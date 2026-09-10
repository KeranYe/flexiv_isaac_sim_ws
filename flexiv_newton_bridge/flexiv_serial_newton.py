"""
FlexivSerialNewton — Direct-Newton mirror of FlexivSerial (Isaac bridge).

Provides the same robot-facing properties and methods as the original
flexiv_serial.FlexivSerial, but reads state from Newton State and writes
torques through Newton Control.joint_f instead of Isaac articulation APIs.

URDF joint labels are prefixed by the Newton importer with the robot name.
For example "Rizon10-ThV7QV_joint1" becomes "Rizon10/Rizon10-ThV7QV_joint1".
This module resolves those names into explicit DoF indices during bind().
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def _warp_assign(warp_array, data):
    """Write numpy data into a warp array.

    Warp arrays do not support in-place item assignment, so we round-trip
    via ``wp.array()`` and ``assign()``.

    Args:
        warp_array: The target warp array to update.
        data: numpy array or list with the same shape/dtype.
    """
    import warp as wp  # lazy import — kept out of module-level to avoid
                        # CUDA warnings when this file is just imported
    if isinstance(data, np.ndarray):
        src = wp.array(dtype=wp.float32, device='cpu', data=data)
    else:
        src = wp.array(dtype=wp.float32, device='cpu', data=np.array(data, dtype=np.float32))
    warp_array.assign(src)


class FlexivSerialNewton:
    """
    Control interface for one Flexiv serial robot backed by direct Newton.

    Mirrors the original Isaac bridge's FlexivSerial in naming, control flow,
    and public properties. The simulator-facing implementation changes from
    Isaac/PhysX articulation views to Newton Model/State/Control.

    Attributes:
        name (str): Human-readable name of this robot instance.
        serial_number (str): Serial number used for the Sim Plugin UserNode.
        arm_dof (int): Degrees of freedom of the robotic arm (excludes gripper).
        has_ft_sensor (bool): Whether this model carries a wrist F/T sensor.
    """

    def __init__(
        self,
        name: str,
        serial_number: str,
        arm_dof: int = 7,
        has_ft_sensor: bool = False,
        kp_hold: float = 100.0,
        kd_hold: float = 20.0,
        max_torque_hold: float = 40.0,
    ) -> None:
        self.name = name
        self.serial_number = serial_number
        self._arm_dof = arm_dof
        self._has_ft_sensor = has_ft_sensor

        # --- Newton objects (set by bridge runner via bind()) ---
        self._model = None
        self._state_0 = None
        self._state_1 = None
        self._control = None

        # --- Resolved joint/DoF indices (set by bind()) ---
        # Map from local arm index [0..arm_dof-1] to Newton joint-qd / Control.joint_f DoF index.
        self._dof_indices: Optional[List[int]] = None
        # Map from local arm index [0..arm_dof-1] to Newton joint-q coordinate index.
        self._joint_q_indices: Optional[List[int]] = None

        # --- Sim Plugin ---
        self.sim_plugin: Optional[object] = None  # flexivsimplugin.UserNode

        # --- Gripper ---
        self.gripper: Optional[object] = None
        # Placeholder — gripper will be bound when the tool is imported (Phase 8).

        # --- Position hold controller (Phase 6) ---
        # When Elements disconnects, the bridge switches to "position" mode.
        # A local PD controller then applies torques to hold the robot at its
        # last-known pose until Elements reconnects and switches back to effort.
        self._mode: str = "effort"  # Default: torque/effort from Elements.
        self._hold_q: Optional[List[float]] = None
        self._kp: float = float(kp_hold)   # PD gain for position hold [Nm/rad].
        self._kd: float = float(kd_hold)   # PD gain for velocity damping [Nm/(rad/s)].
        self._hold_max_torque: float = float(max_torque_hold)  # Per-joint torque clip [Nm].

    # ------------------------------------------------------------------
    # Public properties (mirror original FlexivSerial)
    # ------------------------------------------------------------------

    @property
    def q(self) -> List[float]:
        """Current arm joint positions [rad]."""
        if self._state_0 is None or self._joint_q_indices is None:
            return [0.0] * self._arm_dof
        arr = self._state_0.joint_q.numpy()
        return list(arr[i] for i in self._joint_q_indices)

    @property
    def dq(self) -> List[float]:
        """Current arm joint velocities [rad/s]."""
        if self._state_0 is None or self._dof_indices is None:
            return [0.0] * self._arm_dof
        arr = self._state_0.joint_qd.numpy()
        return list(arr[i] for i in self._dof_indices)

    @property
    def tau(self) -> List[float]:
        """Currently commanded joint torques [Nm]."""
        if self._control is None or self._dof_indices is None:
            return [0.0] * self._arm_dof
        arr = self._control.joint_f.numpy()
        return list(arr[i] for i in self._dof_indices)

    @property
    def has_ft_sensor(self) -> bool:
        """Whether this model carries a wrist force-torque sensor."""
        return self._has_ft_sensor

    @property
    def wrist_wrench(self) -> tuple[List[float], List[float]]:
        """Wrist force and torque readings (Phase 10).

        Reads the external wrench on the flange body from Newton's State.body_f.
        This captures contact reaction forces acting on downstream bodies
        (flange, gripper, tool). The sign convention matches Flexiv: positive
        values represent the force/torque the robot applies ON the environment.

        Returns:
            (force, torque) each as [x, y, z].  Returns zeros when no F/T
            sensor is present or when Newton body_f data is unavailable.

        Notes on Newton semantics (Phase 10 verification):

        * Joint location: body_f reports forces at the body's center of mass.
          For s-model variants with a wrist FT sensor, the flange body sits
          downstream of the sensor joint so this approximation is close.
        * Frame: body_f is expressed in world frame (Newton convention).
          A full implementation would transform to the sensor frame using
          the body pose from State.body_q.
        * Force/torque order: [f_x, f_y, f_z, m_x, m_y, m_z] — 6-vector
          wrench with force first, torque second.
        * Sign: Newton body_f reports external forces ON the body.
          We negate to match Flexiv convention (robot ON environment).
        * Limitation: SolverMuJoCo in Newton 1.5.1 does not currently
          populate State.body_f. This property will return zeros until
          Newton adds reaction force exposure. The implementation is
          structurally correct and ready for when the data becomes available.
        """
        if not self._has_ft_sensor:
            return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

        # Newton body_f is a 6-vector wrench per body: [fx,fy,fz,mx,my,mz].
        # We read from state_0 which holds the current post-step state.
        if self._state_0 is None or not hasattr(self._state_0, 'body_f'):
            return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

        try:
            bf_np = self._state_0.body_f.numpy()
        except Exception:
            return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

        # Find the flange body index in the model.
        # The URDF joint "Rizon10-ThV7QV_link7_to_flange" connects link7 to flange.
        # In Newton the flange body label contains "flange".
        flange_body_idx = self._find_flange_body()
        if flange_body_idx is None or flange_body_idx >= bf_np.shape[0]:
            return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

        # body_f = [fx, fy, fz, mx, my, mz] at center of mass (world frame).
        wrench = bf_np[flange_body_idx]
        force_raw = list(wrench[:3].astype(float))
        torque_raw = list(wrench[3:].astype(float))

        # Negate: Newton reports external ON body; Flexiv wants robot ON env.
        force = [-v for v in force_raw]
        torque = [-v for v in torque_raw]

        return force, torque

    def _find_flange_body(self) -> Optional[int]:
        """Find the Newton body index corresponding to the flange link."""
        if self._model is None:
            return None
        for idx in range(self._model.body_count):
            label = self._model.body_label[idx]
            if "flange" in label.lower():
                return idx
        return None

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def apply_torques(self, tau_d: List[float]) -> None:
        """
        Apply desired torques to all robot arm joints (gripper excluded).

        Writes directly into Control.joint_f at the resolved DoF indices.

        Args:
            tau_d: Desired joint torques [Nm], length must equal arm_dof.
        """
        if len(tau_d) != self._arm_dof:
            raise ValueError(
                f"Expected {self._arm_dof} torques, got {len(tau_d)}"
            )
        if self._control is None or self._dof_indices is None:
            return
        # Read-modify-write to preserve other DoF entries (e.g. gripper).
        f_arr = self._control.joint_f.numpy()
        for idx, dof_idx in enumerate(self._dof_indices):
            f_arr[dof_idx] = tau_d[idx]
        _warp_assign(self._control.joint_f, f_arr)

    def teleport_to(self, q_d: List[float]) -> None:
        """
        Instantly set all arm joint positions to desired values.

        Updates state_0 and state_1 joint_q at the resolved indices and
        zeros the corresponding velocities. Use this for initial pose setup
        or reset semantics (Phase 6).

        Args:
            q_d: Desired joint positions [rad], length must equal arm_dof.
        """
        if len(q_d) != self._arm_dof:
            raise ValueError(
                f"Expected {self._arm_dof} positions, got {len(q_d)}"
            )
        if (
            self._state_0 is None
            or self._state_1 is None
            or self._joint_q_indices is None
            or self._dof_indices is None
        ):
            return

        # Write q to both state buffers.
        for st in (self._state_0, self._state_1):
            q_arr = st.joint_q.numpy()
            for idx, q_idx in enumerate(self._joint_q_indices):
                q_arr[q_idx] = q_d[idx]
            _warp_assign(st.joint_q, q_arr)

        # Zero corresponding velocities in both state buffers.
        for st in (self._state_0, self._state_1):
            qd_arr = st.joint_qd.numpy()
            for dof_idx in self._dof_indices:
                qd_arr[dof_idx] = 0.0
            _warp_assign(st.joint_qd, qd_arr)

    def switch_control_mode(self, mode: str) -> None:
        """
        Switch the control mode for this robot's joints.

        The original Isaac bridge uses articulation_view.switch_control_mode().
        In direct Newton:
          - "effort": torques come from Elements via Control.joint_f.
          - "position": local PD controller holds at snapshot q (disconnect safety).

        Args:
            mode: One of "effort" or "position".
        """
        if mode == "effort":
            self._mode = "effort"
            # Clear hold state when returning to effort control.
            self._hold_q = None
        elif mode == "position":
            self._mode = "position"
            # Snapshot current q as the hold target.
            self._hold_q = list(self.q)
        else:
            raise ValueError(
                f"Unknown control mode '{mode}'; use 'effort' or 'position'"
            )

    def compute_hold_torques(self) -> List[float]:
        """
        Compute PD hold torques for the current state.

        When in position mode this applies:
            tau = Kp * (q_hold - q) - Kd * dq

        Each joint's torque is clipped to [-_hold_max_torque, +_hold_max_torque]
        to prevent numerical blowup from MuJoCo target-mode interaction.

        Returns zero torques when already in effort mode or when no hold target
        is set. NaN guard prevents corruption if state diverges.

        Returns:
            List of 7 torque values [Nm].
        """
        if self._mode != "position" or self._hold_q is None:
            return [0.0] * self._arm_dof

        q_cur = self.q
        dq_cur = self.dq

        # NaN guard: if state has diverged, don't apply stale corrections.
        for i in range(self._arm_dof):
            if not __import__('numpy').isfinite(q_cur[i]) or not __import__('numpy').isfinite(dq_cur[i]):
                return [0.0] * self._arm_dof

        tau_hold: List[float] = []
        clip = self._hold_max_torque
        for i in range(self._arm_dof):
            p_err = self._hold_q[i] - q_cur[i]
            tau_i = self._kp * p_err - self._kd * dq_cur[i]
            # Clip to prevent MuJoCo instability with imported position targets.
            tau_i = max(-clip, min(clip, tau_i))
            tau_hold.append(tau_i)

        return tau_hold

    # ------------------------------------------------------------------
    # Lifecycle — called by the bridge runner after model is finalized
    # ------------------------------------------------------------------

    def bind(self, model, state_0, state_1, control) -> None:
        """
        Attach Newton physics objects and resolve arm joint indices.

        Scans the imported Newton model by joint label to resolve explicit
        index mappings instead of relying on positional assumptions.

        Args:
            model:  Finalized newton.Model.
            state_0: Primary State buffer (read).
            state_1: Secondary State buffer (write/swap target).
            control: Control buffer for applying commands.
        """
        self._model = model
        self._state_0 = state_0
        self._state_1 = state_1
        self._control = control

        # Resolve DoF indices by matching joint labels.
        # Newton prefixes URDF joint names with the articulation label,
        # e.g. "Rizon10/Rizon10-ThV7QV_joint1".
        self._dof_indices = []
        self._joint_q_indices = []

        # Convert all model metadata arrays to numpy upfront for safe indexing
        # (warp arrays do not support item indexing).
        jlabel = model.joint_label
        jq_start = model.joint_q_start.numpy()
        jqd_start = model.joint_qd_start.numpy()
        jdd = model.joint_dof_dim.numpy()

        for arm_j in range(1, self._arm_dof + 1):
            # The URDF joint name without the base prefix.
            bare_name = f"Rizon10-ThV7QV_joint{arm_j}"
            # Try to find this in model.joint_label (which contains the full
            # prefixed names like "Rizon10/Rizon10-ThV7QV_joint1").
            matched = False
            for jidx in range(model.joint_count):
                full_label = jlabel[jidx]
                if bare_name in full_label:
                    # Check this is a revolute (actuated) joint.
                    nqd = int(jdd[jidx, 1])
                    if nqd > 0:
                        self._dof_indices.append(int(jqd_start[jidx]))
                        self._joint_q_indices.append(int(jq_start[jidx]))
                        matched = True
                        break
            if not matched:
                raise RuntimeError(
                    f"Could not resolve arm joint {bare_name} "
                    f"in Newton model (joint_count={model.joint_count})"
                )

        # Validate no NaN/Inf in initial state.
        q_init = state_0.joint_q.numpy()
        qd_init = state_0.joint_qd.numpy()
        if not np.all(np.isfinite(q_init)):
            raise RuntimeError("Initial state has NaN/Inf in joint_q")
        if not np.all(np.isfinite(qd_init)):
            raise RuntimeError("Initial state has NaN/Inf in joint_qd")


# ============================================================================
# Phase 8: Gripper integration
# ============================================================================

class FlexivGripperNewton:
    """
    Newton-backed gripper controller.

    Wraps the finger joints of a gripper imported as part of the arm model.
    Provides ``open()`` and ``close()`` commands that teleport the finger
    joint positions to the configured open/closed values.

    The PGC-140-50 gripper has two prismatic fingers:
      - finger1_joint: axis (1, 0, 0), range [-0.025, 0]
      - finger2_joint: mimics finger1_joint (axis (-1, 0, 0))

    ``open()`` sets finger qd to 0 (retracted).
    ``close()`` sets finger qd to -0.025 (extended / closed).
    """

    def __init__(self) -> None:
        self._model = None
        self._state_0 = None
        self._state_1 = None

        # Resolved DoF indices for gripper fingers (set by bind()).
        self._dof_indices: Optional[List[int]] = None
        # Joint-q coordinate indices.
        self._joint_q_indices: Optional[List[int]] = None

        # Open/closed target positions for each finger DOF.
        self._open_positions: Optional[List[float]] = None
        self._closed_positions: Optional[List[float]] = None

    @property
    def is_bound(self) -> bool:
        """Whether the gripper has been bound to a Newton model."""
        return self._model is not None and self._dof_indices is not None

    def bind(self, model, state_0, state_1, dof_indices=None, joint_q_indices=None,
             open_positions=None, closed_positions=None) -> None:
        """
        Bind the gripper to the Newton model.

        Args:
            model: Finalized newton.Model (arm + gripper combined).
            state_0: Primary State buffer.
            state_1: Secondary State buffer.
            dof_indices: List of DoF indices for gripper finger joints.
                         If None, auto-detect "finger" joints by name.
            joint_q_indices: List of joint-q coordinate indices.
                             If None, derived from dof_indices.
            open_positions: Target q values for fully open state.
                            If None, defaults to [0.0] * len(dof_indices).
            closed_positions: Target q values for fully closed state.
                              If None, defaults to [-0.025] * len(dof_indices).
        """
        self._model = model
        self._state_0 = state_0
        self._state_1 = state_1

        if dof_indices is None:
            # Auto-detect gripper finger joints by name.
            dof_indices, joint_q_indices = self._auto_detect(model)

        self._dof_indices = list(dof_indices)
        self._joint_q_indices = (list(joint_q_indices) if joint_q_indices else list(dof_indices))

        # Default open/closed positions for PGC-140-50 prismatic fingers.
        n_dofs = len(self._dof_indices)
        self._open_positions = list(open_positions) if open_positions else [0.0] * n_dofs
        self._closed_positions = list(closed_positions) if closed_positions else [-0.025] * n_dofs

    def _auto_detect(self, model):
        """Auto-detect gripper finger joints by searching for 'finger' in joint labels."""
        jlabel = model.joint_label
        jq_start = model.joint_q_start.numpy()
        jqd_start = model.joint_qd_start.numpy()
        jdd = model.joint_dof_dim.numpy()

        # Note: Gripper finger joints may be prismatic (type=0) with
        # q_dim=1 but qd_dim=0. We use jq_start for q coordinates and
        # jqd_start for velocity DoFs. For prismatic fingers only q matters.
        dof_indices = []
        q_indices = []
        for jidx in range(model.joint_count):
            label = jlabel[jidx]
            if "finger" in label.lower():
                nq = int(jdd[jidx, 0])  # q_dim (position)
                nqd = int(jdd[jidx, 1])  # qd_dim (velocity)
                # Prismatic joints have q_dim=1, qd_dim=0
                # Revolute joints have q_dim=0 (after first), qd_dim=1
                if nq > 0 or nqd > 0:
                    dof_indices.append(int(jqd_start[jidx]))
                    q_indices.append(int(jq_start[jidx]))
        return dof_indices, q_indices

    def open(self) -> None:
        """Teleport gripper fingers to fully open position."""
        self._set_positions(self._open_positions)

    def close(self) -> None:
        """Teleport gripper fingers to fully closed position."""
        self._set_positions(self._closed_positions)

    def _set_positions(self, target_positions: List[float]) -> None:
        """Set finger joint positions in both state buffers (teleport)."""
        if not self.is_bound or not target_positions:
            return

        import warp as wp
        # Write into both state buffers to maintain consistency.
        for state in (self._state_0, self._state_1):
            q_buf = state.joint_q.numpy().copy()
            for i, idx in enumerate(self._joint_q_indices):
                if i < len(target_positions):
                    q_buf[idx] = float(target_positions[i])
            state.joint_q.assign(
                wp.array(dtype=wp.float32, device='cpu', data=q_buf)
            )

    @property
    def finger_positions(self) -> List[float]:
        """Current finger joint positions."""
        if not self.is_bound:
            return [0.0] * (len(self._dof_indices or []))
        q_np = self._state_0.joint_q.numpy()
        return list(q_np[i] for i in self._joint_q_indices)
