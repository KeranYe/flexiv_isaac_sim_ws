# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Newton-compatible Flexiv serial robot wrapper (Phases 1-6).

Mirrors the public interface of ``flexiv_serial.FlexivSerial`` but uses the
engine-agnostic experimental Articulation API so it works with Newton as well
as PhysX.  It does **not** subclass the deprecated
:py:class:`isaacsim.core.api.robots.robot.Robot`.

Lifecycle
---------
1. ``__init__`` — only stores configuration; no physics access.
2. USD is added to stage and World reset (physics parsed).
3. ``initialize()`` — binds the experimental Articulation, validates entity,
   resolves DoF indices, saves default PD gains.
4. Bridge calls ``teleport_to()``, then switches control mode and enters the
   command loop.
5. On every physics callback: read ``q``, ``dq`` -> send to Elements;
   receive torques from Elements -> ``apply_torques()``.

Phases implemented in this file:
- Phase 2: arm-only Articulation wrapper with q/dq/tau/control
- Phase 5: wrist F/T sensor via get_link_incoming_joint_force()
- Phase 6B: Newton gripper adapter (GripperNewton) for Grav gripper actuation
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import spdlog
from pxr import Sdf, Usd

from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.utils.stage import get_current_stage


# ---------------------------------------------------------------------------
# Backend-agnostic tensor helper
# ---------------------------------------------------------------------------

def _to_numpy(x) -> np.ndarray:
    """Safely convert a backend tensor (Warp, Torch, or NumPy) to host NumPy.

    The experimental Articulation API may return ``wp.array``, PyTorch tensors,
    or plain NumPy arrays depending on the active physics backend.  This helper
    normalises all of them so callers can safely call ``.tolist()``.
    """
    if hasattr(x, "numpy"):
        # Warp array or Torch tensor
        return x.numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# GripperNewton (Phase 6B — Newton gripper actuation)
# ---------------------------------------------------------------------------

class GripperNewton:
    """Newton-compatible gripper actuation using the experimental Articulation API.

    Drives only the gripper DoFs via position targets on the parent
    FlexivSerialNewton articulation, so the arm stays in effort mode.

    Public interface::

        gripper.open()
        gripper.close()

    Params:
        parent_robot: The FlexivSerialNewton instance whose Articulation we share.
        joint_prim_names: Joint names in the USD (e.g. finger_joint).
        opened_positions: Joint positions [rad] for fully open.
        closed_positions: Joint positions [rad] for fully closed.
        logger_name: Logger name prefix.
    """

    def __init__(
        self,
        parent_robot: FlexivSerialNewton,
        joint_prim_names: List[str],
        opened_positions: List[float],
        closed_positions: List[float],
        logger_name: str = "GripperNewton",
    ) -> None:
        self._robot = parent_robot
        self._joint_names = list(joint_prim_names)
        self._open_pos = np.array(opened_positions, dtype=np.float32)
        self._close_pos = np.array(closed_positions, dtype=np.float32)
        self._logger = spdlog.ConsoleLogger(f"flexiv::Newton::{logger_name}")

        # Resolved DoF indices (set during initialize()).
        self._gripper_dof_indices: Optional[np.ndarray] = None
        self._initialized = False

    def initialize(self) -> None:
        """Resolve gripper DoF indices and switch them to position mode.

        Called after FlexivSerialNewton.initialize().  Switches only the
        gripper DoFs to position control mode so the arm stays in effort mode.
        """
        if self._gripper_dof_indices is not None:
            self._logger.info("Already initialized")
            return

        all_dof_names = self._robot._articulation.dof_names
        indices = []
        for jn in self._joint_names:
            try:
                idx = all_dof_names.index(jn)
                indices.append(idx)
            except ValueError:
                # Prefix match fallback.
                matched = [
                    j for j, n in enumerate(all_dof_names) if jn.lower() in n.lower()
                ]
                if matched:
                    indices.append(matched[0])
                else:
                    msg = (
                        f"Gripper DoF '{jn}' not found in articulation "
                        f"DoF names:\n  {all_dof_names}"
                    )
                    raise RuntimeError(msg)

        self._gripper_dof_indices = np.array(indices, dtype=np.int64)
        self._initialized = True

        # Switch only gripper DoFs to position control (arm stays in effort).
        self._robot._articulation.switch_dof_control_mode(
            "position", dof_indices=self._gripper_dof_indices
        )

        msg = (
            f"Gripper initialized: joints={self._joint_names} "
            f"indices={list(self._gripper_dof_indices)}"
        )
        self._logger.info(msg)

    def open(self) -> None:
        """Move gripper to the open position."""
        if not self._initialized or self._gripper_dof_indices is None:
            raise RuntimeError(
                "GripperNewton not initialized. Call initialize() first."
            )
        self._robot._articulation.set_dof_position_targets(
            self._open_pos, dof_indices=self._gripper_dof_indices
        )

    def close(self) -> None:
        """Move gripper to the closed position."""
        if not self._initialized or self._gripper_dof_indices is None:
            raise RuntimeError(
                "GripperNewton not initialized. Call initialize() first."
            )
        self._robot._articulation.set_dof_position_targets(
            self._close_pos, dof_indices=self._gripper_dof_indices
        )


# ---------------------------------------------------------------------------
# FlexivSerialNewton
# ---------------------------------------------------------------------------

class FlexivSerialNewton:
    """Newton-compatible control interface for one Flexiv serial robot.

    Public contract identical to :py:class:`FlexivSerial`::

        q, dq, tau
        has_ft_sensor, wrist_wrench
        switch_control_mode(mode)
        apply_torques(tau_d)
        teleport_to(q_d)

    Params:
        prim_path (str): Primitive path of this robot in the stage.
        name (str): Human-readable name for logging.
        end_effector_prim_name (str): Name used to resolve the end-effector prim.
        arm_dof (int): Arm DoF count (default 7).
        pos_in_world (Optional[List[float]]): Position override (unused here;
            pose should be authored in USD before physics parses).
        ori_in_world (Optional[List[float]]): Orientation override.
        has_ft_sensor (bool): True for ``s`` variants with wrist F/T sensor.
    """

    # Fixed joint name that isolates the wrist force-torque sensor reading zone.
    _FT_JOINT_NAME = "link7_ft_sensor"

    def __init__(
        self,
        prim_path: str,
        name: str,
        end_effector_prim_name: str,
        arm_dof: int = 7,
        pos_in_world: Optional[List[float]] = None,
        ori_in_world: Optional[List[float]] = None,
        has_ft_sensor: bool = False,
    ) -> None:
        self._prim_path = prim_path
        self._name = name
        self._arm_dof = arm_dof
        self._has_ft_sensor = has_ft_sensor
        self._logger = spdlog.ConsoleLogger(f"flexiv::Newton::{name}")

        # -- End-effector path resolution (same logic as FlexivSerial) --------
        self._end_effector_prim_path = self._resolve_end_effector_prim_path(
            prim_path, end_effector_prim_name
        )

        # -- Articulation (bound lazily in initialize()) ---------------------
        self._articulation: Optional[Articulation] = None
        self._arm_dof_indices: Optional[np.ndarray] = None

        # -- Wrist F/T -------------------------------------------------------
        self._ft_force_row: Optional[int] = None

        # -- Gripper (Phase 6B) ----------------------------------------------
        self._gripper: Optional[GripperNewton] = None

        # Log configuration at construction time (safe before physics parse).
        self._logger.info(
            f"[{name}] prim_path={prim_path}  "
            f"end_effector={self._end_effector_prim_path}"
        )
        self._logger.info(f"[{name}] arm_dof={arm_dof}  has_ft_sensor={has_ft_sensor}")

    # ------------------------------------------------------------------
    # Properties (read-only metadata)
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable robot name."""
        return self._name

    @property
    def num_dofs(self) -> int:
        """Total number of articulation DoFs (arm + any attached tool/gripper)."""
        if self._articulation is None:
            return 0
        return self._articulation.num_dofs

    @property
    def gripper(self) -> Optional[GripperNewton]:
        """Return the attached gripper instance, if any."""
        return self._gripper

    # ------------------------------------------------------------------
    # End-effector path resolution (mirrors FlexivSerial)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_end_effector_prim_path(
        prim_path: str, end_effector_prim_name: str
    ) -> str:
        """Resolve the end-effector prim across flat and SimReady USD layouts.

        Two layout patterns are supported:

        1. **Flat** — the EE is a direct child of *prim_path*, so concatenating
           ``prim_path + "/" + end_effector_prim_name`` gives the correct path.
        2. **SimReady** — the EE prim is nested deep in the hierarchy (e.g.
           under ``Geometry/.../link7``).  The direct path doesn't exist; we
           search the entire subtree rooted at *prim_path* for a prim whose
           name matches the *last component* of *end_effector_prim_name*.

        Returns:
            Full USD path to the end-effector prim, or raises ``KeyError``.
        """
        stage = get_current_stage()
        direct = prim_path + "/" + end_effector_prim_name
        if stage.GetPrimAtPath(direct).IsValid():
            return direct

        # Fall back to subtree search using leaf component.
        leaf = end_effector_prim_name.split("/")[-1]
        root = stage.GetPrimAtPath(prim_path)
        if not root.IsValid():
            raise KeyError(
                f"Root prim [{prim_path}] not found on stage; "
                f"cannot resolve end-effector '{end_effector_prim_name}'."
            )

        for prim in Usd.PrimRange(root):
            if prim.GetName() == leaf:
                return str(prim.GetPath())

        raise KeyError(
            f"End-effector not found. Tried direct [{direct}] "
            f"and subtree search for leaf '{leaf}' under [{prim_path}]."
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, physics_sim_view=None) -> None:
        """Bind the experimental Articulation and resolve DoF metadata.

        Called **after** USD references have been added to the stage and
        the physics scene has parsed (i.e. after ``World.reset()``).

        Args:
            physics_sim_view: Unused; kept for signature compatibility with
                legacy ``FlexivSerial.initialize()``.
        """
        self._logger.info(
            f"[{self._name}] Initializing Articulation at {self._prim_path}"
        )

        self._articulation = Articulation(
            paths=self._prim_path, resolve_paths=True
        )

        if not self._articulation.valid:
            raise RuntimeError(
                f"Articulation prim [{self._prim_path}] is not valid. "
                f"Check USD path and that PhysicsScene exists."
            )

        if not self._articulation.is_physics_tensor_entity_valid():
            raise RuntimeError(
                f"Physics tensor entity for [{self._prim_path}] is not valid. "
                f"Ensure physics has parsed (World.reset() called) before "
                f"calling initialize()."
            )

        # Save default PD gains so we can restore position mode later.
        self._default_kps, self._default_kds = self._articulation.get_dof_gains()

        # Resolve arm DoF indices by name (defensive: don't assume layout).
        self._resolve_arm_dof_indices()

        # Resolve F/T sensor row if enabled (Phase 5).
        if self._has_ft_sensor:
            try:
                self._ft_force_row = self._resolve_ft_force_row()
            except Exception as e:
                self._ft_force_row = None
                msg = (
                    f"Failed to resolve F/T sensor joint [{self._FT_JOINT_NAME}], "
                    f"wrist wrench will report zeros: {e}"
                )
                self._logger.error(msg)

        # Initialize gripper if it was attached before init (Phase 6B).
        if self._gripper is not None:
            self._gripper.initialize()
            msg = f"[{self._name}] Gripper initialized alongside arm"
            self._logger.info(msg)

        msg = (
            f"[{self._name}] Articulation OK  num_dofs={self.num_dofs}  "
            f"arm_indices={list(self._arm_dof_indices)}"
        )
        self._logger.info(msg)

    def post_reset(self) -> None:
        """Re-validate after stage / physics reset.

        The experimental Articulation does not require explicit re-binding
        after a reset (unlike the legacy PhysX view), but we log for clarity.
        """
        if self._articulation is not None:
            valid = self._articulation.is_physics_tensor_entity_valid()
            if valid:
                self._logger.info(f"[{self._name}] post_reset: tensor entity valid")
            else:
                self._logger.warn(
                    f"[{self._name}] post_reset: tensor entity invalid — "
                    f"will need re-initialize"
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_arm_dof_indices(self) -> None:
        """Find the seven arm DoF indices inside the full articulation DoF list.

        Flexiv arms have joint names ``joint1`` .. ``joint7``.  We scan the
        resolved DoF names and record their positions.  If a gripper is later
        added its DoFs will appear *after* index 6, which is fine because we
        only touch indices 0..6 in arm commands.

        Raises:
            RuntimeError if fewer than ``_arm_dof`` expected joints are found.
        """
        all_dof_names = self._articulation.dof_names
        arm_names = [f"joint{i}" for i in range(1, self._arm_dof + 1)]

        indices = []
        for aname in arm_names:
            try:
                idx = all_dof_names.index(aname)
                indices.append(idx)
            except ValueError:
                # Prefix match fallback (some USDs embed robot name in DoF label).
                matched = [
                    j for j, n in enumerate(all_dof_names) if aname in n.lower()
                ]
                if matched:
                    indices.append(matched[0])
                else:
                    msg = (
                        f"DoF '{aname}' not found in articulation "
                        f"DoF names:\n  {all_dof_names}"
                    )
                    raise RuntimeError(msg)

        self._arm_dof_indices = np.array(indices, dtype=np.int64)

        name_idx_map = dict(zip(arm_names, list(self._arm_dof_indices)))
        self._logger.info(f"[{self._name}] Arm DoF indices: {name_idx_map}")

    def _resolve_ft_force_row(self) -> int:
        """Resolve the row index into link incoming joint forces for the F/T sensor.

        The wrist F/T sensor in ``s``-variant USDs sits at a fixed joint named
        ``link7_ft_sensor``.  We need its link/joint index to read the reaction
        wrench from ``get_link_incoming_joint_force()``.

        Returns:
            Integer row index into the force tensor output (+1 because row 0 is
            the base-link incoming joint).

        Raises:
            KeyError if the joint or distal link cannot be found.
        """
        link_names = self._articulation.link_names

        # Look for "link7_distal" — the child of the sensor fixed joint whose
        # incoming force IS what the physical wrist sensor reports.
        distal_idx = None
        for i, ln in enumerate(link_names):
            if "link7_distal" in ln.lower():
                distal_idx = i
                break

        if distal_idx is not None:
            return distal_idx + 1

        # Fallback: find the joint by name in articulation metadata.
        try:
            view = self._articulation._physics_articulation_view
            meta = getattr(view, "_metadata", None) or view._get_metadata()
            jindices = getattr(meta, "joint_indices", None)
            if isinstance(jindices, dict) and self._FT_JOINT_NAME in jindices:
                return jindices[self._FT_JOINT_NAME] + 1

            jnames = getattr(meta, "joint_names", None)
            if jnames is not None and self._FT_JOINT_NAME in list(jnames):
                return list(jnames).index(self._FT_JOINT_NAME) + 1
        except Exception:
            pass

        raise KeyError(
            f"F/T sensor joint [{self._FT_JOINT_NAME}] or link7_distal "
            f"not found in articulation metadata (links={link_names})"
        )

    # ------------------------------------------------------------------
    # Public state properties
    # ------------------------------------------------------------------

    @property
    def q(self) -> List[float]:
        """Current arm joint positions (length ``_arm_dof``)."""
        data = self._articulation.get_dof_positions(
            dof_indices=self._arm_dof_indices,
        )
        return _to_numpy(data).flatten().tolist()

    @property
    def dq(self) -> List[float]:
        """Current arm joint velocities (length ``_arm_dof``)."""
        data = self._articulation.get_dof_velocities(
            dof_indices=self._arm_dof_indices,
        )
        return _to_numpy(data).flatten().tolist()

    @property
    def tau(self) -> List[float]:
        """Current *commanded* arm joint efforts (length ``_arm_dof``).

        Returns the effort last set via ``set_dof_efforts()``, not a measured
        sensor value.  Documented to distinguish from PhysX measured torque.
        """
        data = self._articulation.get_dof_efforts(
            dof_indices=self._arm_dof_indices,
        )
        return _to_numpy(data).flatten().tolist()

    @property
    def has_ft_sensor(self) -> bool:
        return self._has_ft_sensor

    @property
    def wrist_wrench(self) -> tuple[List[float], List[float]]:
        """Wrist F/T reading as (force, torque), each length 3.

        Returns zeros with a warning if the sensor row is not resolved or if
        this robot does not carry an ``s``-variant sensor.
        """
        if not self._has_ft_sensor:
            return [0.0] * 3, [0.0] * 3

        if self._ft_force_row is None:
            warnings.warn(
                f"[{self._name}] Wrist F/T row not resolved; returning zeros.",
                UserWarning,
            )
            return [0.0] * 3, [0.0] * 3

        try:
            forces = self._articulation.get_link_incoming_joint_force()
            forces_np = _to_numpy(forces)

            # Clamp row index to valid range.
            row = min(self._ft_force_row, forces_np.shape[0] - 1)

            if forces_np.ndim >= 2:
                wrench = forces_np[row, 0]
            else:
                wrench = forces_np[row]

            # Negate to match Flexiv convention (same as PhysX implementation).
            force = (-wrench[:3]).tolist()
            torque = (-wrench[3:]).tolist()
            return force, torque
        except Exception as e:
            self._logger.error(
                f"[{self._name}] Failed to read wrist wrench: {e}"
            )
            return [0.0] * 3, [0.0] * 3

    # ------------------------------------------------------------------
    # Gripper setup (Phase 6B)
    # ------------------------------------------------------------------

    def initialize_gripper(
        self,
        joint_prim_names: List[str],
        opened_positions: List[float],
        closed_positions: List[float],
    ) -> None:
        """Create and initialize a Newton gripper adapter.

        Called after the robot articulation is initialized but before entering
        the control loop.  The gripper shares the parent Articulation so arm
        stays in effort mode while the gripper DoFs are driven by position targets.

        Params:
            joint_prim_names: Names of the gripper actuation joints in the USD.
            opened_positions: Joint positions [rad] for fully open.
            closed_positions: Joint positions [rad] for fully closed.
        """
        self._gripper = GripperNewton(
            parent_robot=self,
            joint_prim_names=joint_prim_names,
            opened_positions=opened_positions,
            closed_positions=closed_positions,
            logger_name=f"{self._name}_gripper",
        )
        self._gripper.initialize()
        self._logger.info(
            f"[{self._name}] Gripper initialized: {joint_prim_names}"
        )

    # ------------------------------------------------------------------
    # Control methods
    # ------------------------------------------------------------------

    def switch_control_mode(self, mode: str) -> None:
        """Switch arm DoFs to *mode*.

        Args:
            mode: ``"position"``, ``"velocity"``, or ``"effort"``.
        """
        self._articulation.switch_dof_control_mode(
            mode, dof_indices=self._arm_dof_indices
        )
        self._logger.info(f"[{self._name}] switch_control_mode -> {mode}")

    def apply_torques(self, tau_d: List[float]) -> None:
        """Apply desired torques to the seven arm joints.

        Args:
            tau_d: Desired joint torques (length ``_arm_dof``).
        """
        if len(tau_d) != self._arm_dof:
            raise ValueError(
                f"Expected {self._arm_dof} torques, got {len(tau_d)}."
            )

        arr = np.array(tau_d, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            self._logger.warn(f"[{self._name}] Non-finite torque detected: {tau_d}")

        self._articulation.set_dof_efforts(
            arr.astype(np.float32), dof_indices=self._arm_dof_indices
        )

    def teleport_to(self, q_d: List[float]) -> None:
        """Instantly set arm joint positions.  No control is involved.

        Call ``apply_torques()`` immediately after to kick in the joint controls.

        Args:
            q_d: Desired joint positions (length ``_arm_dof``).
        """
        if len(q_d) != self._arm_dof:
            raise ValueError(
                f"Expected {self._arm_dof} positions, got {len(q_d)}."
            )

        arr = np.array(q_d, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            self._logger.warn(f"[{self._name}] Non-finite position detected: {q_d}")

        self._articulation.set_dof_positions(
            arr.astype(np.float32), dof_indices=self._arm_dof_indices
        )

    # ------------------------------------------------------------------
    # Startup diagnostics
    # ------------------------------------------------------------------

    def log_startup_info(
        self, isaac_version: str, dt_physics: float, dt_render: float
    ):
        """Print all required startup diagnostics (Phase 17 of dev plan)."""
        self._logger.info("=== Newton Bridge Startup ===")
        self._logger.info(f"Isaac Sim version: {isaac_version}")

        backend = "N/A"
        if self._articulation is not None:
            try:
                backend = self._articulation.get_backend()
            except Exception:
                pass
        self._logger.info(f"Active physics engine: {backend}")
        self._logger.info(f"Robot USD path: {self._prim_path}")
        self._logger.info(f"Resolved articulation path: {self._prim_path}")

        dof_info = "N/A"
        if self._arm_dof_indices is not None:
            dof_info = (
                f"{list(self._arm_dof_indices)} "
                f"(names={self._articulation.dof_names})"
            )
        self._logger.info(f"Resolved arm DoF indices: {dof_info}")

        gripper_status = "active" if self._gripper is not None else "not attached"
        self._logger.info(f"F/T enabled: {self._has_ft_sensor}")
        self._logger.info(f"Gripper: {gripper_status}")
        self._logger.info(
            f"Physics dt: {dt_physics:.6f}  Render dt: {dt_render:.6f}"
        )

