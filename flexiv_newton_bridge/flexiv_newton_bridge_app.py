"""
flexiv_newton_bridge_app — Direct-Newton bridge app mirroring flexiv_isaac_bridge_app.

Owns the complete Newton scene: ModelBuilder, environment, robot import,
gripper/workpiece additions, model finalization, State/Control buffers,
CollisionPipeline, and SolverMuJoCo.

Communication with Elements Studio flows through Flexiv Sim Plugin 1.3.0
exactly as the original Isaac bridge does.

Usage:
    python flexiv_newton_bridge_app.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import os as _os
import sys
import time
import yaml
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging (spdlog-style timestamps: ISO 8601 with milliseconds)
# ---------------------------------------------------------------------------

import datetime as _dt

# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------

APP_VERSION = "1.4.0"
COMPATIBLE_SIM_PLUGIN_VER = "1.3.0"

# ---------------------------------------------------------------------------
# Import checks (no Isaac dependency)
# ---------------------------------------------------------------------------

import flexivsimplugin

if flexivsimplugin.__version__ != COMPATIBLE_SIM_PLUGIN_VER:
    print(
        f"WARNING: this app targets flexivsimplugin=={COMPATIBLE_SIM_PLUGIN_VER}, "
        f"but found {flexivsimplugin.__version__}. Continuing anyway; behavior may "
        f"differ if the plugin API has changed.",
        file=sys.stderr,
    )

# Warp kernel cache MUST be writable.  /home/keran/.cache/warp is often on a
# read-only overlay in managed environments; redirect to /tmp if needed.
import warp as wp
_cache_dir = "/home/keran/.cache/warp"
if not _os.access(_cache_dir, _os.W_OK):
    wp.config.kernel_cache_dir = "/tmp/warp_kernel_cache"

import newton

# Import our Newton robot wrapper from the same directory.
import importlib.util as _util

_flexiv_dir = _os.path.dirname(_os.path.abspath(__file__))
_fspec = _util.spec_from_file_location(
    "flexiv_serial_newton",
    _os.path.join(_flexiv_dir, "flexiv_serial_newton.py"),
)
_fmod = _util.module_from_spec(_fspec)
_fspec.loader.exec_module(_fmod)
FlexivSerialNewton = _fmod.FlexivSerialNewton

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHYSICS_FREQ = 2000.0
PHYSICS_DT = 1.0 / PHYSICS_FREQ  # 0.0005 s


def _ts(msg: str, prefix: str = "[NewtonBridge]") -> None:
    """Print with timestamp (spdlog-style ISO 8601)."""
    import datetime as _dt
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {prefix} {msg}")

@dataclass
class SingleRobotData:
    """Holds everything the runner needs for one robot."""
    name: str
    serial_number: str
    instance: FlexivSerialNewton = None
# TODO(gripper): Re-enable when gripper is integrated
    sim_plugin: object = None  # flexivsimplugin.UserNode
    last_connected: bool = False
    _last_wait_error: float = 0.0  # Timestamp of last logged WaitForRobotCommands error
    # Feedback blending buffers (Phase 11 stability fix).
    # These hold the "smoothed" q/dq that we feed back to Elements Studio,
    # which is a mix between actual Newton physics and what Elements expects.
    _blend_q: Optional[List[float]] = None
    _blend_dq: Optional[List[float]] = None

    # Settling state (mirrors MuJoCo bridge behavior).
    # After Elements reconnects, hold zero torques for kSettleSec seconds
    # to let ES impedance control stabilize before applying real commands.
    _settle_start: float = 0.0       # perf_counter timestamp when settling began
    _is_settling: bool = False

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class NewtonBridgeRunner:
    """
    Main simulation runner for the direct Newton bridge.

    Mirrors the original BridgeRunner but replaces Isaac World/Physics with
    explicit Newton Model/State/Control/CollisionPipeline/SolverMuJoCo.

    Communication order per cycle (mirrors the original):
        1. Read q/dq from Newton state -> SendRobotStates to Elements
        2. WaitForRobotCommands from Elements
        3. Apply torques to Control.joint_f
        4. CollisionPipeline.collide() + SolverMuJoCo.step()
        5. Swap state buffers, increment servo_cycle
    """

    def __init__(self, config: dict, viewer_backend: str = "null") -> None:
        # Parse config.
        self._config = config
        self._robots: List[SingleRobotData] = []
        self._servo_cycle: int = 0
        self._running: bool = True
        self._viewer_backend: str = viewer_backend
        self._did_reconnect: bool = False      # Flag for solver reset after reconnect.
        self._ever_ran_step: bool = False      # Track if at least one step has completed.

        # ----------------------------------------------------------------
        # Control parameters (tunable via YAML "control" section)
        # ----------------------------------------------------------------
        ctrl = config.get("control", {})
        self._blend_alpha: float = float(ctrl.get("alpha", 0.7))       # q blend weight
        self._blend_beta: float = float(ctrl.get("beta", 0.85))        # dq blend weight
        self._tau_limits_per_joint: List[float] = ctrl.get(
            "tau_limits_per_joint",
            [60.0, 60.0, 40.0, 40.0, 15.0, 15.0, 15.0],
        )
        self._kd_damp: float = float(ctrl.get("kd_damp", 10.0))        # velocity damping in effort mode
        self._settle_duration: float = float(ctrl.get("settle_duration_sec", 2.0))  # seconds of zero-torque settling after Elements connects
        self._gravity_disable: bool = bool(ctrl.get("disable_gravity", False))  # Disable sim gravity for pure-command control
        # Position-hold PD gains (passed to FlexivSerialNewton constructor).
        self._hold_kp: float = float(ctrl.get("hold_kp", 100.0))      # Kp for position hold [Nm/rad]
        self._hold_kd: float = float(ctrl.get("hold_kd", 20.0))       # Kd for velocity damping [Nm/(rad/s)]
        self._hold_max_torque: float = float(ctrl.get("hold_max_torque", 40.0))

        # Timing diagnostics (Phase 5).
        self._timing_stats: Dict[str, List[float]] = {
            "send_states": [],
            "wait_commands": [],
            "apply_torques": [],
            "collision": [],
            "solver": [],
            "cycle_total": [],
        }

        # Newton objects (owned by the runner).
        self._builder: Optional[newton.ModelBuilder] = None
        self._model: Optional[newton.Model] = None
        self._state_0: Optional[newton.State] = None
        self._state_1: Optional[newton.State] = None
        self._control: Optional[newton.Control] = None
        self._collision_pipeline: Optional[object] = None
        self._solver: Optional[object] = None
        self._contacts: Optional[object] = None

        # Initial pose for reset semantics (default Rizon10 home).
        self._initial_q: List[float] = config.get(
            "home_q", [0.0, -0.698132, 0.0, 1.5708, 0.0, 0.698132, 0.0]
        )

        # Build the complete Newton scene.
        self._build_scene()

    def _build_scene(self) -> None:
        """
        Construct the Newton simulation scene.

        Loads URDF(s), adds static environment (Phase 7), finalizes model,
        creates State/Control buffers, CollisionPipeline, and SolverMuJoCo.
        """
        self._builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

        # Record pre-import counts for diagnostics.
        pre_bodies = self._builder.body_count
        pre_joints = self._builder.joint_count
        pre_shapes = self._builder.shape_count

        _ts(f"Builder pre-import: bodies={pre_bodies}, joints={pre_joints}, shapes={pre_shapes}")
        # Load robot(s) from config.
        for rconf in self._config.get("robots", []):
            self._load_robot(rconf)

        # Record post-import counts.
        post_bodies = self._builder.body_count
        post_joints = self._builder.joint_count
        post_shapes = self._builder.shape_count

        _ts(f"Builder post-import: bodies={post_bodies}, joints={post_joints}, shapes={post_shapes}")
        # ---- Phase 7: Ordinary point contact — static environment ----
        # Add ground plane at z=0 and a static table block. Contact gap is
        # explicitly configured rather than relying on Newton defaults.
        _ts("Adding static environment (ground + table)...")

        # Ground plane at z = 0 (infinite XY). Explicit gap=0.01 for contact margin.
        self._builder.add_shape_plane(
            plane=(0.0, 0.0, 1.0, 0.0),
            cfg=newton.ModelBuilder.ShapeConfig(
                gap=0.01,
                margin=0.0,
                mu=1.0,
                restitution=0.0,
            ),
            label="ground_plane",
        )

        # Static table block: 0.4 x 0.6 m top surface, center at z=0.525 m.
        # Positioned so the arm flange (at ~z=0.3) can reach over it.
        # Added as a static shape attached to the world body (body=-1).
        self._builder.add_shape_box(
            body=-1,  # world-static
            xform=wp.transform(wp.vec3(0.0, 0.525, 0.0), wp.quat_identity()),
            hx=0.4, hy=0.3, hz=0.05,
            cfg=newton.ModelBuilder.ShapeConfig(
                gap=0.01,
                margin=0.0,
                mu=1.0,
                restitution=0.0,
            ),
            label="table_top",
        )

        _ts(f"Environment added: bodies={self._builder.body_count}, shapes={self._builder.shape_count}")


        # Finalize model and create state/control buffers.
        self._model = self._builder.finalize()

        # Use pure effort mode (joint_target_mode=0) for all DOFs.
        # The URDF importer sets joint_target_mode=1 by default, but we disable
        # it so that Newton only responds to explicit torque commands in
        # Control.joint_f. This is more predictable than the built-in PD since
        # Elements provides its own impedance control via target drives.
        import numpy as _np
        import warp as _wp
        self._model.joint_target_mode.assign(
            _wp.array(
                dtype=_wp.int32, device='cpu',
                data=_np.zeros(self._model.joint_dof_count, dtype=_np.int32),
            )
        )

        # Simulation gravity controls whether Elements must compensate for it.
        # When gravity is enabled (default), Elements' compensation torques match
        # the physics environment, preventing the mismatch that caused instability.
        if self._gravity_disable:
            import warp as _wp2
            zero_g = _wp2.array(dtype=_wp2.vec3f, device='cpu', data=[(0.0, 0.0, 0.0)])
            self._model.gravity.assign(zero_g)
            _ts("Gravity DISABLED (pure command mode)")
        else:
            # Default: gravity enabled (~-9.81 m/s^2 along -Z). Elements' gravity
            # compensation torques now match the physics environment, which is
            # critical for stable effort-mode control.
            _ts("Gravity ENABLED (default ~-9.81 m/s^2)")

        _ts("Model finalized:")
        _ts(f"  body_count   = {self._model.body_count}")
        _ts(f"  joint_count  = {self._model.joint_count}")
        _ts(f"  joint_dof    = {self._model.joint_dof_count}")
        _ts(f"  shape_count  = {self._model.shape_count}")

        # Create two state buffers (for ping-pong swapping).
        self._state_0 = self._model.state(0)
        self._state_1 = self._model.state(0)

        # Create control buffer.
        self._control = self._model.control()

        # Create collision pipeline.  rigid_contact_max tuned to handle
        # self-collision contacts from detailed URDF mesh geometry.
        # Phase 9: Add hydroelastic SDF config if workpieces are present.
        sdf_hydro_config = None
        has_workpiece = any(rconf.get("workpiece", {}).get("enabled") for rconf in self._config.get("robots", []))
        if has_workpiece:
            sdf_hydro_config = newton.geometry.HydroelasticSDF.Config(
                output_contact_surface=True,
            )

        self._collision_pipeline = newton.CollisionPipeline(
            self._model,
            rigid_contact_max=4096,
            sdf_hydroelastic_config=sdf_hydro_config,
        )

        # Create solver with increased contact buffer to avoid overflow warnings.
        self._solver = newton.solvers.SolverMuJoCo(
            self._model,
            use_mujoco_contacts=False,
            nconmax=8192,
            njmax=4096,
        )

        # Create contacts buffer for collision pipeline.
        self._contacts = newton.Contacts(rigid_contact_max=4096, soft_contact_max=0)

        # Bind all robots to the finalized model + state/control buffers.
        self._bind_robots()

        # Initialize feedback-blend buffers from initial Newton state (home pose).
        for robot in self._robots:
            q_init = robot.instance.q  # already at home pose from _bind_robots
            dq_init = robot.instance.dq  # zero velocities from teleport_to
            robot._blend_q = list(q_init)
            robot._blend_dq = list(dq_init)

        # Phase 11: Setup viewer if requested.
        self._viewer = None
        if self._viewer_backend != "null":
            self._setup_viewer()

    def _setup_viewer(self) -> None:
        """Set up Newton viewer if a backend was requested."""
        try:
            import newton.viewer as nviewer
            _ts(f"Opening {self._viewer_backend.upper()} viewer...")
            if self._viewer_backend == "gl":
                self._viewer = nviewer.ViewerGL(width=1920, height=1080)
            elif self._viewer_backend == "viser":
                self._viewer = nviewer.ViewerViser()
            elif self._viewer_backend == "rtx":
                self._viewer = nviewer.ViewerRTX()
            else:
                _ts(f"WARNING: unknown viewer backend '{self._viewer_backend}'")
                return

            # Feed the model + initial state to the viewer.
            self._viewer.set_model(self._model)
            self._viewer.log_state(self._state_0)
            self._viewer.begin_frame(0.0)
        except Exception as e:
            _ts(f"Viewer init failed ({e}), continuing without window.")
            self._viewer = None

    def _load_robot(self, rconf: dict) -> None:
        """
        Register one robot from the config.

        Creates the FlexivSerialNewton instance and Sim Plugin UserNode.
        Imports the URDF into the Newton builder (Phase 2).
        """
        name = rconf.get("name", "FlexivRobot")
        serial_number = rconf.get("serial_number", "")
        urdf_path = rconf.get("urdf", "")

        arm_dof = rconf.get("arm_dof", 7)
        has_ft_sensor = rconf.get("has_ft_sensor", False)

        # Create our Newton-backed robot instance.
        instance = FlexivSerialNewton(
            name=name,
            serial_number=serial_number,
            arm_dof=arm_dof,
            has_ft_sensor=has_ft_sensor,
            kp_hold=self._hold_kp,
            kd_hold=self._hold_kd,
            max_torque_hold=self._hold_max_torque,
        )

        # Create the Sim Plugin user node for Elements communication.
        sim_plugin = flexivsimplugin.UserNode(serial_number)

        robot = SingleRobotData(
            name=name,
            serial_number=serial_number,
            instance=instance,
            # TODO(gripper): Re-enable when gripper is integrated
            sim_plugin=sim_plugin,
            last_connected=False,
        )

        # Import URDF into the builder if a path is provided.
        if urdf_path and _os.path.isfile(urdf_path):
            pre_bodies = self._builder.body_count
            pre_joints = self._builder.joint_count

            _ts(f"Loading URDF: {urdf_path}")
            self._builder.add_urdf(
                urdf_path,
                floating=False,
                collapse_fixed_joints=False,
                enable_self_collisions=True,
            )

            post_bodies = self._builder.body_count
            post_joints = self._builder.joint_count

            _ts(f"URDF added: bodies +{post_bodies - pre_bodies}, joints +{post_joints - pre_joints}")


            _ts(f"WARNING: no valid URDF path for robot {name}")

        # Phase 8: Import gripper tool if configured.
        tool_config = rconf.get("tool")
        if tool_config and tool_config.get("urdf"):
            tool_urdf = tool_config["urdf"]
            if _os.path.isfile(tool_urdf):
                # TODO(gripper): Re-enable when gripper is integrated
                # self._load_gripper_tool(tool_urdf)
                pass

        # Phase 9: Import workpiece configuration for hydroelastic contact testing.
        workpiece_cfg = rconf.get("workpiece")
        if workpiece_cfg and workpiece_cfg.get("enabled"):
            self._add_workpiece(workpiece_cfg)

        # Store the robot; bind happens later in _build_scene after finalize.
        self._robots.append(robot)

    # TODO(gripper): _load_gripper_tool() commented out
    def _load_gripper_tool(self, tool_urdf: str) -> None:
        #         """
        #         Import a gripper tool URDF into the builder.

        #         The gripper is imported as a separate articulation that shares
        #         the same Newton model as the arm. This means finger joints will
        #         have DoF indices after the arm DoFs in the combined model.

        #         Args:
        #             tool_urdf: Path to the gripper URDF file.
        #         """
        _ts(f"Loading gripper tool URDF: {tool_urdf}")
        #         pre_bodies = self._builder.body_count
        #         pre_joints = self._builder.joint_count

        # Import the gripper as a separate articulation (not floating).
        # The Newton importer will create new bodies and joints for it.
        #         self._builder.add_urdf(
        #             tool_urdf,
        #             floating=False,
        #             collapse_fixed_joints=False,
        #             enable_self_collisions=True,
        #         )

        #         post_bodies = self._builder.body_count
        #         post_joints = self._builder.joint_count

        # TODO(gripper): commented out


        # Create the gripper controller instance on the last robot.
        #         if self._robots:
        #             robot = self._robots[-1]
            # Import FlexivGripperNewton from flexiv_serial_newton module
        #             FlexivGripperNewton = _fmod.FlexivGripperNewton
        #             robot.instance.gripper = FlexivGripperNewton()
        _ts(f"Gripper controller created for [{robot.name}]")
        pass

    def _add_workpiece(self, wp_cfg: dict) -> None:
        """
        Add a simple workpiece shape with SDF/hydroelastic enabled.

        Creates a mesh-based box (convex hull) so that SDF generation is triggered,
        enabling hydroelastic collision between workpiece and gripper fingertips.

        Args:
            wp_cfg: Dict with keys: position, half_size, hydroelastic (bool).
        """
        pos = wp_cfg.get("position", [0.0, 0.575, 0.0])
        hs = wp_cfg.get("half_size", [0.04, 0.04, 0.04])
        mass = wp_cfg.get("mass", 0.5)
        label = wp_cfg.get("label", "workpiece")

        # Build mesh vertices for a box centered at origin with given half-size.
        sx, sy, sz = hs[0], hs[1], hs[2]
        vertices = [
            (-sx,-sy,-sz),( sx,-sy,-sz),( sx, sy,-sz),(-sx, sy,-sz),
            (-sx,-sy, sz),( sx,-sy, sz),( sx, sy, sz),(-sx, sy, sz),
        ]
        indices = [
            0,1,2, 0,2,3, 4,6,5, 4,7,6, 0,4,5, 0,5,1,
            2,6,7, 2,7,3, 0,3,7, 0,7,4, 1,5,6, 1,6,2,
        ]

        workpiece_mesh = newton.Mesh(vertices=vertices, indices=indices)

        body = self._builder.add_body(
            xform=wp.transform(wp.vec3(pos[0], pos[1], pos[2]), wp.quat_identity()),
            mass=mass,
            label=label,
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig(gap=0.01)
        if wp_cfg.get("hydroelastic", False):
            # Enable SDF for hydroelastic contact (Phase 9).
            shape_cfg.configure_sdf(force_sdf=True)
            _ts(f"Workpiece [{label}] SDF/hydroelastic enabled")

        self._builder.add_shape_convex_hull(
            body=body, mesh=workpiece_mesh, cfg=shape_cfg, label=f"{label}_mesh",
        )
        _ts(f"Workpiece [{label}] added at pos=({pos[0]}, {pos[1]}, {pos[2]})")

    def _bind_robots(self) -> None:
        """
        Bind all robots to the finalized model, states, and control.

        Called once from _build_scene after builder.finalize().
        Also binds gripper instances (Phase 8).
        """
        for robot in self._robots:
            _ts(f"Binding robot [{robot.name}]...")
            robot.instance.bind(
                model=self._model,
                state_0=self._state_0,
                state_1=self._state_1,
                control=self._control,
            )

            # Phase 8: Bind gripper if present.
            # TODO(gripper): Re-enable when gripper is integrated
            # if robot.instance.gripper is not None and not robot.instance.gripper.is_bound:
                #                 robot.instance.gripper.bind(
                #                     model=self._model,
                #                     state_0=self._state_0,
                #                     state_1=self._state_1,
                #                 )
                # TODO(gripper): _ts call removed


            # Set home pose on startup so the arm does not fall under gravity.
            # The PD position hold (Phase 6) then keeps it stable until Elements
            # connects and switches to effort mode.
            robot.instance.teleport_to(self._initial_q)
            robot.instance.switch_control_mode("position")
            _ts(f"[{robot.name}] set to home pose, waiting for commands...")

    # ------------------------------------------------------------------
    # Physics step (mirrors original on_physics_step callback)
    # ------------------------------------------------------------------

    def _step_newton(self, dt: float) -> None:
        """
        Single Newton physics step: collide -> solve -> swap state buffers.

        Mirrors the original Isaac flow but with explicit Newton calls.
        Also handles PD hold torques (Phase 6) when disconnected.
        """
        # --- Torque application order (critical for stability) ---
        # Step A: Write Element/PD torques to Control.joint_f BEFORE collision+solve.
        #         Newton's solver reads Control.joint_f during solver.step().
        import numpy as _np_torque
        import warp as _wp_torque

        for robot in self._robots:
            if hasattr(robot, '_pending_torques') and robot._pending_torques is not None:
                # Element sent torque commands → apply them now (effort mode).
                robot.instance.apply_torques(robot._pending_torques)
                robot._pending_torques = None
            else:
                # No pending commands → apply PD hold torques if in position mode.
                hold_tau = robot.instance.compute_hold_torques()
                if any(t != 0.0 for t in hold_tau):
                    tau_buf = self._control.joint_f.numpy().copy()
                    dof_idx = robot.instance._dof_indices
                    if dof_idx is not None:
                        for j, idx in enumerate(dof_idx):
                            tau_buf[idx] = hold_tau[j]
                        self._control.joint_f.assign(
                            _wp_torque.array(dtype=_wp_torque.float32, device='cpu', data=tau_buf)
                        )

        # Step B: Collision detection (populates contact buffer from state_0).
        # Reset solver buffers on REAL reconnect (Phase 6).
        # Only reset if at least one physics step has already completed — prevents
        # clearing Control.joint_f on first-ever connection (Elements connected from startup).
        if self._did_reconnect and self._ever_ran_step:
            JOINT_F = newton.StateFlags.JOINT_F
            FORCE = newton.StateFlags.FORCE
            self._solver.reset(self._state_0, flags=JOINT_F | FORCE)
        self._did_reconnect = False

        # Mark that at least one step has completed (for reconnect tracking).
        if not self._ever_ran_step:
            self._ever_ran_step = True

        # Collision detection (collide populates self._contacts in place).
        t_collide_start = time.perf_counter()
        self._collision_pipeline.collide(self._state_0, self._contacts)
        t_collide_end = time.perf_counter()
        self._timing_stats["collision"].append(t_collide_end - t_collide_start)

        # Solver step: 5 positional args mandatory.
        t_solve_start = time.perf_counter()
        self._solver.step(
            self._state_0,
            self._state_1,
            self._control,
            self._contacts,
            dt,
        )
        t_solve_end = time.perf_counter()
        self._timing_stats["solver"].append(t_solve_end - t_solve_start)

        # Step D: Clear Control.joint_f after solver consumed it.
        #         This resets the buffer for the next cycle and prevents
        #         Newton 1.5.1 CPU corruption from accumulating stale values.
        f_zero = _wp_torque.array(
            dtype=_wp_torque.float32, device='cpu',
            data=_np_torque.zeros(self._model.joint_dof_count, dtype=_np_torque.float32)
        )
        self._control.joint_f.assign(f_zero)

        # Step E: SWAP state buffers — mandatory after solver.step().
        self._state_0, self._state_1 = self._state_1, self._state_0

        # Render viewer frame if active (Phase 11).
        if self._viewer is not None:
            try:
                self._viewer.log_state(self._state_0)
                self._viewer.begin_frame(self._servo_cycle * 0.0005)
                self._viewer.end_frame()
            except Exception:
                pass  # Viewer render errors should not crash the sim loop

    def on_physics_step(self, dt: float) -> None:
        """
        Full physics cycle: communication + Newton step.

        Communication order (mirrors original):
            1. Read q/dq from Newton state -> SendRobotStates to Elements
            2. WaitForRobotCommands from Elements
            3. Apply torques and gripper DOUT commands
        Then the Newton physics step is called separately.

        Returns:
            True if running, False if shutdown requested.
        """
        # --- Communication cycle ---
        t_comm_start = time.perf_counter()

        # Phase 1: Read state, blend, and send to Elements.
        _dbg_count = self._servo_cycle
        for robot in self._robots:
            try:
                import flexivsimplugin as _fsp
                import numpy as _np

                # --- Feedback blending (Phase 11 stability fix) ---
                # Read actual Newton physics state.
                q_phys = robot.instance.q
                dq_phys = robot.instance.dq

                # Blend with previously-fed-back values to keep Elements' internal
                # model close to what it expects. alpha=0..1 controls how much of
                # the raw physics feedback passes through per cycle.
                if robot._blend_q is not None:
                    alpha = self._blend_alpha
                    beta = self._blend_beta
                    blend_q = [alpha * q_phys[i] + (1.0 - alpha) * robot._blend_q[i] for i in range(robot.instance._arm_dof)]
                    blend_dq = [beta * dq_phys[i] + (1.0 - beta) * robot._blend_dq[i] for i in range(robot.instance._arm_dof)]
                    q_es_prev = list(robot._blend_q)  # What ES currently believes
                    dq_es_prev = list(robot._blend_dq)
                else:
                    blend_q = list(q_phys)
                    blend_dq = list(dq_phys)
                    q_es_prev = None
                    dq_es_prev = None
                robot._blend_q = list(blend_q)
                robot._blend_dq = list(blend_dq)

                # --- Debug logging (first 30 cycles or once/second) ---
                if _dbg_count < 30 or _dbg_count % 200 == 0:
                    tt = robot.instance.tau
                    m = robot.instance._mode
                    q_ok = all(_np.isfinite(v) for v in q_phys)
                    d_ok = all(_np.isfinite(v) for v in dq_phys)

                    # Log all 3 state sources: Newton, ES internal, and blended output.
                    # Uses _logger.debug so timestamps show correlation between cycles.
                    q_n_str = ','.join(f'{v:.3f}' for v in q_phys)
                    dq_n_str = ','.join(f'{v:.3f}' for v in dq_phys)
                    _ts(f"[DBG cycle {_dbg_count}] {robot.name} mode={m} "
                          f"q={'OK' if q_ok else 'BAD'} dq={'OK' if d_ok else 'BAD'} | "
                          f"Newton  q=[{q_n_str}]  dq=[{dq_n_str}]", prefix="[DBG]")


                    # Elements Studio internal state (what ES currently believes)
                    if q_es_prev is not None:
                        q_es_str = ','.join(f'{v:.3f}' for v in q_es_prev)
                        dq_es_str = ','.join(f'{v:.3f}' for v in dq_es_prev)
                        _ts(f"              ES      q=[{q_es_str}]  dq=[{dq_es_str}]", prefix="[DBG]")


                    # Blended values being sent to Elements + current control buffer
                    b_q_str = ','.join(f'{v:.3f}' for v in blend_q)
                    b_dq_str = ','.join(f'{v:.3f}' for v in blend_dq)
                    tau_str = ','.join(f'{v:.1f}' for v in tt)
                    _ts(f"              blend   q=[{b_q_str}]  dq=[{b_dq_str}]  tau=[{tau_str}]", prefix="[DBG]")


                if robot.instance.has_ft_sensor:
                    wrist_force, wrist_torque = robot.instance.wrist_wrench
                    states = _fsp.SimRobotStates(
                        servo_cycle=self._servo_cycle,
                        q=blend_q,
                        dq=blend_dq,
                        wrist_force=wrist_force,
                        wrist_torque=wrist_torque,
                    )
                else:
                    states = _fsp.SimRobotStates(
                        servo_cycle=self._servo_cycle,
                        q=blend_q,
                        dq=blend_dq,
                    )
                robot.sim_plugin.SendRobotStates(states)
            except Exception as e:
                _ts(f"SendRobotStates failed for {robot.name}: {e}")

        t_comm_end = time.perf_counter()
        self._timing_stats["send_states"].append(t_comm_end - t_comm_start)

        # Phase 2: Wait for commands from Elements.
        t_wait_start = time.perf_counter()
        connected_flags: List[bool] = []
        for robot in self._robots:
            try:
                ok = robot.sim_plugin.WaitForRobotCommands(100)
                connected_flags.append(ok)
            except Exception as e:
                import time as _time_now
                now = _time_now.time()
                if now - robot._last_wait_error > 1.0:
                    _ts(f"WaitForRobotCommands failed for {robot.name}: {e}")
                    robot._last_wait_error = now
                connected_flags.append(False)

        t_wait_end = time.perf_counter()
        self._timing_stats["wait_commands"].append(t_wait_end - t_wait_start)

        # Phase 3: Apply torque commands and handle DOUT gripper signals.
        t_apply_start = time.perf_counter()
        for idx, robot in enumerate(self._robots):
            is_connected = connected_flags[idx] if idx < len(connected_flags) else False
            was_connected = robot.last_connected

            if is_connected:
                # Read torque commands from Elements.
                try:
                    cmds = robot.sim_plugin.robot_commands()
                    torques = list(cmds.target_drives[:robot.instance._arm_dof])

                    # Clip incoming torques using config-driven per-joint limits.
                    # Limits are typically 10-25% of URDF max to prevent MuJoCo blowup
                    # since Newton has no mechanical impedance like a real robot.
                    _tau_limits = self._tau_limits_per_joint
                    torques = [max(-_tau_limits[i], min(_tau_limits[i], t))
                               for i, t in enumerate(torques)]

                    # Add velocity damping to stabilize effort mode.
                    # Newton/MuJoCo has no inherent joint damping; the real robot's
                    # mechanical impedance prevents oscillation but the simulation
                    # doesn't replicate this. Kd is tunable via config.yaml.
                    _dq = robot.instance.dq
                    torques = [torques[i] - self._kd_damp * _dq[i] for i in range(len(torques))]

                    # Check if still settling — DO NOT apply Element torques during this period.
                    # CRITICAL: We set _pending_torques to None (not [0]*7!) so that the
                    # control loop falls through to PD hold in _step_newton(). This keeps
                    # gravity compensation active and prevents arm drift during settling.
                    # Setting zeros would overwrite control.joint_f and kill PD hold, causing
                    # the arm to free-fall under gravity for the entire settle period.
                    now_now = time.perf_counter()
                    settle_elapsed = now_now - robot._settle_start
                    if robot._is_settling and settle_elapsed < self._settle_duration:
                        # Still settling — hold PD control, skip Element torques.
                        robot._pending_torques = None  # Let PD hold in _step_newton handle it
                        if settle_elapsed < 0.5 or settle_elapsed % 1.0 < 0.01:
                            _ts(f"[{robot.name}] settling ({settle_elapsed:.1f}s / {self._settle_duration:.1f}s)...")
                    else:
                        # Settling period over (or never started) — apply Element commands.
                        if robot._is_settling:
                            # First cycle past threshold — log once, teleport to home, switch mode.
                            _ts(f"[{robot.name}] settling complete, re-applying home pose and switching to effort mode.")
                            robot._is_settling = False
                            robot.instance.teleport_to(self._initial_q)
                            robot.instance.switch_control_mode("effort")
                        robot._pending_torques = list(torques)

                    # Handle gripper DOUT commands.
                    # TODO(gripper): Re-enable when gripper is integrated
                    # Handle gripper DOUT commands.
                    # if robot.instance.gripper is not None:
                        #                         dout = (
                        #                             robot.sim_plugin.robot_commands().digital_outputs
                        #                         )
                        #                         # DOUT[0] high -> open gripper.
                        #                         if len(dout) > 0 and dout[0]:
                        #                             if robot.gripper_status != GripperStatus.OPENED:
                        #                                 robot.instance.gripper.open()
                        #                                 robot.gripper_status = GripperStatus.OPENED
                        #                         # DOUT[1] high -> close gripper.
                        #                         if len(dout) > 1 and dout[1]:
                        #                             if robot.gripper_status != GripperStatus.CLOSED:
                        #                                 robot.instance.gripper.close()
                        #                                 robot.gripper_status = GripperStatus.CLOSED
                except Exception as e:
                    _ts(f"Applying commands failed for {robot.name}: {e}")

                # Transition from disconnected -> connected: start settling.
                # Mirrors MuJoCo bridge behavior: after Elements connects, hold zero
                # torques for kSettleSec seconds to let ES impedance control stabilize
                # before applying real commands. During settling the arm stays at home
                # pose via PD position hold (torques are zeroed each cycle).
                if not was_connected:
                    try:
                        robot._is_settling = True
                        robot._settle_start = time.perf_counter()
                        self._did_reconnect = True
                        _ts(f"Robot [{robot.name}] reconnected, settling for {self._settle_duration:.1f}s (PD hold active)...")
                    except Exception as e:
                        _ts(f"Reconnect failed for {robot.name}: {e}")

            else:
                # Disconnected: switch to position hold (Phase 6) and reset settling.
                if was_connected:
                    robot._is_settling = False  # Cancel any pending settle
                    try:
                        robot.instance.switch_control_mode("position")
                        robot.instance.teleport_to(robot.instance.q)
                        _ts(f"Robot [{robot.name}] disconnected, switched to position hold.")
                    except Exception as e:
                        _ts(f"Disconnect handling failed for {robot.name}: {e}")

                # Reset gripper status on disconnect.
# TODO(gripper): Re-enable when gripper is integrated
                # robot.gripper_status = GripperStatus.INIT

            robot.last_connected = is_connected

        # Debug logging AFTER command processing — shows actual Element commands received
        if _dbg_count < 30 or _dbg_count % 200 == 0:
            for robot in self._robots:
                q_phys_now = robot.instance.q
                dq_phys_now = robot.instance.dq
                m_now = robot.instance._mode
                import numpy as _np_dbg
                q_ok_now = all(_np_dbg.isfinite(v) for v in q_phys_now)

                # Show Element commands that were actually received (before clip/damp)
                try:
                    cmds_now = robot.sim_plugin.robot_commands()
                    raw_tau = list(cmds_now.target_drives[:robot.instance._arm_dof])
                    raw_str = ','.join(f'{v:.1f}' for v in raw_tau)
                except Exception:
                    raw_str = 'N/A'

                # Show pending torques (after clip/damp, about to be applied)
                if hasattr(robot, '_pending_torques') and robot._pending_torques is not None:
                    pend_str = ','.join(f'{v:.1f}' for v in robot._pending_torques)
                else:
                    pend_str = 'None (PD hold)'

                q_n_str_now = ','.join(f'{v:.3f}' for v in q_phys_now)
                dq_n_str_now = ','.join(f'{v:.3f}' for v in dq_phys_now)
                _ts(f"[DBG cycle {_dbg_count}] {robot.name} mode={m_now} "
                      f"q={'OK' if q_ok_now else 'BAD'} | "
                      f"Newton  q=[{q_n_str_now}]  dq=[{dq_n_str_now}]", prefix="[DBG]")

                if robot._blend_q is not None:
                    q_es_str_now = ','.join(f'{v:.3f}' for v in robot._blend_q)
                    _ts(f"              ES      q=[{q_es_str_now}]  "
                          f"raw_cmds={raw_str}  pending={pend_str}", prefix="[DBG]")

        t_apply_end = time.perf_counter()
        self._timing_stats["apply_torques"].append(t_apply_end - t_apply_start)

    def run(self) -> None:
        """
        Main simulation loop.

        Runs at PHYSICS_FREQ (2000 Hz) with dt = 0.0005 s per step.
        Communication cycle (on_physics_step) followed by Newton physics step.
        """
        dt = PHYSICS_DT

        # Log control parameters for debugging.
        _ts("Control params:")
        _ts(f"  blend alpha={self._blend_alpha}, beta={self._blend_beta}")
        _ts(f"  tau_limits_per_joint={self._tau_limits_per_joint}")
        _ts(f"  kd_damp={self._kd_damp} Nm/(rad/s)")
        _ts(f"  hold_kp={self._hold_kp}, hold_kd={self._hold_kd}, hold_max={self._hold_max_torque}")
        _ts(f"  gravity_disabled={self._gravity_disable}")

        _ts(f"Starting main loop at {PHYSICS_FREQ} Hz (dt={dt*1000:.3f} ms)")
        _ts(f"App version: {APP_VERSION}")
        _ts(f"Robots: {[r.name for r in self._robots]}")

        while self._running:
            t_cycle_start = time.perf_counter()

            # 1. Communication cycle (q/dq -> Elements, receive torque/DOUT).
            self.on_physics_step(dt)

            # 2. Newton physics step (collide + solve + swap).
            self._step_newton(dt)

            # Increment servo cycle counter.
            self._servo_cycle += 1

            t_cycle_end = time.perf_counter()
            cycle_ms = (t_cycle_end - t_cycle_start) * 1000
            self._timing_stats["cycle_total"].append(cycle_ms)

        # Print timing summary on exit.
        self._print_timing_summary()

    def _print_timing_summary(self) -> None:
        """Print aggregated timing diagnostics."""
        _ts("Timing summary:")
        for key, values in self._timing_stats.items():
            if values:
                mean_ms = (sum(values) / len(values)) * 1000
                max_ms = max(values) * 1000
                p95_idx = int(len(values) * 0.95)
                sorted_vals = sorted(values)
                p95_ms = sorted_vals[p95_idx] * 1000 if p95_idx < len(sorted_vals) else 0
                _ts(f"  {key:<20s}: mean={mean_ms:.3f} ms, max={max_ms:.3f} ms, p95={p95_ms:.3f} ms")

        # servo_cycle counter
        _ts(f"  servo_cycles    : {self._servo_cycle}")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def run_with_steps(self, step_count: int) -> dict:
        """
        Run a fixed number of Newton physics steps (no DDS communication).
        """
        dt = PHYSICS_DT
        cycle_times: List[float] = []
        read_state_ms: List[float] = []
        collide_ms: List[float] = []
        solve_ms: List[float] = []

        _ts(f"Running {step_count} lockstep cycles (dt={dt*1000:.3f} ms, Newton-only)")

        for _ in range(step_count):
            t_start = time.perf_counter()

            # Read q/dq from current state.
            t0 = time.perf_counter()
            for robot in self._robots:
                _ = robot.instance.q
                _ = robot.instance.dq
            read_state_ms.append((time.perf_counter() - t0) * 1000)

            # Newton physics step.
            self._step_newton(dt)

            cycle_wall = (time.perf_counter() - t_start) * 1000
            cycle_times.append(cycle_wall)

        collide_raw = self._timing_stats.get("collision", [])
        solve_raw = self._timing_stats.get("solver", [])

        total_wall_s = sum(cycle_times) / 1000.0 if cycle_times else 1.0
        effective_hz = step_count / total_wall_s
        rtf = effective_hz / PHYSICS_FREQ

        result = {
            "step_count": step_count,
            "mean_cycle_ms": sum(cycle_times) / len(cycle_times),
            "max_cycle_ms": max(cycle_times),
            "mean_read_ms": sum(read_state_ms) / len(read_state_ms),
            "mean_collide_ms": (sum(collide_raw[-step_count:]) * 1000 if collide_raw else 0.0),
            "mean_solve_ms": (sum(solve_raw[-step_count:]) * 1000 if solve_raw else 0.0),
            "effective_hz": effective_hz,
            "rtf": rtf,
        }

        _ts("Lockstep timing:")
        for k, v in result.items():
            unit = "" if not isinstance(v, float) else (" Hz" if "hz" in k else (" ms" if "ms" in k else ""))
            _ts(f"  {k:<16s}: {v:>10.3f}{unit}")

        return result

    def shutdown(self) -> None:
        """Clean up and exit."""
        self._running = False
        _ts("Shutting down.")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def resolve_urdf_paths(config: dict, workspace_root: str) -> dict:
    """Resolve relative URDF paths in the config against workspace_root."""

    def resolve(path):
        if path and not _os.path.isabs(path):
            return _os.path.join(workspace_root, path)
        return path

    for robot in config.get("robots", []):
        if robot.get("urdf"):
            robot["urdf"] = resolve(robot["urdf"])
        tool = robot.get("tool")
        if tool and tool.get("urdf"):
            tool["urdf"] = resolve(tool["urdf"])
    return config


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flexiv Direct-Newton Bridge",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file describing robots and scene.",
    )
    parser.add_argument(
        "--viewer",
        choices=["null", "gl", "rtx", "viser"],
        default="null",
        help="Render viewer backend (default: null = no window). "
             "Use 'gl' for OpenGL, 'rtx' for RTX ray-tracing, 'viser' for web-based.",
    )
    args = parser.parse_args()

    # Load config.
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Resolve relative paths.
    workspace_root = _os.path.abspath(
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")
    )
    config = resolve_urdf_paths(config, workspace_root)

    # Create and run the bridge (viewer_backend passed to constructor).
    runner = NewtonBridgeRunner(config, viewer_backend=args.viewer)

    try:
        runner.run()
    except KeyboardInterrupt:
        _ts("Interrupted.")
        runner.shutdown()


if __name__ == "__main__":
    main()
