# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Flexiv-Isaac Bridge App -- Newton physics backend (Phases 1-7).

Usage::

    isaac-py standalone_examples/api/isaacsim.robot.manipulators/flexiv/\\
        flexiv_isaac_bridge_app_newton.py \\
        --config standalone_examples/api/isaacsim.robot.manipulators/flexiv/\\
                     single_arm_app_config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import spdlog

# =====================================================================
# 1. Parse CLI args and launch SimulationApp (MUST be first)
# =====================================================================

argparser = argparse.ArgumentParser(description="Flexiv Isaac Bridge App (Newton)")
argparser.add_argument("--config", required=True, help="Path to YAML config file")
args_cli = argparser.parse_args()

from isaacsim import SimulationApp

_run_headless = True if os.environ.get("LIVY_SESSION_ID") else not bool(
    os.environ.get("DISPLAY", "") and os.path.exists(os.environ.get("DISPLAY", ""))
)
_run_headless = False
simulation_app = SimulationApp({
    "headless": _run_headless, "width": 1920, "height": 1080,
})

time.sleep(0.5)

# =====================================================================
# 2. Enable Newton extensions (Phase 1 -- Bootstrap)
# =====================================================================

from isaacsim.core.utils.extensions import enable_extension

NewtonBridge_logger = spdlog.ConsoleLogger("NewtonBridge")

for ext in [
    "isaacsim.robot.manipulators.examples",
    "isaacsim.physics.newton",
    "isaacsim.physics.newton.tensors",
]:
    if not enable_extension(ext):
        NewtonBridge_logger.error(f"[NewtonBridge] Failed to enable extension: {ext}")
    else:
        NewtonBridge_logger.info(f"[NewtonBridge] Enabled extension: {ext}")

simulation_app.update()

from isaacsim.core.simulation_manager import SimulationManager

active_engine = SimulationManager.get_active_physics_engine()
if active_engine != "newton":
    NewtonBridge_logger.warn(
        f"[NewtonBridge] Active engine is '{active_engine}', switching to newton"
    )
    result = SimulationManager.switch_physics_engine("newton", verbose=True)
    if not result:
        NewtonBridge_logger.error("[NewtonBridge] Failed to switch physics engine to newton")
    else:
        NewtonBridge_logger.info("[NewtonBridge] Physics engine switched to newton")
else:
    NewtonBridge_logger.info(f"[NewtonBridge] Active physics engine already newton")

final_engine = SimulationManager.get_active_physics_engine()
NewtonBridge_logger.info(f"[NewtonBridge] Confirmed active engine: {final_engine}")

if final_engine != "newton":
    raise RuntimeError(
        f"Physics engine is '{final_engine}', not 'newton'. "
        "Cannot proceed with Newton bridge."
    )

# =====================================================================
# 3. Load Isaac modules
# =====================================================================

import yaml
import importlib

_script_dir = os.path.dirname(os.path.abspath(__file__))

from pxr import Gf, Usd, UsdGeom, UsdPhysics, Sdf

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage

# Load our workspace copy of FlexivSerialNewton directly by absolute path.
# isaac-py caches isaacsim.robot.manipulators.examples in sys.modules during
# startup, so a normal package import would hit the stale Isaac install copy
# that lacks our flexiv_serial_newton module.
_flexiv_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
                 "exts", "isaacsim.robot.manipulators.examples",
                 "isaacsim", "robot", "manipulators", "examples", "flexiv")
)
_fspec = importlib.util.spec_from_file_location(
    "flexiv_serial_newton", os.path.join(_flexiv_dir, "flexiv_serial_newton.py")
)
_fmod = importlib.util.module_from_spec(_fspec)
_fspec.loader.exec_module(_fmod)
FlexivSerialNewton = _fmod.FlexivSerialNewton

# =====================================================================
# 4. Flexiv Sim Plugin imports (same API as PhysX bridge)
# =====================================================================

import flexivsimplugin


# =====================================================================
# 5. Timing diagnostics (Phase 4)
# =====================================================================

class _TimingDiagnostics:
    """Lightweight ring-buffer diagnostics for the bridge callback loop."""

    def __init__(self, print_interval_sec: float = 2.0):
        self._interval = print_interval_sec
        self._last_print = time.monotonic()
        self._step_times_us: List[float] = []
        self._missed_commands = 0
        self._lock = threading.Lock()

    def record_step(self, duration_ms: float) -> None:
        with self._lock:
            self._step_times_us.append(duration_ms * 1_000)
            if len(self._step_times_us) > 5000:
                self._step_times_us = self._step_times_us[-2500:]

    def record_missed(self) -> None:
        with self._lock:
            self._missed_commands += 1

    def print_report(self, label: str = "Timing") -> None:
        now = time.monotonic()
        if now - self._last_print < self._interval:
            return
        self._last_print = now
        with self._lock:
            if not self._step_times_us:
                return
            recent = self._step_times_us[-500:]
            mean_us = float(np.mean(recent))
            max_us = float(np.max(recent))
            count = len(self._step_times_us)
            missed = self._missed_commands
        NewtonBridge_logger.info(
            f"[{label}] steps={count}  mean_step={mean_us:.0f}us  "
            f"max_step={max_us:.0f}us  missed_cmds={missed}"
        )

    def reset(self) -> None:
        with self._lock:
            self._step_times_us.clear()
            self._missed_commands = 0


_timing = _TimingDiagnostics(print_interval_sec=2.0)


# =====================================================================
# 6. Gripper status enum and profiles (Phase 6, same as PhysX bridge)
# =====================================================================

class GripperStatus(Enum):
    INIT = 0
    OPENED = 1
    CLOSED = 2


GRIPPER_PROFILES = {
    "Grav_gripper": {
        "ee": "right_finger_tip",
        "mount_body": "gripper_base",
        "joints": ["finger_joint", "right_outer_knuckle_joint"],
        "opened": [45.0, 0.0],
        "closed": [-8.88, 0.0],
    },
}


# =====================================================================
# 7. Config resolution helper (same as PhysX bridge)
# =====================================================================

def resolve_usd_paths(config: dict) -> dict:
    """Resolve relative USD paths against the workspace root."""
    _WORKSPACE_ROOT = os.path.abspath(
        os.path.join(_script_dir, "..", "..", "..", "..")
    )

    def resolve(path):
        if path and not os.path.isabs(path):
            return os.path.join(_WORKSPACE_ROOT, path)
        return path

    if config.get("env_usd"):
        config["env_usd"] = resolve(config["env_usd"])
    for robot in config.get("robots", []):
        if robot.get("usd"):
            robot["usd"] = resolve(robot["usd"])
        tool = robot.get("tool")
        if tool and tool.get("usd"):
            tool["usd"] = resolve(tool["usd"])
    return config


# =====================================================================
# 8. Bridge runner (Newton version, Phases 1-7)
# =====================================================================

class NewtonBridgeRunner:
    """Set up world and run Newton-based joint impedance control."""

    ROBOT_DOF = 7

    def __init__(
        self,
        physics_dt: float,
        render_dt: float,
        config: dict,
        initial_q: List[float] = None,
    ) -> None:
        self._logger = spdlog.ConsoleLogger("Flexiv-Isaac Newton Bridge App")

        self._logger.info("-" * 62)
        self._logger.info(
            "-     Flexiv-Isaac Bridge App (Newton Backend) v1.0       -"
        )
        self._logger.info("-" * 62)

        self._initial_q = initial_q or [0.0] * self.ROBOT_DOF
        self._servo_cycle = 0
        self._reset_needed = False
        self._pending_tool_profiles: List[dict] = []

        if config.get("gpu_dynamics", False):
            self._logger.warn(
                "[NewtonBridge] gpu_dynamics is a PhysX option; ignoring for Newton"
            )

        self._world = World(
            stage_units_in_meters=1.0,
            physics_dt=physics_dt,
            rendering_dt=render_dt,
            set_defaults=False,
        )

        env_usd = config.get("env_usd", "")
        if env_usd:
            add_reference_to_stage(usd_path=env_usd, prim_path="/World")
        else:
            self._world.scene.add_default_ground_plane()

        # Add robot data structs.
        self._robots = []
        for r in config.get("robots", []):
            serial_num = r["serial_number"]
            usd_path = r["usd"]
            pos_in_world = [float(r["position"][i]) for i in ["x", "y", "z"]]
            ori_in_world = [float(r["orientation"][i]) for i in ["w", "x", "y", "z"]]

            model_token = serial_num.split("-")[0].strip().lower().replace(" ", "")
            has_ft_sensor = model_token in ("rizon4s", "rizon10s")
            if has_ft_sensor:
                self._logger.info(
                    f"[NewtonBridge] Robot [{serial_num}] is an 's' variant; "
                    f"wrist F/T sensor enabled"
                )

            serial_name = serial_num.replace("-", "_")
            prim_path = "/World/Flexiv/" + serial_name

            self._logger.info(
                f"[NewtonBridge] Adding robot usd [{usd_path}] "
                f"at prim path [{prim_path}]"
            )
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

            end_effector_prim_name = "flange"
            gripper = None
            tool_cfg = r.get("tool")
            if tool_cfg:
                try:
                    gripper, end_effector_prim_name = self._attach_tool(
                        prim_path, tool_cfg
                    )
                except Exception as e:
                    self._logger.error(
                        f"[NewtonBridge] Tool attach failed for "
                        f"[{serial_num}]: {e}"
                    )

            robot_instance = FlexivSerialNewton(
                prim_path=prim_path,
                name=serial_name,
                end_effector_prim_name=end_effector_prim_name,
                arm_dof=self.ROBOT_DOF,
                pos_in_world=pos_in_world,
                ori_in_world=ori_in_world,
                has_ft_sensor=has_ft_sensor,
            )

            prim = get_current_stage().GetPrimAtPath(prim_path)
            if prim.IsValid():
                xf = UsdGeom.Xform(prim)
                xf.AddTranslateOp().Set(Gf.Vec3d(*pos_in_world))
                xf.AddOrientOp().Set(Gf.Quatf(*ori_in_world))

            sim_plugin = flexivsimplugin.UserNode(serial_num)

            self._robots.append(
                {
                    "name": serial_name,
                    "instance": robot_instance,
                    "sim_plugin": sim_plugin,
                    "last_connected": False,
                    "gripper_status": GripperStatus.INIT,
                }
            )

            self._logger.info(
                f"[NewtonBridge] Robot registered: {serial_name} at {prim_path}"
            )

        simulation_app.update()
        self._world.reset()
        self._logger.info("[NewtonBridge] World reset complete")

        for robot in self._robots:
            robot["instance"].initialize()
            self._logger.info(
                f"[NewtonBridge] Robot [{robot['name']}] initialized"
            )

        # Initialize grippers (Phase 6B).
        for profile_info in self._pending_tool_profiles:
            rpath = profile_info["prim_path"]
            profile = profile_info["profile"]
            found_robot = None
            for robot in self._robots:
                if robot["instance"]._prim_path == rpath:
                    found_robot = robot
                    break
            if found_robot is not None:
                joint_names = list(profile["joints"])
                opened_rad = [np.deg2rad(p) for p in profile["opened"]]
                closed_rad = [np.deg2rad(p) for p in profile["closed"]]
                found_robot["instance"].initialize_gripper(
                    joint_prim_names=joint_names,
                    opened_positions=opened_rad,
                    closed_positions=closed_rad,
                )
            else:
                self._logger.warn(
                    f"[NewtonBridge] Could not find robot instance "
                    f"for prim_path [{rpath}] to initialize gripper."
                )

        self._world.add_physics_callback(
            "robot_step", callback_fn=self.on_physics_step
        )

        for robot in self._robots:
            robot["instance"].teleport_to(self._initial_q)
            self._logger.info(
                f"[NewtonBridge] Robot [{robot['name']}] "
                f"teleported to initial q={self._initial_q}"
            )

    # ---------------------------------------------------------------
    # Tool attachment (Phase 6A)
    # ---------------------------------------------------------------

    @staticmethod
    def _find_flange_path(robot_prim_path: str) -> str:
        """Resolve the flange prim path under a robot."""
        stage = get_current_stage()
        direct = robot_prim_path + "/flange"
        if stage.GetPrimAtPath(direct).IsValid():
            return direct
        root = stage.GetPrimAtPath(robot_prim_path)
        for prim in Usd.PrimRange(root):
            if prim.GetName() == "flange":
                return str(prim.GetPath())
        raise RuntimeError(f"No flange prim found under [{robot_prim_path}]")

    def _attach_tool(self, robot_prim_path: str, tool_cfg: dict) -> tuple:
        """Reference a tool USD onto the arm and fix it to the flange."""
        usd_path = tool_cfg["usd"]
        prim_name = tool_cfg["prim_name"]

        profile = GRIPPER_PROFILES.get(prim_name)
        if profile is None:
            raise ValueError(
                f"No gripper profile for [{prim_name}]. "
                f"Known profiles: {sorted(GRIPPER_PROFILES)}"
            )

        tool_prim_path = robot_prim_path + "/" + prim_name
        self._logger.info(
            f"[NewtonBridge] Attaching tool usd [{usd_path}] at "
            f"prim path [{tool_prim_path}]"
        )
        add_reference_to_stage(usd_path=usd_path, prim_path=tool_prim_path)

        stage = get_current_stage()
        flange_path = self._find_flange_path(robot_prim_path)
        mount_body_path = tool_prim_path + "/" + profile["mount_body"]
        mount_joint_path = tool_prim_path + "/flange_to_" + profile["mount_body"]

        mount = UsdPhysics.FixedJoint.Define(stage, mount_joint_path)
        mount.CreateBody0Rel().SetTargets([Sdf.Path(flange_path)])
        mount.CreateBody1Rel().SetTargets([Sdf.Path(mount_body_path)])
        mount_prim = mount.GetPrim()
        mount_prim.CreateAttribute(
            "physics:localPos0", Sdf.ValueTypeNames.Point3f
        ).Set(Gf.Vec3f(0, 0, 0))
        mount_prim.CreateAttribute(
            "physics:localPos1", Sdf.ValueTypeNames.Point3f
        ).Set(Gf.Vec3f(0, 0, 0))
        mount_prim.CreateAttribute(
            "physics:localRot0", Sdf.ValueTypeNames.Quatf
        ).Set(Gf.Quatf(1, 0, 0, 0))
        mount_prim.CreateAttribute(
            "physics:localRot1", Sdf.ValueTypeNames.Quatf
        ).Set(Gf.Quatf(1, 0, 0, 0))

        end_effector_prim_name = prim_name + "/" + profile["ee"]
        self._logger.info(
            f"[NewtonBridge] Tool [{prim_name}] attached; "
            f"gripper control enabled (ee=[{end_effector_prim_name}])"
        )

        self._pending_tool_profiles.append({
            "prim_path": robot_prim_path,
            "profile": profile,
        })

        return None, end_effector_prim_name

    # ---------------------------------------------------------------
    # Physics callback
    # ---------------------------------------------------------------

    def on_physics_step(self, dt: float) -> None:
        """Physics callback for the joint impedance control loop."""
        t_start = time.monotonic()

        for robot in self._robots:
            try:
                if robot["instance"].has_ft_sensor:
                    wrist_force, wrist_torque = robot["instance"].wrist_wrench
                    robot_states = flexivsimplugin.SimRobotStates(
                        self._servo_cycle,
                        robot["instance"].q,
                        robot["instance"].dq,
                        wrist_force,
                        wrist_torque,
                    )
                else:
                    robot_states = flexivsimplugin.SimRobotStates(
                        self._servo_cycle,
                        robot["instance"].q,
                        robot["instance"].dq,
                    )
                robot["sim_plugin"].SendRobotStates(robot_states)
            except Exception as e:
                self._logger.error(
                    f"[NewtonBridge] Failed to send state for "
                    f"robot [{robot['name']}]: {e}"
                )

        for robot in self._robots:
            if robot["sim_plugin"].connected():
                if not robot["last_connected"]:
                    self._logger.info(
                        f"[NewtonBridge] Connected to robot [{robot['name']}]"
                    )
                    robot["instance"].switch_control_mode("effort")

                timeout_ms = 100
                if robot["sim_plugin"].WaitForRobotCommands(timeout_ms):
                    cmds = robot["sim_plugin"].robot_commands()
                    target_drives = getattr(cmds, "target_drives", None)
                    if target_drives is not None:
                        try:
                            robot["instance"].apply_torques(list(target_drives))
                        except Exception as e:
                            self._logger.error(
                                f"[NewtonBridge] Failed to apply torques "
                                f"for [{robot['name']}]: {e}"
                            )

                    dout_list = list(getattr(cmds, "digital_outputs", []))
                    if dout_list and robot["instance"].gripper is not None:
                        if dout_list[0]:
                            if robot["gripper_status"] != GripperStatus.OPENED:
                                self._logger.info(
                                    f"[NewtonBridge] Opening gripper for "
                                    f"robot [{robot['name']}]"
                                )
                                robot["instance"].gripper.open()
                                robot["gripper_status"] = GripperStatus.OPENED

                        if dout_list[1]:
                            if robot["gripper_status"] != GripperStatus.CLOSED:
                                self._logger.info(
                                    f"[NewtonBridge] Closing gripper for "
                                    f"robot [{robot['name']}]"
                                )
                                robot["instance"].gripper.close()
                                robot["gripper_status"] = GripperStatus.CLOSED

                else:
                    self._logger.warn(
                        f"[NewtonBridge] Missed 1 message from [{robot['name']}]"
                    )
                    _timing.record_missed()

                robot["last_connected"] = True

            else:
                if robot["last_connected"]:
                    self._logger.error(
                        f"[NewtonBridge] Disconnected from robot [{robot['name']}]"
                    )
                    robot["instance"].switch_control_mode("position")
                    try:
                        current_q = robot["instance"].q
                        robot["instance"].teleport_to(current_q)
                    except Exception:
                        robot["instance"].teleport_to(self._initial_q)
                    robot["gripper_status"] = GripperStatus.INIT

                robot["last_connected"] = False

        self._servo_cycle += 1

        t_end = time.monotonic()
        _timing.record_step((t_end - t_start) * 1000)
        _timing.print_report("NewtonBridge")

    # ---------------------------------------------------------------
    # Run loop
    # ---------------------------------------------------------------

    def run(self) -> None:
        """Poll world step, which steps physics and rendering."""
        self._logger.info("[NewtonBridge] Entering simulation loop")

        while simulation_app.is_running():
            self._world.step(render=True)

            if self._world.is_stopped() and not self._reset_needed:
                self._reset_needed = True
                for robot in self._robots:
                    robot["last_connected"] = False
            if self._world.is_playing():
                if self._reset_needed:
                    self._world.reset()
                    self._reset_needed = False
                    for robot in self._robots:
                        robot["instance"].initialize()
                        robot["instance"].switch_control_mode("position")
                        robot["instance"].teleport_to(self._initial_q)
                    self._logger.info(
                        "[NewtonBridge] World reset; robots re-initialized"
                    )


# =====================================================================
# 9. Main entry point
# =====================================================================

RENDER_FREQ = 60.0
PHYSICS_FREQ = 2000.0


def main() -> None:
    config_path = args_cli.config
    if not os.path.isfile(config_path):
        alt = os.path.join(_script_dir, config_path)
        if os.path.isfile(alt):
            config_path = alt
        else:
            raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = resolve_usd_paths(yaml.safe_load(open(config_path)))

    runner = NewtonBridgeRunner(
        physics_dt=1.0 / PHYSICS_FREQ,
        render_dt=1.0 / RENDER_FREQ,
        config=cfg,
        initial_q=[0.0, -0.698132, 0.0, 1.5708, 0.0, 0.698132, 0.0],
    )

    isaac_version = "6.0.1"
    for robot in runner._robots:
        robot["instance"].log_startup_info(
            isaac_version,
            dt_physics=runner._world.get_physics_dt(),
            dt_render=runner._world.get_rendering_dt(),
        )

    runner.run()
    simulation_app.close()


if __name__ == "__main__":
    main()

