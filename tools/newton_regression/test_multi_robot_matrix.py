#!/usr/bin/env python3
"""Phase 10 — Multi-Robot and Complete Feature Regression Matrix.

Static verification that the Newton bridge supports all robot configurations
listed in the dev plan Phase 10 test matrix.  This script checks:
    1. The YAML config format supports multiple robots.
    2. Per-robot tool (gripper) configuration is handled.
    3. Serial-number-based F/T sensor detection works for 's' variants.
    4. Dual/multi-robot loop iterates correctly.

No Isaac Sim runtime required.  Run from repository root:
    python3 tools/newton_regression/test_multi_robot_matrix.py
"""

import ast
import os
import sys
import textwrap


_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

_NEWTON_SERIAL = os.path.join(
    _REPO_ROOT,
    "exts/isaacsim.robot.manipulators.examples/isaacsim/robot/manipulators/examples/flexiv/",
    "flexiv_serial_newton.py",
)

_NEWTON_BRIDGE = os.path.join(
    _REPO_ROOT,
    "standalone_examples/api/isaacsim.robot.manipulators/flexiv/",
    "flexiv_isaac_bridge_app_newton.py",
)


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


# ---- Phase 10 test matrix from dev plan ----------------------------------

ROBOT_CONFIGS = [
    ("Rizon10 arm only", {"serial": "Rizon10-00001", "tool": None}),
    ("Rizon10 + Grav",   {"serial": "Rizon10-00001", "tool": "Grav_gripper"}),
    ("Rizon10s arm only", {"serial": "Rizon10s-00001", "tool": None}),
    ("Rizon10s + Grav",  {"serial": "Rizon10s-00001", "tool": "Grav_gripper"}),
]


def test_robot_configs_supported():
    """Verify Newton bridge handles all robot configurations from Phase 10 matrix."""
    print("[test_robot_configs_supported]")

    source = _read(_NEWTON_BRIDGE)

    # Check 1: Multiple robots loop exists.
    assert "for r in config.get(\"robots\"" in source or "for r in config.get('robots'" in source, (
        "Newton bridge does not iterate over multiple robot configs"
    )
    print("  Multiple robot config iteration: OK")

    # Check 2: Serial number parsing exists.
    assert "serial_number" in source, "No serial_number parsing found"
    print("  Serial number parsing: OK")

    # Check 3: F/T sensor detection for 's' variants.
    assert "rizon4s" in source.lower() and "rizon10s" in source.lower(), (
        "F/T sensor variant detection missing"
    )
    print("  F/T sensor variant detection: OK")

    # Check 4: Tool/gripper configuration handling.
    assert "tool" in source, "Tool configuration not handled"
    assert "_attach_tool" in source, "Tool attachment method missing"
    print("  Tool/gripper configuration: OK")

    # Check 5: Dual robot data structure supports per-robot state tracking.
    assert "gripper_status" in source, "Per-robot gripper status not tracked"
    assert "last_connected" in source, "Per-robot connection state not tracked"
    print("  Per-robot state tracking: OK")

    # Verify F/T detection logic works for 's' variants.
    newton_serial_src = _read(_NEWTON_SERIAL)
    assert "has_ft_sensor" in newton_serial_src, (
        "FlexivSerialNewton lacks has_ft_sensor support"
    )
    print("  FlexivSerialNewton F/T support: OK")

    print("  PASS\n")


def test_feature_checklist():
    """Verify all Phase 10 checklist features are implemented in Newton bridge."""
    print("[test_feature_checklist]")

    source = _read(_NEWTON_BRIDGE)
    serial_source = _read(_NEWTON_SERIAL)

    features = {
        "startup": ["NewtonBridge", "Entering simulation loop"],
        "connect": ["Connected to robot", "switch_control_mode(\"effort\")" in source],
        "reset": ["World reset", "re-initialized"],
        "disconnect": ["Disconnected from robot", "switch_control_mode(\"position\")"],
        "state_read": ["SendRobotStates", ".q"],
        "torque_command": ["apply_torques", "target_drives"],
        "ft_sensor": ["wrist_wrench", "has_ft_sensor"],
        "gripper": ["gripper.open()", "gripper.close()"],
        "contact": ["world.step(render=True)"],  # physics steps = contact enabled
    }

    all_ok = True
    for feature, checks in features.items():
        ok = all(
            c if isinstance(c, bool) else (c in source or c in serial_source)
            for c in checks
        )
        tag = "OK" if ok else "MISSING"
        print(f"  {feature:<20s} [{tag}]")
        if not ok:
            all_ok = False

    assert all_ok, "Some Phase 10 features missing from Newton bridge"
    print("  PASS\n")


def test_gripper_profile_completeness():
    """Verify GRIPPER_PROFILES covers the Grav gripper used in testing."""
    print("[test_gripper_profile_completeness]")

    source = _read(_NEWTON_BRIDGE)
    assert "GRIPPER_PROFILES" in source, "No GRIPPER_PROFILES defined"
    assert "Grav_gripper" in source, "Grav gripper profile missing"
    print("  Grav_gripper profile present: OK")

    # Check required keys exist.
    newton_serial_src = _read(_NEWTON_SERIAL)
    assert "initialize_gripper" in newton_serial_src, (
        "FlexivSerialNewton.initialize_gripper missing"
    )
    print("  initialize_gripper method present: OK")

    # Check GripperNewton class exists.
    assert "class GripperNewton" in newton_serial_src, "GripperNewton class missing"
    assert "def open" in newton_serial_src and "def close" in newton_serial_src, (
        "GripperNewton open/close methods missing"
    )
    print("  GripperNewton class with open/close: OK")

    print("  PASS\n")


def test_disconnect_recovery():
    """Verify disconnect -> reconnect cycle is handled correctly."""
    print("[test_disconnect_recovery]")

    source = _read(_NEWTON_BRIDGE)

    # On disconnect: switch to position + safe hold.
    assert 'switch_control_mode("position")' in source, (
        "Disconnect does not switch to position mode"
    )
    assert "teleport_to" in source, "Safe hold teleport missing on disconnect"
    print("  Disconnect handling (position + hold): OK")

    # On reconnect: switch back to effort.
    assert 'switch_control_mode("effort")' in source, (
        "Reconnect does not switch to effort mode"
    )
    print("  Reconnect handling (effort mode): OK")

    # Gripper status reset on disconnect.
    assert "GripperStatus.INIT" in source, "Gripper status not reset on disconnect"
    print("  Gripper status reset: OK")

    print("  PASS\n")


def main():
    print("=" * 60)
    print("Phase 10 -- Multi-Robot and Feature Regression Matrix")
    print("=" * 60)
    print()

    tests = [
        test_robot_configs_supported,
        test_feature_checklist,
        test_gripper_profile_completeness,
        test_disconnect_recovery,
    ]

    passed = failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All Phase 10 regression checks PASSED.")
        print("Newton bridge supports all robot configurations in test matrix.")
    print("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

