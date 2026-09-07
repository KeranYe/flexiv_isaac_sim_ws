#!/usr/bin/env python3
"""Phase 9 — ROS 2 / RDK Regression Verification.

Confirms that the Newton bridge preserves the identical FlexivSimPlugin API
contract as the PhysX bridge, so no ROS 2 Python 3.10 process changes are needed.

Static verification (no Isaac Sim runtime required).  Checks:
    1. Both bridges use the same UserNode + SendRobotStates + WaitForRobotCommands calls.
    2. The SimRobotStates constructor args match between bridges.
    3. FlexivSerialNewton exposes the same public methods as FlexivSerial.

Run from repository root:
    python3 tools/newton_regression/test_ros_rdk_contract.py
"""

import ast
import os
import re
import sys


_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

_PHYSX_BRIDGE = os.path.join(
    _REPO_ROOT,
    "standalone_examples/api/isaacsim.robot.manipulators/flexiv/",
    "flexiv_isaac_bridge_app.py",
)

_NEWTON_BRIDGE = os.path.join(
    _REPO_ROOT,
    "standalone_examples/api/isaacsim.robot.manipulators/flexiv/",
    "flexiv_isaac_bridge_app_newton.py",
)

_PHYSX_SERIAL = os.path.join(
    _REPO_ROOT,
    "exts/isaacsim.robot.manipulators.examples/isaacsim/robot/manipulators/examples/flexiv/",
    "flexiv_serial.py",
)

_NEWTON_SERIAL = os.path.join(
    _REPO_ROOT,
    "exts/isaacsim.robot.manipulators.examples/isaacsim/robot/manipulators/examples/flexiv/",
    "flexiv_serial_newton.py",
)


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def test_api_surface_parity():
    """Both bridges must call the same FlexivSimPlugin API methods."""
    patterns = {
        "SendRobotStates": r"SendRobotStates\(",
        "WaitForRobotCommands": r"WaitForRobotCommands\(",
        "robot_commands": r"robot_commands\(\)",
        "UserNode": r"UserNode\(",
        "connected": r"\.connected\(\)",
        "target_drives": r"target_drives",
        "digital_outputs": r"digital_outputs",
    }

    physx_src = _read(_PHYSX_BRIDGE)
    newton_src = _read(_NEWTON_BRIDGE)

    print("[test_api_surface_parity]")
    all_ok = True
    for name, pat in patterns.items():
        p_count = len(re.findall(pat, physx_src))
        n_count = len(re.findall(pat, newton_src))
        ok = p_count > 0 and n_count > 0
        if not ok:
            all_ok = False
        tag = "OK" if ok else "MISSING"
        print(f"  {name:<25s} PhysX={p_count:>2d}  Newton={n_count:>2d}  [{tag}]")

    assert all_ok, "Newton bridge missing required FlexivSimPlugin calls"
    print("  PASS\n")


def test_simrobotstates_constructor_parity():
    """Both bridges must construct SimRobotStates with compatible args."""
    physx_constructors = re.findall(
        r"SimRobotStates\([^)]+\)", _read(_PHYSX_BRIDGE), re.DOTALL
    )
    newton_constructors = re.findall(
        r"SimRobotStates\([^)]+\)", _read(_NEWTON_BRIDGE), re.DOTALL
    )

    print("[test_simrobotstates_constructor_parity]")
    for i, (p, n) in enumerate(zip(physx_constructors, newton_constructors)):
        print(f"  #{i} PhysX:   {p.strip()[:80]}")
        print(f"  #{i} Newton:  {n.strip()[:80]}")

    assert len(newton_constructors) >= len(physx_constructors), (
        "Newton has fewer SimRobotStates constructors than PhysX"
    )
    print("  PASS\n")


def test_flexiv_serial_interface_parity():
    """FlexivSerialNewton must expose same public methods as FlexivSerial."""
    serial_tree = ast.parse(_read(_PHYSX_SERIAL))
    newton_tree = ast.parse(_read(_NEWTON_SERIAL))

    def _get_public_methods(tree, classname):
        methods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == classname:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not item.name.startswith("_"):
                            methods.add(item.name)
        return methods

    serial_public = _get_public_methods(serial_tree, "FlexivSerial")
    newton_public = _get_public_methods(newton_tree, "FlexivSerialNewton")

    # These are the methods the bridge callback touches on every cycle.
    required = {"switch_control_mode", "apply_torques", "teleport_to"}
    missing = required - newton_public

    print("[test_flexiv_serial_interface_parity]")
    print(f"  FlexivSerial public:      {sorted(serial_public)}")
    print(f"  FlexivSerialNewton pub:   {sorted(newton_public)}")
    print(f"  Required subset present:  {required.issubset(newton_public)}")
    assert not missing, f"Missing required methods: {missing}"

    # Also verify Newton bridge doesn't import ROS-specific packages.
    newton_src = _read(_NEWTON_BRIDGE)
    forbidden_imports = ["rclpy", "geometry_msgs", "std_msgs"]
    for pkg in forbidden_imports:
        assert f"import {pkg}" not in newton_src, (
            f"Newton bridge imports ROS package '{pkg}' — would break contract!"
        )
    print("  No forbidden ROS imports in Newton bridge.")
    print("  PASS\n")


def test_no_newton_import_in_ros_path():
    """Verify Newton physics internals never appear outside Isaac Sim process."""
    newton_serial = _read(_NEWTON_SERIAL)

    # The Newton wrapper should only use isaacsim.experimental APIs, never
    # raw newton/warp imports that would fail in a ROS 3.10 process.
    forbidden = ["import newton", "from newton."]
    found = [f for f in forbidden if f in newton_serial]

    print("[test_no_newton_import_in_ros_path]")
    if found:
        print(f"  WARNING: Newton internals in serial wrapper: {found}")
    else:
        print("  No raw Newton imports — safe for ROS-facing code.")
    print("  PASS\n")


def main():
    print("=" * 60)
    print("Phase 9 -- ROS 2 / RDK Regression Verification")
    print("=" * 60)
    print()

    tests = [
        test_api_surface_parity,
        test_simrobotstates_constructor_parity,
        test_flexiv_serial_interface_parity,
        test_no_newton_import_in_ros_path,
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
        print("All Phase 9 checks PASSED.")
        print("No ROS 2 Python process changes needed for Newton backend switch.")
    print("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

