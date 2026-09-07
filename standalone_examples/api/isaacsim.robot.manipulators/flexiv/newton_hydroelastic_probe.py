#!/usr/bin/env python
"""Newton hydroelastic contact probe (Phase 8A-8C of dev plan).

This standalone script investigates whether Isaac Sim 6.0.1 exposes a practical
path to configure Newton hydroelastic contact through its USD-to-Newton
integration.  It is NOT part of the production bridge.

Run it from the Isaac Sim root:

    ./python.sh standalone_examples/api/isaacsim.robot.manipulators/flexiv/newton_hydroelastic_probe.py --probes all

Probes:
    a — Inspect bundled Newton Python package for hydroelastic symbols.
    b — Inspect Isaac Newton extensions / schema for hydroelastic hooks.
    c — Attempt minimal two-body hydroelastic contact scene inside Isaac.

All results are printed to spdlog and exported as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import spdlog

# Global logger for top-level diagnostics
Probe_logger = spdlog.ConsoleLogger("Probe")

# =====================================================================
# 1. Parse args, launch SimulationApp
# =====================================================================

argparser = argparse.ArgumentParser(
    description="Newton hydroelastic contact investigation (Phase 8)",
)
argparser.add_argument(
    "--probes", choices=["a", "b", "c", "all"], default="all",
    help="Which probe(s) to run.",
)
argparser.add_argument(
    "--report-dir", default=None,
    help="Directory for JSON report (default: cwd).",
)
cli_args = argparser.parse_args()

from isaacsim import SimulationApp

_headless = bool(os.environ.get("LIVY_SESSION_ID")) or not bool(
    os.environ.get("DISPLAY") and os.path.exists(os.environ["DISPLAY"])
)
simulation_app = SimulationApp({"headless": _headless})
time.sleep(0.5)

# =====================================================================
# 2. Enable Newton, switch engine
# =====================================================================

from isaacsim.core.utils.extensions import enable_extension

for ext in [
    "isaacsim.physics.newton",
    "isaacsim.physics.newton.tensors",
]:
    enable_extension(ext)

simulation_app.update()

from isaacsim.core.simulation_manager import SimulationManager

if SimulationManager.get_active_physics_engine() != "newton":
    SimulationManager.switch_physics_engine("newton", verbose=True)
    simulation_app.update()

active = SimulationManager.get_active_physics_engine()
Probe_logger.info(f"[Probe] Active engine: {active}")

# =====================================================================
# Probe result collector
# =====================================================================

results: Dict[str, Any] = {}


def _record(probe_id: str, ok: bool, **kwds: Any) -> None:
    entry = {"ok": ok}
    entry.update(kwds)
    results[probe_id] = entry


# =====================================================================
# Probe A — Inspect bundled Newton for hydroelastic symbols
# =====================================================================

def _probe_a() -> None:
    """Phase 8A: Does the bundled Newton Python package expose hydroelastic APIs?"""
    Probe_logger.info("[Probe A] Inspecting Newton Python package for hydroelastic symbols")
    info: Dict[str, Any] = {}

    try:
        import newton as nw
        info["newton_version"] = getattr(nw, "__version__", "unknown")
    except ImportError as e:
        _record("A", ok=False, import_error=str(e))
        return

    # Search for hydroelastic-relevant attributes.
    target_names = [
        "HydroelasticContactMaterial",
        "HydroelasticSDF",
        "SensorContact",
        "get_contact_surface",
        "hydroelastic",
    ]

    found: Dict[str, str] = {}
    for tname in target_names:
        # Top-level newton module.
        if hasattr(nw, tname):
            found[tname] = f"newton.{tname}"
            continue
        # Recurse one level into sub-modules.
        for submod_name in dir(nw):
            if submod_name.startswith("_"):
                continue
            try:
                submod = getattr(nw, submod_name)
                if hasattr(submod, tname):
                    found[tname] = f"newton.{submod_name}.{tname}"
                    break
            except Exception:
                pass

    info["symbols_found"] = list(found.keys())
    _record("A", ok=len(found) > 0, **info)
    Probe_logger.info(f"[Probe A] Hydro symbols found: {len(found)}")


# =====================================================================
# Probe B — Inspect Isaac Newton extensions / schema for hooks
# =====================================================================

def _probe_b() -> None:
    """Phase 8B: Do Isaac Newton extensions expose hydroelastic configuration?"""
    Probe_logger.info("[Probe B] Inspecting Isaac Newton extensions and schema")
    info: Dict[str, Any] = {}

    # Check Newton USD schema for hydro-related attributes.
    try:
        import omni.usd.schema.newton as ns  # noqa: F401
        info["schema_importable"] = True

        # Recurse into schema sub-modules collecting hydro-related names.
        hydro_names: List[str] = []
        for mod_name in dir(ns):
            if mod_name.startswith("_"):
                continue
            try:
                obj = getattr(ns, mod_name)
                for attr_name in dir(obj):
                    if "hydro" in attr_name.lower() or attr_name.upper() == "SDF":
                        hydro_names.append(f"{mod_name}.{attr_name}")
            except Exception:
                pass

        info["hydro_attributes"] = hydro_names
    except ImportError:
        info["schema_importable"] = False

    # Check UsdPhysics.CollisionAPI for Newton-specific attributes.
    try:
        from pxr import UsdPhysics
        cap = UsdPhysics.CollisionAPI
        newton_attrs = [a for a in dir(cap) if "newton" in a.lower() or "hydro" in a.lower()]
        info["collision_api_newton_attrs"] = newton_attrs
    except Exception as e:
        info["collision_api_error"] = str(e)

    # Check isaacsim.physics.newton.tensors internals.
    try:
        import isaacsim.physics.newton.tensors._impl as impl  # noqa: F401
        newton_impl_attrs = [a for a in dir(impl) if "hydro" in a.lower()]
        info["newton_tensor_hydro_attrs"] = newton_impl_attrs
    except (ImportError, AttributeError) as e:
        info["newton_tensor_error"] = str(e)

    _record("B", ok=bool(info.get("hydro_attributes")), **info)
    Probe_logger.info(f"[Probe B] Schema inspection complete")


# =====================================================================
# Probe C — Attempt minimal two-body hydroelastic scene
# =====================================================================

def _probe_c() -> None:
    """Phase 8C: Create a minimal two-body hydroelastic contact scene inside Isaac.

    Creates two spheres, tries every known path to enable hydroelastic on their
    collision shapes, steps the simulation, and reports findings.
    """
    Probe_logger.info("[Probe C] Attempting minimal hydroelastic two-body scene")
    info: Dict[str, Any] = {}

    import isaacsim.core.experimental.utils.stage as stage_utils
    import omni.kit.app
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    # Fresh stage.
    try:
        await stage_utils.create_new_stage_async(template="default stage")  # type: ignore[name-defined]
    except Exception as e:
        info["stage_error"] = str(e)
        _record("C", ok=False, **info)
        return

    simulation_app.update()
    stage = stage_utils.get_current_stage(backend="usd")

    # Create two spheres (rigid bodies with sphere collision).
    for name, z_pos in [("SphereA", 0.1), ("SphereB", 0.24)]:
        xform = UsdGeom.Xform.Define(stage, f"/World/{name}").GetPrim()
        UsdGeom.XformCommonAPI(xform).SetTranslation(Gf.Vec3f(0, 0, z_pos))
        UsdPhysics.RigidBodyAPI.Apply(xform)
        sphere_prim = UsdGeom.Sphere.Define(stage, f"/World/{name}/sphere")
        sphere_prim.CreateRadiusAttr(0.05)
        collision_api = UsdPhysics.CollisionAPI.Apply(sphere_prim.GetPrim())

    simulation_app.update()

    # Try to enable hydroelastic via every known path.
    attempts_made: Dict[str, bool] = {}

    # Attempt 1 — Create newton:isHydroelastic attribute on CollisionAPI.
    try:
        for name in ["SphereA", "SphereB"]:
            sprim = stage.GetPrimAtPath(f"/World/{name}/sphere")
            cap = UsdPhysics.CollisionAPI.Get(sprim)
            if hasattr(cap, "GetAttribute"):
                attr = cap.GetAttribute("newton:isHydroelastic")
                if not attr.IsValid():
                    try:
                        attr = cap.CreateAttribute(
                            "newton:isHydroelastic", Sdf.ValueTypeNames.Bool,
                        )
                    except Exception:
                        pass
                if attr.IsValid():
                    attr.Set(True)
                    attempts_made["collision_attr"] = True
    except Exception as e:
        info["attempt1_error"] = str(e)

    # Attempt 2 — Use omni.usd.schema.newton to set hydroelastic.
    try:
        import omni.usd.schema.newton as ns
        for name in ["SphereA", "SphereB"]:
            sprim = stage.GetPrimAtPath(f"/World/{name}/sphere")
            # Check for NewtonCollisionAPI in the schema.
            for submod_name in dir(ns):
                if submod_name.startswith("_"):
                    continue
                submod = getattr(ns, submod_name)
                for attr_name in dir(submod):
                    if "hydro" in attr_name.lower():
                        attempts_made[f"schema_{submod_name}.{attr_name}"] = True
    except Exception as e:
        info["attempt2_error"] = str(e)

    # Attempt 3 — Access physics context to set hydroelastic.
    try:
        from isaacsim.core.world import World
        world = World(
            physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, set_defaults=False,
        )
        ctx = world.get_physics_context()
        hydro_attrs = [a for a in dir(ctx) if "hydro" in a.lower()]
        attempts_made["ctx_hydro"] = bool(hydro_attrs)
        info["physics_ctx_hydro_attrs"] = hydro_attrs

        world.reset()
        # Step a few times to generate contacts.
        for _ in range(20):
            world.step(render=False)

        info["steps_completed"] = 20
    except Exception as e:
        info["attempt3_error"] = str(e)

    _record("C", ok=len(attempts_made) > 0, attempts=attempts_made, **info)
    Probe_logger.info(f"[Probe C] Attempts made: {len(attempts_made)}")


# =====================================================================
# Summary and export
# =====================================================================

def _print_summary() -> None:
    """Print structured summary and export JSON report."""
    Probe_logger.info("=" * 62)
    Probe_logger.info("Newton Hydroelastic Probe Summary (Phase 8)")
    Probe_logger.info("=" * 62)

    ok_count = sum(1 for v in results.values() if v.get("ok"))
    total = len(results)

    for pid, r in sorted(results.items()):
        tag = "PASS" if r["ok"] else "NEEDS INVESTIGATION"
        Probe_logger.info(f"  Probe {pid}: {tag}")

    Probe_logger.info(f"\nPassed: {ok_count}/{total}")

    if ok_count >= 3:
        Probe_logger.info("Recommendation: hydroelastic support appears available.")
    elif ok_count >= 1:
        Probe_logger.info(
            "Recommendation: partial support detected. "
            "Investigate specific paths further."
        )
    else:
        Probe_logger.info(
            "Recommendation: no practical hydroelastic configuration path "
            "found in Isaac Sim 6.0.1 + Newton 1.2.1. Document limitation; "
            "consider upgrading Isaac/Newton or writing a custom extension."
        )

    Probe_logger.info("=" * 62)

    # Export JSON report.
    report_dir = cli_args.report_dir or os.getcwd()
    report_path = os.path.join(
        report_dir, "newton_hydroelastic_probe_report.json"
    )
    data = {
        "isaac_version": SimulationApp.get_current().get_version_info(),
        "active_engine": SimulationManager.get_active_physics_engine(),
        "probes": results,
    }
    try:
        with open(report_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        Probe_logger.info(f"Report saved to {report_path}")
    except Exception as e:
        Probe_logger.error(f"Failed to save report: {e}")


# =====================================================================
# Main entry point
# =====================================================================

def main() -> None:
    probe_map = {"a": _probe_a, "b": _probe_b}
    probes_to_run = (
        ["a", "b", "c"] if cli_args.probes == "all" else [cli_args.probes]
    )

    for pid in probes_to_run:
        if pid in probe_map:
            probe_map[pid]()
        elif pid == "c":
            _probe_c()
        else:
            Probe_logger.warn(f"[Probe] Unknown probe '{pid}', skipping")

    _print_summary()


if __name__ == "__main__":
    main()

