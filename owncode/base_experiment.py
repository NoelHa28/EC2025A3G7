from pathlib import Path
import csv
import inspect
import numpy as np
import mujoco as mj
import matplotlib.pyplot as plt
import pandas as pd

from robot import Robot
from ariel.utils.tracker import Tracker
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.controllers.controller import Controller
import controller as ctrl_module
from simulate import experiment

# ======= HARD-CODED SETTINGS =======
TRIALS         = 50
DURATION_SEC   = 15.0
GENOTYPE_SIZE  = 64
SEED           = 42
OUT_CSV        = "_data_/baseline_hardcoded.csv"
VIEW_MODE      = "simple"   # "frame" can yield empty tracker history in this project
# ===================================

rng = np.random.default_rng(SEED)

def _random_body_genotype(size: int) -> list[np.ndarray]:
    return [
        rng.random(size).astype(np.float32),  # type probs
        rng.random(size).astype(np.float32),  # connection probs
        rng.random(size).astype(np.float32),  # rotation probs
    ]

def _simple_forward_distance(xpos_hist: np.ndarray) -> float:
    """Cheap baseline metric: final x - initial x."""
    try:
        x0 = float(xpos_hist[0][0])
        xN = float(xpos_hist[-1][0])
        return xN - x0
    except Exception:
        return float("-inf")

def _try_repo_fitness(history) -> tuple[float | None, str]:
    """
    Try to import and call repo fitness with best-guess signature.
    Returns (value_or_None, error_string_or_empty).
    """
    try:
        from fitness_function import fitness as repo_fitness
    except Exception as e:
        return None, f"import:{e}"

    try:
        sig = inspect.signature(repo_fitness)
        req = [
            p for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and p.default is inspect._empty
        ]
        if len(req) == 1:
            # def fitness(history)
            return float(repo_fitness(history)), ""
        elif len(req) == 2:
            # Try (spawn_pos, history) then (history, spawn_pos)
            try:
                try:
                    from evaluate import SPAWN_POS  # if your code exports it
                except Exception:
                    SPAWN_POS = [-0.8, 0, 0.1]
                try:
                    return float(repo_fitness(SPAWN_POS, history)), ""
                except Exception:
                    return float(repo_fitness(history, SPAWN_POS)), ""
            except Exception as e2:
                return None, f"call2:{e2}"
        else:
            return None, f"unsupported_signature:{sig}"
    except Exception as e:
        return None, f"call:{e}"

def run_trial() -> dict:
    # 1) random body
    try:
        body = _random_body_genotype(GENOTYPE_SIZE)
        robot = Robot(body_genotype=body)  # builds graph, counts hinges, sets EvolvableCPG
    except Exception as e:
        return {"ok": False, "reason": f"robot_init:{e}", "fitness": -100.0, "hinges": 0,
                "xlen": 0, "dx": float("-inf"), "fallback": True}

    # 2) random controller for this morphology
    try:
        mind = robot.brain.generate_random_cpg_genotype(robot._number_of_hinges)
        robot.brain.set_genotype(mind)
    except Exception as e:
        return {"ok": False, "reason": f"mind_init:{e}", "fitness": -100.0, "hinges": robot._number_of_hinges,
                "xlen": 0, "dx": float("-inf"), "fallback": True}

    # 3) run MuJoCo once
    try:
        mj.set_mjcb_control(None)
        core = construct_mjspec_from_graph(robot.graph)
        tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
        ctrl = Controller(controller_callback_function=ctrl_module.cpg, tracker=tracker)
        experiment(robot=robot, core=core, controller=ctrl, mode=VIEW_MODE, duration=DURATION_SEC)
    except Exception as e:
        return {"ok": False, "reason": f"sim:{e}", "fitness": -100.0, "hinges": robot._number_of_hinges,
                "xlen": 0, "dx": float("-inf"), "fallback": True}

    # 4) compute fitness (repo → fallback dx)
    xlen = 0
    dx = float("-inf")
    fallback = True
    f = -100.0
    reason = ""

    try:
        if tracker.history["xpos"] and len(tracker.history["xpos"][0]) > 0:
            xpos = np.array(tracker.history["xpos"][0], dtype=float)
            xlen = len(xpos)
            dx = _simple_forward_distance(xpos)
            # Try repo fitness first
            val, err = _try_repo_fitness(tracker.history["xpos"][0])
            if err == "" and val is not None and np.isfinite(val):
                f = float(val)
                fallback = False
            else:
                reason = f"fitness_fallback:{err}" if err else "fitness_fallback"
                f = float(dx) if np.isfinite(dx) else -100.0
        else:
            reason = "no_history"
            f = -100.0
    except Exception as e:
        reason = f"fitness:{e}"
        f = -100.0

    return {
        "ok": True,
        "reason": reason,
        "fitness": f,
        "hinges": robot._number_of_hinges,
        "xlen": xlen,
        "dx": dx,
        "fallback": fallback,
    }

def main():
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"[BASELINE] trials={TRIALS} duration={DURATION_SEC}s seed={SEED} -> {OUT_CSV}")
    for i in range(TRIALS):
        r = run_trial()
        r["trial"] = i
        rows.append(r)
        status = "OK" if r["ok"] else f"FAIL({r['reason']})"
        fb = " (fallback)" if r.get("fallback") else ""
        print(f"[{i:03d}] {status}{fb}  fitness={r['fitness']:.4f}  hinges={r['hinges']}  xlen={r.get('xlen',0)}  dx={r.get('dx',float('-inf')):.3f}")

    # save csv
    fieldnames = ["trial", "ok", "reason", "fitness", "hinges", "xlen", "dx", "fallback"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[BASELINE] saved {len(rows)} rows -> {OUT_CSV}")

    # ====== PLOTS ======
    try:
        df = pd.DataFrame(rows)

        # 1) Fitness by trial
        plt.figure()
        plt.plot(df["trial"], df["fitness"], marker="o", linestyle="-")
        plt.xlabel("Trial")
        plt.ylabel("Fitness")
        plt.title("Baseline: Fitness by Trial")
        plt.tight_layout()
        plt.show()

        # 2) Fitness distribution
        plt.figure()
        df["fitness"].plot(kind="hist", bins=20)
        plt.xlabel("Fitness")
        plt.ylabel("Count")
        plt.title("Baseline: Fitness Distribution")
        plt.tight_layout()
        plt.show()

        # 3) Fitness vs Hinges
        if "hinges" in df.columns:
            plt.figure()
            plt.scatter(df["hinges"], df["fitness"])
            plt.xlabel("Hinges")
            plt.ylabel("Fitness")
            plt.title("Baseline: Fitness vs Hinges")
            plt.tight_layout()
            plt.show()

        # 4) Cumulative best fitness
        plt.figure()
        cum_best = df["fitness"].cummax()
        plt.plot(df["trial"], cum_best, linestyle="-")
        plt.xlabel("Trial")
        plt.ylabel("Cumulative Best Fitness")
        plt.title("Baseline: Cumulative Best Fitness")
        plt.tight_layout()
        plt.show()

        # Optional: displacement sanity-check
        if "dx" in df.columns:
            plt.figure()
            plt.scatter(df["dx"], df["fitness"])
            plt.xlabel("Forward displacement (dx)")
            plt.ylabel("Fitness")
            plt.title("Baseline: Fitness vs Forward Displacement")
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"[BASELINE] Plotting skipped: {e}")

if __name__ == "__main__":
    main()