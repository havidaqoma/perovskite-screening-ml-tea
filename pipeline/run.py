"""Single entry point.

Legacy reproduction (April 2026 science, flaws included):
    python -m pipeline.run --run-id legacy_april --only s03 --only s04 --only s05 --model-source april_joblib --gate
    python -m pipeline.run --run-id legacy_repro --gate

v2 (Phases C-D: audit-grade ML + stability), needs MP_API_KEY for s10/s20:
    python -m pipeline.run --v2 --run-id v2 --gate
    (Roost s12 and uMLIP s21 run in the WSL GPU env; see REPRODUCIBILITY.md)
    s50 (external HSE06 check and sensitivity checks) needs data/external/walterbos2026/ (scripts/fetch_walterbos2026.py)

Artifacts go to runs/<run-id>/ with one stage_<name>.json manifest per stage
(input hashes, params, output hashes, metrics, package versions, git commit).
"""
from __future__ import annotations

import argparse
import sys

from . import config, s01_featurize, s02_train, s03_generate, s04_screen, s05_tea_legacy

LEGACY_STAGES = [("s01", s01_featurize), ("s02", s02_train), ("s03", s03_generate),
                 ("s04", s04_screen), ("s05", s05_tea_legacy)]


def v2_stages():
    from . import (s10_mp_dataset, s11_ml_eval, s11b_leakage_audit, s13_final_models, s14_screen_v2,
                   s20_stability, s22_shortlist, s30_tea_v2, s40_pareto, s41_robustness, s50_review_checks)
    return [("s10", s10_mp_dataset), ("s11", s11_ml_eval), ("s11b", s11b_leakage_audit),
            ("s13", s13_final_models), ("s14", s14_screen_v2), ("s20", s20_stability), ("s22", s22_shortlist),
            ("s30", s30_tea_v2), ("s40", s40_pareto), ("s41", s41_robustness),
            ("s50", s50_review_checks)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="legacy_repro")
    ap.add_argument("--v2", action="store_true", help="run the v2 stages (s10-s22) instead of the legacy ones")
    ap.add_argument("--only", action="append", help="stage key, e.g. s03 (repeatable)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--gate", action="store_true", help="run the gate(s) after the stages")
    ap.add_argument("--model-source", default="retrained", choices=["retrained", "april_joblib"],
                    help="legacy only: model used by s04/s05")
    a = ap.parse_args(argv)

    run_dir = config.RUNS_DIR / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stages = v2_stages() if a.v2 else LEGACY_STAGES
    for key, mod in stages:
        if a.only and key not in a.only:
            continue
        if key in ("s04", "s05"):
            mod.run(run_dir, force=a.force, model_source=a.model_source)
        else:
            mod.run(run_dir, force=a.force)

    if not a.gate:
        return 0
    if a.v2:
        from . import gate_ml_audit, gate_stability, gate_tea_v2
        results = [gate_ml_audit.check(run_dir), gate_stability.check(run_dir)]
        if (run_dir / "metrics/tea_v2.json").exists():
            results.append(gate_tea_v2.check(run_dir))
        if (run_dir / "pareto/pareto.csv").exists() and (config.ROOT / "paper/figures").exists():
            from . import gate_pareto
            results.append(gate_pareto.check(run_dir))
    else:
        from . import gate_legacy_repro
        results = [gate_legacy_repro.check(run_dir)]
    for res in results:
        for k, v in res["checks"].items():
            print(f"{'PASS' if v['pass'] else 'FAIL'}  {k}")
    ok = all(r["pass"] for r in results)
    print("GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
