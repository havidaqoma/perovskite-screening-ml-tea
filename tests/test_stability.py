import pandas as pd

from src.check_stability import SAMPLE_EHULL, apply_stability_gate


def test_stability_gate_excludes_unknown_at_all_standard_thresholds():
    frame = pd.DataFrame(
        {
            "Formula": list(SAMPLE_EHULL),
            "ehull_meV": list(SAMPLE_EHULL.values()),
        }
    )

    kept, summary = apply_stability_gate(frame, threshold_meV=35.0)

    assert set(kept["Formula"]) == {"Cs2AgBiBr6"}
    assert summary == {
        "n_total": 3,
        "n_kept_20": 1,
        "n_kept_35": 1,
        "n_kept_50": 2,
        "n_unknown": 1,
    }
