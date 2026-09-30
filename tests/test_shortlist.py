"""Regression test: known-MP compounds must never be labelled 'novel' by the shortlist stage."""
import pandas as pd

from pipeline import s22_shortlist


def test_shortlist_roles_follow_status_after_resort(tmp_path, monkeypatch):
    rows = []
    # known compounds get the HIGHEST scores so a re-sort moves them to the top (the bug's trigger)
    for i in range(6):
        rows.append({"formula": f"Cs2AgBi{'Br' if i % 2 else 'Cl'}6".replace("Bi", ["Bi", "Sb", "In"][i % 3]),
                     "status": "known_mp", "mp_ehull_meV": 0.0, "ehull_pred_meV": 0.0, "p_ehull_le_50": 1.0,
                     "p_semi": 0.95, "p_gap_pv": 0.9, "ox_states": "Cs+1;Ag+1;Bi+3;Br-1"})
    for i in range(20):
        rows.append({"formula": "K2GeCuF3Cl3", "status": "stable_likely", "mp_ehull_meV": float("nan"),
                     "ehull_pred_meV": 10.0, "p_ehull_le_50": 0.8, "p_semi": 0.7, "p_gap_pv": 0.5 - i * 0.01,
                     "ox_states": "K+1;Ge+2;Cu+2;F-1;Cl-1"})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)   # unsorted input
    (tmp_path / "stability").mkdir()
    df.to_csv(tmp_path / "stability/v2_stability.csv", index=False)
    s22_shortlist.run(tmp_path, force=True)
    short = pd.read_csv(tmp_path / "stability/shortlist_for_umlip.csv")
    assert (short.loc[short.role == "novel", "status"] != "known_mp").all()
    assert (short.loc[short.role == "control_known_mp", "status"] == "known_mp").all()
