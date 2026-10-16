from __future__ import annotations

import json
from pathlib import Path

from exp06_matched_drift.prep_data import RECIPES

SETUP = Path(__file__).resolve().parents[1] / "exp06_matched_drift" / "logs" / "setup" / "data"


def test_pinned_revisions_are_the_ones_exp06_recorded() -> None:
    for r in RECIPES:
        prov = json.loads((SETUP / f"{r.name}.prov.json").read_text())
        assert (r.dataset, r.config, r.split) == (prov["dataset"], prov["config"], prov["split"])
        assert r.revision == prov["revision"] and len(r.revision) == 40
