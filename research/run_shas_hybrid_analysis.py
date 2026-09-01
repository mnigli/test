from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("analyze_shas_hybrid_ballots.py")
spec = importlib.util.spec_from_file_location("hybrid", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
hybrid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hybrid)

_original_prepare = hybrid.prepare_election


def prepare_election_fixed(df, election):
    grouped, external = _original_prepare(df, election)
    year = 2021 if election == 24 else 2022

    # The source script uses Knesset numbers (24/25) in generated column names,
    # while the analysis logic refers to calendar election years (2021/2022).
    # Preserve the already-built 2021 source-group columns and map the remaining
    # election-specific fields to the expected year suffix.
    suffix = f"_{election}"
    target_suffix = f"_{year}"
    for column in list(grouped.columns):
        if not column.endswith(suffix):
            continue
        target = column[: -len(suffix)] + target_suffix
        if target in grouped.columns:
            grouped = grouped.drop(columns=[column])
        else:
            grouped = grouped.rename(columns={column: target})
    return grouped, external


hybrid.prepare_election = prepare_election_fixed
hybrid.main()
