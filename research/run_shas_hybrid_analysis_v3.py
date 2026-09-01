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

# The city table already contains 2021 source-pool counts. Later, the source
# attribution pivot uses the same column names for estimated contributions to
# Shas in 2022. Drop the left-side source pools only in that specific merge so
# the contribution columns remain unsuffixed; source capacities are read later
# from the original 2021 city table.
_original_merge = hybrid.pd.DataFrame.merge


def merge_fixed(self, right, *args, **kwargs):
    source_groups = list(hybrid.SOURCE_GROUPS)
    overlap = all(c in self.columns for c in source_groups) and all(c in right.columns for c in source_groups)
    key_match = {"city_code", "city"}.issubset(self.columns) and {"city_code", "city"}.issubset(right.columns)
    if overlap and key_match:
        self = self.drop(columns=source_groups)
    return _original_merge(self, right, *args, **kwargs)


hybrid.pd.DataFrame.merge = merge_fixed
hybrid.main()
