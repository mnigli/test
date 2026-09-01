from __future__ import annotations

import hashlib
import io
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from scipy.optimize import lsq_linear

OUT = Path("research/hybrid_output")
OUT.mkdir(parents=True, exist_ok=True)

BALLOT24_URL = "https://raw.githubusercontent.com/harelc/elections-vote-transfer/refs/heads/master/ballot24.csv"
BALLOT25_URL = "https://raw.githubusercontent.com/harelc/elections-vote-transfer/refs/heads/master/ballot25.csv"
EDU_URLS = [
    "https://mos.eddata.live/data/institutions.json",
    "https://mos.eddata.live/institutions.json",
]

CORE4 = ["אלעד", "ביתר עילית", "בני ברק", "מודיעין עילית"]
YOUNG3 = ["אלעד", "ביתר עילית", "מודיעין עילית"]
SOURCE_GROUPS = [
    "shas_2021",
    "utj_2021",
    "likud_2021",
    "religious_right_2021",
    "new_hope_2021",
    "yisrael_beiteinu_2021",
    "center_2021",
    "left_2021",
    "arab_2021",
    "other_invalid_2021",
    "abstain_2021",
]
SOURCE_LABELS = {
    "shas_2021": "ש״ס 2021",
    "utj_2021": "יהדות התורה 2021",
    "likud_2021": "הליכוד 2021",
    "religious_right_2021": "ימינה והציונות הדתית 2021",
    "new_hope_2021": "תקווה חדשה 2021",
    "yisrael_beiteinu_2021": "ישראל ביתנו 2021",
    "center_2021": "יש עתיד וכחול לבן 2021",
    "left_2021": "העבודה ומרצ 2021",
    "arab_2021": "מפלגות ערביות 2021",
    "other_invalid_2021": "מפלגות אחרות ופסולים 2021",
    "abstain_2021": "לא הצביעו ב־2021",
}

HEBREW_MARKS = re.compile(r"[\u0591-\u05C7]")
NON_ALNUM = re.compile(r"[^0-9A-Za-z\u0590-\u05FF]+")
SPACES = re.compile(r"\s+")


def clean_text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    s = unicodedata.normalize("NFKC", str(value))
    s = HEBREW_MARKS.sub("", s)
    s = s.replace("–", "-").replace("—", "-").replace("־", "-")
    s = NON_ALNUM.sub(" ", s)
    return SPACES.sub(" ", s).strip()


def norm_city(value) -> str:
    s = clean_text(value).replace("קריית", "קרית")
    aliases = {
        "מודיעין מכבים רעות": "מודיעין מכבים רעות",
        "מודיעין-מכבים-רעות": "מודיעין מכבים רעות",
        "תל אביב יפו": "תל אביב יפו",
        "קרית יערים טלז סטון": "קרית יערים",
        "טלז סטון": "קרית יערים",
        "נצרת עילית": "נוף הגליל",
        "באקה אל גרביה": "באקה אל גרביה",
        "מעלות תרשיחא": "מעלות תרשיחא",
        "יהוד מונוסון": "יהוד מונוסון",
    }
    return aliases.get(s, s)


def ballot_base(value) -> str:
    s = clean_text(value)
    if not s:
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return str(int(math.floor(f)))
    except Exception:
        return s.split(".", 1)[0]


def get_bytes(url: str, timeout: int = 240) -> bytes:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "shas-hybrid-research/1.0"})
    r.raise_for_status()
    return r.content


def read_csv_url(url: str, out_name: str) -> pd.DataFrame:
    raw = get_bytes(url)
    (OUT / out_name).write_bytes(raw)
    last = None
    for enc in ["utf-8-sig", "utf-8", "cp1255", "iso-8859-8"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, dtype={"קלפי": str})
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Unable to decode {url}: {last}")


def read_education() -> tuple[pd.DataFrame, str]:
    last = None
    for url in EDU_URLS:
        try:
            raw = get_bytes(url)
            payload = json.loads(raw)
            rows = payload.get("institutions", payload)
            if not isinstance(rows, list):
                raise TypeError(type(rows))
            (OUT / "institutions.json").write_bytes(raw)
            return pd.DataFrame(rows), url
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Unable to fetch education data: {last}")


def numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for c in columns:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)


def prepare_election(df: pd.DataFrame, election: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    numeric(df, ["בזב", "מצביעים", "פסולים", "כשרים", "שס", "ג", "מחל", "ט", "ב", "ת", "ל", "פה", "כן", "אמת", "מרצ", "ודעם", "עם"])
    df["city"] = df["שם ישוב"].map(norm_city)
    df["city_code"] = pd.to_numeric(df["סמל ישוב"], errors="coerce").fillna(-1).astype(int)
    df["ballot_base"] = df["קלפי"].map(ballot_base)
    ordinary = ~df["city"].str.contains("מעטפות", na=False)
    ordinary &= df["city_code"] > 0
    external = df.loc[~ordinary].copy()
    df = df.loc[ordinary].copy()

    if election == 24:
        known = {
            "shas_2021": df["שס"],
            "utj_2021": df["ג"],
            "likud_2021": df["מחל"],
            "religious_right_2021": df["ב"] + df["ט"],
            "new_hope_2021": df["ת"],
            "yisrael_beiteinu_2021": df["ל"],
            "center_2021": df["פה"] + df["כן"],
            "left_2021": df["אמת"] + df["מרצ"],
            "arab_2021": df["ודעם"] + df["עם"],
        }
        for name, values in known.items():
            df[name] = values
        known_sum = sum(known.values())
        df["other_invalid_2021"] = (df["כשרים"] - known_sum).clip(lower=0) + df["פסולים"]
        df["abstain_2021"] = (df["בזב"] - df["מצביעים"]).clip(lower=0)
        agg_map = {
            "city": "first",
            "בזב": "sum",
            "מצביעים": "sum",
            "פסולים": "sum",
            "כשרים": "sum",
            "שס": "sum",
            "ג": "sum",
            "מחל": "sum",
            "ט": "sum",
            "ב": "sum",
            "ת": "sum",
            "ל": "sum",
            "פה": "sum",
            "כן": "sum",
            "אמת": "sum",
            "מרצ": "sum",
            "ודעם": "sum",
            "עם": "sum",
            **{c: "sum" for c in SOURCE_GROUPS},
        }
    else:
        agg_map = {
            "city": "first",
            "בזב": "sum",
            "מצביעים": "sum",
            "פסולים": "sum",
            "כשרים": "sum",
            "שס": "sum",
            "ג": "sum",
            "מחל": "sum",
            "ט": "sum",
            "ל": "sum",
            "פה": "sum",
            "כן": "sum",
            "אמת": "sum",
            "מרצ": "sum",
            "ודעם": "sum",
            "עם": "sum",
        }

    grouped = df.groupby(["city_code", "ballot_base"], as_index=False).agg(agg_map)
    grouped = grouped.rename(columns={
        "בזב": f"bav_{election}",
        "מצביעים": f"voters_{election}",
        "פסולים": f"invalid_{election}",
        "כשרים": f"valid_{election}",
        "שס": f"shas_{election}",
        "ג": f"utj_{election}",
        "מחל": f"likud_{election}",
        "ט": f"rz_{election}",
        "ב": f"yamina_{election}",
        "ת": f"new_hope_{election}",
        "ל": f"yb_{election}",
        "פה": f"yesh_atid_{election}",
        "כן": f"blue_white_{election}",
        "אמת": f"labor_{election}",
        "מרצ": f"meretz_{election}",
        "ודעם": f"joint_{election}",
        "עם": f"raam_{election}",
    })
    return grouped, external


def city_totals(ballots: pd.DataFrame, election: int) -> pd.DataFrame:
    numeric_cols = [c for c in ballots.columns if c not in {"city", "ballot_base", "city_code"}]
    agg = {c: "sum" for c in numeric_cols}
    agg["city"] = "first"
    return ballots.groupby("city_code", as_index=False).agg(agg)


def education_by_city(edu: pd.DataFrame) -> pd.DataFrame:
    e = edu.copy()
    e["city"] = e["city"].map(norm_city)
    e["students"] = pd.to_numeric(e["students"], errors="coerce")
    e = e[e["students"].notna() & (e["students"] >= 0)].copy()
    e["is_haredi"] = e["supervision"].map(clean_text).str.contains("חרדי", na=False)
    rows = []
    for city, g in e.groupby("city"):
        h = g[g["is_haredi"]]
        rows.append({
            "city": city,
            "students_all": float(g["students"].sum()),
            "students_haredi": float(h["students"].sum()),
            "haredi_institutions": int(len(h)),
        })
    out = pd.DataFrame(rows)
    out["student_haredi_share"] = np.where(out["students_all"] > 0, out["students_haredi"] / out["students_all"], np.nan)
    return out


def weighted_metrics(actual: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    weights = np.asarray(weights, float)
    actual = np.asarray(actual, float)
    predicted = np.asarray(predicted, float)
    wsum = weights.sum()
    if wsum <= 0:
        return {"mae_pp": np.nan, "rmse_pp": np.nan, "r2": np.nan}
    err = predicted - actual
    mae = np.sum(weights * np.abs(err)) / wsum
    rmse = math.sqrt(np.sum(weights * err * err) / wsum)
    mean = np.sum(weights * actual) / wsum
    denom = np.sum(weights * (actual - mean) ** 2)
    r2 = 1 - np.sum(weights * err * err) / denom if denom > 0 else np.nan
    return {"mae_pp": mae * 100, "rmse_pp": rmse * 100, "r2": r2}


def fit_beta(frame: pd.DataFrame, prior: np.ndarray | None = None, ridge: float = 0.0) -> np.ndarray:
    x = frame[[f"share_{c}" for c in SOURCE_GROUPS]].to_numpy(float)
    y = frame["target_shas_share_2022"].to_numpy(float)
    weights = np.sqrt(frame["bav_2022"].clip(lower=1).to_numpy(float))
    a = x * weights[:, None]
    b = y * weights
    if prior is not None and ridge > 0:
        a = np.vstack([a, math.sqrt(ridge) * np.eye(len(SOURCE_GROUPS))])
        b = np.concatenate([b, math.sqrt(ridge) * prior])
    result = lsq_linear(a, b, bounds=(0.0, 1.0), method="trf", lsmr_tol="auto", max_iter=5000)
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def predict(frame: pd.DataFrame, beta: np.ndarray) -> np.ndarray:
    x = frame[[f"share_{c}" for c in SOURCE_GROUPS]].to_numpy(float)
    return x @ beta


def capped_proportional_allocation(total: float, probabilities: np.ndarray, capacities: np.ndarray) -> np.ndarray:
    total = max(0.0, min(float(total), float(capacities.sum())))
    p = np.asarray(probabilities, float).copy()
    cap = np.asarray(capacities, float).copy()
    p = np.maximum(p, 0)
    if p.sum() <= 0 or total <= 0:
        return np.zeros_like(cap)
    p /= p.sum()
    alloc = np.zeros_like(cap)
    active = np.ones(len(cap), dtype=bool)
    remaining = total
    for _ in range(len(cap) + 2):
        if remaining <= 1e-9 or not active.any():
            break
        pa = p * active
        if pa.sum() <= 0:
            pa = active.astype(float)
        proposed = remaining * pa / pa.sum()
        room = cap - alloc
        hit = active & (proposed >= room - 1e-9)
        if not hit.any():
            alloc[active] += proposed[active]
            remaining = 0
            break
        alloc[hit] += np.maximum(room[hit], 0)
        remaining -= np.maximum(room[hit], 0).sum()
        active[hit] = False
    if remaining > 1e-6 and active.any():
        room = np.maximum(cap - alloc, 0)
        alloc += remaining * room / room.sum()
    return np.minimum(alloc, cap)


def main() -> None:
    raw24 = read_csv_url(BALLOT24_URL, "ballot24.csv")
    raw25 = read_csv_url(BALLOT25_URL, "ballot25.csv")
    edu_raw, edu_url = read_education()

    b24, ext24 = prepare_election(raw24, 24)
    b25, ext25 = prepare_election(raw25, 25)
    c24 = city_totals(b24, 24)
    c25 = city_totals(b25, 25)
    edu_city = education_by_city(edu_raw)

    matched = b24.merge(b25, on=["city_code", "ballot_base"], how="inner", suffixes=("_24name", "_25name"))
    matched["city"] = matched["city_25name"].fillna(matched["city_24name"])
    matched = matched[(matched["bav_2021"] >= 80) & (matched["bav_2022"] >= 80)].copy()
    for c in SOURCE_GROUPS:
        matched[f"share_{c}"] = matched[c] / matched["bav_2021"].replace(0, np.nan)
    matched["target_shas_share_2022"] = matched["shas_2022"] / matched["bav_2022"].replace(0, np.nan)
    matched["turnout_2021"] = matched["voters_2021"] / matched["bav_2021"].replace(0, np.nan)
    matched["turnout_2022"] = matched["voters_2022"] / matched["bav_2022"].replace(0, np.nan)
    matched["shas_share_2021"] = matched["shas_2021"] / matched["bav_2021"].replace(0, np.nan)
    matched["shas_share_2022"] = matched["target_shas_share_2022"]
    matched["shas_share_change_pp"] = (matched["shas_share_2022"] - matched["shas_share_2021"]) * 100
    matched["turnout_change_pp"] = (matched["turnout_2022"] - matched["turnout_2021"]) * 100
    matched = matched.replace([np.inf, -np.inf], np.nan).dropna(subset=["target_shas_share_2022"] + [f"share_{c}" for c in SOURCE_GROUPS])

    city = c24.merge(c25, on="city_code", how="outer", suffixes=("_24name", "_25name"))
    city["city"] = city["city_25name"].fillna(city["city_24name"])
    city = city.merge(edu_city, on="city", how="left")
    city["student_haredi_share"] = city["student_haredi_share"].fillna(0)
    city["students_haredi"] = city["students_haredi"].fillna(0)
    city["students_all"] = city["students_all"].fillna(0)
    city["arab_share_2022"] = (city.get("joint_2022", 0) + city.get("raam_2022", 0)) / city["valid_2022"].replace(0, np.nan)
    city["is_core4"] = city["city"].isin(CORE4)
    city["is_haredi_dominant"] = city["is_core4"] | ((city["student_haredi_share"] >= 0.80) & ((city["shas_2022"] + city["utj_2022"]) / city["valid_2022"].replace(0, np.nan) >= 0.65))
    city["is_mixed_model"] = (
        (~city["is_haredi_dominant"])
        & (city["students_haredi"] >= 30)
        & (city["shas_2022"] >= 30)
        & (city["arab_share_2022"].fillna(0) < 0.50)
    )

    # Student anchor calibration by election year.
    calibration_rows = []
    student_coefficients = {}
    for year in [2021, 2022]:
        s_col = f"shas_{year}"
        g_col = f"utj_{year}"
        bav_col = f"bav_{year}"
        core = city[city["city"].isin(CORE4) & (city["students_haredi"] > 0)].copy()
        core["bloc"] = core[s_col] + core[g_col]
        core["bloc_per_student"] = core["bloc"] / core["students_haredi"]
        core["bav_per_student"] = core[bav_col] / core["students_haredi"]
        for _, r in core.iterrows():
            calibration_rows.append({
                "year": year,
                "city": r["city"],
                "students_haredi": r["students_haredi"],
                "shas": r[s_col],
                "utj": r[g_col],
                "bloc": r["bloc"],
                "bav": r[bav_col],
                "bloc_per_student": r["bloc_per_student"],
                "bav_per_student": r["bav_per_student"],
            })
        young = core[core["city"].isin(YOUNG3)]
        student_coefficients[year] = {
            "bloc_low": float(young["bloc_per_student"].median()),
            "bloc_central": float(core["bloc_per_student"].median()),
            "bloc_high": float(core["bloc"].sum() / core["students_haredi"].sum()),
            "bav_central": float(core["bav_per_student"].median()),
            "bav_weighted": float(core[bav_col].sum() / core["students_haredi"].sum()),
        }
        for scenario in ["low", "central", "high"]:
            coef = student_coefficients[year][f"bloc_{scenario}"]
            pred_bloc = np.minimum(city[s_col] + city[g_col], city["students_haredi"] * coef)
            h_shas = np.maximum(0, np.minimum(city[s_col], pred_bloc - city[g_col]))
            h_shas = np.where(city["is_haredi_dominant"], city[s_col], h_shas)
            city[f"anchor_haredi_shas_{year}_{scenario}"] = h_shas

    # Select matched mixed-city ballots for source model.
    model_cities = set(city.loc[city["is_mixed_model"], "city_code"].astype(int))
    train = matched[matched["city_code"].isin(model_cities)].copy()
    # Remove extreme data errors but retain genuine high-turnout ballots.
    train = train[(train["turnout_2021"].between(0.20, 1.02)) & (train["turnout_2022"].between(0.20, 1.02))]

    beta_global = fit_beta(train)
    train["pred_global"] = predict(train, beta_global)
    global_metrics = weighted_metrics(train["target_shas_share_2022"].to_numpy(), train["pred_global"].to_numpy(), train["bav_2022"].to_numpy())

    # Five-fold cross-validation grouped by city.
    train["fold"] = train["city_code"].map(lambda x: int(hashlib.md5(str(int(x)).encode()).hexdigest(), 16) % 5)
    cv_parts = []
    cv_coefficients = []
    for fold in range(5):
        tr = train[train["fold"] != fold]
        te = train[train["fold"] == fold].copy()
        beta = fit_beta(tr)
        te["prediction"] = predict(te, beta)
        te["cv_fold"] = fold
        cv_parts.append(te[["city_code", "city", "ballot_base", "bav_2022", "target_shas_share_2022", "prediction", "cv_fold"]])
        for source, value in zip(SOURCE_GROUPS, beta):
            cv_coefficients.append({"fold": fold, "source": source, "source_label": SOURCE_LABELS[source], "coefficient": value})
    cv = pd.concat(cv_parts, ignore_index=True)
    cv_metrics = weighted_metrics(cv["target_shas_share_2022"].to_numpy(), cv["prediction"].to_numpy(), cv["bav_2022"].to_numpy())

    # City-specific ridge estimates centered on global coefficients.
    city_beta_rows = []
    contribution_rows = []
    ballot_rows = []
    for city_code, g in train.groupby("city_code"):
        g = g.copy()
        if len(g) >= 8:
            ridge = max(20.0, len(g) * 0.75)
            beta = fit_beta(g, prior=beta_global, ridge=ridge)
            model_type = "city_ridge"
        else:
            beta = beta_global.copy()
            model_type = "global"
        predicted = predict(g, beta)
        metrics = weighted_metrics(g["target_shas_share_2022"].to_numpy(), predicted, g["bav_2022"].to_numpy())
        city_name = g["city"].iloc[0]
        for source, value in zip(SOURCE_GROUPS, beta):
            city_beta_rows.append({
                "city_code": int(city_code),
                "city": city_name,
                "matched_ballots": int(len(g)),
                "model_type": model_type,
                "source": source,
                "source_label": SOURCE_LABELS[source],
                "coefficient": float(value),
                **metrics,
            })
        x = g[[f"share_{c}" for c in SOURCE_GROUPS]].to_numpy(float)
        raw = x * beta[None, :]
        raw_sum = raw.sum(axis=1)
        shares = np.divide(raw, raw_sum[:, None], out=np.zeros_like(raw), where=raw_sum[:, None] > 0)
        actual_counts = g["shas_2022"].to_numpy(float)
        contributions = shares * actual_counts[:, None]
        for row_idx, (_, r) in enumerate(g.iterrows()):
            base = {
                "city_code": int(city_code),
                "city": city_name,
                "ballot_base": r["ballot_base"],
                "bav_2021": r["bav_2021"],
                "bav_2022": r["bav_2022"],
                "shas_2021": r["shas_2021"],
                "shas_2022": r["shas_2022"],
                "utj_2021": r["utj_2021"],
                "utj_2022": r["utj_2022"],
                "likud_2021": r["likud_2021"],
                "likud_2022": r["likud_2022"],
                "shas_share_change_pp": r["shas_share_change_pp"],
                "turnout_change_pp": r["turnout_change_pp"],
                "model_predicted_shas": predicted[row_idx] * r["bav_2022"],
                "model_residual_votes": r["shas_2022"] - predicted[row_idx] * r["bav_2022"],
            }
            for j, source in enumerate(SOURCE_GROUPS):
                base[f"from_{source}"] = contributions[row_idx, j]
                contribution_rows.append({
                    "city_code": int(city_code),
                    "city": city_name,
                    "ballot_base": r["ballot_base"],
                    "source": source,
                    "source_label": SOURCE_LABELS[source],
                    "source_votes_2021": r[source],
                    "estimated_shas_2022_from_source": contributions[row_idx, j],
                })
            ballot_rows.append(base)

    city_betas = pd.DataFrame(city_beta_rows)
    contributions_long = pd.DataFrame(contribution_rows)
    ballot_sources = pd.DataFrame(ballot_rows)
    city_source = contributions_long.groupby(["city_code", "city", "source", "source_label"], as_index=False).agg(
        source_votes_2021=("source_votes_2021", "sum"),
        shas_2022_from_source=("estimated_shas_2022_from_source", "sum"),
    )

    # Derive Haredi fractions within each 2021 source pool using core-city electorate composition.
    core24 = b24.merge(edu_city[["city", "students_haredi"]], on="city", how="left")
    core24 = core24[core24["city"].isin(CORE4)]
    core_source = core24.groupby("city", as_index=False).agg({**{c: "sum" for c in SOURCE_GROUPS}, "bav_2021": "sum", "students_haredi": "first"})
    core_source_totals = core_source[SOURCE_GROUPS].sum().to_numpy(float)
    core_probabilities = core_source_totals / core_source_totals.sum()
    bav_per_student = float((core_source["bav_2021"] / core_source["students_haredi"]).median())

    city_source_wide = city_source.pivot(index=["city_code", "city"], columns="source", values="shas_2022_from_source").fillna(0).reset_index()
    for source in SOURCE_GROUPS:
        if source not in city_source_wide.columns:
            city_source_wide[source] = 0.0

    city_summary = city.merge(city_source_wide, on=["city_code", "city"], how="left")
    for source in SOURCE_GROUPS:
        city_summary[source] = city_summary[source].fillna(0)

    haredi_fraction_rows = []
    hybrid_source_rows = []
    hybrid_counts = []
    for _, r in city_summary.iterrows():
        actual_source = np.array([float(r.get(c, 0)) for c in SOURCE_GROUPS])
        # For city rows not modeled, contributions are zero; Haredi-dominant cities are handled separately.
        estimated_haredi_bav_2021 = min(float(r.get("bav_2021", 0)), float(r.get("students_haredi", 0)) * bav_per_student)
        source_capacity = np.array([float(r.get(c + "_actual", 0)) for c in SOURCE_GROUPS])
        # Source counts reside in c24, not contribution pivot.
        c24row = c24[c24["city_code"] == r["city_code"]]
        if not c24row.empty:
            source_capacity = c24row.iloc[0][SOURCE_GROUPS].to_numpy(float)
        else:
            source_capacity = np.zeros(len(SOURCE_GROUPS))
        h_alloc = capped_proportional_allocation(estimated_haredi_bav_2021, core_probabilities, source_capacity)
        h_frac = np.divide(h_alloc, source_capacity, out=np.zeros_like(h_alloc), where=source_capacity > 0)
        transfer_contrib = actual_source
        h_contrib = transfer_contrib * h_frac
        transfer_haredi = float(h_contrib.sum())
        transfer_nonharedi = max(0.0, float(r.get("shas_2022", 0)) - transfer_haredi)
        anchor = float(r.get("anchor_haredi_shas_2022_central", 0))
        if bool(r.get("is_haredi_dominant", False)):
            transfer_haredi = float(r.get("shas_2022", 0))
            transfer_nonharedi = 0.0
            h_contrib = transfer_contrib.copy()
        hybrid_counts.append({
            "city_code": int(r["city_code"]),
            "city": r["city"],
            "shas_2022": float(r.get("shas_2022", 0)),
            "utj_2022": float(r.get("utj_2022", 0)),
            "students_haredi": float(r.get("students_haredi", 0)),
            "student_haredi_share": float(r.get("student_haredi_share", 0)),
            "is_haredi_dominant": bool(r.get("is_haredi_dominant", False)),
            "is_mixed_model": bool(r.get("is_mixed_model", False)),
            "anchor_haredi_shas_2022": anchor,
            "hybrid_haredi_shas_2022": transfer_haredi,
            "hybrid_nonharedi_shas_2022": transfer_nonharedi,
            "anchor_minus_hybrid": anchor - transfer_haredi,
            "hybrid_haredi_share": transfer_haredi / r["shas_2022"] if r.get("shas_2022", 0) else np.nan,
            "estimated_haredi_bav_2021": estimated_haredi_bav_2021,
        })
        for j, source in enumerate(SOURCE_GROUPS):
            haredi_fraction_rows.append({
                "city_code": int(r["city_code"]),
                "city": r["city"],
                "source": source,
                "source_label": SOURCE_LABELS[source],
                "actual_source_voters_2021": source_capacity[j],
                "estimated_haredi_source_voters_2021": h_alloc[j],
                "estimated_haredi_fraction": h_frac[j],
            })
            hybrid_source_rows.append({
                "city_code": int(r["city_code"]),
                "city": r["city"],
                "source": source,
                "source_label": SOURCE_LABELS[source],
                "shas_2022_from_source": transfer_contrib[j],
                "haredi_shas_2022_from_source": h_contrib[j],
                "nonharedi_shas_2022_from_source": max(0.0, transfer_contrib[j] - h_contrib[j]),
            })

    hybrid_city = pd.DataFrame(hybrid_counts)
    haredi_fractions = pd.DataFrame(haredi_fraction_rows)
    hybrid_sources = pd.DataFrame(hybrid_source_rows)

    # Non-modeled education-matched cities: fall back to city anchor; unmatched are reported separately.
    modeled_codes = set(hybrid_city.loc[hybrid_city["is_mixed_model"] | hybrid_city["is_haredi_dominant"], "city_code"])
    fallback_mask = ~hybrid_city["city_code"].isin(modeled_codes)
    has_students = hybrid_city["students_haredi"] > 0
    hybrid_city.loc[fallback_mask & has_students, "hybrid_haredi_shas_2022"] = hybrid_city.loc[fallback_mask & has_students, "anchor_haredi_shas_2022"]
    hybrid_city.loc[fallback_mask & has_students, "hybrid_nonharedi_shas_2022"] = hybrid_city.loc[fallback_mask & has_students, "shas_2022"] - hybrid_city.loc[fallback_mask & has_students, "hybrid_haredi_shas_2022"]
    hybrid_city.loc[fallback_mask & has_students, "hybrid_haredi_share"] = hybrid_city.loc[fallback_mask & has_students, "hybrid_haredi_shas_2022"] / hybrid_city.loc[fallback_mask & has_students, "shas_2022"].replace(0, np.nan)

    ordinary_haredi = float(hybrid_city["hybrid_haredi_shas_2022"].sum())
    ordinary_shas = float(c25["shas_2022"].sum())
    external_shas = float(pd.to_numeric(ext25.get("שס", 0), errors="coerce").fillna(0).sum())
    external_utj = float(pd.to_numeric(ext25.get("ג", 0), errors="coerce").fillna(0).sum())
    external_haredi = min(external_shas, external_utj * (28 / 58))
    total_haredi = ordinary_haredi + external_haredi
    total_shas = ordinary_shas + external_shas

    national_sources = hybrid_sources.groupby(["source", "source_label"], as_index=False).agg(
        shas_2022_from_source=("shas_2022_from_source", "sum"),
        haredi_shas_2022_from_source=("haredi_shas_2022_from_source", "sum"),
        nonharedi_shas_2022_from_source=("nonharedi_shas_2022_from_source", "sum"),
    )
    national_sources["nonharedi_source_share"] = national_sources["nonharedi_shas_2022_from_source"] / national_sources["nonharedi_shas_2022_from_source"].sum()

    national_summary = pd.DataFrame([
        {"metric": "ordinary_shas_2022", "value": ordinary_shas},
        {"metric": "external_shas_2022", "value": external_shas},
        {"metric": "total_shas_2022", "value": total_shas},
        {"metric": "hybrid_haredi_shas_ordinary", "value": ordinary_haredi},
        {"metric": "external_haredi_shas_imputed", "value": external_haredi},
        {"metric": "hybrid_haredi_shas_total", "value": total_haredi},
        {"metric": "hybrid_haredi_share_total", "value": total_haredi / total_shas if total_shas else np.nan},
        {"metric": "student_anchor_haredi_shas_ordinary", "value": float(hybrid_city["anchor_haredi_shas_2022"].sum())},
        {"metric": "source_model_train_ballots", "value": len(train)},
        {"metric": "source_model_global_mae_pp", "value": global_metrics["mae_pp"]},
        {"metric": "source_model_global_rmse_pp", "value": global_metrics["rmse_pp"]},
        {"metric": "source_model_global_r2", "value": global_metrics["r2"]},
        {"metric": "source_model_cv_mae_pp", "value": cv_metrics["mae_pp"]},
        {"metric": "source_model_cv_rmse_pp", "value": cv_metrics["rmse_pp"]},
        {"metric": "source_model_cv_r2", "value": cv_metrics["r2"]},
        {"metric": "bav_per_haredi_student_2021", "value": bav_per_student},
    ])

    beta_df = pd.DataFrame({
        "source": SOURCE_GROUPS,
        "source_label": [SOURCE_LABELS[c] for c in SOURCE_GROUPS],
        "global_coefficient_to_shas_2022": beta_global,
    })

    # City observed changes for interpretation.
    hybrid_city = hybrid_city.merge(
        city[["city_code", "bav_2021", "bav_2022", "shas_2021", "shas_2022", "utj_2021", "utj_2022", "likud_2021", "likud_2022"]],
        on=["city_code", "shas_2022", "utj_2022"], how="left"
    )
    hybrid_city["shas_2021_scaled_to_2022_bav"] = hybrid_city["shas_2021"] / hybrid_city["bav_2021"].replace(0, np.nan) * hybrid_city["bav_2022"]
    hybrid_city["shas_net_gain_vs_constant_share"] = hybrid_city["shas_2022"] - hybrid_city["shas_2021_scaled_to_2022_bav"]
    hybrid_city["likud_share_change_pp"] = (
        hybrid_city["likud_2022"] / hybrid_city["bav_2022"].replace(0, np.nan)
        - hybrid_city["likud_2021"] / hybrid_city["bav_2021"].replace(0, np.nan)
    ) * 100

    # Save outputs.
    pd.DataFrame(calibration_rows).to_csv(OUT / "calibration_core4.csv", index=False, encoding="utf-8-sig")
    beta_df.to_csv(OUT / "global_source_coefficients.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cv_coefficients).to_csv(OUT / "cv_coefficients.csv", index=False, encoding="utf-8-sig")
    cv.to_csv(OUT / "cv_predictions.csv", index=False, encoding="utf-8-sig")
    city_betas.to_csv(OUT / "city_coefficients.csv", index=False, encoding="utf-8-sig")
    ballot_sources.to_csv(OUT / "ballot_source_attribution.csv", index=False, encoding="utf-8-sig")
    contributions_long.to_csv(OUT / "ballot_source_attribution_long.csv", index=False, encoding="utf-8-sig")
    city_source.to_csv(OUT / "city_source_attribution.csv", index=False, encoding="utf-8-sig")
    haredi_fractions.to_csv(OUT / "city_source_haredi_fractions.csv", index=False, encoding="utf-8-sig")
    hybrid_sources.to_csv(OUT / "city_hybrid_sources.csv", index=False, encoding="utf-8-sig")
    hybrid_city.sort_values("shas_2022", ascending=False).to_csv(OUT / "hybrid_city_summary.csv", index=False, encoding="utf-8-sig")
    national_sources.to_csv(OUT / "national_source_attribution.csv", index=False, encoding="utf-8-sig")
    national_summary.to_csv(OUT / "national_summary.csv", index=False, encoding="utf-8-sig")
    matched[matched["city"] == "בת ים"].to_csv(OUT / "bat_yam_matched_ballots.csv", index=False, encoding="utf-8-sig")
    ballot_sources[ballot_sources["city"] == "בת ים"].to_csv(OUT / "bat_yam_source_ballots.csv", index=False, encoding="utf-8-sig")
    hybrid_sources[hybrid_sources["city"] == "בת ים"].to_csv(OUT / "bat_yam_sources.csv", index=False, encoding="utf-8-sig")
    hybrid_city[hybrid_city["city"] == "בת ים"].to_csv(OUT / "bat_yam_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "education_url": edu_url,
        "student_coefficients": student_coefficients,
        "source_groups": SOURCE_LABELS,
        "global_beta": dict(zip(SOURCE_GROUPS, map(float, beta_global))),
        "global_metrics": global_metrics,
        "cv_metrics": cv_metrics,
        "bav_per_student_2021": bav_per_student,
        "core_source_probabilities": dict(zip(SOURCE_GROUPS, map(float, core_probabilities))),
        "national_summary": dict(zip(national_summary["metric"], national_summary["value"])),
        "bat_yam": hybrid_city[hybrid_city["city"] == "בת ים"].to_dict(orient="records"),
        "top_nonharedi_sources": national_sources.sort_values("nonharedi_shas_2022_from_source", ascending=False).to_dict(orient="records"),
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str)[:50000])


if __name__ == "__main__":
    main()
