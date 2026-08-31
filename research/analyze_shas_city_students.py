# /// script
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "requests>=2.32",
#   "openpyxl>=3.1"
# ]
# ///

from __future__ import annotations

import io
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

OUT = Path("research/city_student_output")
OUT.mkdir(parents=True, exist_ok=True)

EDU_URL = "https://aws-e.data.gov.il/dataset/5a9278c8-fa76-46e1-bf2d-0368078c9cc7/resource/5548fd63-5868-4053-ad81-98caddc5e232/download/mosdot.xlsx"
BALLOT_URL = "https://raw.githubusercontent.com/harelc/elections-vote-transfer/master/ballot25.csv"

CALIBRATION_SETS = {
    "large4": ["בני ברק", "מודיעין עילית", "ביתר עילית", "אלעד"],
    "pure6": ["בני ברק", "מודיעין עילית", "ביתר עילית", "אלעד", "עמנואל", "קרית יערים"],
    "pure7": ["בני ברק", "מודיעין עילית", "ביתר עילית", "אלעד", "עמנואל", "קרית יערים", "רכסים"],
}

HEBREW_MARKS = re.compile(r"[\u0591-\u05C7]")
NON_ALNUM = re.compile(r"[^0-9A-Za-z\u0590-\u05FF]+")
SPACES = re.compile(r"\s+")


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    s = unicodedata.normalize("NFKC", str(value))
    s = HEBREW_MARKS.sub("", s)
    s = s.replace("–", "-").replace("—", "-").replace("־", "-")
    s = NON_ALNUM.sub(" ", s)
    return SPACES.sub(" ", s).strip()


def norm_place(value: Any) -> str:
    s = clean_text(value)
    s = s.replace("קריית", "קרית")
    aliases = {
        "תל אביב יפו": "תל אביב יפו",
        "תל אביב יפו ": "תל אביב יפו",
        "קרית יערים טלז סטון": "קרית יערים",
        "טלז סטון": "קרית יערים",
        "נצרת עילית": "נוף הגליל",
        "מודיעין מכבים רעות": "מודיעין מכבים רעות",
        "באקה אל גרביה": "באקה אל גרביה",
        "מעלות תרשיחא": "מעלות תרשיחא",
        "יהוד מונוסון": "יהוד מונוסון",
    }
    return aliases.get(s, s)


def read_official_education() -> tuple[pd.DataFrame, dict[str, Any]]:
    r = requests.get(EDU_URL, timeout=240, headers={"User-Agent": "shas-city-research/1.0"})
    r.raise_for_status()
    raw = r.content
    (OUT / "mosdot_official.xlsx").write_bytes(raw)
    sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None, engine="openpyxl")
    usable = []
    sheet_info = {}
    for name, frame in sheets.items():
        frame.columns = [clean_text(c) for c in frame.columns]
        sheet_info[name] = {"rows": int(len(frame)), "columns": list(frame.columns)}
        required = {"שנה", "שם ישוב", "פיקוח", "סהכ תלמידים במוסד"}
        if required.issubset(frame.columns):
            usable.append(frame)
    if not usable:
        raise RuntimeError(f"No usable education sheet. Sheets: {sheet_info}")
    df = pd.concat(usable, ignore_index=True)
    return df, {"bytes": len(raw), "sheets": sheet_info, "rows": len(df)}


def read_ballots() -> pd.DataFrame:
    r = requests.get(BALLOT_URL, timeout=180, headers={"User-Agent": "shas-city-research/1.0"})
    r.raise_for_status()
    raw = r.content
    (OUT / "ballot25.csv").write_bytes(raw)
    last: Exception | None = None
    for enc in ["utf-8-sig", "utf-8", "cp1255", "iso-8859-8"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Could not decode ballot file: {last}")


def prepare_election(ballots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ballots = ballots.copy()
    for c in ["כשרים", "מצביעים", "בזב", "שס", "ג"]:
        ballots[c] = pd.to_numeric(ballots[c], errors="coerce").fillna(0)
    ballots["שם ישוב"] = ballots["שם ישוב"].astype(str).map(clean_text)
    ballots["מפתח"] = ballots["שם ישוב"].map(norm_place)

    grouped = ballots.groupby(["מפתח", "שם ישוב"], as_index=False).agg(
        כשרים=("כשרים", "sum"),
        מצביעים=("מצביעים", "sum"),
        בעלי_זכות=("בזב", "sum"),
        שס=("שס", "sum"),
        ג=("ג", "sum"),
        קלפיות=("קלפי", "count"),
    )
    grouped["שס_וג"] = grouped["שס"] + grouped["ג"]
    external_mask = grouped["שם ישוב"].str.contains("מעטפות", na=False)
    external = grouped.loc[external_mask].copy()
    ordinary = grouped.loc[~external_mask].copy()
    return ordinary, external


def prepare_education(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df = df.copy()
    df["שנה"] = pd.to_numeric(df["שנה"], errors="coerce")
    df = df[df["שנה"].notna()].copy()
    df["שנה"] = df["שנה"].astype(int)
    df["תלמידים"] = pd.to_numeric(df["סהכ תלמידים במוסד"], errors="coerce")
    df["חסר_תלמידים"] = df["תלמידים"].isna()
    df["תלמידים"] = df["תלמידים"].fillna(0).clip(lower=0)
    df["פיקוח_נקי"] = df["פיקוח"].map(clean_text)
    df["חרדי"] = df["פיקוח_נקי"].str.contains("חרדי", na=False)
    df["יישוב_מקור"] = df["שם ישוב"].map(clean_text)
    df["מפתח"] = df["יישוב_מקור"].map(norm_place)

    type_text = pd.Series("", index=df.index, dtype="object")
    for col in ["סוג מוסד", "סוג מסגרת אירגונית", "סוג חינוך מוסד"]:
        if col in df.columns:
            type_text = type_text + " " + df[col].map(clean_text)
    df["גן"] = type_text.str.contains("גן ילדים|גני ילדים", regex=True, na=False)
    df["מיוחד"] = type_text.str.contains("מיוחד", na=False)

    rows = []
    for (year, key), g in df.groupby(["שנה", "מפתח"], dropna=False):
        h = g[g["חרדי"]]
        rows.append({
            "שנה": int(year),
            "מפתח": key,
            "שם_ישוב_חינוך": g["יישוב_מקור"].mode().iat[0] if not g.empty else "",
            "כלל_התלמידים": float(g["תלמידים"].sum()),
            "תלמידים_חרדים_הכל": float(h["תלמידים"].sum()),
            "תלמידים_חרדים_ללא_גנים": float(h.loc[~h["גן"], "תלמידים"].sum()),
            "תלמידים_חרדים_רגיל": float(h.loc[~h["מיוחד"], "תלמידים"].sum()),
            "תלמידים_חרדים_ללא_גנים_רגיל": float(h.loc[(~h["גן"]) & (~h["מיוחד"]), "תלמידים"].sum()),
            "תלמידים_חרדים_גנים": float(h.loc[h["גן"], "תלמידים"].sum()),
            "מוסדות_חרדיים": int(len(h)),
            "מוסדות_חרדיים_חסר_מספר": int(h["חסר_תלמידים"].sum()),
        })
    agg = pd.DataFrame(rows)

    yearly = df.groupby("שנה", as_index=False).agg(
        רשומות=("מפתח", "size"),
        תלמידים=("תלמידים", "sum"),
        תלמידים_חרדים=("תלמידים", lambda s: float(s[df.loc[s.index, "חרדי"]].sum())),
        חסרי_מספר=("חסר_תלמידים", "sum"),
    )
    meta = {
        "years": sorted(int(x) for x in df["שנה"].unique()),
        "supervision_values": df["פיקוח_נקי"].value_counts(dropna=False).head(30).to_dict(),
        "rows": int(len(df)),
    }
    return agg, yearly, meta


def build_join(election: pd.DataFrame, edu_agg: pd.DataFrame, year: int) -> pd.DataFrame:
    e = election.copy()
    y = edu_agg[edu_agg["שנה"] == year].copy()
    out = e.merge(y, on="מפתח", how="left", validate="one_to_one")
    student_cols = [c for c in y.columns if c.startswith("תלמידים_") or c in {"כלל_התלמידים", "מוסדות_חרדיים", "מוסדות_חרדיים_חסר_מספר"}]
    out["נמצאה_התאמת_חינוך"] = out["שנה"].notna()
    for c in student_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["אחוז_תלמידים_חרדים"] = np.where(
        out["כלל_התלמידים"].fillna(0) > 0,
        out["תלמידים_חרדים_הכל"] / out["כלל_התלמידים"],
        np.nan,
    )
    return out


def get_calibration(joined: pd.DataFrame, cities: list[str], student_col: str) -> pd.DataFrame:
    keys = [norm_place(x) for x in cities]
    c = joined[joined["מפתח"].isin(keys)].copy()
    c["עיר_כיול"] = c["מפתח"]
    c["תלמידי_כיול"] = pd.to_numeric(c[student_col], errors="coerce")
    c = c[c["תלמידי_כיול"].fillna(0) > 0].copy()
    c["קולות_חרדיים_בפועל"] = c["שס_וג"]
    c["קולות_לתלמיד"] = c["קולות_חרדיים_בפועל"] / c["תלמידי_כיול"]
    return c


def loo_metrics(cal: pd.DataFrame, estimator: str) -> tuple[pd.DataFrame, dict[str, float]]:
    detail = []
    for idx, row in cal.iterrows():
        train = cal.drop(index=idx)
        if estimator == "weighted":
            coefficient = train["קולות_חרדיים_בפועל"].sum() / train["תלמידי_כיול"].sum()
        elif estimator == "median":
            coefficient = train["קולות_לתלמיד"].median()
        else:
            raise ValueError(estimator)
        pred = coefficient * row["תלמידי_כיול"]
        actual = row["קולות_חרדיים_בפועל"]
        err = pred - actual
        pct = err / actual if actual else np.nan
        detail.append({
            "עיר": row["מפתח"],
            "תלמידים": row["תלמידי_כיול"],
            "שס": row["שס"],
            "ג": row["ג"],
            "קולות_בפועל": actual,
            "מקדם_ללא_העיר": coefficient,
            "תחזית": pred,
            "שגיאה": err,
            "שגיאה_באחוזים": pct,
        })
    d = pd.DataFrame(detail)
    abs_pct = d["שגיאה_באחוזים"].abs().dropna()
    metrics = {
        "מספר_ערים": float(len(d)),
        "MAPE": float(abs_pct.mean()) if len(abs_pct) else np.nan,
        "MdAPE": float(abs_pct.median()) if len(abs_pct) else np.nan,
        "MaxAPE": float(abs_pct.max()) if len(abs_pct) else np.nan,
        "Bias": float(d["שגיאה"].sum() / d["קולות_בפועל"].sum()) if d["קולות_בפועל"].sum() else np.nan,
    }
    return d, metrics


def model_coefficient(cal: pd.DataFrame, estimator: str) -> float:
    if estimator == "weighted":
        return float(cal["קולות_חרדיים_בפועל"].sum() / cal["תלמידי_כיול"].sum())
    return float(cal["קולות_לתלמיד"].median())


def estimate_cities(joined: pd.DataFrame, student_col: str, coefficient: float, calibration_cities: list[str]) -> pd.DataFrame:
    out = joined.copy()
    pure_keys = set(norm_place(x) for x in calibration_cities)
    out["תלמידי_מודל"] = pd.to_numeric(out[student_col], errors="coerce")
    out["תחזית_בלוק_חרדי_לפני_חיתוך"] = out["תלמידי_מודל"] * coefficient
    out["תחזית_בלוק_חרדי"] = out["תחזית_בלוק_חרדי_לפני_חיתוך"].clip(lower=0)
    out["תחזית_בלוק_חרדי"] = np.minimum(out["תחזית_בלוק_חרדי"], out["שס_וג"])
    out["שס_חרדי_משוער"] = (out["תחזית_בלוק_חרדי"] - out["ג"]).clip(lower=0)
    out["שס_חרדי_משוער"] = np.minimum(out["שס_חרדי_משוער"], out["שס"])
    out["עיר_חרדית_כיול"] = out["מפתח"].isin(pure_keys)
    # In calibration localities, Shas votes are treated as Haredi ground truth.
    out.loc[out["עיר_חרדית_כיול"], "שס_חרדי_משוער"] = out.loc[out["עיר_חרדית_כיול"], "שס"]
    out.loc[out["עיר_חרדית_כיול"], "תחזית_בלוק_חרדי"] = out.loc[out["עיר_חרדית_כיול"], "שס_וג"]
    out["שס_לא_חרדי_משוער"] = out["שס"] - out["שס_חרדי_משוער"]
    out["אחוז_שס_חרדי"] = np.where(out["שס"] > 0, out["שס_חרדי_משוער"] / out["שס"], np.nan)
    out["מצב_חיתוך"] = np.select(
        [
            ~out["נמצאה_התאמת_חינוך"],
            out["תחזית_בלוק_חרדי_לפני_חיתוך"] < out["ג"],
            out["תחזית_בלוק_חרדי_לפני_חיתוך"] > out["שס_וג"],
        ],
        ["ללא נתוני חינוך", "תחזית נמוכה מקולות ג", "תחזית גבוהה משס+ג"],
        default="ללא חיתוך",
    )
    return out


def main() -> None:
    edu_raw, edu_download_meta = read_official_education()
    ballots = read_ballots()
    election, external = prepare_election(ballots)
    edu_agg, edu_yearly, edu_meta = prepare_education(edu_raw)

    years = edu_meta["years"]
    target_years = [y for y in [2021, 2022, 2023, 2024, 2025, 2026] if y in years]
    if not target_years:
        target_years = years[-3:]

    scopes = {
        "all_haredi": "תלמידים_חרדים_הכל",
        "no_kindergarten": "תלמידים_חרדים_ללא_גנים",
        "regular_only": "תלמידים_חרדים_רגיל",
        "school_regular": "תלמידים_חרדים_ללא_גנים_רגיל",
    }

    metrics_rows = []
    calibration_rows = []
    estimates_by_scenario = []
    joined_by_year: dict[int, pd.DataFrame] = {}

    for year in target_years:
        joined = build_join(election, edu_agg, year)
        joined_by_year[year] = joined
        for scope_name, student_col in scopes.items():
            for set_name, cities in CALIBRATION_SETS.items():
                cal = get_calibration(joined, cities, student_col)
                missing = sorted(set(norm_place(x) for x in cities) - set(cal["מפתח"]))
                if len(cal) < 3:
                    continue
                for estimator in ["weighted", "median"]:
                    detail, metrics = loo_metrics(cal, estimator)
                    coefficient = model_coefficient(cal, estimator)
                    metrics_rows.append({
                        "שנה": year,
                        "הגדרת_תלמידים": scope_name,
                        "עמודת_תלמידים": student_col,
                        "קבוצת_כיול": set_name,
                        "אומד": estimator,
                        "מקדם_קולות_לתלמיד": coefficient,
                        "ערי_כיול_שנמצאו": len(cal),
                        "ערי_כיול_חסרות": ", ".join(missing),
                        **metrics,
                    })
                    detail.insert(0, "שנה", year)
                    detail.insert(1, "הגדרת_תלמידים", scope_name)
                    detail.insert(2, "קבוצת_כיול", set_name)
                    detail.insert(3, "אומד", estimator)
                    calibration_rows.append(detail)

                    est = estimate_cities(joined, student_col, coefficient, cities)
                    est["שנה"] = year
                    est["הגדרת_תלמידים"] = scope_name
                    est["קבוצת_כיול"] = set_name
                    est["אומד"] = estimator
                    est["מקדם"] = coefficient
                    estimates_by_scenario.append(est)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["שנה", "MAPE", "MdAPE"])
    calibration_df = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()
    all_estimates = pd.concat(estimates_by_scenario, ignore_index=True) if estimates_by_scenario else pd.DataFrame()

    preferred_year = 2022 if 2022 in target_years else min(target_years, key=lambda y: abs(y - 2022))
    preferred_mask = (
        (metrics_df["שנה"] == preferred_year)
        & (metrics_df["הגדרת_תלמידים"] == "all_haredi")
        & (metrics_df["קבוצת_כיול"] == "pure7")
        & (metrics_df["אומד"] == "weighted")
    )
    if preferred_mask.any():
        main_metric = metrics_df.loc[preferred_mask].iloc[0]
    else:
        # Deterministic fallback: closest year, then lowest MAPE among weighted pure7 models.
        fallback = metrics_df[(metrics_df["קבוצת_כיול"] == "pure7") & (metrics_df["אומד"] == "weighted")].copy()
        fallback["מרחק_מבחירות"] = (fallback["שנה"] - 2022).abs()
        main_metric = fallback.sort_values(["מרחק_מבחירות", "MAPE"]).iloc[0]

    main_filter = (
        (all_estimates["שנה"] == int(main_metric["שנה"]))
        & (all_estimates["הגדרת_תלמידים"] == main_metric["הגדרת_תלמידים"])
        & (all_estimates["קבוצת_כיול"] == main_metric["קבוצת_כיול"])
        & (all_estimates["אומד"] == main_metric["אומד"])
    )
    main_est = all_estimates.loc[main_filter].copy()

    matched = main_est[main_est["נמצאה_התאמת_חינוך"]].copy()
    unmatched = main_est[~main_est["נמצאה_התאמת_חינוך"]].copy()
    external_shas = float(external["שס"].sum())
    external_g = float(external["ג"].sum())
    ordinary_shas = float(election["שס"].sum())
    estimated_haredi_matched = float(matched["שס_חרדי_משוער"].sum())
    estimated_non_haredi_matched = float(matched["שס_לא_חרדי_משוער"].sum())
    unmatched_shas = float(unmatched["שס"].sum())

    # Optional imputation for external envelopes using the estimated Haredi Shas:G split in ordinary ballots.
    ordinary_g = float(election["ג"].sum())
    ratio_hs_to_g = estimated_haredi_matched / ordinary_g if ordinary_g else np.nan
    external_haredi_imputed = min(external_shas, external_g * ratio_hs_to_g) if np.isfinite(ratio_hs_to_g) else np.nan

    national_summary = pd.DataFrame([
        {"מדד": "שנת נתוני חינוך במודל הראשי", "ערך": int(main_metric["שנה"])},
        {"מדד": "הגדרת תלמידים במודל הראשי", "ערך": main_metric["הגדרת_תלמידים"]},
        {"מדד": "קבוצת כיול", "ערך": main_metric["קבוצת_כיול"]},
        {"מדד": "מקדם קולות חרדיים לתלמיד", "ערך": float(main_metric["מקדם_קולות_לתלמיד"])},
        {"מדד": "MAPE השארת עיר בחוץ", "ערך": float(main_metric["MAPE"])},
        {"מדד": "MdAPE השארת עיר בחוץ", "ערך": float(main_metric["MdAPE"])},
        {"מדד": "קולות שס ביישובים רגילים", "ערך": ordinary_shas},
        {"מדד": "קולות שס ביישובים עם התאמת חינוך", "ערך": float(matched["שס"].sum())},
        {"מדד": "קולות שס ללא התאמת חינוך", "ערך": unmatched_shas},
        {"מדד": "שס חרדי משוער ביישובים מותאמים", "ערך": estimated_haredi_matched},
        {"מדד": "שס לא חרדי משוער ביישובים מותאמים", "ערך": estimated_non_haredi_matched},
        {"מדד": "קולות שס במעטפות חיצוניות", "ערך": external_shas},
        {"מדד": "קולות ג במעטפות חיצוניות", "ערך": external_g},
        {"מדד": "אומדן שס חרדי במעטפות לפי יחס שס חרדי/ג", "ערך": external_haredi_imputed},
        {"מדד": "אומדן שס חרדי ארצי כולל זקיפת מעטפות", "ערך": estimated_haredi_matched + (external_haredi_imputed if np.isfinite(external_haredi_imputed) else 0)},
        {"מדד": "אחוז שס חרדי ארצי כולל זקיפת מעטפות", "ערך": (estimated_haredi_matched + (external_haredi_imputed if np.isfinite(external_haredi_imputed) else 0)) / (ordinary_shas + external_shas)},
    ])

    scenario_summary = all_estimates.groupby(["שנה", "הגדרת_תלמידים", "קבוצת_כיול", "אומד", "מקדם"], as_index=False).agg(
        שס_חרדי_משוער=("שס_חרדי_משוער", "sum"),
        שס_לא_חרדי_משוער=("שס_לא_חרדי_משוער", "sum"),
        שס_מכוסה=("שס", "sum"),
        יישובים=("מפתח", "count"),
        יישובים_מותאמים=("נמצאה_התאמת_חינוך", "sum"),
    )
    scenario_summary = scenario_summary.merge(
        metrics_df[["שנה", "הגדרת_תלמידים", "קבוצת_כיול", "אומד", "MAPE", "MdAPE", "MaxAPE"]],
        on=["שנה", "הגדרת_תלמידים", "קבוצת_כיול", "אומד"],
        how="left",
    )

    main_cols = [
        "שם ישוב", "מפתח", "כשרים", "שס", "ג", "שס_וג", "כלל_התלמידים",
        "תלמידים_חרדים_הכל", "תלמידים_חרדים_ללא_גנים", "תלמידים_חרדים_גנים",
        "אחוז_תלמידים_חרדים", "מוסדות_חרדיים", "תחזית_בלוק_חרדי_לפני_חיתוך",
        "תחזית_בלוק_חרדי", "שס_חרדי_משוער", "שס_לא_חרדי_משוער", "אחוז_שס_חרדי",
        "עיר_חרדית_כיול", "מצב_חיתוך", "נמצאה_התאמת_חינוך",
    ]
    main_est = main_est[main_cols].sort_values("שס", ascending=False)

    bat_yam = main_est[main_est["מפתח"] == norm_place("בת ים")].copy()
    calibration_main = calibration_df[
        (calibration_df["שנה"] == int(main_metric["שנה"]))
        & (calibration_df["הגדרת_תלמידים"] == main_metric["הגדרת_תלמידים"])
        & (calibration_df["קבוצת_כיול"] == main_metric["קבוצת_כיול"])
        & (calibration_df["אומד"] == main_metric["אומד"])
    ].copy()

    unmatched_out = unmatched[["שם ישוב", "מפתח", "שס", "ג", "כשרים"]].sort_values("שס", ascending=False)

    edu_agg.to_csv(OUT / "education_by_locality_year.csv", index=False, encoding="utf-8-sig")
    edu_yearly.to_csv(OUT / "education_yearly_coverage.csv", index=False, encoding="utf-8-sig")
    election.to_csv(OUT / "election_by_locality.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(OUT / "model_metrics.csv", index=False, encoding="utf-8-sig")
    calibration_df.to_csv(OUT / "calibration_loo_all.csv", index=False, encoding="utf-8-sig")
    calibration_main.to_csv(OUT / "calibration_loo_main.csv", index=False, encoding="utf-8-sig")
    scenario_summary.to_csv(OUT / "scenario_summary.csv", index=False, encoding="utf-8-sig")
    main_est.to_csv(OUT / "city_estimates_main.csv", index=False, encoding="utf-8-sig")
    bat_yam.to_csv(OUT / "bat_yam_main.csv", index=False, encoding="utf-8-sig")
    unmatched_out.to_csv(OUT / "unmatched_localities.csv", index=False, encoding="utf-8-sig")
    external.to_csv(OUT / "external_envelopes.csv", index=False, encoding="utf-8-sig")
    national_summary.to_csv(OUT / "national_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "education_download": edu_download_meta,
        "education": edu_meta,
        "target_years": target_years,
        "main_model": main_metric.to_dict(),
        "national_summary": dict(zip(national_summary["מדד"], national_summary["ערך"])),
        "bat_yam": bat_yam.to_dict(orient="records"),
        "top_metrics": metrics_df.head(20).to_dict(orient="records"),
        "top_unmatched": unmatched_out.head(30).to_dict(orient="records"),
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str)[:40000])


if __name__ == "__main__":
    main()
