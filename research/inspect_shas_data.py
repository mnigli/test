# /// script
# dependencies = ["pandas", "requests", "openpyxl"]
# ///

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

OUT = Path("research/output")
OUT.mkdir(parents=True, exist_ok=True)
S = requests.Session()
S.headers.update({"User-Agent": "shas-2022-research/1.0", "Accept": "*/*"})


def get_bytes(url: str) -> bytes:
    r = S.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def fetch_datastore(resource_id: str, filters: dict[str, Any] | None = None, page_size: int = 10000) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    offset = 0
    fields = []
    total = None
    while True:
        params: dict[str, Any] = {"resource_id": resource_id, "limit": page_size, "offset": offset}
        if filters:
            params["filters"] = json.dumps(filters, ensure_ascii=False)
        r = S.get("https://data.gov.il/api/3/action/datastore_search", params=params, timeout=120)
        r.raise_for_status()
        payload = r.json()
        if not payload.get("success"):
            raise RuntimeError(payload)
        result = payload["result"]
        if not fields:
            fields = result.get("fields", [])
            total = result.get("total")
        batch = result.get("records", [])
        rows.extend(batch)
        if not batch or len(rows) >= int(total or 0) or len(batch) < page_size:
            break
        offset += len(batch)
    return pd.DataFrame(rows), {"total": total, "fields": fields, "rows_fetched": len(rows)}


def read_csv_bytes(data: bytes) -> tuple[pd.DataFrame, str]:
    p = OUT / "ballot25.csv"
    p.write_bytes(data)
    last = None
    for enc in ["utf-8-sig", "utf-8", "cp1255", "iso-8859-8"]:
        try:
            return pd.read_csv(p, encoding=enc), enc
        except Exception as e:
            last = e
    raise last  # type: ignore[misc]


def summarize_json(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"top_type": type(obj).__name__}
    if isinstance(obj, dict):
        out["top_keys"] = list(obj.keys())[:40]
        out["length"] = len(obj)
        for k, v in list(obj.items())[:3]:
            out.setdefault("samples", []).append({"key": str(k), "type": type(v).__name__, "value": v})
    elif isinstance(obj, list):
        out["length"] = len(obj)
        out["samples"] = obj[:3]
    return out


def main() -> None:
    report: dict[str, Any] = {}

    ballot_url = "https://raw.githubusercontent.com/harelc/elections-vote-transfer/master/ballot25.csv"
    df, enc = read_csv_bytes(get_bytes(ballot_url))
    report["ballot"] = {
        "encoding": enc,
        "shape": list(df.shape),
        "columns": [str(c) for c in df.columns],
        "head": df.head(3).where(pd.notna(df.head(3)), None).to_dict(orient="records"),
    }

    for name, url in {
        "locations": "https://raw.githubusercontent.com/harelc/elections-vote-transfer/master/data/ballot_locations_25.json",
        "map": "https://raw.githubusercontent.com/harelc/elections-vote-transfer/master/site/data/map_25.json",
    }.items():
        try:
            raw = get_bytes(url)
            (OUT / f"{name}.json").write_bytes(raw)
            obj = json.loads(raw)
            report[name] = summarize_json(obj)
        except Exception as e:
            report[name] = {"error": repr(e)}

    mosdot, mos_meta = fetch_datastore("5548fd63-5868-4053-ad81-98caddc5e232", {"שנה": 2022})
    report["mosdot"] = {
        "meta": mos_meta,
        "shape": list(mosdot.shape),
        "columns": [str(c) for c in mosdot.columns],
        "head": mosdot.head(5).where(pd.notna(mosdot.head(5)), None).to_dict(orient="records"),
        "פיקוח_counts": mosdot["פיקוח"].value_counts(dropna=False).head(20).to_dict() if "פיקוח" in mosdot else {},
        "גורם_counts": mosdot["גורם מדווח"].value_counts(dropna=False).head(30).to_dict() if "גורם מדווח" in mosdot else {},
    }
    mosdot.to_csv(OUT / "mosdot_2022.csv", index=False, encoding="utf-8-sig")

    coords, coord_meta = fetch_datastore("5c5d6bb0-755d-470d-84b6-d7dd3135ba9c")
    report["coords"] = {
        "meta": coord_meta,
        "shape": list(coords.shape),
        "columns": [str(c) for c in coords.columns],
        "head": coords.head(5).where(pd.notna(coords.head(5)), None).to_dict(orient="records"),
    }
    coords.to_csv(OUT / "education_coordinates.csv", index=False, encoding="utf-8-sig")

    (OUT / "inspection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str)[:30000])


if __name__ == "__main__":
    main()
