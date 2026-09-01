from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

OUT = Path("research/official_ballots")
OUT.mkdir(parents=True, exist_ok=True)

RESOURCES = {
    "ballot24_official": "419be3b0-fd30-455a-afc0-034ec36be990",
    "ballot25_official": "cc223336-07bc-485d-b160-62df92967c0a",
}

S = requests.Session()
S.headers.update({"User-Agent": "shas-research/1.0", "Accept": "application/json"})


def fetch_resource(resource_id: str, page_size: int = 5000) -> tuple[pd.DataFrame, dict]:
    records = []
    offset = 0
    fields = None
    total = None
    while True:
        r = S.get(
            "https://data.gov.il/api/3/action/datastore_search",
            params={"resource_id": resource_id, "limit": page_size, "offset": offset},
            timeout=120,
        )
        r.raise_for_status()
        payload = r.json()
        if not payload.get("success"):
            raise RuntimeError(payload)
        result = payload["result"]
        if fields is None:
            fields = result.get("fields", [])
            total = int(result.get("total", 0))
        batch = result.get("records", [])
        records.extend(batch)
        print(resource_id, "offset", offset, "batch", len(batch), "total", total)
        if not batch or len(records) >= total:
            break
        offset += len(batch)
    return pd.DataFrame(records), {"resource_id": resource_id, "total": total, "fetched": len(records), "fields": fields}


def main() -> None:
    metadata = {}
    for name, rid in RESOURCES.items():
        df, meta = fetch_resource(rid)
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
        df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
        metadata[name] = meta
        print(name, df.shape)
        for col in ["שס", "ג", "כשרים", "בזב", "מצביעים"]:
            if col in df.columns:
                print(name, col, pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
