"""Append-only, queryable experimental decision records."""
import json
import math
import sqlite3
from pathlib import Path

from .research_store import utc_now


class HypothesisMemory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, hypothesis_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL)")

    def record(self, event_id, hypothesis_id, kind, payload):
        encoded = json.dumps(payload, sort_keys=True, default=str)
        with sqlite3.connect(self.path) as conn:
            old = conn.execute("SELECT payload FROM events WHERE event_id=?", (event_id,)).fetchone()
            if old and old[0] != encoded:
                raise ValueError("Conflicting hypothesis event replay")
            conn.execute("INSERT OR IGNORE INTO events VALUES (?,?,?,?,?)", (event_id, hypothesis_id, kind, encoded, utc_now()))

    def retrieve(self, query="", limit=20):
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute("SELECT hypothesis_id,kind,payload,created_at FROM events WHERE payload LIKE ? ORDER BY rowid DESC LIMIT ?", (f"%{query}%", min(limit,100))).fetchall()
        return [dict(hypothesis_id=h, kind=k, payload=json.loads(p), created_at=t) for h,k,p,t in rows]


def champion_snapshot(terminal):
    groups, families = {}, {}
    for row in terminal:
        result = row.get("result", {})
        if row.get("status") != "completed" or result.get("objective_value") is None or not math.isfinite(result["objective_value"]):
            continue
        recipe = row["payload"]["recipe"]
        lineage = result.get("lineage", {})
        key = json.dumps({"target":recipe["target"],"objective":result["objective_name"],"protocol":lineage.get("protocol_id",lineage.get("protocol_hash")),"metric_contract":result.get("metrics",{}).get("metric_contract_version")}, sort_keys=True)
        candidate = {"attempt_id":row["attempt_id"],"objective_name":result["objective_name"],"objective_value":result["objective_value"],"model_kind":recipe["model"]["kind"],"feature_discovery":recipe.get("feature_discovery"),"recipe":recipe}
        for mapping, index in ((groups,key),(families,key+":"+recipe["model"]["kind"])):
            if index not in mapping or candidate["objective_value"] < mapping[index]["objective_value"]:
                mapping[index] = candidate
    return {"global_champions":groups,"family_champions":families,"terminal_watermark":len(terminal),"scope":"development only; objective values minimized; ranking exposes negative NDCG"}
