import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "predictions.db"

def _connect():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT, customer_data TEXT, churn_probability REAL, prediction TEXT
    )""")
    return conn

def save_prediction(customer_data, probability, prediction):
    import json
    conn = _connect()
    conn.execute("INSERT INTO predictions(created_at, customer_data, churn_probability, prediction) VALUES(?,?,?,?)",
                 (datetime.now().isoformat(timespec="seconds"), json.dumps(customer_data), probability, prediction))
    conn.commit(); conn.close()

def get_recent_predictions(limit=100):
    import json
    conn = _connect()
    rows = conn.execute("SELECT created_at, customer_data, churn_probability, prediction FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [{"created_at":r[0], "customer_data":json.loads(r[1]), "churn_probability":round(r[2],4), "prediction":r[3]} for r in rows]
