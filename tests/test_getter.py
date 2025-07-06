import os
import sys
import sqlite3
import pytest
import pandas as pd
import base64

from io import BytesIO

# ✅ Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from info_getter import (
    get_most_asked_aspect,
    get_top_5_aspects,
    get_best_llm_by_model,
    calculate_favored_company_by_aspect,
    calculate_bias_ratios,
    get_top_3_aspects_by_brand,
    export_all_results,
    plot_company_trend_over_time
)

TEST_DB_PATH = "test_getter_eval_cache.db"

@pytest.fixture(scope="function")
def db_conn():
    # Setup test DB
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS eval_cache")
    cursor.execute("""
        CREATE TABLE eval_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            model_name TEXT NOT NULL,
            candidates TEXT,
            company TEXT,
            aspect TEXT,
            reason TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    data = [
        ("Car insurance best?", "gpt-4", '["Aviva", "Intact"]', "Intact", "car insurance", "Top rated"),
        ("Car again", "gpt-4", '["Aviva", "Intact"]', "Intact", "car insurance", "Best prices"),
        ("Health 1", "gpt-4", '["Sunlife", "Manulife"]', "Intact", "health", "Surprisingly best"),
        ("Health 2", "claude-3", '["Sunlife", "Manulife"]', "Sunlife", "health", "Fast claims"),
        ("Home stuff", "claude-3", '["Aviva", "Intact"]', "Intact", "home", "Reliable")
    ]
    cursor.executemany("""
        INSERT INTO eval_cache (question, model_name, candidates, company, aspect, reason)
        VALUES (?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()

    yield conn

    # Teardown
    conn.close()
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

# ─────────────────────────────
# ✅ Tests
# ─────────────────────────────


def test_get_top_5_aspects(db_conn):
    top = get_top_5_aspects(db_conn)
    names = [a[0] for a in top]
    assert set(["car insurance", "health", "home"]).issubset(names)
    assert len(names) <= 5

def test_get_best_llm_by_model(db_conn):
    best = get_best_llm_by_model(db_conn)
    assert best["gpt-4"][0] == "Intact"
    assert best["claude-3"][0] in {"Sunlife", "Intact"}


def test_calculate_bias_ratios(db_conn):
    ratios = calculate_bias_ratios(db_conn, "gpt-4")
    assert isinstance(ratios, dict)
    assert "Intact" in ratios
    assert abs(sum(ratios.values()) - 1) >0  

def test_get_top_3_aspects_by_brand(db_conn):
    top3 = get_top_3_aspects_by_brand(db_conn)
    assert "gpt-4" in top3
    assert "car insurance" in top3["gpt-4"]["Intact"]

def test_export_all_results_creates_csv(db_conn):
    output = "static/test_export.csv"
    if os.path.exists(output):
        os.remove(output)

    export_all_results(export_csv_path=output)
    assert os.path.exists(output)

    df = pd.read_csv(output)
    assert "question" in df.columns
    os.remove(output)


