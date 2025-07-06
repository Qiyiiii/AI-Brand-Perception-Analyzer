import os
import sys
import sqlite3
import pytest

# ✅ Ensure the project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from info_getter import (
    get_most_asked_aspect,
    get_top_5_aspects,
    get_best_llm_by_model
)

TEST_DB_PATH = "test_getter_eval_cache.db"

@pytest.fixture(scope="function")
def db_conn():
    # Setup: create test db
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""DROP TABLE IF EXISTS eval_cache""")
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
    # ✅ Deterministic mock data
    sample_data = [
        ("What's best for car insurance?", "gpt-4", '["Aviva", "Intact"]', "Intact", "car insurance", "Best coverage"),
        ("Top for car?", "gpt-4", '["Aviva", "Intact"]', "Intact", "car insurance", "Good rates"),
        ("Which is better for health?", "gpt-4", '["Manulife", "Sunlife"]', "Intact", "health", "Surprisingly best"),
        ("Who's good at health?", "claude-3", '["Manulife", "Sunlife"]', "Sunlife", "health", "Fast claims"),
        ("Best home coverage?", "claude-3", '["Aviva", "Intact"]', "Intact", "home", "Trusted service"),
    ]
    cursor.executemany("""
        INSERT INTO eval_cache (question, model_name, candidates, company, aspect, reason)
        VALUES (?, ?, ?, ?, ?, ?)
    """, sample_data)
    conn.commit()

    yield conn

    # Teardown
    conn.close()
    os.remove(TEST_DB_PATH)

def test_get_most_asked_aspect(db_conn):
    aspect, freq = get_most_asked_aspect(db_conn)
    assert aspect != ""
    assert freq == 2

def test_get_top_5_aspects(db_conn):
    aspects = get_top_5_aspects(db_conn)
    aspect_names = [a[0] for a in aspects]
    assert "car insurance" in aspect_names
    assert "health" in aspect_names
    assert "home" in aspect_names
    assert len(aspect_names) <= 5

def test_get_best_llm_by_model(db_conn):
    best = get_best_llm_by_model(db_conn)
    assert "gpt-4" in best
    assert best["gpt-4"][0] == "Intact"  # Appears 3 times for gpt-4
    assert "claude-3" in best
    assert best["claude-3"][0] in {"Sunlife", "Intact"}  # Both chosen once
