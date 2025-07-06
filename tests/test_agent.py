import os
import sys
import json
import pytest
from unittest.mock import patch

# ✅ Ensure project root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent import Evaluation_agent

TEST_DB_PATH = "test_eval_cache.db"

@pytest.fixture(scope="function")
def agent():
    agent = Evaluation_agent("TEST_MODEL", db_path=TEST_DB_PATH)
    yield agent
    agent.conn.close()  # ✅ Close DB connection
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

# ✅ Patch where generate is actually used (in agent.py)
@patch("agent.generate", return_value=json.dumps({
    "company": "Intact",
    "aspect": "home coverage",
    "reason": "Better claim support",
    "candidates": ["Aviva", "Intact"]
}))
def test_good_input(mock_generate, agent):
    user_question = "Which insurance is better for home coverage, Aviva or Intact?"
    result = agent.comp({"user_input": user_question})["result"]

    assert result.company == "Intact"
    assert result.aspect == "home coverage"
    assert "Better" in result.reason
    assert "Aviva" in result.candidates and "Intact" in result.candidates

@patch("agent.generate", return_value=json.dumps({
    "company": "",
    "aspect": "",
    "reason": "Input unclear",
    "candidates": []
}))
def test_bad_input(mock_generate, agent):
    user_question = "Purple bananas running backwards"
    result = agent.comp({"user_input": user_question})["result"]

    assert result.company == ""
    assert result.aspect == ""
    assert "unclear" in result.reason.lower()
    assert result.candidates == []
