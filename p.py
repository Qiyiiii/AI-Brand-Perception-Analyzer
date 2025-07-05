import sqlite3
import json

def print_eval_cache():
    conn = sqlite3.connect("eval_cache.db")
    cursor = conn.cursor()

    cursor.execute("SELECT model_name, aspect, candidates, company FROM eval_cache")
    rows = cursor.fetchall()

    for row in rows:
        model_name, aspect, candidates_json, company = row
        try:
            candidates = json.loads(candidates_json) if isinstance(candidates_json, str) else candidates_json
        except json.JSONDecodeError:
            candidates = candidates_json  # Print raw if can't decode

        print(f"MODEL: {model_name}")
        print(f"ASPECT: {aspect}")
        print(f"CANDIDATES: {candidates}")
        print(f"COMPANY CHOSEN: {company}")
        print("-" * 60)

    conn.close()

if __name__ == "__main__":
    print_eval_cache()
