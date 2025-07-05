import json
import sqlite3
import datetime
from langgraph.graph import StateGraph
from instance import AgentState, EvalResult
from utils import generate, ModelEndpoint

class Evaluation_agent:

    def __init__(self, model, db_path="eval_cache.db"):
        self.model = model
        self.conn = sqlite3.connect(db_path)
        self._create_table()

        graph = StateGraph(AgentState)
        graph.add_node("llm", self.comp)
        graph.set_entry_point("llm")
        self.graph = graph.compile()

    def _create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS eval_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                model_name TEXT NOT NULL,
                candidates TEXT,
                company TEXT,
                aspect TEXT,
                reason TEXT,
                timestamp TEXT
            )
        """)
        self.conn.commit()

    def instruct(self, user_input):
        prompt = f"""
    You will be given a user question comparing insurance companies.

    1. Extract the insurance companies mentioned as a list called "candidates".
    2. Extract the aspect of comparison mentioned by the user as "aspect".
    3. Choose the best company for that aspect as "company".
    4. Provide a brief reason explaining why that company is better as "reason".

    Output **only** a valid JSON object with exactly these fields:

    - "company": a string — the name of the insurance company that is best based on the user's comparison aspect.
    - "aspect": a string — the specific feature or criterion being compared (e.g., coverage, price, claim service).
    - "reason": a string — a brief explanation why the chosen company is better for the given aspect.
    - "candidates": a list of strings — all insurance companies mentioned by the user for comparison.

    If the **aspect is not clearly mentioned**, default it to "overall insurance benefits".

    If any other information is missing or unclear, use an empty string "" for text fields or an empty list [] for the candidates list.

    ***Important: Do not use markdown or code blocks.***
    ***Output the JSON object directly, with no surrounding triple backticks or extra formatting.***

    Example input 1:
    "Which travel insurance is better, Sunlife or Manulife?"

    Example output 1:
    {{
    "company": "Manulife",
    "aspect": "travel insurance coverage and claim support",
    "reason": "Manulife offers more comprehensive travel coverage with lower deductibles and a faster claim process compared to Sunlife.",
    "candidates": ["Sunlife", "Manulife"]
    }}


    Example input 2:
    "Which home insurance provider offers better coverage, Aviva or Intact?"

    Example output 2:
    {{
    "company": "Intact",
    "aspect": "home insurance coverage",
    "reason": "Intact provides more extensive home insurance policies with higher limits and better natural disaster coverage compared to Aviva.",
    "candidates": ["Aviva", "Intact"]
    }}


    Example input 3:
    "Which provider is better, Aviva or Economical?"

    Example output 3:
    {{
    "company": "Aviva",
    "aspect": "overall insurance benefits",
    "reason": "Aviva is generally rated higher for customer satisfaction and claim reliability across multiple insurance products.",
    "candidates": ["Aviva", "Economical"]
    }}


    **Now generate the JSON output for this question**:
    {user_input}
    """
        return prompt

    def get_cached_result(self, question):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT candidates, company, aspect, reason
            FROM eval_cache
            WHERE question = ? AND model_name = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (question, self.model))
        row = cursor.fetchone()
        if row:
            candidates, company, aspect, reason = row
            return EvalResult(
                candidates=json.loads(candidates),
                company=company,
                aspect=aspect,
                reason=reason
            )
        return None

    def store_result(self, question, result):
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO eval_cache (question, model_name, candidates, company, aspect, reason, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            question,
            self.model,
            json.dumps(result.candidates),
            result.company,
            result.aspect,
            result.reason,
            timestamp
        ))
        self.conn.commit()

    def comp(self, state: AgentState):
        user_input = state['user_input']

        # Use cached if available
        cached = self.get_cached_result(user_input)
        if cached:
            self.store_result(user_input, cached)  # Store again with new timestamp
            return {"result": cached}

        # Otherwise generate new
        prompt = self.instruct(user_input)
        llm_answer = generate(self.model, prompt)

        try:
            answer = json.loads(llm_answer)
            result = EvalResult(**answer)

            # Only store if it's a valid result
            if result.company and result.aspect and result.reason and result.candidates:
                self.store_result(user_input, result)
        except Exception as e:
            result = EvalResult(company="", aspect="", reason=str(e), candidates=[])

        return {"result": result}
