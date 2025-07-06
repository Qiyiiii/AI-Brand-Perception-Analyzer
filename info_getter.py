import json
from collections import defaultdict, Counter
import pandas as pd
from agent import Evaluation_agent
from utils import ModelEndpoint
from instance import EvalResult
from math import log
import sqlite3
import math
from typing import List
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import base64
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
def get_most_asked_aspect(conn):
    """
    Retrieve the most frequently asked aspect across all records in the eval_cache table.

    Args:
        conn (sqlite3.Connection): SQLite database connection.

    Returns:
        tuple: (aspect, frequency) of the most asked aspect.
               Returns None if no records found.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT aspect, COUNT(*) as freq
        FROM eval_cache
        GROUP BY aspect
        ORDER BY freq DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    return row  


def get_top_5_aspects(conn):
    """
    Retrieve the top 5 most frequently asked aspects from the eval_cache table.

    Args:
        conn (sqlite3.Connection): SQLite database connection.

    Returns:
        list[tuple]: A list of up to 5 tuples, each containing (aspect, frequency).
                     Returns an empty list if no records are found.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT aspect, COUNT(*) as freq
        FROM eval_cache
        GROUP BY aspect
        ORDER BY freq DESC
        LIMIT 5
    """)
    rows = cursor.fetchall()
    return rows  # list of (aspect, freq)
    
def get_best_llm_by_model(conn):
    """
    Retrieve the most frequently chosen company for each model.

    Args:
        conn (sqlite3.Connection): SQLite database connection.

    Returns:
        dict: Mapping from model_name to a tuple of (company, frequency),
              representing the best company for that model.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT model_name, company, COUNT(*) as freq
        FROM eval_cache
        GROUP BY model_name, company
        ORDER BY model_name, freq DESC
    """)
    rows = cursor.fetchall()

    # Aggregate to get the top company per model
    best_per_model = {}
    for model_name, company, freq in rows:
        if model_name not in best_per_model:
            best_per_model[model_name] = (company, freq)
    return best_per_model


def get_records_by_models_any(conn, models):
    """
    Retrieve all records where the model_name is any of the specified models.

    Args:
        conn (sqlite3.Connection): SQLite database connection.
        models (list[str]): List of model names.

    Returns:
        list of tuples: All matching records from eval_cache.
    """
    placeholders = ",".join("?" for _ in models)
    query = f"SELECT * FROM eval_cache WHERE model_name IN ({placeholders})"
    cursor = conn.cursor()
    cursor.execute(query, models)
    return cursor.fetchall()


def get_records_by_models_all(conn, models):
    """
    Retrieve records for questions that appear under all specified models.

    Args:
        conn (sqlite3.Connection): SQLite database connection.
        models (list[str]): List of model names.

    Returns:
        list of tuples: Records for questions common to all specified models.
    """
    cursor = conn.cursor()
    n = len(models)
    placeholders = ",".join("?" for _ in models)
    query = f"""
    SELECT question
    FROM eval_cache
    WHERE model_name IN ({placeholders})
    GROUP BY question
    HAVING COUNT(DISTINCT model_name) = ?
    """
    cursor.execute(query, (*models, n))
    questions_in_all_models = [row[0] for row in cursor.fetchall()]

    # Now get all records for those questions and specified models
    if not questions_in_all_models:
        return []

    placeholders_q = ",".join("?" for _ in questions_in_all_models)
    query2 = f"""
    SELECT * FROM eval_cache
    WHERE model_name IN ({placeholders})
      AND question IN ({placeholders_q})
    """
    cursor.execute(query2, (*models, *questions_in_all_models))
    return cursor.fetchall()


def evaluate_question_with_models(agent_class, model_names, question):
    """
    Evaluate a user question with multiple models using the specified agent class.

    Args:
        agent_class (class): The agent class to instantiate for each model.
        model_names (list[str]): List of model names.
        question (str): User question to evaluate.

    Returns:
        dict: Mapping from model name to the evaluation result.
    """
    results = {}
    for model_name in model_names:
        # Create a fresh tracker DataFrame for each agent instance
        tracker_df = pd.DataFrame(columns=[
            "question", "model_name", "candidates", "company", "aspect", "reason"
        ])

        # Instantiate the agent for this model
        agent = agent_class(model=model_name, db_path='eval_cache.db')
 
        # Run the question through the agent
        response = agent.graph.invoke({"user_input": question})

        results[model_name] = response["result"]

    return results


def get_company_probability(conn, model_name, company_name):
    """
    Calculate the probability that the specified company is chosen by the model,
    given that the company appears in the candidates.

    Args:
        conn (sqlite3.Connection): SQLite database connection.
        model_name (str): The model name to filter records.
        company_name (str): The company name to calculate probability for.

    Returns:
        float: Probability value between 0 and 1.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT candidates, company FROM eval_cache WHERE model_name = ?
    """, (model_name,))
    records = cursor.fetchall()

    if not records:
        return 0.0  # no data, zero probability

    total_candidate_occurrences = 0
    total_chosen_occurrences = 0

    for candidates_json, chosen_company in records:
        candidates = json.loads(candidates_json) if isinstance(candidates_json, str) else candidates_json
        if company_name in candidates:
            total_candidate_occurrences += 1
            if chosen_company == company_name:
                total_chosen_occurrences += 1

    if total_candidate_occurrences == 0:
        return 0.0  # company never appeared as candidate in this model

    probability = total_chosen_occurrences / total_candidate_occurrences
    return probability


def get_favorite_company_and_probabilities(conn, model_name):
    """
    For a given model, calculate probabilities for all companies and return the company
    with the highest probability along with the full probability dictionary.

    Args:
        conn (sqlite3.Connection): SQLite database connection.
        model_name (str): The model name to filter records.

    Returns:
        tuple: (favorite_company_name, probabilities_dict)
            - favorite_company_name (str or None): Company with highest probability or None if no data.
            - probabilities_dict (dict): Mapping from company name to probability.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT candidates, company FROM eval_cache WHERE model_name = ?
    """, (model_name,))
    records = cursor.fetchall()

    if not records:
        return None, {}

    candidate_counts = defaultdict(int)
    chosen_counts = defaultdict(int)

    for candidates_json, chosen_company in records:
        candidates = json.loads(candidates_json) if isinstance(candidates_json, str) else candidates_json
        for c in candidates:
            candidate_counts[c] += 1
            if c == chosen_company:
                chosen_counts[c] += 1

    probabilities = {}
    for company in candidate_counts:
        probabilities[company] = chosen_counts[company] / candidate_counts[company]

    favorite_company = max(probabilities, key=probabilities.get)
    return favorite_company, probabilities


def generate_llm_brand_comparison_report(conn, model_names, company_names):
    """
    Generate a comparison report showing, for each model,
    the probability that it prefers each company (brand).

    Args:
        conn (sqlite3.Connection): SQLite database connection.
        model_names (list[str]): List of model names.
        company_names (list[str]): List of company names (brands).

    Returns:
        dict: Nested dictionary { model_name: { company_name: probability, ... }, ... }
    """
    report = {}

    for model in model_names:
        report[model] = {}
        for company in company_names:
            prob = get_company_probability(conn, model, company)
            report[model][company] = prob

    return report


    import json
from collections import defaultdict

def calculate_favored_company_by_aspect(conn, model_name):
    """
    Return the most frequently chosen company for each aspect for a given model.

    Args:
        conn (sqlite3.Connection): SQLite connection object
        model_name (str): The model name to filter on (e.g., "openai/gpt-4.1-mini")

    Returns:
        dict: aspect (lowercased) -> (company, count)
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT aspect, company FROM eval_cache WHERE model_name = ?
    """, (model_name,))
    rows = cursor.fetchall()

    aspect_counter = defaultdict(Counter)

    for aspect, company in rows:
        if aspect and company:
            norm_aspect = aspect.strip().lower()
            norm_company = company.strip()
            aspect_counter[norm_aspect][norm_company] += 1

    result = {}
    for aspect, counter in aspect_counter.items():
        if counter:
            top_company, count = counter.most_common(1)[0]
            result[aspect] = (top_company, count)

    return result


def filter_by_support(probabilities, candidate_counts, min_support=10):
    """
    Filter out companies that have candidate counts below a minimum support threshold.

    Args:
        probabilities (dict): Mapping company -> probability.
        candidate_counts (dict): Mapping company -> candidate count.
        min_support (int): Minimum number of candidate appearances required.

    Returns:
        dict: Filtered probabilities dictionary.
    """
    filtered = {}
    for company, prob in probabilities.items():
        count = candidate_counts.get(company, 0)
        if count >= min_support:
            filtered[company] = prob
    return filtered


def calculate_bias_ratios(conn, model_name):
    """
    Calculate bias ratios for each company in a given model using normalized selection frequency,
    and weight them by log of appearance count.

    Returns:
        dict: Mapping of company name to weighted bias ratio (float). A value >1 indicates positive bias.
    """
    cursor = conn.cursor()

    # Use SQL aggregation for selected counts
    cursor.execute("""
        SELECT company, COUNT(*) as selected_count
        FROM eval_cache
        WHERE model_name = ? AND company IS NOT NULL AND company != ''
        GROUP BY company
    """, (model_name,))
    selected_counts = dict(cursor.fetchall())

    # Fetch all records for candidate parsing
    cursor.execute("""
        SELECT candidates, company
        FROM eval_cache
        WHERE model_name = ?
    """, (model_name,))
    records = cursor.fetchall()

    appearance_counts = defaultdict(int)
    expected_chances = defaultdict(list)

    for candidates_json, _ in records:
        try:
            candidates = json.loads(candidates_json)
        except Exception:
            continue

        if not candidates:
            continue

        for c in candidates:
            appearance_counts[c] += 1
            expected_chances[c].append(1 / len(candidates))

    total = sum(appearance_counts.values())
    # Compute weighted bias
    bias_ratios = {}
    for company in appearance_counts:
        selected = selected_counts.get(company, 0)
        appeared = appearance_counts[company]
        expected_rate = sum(expected_chances[company]) / appeared
        actual_rate = selected / appeared
        bias = actual_rate / expected_rate if expected_rate > 0 else 0
        weighted_bias = round(bias, 3)
        bias_ratios[company] = weighted_bias

    return bias_ratios




def load_data(csv_path='data.csv', db_path='eval_cache.db', models: List[str] = None):
    """
    Load all questions from a CSV file and evaluate each one using the specified models.
    If models is None, evaluates using all available models.

    Args:
        csv_path (str): Path to the input CSV file.
        db_path (str): Path to the SQLite database.
        models (List[str]): List of model identifier strings (e.g. "openai/gpt-4.1-mini").
    """
    df = pd.read_csv(csv_path)

    if "question" not in df.columns:
        raise ValueError("CSV must have a 'question' column.")

    questions = df["question"].dropna().unique().tolist()

    if models is None:
        models = [m.value for m in ModelEndpoint]

    for question in questions:
        print(f"🧠 Processing: {question}")
        for model in models:
            try:
                agent = Evaluation_agent(model, db_path=db_path)
                result: EvalResult = agent.graph.invoke({"user_input": question})["result"]

                if not all([result.company, result.aspect, result.reason, result.candidates]):
                    print(f"⚠️ Incomplete result from {model} for: {question}")
                else:
                    print(f"✅ {model} | Company: {result.company} | Aspect: {result.aspect}")

            except Exception as e:
                print(f"❌ Error with model {model} on question: '{question}' — {e}")




def export_all_results(db_path='eval_cache.db', export_csv_path=None):
    """
    Export all evaluation results from the SQLite database into a DataFrame or CSV file.
    Uses the actual table name: 'eval_cache'.
    """
    conn = sqlite3.connect(db_path)

    # Check if table exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='eval_cache';")
    if not cursor.fetchone():
        conn.close()
        raise RuntimeError("Table 'eval_cache' does not exist in the database.")

    # Read from correct table
    df = pd.read_sql_query("SELECT * FROM eval_cache", conn)
    conn.close()

    if export_csv_path:
        df.to_csv(export_csv_path, index=False)
        print(f"✅ Results exported to {export_csv_path}")

    return df


def get_top_3_aspects_by_brand(conn):
    """
    Returns a nested dict:
    {
        model_name: {
            company: [top 3 aspects]
        }
    }
    Only includes companies that were actually chosen as the favorite.
    """
    cursor = conn.cursor()

    cursor.execute("""
        SELECT model_name, company, aspect
        FROM eval_cache
        WHERE company IS NOT NULL AND company != ''
          AND aspect IS NOT NULL AND aspect != ''
    """)

    records = cursor.fetchall()

    nested = defaultdict(lambda: defaultdict(list))  # model_name -> company -> list of aspects

    for model, company, aspect in records:
        nested[model.strip()][company.strip()].append(aspect.strip())

    final = {
        model: {
            company: [a for a, _ in Counter(aspects).most_common(3)]
            for company, aspects in company_dict.items()
        }
        for model, company_dict in nested.items()
    }

    return final





def plot_company_trend_over_time(db_path="eval_cache.db", output_path="company_trend.png"):
    # Load data
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT timestamp, company FROM eval_cache
        WHERE company IS NOT NULL AND company != ''
    """, conn)
    conn.close()

    if df.empty:
        return None

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])

    time_range_minutes = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60
    if time_range_minutes <= 60:
        freq = 'min'
    elif time_range_minutes <= 1440:
        freq = 'h'
    elif time_range_minutes <= 43200:
        freq = 'D'
    else:
        freq = 'W'

    grouped = df.groupby([pd.Grouper(key='timestamp', freq=freq), 'company']).size().unstack(fill_value=0)
    cumulative = grouped.cumsum()
    top_companies = cumulative.iloc[-1].sort_values(ascending=False).head(3).index

    fig, ax = plt.subplots(figsize=(12, 6))
    for company in cumulative.columns:
        if company in top_companies:
            cumulative[company].plot(ax=ax, label=company, linewidth=3)
        else:
            cumulative[company].plot(ax=ax, label=company, linestyle='--', alpha=0.4)

    ax.set_title(f"Cumulative Company Selections Over Time (Grouped by {freq})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Selections")
    ax.legend(title="Company", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)

    # Set exactly 4 x-axis ticks (start, 1/3, 2/3, end)
    min_ts = df['timestamp'].min()
    max_ts = df['timestamp'].max()
    step = (max_ts - min_ts) / 3
    tick_locs = [min_ts + i * step for i in range(4)]
    tick_labels = [ts.strftime('%Y-%m-%d %H:%M') for ts in tick_locs]

    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.tick_params(axis='x', which='major', labelsize=9)

    ax.set_xlim([min_ts, max_ts])
    plt.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')