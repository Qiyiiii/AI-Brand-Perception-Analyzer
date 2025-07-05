from utils import ModelEndpoint
from instance import EvalResult
from utils import generate
import json


def generate_summary_report(top_aspects, best_by_model, bias_ratios_by_model, top3_aspects_by_brand, model=ModelEndpoint.GPT_4_1_MINI.value):
    """
    Generate a short comparative report using LLM evaluation data.

    Focus: Identify which brands are most favored and how LLMs differ in perception.

    Args:
        top_aspects (list): Frequently asked aspects across all questions.
        best_by_model (dict): Model name -> (most-picked brand, count).
        bias_ratios_by_model (dict): Model name -> {company: bias_ratio}.
        top3_aspects_by_brand (dict): Model name -> {company -> [top 3 aspects]}.
        model (str): LLM model used for summarization.

    Returns:
        str: A concise report.
    """
    prompt = f"""
You are analyzing outputs from different LLMs that compare insurance brands.

Your task:
- Briefly identify which brands are most favored overall.
- Compare how each LLM perceives brands (e.g., stronger bias or preference).
- Focus on key trends and differences — no fluff.

Data:
Top aspects asked: {json.dumps(top_aspects, indent=2)}
Most selected company per model: {json.dumps(best_by_model, indent=2)}
Bias ratios by model: {json.dumps(bias_ratios_by_model, indent=2)}
Top 3 aspects per brand: {json.dumps(top3_aspects_by_brand, indent=2)}

Now, write a short summary (max 6-7 sentences) comparing brand perception across LLMs.
""".strip()

    return generate(model, prompt)

