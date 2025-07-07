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
    You are summarizing how different LLMs perceive and compare insurance brands based on user questions and extracted evaluation data.

    Instructions:
    - Use the **bias ratios** to determine which brands each model is inclined to favor. Bias ratio is the normalized selection rate of a brand relative to its expected appearance frequency.
    - Refer to the **most picked brand per model** to reinforce or challenge the bias findings based on raw selection frequency.
    - Use the **top aspects asked** and the **top 3 aspects per brand** to interpret *why* certain brands may be favored (e.g., a brand often selected in high-priority aspects like price or customer service).
    - If a brand shows high bias but is rarely selected, or is selected mostly on less important aspects, mention this as a limiting factor.
    - Avoid speculation or general praise; rely only on the statistics provided.

    Data:
    Top aspects asked: {json.dumps(top_aspects, indent=2)}
    Most selected company per model: {json.dumps(best_by_model, indent=2)}
    Bias ratios by model: {json.dumps(bias_ratios_by_model, indent=2)}
    Top 3 aspects per brand: {json.dumps(top3_aspects_by_brand, indent=2)}

    Write a short comparative summary (maximum 6–7 sentences) highlighting brand favorability and differences across LLMs.
    """.strip()

    return generate(model, prompt)

