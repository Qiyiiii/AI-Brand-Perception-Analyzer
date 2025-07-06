# AI-Brand-Perception-Analyzer

AI-Brand-Perception-Analyzer is a Python-based tool designed to analyze and compare brand perception, specifically for insurance companies, using advanced language models. It leverages multiple LLM endpoints to extract, evaluate, and cache structured insights from user queries about insurance providers.

## Features
- **Multi-Model Support:** Easily switch between various LLM endpoints (OpenAI, Anthropic, Google, X.ai, Meta, DeepSeek, Mistral) via a unified interface.
- **Structured Extraction:** Automatically extracts companies, comparison aspects, and provides a reasoned evaluation in JSON format.
- **Caching:** Results are cached in a local SQLite database to avoid redundant LLM calls and speed up repeated queries.
- **Extensible:** Modular design allows for easy extension to other domains or evaluation criteria.

## How It Works
1. **User Input:** The user provides a question comparing insurance companies (e.g., "Which travel insurance is better, Sunlife or Manulife?").
2. **Prompt Generation:** The system generates a detailed prompt instructing the LLM to extract companies, aspect, best company, and reasoning.
3. **LLM Evaluation:** The selected model endpoint processes the prompt and returns a structured JSON response.
4. **Caching:** The result is stored in a local SQLite database for future reuse.

## Example
**Input:**
```
Which home insurance provider offers better coverage, Aviva or Intact?
```
**Output:**
```
{
  "company": "Intact",
  "aspect": "home insurance coverage",
  "reason": "Intact provides more extensive home insurance policies with higher limits and better natural disaster coverage compared to Aviva.",
  "candidates": ["Aviva", "Intact"]
}
```

## Setup
1. **Clone the repository:**
   ```sh
   git clone https://github.com/Qiyiiii/AI-Brand-Perception-Analyzer.git
   cd AI-Brand-Perception-Analyzer
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Configure API Key:**
   - Set your OpenRouter API key in `utils.py` (`OPENROUTER_API_KEY`).

## Usage
Run the main agent script:
```sh
python agent.py
```

You can modify the user input and model endpoint in the `__main__` section of `agent.py`.

## File Structure
- `agent.py` — Main logic for insurance company comparison.
- `utils.py` — LLM API integration and model endpoint definitions.
- `instance.py` — Data models and type definitions.
- `info_getter.py` — Functions used to get and handle information from the databse.
- `requirements.txt` — Python dependencies.

## License
This project is licensed under the Apache-2.0 license.