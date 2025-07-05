from flask import Flask, render_template, request, redirect, url_for
from instance import EvalResult
from info_getter import evaluate_question_with_models, get_most_asked_aspect, get_best_llm_by_model, calculate_favored_company_by_aspect,calculate_favored_company_by_aspect,get_top_5_aspects, calculate_bias_ratios, load_data, get_top_3_aspects_by_brand,plot_company_trend_over_time
from agent import Evaluation_agent
from utils import ModelEndpoint
import sqlite3
from flask import send_file
from info_getter import export_all_results
from reporter import generate_summary_report
from flask import request, jsonify
import os
import base64
from io import BytesIO
app = Flask(__name__)

def get_model_label(model_enum):
    return model_enum.name.replace("_", " ").title()

@app.route("/", methods=["GET", "POST"])
def index():
    models = list(ModelEndpoint)

    if request.method == "POST":
        selected_models = request.form.getlist("models")
        mode = request.form.get("mode")

        if not selected_models:
            return render_template("index.html", models=models, error="Please select at least one model.")

        try:
            selected_model_values = [ModelEndpoint[m].value for m in selected_models]
        except KeyError:
            return render_template("index.html", models=models, error="Invalid model selection.")

        # Handle batch CSV upload
        if mode == "batch":
            file = request.files.get("csv_file")
            if file and file.filename.endswith(".csv"):
                filepath = os.path.join("uploads", "data.csv")
                os.makedirs("uploads", exist_ok=True)
                file.save(filepath)

                load_data(csv_path=filepath, models=selected_model_values)

                return render_template("index.html", models=models, message="✅ CSV uploaded and processed successfully.")
            else:
                return render_template("index.html", models=models, error="⚠️ Please upload a valid CSV file.")

        # Handle single question
        question = request.form.get("question", "").strip()
        if not question:
            return render_template("index.html", models=models, error="Please enter a question and select at least one model.")

        return redirect(url_for("results", question=question, models=",".join(selected_models)))

    return render_template("index.html", models=models)


@app.route("/results", methods=["GET"])
def results():
    question = request.args.get("question", "").strip()
    model_values = request.args.get("models", "").split(",")
    selected_models = [m for m in model_values if m]

    results = []
    model_failures = []
    user_input_failures = []

    print(f"📥 Question: {question}")
    print(f"✅ Selected models: {selected_models}")

    for model in selected_models:
        try:
            # ✅ FIX: Convert string name to ModelEndpoint enum and get the .value
            model_enum = ModelEndpoint[model]
            agent = Evaluation_agent(model_enum.value, db_path='eval_cache.db')
            output = agent.graph.invoke({"user_input": question})
            print(output)

            if "result" not in output:
                print(f"⚠️ No 'result' field in response from {model}: {output}")
                model_failures.append(model)
                continue

            result = output["result"]

            if not all([
                getattr(result, "company", "").strip(),
                getattr(result, "aspect", "").strip(),
                getattr(result, "reason", "").strip(),
                getattr(result, "candidates", []),
            ]):
                print(f"⚠️ Incomplete result from {model}: {result}")
                user_input_failures.append(model)
                continue

            results.append({
                "model": get_model_label(model_enum),
                "company": result.company,
                "aspect": result.aspect,
                "reason": result.reason,
                "candidates": result.candidates
            })

        except Exception as e:
            print(f"❌ Error with model {model}: {str(e)}")
            model_failures.append(model)
            continue

    all_failed_due_to_user = len(results) == 0 and len(user_input_failures) > 0
    

    return render_template(
        "results.html",
        question=question,
        results=results,
        model_failures=model_failures,
        user_input_failures=user_input_failures,
        all_failed_due_to_user=all_failed_due_to_user,
        failed_models=user_input_failures + model_failures,
        
    )




    


@app.route("/summary", methods=["GET"])
def summary():
    conn = sqlite3.connect("eval_cache.db")
    top_aspects = get_top_5_aspects(conn)
    best_by_model = get_best_llm_by_model(conn)

    bias_ratios_by_model = {
        model.name: calculate_bias_ratios(conn, model.value)
        for model in ModelEndpoint
    }

    raw_top3_aspects = get_top_3_aspects_by_brand(conn)
    top3_aspects_by_brand = {
        model_enum.name: raw_top3_aspects.get(model_enum.value, {})
        for model_enum in ModelEndpoint
    }

    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    company_trend_img_base64 = plot_company_trend_over_time()

    return render_template(
        "summary.html",
        top_aspects=top_aspects,
        best_by_model=best_by_model,
        bias_ratios_by_model=bias_ratios_by_model,
        top3_aspects_by_brand=top3_aspects_by_brand,
        company_trend=company_trend_img_base64  
    )

@app.route("/download", methods=["GET"])
def download():
    export_path = "static/eval_results_export.csv"
    export_all_results(export_csv_path=export_path)
    return send_file(export_path, as_attachment=True)


@app.route("/generate_report", methods=["POST"])
def generate_report():
    try:
        conn = sqlite3.connect("eval_cache.db")
        top_aspects = get_top_5_aspects(conn)
        best_by_model = get_best_llm_by_model(conn)
        top3_aspects_by_brand = get_top_3_aspects_by_brand(conn)

        bias_ratios_by_model = {}
        for model in ModelEndpoint:
            model_id = model.value
            bias_ratios_by_model[model.name] = calculate_bias_ratios(conn, model_id)

        conn.close()

        report = generate_summary_report(
            top_aspects,
            best_by_model,
            bias_ratios_by_model,
            top3_aspects_by_brand
        )
        return jsonify({"report": report})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
if __name__ == "__main__":
    app.run(debug=True)