import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Path to trained SVR model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Exam Score Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4F46E5;
            --primary-hover: #4338CA;
            --bg: #F3F4F6;
            --card-bg: #FFFFFF;
            --text: #1F2937;
            --text-light: #6B7280;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg);
            color: var(--text);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 2rem 1rem;
        }

        .container {
            background: var(--card-bg);
            max-width: 750px;
            width: 100%;
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 1.875rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-light);
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-group label {
            font-size: 0.875rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
            color: var(--text);
        }

        .form-group input, .form-group select {
            padding: 0.65rem 0.85rem;
            border: 1px solid #D1D5DB;
            border-radius: 8px;
            font-size: 0.95rem;
            outline: none;
            background-color: #FAFAFA;
        }

        .form-group input:focus, .form-group select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2);
            background-color: #FFFFFF;
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            padding: 0.85rem;
            background-color: var(--primary);
            color: white;
            font-size: 1rem;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.3);
        }

        .btn-submit:hover {
            background-color: var(--primary-hover);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.25rem;
            border-radius: 12px;
            background: #EEF2FF;
            border: 1px solid #C7D2FE;
            text-align: center;
        }

        .result-box h2 {
            font-size: 1.1rem;
            color: var(--primary-hover);
            margin-bottom: 0.25rem;
        }

        .result-box .score {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
        }

        .error-box {
            margin-top: 1.5rem;
            padding: 1rem;
            background-color: #FEE2E2;
            border: 1px solid #FCA5A5;
            color: #991B1B;
            border-radius: 8px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Student Performance Predictor</h1>
            <p>Input student metrics to estimate expected score using the SVR model.</p>
        </div>

        <form action="/" method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label for="age">Age</label>
                    <input type="number" id="age" name="age" step="any" required placeholder="e.g. 20">
                </div>
                <div class="form-group">
                    <label for="gender">Gender</label>
                    <select id="gender" name="gender" required>
                        <option value="0">Female</option>
                        <option value="1">Male</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="course">Course Code/ID</label>
                    <input type="number" id="course" name="course" step="any" required placeholder="e.g. 1">
                </div>
                <div class="form-group">
                    <label for="study_hours">Study Hours / Week</label>
                    <input type="number" id="study_hours" name="study_hours" step="any" required placeholder="e.g. 15">
                </div>
                <div class="form-group">
                    <label for="class_attendance">Class Attendance (%)</label>
                    <input type="number" id="class_attendance" name="class_attendance" step="any" required placeholder="e.g. 85">
                </div>
                <div class="form-group">
                    <label for="internet_access">Internet Access</label>
                    <select id="internet_access" name="internet_access" required>
                        <option value="1">Yes</option>
                        <option value="0">No</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="sleep_hours">Sleep Hours / Night</label>
                    <input type="number" id="sleep_hours" name="sleep_hours" step="any" required placeholder="e.g. 7">
                </div>
                <div class="form-group">
                    <label for="sleep_quality">Sleep Quality (1-5)</label>
                    <input type="number" id="sleep_quality" name="sleep_quality" step="any" required placeholder="e.g. 4">
                </div>
                <div class="form-group">
                    <label for="study_method">Study Method ID</label>
                    <input type="number" id="study_method" name="study_method" step="any" required placeholder="e.g. 2">
                </div>
                <div class="form-group">
                    <label for="facility_rating">Facility Rating (1-5)</label>
                    <input type="number" id="facility_rating" name="facility_rating" step="any" required placeholder="e.g. 3">
                </div>
                <div class="form-group">
                    <label for="exam_difficulty">Exam Difficulty (1-5)</label>
                    <input type="number" id="exam_difficulty" name="exam_difficulty" step="any" required placeholder="e.g. 3">
                </div>

                <button type="submit" class="btn-submit">Predict Performance Score</button>
            </div>
        </form>

        {% if prediction %}
        <div class="result-box">
            <h2>Predicted Score</h2>
            <div class="score">{{ prediction }}</div>
        </div>
        {% endif %}

        {% if error %}
        <div class="error-box">
            {{ error }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        if model is None:
            error = "Model file ('model.pkl') not found. Please ensure it is uploaded in the root directory."
        else:
            try:
                features = [
                    float(request.form.get("age", 0)),
                    float(request.form.get("gender", 0)),
                    float(request.form.get("course", 0)),
                    float(request.form.get("study_hours", 0)),
                    float(request.form.get("class_attendance", 0)),
                    float(request.form.get("internet_access", 0)),
                    float(request.form.get("sleep_hours", 0)),
                    float(request.form.get("sleep_quality", 0)),
                    float(request.form.get("study_method", 0)),
                    float(request.form.get("facility_rating", 0)),
                    float(request.form.get("exam_difficulty", 0)),
                ]
                
                input_array = np.array([features])
                pred_val = model.predict(input_array)[0]
                prediction = f"{pred_val:.2f}"
            except Exception as e:
                error = f"Error processing prediction: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction, error=error)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
