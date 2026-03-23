import time
from flask import Flask, request, jsonify, send_from_directory
from model import predict_interest_rate, _initialise

app = Flask(__name__, static_folder=".", template_folder=".")

print("⏳  Initialising ANN model…")
_initialise("loan_data.csv")
print("✅  Model ready.")

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON payload."}), 400

    for required_field in ["Principal", "Income", "credit_score"]:
        if required_field not in data:
            return jsonify({"error": f"Missing required field: {required_field}"}), 400

    try:
        t0     = time.perf_counter()
        result = predict_interest_rate(data)
        t1     = time.perf_counter()
        result["proc_ms"] = round((t1 - t0) * 1000)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
