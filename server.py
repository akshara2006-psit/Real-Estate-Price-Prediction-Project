
from flask import Flask, request, jsonify, render_template
import pickle
import json
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("banglore_home_prices_model.pickle", "rb"))

with open("columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]

@app.route("/")
def home():
    return render_template("index.html", locations=data_columns[3:])


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        sqft = float(data["sqft"])
        bath = int(data["bath"])
        bhk = int(data["bhk"])
        location = data["location"]

        # Input validation
        if sqft < 300 or bath < 1 or bhk < 1:
            return jsonify({"error": "Invalid input values"}), 400

        x = np.zeros(len(data_columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk

        if location in data_columns:
            loc_index = data_columns.index(location)
            x[loc_index] = 1

        prediction = model.predict([x])[0]

        # Prevent negative output
        prediction = max(prediction, 0)

        return jsonify({
            "estimated_price": round(prediction, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)