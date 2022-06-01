from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

with open("PCA_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("pca.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        input_data = [float(x) for x in request.form.values()]
        features_value = [np.array(input_data)]
        features_name = ["Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
        df = pd.DataFrame(features_value, columns=features_name)
        output = model.predict(df)[0][0]
        return render_template("result.html", value=output)
    else:
        return render_template("result.html", value="  invalid request  ")

if __name__ == "__main__":
    app.run()
