from flask import Flask, request, render_template
import requests

def get_prediction(data):
    API_KEY = "0iiOUE_frC77F7i8dmSNkR8W0XE86s85eIsThHAdETCS"
    token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
    API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
    mltoken = token_response.json()["access_token"]

    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

    payload_scoring = {"input_data": [{"field": ["Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"], "values": [data]}]}

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/a10759eb-2aa7-4316-8fbc-0ef91a500e56/predictions?version=2022-05-27', json=payload_scoring,
    headers={'Authorization': 'Bearer ' + mltoken})
    prediction = response_scoring.json()
    return prediction["predictions"][0]["values"][0][0][0]


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("pca.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        input_data = [float(x) for x in request.form.values()]
        output = get_prediction(input_data)
        return render_template("result.html", value=output)
    else:
        return render_template("result.html", value="  invalid request  ")



if __name__ == "__main__":
    app.run(debug=False)
