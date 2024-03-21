from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)

# Load the pre-trained model
model = load('stroke.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Extract features from the received JSON data
    features = [data['features']]
    prediction = model.predict(features)
    output = prediction[0]
    return jsonify({'stroke_prediction': int(output)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
