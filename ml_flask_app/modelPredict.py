# Load the model and scaler in Flask app
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pickle import load

app = Flask(__name__)

# Load the model from a file
with open('lgb_model.pkl', 'rb') as file:
    lgb_model = load(file)

# The browser was sending an OPTIONS method before POST method
# This is to handle the OPTIONS method
@app.before_request
def option_autoreply():
    if request.method == 'OPTIONS':
        resp = app.make_default_options_response()
        headers = None
        if 'ACCESS_CONTROL_REQUEST_HEADERS' in request.headers:
            headers = request.headers['ACCESS_CONTROL_REQUEST_HEADERS']
        h = resp.headers
        h['Access-Control-Allow-Origin'] = request.headers['Origin']
        h['Access-Control-Allow-Methods'] = "POST"
        h['Access-Control-Allow-Headers'] = headers
        h['Access-Control-Allow-Credentials'] = 'true'
        h['Access-Control-Max-Age'] = '90000'
        return resp

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)
    
    print(data)
        
    # Convert the data to a pandas dataframe
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.transpose()
    
        # Bin the age and create new columns
    df['is_infant'] = 0
    df['is_toddler'] = 0
    df['is_child'] = 0
    df['is_teenager'] = 0
    df['is_young_adult'] = 0
    df['is_middle_aged'] = 0
    df['is_pre_senior'] = 0
    df['is_senior'] = 0
    age = int(df['AGE'][0])
    
    print(age)
    
    if age < 2:
        df['is_infant'] = 1
    elif age >= 3 and age < 4:
        df['is_toddler'] = 1
    elif age >= 4 and age < 13:
        df['is_child'] = 1
    elif age >= 13 and age < 20:
        df['is_teenager'] = 1
    elif age >= 20  and age < 30:
        df['is_young_adult'] = 1
    elif age >= 30 and age < 49:
        df['is_middle_aged'] = 1
    elif age >= 49 and age < 60:
        df['is_pre_senior'] = 1
    elif age >= 60:
        df['is_senior'] = 1
    
    print("df after", df)
    # Reshaping
    df = df.values.reshape(1,-1)

    print(df)
    
    # Make prediction using the loaded model and scaled data
    predictions = lgb_model.predict(df)

    print(predictions[0])

    output = int(predictions[0])
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
