# import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from flask import make_response

app = Flask(__name__)
model = pickle.load(open('venv/model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['get'])
def predict():
    # print(int(request.args['age']))
    # int_features = [float(x) for x in request.form.values()]
    int_features=request.args
    final_features = pd.DataFrame(
        {'age': [float(int_features['age'])/80], 'bp': [float(int_features['bp'])/10], 'al':[ float(int_features['al'])/5], 'pcc': [float(int_features['pcc'])],
         'bgr': [float(int_features['bgr'])/146], 'bu': [float(int_features['bu'])/418], 'sod': [float(int_features['sod'])/164], 'pot': [float(int_features['pot'])/130.1],
         'hemo': [float(int_features['hemo'])/21.7], 'rbcc': [float(int_features['rbcc'])/12.8], 'dm': [float(int_features['dm'])], 'appet': [float(int_features['appet'])]})
    prediction = model.predict(final_features)
    print(prediction[0][0])
    pred=str(prediction[0][0]*0.96)

    # return render_template('index.html', prediction_text='you have this disease ${}'.format(pred))
    return make_response(pred)


if __name__ == "__main__":
    app.run(debug=True)
