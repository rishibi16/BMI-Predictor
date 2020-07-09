from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('bmi.pkl')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == "POST":
        myDict = request.form
        gen = int(myDict['gender'])
        height = int(myDict['height'])
        weight = int(myDict['weight'])
        input_feature = [gen, height, weight]
        sc = model.predict([input_feature])[0]
        score = round(sc)
        if(score == 0):
            final = 'Extremely Weak'
        elif(score == 1):
            final = 'Weak'
        elif(score == 2):
            final = 'Normal'
        elif(score == 3):
            final = 'Overweight'
        elif(score == 4):
            final = 'Obesity'
        elif(score == 5):
            final = 'Extreme Obesity'
        else:
            final = 'Can\'t predict'
        #return render_template('show.html', prediction = 'Person is in {}'.format(final))
    return render_template('index.html', prediction = 'Person is in {} Condition'.format(final))

if __name__ == "__main__":
    app.run(debug=True)
    