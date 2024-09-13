from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('best_pipeline.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define mappings
sex_mapping = {'female': 0, 'male': 1}
embarked_mapping = {'C': 1, 'Q': 0, 'S': 0}  # Only one value should be 1
title_mapping = {'Mr': 1, 'Mrs': 0, 'Miss': 0, 'Other': 0}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    pclass = int(request.form['Pclass'])
    age = float(request.form['Age'])
    fare = float(request.form['Fare'])
    family_size = int(request.form['FamilySize'])
    sex = request.form['Sex']
    embarked = request.form['Embarked']
    title = request.form['Title']

    # Convert categorical data to numerical
    sex = sex_mapping[sex]
    embarked_q = 1 if embarked == 'Q' else 0
    embarked_s = 1 if embarked == 'S' else 0
    title_miss = 1 if title == 'Miss' else 0
    title_mr = 1 if title == 'Mr' else 0
    title_mrs = 1 if title == 'Mrs' else 0
    title_other = 1 if title == 'Other' else 0
    title_the_countess = 0  # Assuming not used

    # Prepare data for prediction
    data = np.array([[pclass, age, fare, family_size, sex, embarked_q, embarked_s, title_miss, title_mr, title_mrs, title_other, title_the_countess]])
    print(data)
    data = pd.DataFrame(data, columns=['Pclass', 'Age', 'Fare', 'FamilySize', 'Sex_male', 'Embarked_Q', 'Embarked_S', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Other', 'Title_the Countess'])

    # Scale the data
    data[['Age', 'Fare', 'FamilySize']] = scaler.transform(data[['Age', 'Fare', 'FamilySize']])

    # Make prediction
    prediction = model.predict(data)

    # Return the result
    result = 'Survived' if prediction[0] == 1 else 'Not Survived'
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
