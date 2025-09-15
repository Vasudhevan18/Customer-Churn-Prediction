from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# Load the saved model
model_filename = 'rf_churn_model_improved.pkl'
loaded_model = joblib.load(model_filename)
print(f"Model loaded from '{model_filename}'")

# Function to preprocess input data (matches training preprocessing)
def preprocess_input(df_input):
    le = LabelEncoder()
    categorical_cols = ['Contract_Type', 'Payment_Method', 'Tech_Support', 'Internet_Service']
    
    df_processed = df_input.copy()
    
    for col in categorical_cols:
        le.fit(['Month-to-month', 'One year', 'Two year'] if col == 'Contract_Type' else 
               ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'] if col == 'Payment_Method' else 
               ['Yes', 'No'] if col == 'Tech_Support' else 
               ['DSL', 'Fiber optic', 'No'])
        df_processed[col] = le.transform(df_processed[col])
    
    return df_processed

# Landing page
@app.route('/')
def home():
    return render_template('home.html')

# Prediction form page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        customer_data = {
            'Age': int(request.form['Age']),
            'Tenure_Months': int(request.form['Tenure_Months']),
            'Monthly_Charges': float(request.form['Monthly_Charges']),
            'Total_Charges': float(request.form['Total_Charges']),
            'Contract_Type': request.form['Contract_Type'],
            'Payment_Method': request.form['Payment_Method'],
            'Tech_Support': request.form['Tech_Support'],
            'Internet_Service': request.form['Internet_Service']
        }
        
        # Convert to DataFrame
        single_df = pd.DataFrame([customer_data])
        
        # Preprocess the data
        X_single = preprocess_input(single_df)
        
        # Make prediction
        prediction_value = loaded_model.predict(X_single)[0]
        prediction = 'Churn' if prediction_value == 1 else 'Not Churn'
        
        # Redirect to result page with the prediction
        return render_template('result.html', prediction=prediction)
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)