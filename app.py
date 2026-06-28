from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# 1. Load your trained machine learning model
with open('model.pkl', 'rb') as model_file:
    model = pickle.pickle.load(model_file) if hasattr(pickle, 'pickle') else pickle.load(model_file)

@app.route('/')
def home():
    # This displays your main input form page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 2. Get data entered by the user in the web form
        gender = int(request.form['Gender'])
        married = int(request.form['Married'])
        dependents = int(request.form['Dependents'])
        education = int(request.form['Education'])
        self_employed = int(request.form['Self_Employed'])
        applicant_income = float(request.form['Applicant_Income'])
        coapplicant_income = float(request.form['Coapplicant_Income'])
        loan_amount = float(request.form['Loan_Amount'])
        loan_amount_term = float(request.form['Loan_Amount_Term'])
        credit_history = float(request.form['Credit_History'])
        property_area = int(request.form['Property_Area'])
        
        # 3. Combine inputs into an array for the model
        features = np.array([[gender, married, dependents, education, self_employed, 
                              applicant_income, coapplicant_income, loan_amount, 
                              loan_amount_term, credit_history, property_area]])
        
        # 4. Make the loan prediction (0 = Rejected, 1 = Approved)
        prediction = model.predict(features)[0]
        
        if prediction == 1:
            result_text = "Congratulations! You are ELIGIBLE for the loan."
        else:
            result_text = "Sorry, you are NOT ELIGIBLE for the loan at this time."
            
        return render_template('index.html', prediction_text=result_text)

if __name__ == "__main__":
    app.run(debug=True)
