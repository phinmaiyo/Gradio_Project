import gradio as gr
import pandas as pd 
import numpy as np
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier


# key lists
expected_inputs = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges']
numerics = ['tenure', 'MonthlyCharges', 'TotalCharges']
categoricals = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract', 'PaperlessBilling', 'PaymentMethod']

# Define helper functions
# Function to load the toolkit
def load_toolkit(file_path = r"src\Gradio_toolkit"):
    with open(file_path, "rb") as file:
        toolkit = pickle.load(file)
    return toolkit

# Load_toolkit
loaded_toolkit = load_toolkit(r"src\Gradio_toolkit")

# Import the model
model= RandomForestClassifier()
model=joblib.load(r"src\customer_churn_model.pkl")

# Instantiate the items in the toolkit
encoder= loaded_toolkit["encoder"]
scaler= loaded_toolkit["scaler"]

# Function to process inputs and return prediction


def predict_customer_attrition(model=model, scaler=scaler, encoder=encoder):
    # Scale the numeric columns (you should use the scaler you have)
    scaled_num = pd.DataFrame(scaler.transform(numerics), columns=numerics.columns, index=numerics.index)

    # Encode the categorical columns 
    encoded_cat = pd.DataFrame(encoder.transform(categoricals).toarray(), index=categoricals.index, columns=encoder.get_feature_names_out(categoricals.columns))

    # Combine the scaled numeric and encoded categorical features for input data
    input_data = pd.concat([scaled_num, encoded_cat], axis=1)

    # Make the prediction 
    model_output = model.predict(input_data)

    # Format the prediction result
    return{"Prediction: Customer is likely to LEAVE": float(model_output[0]),
           "Prediction: Customer is likely to STAY": 1 - float(model_output[0])}


# Set up interface
# Inputs
gender= gr.Dropdown(label = "What is the gender of the customer?", choices = ["Female", "Male"], value= "Male")
SeniorCitizen= gr.Dropdown( label="Is the customer a senior citizen?", choices= ["No","Yes"], value="No")
Partner= gr.Radio(label= "Does the customer have a partner?", choices= ["No", "Yes"], value="No")
Dependents= gr.Radio(label= "Does the customer have dependents?", choices= ["No", "Yes"], value="No")
tenure= gr.Number(label= "How many months has the customer stayed with the company?", minimum= 1, maximum= 72, interactive= True, value= 1, step =1)
PhoneService= gr.Dropdown(label=" Does the customer has a phone service?", choices=["No", "Yes"], value= "Yes")
MultipleLines= gr.Radio(label="Does the customer has multiple lines?", choices=["No", "Yes", "No phone service"], value="No")
InternetService= gr.Dropdown(label="What is the customer's internet service provider?", choices=["Fiber optic", "DSL", "No Internet Service"], value="Fiber optic")
OnlineSecurity= gr.Radio(label="Does the customer has online security?", choices=["No", "Yes", "No internet service"], value="No")
OnlineBackup= gr.Radio(label="Does the customer has online backup?", choices=["No", "Yes", "No internet service"], value="No")
DeviceProtection= gr.Dropdown(label="Does the customer has device protection?", choices=["No", "Yes", "No internet service"], value="No")
TechSupport= gr.Radio(label="Does the customer have tech support?", choices=["No", "Yes", "No internet service"], value="No")
StreamingTV= gr.Dropdown(label="Does the customer stream TV?", choices=["No", "Yes", "No internet service"], value="No")
StreamingMovies= gr.Dropdown(label="Does the customer stream movies?", choices=["No", "Yes", "No internet service"], value="No")
Contract= gr.Radio(label="What is the contract term of the customer?", choices=["Month-to-month", "Two year", "One year"], value="Month-to-month")
PaperlessBilling= gr.Dropdown(label="Does the customer has paperless billing?", choices=["No", "Yes"], value="Yes")
PaymentMethod= gr.Dropdown(label="What is the customer's payment method?", choices=["Electronic check", "Mailed check", "Credit card (automatic)","Bank transfer (automatic)"], value="Electronic check")
MonthlyCharges= gr.Slider(label= "What is the monthly amount charged to the customer?", minimum= 15, maximum= 150, value= 20, interactive= True)    
TotalCharges= gr.Slider(label= "What is the total amount charged to the customer?", minimum= 15, maximum= 9000, value= 220, interactive= True)



# outputs
iface = gr.Interface(
    inputs= ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure','PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity','OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV','StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'],
    fn=predict_customer_attrition, 
    outputs=gr.Label("Awaiting Submission..."),
    title="Customer Attrition Prediction App",
    description="This app was created by Santorini during our LP4 EDS",
    live=True
)

iface.launch(inbrowser=True, show_error=True)



