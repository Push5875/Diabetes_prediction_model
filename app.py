from flask import Flask,jsonify,request,render_template
import sqlite3
import pickle
# from diabetes_prediction import *

filename = 'diabetes_prediction_model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home_page():
    return render_template('index.html')
    
@app.route("/predict", methods=['POST'])	
def predict():
    if request.method == 'POST':
        Pregnancies = int(request.form['Pregnancies'])
        Glucose=int(request.form['Glucose'])
        Insulin=int(request.form['Insulin'])
        if Insulin=='Yes':
            Insulin = 1
        else:
            Insulin = 0
        BMI=float(request.form['BMI'])
        DiabetesPedigreeFunction=float(request.form['DiabetesPedigreeFunction'])
        Age=int(request.form['Age'])
        prediction=model.predict([[Pregnancies,Glucose,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        output = prediction[0]
        print(output)
        if output==0:
            return render_template('index.html',prediction_text="You are safe")
        else:
            return render_template('index.html',prediction_text="You have Diabetes")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
