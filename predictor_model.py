import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
from sklearn import metrics
#model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))
class diabetes_prediction:
    def __init__(self):
        self.dataset = pd.read_csv('F:\python\python_resume_project\DIABETES_PREDICTION_DEPLOYMENT\diabetes.csv')
        self.split_data()

    def split_data(self):
        feature_cols = ['Pregnancies','Glucose','Insulin','BMI','DiabetesPedigreeFunction','Age']
        x = self.dataset[feature_cols] # Features
        y = self.dataset['Outcome'] # Target variable
        self.create_model(x,y)
        
    def create_model(self,x,y):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=767)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        filename = 'diabetes_prediction_model.pkl'
        pickle.dump(model, open(filename, 'wb'))

    # def predict_output(self,user_input):
    #     predictions = self.model.predict([user_input])
    #     print("Accuracy:",(metrics.accuracy_score([[0]], predictions)*100))
    #     return predictions[0]

# if __name__=="__main__":
#     app.run(debug=True)

obj = diabetes_prediction()
