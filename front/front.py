import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import datetime as dt
from sklearn.linear_model import  Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, f1_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load dataset
df = pd.read_csv("dataset/data.csv")

# Preprocess the dataset
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

df['gender_Male'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
df['gender_Female'] = df['gender'].apply(lambda x: 1 if x == 'Female' else 0)
df['ever_married'] = df['ever_married'].apply(lambda x: 1 if x == 'Yes' else 0)

df['work_type_Private'] = df['work_type'].apply(lambda x: 1 if x == 'Private' else 0)
df['work_type_Self_employed'] = df['work_type'].apply(lambda x: 1 if x == 'Self-employed' else 0)
df['work_type_Govt_job'] = df['work_type'].apply(lambda x: 1 if x == 'Govt_job' else 0)
df['work_type_children'] = df['work_type'].apply(lambda x: 1 if x == 'children' else 0)
df['work_type_Never_worked'] = df['work_type'].apply(lambda x: 1 if x == 'Never_worked' else 0)

df['Residence_type'] = df['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)

df['smoking_status_formerly_smoked'] = df['smoking_status'].apply(lambda x: 1 if x == 'formerly smoked' else 0)
df['smoking_status_never_smoked'] = df['smoking_status'].apply(lambda x: 1 if x == 'never smoked' else 0)
df['smoking_status_smokes'] = df['smoking_status'].apply(lambda x: 1 if x == 'smokes' else 0)
df['smoking_status_Unknown'] = df['smoking_status'].apply(lambda x: 1 if x == 'Unknown' else 0)

df_model = df.copy()
df_model.drop(['Residence_type', 'work_type', 'smoking_status', 'gender', 'ever_married'], axis=1, inplace=True)

X = df_model.drop('stroke', axis=1)
y = df_model['stroke']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return np.array(class_pred)

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred) / len(y)
        return accuracy


linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_reg_predictions = linear_reg.predict(X_test)
linear_reg_rmse = np.sqrt(mean_squared_error(y_test, linear_reg_predictions))
linear_reg_acc = linear_reg.score(X_test, y_test)

lasso_reg = Lasso()
lasso_reg.fit(X_train, y_train)
lasso_reg_predictions = lasso_reg.predict(X_test)
lasso_reg_rmse = np.sqrt(mean_squared_error(y_test, lasso_reg_predictions))
lasso_reg_acc = lasso_reg.score(X_test, y_test)

Rigid_reg = Ridge()
Rigid_reg.fit(X_train, y_train)
Rigid_reg_predictions = Rigid_reg.predict(X_test)
Rigid_reg_rmse = np.sqrt(mean_squared_error(y_test, Rigid_reg_predictions))
Rigid_reg_acce = Rigid_reg.score(X_test, y_test)

logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, y_train)
Logistic_pred = logistic_reg.predict(X_test)
Log_reg_rmse = np.sqrt(mean_squared_error(y_test, Logistic_pred))
logistic_reg_acc = logistic_reg.score(X_test, y_test)


results = pd.DataFrame({
    'Model': ['Linear Regression', 'Lasso Regression', 'Ridge Regression', 'Logistic Regression'],
    'Accuracy': [linear_reg_acc, lasso_reg_acc, Rigid_reg_acce, logistic_reg_acc],
    'RMSE': [linear_reg_rmse, lasso_reg_rmse, Rigid_reg_rmse, Log_reg_rmse]
})


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.barplot(x='Model', y='Accuracy', data=results, palette='Blues_d', ax=axes[0])
axes[0].set_title('Model Accuracy')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

sns.barplot(x='Model', y='RMSE', data=results, palette='Reds_d', ax=axes[1])
axes[1].set_title('Model RMSE')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)

plt.tight_layout()
st.pyplot(fig)


Confusion_matrix = confusion_matrix(y_test, Logistic_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=Confusion_matrix, display_labels=[0, 1])

fig, ax = plt.subplots(figsize=(6, 6))
cm_display.plot(ax=ax)
st.pyplot(fig)


precision = precision_score(y_test, Logistic_pred)
f1 = f1_score(y_test, Logistic_pred)
recall = recall_score(y_test, Logistic_pred)

st.write(f"Precision Score: {precision:.2f}")
st.write(f"F1 Score: {f1:.2f}")
st.write(f"Recall Score: {recall:.2f}")


st.title("Stroke Risk Prediction")

st.sidebar.header("Enter Your Details")


age = st.sidebar.slider('Age', 18, 100, 30)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
hypertension = st.sidebar.selectbox('Hypertension', ['Yes', 'No'])
heart_disease = st.sidebar.selectbox('Heart Disease', ['Yes', 'No'])
ever_married = st.sidebar.selectbox('Ever Married', ['Yes', 'No'])
work_type = st.sidebar.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
residence_type = st.sidebar.selectbox('Residence Type', ['Urban', 'Rural'])
avg_glucose_level = st.sidebar.number_input('Average Glucose Level', 50, 300, 100)
bmi = st.sidebar.number_input('BMI', 10, 50, 20)
smoking_status = st.sidebar.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

predict_button = st.sidebar.button('Predict Stroke Risk')

user_data = {
    'age': age,
    'gender_Male': 1 if gender == 'Male' else 0,
    'gender_Female': 1 if gender == 'Female' else 0,
    'hypertension': 1 if hypertension == 'Yes' else 0,
    'heart_disease': 1 if heart_disease == 'Yes' else 0,
    'ever_married': 1 if ever_married == 'Yes' else 0,
    'work_type_Private': 1 if work_type == 'Private' else 0,
    'work_type_Self_employed': 1 if work_type == 'Self-employed' else 0,
    'work_type_Govt_job': 1 if work_type == 'Govt_job' else 0,
    'work_type_children': 1 if work_type == 'children' else 0,
    'work_type_Never_worked': 1 if work_type == 'Never_worked' else 0,
    'Residence_type': 1 if residence_type == 'Urban' else 0,
    'smoking_status_formerly_smoked': 1 if smoking_status == 'formerly smoked' else 0,
    'smoking_status_never_smoked': 1 if smoking_status == 'never smoked' else 0,
    'smoking_status_smokes': 1 if smoking_status == 'smokes' else 0,
    'smoking_status_Unknown': 1 if smoking_status == 'Unknown' else 0,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi
}


user_input_df = pd.DataFrame([user_data], columns=X.columns)


user_input_df = user_input_df.fillna(X.median())


if predict_button:
    stroke_prediction = logistic_reg.predict(user_input_df)


    if stroke_prediction == 1:
        st.subheader("You are at risk of a stroke.")
    else:
        st.subheader("You are not at risk of a stroke.")
