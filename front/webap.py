import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings
from sklearn.linear_model import  Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics

warnings.filterwarnings('ignore')


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

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)
        return np.array([1 if i > 0.5 else 0 for i in y_pred])

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)


st.title("Stroke Risk Prediction and Data Analysis")

@st.cache_data
def load_data():
    df = pd.read_csv("dataset/data.csv")
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    return df

df = load_data()


st.sidebar.title("Navigation")
options = st.sidebar.radio("Select Page", ["Data Visualization", "Stroke Prediction"])

if options == "Data Visualization":
    st.header("Exploratory Data Analysis and Visualizations")


    st.subheader("Age Distribution of Patients")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['age'], kde=True, ax=ax)
    ax.set_title("Age Distribution of Patients")
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.markdown(
        "**Observations:** The age distribution is roughly bimodal, with peaks around ages 40–60 and 80."
    )


    st.subheader("Average Glucose Level by Stroke Status")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='stroke', y='avg_glucose_level', data=df, ax=ax)
    ax.set_title("Average Glucose Level by Stroke Status")
    ax.set_xlabel("Stroke")
    ax.set_ylabel("Average Glucose Level")
    st.pyplot(fig)

    st.markdown(
        "**Observations:** Stroke patients generally have higher glucose levels."
    )

  
    st.subheader("Hypertension vs Stroke")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='hypertension', hue='stroke', data=df, ax=ax)
    ax.set_title("Hypertension vs Stroke")
    ax.set_xlabel("Hypertension")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.markdown(
        "**Observations:** Most hypertensive patients do not experience a stroke, but hypertension increases stroke likelihood."
    )

 
    st.subheader("BMI vs Average Glucose Level by Stroke")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='bmi', y='avg_glucose_level', hue='stroke', data=df, palette="viridis", ax=ax)
    ax.set_title("BMI vs Average Glucose Level by Stroke")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Average Glucose Level")
    st.pyplot(fig)

    st.markdown(
        "**Observations:** Elevated glucose levels correlate more strongly with stroke risk than BMI."
    )


    st.subheader("Model Performance Evaluation")

    df['gender_Male'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
    df['gender_Female'] = df['gender'].apply(lambda x: 1 if x == 'Female' else 0)
    df['ever_married'] = df['ever_married'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['Residence_type'] = df['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)
    df['work_type_Private'] = df['work_type'].apply(lambda x: 1 if x == 'Private' else 0)
    df['work_type_Self_employed'] = df['work_type'].apply(lambda x: 1 if x == 'Self-employed' else 0)
    df['work_type_Govt_job'] = df['work_type'].apply(lambda x: 1 if x == 'Govt_job' else 0)
    df['work_type_children'] = df['work_type'].apply(lambda x: 1 if x == 'children' else 0)
    df['work_type_Never_worked'] = df['work_type'].apply(lambda x: 1 if x == 'Never_worked' else 0)
    df['smoking_status_formerly_smoked'] = df['smoking_status'].apply(lambda x: 1 if x == 'formerly smoked' else 0)
    df['smoking_status_never_smoked'] = df['smoking_status'].apply(lambda x: 1 if x == 'never smoked' else 0)
    df['smoking_status_smokes'] = df['smoking_status'].apply(lambda x: 1 if x == 'smokes' else 0)
    df['smoking_status_Unknown'] = df['smoking_status'].apply(lambda x: 1 if x == 'Unknown' else 0)

    df_model = df.copy()
    df_model.drop(['Residence_type', 'work_type', 'smoking_status', 'gender', 'ever_married'], axis=1, inplace=True)

    X = df_model.drop('stroke', axis=1)
    y = df_model['stroke']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)


    logistic_reg = LogisticRegression(learning_rate=0.01, n_iters=1000)
    logistic_reg.fit(X_train, y_train)


    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    lasso_reg = Lasso()
    lasso_reg.fit(X_train, y_train)
    rigid_reg = Ridge()
    rigid_reg.fit(X_train, y_train)


    logistic_pred = logistic_reg.predict(X_test)
    logistic_acc = logistic_reg.score(X_test, y_test)
    log_reg_rmse = np.sqrt(mean_squared_error(y_test, logistic_pred))


    results = pd.DataFrame({
        'Model': ['Linear Regression', 'Lasso Regression', 'Ridge Regression', 'Logistic Regression'],
        'Accuracy': [
            linear_reg.score(X_test, y_test),
            lasso_reg.score(X_test, y_test),
            rigid_reg.score(X_test, y_test),
            logistic_acc
        ],
        'RMSE': [
            np.sqrt(mean_squared_error(y_test, linear_reg.predict(X_test))),
            np.sqrt(mean_squared_error(y_test, lasso_reg.predict(X_test))),
            np.sqrt(mean_squared_error(y_test, rigid_reg.predict(X_test))),
            log_reg_rmse
        ]
    })


    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.barplot(x='Model', y='Accuracy', data=results, palette='Blues_d', ax=ax[0])
    ax[0].set_title('Model Accuracy')
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)

    sns.barplot(x='Model', y='RMSE', data=results, palette='Reds_d', ax=ax[1])
    ax[1].set_title('Model RMSE')
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)

    st.pyplot(fig)

elif options == "Stroke Prediction":
    st.header("Predict Stroke Risk Using Machine Learning")


    df['gender_Male'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
    df['gender_Female'] = df['gender'].apply(lambda x: 1 if x == 'Female' else 0)
    df['ever_married'] = df['ever_married'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['Residence_type'] = df['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)
    df['work_type_Private'] = df['work_type'].apply(lambda x: 1 if x == 'Private' else 0)
    df['work_type_Self_employed'] = df['work_type'].apply(lambda x: 1 if x == 'Self-employed' else 0)
    df['work_type_Govt_job'] = df['work_type'].apply(lambda x: 1 if x == 'Govt_job' else 0)
    df['work_type_children'] = df['work_type'].apply(lambda x: 1 if x == 'children' else 0)
    df['work_type_Never_worked'] = df['work_type'].apply(lambda x: 1 if x == 'Never_worked' else 0)
    df['smoking_status_formerly_smoked'] = df['smoking_status'].apply(lambda x: 1 if x == 'formerly smoked' else 0)
    df['smoking_status_never_smoked'] = df['smoking_status'].apply(lambda x: 1 if x == 'never smoked' else 0)
    df['smoking_status_smokes'] = df['smoking_status'].apply(lambda x: 1 if x == 'smokes' else 0)
    df['smoking_status_Unknown'] = df['smoking_status'].apply(lambda x: 1 if x == 'Unknown' else 0)

    df_model = df.copy()
    df_model.drop(['Residence_type', 'work_type', 'smoking_status', 'gender', 'ever_married'], axis=1, inplace=True)

    X = df_model.drop('stroke', axis=1)
    y = df_model['stroke']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)


    logistic_reg = LogisticRegression( learning_rate=0.01, n_iters=1000)
    logistic_reg.fit(X_train, y_train)

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
        with st.spinner('Predicting stroke risk...'):
            stroke_prediction = logistic_reg.predict(user_input_df)
        if stroke_prediction == 1:
            st.success("You are at risk of a stroke.")
        else:
            st.success("You are not at risk of a stroke.")
