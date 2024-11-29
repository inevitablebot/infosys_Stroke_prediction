import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import  Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, f1_score, recall_score, precision_recall_curve, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
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
    return df, df_model

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

        for _ in range(self.n_iters):
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



def train_models(X_train, X_test, y_train, y_test):
    results = []


    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    linear_acc = linear_reg.score(X_test, y_test)
    linear_rmse = np.sqrt(mean_squared_error(y_test, linear_reg.predict(X_test)))
    results.append(('Linear Regression', linear_acc, linear_rmse))

 
    lasso_reg = Lasso()
    lasso_reg.fit(X_train, y_train)
    lasso_acc = lasso_reg.score(X_test, y_test)
    lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_reg.predict(X_test)))
    results.append(('Lasso Regression', lasso_acc, lasso_rmse))


    ridge_reg = Ridge()
    ridge_reg.fit(X_train, y_train)
    ridge_acc = ridge_reg.score(X_test, y_test)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_reg.predict(X_test)))
    results.append(('Ridge Regression', ridge_acc, ridge_rmse))


    logistic_reg = LogisticRegression()
    logistic_reg.fit(X_train, y_train)
    logistic_acc = logistic_reg.score(X_test, y_test)
    logistic_rmse = np.sqrt(mean_squared_error(y_test, logistic_reg.predict(X_test)))
    results.append(('Logistic Regression', logistic_acc, logistic_rmse))

    return results, logistic_reg

file_path = "dataset/data.csv"
if file_path:
    df, df_model = load_data(file_path)
    st.write("Preview of the dataset:")
    st.dataframe(df.head())

 
    X = df_model.drop('stroke', axis=1)
    y = df_model['stroke']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)


    results, logistic_reg = train_models(X_train, X_test, y_train, y_test)


    st.write("### Model Results")
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "RMSE"])
    st.dataframe(results_df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.barplot(x="Model", y="Accuracy", data=results_df, ax=axes[0], palette="Blues_d")
    sns.barplot(x="Model", y="RMSE", data=results_df, ax=axes[1], palette="Reds_d")
    axes[0].set_title("Model Accuracy")
    axes[1].set_title("Model RMSE")
    st.pyplot(fig)


    logistic_pred = logistic_reg.predict(X_test)
    cm = confusion_matrix(y_test, logistic_pred)
    st.write("### Confusion Matrix")
    st.dataframe(cm)

    precision = precision_score(y_test, logistic_pred)
    recall = recall_score(y_test, logistic_pred)
    f1 = f1_score(y_test, logistic_pred)
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

    precision, recall, _ = precision_recall_curve(y_test, logistic_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    st.pyplot(plt)


    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    hypertension = st.selectbox("Hypertension (0: No, 1: Yes)", [0, 1])
    heart_disease = st.selectbox("Heart Disease (0: No, 1: Yes)", [0, 1])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, step=0.1)
    bmi = st.number_input("BMI", min_value=0.0, step=0.1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox(
        "Work Type",
        ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
    )
    smoking_status = st.selectbox(
        "Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"]
    )

input_data = {
    "Age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "gender_Male": 1 if gender == "Male" else 0,
    "gender_Female": 1 if gender == "Female" else 0,
    "work_type_Private": 1 if work_type == "Private" else 0,
    "work_type_Self_employed": 1 if work_type == "Self-employed" else 0,
    "work_type_Govt_job": 1 if work_type == "Govt_job" else 0,
    "work_type_children": 1 if work_type == "children" else 0,
    "work_type_Never_worked": 1 if work_type == "Never_worked" else 0,
    "smoking_status_formerly_smoked": 1 if smoking_status == "formerly smoked" else 0,
    "smoking_status_never_smoked": 1 if smoking_status == "never smoked" else 0,
    "smoking_status_smokes": 1 if smoking_status == "smokes" else 0,
    "smoking_status_Unknown": 1 if smoking_status == "Unknown" else 0,
}


input_df = pd.DataFrame([input_data])

model_columns = X_train.columns  
input_df = input_df.reindex(columns=model_columns, fill_value=0)

if st.button("Predict Stroke Probability"):
    prediction_proba = logistic_reg.predict_proba(input_df)[0][1]
    st.write(f"### Stroke Probability: {prediction_proba * 100:.2f}%")
    if prediction_proba >= 0.5:
        st.error("High risk of stroke detected! Recommend medical consultation.")
    else:
        st.success("Low risk of stroke detected.")
