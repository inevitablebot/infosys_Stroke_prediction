import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import precision_score, f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt


df = pd.read_csv("dataset\data.csv")


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


linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_reg_predictions = linear_reg.predict(X_test)
linear_reg_rmse = np.sqrt(mean_squared_error(y_test, linear_reg_predictions))
linear_reg_acc = linear_reg.score(X_test, y_test)
print(f"Linear Regression  Score: {linear_reg_acc * 100:.2f}%")
print(f"Linear Regression RMSE: {linear_reg_rmse * 100:.2f}%")


lasso_reg = Lasso()
lasso_reg.fit(X_train, y_train)
lasso_reg_predictions = lasso_reg.predict(X_test)
lasso_reg_rmse = np.sqrt(mean_squared_error(y_test, lasso_reg_predictions))
lasso_reg_acc = lasso_reg.score(X_test, y_test)
print(f"Lasso Regression  Score: {lasso_reg_acc * 100:.2f}%")
print(f"Lasso Regression RMSE: {lasso_reg_rmse * 100:.2f}%")


Rigid_reg = Ridge()
Rigid_reg.fit(X_train, y_train)
Rigid_reg_predictions = Rigid_reg.predict(X_test)
Rigid_reg_rmse = np.sqrt(mean_squared_error(y_test, Rigid_reg_predictions))
Rigid_reg_acce = Rigid_reg.score(X_test, y_test)
print(f"Ridge Regression  Score: {Rigid_reg_acce * 100:.2f}%")
print(f"Ridge Regression RMSE: {Rigid_reg_rmse * 100:.2f}%")


logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, y_train)
Logistic_pred = logistic_reg.predict(X_test)
Log_reg_rmse = np.sqrt(mean_squared_error(y_test, Logistic_pred))
logistic_reg_acc = logistic_reg.score(X_test, y_test)
print(f"Logistic Regression  Score: {logistic_reg_acc * 100:.2f}%")
print(f"Logistic Regression RMSE: {Log_reg_rmse * 100:.2f}%")

results = pd.DataFrame({
    'Model': ['Linear Regression', 'Lasso Regression', 'Ridge Regression', 'Logistic Regression'],
    'Accuracy': [linear_reg_acc, lasso_reg_acc, Rigid_reg_acce, logistic_reg_acc],
    'RMSE': [linear_reg_rmse, lasso_reg_rmse, Rigid_reg_rmse, Log_reg_rmse]
})
print(results)

plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='Accuracy', data=results, palette='Blues_d')
plt.title('Model Accuracy')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='RMSE', data=results, palette='Reds_d')
plt.title('Model RMSE')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

Confusion_matrix = metrics.confusion_matrix(y_test, Logistic_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = Confusion_matrix, display_labels = [0, 1])
print(Confusion_matrix)
cm_display.plot()
plt.show()
tp = Confusion_matrix[0][0]
fn = Confusion_matrix[0][1]
fp = Confusion_matrix[1][0]
tn = Confusion_matrix[1][1]
precision = precision_score(y_test, Logistic_pred)
f1 = f1_score(y_test, Logistic_pred)
recall=recall_score(y_test, Logistic_pred)
print(f"Precision Score: {precision:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Recall Score: {recall:.2f}")