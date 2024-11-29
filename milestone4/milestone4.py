import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import datetime as dt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import precision_score, f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

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


precision, recall, thresholds = precision_recall_curve(y_test, Logistic_pred)
thresholds

display = PrecisionRecallDisplay.from_estimator(logistic_reg, X_test, y_test, name="Logistic Regression", plot_chance_level=True) 
display.ax_.set_title("2-class Precision-Recall curve")
plt.show()


#---------------------------------Comparison-------------------------------#
#--------------------------------------------------------------------------#
"""
Precision Comparison
   # Precision Score
    The precision score of 0.50 indicates that when the model predicts a
    stroke (positive class), it is correct 50% of the time. This suggests the
    model is often wrong when predicting the positive class, leading to a
    significant number of false positives.

F1 Score Comparison
   # F1 Score
   The F1 score of 0.03 is extremely low, indicating a poor balance between 
   precision and recall. This score reveals that the model's ability to predict 
   stroke cases accurately is limited, and it misses most positive cases.

Recall Score Comparison
   # Recall Score
   With a recall score of 0.02, the model fails to identify the true positive 
   stroke cases. This suggests that the model's performance is skewed, heavily 
   favoring the majority class (no stroke), and missing most of the minority class 
   (stroke).
"""
#--------------------------------------------------------------------------#

#------------------------------- Precision-Recall curve--------------------#
#--------------------------------------------------------------------------#
"""
This Precision-Recall curve evaluates the performance of a logistic 
regression model for classifying stroke cases (positive label). 
The area under the curve (AP = 0.16) indicates modest performance, slightly
better than random chance (AP = 0.06). 

High recall comes at the cost of lower precision, as the model struggles to
correctly identify stroke cases while overpredicting them, 
leading to a high number of false positives.
"""
#--------------------------------------------------------------------------#




