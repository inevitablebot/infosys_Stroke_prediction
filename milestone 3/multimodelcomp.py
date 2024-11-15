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


df = pd.read_csv("D:\\coding stuff\\infosys\\data.csv")


df['bmi'] = df['bmi'].fillna(df['bmi'].median())


# Convert 'gender' to binary variables
df['gender_Male'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
df['gender_Female'] = df['gender'].apply(lambda x: 1 if x == 'Female' else 0)

# Convert 'ever_married' to binary (Yes=1, No=0)
df['ever_married'] = df['ever_married'].apply(lambda x: 1 if x == 'Yes' else 0)

# One-hot encode 'work_type'
df['work_type_Private'] = df['work_type'].apply(lambda x: 1 if x == 'Private' else 0)
df['work_type_Self_employed'] = df['work_type'].apply(lambda x: 1 if x == 'Self-employed' else 0)
df['work_type_Govt_job'] = df['work_type'].apply(lambda x: 1 if x == 'Govt_job' else 0)
df['work_type_children'] = df['work_type'].apply(lambda x: 1 if x == 'children' else 0)
df['work_type_Never_worked'] = df['work_type'].apply(lambda x: 1 if x == 'Never_worked' else 0)

# Convert 'Residence_type' to binary (Urban=1, Rural=0)
df['Residence_type'] = df['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)

# One-hot encode 'smoking_status'
df['smoking_status_formerly_smoked'] = df['smoking_status'].apply(lambda x: 1 if x == 'formerly smoked' else 0)
df['smoking_status_never_smoked'] = df['smoking_status'].apply(lambda x: 1 if x == 'never smoked' else 0)
df['smoking_status_smokes'] = df['smoking_status'].apply(lambda x: 1 if x == 'smokes' else 0)
df['smoking_status_Unknown'] = df['smoking_status'].apply(lambda x: 1 if x == 'Unknown' else 0)

# Drop original categorical columns as they are now encoded



df_model = df.copy()
df_model.drop(['Residence_type', 'work_type', 'smoking_status', 'gender', 'ever_married'], axis=1, inplace=True)


X = df_model.drop('stroke', axis=1)
y = df_model['stroke']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_reg_predictions = linear_reg.predict(X_test)
linear_reg_rmse = np.sqrt(mean_squared_error(y_test, linear_reg_predictions))*100
linear_reg_acc = linear_reg.score(X_test, y_test)*100
print(f"Linear Regression  Score: {linear_reg_acc :.2f}%")
print(f"Linear Regression RMSE: {linear_reg_rmse :.2f}%")


lasso_reg = Lasso()
lasso_reg.fit(X_train, y_train)
lasso_reg_predictions = lasso_reg.predict(X_test)
lasso_reg_rmse = np.sqrt(mean_squared_error(y_test, lasso_reg_predictions))*100
lasso_reg_acc = lasso_reg.score(X_test, y_test)*100
print(f"Lasso Regression  Score: {lasso_reg_acc :.2f}%")
print(f"Lasso Regression RMSE: {lasso_reg_rmse :.2f}%")


Rigid_reg = Ridge()
Rigid_reg.fit(X_train, y_train)
Rigid_reg_predictions = Rigid_reg.predict(X_test)
Rigid_reg_rmse = np.sqrt(mean_squared_error(y_test, Rigid_reg_predictions))*100
Rigid_reg_acce = Rigid_reg.score(X_test, y_test)*100
print(f"Ridge Regression  Score: {Rigid_reg_acce :.2f}%")
print(f"Ridge Regression RMSE: {Rigid_reg_rmse :.2f}%")


logistic_reg = LogisticRegression(max_iter=1000)
logistic_reg.fit(X_train, y_train)
Logistic_pred = logistic_reg.predict(X_test)
Log_reg_rmse = np.sqrt(mean_squared_error(y_test, Logistic_pred))*100
logistic_reg_acc = logistic_reg.score(X_test, y_test)*100
print(f"Logistic Regression  Score: {logistic_reg_acc :.2f}%")
print(f"Logistic Regression RMSE: {Log_reg_rmse :.2f}%")

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
plt.savefig(" Accuracy vs. RMSE .png")
plt.show()


#--------------------------------------------------------------------------#
#----------------------------------Output----------------------------------#
"""
Linear Regression  Score: 9.10%
Linear Regression RMSE: 22.76%

Lasso Regression  Score: 0.94%
Lasso Regression RMSE: 23.76%

Ridge Regression  Score: 9.10%
Ridge Regression RMSE: 22.76%

Logistic Regression  Score: 93.93%
Logistic Regression RMSE: 24.63%

                 Model   Accuracy       RMSE
0    Linear Regression   9.355712  22.727454
1     Lasso Regression   0.942424  23.758795
2     Ridge Regression   9.354708  22.727580
3  Logistic Regression  93.933464  24.630339


"""

#--------------------------------------------------------------------------#
"""
Accuracy Comparison
   # Logistic Regression 
        it has the highest accuracy at 93.93%,
        which significantly outperforms the other models. This high accuracy 
        suggests that logistic regression is well-suited for the dataset, 
        especially given that its designed for binary classification 
        (i.e., predicting stroke or no stroke).
   # The other models 
        (Linear, Lasso, and Ridge Regression) show much lower accuracy, 
        with values below 10%. This indicates that these regression models
        may not be appropriate for this binary classification task, as they
        are better suited for predicting continuous variables rather than
        categorical outcomes.    
"""
#--------------------------------------------------------------------------#

#--------------------------------------------------------------------------#
"""
RMSE Comparison:
    Linear, Lasso, and Ridge Regression 
        these models have similar RMSE values 
        (around 0.23), indicating consistent performance in predicting 
        continuous values.
    Logistic Regression 
        shows a slightly higher RMSE of 0.246, but in this context, RMSE is 
        less relevant since logistic regression is used for classification
        rather than for continuous value prediction    
"""
#--------------------------------------------------------------------------#


#--------------------------------------------------------------------------#
"""
Why Logistic Regression Works Well
    Logistic regression is designed for binary classification and works 
    well when the relationship between the target variable and the 
    predictors is linear. In our case, predicting stroke (1) or no stroke 
    (0) based on features like BMI, age, gender, etc.,
    aligns well with logistic regression's strengths.


Why the Other Models (Linear, Lasso, Ridge) Don’t Perform Well

    Linear regression models are intended for predicting continuous
    values. Since the target variable here is binary (stroke/no stroke),
    these models struggle to classify it effectively, which is why their 
    accuracy is so low. The same issue applies to Ridge and Lasso 
    regression models. Although these are regularized versions of linear
    regression, they still approach the problem as if predicting a
    continuous outcome, making them less suitable for binary
    classification tasks like this one.

"""
#--------------------------------------------------------------------------#

#-------------------------------Conclusion---------------------------------#

"""
Logistic regression is the most effective model for this dataset because 
it is specifically designed for binary classification, and the relationship 
between the target variable (stroke/no stroke) and the predictors is linear. 
The other models (Linear, Lasso, and Ridge regression) are not suitable for 
this task, as they are designed for predicting continuous values rather 
than binary outcomes. This makes logistic regression the best fit for 
predicting binary outcomes like this one.

"""
#----------------------------------END--------------------------------------#