import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('D:\\coding stuff\\infosys\\github\\dataset\\data.csv')


data.head()
data['Urban/Rural'] = data['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)


#One-Hot Encode

data['work_type_Never_worked'] = data['work_type'].apply(lambda x: 1 if x == 'Never_worked' else 0)
data['work_type_Private'] = data['work_type'].apply(lambda x: 1 if x == 'Private' else 0)
data['work_type_Self_employed'] = data['work_type'].apply(lambda x: 1 if x == 'Self-employed' else 0)


data['smoking_status_formerly_smoked'] = data['smoking_status'].apply(lambda x: 1 if x == 'formerly smoked' else 0)
data['smoking_status_never_smoked'] = data['smoking_status'].apply(lambda x: 1 if x == 'never smoked' else 0)
data['smoking_status_smokes'] = data['smoking_status'].apply(lambda x: 1 if x == 'smokes' else 0)




model_data = data.copy()


model_data.drop(['Residence_type', 'work_type', 'smoking_status'], axis=1, inplace=True)
print(model_data)