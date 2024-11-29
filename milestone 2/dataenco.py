
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns 

data = pd.read_csv("dataset\data.csv")

data.head()  

    # Encode 'Residence_type' as a binary feature 'Urban/Rural'
data['Urban/Rural'] = data['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)
    # Converting 'Residence_type' to binary (Urban=1, Rural=0) helps standardize data and makes it suitable for machine learning algorithms.

    # One-Hot Encode the 'work_type' variable
data['work_type_Never_worked'] = data['work_type'].apply(lambda x: 1 if x == 'Never_worked' else 0)
data['work_type_Private'] = data['work_type'].apply(lambda x: 1 if x == 'Private' else 0)
data['work_type_Self_employed'] = data['work_type'].apply(lambda x: 1 if x == 'Self-employed' else 0)
    # This process transforms categorical data into a series of binary columns. 
    # Each row will have a 1 in the column representing its category and 0 elsewhere. 
    # This is beneficial as it makes the data numeric and suitable for machine learning algorithms.

    # One-Hot Encode the 'smoking_status' variable
data['smoking_status_formerly_smoked'] = data['smoking_status'].apply(lambda x: 1 if x == 'formerly smoked' else 0)
data['smoking_status_never_smoked'] = data['smoking_status'].apply(lambda x: 1 if x == 'never smoked' else 0)
data['smoking_status_smokes'] = data['smoking_status'].apply(lambda x: 1 if x == 'smokes' else 0)
    # Similar to 'work_type', converting 'smoking_status' to a set of binary columns to prepare data for model training.
    # This step standardizes the categorical information, which helps improve model performance.

    # Create a copy of the data for model training and transformation
model_data = data.copy()  
    # Keeping a copy of the transformed data allows flexibility to use it for training without altering the original dataset.

    # Drop the original categorical columns that were one-hot encoded
model_data.drop(['Residence_type', 'work_type', 'smoking_status'], axis=1, inplace=True)
    # Removing the original columns reduces redundancy in the data, preventing multicollinearity issues.
    # The resulting dataset is now entirely numeric, making it ready for most machine learning models.

    # Print the final transformed dataset for verification
print(model_data)  
    # Displaying the transformed dataset provides a quick check to ensure the data is in the expected format for further analysis or model building.
