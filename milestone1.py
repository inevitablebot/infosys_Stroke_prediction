import pandas as pd

# Load dataset
df = pd.read_csv('dataset/data.csv')

# Basic exploration
print("Basic statistical description of numerical data:")
print(df.describe()) 

print("\nDataset information (columns, types, and non-null counts):")
print(df.info()) 

print("\nShape of the dataset (rows, columns):")
print(df.shape)  
print("\nBasic statistical description of categorical data:")
print(df.describe(include='object'))  

# Unique values and null value analysis
print("\nUnique values in 'gender' column:")
gender_unique = df['gender'].unique()
print(gender_unique)  

print("\nUnique values in 'smoking_status' column:")
smoking_status_unique = df['smoking_status'].unique()
print(smoking_status_unique)  

# Check for null values
null_values = df.isnull().sum()  
print("\nNull values in each column:")
print(null_values)

# Percentage of null values
null_percentage = df.isnull().mean() * 100 
print("\nPercentage of null values in each column:")
print(null_percentage)

# Observations
print("\nObservations:")

# Display the data types of each column to ensure they align with expected types
print("\nData types of each column:")
print(df.dtypes)

# **Observation 1:**
#  The dataset contains 5110 rows and 12 columns.
print(f"1. The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

# **Observation 2:**
#  The 'bmi' column originally contained 201 missing values.
missing_bmi_count = null_values['bmi']
if missing_bmi_count > 0:
    print(f"2. The 'bmi' column originally contained {missing_bmi_count} missing values.")
else:
    print("2. The 'bmi' column contains no missing values.")

# **Observation 3:**
#  The 'gender' column contains the following unique values: ['Male', 'Female', 'Other'].
print(f"3. The 'gender' column contains the following unique values: {gender_unique}")

# **Observation 4:**
#  The 'smoking_status' column contains the following unique values: ['formerly smoked', 'never smoked', 'smokes', 'Unknown'].
print(f"4. The 'smoking_status' column contains the following unique values: {smoking_status_unique}")

# **Observation 5:**
#  Missing data is found in the 'bmi' column, with a total of 3.93% of its values missing.
print(f"5. The dataset contains missing data in the following columns (with percentages):")
print(null_percentage[null_percentage > 0])

# **Observation 6:**
#  After handling missing values, no missing values remain in the 'bmi' column.
# Option 1: Dropping rows with missing values in 'bmi'
df_dropped = df.dropna(subset=['bmi'])  # Option 1
print(f"6. After dropping rows with missing 'bmi' values, the dataset contains {df_dropped.shape[0]} rows.")

# Option 2: Impute missing 'bmi' values with the mean
mean_bmi = df['bmi'].mean()
df['bmi'].fillna(mean_bmi, inplace=True)  # Option 2
print(f"\nImputing missing 'bmi' values with mean value: {mean_bmi}")

# Checking null values again after imputation
null_values_after = df.isnull().sum()
print("\nNull values after imputing missing 'bmi' values:")
print(null_values_after)

# **Observation 7:**
# After imputing missing values, the 'bmi' column has no missing values.
if null_values_after['bmi'] == 0:
    print("7. After imputing, the 'bmi' column has no missing values.")

# Identify if there are any duplicate rows in the dataset.
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# **Observation 8:**
#strok rate by genders 
print("\nStroke rate by gender:")
print(df.groupby('gender')['stroke'].mean())
#observation of Stroke rate by gender:
#gender
#Female    0.047094
#Male      0.051064
#Other     0.000000

#percentage
total = df['stroke'].sum()
strokes_gender = df[df['stroke'] == 1].groupby('gender')['stroke'].count()
stroke_per = (strokes_gender / total) * 100

total = df['stroke'].sum()
strokes_gender = df[df['stroke'] == 1].groupby('gender')['stroke'].count()
stroke_per = (strokes_gender / total) * 100
print("\nStroke percentage by gender (relative to all stroke cases):")
print(stroke_per)