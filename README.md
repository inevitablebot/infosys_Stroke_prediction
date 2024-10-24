# infosys_Stroke_prediction

-----------------------------------------------------
## Milestone 1

### Dataset Observations

  

### 1. Basic Dataset Information

- The dataset contains **5110 rows** and **12 columns**.

  

### 2. Missing Values in 'bmi' Column

- The 'bmi' column originally contained **201 missing values**, which is approximately **3.93%** of the column's total data.

  

### 3. Unique Values in 'gender' Column

- The 'gender' column contains the following unique values: **['Male', 'Female', 'Other']**.

  

### 4. Unique Values in 'smoking_status' Column

- The 'smoking_status' column contains the following unique values: **['formerly smoked', 'never smoked', 'smokes', 'Unknown']**.

  

### 5. Missing Data

- The dataset contains missing data in the following columns (with percentages):

  - 'bmi': **3.93%**.

  

### 6. Handling Missing 'bmi' Values

- After dropping rows with missing 'bmi' values, the dataset contains **4909 rows**.

- Alternatively, after imputing the missing 'bmi' values with the mean value of **28.89**, the 'bmi' column no longer has missing values.

  

### 7. Duplicate Rows

- There are **no duplicate rows** in the dataset.

  

### 8. Stroke Rate by Gender (Proportion within Gender)

- **Female**: **4.71%** stroke rate.

- **Male**: **5.11%** stroke rate.

- **Other**: **0%** stroke rate.



### 9. Stroke Percentage by Gender (Relative to All Stroke Cases):

- **Female**: **56.63%** of total stroke cases

- **Male**: **43.37%** of total stroke cases
