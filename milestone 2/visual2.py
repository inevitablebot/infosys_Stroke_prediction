import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('D:\\coding stuff\\infosys\\github\\dataset\\data.csv')

# Display the first few rows to check the data
data.head()


# What is the age distribution among patients?


#Histogram for Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['age'], kde=True)
plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# How does average glucose level differ between patients with and without a stroke?

#Boxplot for Average Glucose Level by Stroke Status
plt.figure(figsize=(10, 6))
sns.boxplot(x='stroke', y='avg_glucose_level', data=data)
plt.title("Average Glucose Level by Stroke Status")
plt.xlabel("Stroke")
plt.ylabel("Average Glucose Level")
plt.show()

#  Is there a relationship between hypertension and the likelihood of having a stroke?

#Bar Chart for Hypertension vs. Stroke Status
plt.figure(figsize=(8, 6))
sns.countplot(x='hypertension', data=data)
plt.title("Hypertension vs Stroke Status")
plt.xlabel("Hypertension")
plt.ylabel("Count")
plt.legend(title="Stroke")
plt.show()

# What is the distribution of residence types (Urban/Rural) among stroke patients?

#Pie Chart for Residence Type among Stroke Patients
stroke_residence = data[data['stroke'] == 1]['Residence_type'].value_counts()
plt.figure(figsize=(6, 6))
stroke_residence.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title("Residence Type Distribution Among Stroke Patients")
plt.ylabel("")  # Hide the y-label
plt.show()

# Is there any correlation between BMI and average glucose level among stroke patients?
plt.figure(figsize=(10, 6))
sns.scatterplot(x='bmi', y='avg_glucose_level', hue='stroke', data=data, palette="viridis")
plt.title("BMI vs Average Glucose Level by Stroke Status")
plt.xlabel("BMI")
plt.ylabel("Average Glucose Level")
plt.legend(title="Stroke")
plt.show()

#How do age, average glucose level, and BMI interact in relation to stroke status?

#Scatter Plot for BMI vs Average Glucose Level
#Pairplot
sns.pairplot(data, vars=['age', 'avg_glucose_level', 'bmi'], hue='stroke')
plt.suptitle("Pairplot of Age, Glucose Level, and BMI by Stroke Status", y=1.02)
plt.show()
