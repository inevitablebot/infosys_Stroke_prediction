import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
# Load dataset
df = pd.read_csv('D:\\coding stuff\\infosys\\github\\dataset\\data.csv')

# Fill missing BMI values with mean for consistency in some analyses
df = df.assign(bmi=df['bmi'].fillna(df['bmi'].mean()))


# Define a consistent style for all plots
sns.set(style="whitegrid")

# Age Distribution of Stroke Patients

plt.figure(figsize=(8, 6))
gender_stroke = df.groupby('gender')['stroke'].mean() * 100  # Convert to percentage
sns.barplot(x=gender_stroke.index, y=gender_stroke.values, palette='viridis')
plt.title('Stroke Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Stroke Rate (%)')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())  # Set y-axis to percentage format
plt.show()

# Stroke Rate by Gender
plt.figure(figsize=(8, 6))
gender_stroke = df.groupby('gender')['stroke'].mean()
sns.barplot(x=gender_stroke.index, y=gender_stroke.values, palette='viridis', hue=gender_stroke.index, dodge=False)

plt.title('Stroke Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Stroke Rate')
plt.show()

# Impact of Glucose Levels on Stroke Occurrence
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='avg_glucose_level', y='stroke', hue='stroke', palette={0: 'lightblue', 1: 'salmon'})
plt.title('Impact of Glucose Levels on Stroke Occurrence')
plt.xlabel('Average Glucose Level')
plt.ylabel('Stroke Occurrence (0 = No, 1 = Yes)')
plt.show()

# Stroke Rate by Hypertension and Heart Disease
plt.figure(figsize=(12, 6))
hypertension_heart_disease = df.groupby(['hypertension', 'heart_disease'])['stroke'].mean().unstack()
hypertension_heart_disease.plot(kind='bar', stacked=False, color=['skyblue', 'salmon'], ax=plt.gca())
plt.title('Stroke Rate by Hypertension and Heart Disease')
plt.xlabel('Hypertension (0 = No, 1 = Yes)')
plt.ylabel('Stroke Rate')
plt.legend(title='Heart Disease')
plt.show()


# BMI Distribution of Stroke Patients

plt.figure(figsize=(10, 6))
sns.histplot(df[df['stroke'] == 1]['bmi'], bins=20, kde=True, color='lightgreen')
plt.title('BMI Distribution of Stroke Patients')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()


# Stroke Rate by Smoking Status
plt.figure(figsize=(10, 6))
smoking_status_stroke = df.groupby('smoking_status')['stroke'].mean()
sns.barplot(x=smoking_status_stroke.index, y=smoking_status_stroke.values, palette='magma')
plt.title('Stroke Rate by Smoking Status')
plt.xlabel('Smoking Status')
plt.ylabel('Stroke Rate')
plt.show()
