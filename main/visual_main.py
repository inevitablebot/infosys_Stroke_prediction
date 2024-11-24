import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('D:\\coding stuff\\infosys\\github\\dataset\\data.csv')

data.head()

# What is the age distribution among patients?

# Histogram for Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['age'], kde=True)
plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig('age_distribution.png') 
plt.show()

#...............................................................................................#
#                                           observation                                         #
# The age distribution shows a roughly bimodal trend, with peaks around the ages of 40–60 
# and 80. There is a steady increase in patient numbers from younger ages up to middle age,
# after which it drops slightly before peaking again in older age groups.
#...............................................END.............................................#
#...............................................................................................#

# How does average glucose level differ between patients with and without a stroke?

# Boxplot for Average Glucose Level by Stroke Status
plt.figure(figsize=(10, 6))
sns.boxplot(x='stroke', y='avg_glucose_level', data=data)
plt.title("Average Glucose Level by Stroke Status")
plt.xlabel("Stroke")
plt.ylabel("Average Glucose Level")
plt.savefig('glucose_by_stroke.png')  
plt.show()

#...............................................................................................#
#                                           observation                                         #
# 1 The median glucose level is higher for patients who have experienced a stroke compared to 
#   those who have not.
# 2 There is a wider spread of glucose levels in stroke patients, indicating that high glucose 
#   levels might be more common in this group.
#...............................................END.............................................#
#...............................................................................................#

# Is there a relationship between hypertension and the likelihood of having a stroke?

# Bar Chart for Hypertension vs. Stroke Status
plt.figure(figsize=(8, 6))
sns.countplot(x='hypertension', data=data)
plt.title("Hypertension vs Stroke Status")
plt.xlabel("Hypertension")
plt.ylabel("Count")
plt.legend(title="Stroke")
plt.savefig('hypertension_vs_stroke.png') 
plt.show()

#...............................................................................................#
#                                           observation                                         #
# Most patients with hypertension do not have a stroke, as indicated by the large bar at the 
# zero mark. This suggests that while hypertension is a known risk factor, the majority of 
# hypertensive patients in this dataset have not experienced a stroke.
#...............................................END.............................................#
#...............................................................................................#


# What is the distribution of residence types (Urban/Rural) among stroke patients?

# Pie Chart for Residence Type among Stroke Patients
stroke_residence = data[data['stroke'] == 1]['Residence_type'].value_counts()
plt.figure(figsize=(6, 6))
stroke_residence.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title("Residence Type Distribution Among Stroke Patients")
plt.ylabel("")  
plt.savefig('residence_type_distribution.png')  
plt.show()

#...............................................................................................#
#                                           observation                                         #
# 1 Stroke patients are slightly more likely to be from urban areas (54.2%) than rural areas 
#   (45.8%).
# 2 This distribution suggests that stroke cases are fairly common in both urban and rural 
#   populations, with a minor inclination toward urban residents.
#...............................................END.............................................#
#...............................................................................................#

# Is there any correlation between BMI and average glucose level among stroke patients?
plt.figure(figsize=(10, 6))
sns.scatterplot(x='bmi', y='avg_glucose_level', hue='stroke', data=data, palette="viridis")
plt.title("BMI vs Average Glucose Level by Stroke Status")
plt.xlabel("BMI")
plt.ylabel("Average Glucose Level")
plt.legend(title="Stroke")
plt.savefig('bmi_vs_glucose.png') 
plt.show()

#...............................................................................................#
#                                           observation                                         #
# 1     Patients with higher glucose levels and a wide range of BMIs 
#       (especially in the 20-60 range) are present among both stroke and non-stroke groups.
# 2     Stroke cases (marked by a different color) tend to cluster more towards higher glucose
#       levels, suggesting a possible link between elevated glucose levels and stroke occurrence
#...............................................END.............................................#
#...............................................................................................#

# How do age, average glucose level, and BMI interact in relation to stroke status?

# Pairplot for Age, Average Glucose Level, and BMI by Stroke Status
sns.pairplot(data, vars=['age', 'avg_glucose_level', 'bmi'], hue='stroke')
plt.suptitle("Pairplot of Age, Glucose Level, and BMI by Stroke Status", y=1.02)
plt.savefig('pairplot_age_glucose_bmi.png') 
plt.show()

#...............................................................................................#
#                                           observation                                         #
# Patients who have experienced a stroke (indicated by orange markers) appear to cluster around 
# higher glucose levels and a wide range of ages, mostly over 40. There are a few outliers, with
# higher BMI values but no clear pattern between BMI and stroke status. This suggests that age 
# and glucose level may have a stronger correlation with stroke than BMI alone.
#...............................................END.............................................#
#...............................................................................................#