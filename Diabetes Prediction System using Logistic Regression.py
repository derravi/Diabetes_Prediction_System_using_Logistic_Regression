    #IMport The Librarys
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#Import the Dataset

df = pd.read_csv("diabetes.csv")
print("\nDataset Values.\n")
df.head()

#Find the Shape of the Dataset
print(f"Total Columns of the Data set is {df.shape[1]} and the Rows is {df.shape[0]}")

#Describe the Dataset
print("\nLest see the Describe Values of the Datasets\n")
df.describe()

#Check the Datatpes of the Dataset columns
print("\nLets see all the columns Data types\n")
df.dtypes

#Check any null values present or not.
print("\nLets see the Null values of this datasets\n")
df.isnull().sum()

#Give the Input Data and Output Dta
x = df.iloc[:,:-1]
y = df["Outcome"]

#Train Test And Split Functions

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


#Scal Down all the Fetures
std = StandardScaler()
x_train_scaled = std.fit_transform(x_train)
x_test_scaled = std.transform(x_test)

model = LogisticRegression()
model.fit(x_train_scaled,y_train)

y_predict = model.predict(x_test_scaled)

#Model Evaluation

print("\nModel Accuracy:\n",accuracy_score(y_test,y_predict))
print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_predict))
print("\nClassification Report:\n",classification_report(y_test,y_predict))

#Confustion Matrix Heatmap

plt.figure(figsize=(10,5))
sns.heatmap(confusion_matrix(y_test,y_predict),annot=True,fmt="d",cmap="Blues")
plt.xlabel("Predictino")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("Confusion_matrix.png",dpi=500,bbox_inches='tight')
plt.show()


#Distribution plots for each feature

plt.figure(figsize=(16, 12))
for i,column in enumerate(x.columns,1):
    plt.subplot(3,3,i)
    sns.histplot(df[column],kde=True,color='green',edgecolor='black')
    plt.title(f"Distribution of {column}",fontsize=12)
plt.tight_layout()
plt.savefig("Distribution_plots_for_each_feature.png",dpi=500,bbox_inches='tight')
plt.show()

#Correlation Heatmap of Diabetes Dataset

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Diabetes Dataset", fontsize=15)
plt.savefig("Correlation_Heatmap_of_Diabetes_Dataset.png",dpi=500,bbox_inches='tight')
plt.show()

#Count of Diabetic vs Non-Diabetic Patients

plt.figure(figsize=(6, 4))
sns.countplot(x="Outcome", data=df, palette="pastel")
plt.title("Count of Diabetic vs Non-Diabetic Patients", fontsize=14)
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Count")
plt.savefig("Count_of_Diabetic_vs_Non-Diabetic_Patients.png",dpi=500,bbox_inches='tight')
plt.show()

#Glucose Levels by Diabetes Outcome

plt.figure(figsize=(6, 4))
sns.boxplot(x="Outcome", y="Glucose", data=df, palette="Set2")
plt.title("Glucose Levels by Diabetes Outcome", fontsize=14)
plt.xlabel("Outcome")
plt.ylabel("Glucose Level")
plt.savefig("Glucose_Levels_by_Diabetes_Outcome.png",dpi=500,bbox_inches='tight')
plt.show()



print("\n Enter Your Helth Data:\n")

Pregnancies = int(input("Enter the Pregnancies(like 8):"))
Glucose = int(input("Enter the Glucose(like 130):"))
BloodPressure = int(input("Enter the BloodPressure(like 50 to 90):"))
SkinThickness = int(input("Enter the SkinThickness(like 20 to 40):"))
Insulin = int(input("Enter the Insulin(like 0 to 200):"))
BMI = int(input("Enter the BMI(like 20 to 60):"))
DiabetesPedigreeFunction = float(input("Enter the DiabetesPedigreeFunction(like 0.00 to 5):"))
Age = int(input("Enter the Age:"))

output = model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

if output == 1:
    print("You have a risk of Diabetes.")
else:
    print("You don't have Diabetes.")

