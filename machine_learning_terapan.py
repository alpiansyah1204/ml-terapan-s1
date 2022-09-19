# -*- coding: utf-8 -*-
"""Machine Learning Terapan.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Z1UBMg9SQaT7DhFFqN32IaEzpMvctOrk

#Import Library
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

"""#Import kaggle """

pip install -q kaggle

from google.colab import files

files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle

!kaggle datasets download -d alexteboul/diabetes-health-indicators-dataset

local_zip = '/content/diabetes-health-indicators-dataset.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content')
zip_ref.close()

"""#Import Dataset"""

df = pd.read_csv('/content/diabetes_binary_health_indicators_BRFSS2015.csv')
df.head()

"""#EXPLORATORY DATA ANALYSIS (EDA) & DATA CLEANING"""

df.info()

df.describe()

df.isnull().sum()

df['Diabetes_binary'].value_counts()

"""Melihat korelasi antar column"""

plt.figure(figsize=(20, 15))
sns.heatmap(data=df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")

plt.pie(df["Diabetes_binary"].value_counts(),labels = df["Diabetes_binary"].unique(),autopct = "%.2f%%",shadow=True,explode=(0,0.3),radius=1.5)
plt.legend()
plt.show()

"""dari piechart didapatkan 86.07% tidak mengidap diabetes dan 13.93 mengidap diabetes

Distribusi setiap variable yang ada di dalam dataset
"""

df.hist(figsize=(16,12))

"""disini saya menggunakan pairplot untuk melihat grafik mana yang memiliki kesamaan sehingga akan mempermudah untuk melakukan prediksi"""

sns.pairplot(df,x_vars=['HighBP','HighChol','CholCheck','BMI','Smoker','Stroke'],y_vars=['Diabetes_binary'])

sns.pairplot(df,x_vars=['HeartDiseaseorAttack','PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare'],y_vars=['Diabetes_binary'])

sns.pairplot(df,x_vars=['NoDocbcCost','GenHlth','MentHlth','PhysHlth','DiffWalk','Sex'],y_vars=['Diabetes_binary'])

sns.pairplot(df,x_vars=['Age','Education','Income'],y_vars=['Diabetes_binary'])

"""Mengecek value setiap column apakah ada data outlier"""

px = 1
plt.figure(figsize=(20,20))
for i in ['Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
       'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
       'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education','Income']:
    if px<23:
        plt.subplot(6,5,px)
        plt.boxplot(df[i])
        plt.title(i)
        px=px+1

def outlier():
    l = [ 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
       'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
       'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
       'Income']
    for i in l:
        x = np.quantile(df[i],[0.25,0.75])
        iqr = x[1]-x[0]   
        lof = x[0]-1.5*iqr   
        upf = x[1]+1.5*iqr   
        df[i] = np.where(df[i]>upf,upf,(np.where(df[i]<lof,lof,df[i])))
outlier()

px = 1
plt.figure(figsize=(20,20))
for i in ['Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
       'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
       'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education','Income']:
    if px<23:
        plt.subplot(6,5,px)
        plt.boxplot(df[i])
        plt.title(i)
        px=px+1

"""Melakukan drop pada beberapa column karena memiliki relasi yang tidak begitu baik dengan variable diabetes_binary. hal ini ditujukan agar dapat meningkatkan score prediksi"""

df = df.drop(columns=['CholCheck','Smoker','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','MentHlth','Sex'])

"""#Data preperation

disini saya akan melakukan Data preperation sehingga data tersebut dapat di training dan akan menghasilkan hasil prediksi yang baik
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

"""## Normalize"""

abs_scaler = MaxAbsScaler()
abs_scaler.fit(df)
scaled_data = abs_scaler.transform(df)
df_scaled = pd.DataFrame(scaled_data, columns = df.columns)
df_scaled.describe()

y = df_scaled['Diabetes_binary']
X = df_scaled.drop(columns=['Diabetes_binary'])
X

y.value_counts()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=93)

"""#Model Development dan Evaluasi Model"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score,r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

"""Logisticregresi"""

logisticRegressionModel = LogisticRegression().fit(X_train, y_train)
y_pred = logisticRegressionModel.predict(X_test)
print('mse:',mean_squared_error(y_test,y_pred))
print('rmse:',np.sqrt(mean_squared_error(y_test,y_pred)))
print('r^2:',r2_score(y_test,y_pred))
print('accuracy score:',accuracy_score(y_test, y_pred))

"""DecisionTreeClassifier"""

treeModel = DecisionTreeClassifier(min_samples_split = 60).fit(X_train, y_train)
y_pred_tree = treeModel.predict(X_test)
print('mse:',mean_squared_error(y_test,y_pred_tree))
print('rmse:',np.sqrt(mean_squared_error(y_test,y_pred_tree)))
print('r^2:',r2_score(y_test,y_pred_tree))
print('accuracy score:',accuracy_score(y_test, y_pred_tree))

"""RandomForestClassifier"""

forestModel = RandomForestClassifier(min_samples_split = 60).fit(X_train, y_train)
y_pred_forest = forestModel.predict(X_test)
print('mse:',mean_squared_error(y_test,y_pred_forest))
print('rmse:',np.sqrt(mean_squared_error(y_test,y_pred_forest)))
print('r^2:',r2_score(y_test,y_pred_forest))
print('accuracy score:',accuracy_score(y_test, y_pred_forest))

bayesModel = GaussianNB().fit(X_train, y_train)
y_pred_bayes = bayesModel.predict(X_test)
print('mse:',mean_squared_error(y_test,y_pred_bayes))
print('rmse:',np.sqrt(mean_squared_error(y_test,y_pred_bayes)))
print('r^2:',r2_score(y_test,y_pred_bayes))
print('accuracy score:',accuracy_score(y_test, y_pred_bayes))

"""dari hasil yang didapat dengan dataset yang sudah saya bersihkan paling baik menggunakan RandomForestClassifier dengan score 0.8646128981393882"""