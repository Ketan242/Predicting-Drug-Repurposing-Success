# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 19:20:54 2024

@author: Ketan Inamdar
"""
#PROJECT - To analyze the given drug dataset and apply various data exploration and machine learning techniques to predict drug repurposing success.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score

df = pd.read_excel("C:/Users/Ketan Inamdar/Desktop/pharm d/Simulated_Dataset_Drug_Repurposing.xlsx")
print(df)


#1. Data Exploration
#a. Analyze the Distribution of Drug Properties
continuous_vars_1 = ['IC50', 'EC50', 'Molecular_Weight', 'LogP', 'Solubility', 'Protein_Binding', 'AlogP']
fig, axes = plt.subplots(nrows=len(continuous_vars_1), ncols=2, figsize=(15, 10))
for i, var in enumerate(continuous_vars_1):
    sns.histplot(df[var], kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f'Histogram of {var}')
    sns.boxplot(x=df[var], ax=axes[i, 1])
    axes[i, 1].set_title(f'Box Plot of {var}')
plt.tight_layout()
plt.show()

continuous_vars_2 = ['Heavy_Atom_Count', 'Complexity', 
                     'Hydrophilic_Lipophilic_Balance', 'Polarizability', 'Refractivity', 'pKa', 'Isoelectric_Point' ]
fig, axes = plt.subplots(nrows=len(continuous_vars_2), ncols=2, figsize=(15, 10))
for i, var in enumerate(continuous_vars_2):
    sns.histplot(df[var], kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f'Histogram of {var}')
    sns.boxplot(x=df[var], ax=axes[i, 1])
    axes[i, 1].set_title(f'Box Plot of {var}')
plt.tight_layout()
plt.show()

Skewness = df[['IC50', 'EC50', 'LogP', 'AlogP', 'Molecular_Weight','Solubility', 'Protein_Binding', 
               'Number_of_Rotatable_Bonds', 'Number_of_H_Bond_Donors', 'Number_of_H_Bond_Acceptors', 
               'TPSA', 'Aromatic_Rings', 'Heavy_Atom_Count', 'Complexity', 'Hydrophilic_Lipophilic_Balance', 
               'Polarizability', 'Refractivity', 'pKa', 'Isoelectric_Point']].skew()
print(Skewness)

#B. Visualize Correlations Between Drug Properties and Their Existing Use
Missing_values = df.isnull().sum()
print(Missing_values)
df.fillna(df.mean(numeric_only=True), inplace=True)
print('Missing Values After Handling')
print(df.isnull().sum())
def detect_outliers_iqr(df, column):
    q_1= df['EC50'].quantile(0.25)
    q_3= df['EC50'].quantile(0.75)
    iqr_1= q_3 - q_1
    lower_bound = q_1 - 1.5 * iqr_1
    print(lower_bound)
    upper_bound = q_3 + 1.5 * iqr_1
    print(upper_bound)
    df['Outlier'] = (df['EC50'] < q_1 - 1.5*iqr_1) | (df['EC50'] > q_3 + 1.5 * iqr_1)
    print(df['Outlier'])
    df['EC50_capped'] = np.clip( df['EC50'],lower_bound, upper_bound)
    print(df['EC50_capped'])
    return df[(df['EC50'] < (q_1 - 1.5 * iqr_1)) | (df['EC50'] > (q_3 + 1.5 * iqr_1))]

df_encoded = pd.get_dummies(df, drop_first=True)
df_encoded = df_encoded.astype(int)
print(df_encoded)
print("Correlation matrix:")
print(df_encoded.corr())
plt.figure(figsize=(10, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt='.2f') 
plt.title('Correlation Matrix')
plt.show()


#C. Generate Summary Statistics for Continuous Variables
Mean = df[['IC50', 'EC50', 'LogP', 'AlogP', 'Molecular_Weight','Solubility', 'Protein_Binding', 
           'Number_of_Rotatable_Bonds', 'Number_of_H_Bond_Donors', 'Number_of_H_Bond_Acceptors', 
           'TPSA', 'Aromatic_Rings', 'Heavy_Atom_Count', 'Complexity', 'Hydrophilic_Lipophilic_Balance', 
           'Polarizability', 'Refractivity', 'pKa', 'Isoelectric_Point']].mean()
print(Mean)
Median = df[['IC50', 'EC50', 'LogP', 'AlogP', 'Molecular_Weight','Solubility', 'Protein_Binding', 
             'Number_of_Rotatable_Bonds', 'Number_of_H_Bond_Donors', 'Number_of_H_Bond_Acceptors', 
             'TPSA', 'Aromatic_Rings', 'Heavy_Atom_Count', 'Complexity', 'Hydrophilic_Lipophilic_Balance', 
             'Polarizability', 'Refractivity', 'pKa', 'Isoelectric_Point']].median()
print(Median)
Mode = df[['IC50', 'EC50', 'LogP', 'AlogP', 'Molecular_Weight','Solubility', 'Protein_Binding', 
           'Number_of_Rotatable_Bonds', 'Number_of_H_Bond_Donors', 'Number_of_H_Bond_Acceptors', 
           'TPSA', 'Aromatic_Rings', 'Heavy_Atom_Count', 'Complexity', 'Hydrophilic_Lipophilic_Balance', 
           'Polarizability', 'Refractivity', 'pKa', 'Isoelectric_Point']].mode()
print(Mode)
Standard_Deviation = df[['IC50', 'EC50', 'LogP', 'AlogP', 'Molecular_Weight','Solubility', 'Protein_Binding', 
                         'Number_of_Rotatable_Bonds', 'Number_of_H_Bond_Donors', 'Number_of_H_Bond_Acceptors', 
                         'TPSA', 'Aromatic_Rings', 'Heavy_Atom_Count', 'Complexity', 'Hydrophilic_Lipophilic_Balance', 
                         'Polarizability', 'Refractivity', 'pKa', 'Isoelectric_Point']].std()
print(Standard_Deviation)
Q1 = df[['IC50', 'EC50', 'LogP', 'AlogP', 'Molecular_Weight','Solubility', 'Protein_Binding', 
         'Number_of_Rotatable_Bonds', 'Number_of_H_Bond_Donors', 'Number_of_H_Bond_Acceptors', 'TPSA', 
         'Aromatic_Rings', 'Heavy_Atom_Count', 'Complexity', 'Hydrophilic_Lipophilic_Balance', 'Polarizability', 
         'Refractivity', 'pKa', 'Isoelectric_Point']].quantile(0.25)
print(Q1)
Q3 = df[['IC50', 'EC50', 'LogP', 'AlogP', 'Molecular_Weight','Solubility', 'Protein_Binding', 
         'Number_of_Rotatable_Bonds', 'Number_of_H_Bond_Donors', 'Number_of_H_Bond_Acceptors', 
         'TPSA', 'Aromatic_Rings', 'Heavy_Atom_Count', 'Complexity', 'Hydrophilic_Lipophilic_Balance', 
         'Polarizability', 'Refractivity', 'pKa', 'Isoelectric_Point']].quantile(0.75)
print(Q3)
IQR = Q3 - Q1
print(IQR)


#2. Missing Value Handling
Missing_values = df.isnull().sum()
print(Missing_values)
#Replacing missing values by Mean/Median/Mode
Mean = df[['IC50', 'EC50', 'LogP', 'AlogP', 'Molecular_Weight','Solubility', 'Protein_Binding', 
           'Number_of_Rotatable_Bonds', 'Number_of_H_Bond_Donors', 'Number_of_H_Bond_Acceptors', 'TPSA', 
           'Aromatic_Rings', 'Heavy_Atom_Count', 'Complexity', 'Hydrophilic_Lipophilic_Balance', 'Polarizability', 
           'Refractivity', 'pKa', 'Isoelectric_Point']].mean()
print(Mean)
Median = df[['LogP', 'Molecular_Weight']].median()
print(Median)
df_new = df.fillna({'IC50': Mean['IC50'],'EC50': Mean['EC50'],'LogP': Median['LogP'], 'AlogP': Mean['AlogP'],
                    'Molecular_Weight': Median['Molecular_Weight'], 'Solubility': Mean['Solubility'], 
                    'Protein_Binding': Mean['Protein_Binding'], 'Number_of_Rotatable_Bonds': Mean['Number_of_Rotatable_Bonds'], 'Number_of_H_Bond_Acceptors': Mean['Number_of_H_Bond_Acceptors'], 
                    'TPSA': Mean['TPSA'], 'Aromatic_Rings': Mean['Aromatic_Rings'], 'Heavy_Atom_Count': Mean['Heavy_Atom_Count'], 'Complexity': Mean['Complexity'], 
                    'Hydrophilic_Lipophilic_Balance': Mean['Hydrophilic_Lipophilic_Balance'], 'Polarizability': Mean['Polarizability'], 
                    'Refractivity': Mean['Refractivity'],  'pKa': Mean['pKa'], 'Isoelectric_Point': Mean['Isoelectric_Point']})
print(df_new)


#3. Outlier Detection
Q1 = df_new[['IC50', 'EC50', 'LogP', 'AlogP', 'Molecular_Weight','Solubility', 'Protein_Binding', 'Number_of_Rotatable_Bonds', 'Number_of_H_Bond_Donors', 'Number_of_H_Bond_Acceptors', 'TPSA', 'Aromatic_Rings', 'Heavy_Atom_Count', 'Complexity', 'Hydrophilic_Lipophilic_Balance', 'Polarizability', 'Refractivity', 'pKa', 'Isoelectric_Point']].quantile(0.25)
print(Q1)
Q3 = df_new[['IC50', 'EC50', 'LogP', 'AlogP', 'Molecular_Weight','Solubility', 'Protein_Binding', 'Number_of_Rotatable_Bonds', 'Number_of_H_Bond_Donors', 'Number_of_H_Bond_Acceptors', 'TPSA', 'Aromatic_Rings', 'Heavy_Atom_Count', 'Complexity', 'Hydrophilic_Lipophilic_Balance', 'Polarizability', 'Refractivity', 'pKa', 'Isoelectric_Point']].quantile(0.75)
print(Q3)
IQR = Q3 - Q1
print(IQR)
#Detecting and Replacing outliers
q1 = df['LogP'].quantile(0.25)
print(q1)
q3 = df['LogP'].quantile(0.75)
print(q3)
iqr = q3 - q1
print(iqr)
lower_bound = q1 - 1.5 * iqr
print(lower_bound)
upper_bound = q3 + 1.5 * iqr
print(upper_bound)
df['Outlier'] = (df['LogP'] < q1 - 1.5*iqr) | (df['LogP'] > q3 + 1.5 * iqr)
print(df['Outlier'])
df['LogP_capped'] = np.clip( df['LogP'],lower_bound, upper_bound)
print(df['LogP_capped'])
q1 = df['Molecular_Weight'].quantile(0.25)
print(q1)
q3 = df['Molecular_Weight'].quantile(0.75)
print(q3)
iqr = q3 - q1
print(iqr)
lower_bound = q1 - 1.5 * iqr
print(lower_bound)
upper_bound = q3 + 1.5 * iqr
print(upper_bound)
df['Outlier'] = (df['Molecular_Weight'] < q1 - 1.5*iqr) | (df['Molecular_Weight'] > q3 + 1.5 * iqr)
print(df['Outlier'])
df['Molecular_Weight_capped'] = np.clip( df['Molecular_Weight'],lower_bound, upper_bound)
print(df['Molecular_Weight_capped'])


#4. Modeling Techniques
#A. Linear Regression to Predict Success Probability
df_1 = pd.get_dummies(df, drop_first=True)
df_1 = df_1.astype(int)
print(df_1)
df_1.head()
X = df_1[['Drug_Class_Antiviral', 'Drug_Class_Antibiotic', 
          'Target_Organ_Kidney', 'Target_Organ_Heart', 
          'Indication_Inflammation','Indication_Neurological', 'Indication_Pain' ,
          'Molecular_Weight', 'Protein_Binding', 'Solubility', 'Number_of_Rotatable_Bonds', 
          'Number_of_H_Bond_Donors', 'Number_of_H_Bond_Acceptors', 'TPSA', 'Aromatic_Rings', 
          'Heavy_Atom_Count', 'Complexity', 'Hydrophilic_Lipophilic_Balance', 'Polarizability', 
          'Refractivity', 'pKa', 'Isoelectric_Point' ]]
Y = df_1[['EC50']]

X = X.dropna()
Y = Y.loc[X.index]

print("Missing Valeus in X and Y")
print("X missing values")
print(X.isnull().sum())
print("Y missing values:")

model = LinearRegression()
model.fit(X, Y)
Y_pred = model.predict(X)

print(f'Mean Squred Error: {mean_squared_error(Y, Y_pred)}')
print(f'R2 Score: {r2_score(Y,Y_pred)}')
print(f'Slope(coefficient): {model.coef_[0]}')
print(f'Interept: {model.intercept_}')

#B. Logistic Regression to Classify Drug Repurposing Success
df_1 = pd.get_dummies(df, columns=['Drug_Class', 'Target_Organ', 'Indication'], drop_first=True) 
numeric_cols = df_1.select_dtypes(include=['number']).columns
df_1[numeric_cols] = df_1[numeric_cols].astype(int) 
X = df_1[['Drug_Class_Antiviral', 'Drug_Class_Antibiotic', 
          'Target_Organ_Kidney', 'Target_Organ_Heart', 
          'Indication_Inflammation','Indication_Neurological', 'Indication_Pain' ,
          'Molecular_Weight', 'Protein_Binding', 'Solubility', 'Number_of_Rotatable_Bonds', 
          'Number_of_H_Bond_Donors', 'Number_of_H_Bond_Acceptors', 'TPSA', 'Aromatic_Rings', 
          'Heavy_Atom_Count', 'Complexity', 'Hydrophilic_Lipophilic_Balance', 'Polarizability', 
          'Refractivity', 'pKa', 'Isoelectric_Point' ]]
Y = df_1[['Success_Status']]  

logistic_model = LogisticRegression()

logistic_model.fit(X, Y)

logistic_predictions = logistic_model.predict(X)
print('Logistic Regression Predictions:', logistic_predictions)

logistic_accuracy = accuracy_score(Y, logistic_predictions)
print(f'Logistic Regression Accuracy: {logistic_accuracy * 100:.2f}%')

#C. Decision Tree for Interpretability on Drug Features
df_1 = pd.get_dummies(df, columns=['Drug_Class', 'Target_Organ', 'Indication'], drop_first=True) 
numeric_cols = df_1.select_dtypes(include=['number']).columns
df_1[numeric_cols] = df_1[numeric_cols].astype(int) 
X = df_1[['Drug_Class_Antiviral', 'Drug_Class_Antibiotic', 
          'Target_Organ_Kidney', 'Target_Organ_Heart', 
          'Indication_Inflammation','Indication_Neurological', 'Indication_Pain' ,
          'Molecular_Weight', 'Protein_Binding', 'Solubility', 'Number_of_Rotatable_Bonds', 
          'Number_of_H_Bond_Donors', 'Number_of_H_Bond_Acceptors', 'TPSA', 'Aromatic_Rings', 
          'Heavy_Atom_Count', 'Complexity', 'Hydrophilic_Lipophilic_Balance', 'Polarizability', 
          'Refractivity', 'pKa', 'Isoelectric_Point' ]]
Y = df_1[['Success_Status']]

logistic_model = LogisticRegression()
logistic_model.fit(X, Y)
logistic_predictions = logistic_model.predict(X)
print('Logistic Regression Predictions:', logistic_predictions)
logistic_accuracy = accuracy_score(Y, logistic_predictions)
print(f'Logistic Regression Accuracy: {logistic_accuracy * 100:.2f}%')

decision_tree_model = DecisionTreeClassifier(min_samples_leaf=5, max_features=10, random_state=42)
decision_tree_model.fit(X, Y)

tree_predictions = decision_tree_model.predict(X)
print('Decision Tree Predictions:', tree_predictions)

tree_accuracy = accuracy_score(Y, tree_predictions)
print(f'Decision Tree Accuracy: {tree_accuracy * 100:.2f}%')

#D. K-Means Clustering to Group Drugs Based on Structural Similarities
df_1 = pd.get_dummies(df, columns = [ 'Drug_Class', 'Target_Organ', 'Indication'], drop_first=True)
numeric_cols = df_1.select_dtypes(include=['number']).columns
df_1[numeric_cols] = df_1[numeric_cols].astype(int) 
print(df_1)
df_1.head()
features = df_1.drop(columns=['Success_Status', 'Drug_Name'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=2, random_state=45)
df_1['Cluster'] = kmeans.fit_predict(scaled_data)
silhouette_avg = silhouette_score(scaled_data, df_1['Cluster'])
print(f"\nSilhouette Score : {silhouette_avg:.3f}")

silhouette_avg = silhouette_score(scaled_data, df_1['Cluster'])
print(f"\nSilhoutte Score : {silhouette_avg:.3f}")

print("\nClustered User Demographics Data:")
print(df_1)

def success_status(Cluster) :
    if Cluster == 0:
        return['Success']
    elif Cluster == 1:
        return['Failure']
    else:
        return[]
    
    df_1['Success_Status'] = df_1['Cluster'].apply(success_status)
    print("\nSuccess_Status")
    print(df_1[[ 'Molecular_Weight', 'Protein_Binding', 'Solubility', 'Number_of_Rotatable_Bonds', 
     'Number_of_H_Bond_Donors', 'Number_of_H_Bond_Acceptors', 'TPSA', 'Aromatic_Rings', 
     'Heavy_Atom_Count', 'Complexity', 'Hydrophilic_Lipophilic_Balance', 'Polarizability', 
     'Refractivity', 'pKa', 'Isoelectric_Point', 'Success_Status']])














