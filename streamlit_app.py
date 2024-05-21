import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler


hcc= pd.read_csv("hcc_dataset.csv", sep= ",", na_values= ['?'])
#table = pd.read_csv('comparison_results.csv')

# Use Streamlit to display data
st.title('The Hepatocellular Carcinoma Dataset')
st.write("The analyzed dataset contains clinical records of 165 patients diagnosed with hepatocellular carcinoma (HCC), collected at the Centro Hospitalar e Universitário de Coimbra in Portugal.")
st.write("The main goal of this project is to develop a machine learning pipeline capable of determining the survivability of patients at 1 year after diagnosis.")

st.write("Let's start exploring this data")
st.write(hcc.head())
st.write("As you can see, there are some values missing on this dataset,let's explore it!! Click the button below to see a summary of missing data in each column.")

if st.button('Show Missing Data Summary'):
    missing_data = hcc.isnull().sum().sort_values(ascending=False)
    st.write("### Missing Data Summary")
    st.write("Oops!! As you can see, there's something wrong. There are two colmuns where 'None' is being considered as a missing value and we also noticed that some numbers were categorized as 'objects', we also have to fix this. Let's fix it")
    st.write(missing_data)

if st.button('Click here to fix the dataset'):
    # Then replace NaN with 'None'
    hcc.replace('?', np.nan, inplace=True)  # Replace '?' with NaN first
    #Convertendo as colunas numéricas
    for column in hcc.columns:
    #Convertendo os valores que são numéricos para float
       if hcc[column].dtype == 'object':
          try:
             if hcc[column].apply(lambda x: pd.to_numeric(x, errors='coerce')).notnull().any() and column != 'Nodules': # Pois Nodules é uma categoria apesar de ser definido por números
                 hcc[column] = pd.to_numeric(hcc[column], errors='coerce')
          except ValueError:
             try:
                 hcc[column] = pd.to_datetime(hcc[column], errors='coerce')
             except ValueError:
                 pass
    st.write("The dataset has been fixed. Here's the updated info:")
    st.write(hcc.head())
    st.write("As you can see, now only the rows")

st.write("Now that 'None' no longer being considered a missing value") 

# Separating the numerical columns
numerical_cols = hcc.select_dtypes(include=['float64', 'int64'])
numerical_cols['class_encoded'] = hcc['Class'].astype('category').cat.codes

# Calculating the correlation matrix for numerical variables, including the encoded target variable
correlation_matrix = numerical_cols.corr()

# Enhancing the heatmap visualization
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f',
            annot_kws={"size": 8}, linewidths=0.5, linecolor='gray')
plt.title('Correlation matrix', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Specific correlation of numerical variables with the target variable
target_corr = correlation_matrix['class_encoded'].drop('class_encoded')
st.write("Correlation with the target variable (class):")
st.write("## In order to understand the most relevants variables, let's use the correlation matrix and the Cramer's method. ")
if st.button('Table with correlation matrix values'):
    st.write(target_corr.sort_values(ascending=False))

from scipy.stats import chi2_contingency

# Define Cramer's V function
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / (min(k-1, r-1)))

if st.button("Cramer's method values"):
    # Assuming 'hcc' is your DataFrame
    categorical_cols = hcc.select_dtypes(include=['object', 'category'])
    categorical_cols = categorical_cols.drop(columns=['Class'])

    # Calculate Cramér's V for categorical variables
    cramers = {}
    for column in categorical_cols:
        contingency_table = pd.crosstab(hcc[column], hcc['Class'])
        cramers_v_value = cramers_v(contingency_table.to_numpy())
        cramers[column] = cramers_v_value

    # Prepare data for display
    cramers_df = pd.DataFrame(list(cramers.items()), columns=['Variable', 'Cramér\'s V'])
    cramers_df.sort_values(by='Cramér\'s V', ascending=False, inplace=True)

    # Display results in a table
    st.table(cramers_df)
