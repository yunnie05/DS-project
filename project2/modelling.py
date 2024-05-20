import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from copy import deepcopy

#Carregando o dataset
hcc = pd.read_csv("hcc_dataset.csv", sep=",")

#Ajetiando os valores nulos
hcc.replace(np.nan, 'None', inplace=True) #Na tabela tem células com o valor None que ele interpreta como um np.nan, então precisamos garantir que ele vai entender isso como um valor válido
hcc.replace('?', np.nan, inplace=True) #As células vazias possuem uma '?', então aqui dizemos que essas células sõa NaN

for column in hcc.columns:
    #Convertendo os valores que são numéricos para float
    if hcc[column].dtype == 'object':
        try:
            if hcc[column].apply(lambda x: pd.to_numeric(x, errors='coerce')).notnull().any() and column != 'Nodules':
                hcc[column] = pd.to_numeric(hcc[column], errors='coerce')
        except ValueError:
            try:
                hcc[column]
            except ValueError:
                pass

    #Tratando os valores de texto -  colocando para UPPERCASE e colocando algum valor nas células vazias
    if(hcc[column].dtype == 'object'):
        hcc[column] = hcc[column].str.upper()
        value = hcc[column].value_counts().idxmax() # para variáveis categóricas colocamos o valor mais frequente
    else:
        value = hcc[column].mean() # para valores numéricos colocamos a média
    hcc[column].replace(np.nan, value, inplace=True)
    
# Normalize numeric values
scaler = MinMaxScaler()
numeric_columns = hcc.select_dtypes(include=['float64', 'int64']).columns
hcc[numeric_columns] = scaler.fit_transform(hcc[numeric_columns])

# Display the first few rows to verify normalization
hcc.head()
