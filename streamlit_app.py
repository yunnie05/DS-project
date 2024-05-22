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
    hcc.replace(np.nan, 'None', inplace=True) 
    hcc.replace('?', np.nan, inplace=True) 
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

st.write("Now that 'None' no longer being considered a missing value we need to understand which variables correlate the most") 

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
#plt.show()

from scipy.stats import chi2_contingency

# Define Cramer's V function
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / (min(k-1, r-1)))


# Specific correlation of numerical variables with the target variable
target_corr = correlation_matrix['class_encoded'].drop('class_encoded')
st.write("## In order to understand the most relevants variables, let's use the correlation matrix and the Cramer's method. ")
option = st.selectbox(
    'Select one:',
    ('Correlation Matrix', "Cramer's method")
)

if st.button('Show Table'):
    if option == 'Correlation Matrix':
        st.write(target_corr.sort_values(ascending=False))
        st.write('### -Positive Correlations')
        st.write(' #### Hemoglobin (0.292), Iron (0.291), Albumin (0.288):')
        st.write("These three show a moderate positive correlation with the target variable. As the levels of Hemoglobin, Iron, and Albumin increase, there is a corresponding moderate increase in the target variable. This suggests that higher levels of these biomarkers are associated with a stronger presence or higher level of the target condition.")
        st.write('### -Low Positive Correlations')
        st.write('#### Sat (0.043), MCV (0.042):')
        st.write('These values suggest very weak positive correlations. Changes in Saturation and Mean Corpuscular Volume have a slight association with changes in the target variable, but these relationships are weak and might not be significant in practical terms.')
        st.write('### -No or Negligible Correlation')
        st.write('#### AFP (0.002), ALT (-0.006):')
        st.write('These correlations are close to zero, indicating no meaningful relationship with the target variable. Changes in Alpha-Fetoprotein and Alanine Aminotransferase levels do not significantly affect the target variable.')
        st.write('### -Negative Correlations')
        st.write('#### Mild Negative: ')
        st.write('Minor negative correlations (like with TP, Grams/day, and Nodules) indicate that as these values increase, there is a slight decrease in the target variable. These are not strong correlations but suggest a mild inverse relationship.')
        st.write('#### Moderate to Strong Negative')
        st.write('##### Major_Dim (-0.194), INR (-0.202), Total_Bil (-0.224), Dir_Bil (-0.265), ALP (-0.294), Ferritin (-0.321): ')
        st.write('-These show stronger negative correlations. Especially noteworthy are Ferritin and ALP (Alkaline Phosphatase), where higher levels are strongly associated with a decrease in the target variable. This might suggest that increased levels of these biomarkers are protective, or inversely related to the severity or occurrence of the target condition.')
        st.write('-Variables like AST, GGT, Platelets, and particularly INR and bilirubin forms (Direct Bilirubin, Total Bilirubin) also display negative correlations, reinforcing the notion that these factors are inversely associated with the condition being studied.')
        st.write('## Implications for correlation matrixes:')
        st.write('Hemoglobin, Iron, and Albumin may be significant predictors for the positive side of the target condition, while Ferritin, ALP, and bilirubin levels are significant on the negative side.')

    elif option == "Cramer's method":
         # Assuming 'hcc' is your DataFrame
        categorical_cols = hcc.select_dtypes(include=['object', 'category'])
        categorical_cols = categorical_cols.drop(columns=['Class'])
        
        cramers = {}
        for column in categorical_cols:
            contingency_table = pd.crosstab(hcc[column], hcc['Class'])
            cramers_v_value = cramers_v(contingency_table.to_numpy())
            cramers[column] = cramers_v_value # Prepare data for display
            
        cramers_df = pd.DataFrame(list(cramers.items()), columns=['Variable', 'Cramér\'s V'])
        cramers_df.sort_values(by='Cramér\'s V', ascending=False, inplace=True) # Display results in a table
        st.table(cramers_df)
        st.write('### -Strong Associations')
        st.write('#### PS (Performance Status) (0.396):')
        st.write("This suggests a moderately strong association. PS typically reflects how a patient's daily living abilities are affected by their condition, indicating that it could be a significant factor in the progression or severity of the disease.")
        st.write('#### Encephalopathy (0.354):')
        st.write(" This also indicates a moderate association. In the context of liver diseases, for example, encephalopathy can significantly affect patient outcomes, aligning with a higher Cramer's V value.")
        st.write('### -Moderate Associations')
        st.write('#### Symptoms (0.283):')
        st.write('This value suggests a fair association with the condition, indicating that the presence or type of symptoms can reliably vary with the disease state or severity.')
        st.write('#### Metastasis (0.234) and PVT (Portal Vein Thrombosis) (0.198):')
        st.write('These values indicate a moderate association. Metastasis points towards the stage of cancer, while PVT is often associated with liver diseases, both critical in determining disease outcomes.')
        st.write('### -Weak Associations')
        st.write('#### Ascites (0.119), HCVAb (Hepatitis C Virus Antibodies) (0.099), and Diabetes (0.099):')
        st.write('These variables show weak associations. Although these factors are clinically relevant, their individual predictive power regarding the target condition is limited.')
        st.write('#### Endemic (0.079) to HBeAg (Hepatitis B e Antigen) (0.025):')
        st.write('These show very weak associations with the target condition, indicating that while they may contribute to the overall clinical picture, they do so to a lesser extent.')
        st.write('### -Very Weak or No Association')
        st.write('#### Gender (0.023), Cirrhosis (0.016), Spleno (splenomegaly) (0.016) down to Obesity (0.004) and HIV (0.0): ')
        st.write("These show minimal to no measurable association with the target condition. It's notable that some typically significant factors like Cirrhosis show very low association, which might be due to the specific context of the dataset or the outcome being measured.")
        st.write('## Implications for Cramer')
        st.write('- PS and Encephalopathy are notably higher and could be pivotal in analyses or model development. Their strong association might indicate them as primary factors or symptoms in the condition being studied.')
        st.write("- Variables with moderate to weak associations shouldn't be ignored, as their combined effects or interactions with other variables might be significant.")
        st.write("- Low Cramer's V values suggest these variables alone may not be strong predictors for the condition. Including them might add noise or reduce the predictive accuracy of models unless combined intelligently with other data.")




st.write('## In order to have an even better understanding of the data, lets see some graph visualizations, to analyse them:')

gender_count= hcc['Gender'].value_counts(normalize= True)
gender_percentages= gender_count* 100

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{v:.1f}% ({p:.2f}%)'.format(p=pct, v=val)
    return my_format

def create_pie_chart(data, labels):
    plt.figure(figsize=(8, 8))
    plt.pie(data, labels=labels, autopct=autopct_format(data), startangle=90, colors=['blue', 'green'])
    plt.title('Gender Distribution')
    st.pyplot(plt)

gender_count= hcc['Gender'].value_counts(normalize= True)
gender_percentages= gender_count* 100
st.write(gender_percentages) 

# Example data, replace this with your actual data
gender_percentages = [81, 19]
gender_labels = ['Male', 'Female']

st.write("### Here's the Gender Distribution")

if st.button('Show Pie Chart'):
    create_pie_chart(gender_percentages, gender_labels)

st.write("##### This allows us to understand that this dataset has more men than women, here's what we can extract from this information: ")
st.write("- There's an increased power for male data: with fewer data points for women, any conclusions or models developed might not generalize well to female populations. The patterns observed in men may not hold true for women due to biological, behavioral, or social differences.")
st.write("- There's imited Generalizability for Women: with fewer data points for women, any conclusions or models developed might not generalize well to female populations. The patterns observed in men may not hold true for women due to biological, behavioral, or social differences.")
st.write("- There might be gender-specific Insights: we might be able to explore whether certain conditions, behaviors, or outcomes are disproportionately associated with one gender. This could uncover gender-specific risk factors or protective factors.")
st.write('- For example, certain treatments might be more or less effective based on gender, or certain risk factors might be amplified or mitigated.')

st.write("## Before implementing the classification models it's very important to preprocess the data")
if st.button('Know why'):
    st.write("this is the part where we will adress some of the issues observed in the Exploratory fase: imput missing values, transform non-numeric values in numbers, etc...")
st.write("#### Step 1:")
st.write("- Let's start by handling the missing values, for that we will calculate the mean of all values on a numerical column and we will replace the result to the missing value")

#Carregando o dataset
hcc = pd.read_csv("hcc_dataset.csv", sep=",")

#AjeITando os valores nulos
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
                hcc[column] = pd.to_datetime(hcc[column], errors='coerce')
            except ValueError:
                pass

    #Tratando os valores de texto -  colocando para UPPERCASE e colocando algum valor nas células vazias
    if(hcc[column].dtype == 'object'):
        hcc[column] = hcc[column].str.upper()
        value = hcc[column].value_counts().idxmax() # para variáveis categóricas colocamos o valor mais frequente
    else:
        value = hcc[column].mean() # para valores numéricos colocamos a média
    hcc[column].replace(np.nan, value, inplace=True)

hcc.to_csv('imputed_hcc.csv', index=False) # criando um arquivo com esses dados
if st.button('Click here to do that'):
    st.write(hcc.head())

st.write("#### Step 2:")
st.write("Encoding categorical values into numerics Encoding values into numbers to be used in algorithms")
# Criando uma cópia do dataset original para encoding
encoded_hcc = deepcopy(hcc)

# Definindo as colunas com classificação ordinal
columns_classification = {
    'PS': ['ACTIVE', 'RESTRICTED', 'AMBULATORY', 'SELFCARE', 'DISABLED'],
    'Encephalopathy': ['NONE', 'GRADE I/II', 'GRADE III/IV'],
    'Ascites': ['NONE', 'MILD', 'MODERATE/SEVERE']
}

# Transformando as variáveis categóricas em números
for column in encoded_hcc.columns:
    if encoded_hcc[column].dtype == 'object' and column != 'Nodules':  # Verifica se é categórica e não 'Nodules'
        if column in columns_classification:
            # Ensure correct usage of OrdinalEncoder
            categories = [columns_classification[column] + ['UNKNOWN']]  # Adding 'UNKNOWN' to handle unexpected categories
            encoded_hcc[column] = encoded_hcc[column].apply(lambda x: x if x in columns_classification[column] else 'UNKNOWN')
            enc = OrdinalEncoder(categories=categories)
            enc_transform = pd.DataFrame(enc.fit_transform(encoded_hcc[[column]]), columns=[column])
        else:
            enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            enc_transform = enc.fit_transform(encoded_hcc[[column]])
            enc_transform = pd.DataFrame(enc_transform, columns=enc.get_feature_names_out([column]))

        # Tratamento especial para colunas específicas
        if not(column in columns_classification or column == 'Gender' or column == 'Class'):
            enc_transform = enc_transform.filter(regex='_YES$')  # Filtrar apenas colunas com '_YES$'
        elif column == 'Gender':
            enc_transform = enc_transform.filter(regex='_MALE$')  # Considerar apenas '_MALE$'
        elif column == 'Class':
            enc_transform = enc_transform.filter(regex='_LIVES$')  # Considerar apenas '_LIVES$'

        encoded_hcc = encoded_hcc.drop(columns=[column])  # Excluir a coluna original
        encoded_hcc = pd.concat([encoded_hcc, enc_transform], axis=1)  # Concatenar a transformação ao dataset
if st.button('Encode here'):
    st.write(encoded_hcc)
    st.write("As you can see, now the values of PS, for example, range from 0 to 4: 0- ACTIVE, 1 RESTRICTED', 2- AMBULATORY, 3- SELFCARE, 4- DISABLED")

st.write("#### Step 3:")
st.write("Scalling the data: In this part we transform all the number in values from 0 to 1(scalling the data with values between 0 and 1), so that the algorithm doesn't 'privileged', the data with bigger numbers")

colunas_numericas = encoded_hcc.select_dtypes(include=['int', 'float']).columns #seleciona apenas as colunas numéricas
colunas_nao_numericas=encoded_hcc.select_dtypes(exclude=['int', 'float']).columns#exclui as colunas numéricas, para depois concatenarmos com a anterior
scaler=MinMaxScaler()
dados_escalados_numericos=scaler.fit_transform(encoded_hcc[colunas_numericas])#chamamos o MinMaxScaler pra ele transformar os dados, somente das colunas numéricas
dados_escalados_numericos=pd.DataFrame(dados_escalados_numericos,columns=colunas_numericas) ## converte o array numpy dos dados escalados de volta para um DataFrame com os nomes originais das colunas numéricas
dados_escalados = pd.concat([dados_escalados_numericos, encoded_hcc[colunas_nao_numericas]], axis=1)#concatena os dados numéricos escalonadas com os dados das colunas não numéricas
if st.button("Scaled data"):
    st.write(dados_escalados)

st.write("#### Step 4:")
st.write("It's very important to select the most important variables, because this are the ones that will be used on the classification algorithms. This selection was made based on the matrix of correlation and Cremer's methods, shown on the exploratory phase.")
#colunas selecionadas, a partir da relevância medida nas etapas anteriores
colunas_selecionadas=['Hemoglobin','Iron',' Albumin','Sat','MCV','AFP','Ferritin','ALP', 'PS','Symptoms_YES','Ascites','Metastasis_YES','Encephalopathy','Class_LIVES']
dados_selecionados=dados_escalados[colunas_selecionadas].copy()
if st.button('Select'):
    st.write(dados_selecionados)

st.write("## Now the dataset is ready for data modelling")
st.write("Three classification models were built")
st.write("## 1: Decidion Tree Algorithm")


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  precision_score, recall_score, f1_score


def decision_tree(subdataset):
    X = subdataset.drop(columns=["Class_LIVES"])
    y = subdataset["Class_LIVES"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_leaf_nodes=8, criterion="gini", random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy:", accuracy)

    st.write("Classification report")
    st.text(classification_report(y_test, y_pred))

    st.write("Confusion matrix")
    st.text(str(confusion_matrix(y_test, y_pred)))

    auc = roc_auc_score(y_test, y_pred_proba)
    st.write("AUC:", auc)

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC Curve (área = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for DT')
    plt.legend(loc="lower right")
    st.pyplot(plt)

    cm = confusion_matrix(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(j+0.5, i+0.5, f'{cm[i, j]}\nP: {precision[i]:.2f}\nR: {recall[i]:.2f}\nF1: {f1[i]:.2f}',
                     ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.5))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix with Precision, Recall, and F1-Score')
    st.pyplot(plt)

st.write('### Decision Tree Classifier Analysis')

if st.button('Run Decision Tree'):
    decision_tree(dados_selecionados)

st.write("## 2: KNN Algorithm")
st.write("KNN ALGORITHM (from k= 1 to k= 21)")

def knn_analysis(X_train, X_test, y_train, y_test):
    best_accuracy = 0
    best_k = 0

    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    # Displaying the best k and its accuracy
    st.write(f'Best k: {best_k}, with Accuracy: {best_accuracy:.2f}')


st.write('### KNN Classification Analysis')
if st.button('Run KNN Analysis'):
    X = dados_selecionados.drop(columns=['Class_LIVES'])
    y = dados_selecionados['Class_LIVES']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn_analysis(X_train, X_test, y_train, y_test)

st.write("Let's analyse the ROC curves for k values with high accuracies")
def plot_roc_curves(X_train, X_test, y_train, y_test, k_values, accuracy):
    # Initialize lists to store ROC curves and AUCs
    fpr_list = []
    tpr_list = []
    roc_auc_list = []

    # Function to compute ROC curve and AUC for a list of k values
    def compute_roc_auc(k_values):
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            # Predict probabilities for the positive class
            y_pred_prob = knn.predict_proba(X_test)[:, 1]
            # Calculate the ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            # Calculate the AUC
            roc_auc = roc_auc_score(y_test, y_pred_prob)
            # Store the results
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            roc_auc_list.append(roc_auc)

    # Compute ROC and AUC
    compute_roc_auc(k_values)

    # Plot the ROC curves
    plt.figure(figsize=(10, 7))
    for i, k in enumerate(k_values):
        plt.plot(fpr_list[i], tpr_list[i], lw=2, label=f'K={k} (AUC = {roc_auc_list[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for Different K Values (Accuracy= {accuracy})')
    plt.legend(loc="lower right")
    plt.grid(True)
    st.pyplot(plt)


# Assuming 'dados_selecionados' is already defined
X = dados_selecionados.drop(columns=['Class_LIVES'])
y = dados_selecionados['Class_LIVES']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write('### KNN ROC Curves')

k_values_group1 = [4, 9, 15, 18, 20]
k_values_group2 = [10, 11, 12, 14, 16]

# Streamlit selectors for different groups
if st.button('Show ROC Curves for Accuracy 0.70'):
    plot_roc_curves(X_train, X_test, y_train, y_test, k_values_group1, '0.70')

if st.button('Show ROC Curves for Accuracy 0.73'):
    plot_roc_curves(X_train, X_test, y_train, y_test, k_values_group2, '0.73')
    display_confusion_matrix(X, y)

from sklearn.ensemble import RandomForestClassifier

st.write("## 3: Random Forest")


def train_and_evaluate_rf(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    rf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Display results
    st.write(f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')

    # Classification report
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    # Plot the confusion matrix with additional metrics
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            ax.text(j+0.5, i+0.5, f'P: {precision:.2f}\nR: {recall:.2f}\nF1: {f1:.2f}',
                     ha='center', va='center', color='black')

st.write('### Random Forest Classifier Evaluation')


# Separating the predictor variables (X) and the target variable (y)
X = dados_selecionados.drop(columns=['Class_LIVES'])
y = dados_selecionados['Class_LIVES']

if st.button('Evaluate Random Forest'):
    train_and_evaluate_rf(X, y)


def plot_roc_curves(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the models
    knn = KNeighborsClassifier(n_neighbors=12)
    decision_tree = DecisionTreeClassifier(max_leaf_nodes=8, criterion="gini", random_state=0)
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the models
    knn.fit(X_train, y_train)
    decision_tree.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)

    # Make predictions and calculate probabilities
    y_pred_proba_knn = knn.predict_proba(X_test)[:, 1]
    y_pred_proba_dt = decision_tree.predict_proba(X_test)[:, 1]
    y_pred_proba_rf = random_forest.predict_proba(X_test)[:, 1]

    # Calculate ROC curve and AUC
    fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_proba_knn)
    roc_auc_knn = roc_auc_score(y_test, y_pred_proba_knn)
    fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
    roc_auc_dt = roc_auc_score(y_test, y_pred_proba_dt)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
    roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

    # Plot ROC curves
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_knn, tpr_knn, color='blue', lw=2, label=f'KNN (AUC = {roc_auc_knn:.2f})')
    plt.plot(fpr_dt, tpr_dt, color='green', lw=2, label=f'Decision Tree (AUC = {roc_auc_dt:.2f})')
    plt.plot(fpr_rf, tpr_rf, color='red', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    st.pyplot(plt)

st.write("# Let's now compare the 3 algorithms")
st.write('## Model ROC Curves Comparison')

X = dados_selecionados.drop(columns=['Class_LIVES'])
y = dados_selecionados['Class_LIVES']

if st.button('Show ROC Curves'):
    plot_roc_curves(X, y)


st.write("#### From this graphic, it's clear that Random Forest is the best algorithm. Thank you!!")


