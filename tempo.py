if st.button('Table with correlation matrix values'):
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
