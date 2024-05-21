# Data Science Pipeline for Hepatocellular Carcinoma Dataset
- This project was developed for the EIACD (Elements of Artificial Intelligence and Data Science) course.

## Project Overview
The second practical assignment involves developing a comprehensive data science pipeline, from exploratory data analysis and preprocessing to applying supervised learning techniques for classification and evaluating their performance. Optionally, the project may also explore additional techniques such as clustering, handling missing or imbalanced data, among others, to improve system performance.

## The Hepatocellular Carcinoma Dataset
This project aims to address a real data science use case using the Hepatocellular Carcinoma (HCC) dataset, collected at the Coimbra Hospital and University Center (CHUC) in Portugal. The dataset contains real clinical data of patients diagnosed with HCC. The primary objective is to develop a machine learning pipeline capable of determining the survivability of patients one year after diagnosis (e.g., “lives” or “dies”).

## Test Code
The project includes a separate folder named "test_code" which contains additional scripts and notebooks used for testing and experimenting with different approaches, algorithms, or techniques. These scripts and notebooks are not part of the main project execution but serve as supplementary resources for development and validation purposes.

While the main code file to be executed is DSproject.ipynb, you may find it useful to explore the contents of the "test_code" folder for additional insights or experimentation.


## Libraries and Dependencies
This project uses several libraries and dependencies for data analysis, visualization, preprocessing, and modeling. Below is a list of the main libraries used and their respective functionalities:
### Data Analysis and Visualization Libraries
- NumPy: Supports high-performance arrays and mathematical operations.
- Pandas: Data structures for data manipulation and analysis.
- Matplotlib: Library for creating static, animated, and interactive plots.
- Seaborn: Data visualization library based on Matplotlib, providing a high-level interface.
### Data Preprocessing Libraries
- Scikit-learn (sklearn): Tools for data modeling and preprocessing.
  - OneHotEncoder and OrdinalEncoder for encoding categorical variables
  - MinMaxScaler for data normalization.
### Modeling and Evaluation Libraries
- Scikit-learn (sklearn): Tools for model building and evaluation.
  - DecisionTreeClassifier for classification using decision trees.
  - KNeighborsClassifier for classification using the k-nearest neighbors algorithm.
  - RandomForestClassifier for classification using random forests.
  - train_test_split for splitting the data into training and testing sets.
  - cross_val_score for cross-validation of models
  - Evaluation metrics like accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, and f1_score.
### Other Libraries
- deepcopy: Used to create deep copies of objects.


## Installing Dependencies
To run this project locally, you need to have Python installed on your machine. You can download and install [Python](https://www.python.org/downloads/) from the official Python website.

You can install all the necessary dependencies listed above using the included requirements.txt file. To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```


## Running the Code Locally
To execute the code on your local machine, ensure that all dependencies are installed. Then, follow these steps:

Make sure you have [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/index.html) installed. If not, you can install it using the following command:

```bash
pip install jupyterlab
```
Download the project files and ensure the dataset file hcc_dataset.csv is in the same directory as the notebook DSproject.ipynb.

Open a terminal and navigate to the directory containing the project files.

Start Jupyter Lab by running:

```bash
jupyter lab
```
In the Jupyter Lab interface, open DSproject.ipynb and execute the cells to run the code.


## Running the Code in Google Colab
If you prefer, you can run the code in [Google Colab](https://colab.google/). This allows you to execute the code in a cloud environment without needing to set up anything locally.

Download the project files and upload them to Google Colab.
Make sure to add the dataset file hcc_dataset.csv in the same directory as your notebook.


## Authors
- [Isabela Cartaxo](https://github.com/belacartaxo)
- [Telma Freitas](https://github.com/telmsgiovana)
- [Yunnie Zita](https://github.com/yunnie05)


## Bibliography
### Dataset: 
- Hepatocellular Carcinoma (HCC) from the Coimbra Hospital and University Center
  
### Virtual Assistant:
- Chat GPT

### Articles:
- [Computational intelligence identifies alkaline phosphatase (ALP), alpha-fetoprotein (AFP), and hemoglobin levels as most predictive survival factors for hepatocellular carcinoma - Davide Chicco, Luca Oneto, 2021](https://journals.sagepub.com/doi/10.1177/1460458220984205)
- [A new cluster-based oversampling method for improving survival prediction of hepatocellular carcinoma patients - Miriam Santos, 2015](https://www.sciencedirect.com/science/article/pii/S1532046415002063)
- [Gini Index & Information Gain in Machine Learning](https://www.linkedin.com/pulse/gini-index-information-gain-machine-learning-dhiraj-patra/)

### Video Lessons:
- [Ordinal Encoder with Python Machine Learning(scikit learn)](https://www.youtube.com/watch?v=15uClAVV-rI&t=300s)
- [Decision and classification trees, clearly explained](https://www.youtube.com/watch?v=_L39rN6gz7Y)
- [How to Build your first decion tree in python(scikit-leanr)](https://www.youtube.com/watch?v=YkYpGhsCx4c)
