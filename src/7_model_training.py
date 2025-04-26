import os, multiprocessing, pickle, json, sys, random
sys.path.append(".")
sys.path.append("..")
from multiprocessing import Pool
from glob import glob
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
from xgboost import XGBClassifier
# from xgboost import xgb 
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE #,ADASYN
from collections import Counter
import numpy as np
import warnings
import logging
warnings.filterwarnings("ignore", category=UserWarning)  # Adjust the category as needed
logging.disable(logging.CRITICAL)
PARAM_GRID_DT = {
    'splitter': ["best", "random"],
    'criterion': ["gini", "entropy"],
    'min_samples_split': [2, random.uniform(0.0, 1.0)]
}

PARAM_GRID_RF = {
 'criterion':['entropy','gini'],
 'n_estimators': [50, random.randint(50, 100)] 
}


PARAM_GRID_XGB = {
    'n_estimators': np.linspace(50, 150, 3, dtype=int).tolist()  
}

PARAM_GRID_CAT = {
            'loss_function': ['Logloss', 'CrossEntropy']
            ,'bootstrap_type': ['Bernoulli', 'Bayesian', 'Poisson', 'No']
            ,'n_estimators': [50, random.randint(50, 100)] 
            }

PARAM_GRID_LGBM = {
            'n_estimators': [50, random.randint(50, 100)],
            'force_row_wise': [True],
            }

PARAM_GRID_ADA = {
            'n_estimators': [50, random.randint(50, 100)] 
            }


"""
    Functionality: Loads and preprocesses the project data from a CSV file. It filters specific columns for feature and target extraction.

    Expected Input:
    project (str): The name of the project, used to locate the corresponding dataset file.
    threshold (float, optional): A threshold value that could be used for data filtering or threshold-based feature selection, if implemented.

    Expected Output: 
    x (DataFrame): A DataFrame containing all the feature columns extracted and processed from the dataset.
    y (Series): A Series containing the target variable extracted from the dataset.
"""
def load_data(project, threshold=0.5):
    # load project data
    reusable_clone_df = pd.read_csv('../data/clones/%s_raw_dataset.csv' % (project))
    reusable_clone_df = reusable_clone_df[['cnt_distinct_contributors', 'CountPath', 'CountOutput', 'Essential', 'cnt_group_followers', 
            'CountLineComment', 'CountStmtDecl', 'path_jaccard_similarity', 'CountInput', 'cnt_group_paras', 'CountLine', 'CountLineBlank',
            'is_reusable']]
    x = reusable_clone_df.drop('is_reusable', axis=1)
    y = reusable_clone_df['is_reusable']
    return x, y


"""
    Functionality: Conducts a grid search to find the best parameters for the specified model using cross-validation. 
                   The function uses multiple scoring metrics and returns both the grid search result object and the best model.

    Expected Input:
    x_train (DataFrame): The feature set used for training the model.
    y_train (Series): The target variable used for training the model.
    model (estimator): The machine learning model for which the grid search is to be performed.
    model_paras (dict): A dictionary containing the parameter grid over which the grid search will iterate.

    Expected Output:
    grid_search (GridSearchCV object): The GridSearchCV instance after fitting it to the training data. 
                                      This object contains results and configuration details of the grid search.
    best_model (estimator): The best model obtained from the grid search, fitted with the optimal parameters found.
"""
def grid_search(x_train, y_train, model, model_paras):
    #rcv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1) # 10 * 10 repeated k-fold cv
    rcv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)  # 10 * 10 repeated k-fold cv

    # Multiple metrics can be specified in a dictionary
    scoring = {
        'AUC': 'roc_auc',
        #'Accuracy': make_scorer(accuracy_score),
        #'F1': 'f1',
        #'Recall': 'recall',
        #'Precision': 'precision',
    }

    grid_search = GridSearchCV(estimator=model #  
                               , param_grid=model_paras   
                               , cv=rcv  
                               , scoring=scoring 
                               , refit='AUC'
                               , n_jobs=5
                               , verbose = 0
                               )
    grid_search.fit(x_train.values, y_train)
    # directly used the returned best_estimator model from cross-validation to predict testing datasets with roc_auc_score provided by cross-validation
    best_model = grid_search.best_estimator_
    return grid_search, best_model


"""
    Functionality: Fine-tunes a machine learning model using SMOTE for handling class imbalance and GridSearchCV for parameter optimization. 
                   It assesses the model performance using several metrics: AUC, micro-averaged precision, micro-averaged recall, and MCC.

    Expected Input:
    project (str): The name of the project, used to identify and load the dataset.
    model (estimator): The machine learning model to be fine-tuned.
    model_paras (dict): A dictionary specifying the parameter grid for GridSearchCV.
    threshold (float): A threshold value that might be used for initial data filtering or within the model training process.

    Expected Output:
    Returns a tuple containing:
    - Classifier name (str): A string representing the type of the classifier used.
    - AUC score (float): The area under the ROC curve for the test set.
    - Micro-averaged precision (float): The micro-averaged precision score for the test set.
    - Micro-averaged recall (float): The micro-averaged recall score for the test set.
    - MCC score (float): The Matthews correlation coefficient for the test set.
"""
def fine_tune(project, model, model_paras, threshold):
    
    x, y = load_data(project, threshold)
    #x, y = sm.fit_resample(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # counter1 = Counter(y_train)
    sm = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=min(5, y_train.value_counts().min() - 1))
    x_train, y_train = sm.fit_resample(x_train, y_train)
    #counter2 = Counter(y_train_sm)
    grid_search_model, best_model = grid_search(x_train, y_train, model, model_paras)
    
    y_pred = grid_search_model.predict(x_test)
    auc_score = grid_search_model.score(x_test, y_test)
    # Compute micro-averaging precision and recall
    micro_precision = precision_score(y_test, y_pred, average='micro')
    micro_recall = recall_score(y_test, y_pred, average='micro')
    mcc_score = matthews_corrcoef(y_test, y_pred)
    # save auc
    return type(model).__name__, auc_score, micro_precision, micro_recall, mcc_score
    # load tuple
    # (grid_search_model, x_train, y_train, x_test, y_test, auc_score) = pickle.load(open("tuple_model.pkl", 'rb'))


if __name__=='__main__':
    if len(sys.argv) > 1:
        project = sys.argv[1]

    model_dict = {
        'DecisionTrieeClassifier': PARAM_GRID_DT,
        #'RandomForestClassifier': PARAM_GRID_RF,
        #'CatBoostClassifier': PARAM_GRID_CAT,
        #'XGBClassifier': PARAM_GRID_XGB,
        #'LGBMClassifier': PARAM_GRID_LGBM,
        #'AdaBoostClassifier': PARAM_GRID_ADA
    }

    dataset_files = glob('../data/clones/*_raw_dataset.csv')
    for classifier in model_dict:
        model = DecisionTreeClassifier(random_state=0)
        if classifier == 'DecisionTreeClassifier':
            model = DecisionTreeClassifier(random_state=0)
        elif classifier == 'RandomForestClassifier':
            model = RandomForestClassifier(oob_score=True, random_state=42)  # n_estimators=10000
        elif classifier == 'XGBClassifier':
            model = XGBClassifier() 
        elif classifier == 'CatBoostClassifier':
            # model = CatBoostClassifier(bootstrap_type='Bayesian', verbose=0) 
            model = CatBoostClassifier()
        elif classifier == 'LGBMClassifier':
            model = LGBMClassifier(objective='binary', random_state=5, verbosity=-1)
        elif classifier == 'AdaBoostClassifier':
            model = AdaBoostClassifier(random_state=0)
        elif classifier == 'SVC':
            model = svm.SVC(random_state=42)
        model_paras = model_dict[classifier]
        classifier, auc_score, micro_precision, micro_recall, mcc_score = fine_tune(project, model, model_paras, 0.5)
        perf_df = pd.DataFrame(columns=['project', 'classifier', 'auc', 'auc_score', 'micro_precision', 'micro_recall', 'mcc_score'])
        perf_df.loc[len(perf_df)] = [project, classifier, auc_score, auc_score, micro_precision, micro_recall, mcc_score]
        perf_df.to_csv('result_AUC.csv', mode='a', index=False, header=False)
    print("done evaluation successfully !")
