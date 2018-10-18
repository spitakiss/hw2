# -*- coding: utf-8 -*-
"""
author:  Aaron Grzasko
"""
import pickle
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.utils.multiclass import unique_labels 
 

def import_pickled_model(file):
    """
    function to read in pickled model
        file = .pkl file
    """
    try:
        with open(file, 'rb') as f:
            model = pickle.load(f)
    except:
        print('There was an issue reading in the .pkl file.')
    return(model)
        
  
def read_data(file):
    """
    function to read in data
    assumes file is in csv format in working dir
    """
    try:
        df = pd.read_csv(file)
    except:
        print('There was a problem reading the file.')
    return(df)
   

def del_na_rows(df, cols):
    """"
    function deletes rows with missing values
        df: dataframe
        cols: list of columns to search for NAs
    """
    new_df = df.dropna(subset=cols)
    return(new_df)


def drop_cols(df, cols):
    """
    function deletes uneccessary columns
        df: dataframe
        cols = list of unnecessary columns
    """
    
    new_df = df.drop(cols, axis= 1)
    return(new_df)
    

def mod_categories(df):
    """"
    function to convert non-numerical cateogires to binary values
    function will also delete the first binarized category
        df: dataframe
        
    """
    new_df = pd.get_dummies(df, drop_first=True)
    return(new_df)

def impute_missing(df, cols):
    """
    function imputes missing values using mean imputation
        df: dataframe
        cols: list of columns to impute
    """
    imp = Imputer(missing_values='NaN',strategy='mean', axis=0)
    df = imp.fit(df)
    return(df)
    


def make_preds(model, X):
    """
    function takes makes prediction on test set
    and produces text file output of accuracy report
        X: test feature matrix
        y: test target actual output
    """
    try:
        y_pred = model.predict(X)
        return(y_pred)
    except:
        print('There was a problem making predictions.')
        
    
def score(y, y_pred, file):
    """
    function to produce accuracy of model and output classification report to csv
        y = true y values
        y_pred = predicted y values
        file = name of file to export
    """
    try:
        # print accuracy
        acc_report =  metrics.accuracy_score(y, y_pred)
        print('The model accuracy on the test portion of the training set is {}.'.format(acc_report))
        
        
        # export classification report to csv
        labels = unique_labels(y, y_pred)
        precision, recall, f_score, support = metrics.precision_recall_fscore_support(y, y_pred, labels=labels)
        results_pd = pd.DataFrame({"class": labels, "precision": precision,"recall": recall, "f_score": f_score, \
                                   "support": support})
        results_pd.to_csv(file, index=False)
        
        print('Please refer to the {} file for the classifcation metrics.'.format(file))
    
    except:
        print('There was a problem generating the classification metrics report')

    

def titanic_preds():
    
    # make predictions and produce accuracy from subset of training data split into a test portion
    model = import_pickled_model('rf_model.pkl')
    X = read_data('test_features.csv')
    y = read_data('test_target.csv')
    y_pred =  make_preds(model,X)
    score(y, y_pred, 'classification_report.csv')
    
    # make predictons on the test data set provided directly from the kaggle website
    test = read_data('test.csv')
    test_pass_id = pd.Series(test['PassengerId'], name='PassengerId')
    test = drop_cols(test,['PassengerId','Name','Ticket','Cabin']) 
    test = mod_categories(test)
    test = impute_missing(test, ['Fare'])
    test_preds = pd.Series(make_preds(model,X), name='Survived')
    test_output = pd.concat([test_pass_id,test_preds], axis=1)
    test_output.to_csv('preds_kaggle_test_set.csv', index=False)
    

if __name__ == '__main__':
    titanic_preds()
    
'''
References:
-https://stackoverflow.com/questions/45003577/classification-report-in-sklearn
'''
    
    
    