# -*- coding: utf-8 -*-
"""
author: Aaron Grzasko


Based on EDA--see EDA.ipynb--we're keeping the following variables:

- Survived
- PClass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked
- Cabin Known

We will drop two missing values in Embarked

We will impute missing values for Age

We need to convert the following categorical variables to dummy number format:
    - Pclass
    - Sex
    - SibSp
    - Parch
    - Embarked



"""

import pandas as pd
from sklearn.preprocessing import Imputer 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pickle


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
 
    
def RF_create(X,y):
    """
    function fits Random Forest models, saves model to disk
    saves feature dataframe and target variable to disk
        X = feature matrix in df form
        y = target vector
    
    """
    
    # initialize imputer using mean imputation
    imp = Imputer(missing_values='NaN',strategy='mean', axis=0)
    
    # initialize RF classifer
    clf = RandomForestClassifier(n_estimators=100, random_state=25)
    
    # build pipeline steps: imputation + model fit
    steps = [('imputation', imp), ('RFClassifier',clf)]
    pipeline = Pipeline(steps)
    
    # create sepate training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=81)
    
    # fit model to data
    rf_model = pipeline.fit(X_train, y_train)
    
        
    # save model and test data to disk
    filename = 'rf_model.pkl'
    
    try: 
        pickle.dump(rf_model, open(filename, 'wb'))
        X_test.to_csv('test_features.csv', index=False)
        y_test.to_csv('test_target.csv', index=False, header='Survived')
        print('The Random Forest model was trained to the data!')
    
    except:
        print("There was a problem saving model information!")
  
    
 
def titanic_RF():
    """
    function explicitly designed to scrub data,
    and build RF classifer on Titanic dataset from Kaggle
    """
    # scrub dataframe for fitting to model
    df = read_data('train.csv')
    df = del_na_rows(df,['Embarked'])    
    df = drop_cols(df,['PassengerId','Name','Ticket','Cabin']) 
    df = mod_categories(df)
    
    # separate out features and target
    X = df.loc[:, df.columns != 'Survived']
    y = df.loc[:,'Survived']
    
    # fit titanic data to RF model
    RF_create(X,y)
    


if __name__ == '__main__':
    
    titanic_RF()
    


"""
References
https://www.kaggle.com/athi94/investigating-imputation-methods
https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

"""