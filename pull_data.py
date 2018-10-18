# -*- coding: utf-8 -*-
"""
author: Aaron Grzasko

"""

import requests 
import os
from getpass import getpass



def get_credentials():
    """
    function to import kaggle credentials
    or do manual entry 
     """
 
    # try importing python file input_creds.py with dictionary of user/pw
    try:
        from input_creds import CREDENTIALS
    
    # else: manually enter credentials.  Must be run in terminal; won't work in most IDEs  
    except:
        user = getpass('Enter Kaggle user login:')
        pw = getpass('Enter Kaggle password:')
        CREDENTIALS = {'__RequestVerificationToken': '', \
                       'username': user, \
                       'password': pw}  
    return(CREDENTIALS)
        

def download_kaggle_data(url, filenames):
    """
    function to download kaggle data sets:
        url = web address excluding file name 
        filenames = string with single filename, or list of multiple file names
    """
        
    CREDENTIALS = get_credentials()
    
    loginURL = 'https://www.kaggle.com/account/login'
    
    # if single file supplied, convert to list
    if type(filenames) != list:
        filenames = list(filenames)
    
    # loop through specified files
    for file in filenames:
        file_path = os.path.join(url,file)
        
    
        # supply kaggle credentials
        with requests.Session() as c:
            response = c.get(loginURL).text
            AFToken = response[response.index('antiForgeryToken')+19:response.index('isAnonymous: ')-12]
            CREDENTIALS['__RequestVerificationToken']=AFToken
            c.post(loginURL + "?isModal=true&returnUrl=/", data=CREDENTIALS)
            
            # get data
            response = c.get(file_path)
             
            # Iterate through data in chunks, write to csv
            with open(file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=(512 * 1024)):
                    if chunk: 
                        f.write(chunk)
        
 
def check_download_status(filenames):
    
    # if single file supplied, convert to list
    if type(filenames) != list:
        filenames = list(filenames)
    
    # confirm that specified files exist in cwd and are greater than 0 in size.
    for file in filenames:
        size = os.path.getsize(file)
        if os.path.exists(file) and size > 0:
            print('File {} exists and has the following size: {}'.format(file, size))
        else:
            print('There was a problem with file {}'.format(file))
        
        
    

       
def get_titanic():
    """
    function to pull titanic data sets from kaggle.
   
    """    
    # specify parameters
    url = "https://www.kaggle.com/c/titanic/download"
    files = ['train.csv','test.csv']
    
    # download files
    download_kaggle_data(url,files)
    
    # verify download
    check_download_status(files)
    
    
if __name__ == '__main__':
    get_titanic()
    

"""
References to supply kaggle credentials and download files:
    - https://stackoverflow.com/questions/43516982/import-kaggle-csv-from-download-url-to-pandas-dataframe
    - https://stackoverflow.com/questions/50863516/issue-in-extracting-titanic-training-data-from-kaggle-using-jupyter-notebook
    - https://stackoverflow.com/questions/49386920/download-kaggle-dataset-by-using-python
    - https://ramhiser.com/2012/11/23/how-to-download-kaggle-data-with-python-and-requests-dot-py/

"""