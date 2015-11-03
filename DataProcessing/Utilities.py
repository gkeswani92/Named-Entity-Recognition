'''
Created on Nov 2, 2015

@author: gaurav
'''

import json
import csv
import os

dir_path      = os.path.dirname(__file__) + '/'
training_file = 'Training_Test_Data/train.txt'
test_file     = 'Training_Test_Data/test.txt'
predictions   = 'Training_Test_Data/predictions.csv'

def loadFile(filepath):
    '''
        Gets an instance of the file depending on the path
    '''
    f = open(filepath, 'r')
    return f

def getDataFromFile(f):
    '''
        Given a file instance, returns the data in the form of list
    '''
    data = f.readlines()
    return data

def pprint(myDict):
    '''
        Pretty print the default dictionary
    '''
    print(json.dumps(myDict, indent = 4))

def savePredictionsToCSV(test_predictions):
    '''
        Saves the final test predictions to CSV format in the format needed
        for kaggle
    '''
    f = open(dir_path + predictions, "w")
    writer = csv.writer(f)
    writer.writerows([['Type','Prediction']])
    
    for key, value in test_predictions.iteritems():
        writer.writerows([[key, ' '.join(value)]])
        
    f.close()
        
    