'''
Created on Nov 2, 2015

@author: gaurav
'''

from Utility import loadFile, getDataFromFile, dir_path, training_file, test_file
from pandas.compat import OrderedDefaultdict

def parseTrainingData(training_data):
    '''
        Given the training data, separates the context, pos tags and the ner tags 
        into separate lists while maintaining order
    '''
    context = []
    pos     = []
    ner     = []
    
    #1st line is context, 2nd is POS and 3rd is named entity category
    for i in xrange(len(training_data)):
        if i % 3 == 0:
            context += training_data[i].strip().split('\t')
        elif i % 3 == 1:
            pos += training_data[i].strip().split('\t')
        else:
            ner += training_data[i].strip().split('\t')
    
    return context, pos, ner

def processTrainingData(context, pos, ner):
    '''
        Given the context, part of speeches and the named entity tags, 
        create a data structure that simplifies look ups
    '''
    training_data = OrderedDefaultdict(dict);
    
    for i in xrange(len(context)):
        training_data[context[i]] = {'POS' : pos[i], 'NE' : ner[i]}
    
    return training_data

def getTrainingData():
    '''
        Loads the training data from the appropriate directory
    '''
    #Load the data present in the training file
    f = loadFile(dir_path + training_file)
    training_data = getDataFromFile(f)
    context, pos, ner = parseTrainingData(training_data)
    training_data = processTrainingData(context, pos, ner)    

def main():
    getTrainingData()

if __name__ == '__main__':
    main()
    