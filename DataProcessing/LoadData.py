'''
Created on Nov 2, 2015

@author: gaurav
'''

from DataProcessing.Utilities import loadFile, getDataFromFile, dir_path, training_file, test_file, pprint
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

def parseTrainingDataHMM(training_data):
    '''
        Given the training data, separates the context, pos tags and the ner tags
        into separate *nested* lists while maintaining order
    '''
    context = []
    pos     = []
    ner     = []

    #1st line is context, 2nd is POS and 3rd is named entity category
    for i in xrange(len(training_data)):
        if i % 3 == 0:
            context.append(training_data[i].strip().split('\t'))
        elif i % 3 == 1:
            pos.append(training_data[i].strip().split('\t'))
        else:
            ner.append(training_data[i].strip().split('\t'))

    return context, pos, ner

def processTrainingData(context, pos, ner):
    '''
        Given the context, part of speeches and the named entity tags, 
        create a data structure that simplifies look ups
    '''
    training_data = OrderedDefaultdict(dict);

    for i in xrange(len(context)):
        #if ner[i] != "": Reduced the accuracy down to 34%
            training_data[context[i]] = {'POS' : pos[i], 'NE' : ner[i]}

    pprint(training_data)
    return training_data

def getTrainingData(HMM=False):
    '''
        Loads the training data from the appropriate directory
    '''
    #Load the data present in the training file
    f = loadFile(dir_path + training_file)
    training_data = getDataFromFile(f)
    if not HMM:
        context, pos, ner = parseTrainingData(training_data)
        training_data = processTrainingData(context, pos, ner)
        return training_data
    else:
        context, pos, ner = parseTrainingDataHMM(training_data)
        return context, pos, ner

def parseTestData(test_data):
    '''
        Parses the test data to extract context, pos and index
    '''
    context = []
    pos     = []
    index   = []
    
    #1st line is context, 2nd is POS and 3rd is index
    for i in xrange(len(test_data)):
        if i % 3 == 0:
            context += test_data[i].strip().split('\t')
        elif i % 3 == 1:
            pos += test_data[i].strip().split('\t')
        else:
            index += test_data[i].strip().split('\t')
    
    return context, pos, index

def parseTestDataHMM(test_data):
    '''
        Parses the test data to extract *nested* context, pos and index
    '''
    context = []
    pos     = []
    index   = []

    #1st line is context, 2nd is POS and 3rd is index
    for i in xrange(len(test_data)):
        if i % 3 == 0:
            context.append(test_data[i].strip().split('\t'))
        elif i % 3 == 1:
            pos.append(test_data[i].strip().split('\t'))
        else:
            index.append(test_data[i].strip().split(' '))  # this uses spaces, not tabs...

    return context, pos, index

def getTestData(HMM=False):
    '''
        Loads the test data from the appropriate directory
    '''
    #Load the data present in the test file
    f = loadFile(dir_path + test_file)
    test_data = getDataFromFile(f)
    if not HMM:
        context, pos, index = parseTestData(test_data)
        return context, pos, index
    else:
        context, pos, index = parseTestDataHMM(test_data)
        return context, pos, index