'''
Created on Nov 8, 2015

@author: gaurav
'''

from DataProcessing.Utilities import loadFile, getDataFromFile, dir_path, training_file, pprint
from collections import defaultdict
from copy import deepcopy
import csv

def getTrainingData(text_features = False):
    '''
        Loads the training data from the appropriate directory
    '''
    #Load the data present in the training file
    f = loadFile(dir_path + training_file)
    training_data = getDataFromFile(f)
    context_ner, context_pos = parseTrainingData(training_data)
    low_frequency_ner, low_frequency_pos = findLowFrequencyWord(context_ner, context_pos)
    
    if text_features:
        features = findFeaturesForText(low_frequency_ner)
    else:
        features = findFeaturesForPOS(low_frequency_pos, low_frequency_ner)
        
    feature_probabilities = findProbabilityForFeatures(features)
    #pprint(feature_probabilities)
    return feature_probabilities
    
def parseTrainingData(training_data):
    '''
        Given the training data, separates the context and pos tags and the context and ner
        tags into separate dicts while maintaining order
    '''
    context = []
    ner     = []
    pos     = []

    #1st line is context, 2nd is POS and 3rd is named entity category
    for i in xrange(len(training_data)):
        if i % 3 == 0:
            context += training_data[i].strip().split('\t')
        elif i % 3 == 2:
            ner += training_data[i].strip().split('\t')
        else:
            pos += training_data[i].strip().split('\t')

    #Mapping of context to named entities
    context_ner = defaultdict(list)  
    context_pos = defaultdict(list)
      
    for i in xrange(len(context)):
        context_ner[context[i]].append(ner[i])
        
    for i in xrange(len(context)):
        context_pos[context[i]].append(pos[i])    

    return context_ner, context_pos

def findLowFrequencyWord(context_ner, context_pos):
    '''
        Only keeps the low frequency words i.e. words that have count 2 or less
        and that do not have the Named Entity tag O
    '''
    #Finding the low frequency context        
    low_frequency_ner = {}
    low_frequency_pos = {}
    
    for key, value in context_ner.iteritems():
        #if len(value) < 3 and 'O' not in value:
        #if len(value) < 3:

        low_frequency_ner[key] = value
        low_frequency_pos[key] = context_pos[key]
    
    return low_frequency_ner, low_frequency_pos

def findFeaturesForPOS(low_frequency_pos, low_frequency_ner):
    
    probabilities = {"PER":0, "LOC":0, "ORG":0, "MISC":0, "O":0}
    features = {}
    
    for key, value in low_frequency_pos.iteritems():
        for pos in value:
            if pos not in features:
                features[pos] = deepcopy(probabilities)
            
            nets = low_frequency_ner[key]
            
            for net in nets:
                if '-' in net:
                    features[pos][net.split('-')[1]] += 1
                else:
                    features[pos][net] += 1
     
    pprint(features)
    return features
        
def findFeaturesForText(context_ner):
    
    probabilities = {"PER":0, "LOC":0, "ORG":0, "MISC":0, "O":0}
    features = {"upper-case": deepcopy(probabilities),
                "digit": deepcopy(probabilities),
                "contains-digit": deepcopy(probabilities),
                "first-char-upper": deepcopy(probabilities),
                "lower-case": deepcopy(probabilities),
                "other":deepcopy(probabilities)
                }

    for key, values in context_ner.iteritems():
        feature_class = findFeatureClass(key)
        for net in values:
            if '-' in net:
                features[feature_class][net.split('-')[1]] += 1
            else:
                features[feature_class][net] += 1
     
    return features

def findFeatureClass(token):
    '''
        Given a low frequency word, finds the feature class it is most suitable
        for
    '''
    feature_class = "other"
        
    #Digits are always MISC
    if token.isdigit():
        feature_class = "digit"
        
    #Upper case unknown keys is generally an organisation
    elif token.isupper():
        feature_class = "upper-case"
        
    #IF first character is upper case, it is mostly a person or a location
    elif token[0].isupper():
        feature_class = "first-char-upper"
    
    #If the word contains a digit it seems to be MISC most of the time
    elif any([char.isdigit() for char in token]):
        feature_class = "contains-digit"
            
    elif token.islower():
        feature_class = "lower-case"
    
    #If the word contains some weird special characters
    else:
        feature_class = "other"
        
    return feature_class
          
def findProbabilityForFeatures(features):
    '''
        Finds the probabilities of every state for every feature class
    '''
    for _, state_counts in features.iteritems():
        total_count = sum(state_counts.values())
        for key, value in state_counts.iteritems():
            state_counts[key] = value * 1.0 / total_count;
    
    return features

def savePredictionsToCSV(low_frequency_ner):
    '''
        Saves the final test predictions to CSV format in the format needed
        for kaggle
    '''
    f = open(dir_path + 'Training_Test_Data/low_frequency_analysis', "w")
    writer = csv.writer(f)
    
    for key, value in low_frequency_ner.iteritems():
        writer.writerows([[key, ' '.join(value)]])
        
    f.close()
    
def getLowFrequencyWordProbabilities():
    return getTrainingData()

# getTrainingData()
