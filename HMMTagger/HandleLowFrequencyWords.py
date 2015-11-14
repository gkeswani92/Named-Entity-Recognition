'''
Created on Nov 8, 2015

@author: gaurav
'''

from DataProcessing.Utilities import loadFile, getDataFromFile, dir_path, training_file, pprint
from collections import defaultdict
from copy import deepcopy
import json

def getTrainingData(text_features = True):
    '''
        Loads the training data from the appropriate directory
    '''
    #Load the data present in the training file
    f = loadFile(dir_path + training_file)
    training_data = getDataFromFile(f)
    token_ner, token_pos = parseTrainingData(training_data)
    
    feature_type = ""
    state_features = {}

    #Considering only words with count less than 3 for the similarity based classifier
    if text_features:
        low_frequency_token_ner = findLowFrequencyWord(token_ner)
        state_features = findFeaturesForText(low_frequency_token_ner)
        feature_type = "text_features"
    
    #Considering all words for the POS based classifier
    else:
        state_features = findFeaturesForPOS(token_pos, token_ner)
        feature_type = "pos_features"
    
    #Finding the probabilities for the features
    feature_probabilities = findProbabilityForFeatures(state_features)
    saveFeaturesToDisk(feature_probabilities, feature_type)
    pprint(feature_probabilities)
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

def findLowFrequencyWord(context_ner):
    '''
        Only keeps the low frequency words i.e. words that have count 2 or less
        and that do not have the Named Entity tag O
    '''
    #Finding the low frequency context        
    low_frequency_token_ner = {}
    
    for key, value in context_ner.iteritems():
        #if len(value) < 5:
        low_frequency_token_ner[key] = value
        
    return low_frequency_token_ner

 
def findFeaturesForText(context_ner):
    
    features = {"upper-case":0, "digit": 0, "contains-digit": 0, "first-char-upper": 0, "lower-case": 0,"other":0}
    state_features = {"B-PER":deepcopy(features), 
                      "I-PER":deepcopy(features), 
                      "B-LOC":deepcopy(features),
                      "I-LOC":deepcopy(features), 
                      "B-ORG":deepcopy(features), 
                      "I-ORG":deepcopy(features),
                      "B-MISC":deepcopy(features),
                      "I-MISC":deepcopy(features), 
                      "O":deepcopy(features)}

    for key, values in context_ner.iteritems():
        feature_class = findFeatureClass(key)
        for net in values:
            state_features[net][feature_class] += 1
     
    return state_features

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
    for _, state_features in features.iteritems():
        total_tokens = sum(state_features.values())
        for key, value in state_features.iteritems():
            state_features[key] = value * 1.0 / total_tokens;
    
    return features

def saveFeaturesToDisk(features, feature_type):
    '''
        Saves the final test predictions to CSV format in the format needed
        for kaggle
    '''
    print("Saving {0} to disk".format(feature_type))
    f = open(dir_path + 'Training_Test_Data/{0}'.format(feature_type), "w")
    json.dump(features, f)
    f.close();
    
def getLowFrequencyWordProbabilities():
    '''
        Caller method to get the features
    '''
    return getTrainingData()

def findFeaturesForPOS(low_frequency_pos, low_frequency_ner):
    '''
        Determine the distribution of the named entities under each part of speech
        tag
    '''
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
    
    return features

getTrainingData()
