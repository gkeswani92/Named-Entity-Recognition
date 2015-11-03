'''
Created on Nov 2, 2015

@author: gaurav
'''

from DataProcessing.Utilities import savePredictionsToCSV, dir_path, training_file, getDataFromFile, loadFile, pprint
from DataProcessing.LoadData import getTrainingData, getTestData, parseTrainingData
from collections import defaultdict

def processTrainingData(context, pos, ner):
    '''
        Given the context, part of speeches and the named entity tags, 
        create a data structure that simplifies look ups
    '''

    current_namedEntity_sequence = []
    current_namedEntity_type = ""
    training_data = {}
    i = 0
    
    while i < len(ner):
        named_entity = ner[i]
        
        #This is one of the named entities we are interested in
        if '-' in named_entity and named_entity.split('-')[1] in ['MISC', 'ORG', 'PER', 'LOC']:
            
            #If this is the first part of a named entity sequence
            if named_entity.startswith('B'):
                
                #If there is a previous entity set in the buffer, we need to flush it out
                if current_namedEntity_sequence and current_namedEntity_type:
                    if len(current_namedEntity_sequence) > 1:
                        training_data[tuple(current_namedEntity_sequence)] = current_namedEntity_type
                    else:
                        training_data[current_namedEntity_sequence[0]] = current_namedEntity_type
                
                #Set the buffers to contains only this new entity tag
                current_namedEntity_type = named_entity.split('-')[1]
                current_namedEntity_sequence = [context[i]]
            
            #If this is an internal part of a sequence of named entity tags
            else:
                current_namedEntity_sequence.append(context[i])
                
        #We have encountered an 'O'. Need to flush out data and reset the sequence markers
        else:
            if current_namedEntity_sequence and current_namedEntity_type:
                if len(current_namedEntity_sequence) > 1:
                    training_data[tuple(current_namedEntity_sequence)] = current_namedEntity_type
                else:
                    training_data[current_namedEntity_sequence[0]] = current_namedEntity_type
                    
            current_namedEntity_sequence = []
            current_namedEntity_type = ""
        
        i += 1

    return training_data

def getMaxLengthKey(training_data):
    '''
        Gets the key with the maximum length
    '''
    max_length = 0
    for key in training_data:
        if isinstance(key, tuple):
            if len(key) > max_length:
                max_length = len(key)
    return max_length

def getTrainingData():
    '''
        Loads the training data from the appropriate directory
    '''
    #Load the data present in the training file
    f = loadFile(dir_path + training_file)
    training_data = getDataFromFile(f)
    context, pos, ner = parseTrainingData(training_data)
    training_data = processTrainingData(context, pos, ner)
    largest_key_size = getMaxLengthKey(training_data)
    return training_data, largest_key_size

def makePredictions(training_data, context, pos, index, largest_key_size):
    
    predictions = {"ORG" : [], "MISC" : [], "PER" : [], "LOC" : []}
    i = 0
    
    while i < len(context):
        sequences = createSequenceFromContext(context[i:i+10], largest_key_size)
        
        #Going through the sequences in the reverse order. Longest sequence 
        for j in range(len(sequences)-1,-1,-1):
            sequence = sequences[j]
            
            if sequence in training_data:
                named_entity = training_data[sequence]
                end_index    = i + len(sequence[1:]) if isinstance(sequence, tuple) else i
                predictions[named_entity].append("{0}-{1}".format(str(i), str(end_index)))
                
                #Dont need to process words that have been clumped together in this 
                #entity sequence again
                i = end_index
                
                #Dont need to process for this sequence anymore
                break;
        
        i += 1
            
    return predictions
    
def createSequenceFromContext(context, length):
    '''
        Creates a sequence of all words that should be checked for in the training
        data
    '''
    final_sequences = []
    current_sequence = []
    for i in range(len(context)):
        current_sequence.append(context[i])
        
        if len(current_sequence) > 1:
            final_sequences.append(tuple(current_sequence))
        else:
            final_sequences.append(current_sequence[0])
            
    return final_sequences

def main():
    training_data, largest_key_size = getTrainingData()
    context, pos, index = getTestData()
    predictions = makePredictions(training_data, context, pos, index, largest_key_size)
    savePredictionsToCSV(predictions)

if __name__ == '__main__':
    main()