'''
Created on Nov 2, 2015

@author: gaurav
'''

from DataProcessing.Utilities import savePredictionsToCSV
from DataProcessing.LoadData import getTrainingData, getTestData

def makePredictions(training_data, context, pos, index):
    
    predictions = {"ORG" : [], "MISC" : [], "PER" : [], "LOC" : []}
    current_namedEntity_index = {'start':0, 'end':0}
    current_namedEntity = ""
    
    for i in xrange(len(context)):
        word = context[i]
        pos_ne_information = training_data.get(word, None)
        
        #If we have seen this word before in our training data
        if pos_ne_information:
            named_entity = pos_ne_information.get('NE', None)
            
            #If the named entity is one we care about
            if named_entity in ['B-MISC','I-MISC','B-ORG','I-ORG','B-PER','I-PER','B-LOC','I-LOC']:
                
                #A new beginning for a named entity and thus the start and the end indexes are the same
                if not current_namedEntity:
                    current_namedEntity = named_entity
                    current_namedEntity_index['start'] = i
                    current_namedEntity_index['end'] = i
                    
                #We have found a pair of words that are a part of the same named entity
                #Thus we update the end index
                elif named_entity.endswith(current_namedEntity):
                    current_namedEntity_index['end'] = i
                
                #We have found the beginning of a new named entity
                #Store the current prediction in predictions and update the start 
                #and end indexes for the new one
                else:
                    category = current_namedEntity.split('-')[1]
                    predictions[category].append("{0}-{1}".format(current_namedEntity_index['start'],current_namedEntity_index['end']))
                    
                    #Setting the variables up to consider the new entity from now on
                    current_namedEntity = named_entity
                    current_namedEntity_index['start'] = i
                    current_namedEntity_index['end'] = i
            
            #Current token is not a named entity. 
            else:
                
                #Tokens uptill now were named entities. Flush data to predictions
                if current_namedEntity:
                    category = current_namedEntity.split('-')[1]
                    predictions[category].append("{0}-{1}".format(current_namedEntity_index['start'],current_namedEntity_index['end']))
                
                #Reset data in preparation of new named entities 
                current_namedEntity = ""
                current_namedEntity_index['start'] = 0
                current_namedEntity_index['end'] = 0
        
        #Current token has never been seen before. 
        else:
            
            #Tokens up till now were named entities. Flush data to predictions
            if current_namedEntity:
                category = current_namedEntity.split('-')[1]
                predictions[category].append("{0}-{1}".format(current_namedEntity_index['start'],current_namedEntity_index['end']))
            
            #Reset data in preparation of new named entities 
            current_namedEntity = ""
            current_namedEntity_index['start'] = 0
            current_namedEntity_index['end'] = 0
       
    return predictions
    
def main():
    training_data = getTrainingData()
    context, pos, index = getTestData()
    predictions = makePredictions(training_data, context, pos, index)
    savePredictionsToCSV(predictions)
    
if __name__ == '__main__':
    main()