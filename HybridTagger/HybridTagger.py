__author__ = 'Jonathan Simon'

import os
import pandas as pd
from copy import deepcopy


def loadPredictions(pred_file_path):
    '''
    Read in the csv file, and convert it to a dict of lists
    '''

    preds = pd.read_csv(pred_file_path)
    pred_dict = {}
    for idx, row in preds.iterrows():
        pred_dict[row['Type']] = row['Prediction'].strip().split(' ')

    return pred_dict


def mergePredictions(core_preds, extra_preds):
    '''
    For each prediction in "extra_preds", add it to "core_preds" *if* there is no conflict
    '''

    # Create a set of all "occupied" indices
    core_pred_set = set()
    for pred_list in core_preds.values():
        for pred in pred_list:
            pred_start, pred_end = map(int, pred.split('-'))
            core_pred_set.update([tag_idx for tag_idx in range(pred_start, pred_end+1)])

    # Add in all predictions which don't overlap
    merged_preds = deepcopy(core_preds)
    for tag_key, tag_inds in extra_preds.iteritems():
        for pred in tag_inds:
            pred_start, pred_end = map(int, pred.split('-'))
            num_overlaps = len([this_idx for this_idx in range(pred_start, pred_end+1) if this_idx in core_pred_set])
            if num_overlaps == 0:
                merged_preds[tag_key].append(pred)

    # Sort the predictions so that the added ones are in the right spots
    for tag_key in merged_preds:
        merged_preds[tag_key].sort(key=lambda x: int(x.split('-')[0]))

    return merged_preds


def saveMergedPreds(merged_preds, pred_outfile):
    with open(pred_outfile, 'w') as outfile:
        outfile.write("Type,Prediction\n")
        ordered_keys = ["ORG", "MISC", "PER", "LOC"]
        for key in ordered_keys:
            outfile.write(key + ',' + ' '.join(merged_preds[key]) + '\n')


def main():
    pred_dir = '/'.join(os.path.abspath(__file__).split('/')[:-2]) + '/DataProcessing/Training_Test_Data/'
    # core_pred_file = pred_dir+'predictions_dynamicTagger.csv'
    # extra_pred_file = pred_dir+'predictions_smoothedWordBigramHMMTagger.csv'
    # core_pred_file = pred_dir+'predictions_smoothedWordBigramHMMTagger.csv'
    # extra_pred_file = pred_dir+'predictions_dynamicTagger.csv'
    core_pred_file = pred_dir+'predictions_smoothedWordBigramHMMTagger10.csv'
    extra_pred_file = pred_dir+'predictions_dynamicTagger.csv'

    core_preds = loadPredictions(core_pred_file)
    extra_preds = loadPredictions(extra_pred_file)

    merged_preds = mergePredictions(core_preds, extra_preds)

    saveMergedPreds(merged_preds, pred_dir+'merged_predictions.csv')

if __name__ == '__main__':
    main()