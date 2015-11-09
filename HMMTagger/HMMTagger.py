__author__ = 'Jonathan Simon'

from DataProcessing.LoadData import getTrainingData, getTestData
from DataProcessing.Utilities import savePredictionsToCSV
from GetHMMProbabilities import getEmissionProbabilities, getStateProbabilities
import numpy as np

def Viterbi(emission_probs, state_init_probs, state_trans_probs, test_subseq):
    '''
    For now we're ignoring the <UNK> tokens that were inserted
    If we lookup an emission that doesn't exist, it will have probability 0, since we're using a Counter
    '''
    # Initialize paths and probabilities
    path_dict = dict((state, [state]) for state in state_init_probs.keys())
    prev_probs = {}
    for state in state_init_probs.keys():
        prev_probs[state] = state_init_probs[state] * emission_probs[state][test_subseq[0]]

    # Iterate over the sentence
    all_states = set(state2 for state1 in state_trans_probs for state2 in state_trans_probs[state1])
    for emission in test_subseq[1:]:
        new_path_dict = {}
        curr_state_probs = {}
        for curr_state in all_states:
            temp_state_probs = {}
            for prev_state in path_dict:
                temp_state_probs[prev_state] = prev_probs[prev_state] * state_trans_probs[prev_state][curr_state] * \
                                               emission_probs[curr_state][emission]

            max_idx = np.argmax(temp_state_probs.values())
            max_prob = temp_state_probs.values()[max_idx]
            max_state = temp_state_probs.keys()[max_idx]

            curr_state_probs[curr_state] = max_prob
            new_path_dict[curr_state] = path_dict[max_state] + [curr_state]

        prev_probs = curr_state_probs.copy()
        path_dict = new_path_dict.copy()

    # Identify overall most probable path
    overall_max_idx = np.argmax(prev_probs.values())
    overall_max_state = prev_probs.keys()[overall_max_idx]

    return path_dict[overall_max_state]


def getTestPreds(train_pos_list, train_ne_list, test_pos_list):
    emission_probs = getEmissionProbabilities(train_pos_list, train_ne_list)
    state_init_probs, state_trans_probs = getStateProbabilities(train_ne_list)
    pred_ne_list = []
    for i in xrange(len(test_pos_list)):
        predicted_states = Viterbi(emission_probs, state_init_probs, state_trans_probs, test_pos_list[i])
        pred_ne_list.append(predicted_states)
    return pred_ne_list


def formatTestPreds(preds, inds):
    '''
    Could break if HMM does not follow B-before-I rule
    In particular, if I-* occurs before B-* in the sentence, then the formatted output
    will have "None" as the starting index
    '''
    formatted_preds = {"ORG": [], "MISC": [], "PER": [], "LOC": []}
    for i in xrange(len(preds)):  # sentences
        word_idx = 0
        in_ne = False
        ne_start = None
        ne_end = None
        ne_tag = None
        while word_idx < len(preds[i]):
            if not in_ne:  # if not inside a named entity
                if preds[i][word_idx][0] == 'B':  # if encountered the start of a named entity
                    in_ne = True
                    ne_start = inds[i][word_idx]
                    ne_tag = preds[i][word_idx].split('-')[1]
            else:  # if inside a named entity
                if preds[i][word_idx][0] != 'I':  # if past the end of a named entity
                    ne_end = inds[i][word_idx-1]
                    formatted_preds[ne_tag].append("{0}-{1}".format(ne_start, ne_end))
                    if preds[i][word_idx][0] == 'B':  # start of new named entity
                        ne_start = word_idx
                        ne_tag = preds[i][word_idx].split('-')[1]
                    else:  # outside of named entity
                        in_ne = False
            word_idx += 1
    return formatted_preds


def main():
    _, train_pos_list, train_ne_list = getTrainingData(HMM=True)
    _, test_pos_list, test_idx_list = getTestData(HMM=True)

    tag_seq_preds = getTestPreds(train_pos_list, train_ne_list, test_pos_list)
    formatted_preds = formatTestPreds(tag_seq_preds, test_idx_list)
    savePredictionsToCSV(formatted_preds)

if __name__ == '__main__':
    main()