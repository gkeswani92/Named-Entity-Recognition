__author__ = 'Jonathan Simon'

from DataProcessing.LoadData import getTrainingData, getTestData
from DataProcessing.Utilities import savePredictionsToCSV
from GetHMMProbabilities import getEmissionProbabilities, getBigramStateProbabilities
import numpy as np
from copy import deepcopy

def Viterbi(emission_probs, state_init_probs, state_trans_probs, test_subseq):
    '''
    For now we're ignoring the <UNK> tokens that were inserted
    If we lookup an emission that doesn't exist, it will have probability 0, since we're using a Counter
    '''
    # Initialize paths and probabilities
    path_dict = dict((state, [state[1]]) for state in state_init_probs.keys())
    prev_probs = {}
    for state in state_init_probs.keys():
        prev_probs[state] = state_init_probs[state] * emission_probs[state[1]][test_subseq[0]]

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

            curr_bigram = (max_state[1], curr_state)
            curr_state_probs[curr_bigram] = max_prob
            new_path_dict[curr_bigram] = path_dict[max_state] + [curr_state]

        prev_probs = curr_state_probs.copy()
        path_dict = new_path_dict.copy()

    # Identify overall most probable path
    overall_max_idx = np.argmax(prev_probs.values())
    overall_max_state = prev_probs.keys()[overall_max_idx]

    return path_dict[overall_max_state]


def SmoothViterbi(emission_probs, state_init_probs, state_trans_probs, test_subseq):
    '''
    For now we're ignoring the <UNK> tokens that were inserted
    If we lookup an emission that doesn't exist, it will have probability 0, since we're using a Counter
    '''
    # Initialize paths and probabilities
    path_dict = dict((state, [state[1]]) for state in state_init_probs.keys())
    prev_probs = {}
    for state in state_init_probs.keys():
        prev_probs[state] = state_init_probs[state] * emission_probs[state[1]][test_subseq[0]]

    # Iterate over the sentence
    all_states = set(state2 for state1 in state_trans_probs for state2 in state_trans_probs[state1])
    for emission in test_subseq[1:]:
        new_path_dict = {}
        curr_state_probs = {}
        for curr_state in all_states:
            temp_state_probs = {}
            for prev_state in path_dict:

                if emission not in emission_probs[curr_state]:
                    this_emission_prob = emission_probs[curr_state]["<UNK>"]
                else:
                    this_emission_prob = emission_probs[curr_state][emission]

                if prev_state not in state_trans_probs:
                    this_trans_prob = state_trans_probs["<UNK>"][curr_state]
                elif curr_state not in state_trans_probs[prev_state]:
                    this_trans_prob = state_trans_probs[prev_state]["<UNK>"]
                else:
                    this_trans_prob = state_trans_probs[prev_state][curr_state]

                temp_state_probs[prev_state] = prev_probs[prev_state] * this_trans_prob * this_emission_prob

            max_idx = np.argmax(temp_state_probs.values())
            max_prob = temp_state_probs.values()[max_idx]
            max_state = temp_state_probs.keys()[max_idx]

            curr_bigram = (max_state[1], curr_state)
            curr_state_probs[curr_bigram] = max_prob
            new_path_dict[curr_bigram] = path_dict[max_state] + [curr_state]

        prev_probs = curr_state_probs.copy()
        path_dict = new_path_dict.copy()

    # Identify overall most probable path
    overall_max_idx = np.argmax(prev_probs.values())
    overall_max_state = prev_probs.keys()[overall_max_idx]

    return path_dict[overall_max_state]


def getTestPreds(train_obs_list, train_ne_list, test_obs_list, smooth):
    emission_probs = getEmissionProbabilities(train_obs_list, train_ne_list)
    state_init_probs, state_trans_probs = getBigramStateProbabilities(train_ne_list)
    pred_ne_list = []
    if smooth:
        smoothed_emission_probs = getSmoothEmissionProbs(emission_probs)
        smoothed_state_trans_probs = getSmoothTransitionProbs(state_trans_probs)
        for i in xrange(len(test_obs_list)):
            predicted_states = SmoothViterbi(smoothed_emission_probs, state_init_probs, smoothed_state_trans_probs, test_obs_list[i])
            pred_ne_list.append(predicted_states)
    else:
        for i in xrange(len(test_obs_list)):
            predicted_states = Viterbi(emission_probs, state_init_probs, state_trans_probs, test_obs_list[i])
            pred_ne_list.append(predicted_states)

    return pred_ne_list


def getSmoothEmissionProbs(emission_probs):
    '''
    For each state, consider the probability of an unseen emission to be the probability the of least common emission
    '''
    smoothed_emission_probs = deepcopy(emission_probs)
    for state in smoothed_emission_probs:
        smoothed_emission_probs[state]["<UNK>"] = min(smoothed_emission_probs[state].values()) / 10 # state known, emission unknown

    return smoothed_emission_probs


def getSmoothTransitionProbs(state_trans_probs):
    '''
    For each state, consider the probability of an unseen emission to be the probability the of least common emission
    '''
    all_known_states = set(state2 for state1 in state_trans_probs for state2 in state_trans_probs[state1])
    smoothed_state_trans_probs = deepcopy(state_trans_probs)

    # preceding bigram known, current state unknown
    for bigram in smoothed_state_trans_probs:
        smoothed_state_trans_probs[bigram]["<UNK>"] = min(smoothed_state_trans_probs[bigram].values()) / 10

    # preceding bigram unknown, current state known
    for state in all_known_states:
        min_state_prob = 1
        for bigram in smoothed_state_trans_probs:
            if state in smoothed_state_trans_probs[bigram] and smoothed_state_trans_probs[bigram][state] < min_state_prob:
                min_state_prob = smoothed_state_trans_probs[bigram][state]
        smoothed_state_trans_probs["<UNK>"][state] = min_state_prob / 10

    return smoothed_state_trans_probs


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
                        ne_start = inds[i][word_idx]
                        ne_tag = preds[i][word_idx].split('-')[1]
                    else:  # outside of named entity
                        in_ne = False
            word_idx += 1
    return formatted_preds


def main():
    train_word_list, train_pos_list, train_ne_list = getTrainingData(HMM=True)
    test_word_list, test_pos_list, test_idx_list = getTestData(HMM=True)

    # tag_seq_preds = getTestPreds(train_pos_list, train_ne_list, test_pos_list, smooth=True)
    tag_seq_preds = getTestPreds(train_word_list, train_ne_list, test_word_list, smooth=True)
    formatted_preds = formatTestPreds(tag_seq_preds, test_idx_list)
    savePredictionsToCSV(formatted_preds)

if __name__ == '__main__':
    main()