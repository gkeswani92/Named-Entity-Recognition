__author__ = 'Jonathan Simon'

from DataProcessing.LoadData import getTrainingData, getTestData
from DataProcessing.Utilities import savePredictionsToCSV
from HMMTagger.GetHMMProbabilities import getEmissionProbabilities, getStateProbabilities


def Viterbi(test_subseq, emission_probs, state_init_probs, state_trans_probs):
    pass


def getTestPreds(train_pos_list, train_ne_list, test_pos_list, test_idx_list):
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
                    ne_end = inds[i][word_idx]
                    formatted_preds[ne_tag] = "{0}-{1}".format(ne_start, ne_end)
                    if preds[i][word_idx][0] == 'B':  # start of new named entity
                        ne_start = word_idx
                        ne_tag = preds[i][word_idx].split('-')[1]
                    else:  # outside of named entity
                        in_ne = False
            word_idx += 1
    return formatted_preds


def main():
    train_word_list, train_pos_list, train_ne_list = getTrainingData(HMM=True)
    test_word_list, test_pos_list, test_idx_list = getTestData(HMM=True)

    tag_seq_preds = getTestPreds(train_pos_list, train_ne_list, test_pos_list, test_idx_list)
    formatted_preds = formatTestPreds(tag_seq_preds)
    savePredictionsToCSV(formatted_preds)

if __name__ == '__main__':
    main()