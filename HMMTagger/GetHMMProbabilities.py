__author__ = 'Jonathan Simon'

from collections import defaultdict, Counter
from copy import deepcopy


def applyGoodTuringSmoothing(emission_freqs, max_freq=5):
    '''
        Takes the frequency distribution as an input and returns the smoothed
        frequency distribution
    '''

    #Zero count estimate
    zero_counts = Counter()
    for this_ne_state, this_word_counter in emission_freqs.iteritems():
        seen_word_set = set(this_word_counter.keys())
        unseen_set = set()
        for other_ne_state, other_word_counter in emission_freqs.iteritems():
            if this_ne_state != other_ne_state:
                unseen_set |= (set(other_word_counter.keys()) - seen_word_set)
        zero_counts[this_ne_state] = len(unseen_set)


    #Counter to determine how many unigrams occur 0 times, 1 times, 2 times ...
    freq_distribution = {}
    for ne_state, word_counts in emission_freqs.iteritems():
        freq_distribution[ne_state] = Counter(word_counts.values())
        freq_distribution[ne_state][0] = zero_counts[ne_state] # estimate of unseen counts
        # freq_distribution[ne_state][0] = freq_distribution[ne_state][1] # estimate of unseen counts

    #Applying good turing smoothing on frequencies less than n
    smoothed_frequencies = defaultdict(dict)
    for ne_state, count_freq in freq_distribution.iteritems():
        for i in range(max_freq+1):
            smoothed_frequencies[ne_state][i] = (i + 1) * (float(count_freq[i+1])/count_freq[i])

    #Updating the counts which were less than n in the original frequency distribution
    #with the new smoothed counts
    smoothed_emission_freqs = deepcopy(emission_freqs)
    for ne_state, word_counts in smoothed_emission_freqs.iteritems():
        for word, count in word_counts.iteritems():
            if count <= max_freq:
                word_counts[word] = smoothed_frequencies[ne_state][count]
        word_counts['<UNK>'] = smoothed_frequencies[ne_state][0]

    return smoothed_emission_freqs


def getEmissionProbabilities(obs_list, ne_list, smooth=None):
    '''
    For each named entity, need to compute the emmission probabilities for each part of speech
    Because some parts of speech never occur within a named entity type, we also need to include
    an "<UNK>" pos tag for each named entity type, whose value is determined by Laplacian smoothing

    Should return a dict of dicts, mapping named entities to parts of speech, and parts of speech to probabilities
    '''

    # Fill up the dictionaries with counts
    emission_probs = defaultdict(Counter)
    for i in xrange(len(ne_list)): # for each sentence in the dataset
        for j in xrange(len(ne_list[i])): # for each word in the sentence
            this_ne = ne_list[i][j]
            this_obs = obs_list[i][j]
            emission_probs[this_ne][this_obs] += 1

    # Contains its own loop; consider refactoring
    if smooth == 'Good-Turing':
        emission_probs = applyGoodTuringSmoothing(emission_probs)

    # Add the <UNK> token, and normalize the counts
    for ne in emission_probs:
        if smooth == 'Laplacian':
            emission_probs[ne]['<UNK>'] = Counter(emission_probs[ne].values())[1]
        total_count = 1.0*sum(emission_probs[ne].values())
        for obs in emission_probs[ne]:
            emission_probs[ne][obs] /= total_count

    return emission_probs

def getStateProbabilities(ne_list):
    '''
    For each state (named entity), compute the probability that it begins a sentence (init_probs),
    and the probability that it transitions to another named_entity (trans_probs)

    Should consider merging B-* and I-* tokens
    Should add some small probability mass for unseen state transitions
    '''

    # Fill up the dictionaries with counts
    init_probs = Counter()
    trans_probs = defaultdict(Counter)
    for i in xrange(len(ne_list)): # for each sentence in the dataset
        init_probs[ne_list[i][0]] += 1
        for j in xrange(len(ne_list[i])-1): # for each word in the sentence
            current_ne = ne_list[i][j]
            next_ne = ne_list[i][j+1]
            trans_probs[current_ne][next_ne] += 1

    # Normalize the initial probabilities
    total_count = 1.0*sum(init_probs.values())
    for ne in init_probs:
        init_probs[ne] /= total_count

    # Normalize the transition probabilities
    for ne1 in trans_probs:
        total_count = 1.0*sum(trans_probs[ne1].values())
        for ne2 in trans_probs[ne1]:
            trans_probs[ne1][ne2] /= total_count

    return init_probs, trans_probs