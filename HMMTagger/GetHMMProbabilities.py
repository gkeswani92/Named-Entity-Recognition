__author__ = 'Jonathan Simon'

from collections import defaultdict, Counter


def getEmissionProbabilities(pos_list, ne_list):
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
            this_pos = pos_list[i][j]
            emission_probs[this_ne][this_pos] += 1

    # Add the <UNK> token, and normalize the counts
    # NOTE: Should be smarter about adding this "unknown" probability mass
    for ne in emission_probs:
        emission_probs[ne]['<UNK>'] = 1
        total_count = 1.0*sum(emission_probs[ne].values())
        for pos in emission_probs[ne]:
            emission_probs[ne][pos] /= total_count

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