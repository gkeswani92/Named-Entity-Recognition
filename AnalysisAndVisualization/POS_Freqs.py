'''
Repurposed plotting code found here:
http://matplotlib.org/examples/pylab_examples/barchart_demo.html
'''

__author__ = 'Jonathan Simon'

from DataProcessing.LoadData import getTrainingData, getTestData
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

train_word_list, train_pos_list, train_ne_list = getTrainingData(HMM=True)
test_word_list, test_pos_list, test_idx_list = getTestData(HMM=True)

# Get POS counts in training set
all_train_pos = []
for sentence in train_pos_list:
    for pos in sentence:
        all_train_pos.append(pos)

train_pos_counter = Counter(all_train_pos)
train_pos_names, train_pos_counts = zip(*train_pos_counter.most_common())

# Get POS counts in test set
all_test_pos = []
for sentence in test_pos_list:
    for pos in sentence:
        all_test_pos.append(pos)

test_pos_counter = Counter(all_test_pos) # all test keys present in the training data

# Order the counts in the same way for the test set
test_pos_names = train_pos_names
test_pos_counts = [test_pos_counter[key] for key in train_pos_names]


# Turn counts into probs
train_count_total = sum(train_pos_counts)
train_pos_probs = [1.0*x/train_count_total for x in train_pos_counts]
test_count_total = sum(test_pos_counts)
test_pos_probs = [1.0*x/test_count_total for x in test_pos_counts]


# fig, ax = plt.subplots()
fig = plt.figure(figsize=(14,6))

index = np.arange(len(train_pos_counter))
bar_width = 0.35

opacity = 0.4

# rects1 = plt.bar(index, train_pos_counts, bar_width,
rects1 = plt.bar(index, train_pos_probs, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Training Dataset')

# rects2 = plt.bar(index + bar_width, test_pos_counts, bar_width,
rects2 = plt.bar(index + bar_width, test_pos_probs, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Test Dataset')

# ax.set_yscale('log', basey=10)
plt.xlabel('Part-of-Speech')
# plt.ylabel('Count')
plt.ylabel('Probability Of Occurrence')
plt.title('POS Frequencies in Training and Test Data')
plt.xticks(index + bar_width, train_pos_names, rotation=-45)
plt.legend()

plt.tight_layout()
# plt.show()

dir_path = "/Users/Macbook/Documents/Cornell/CS 5740 - Natural Language Processing/Project 3/Named-Entity-Recognition/AnalysisAndVisualization/"
filename = "POS_barplot.png"
plt.savefig(dir_path+filename)