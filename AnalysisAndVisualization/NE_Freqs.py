__author__ = 'Jonathan Simon'

'''
Repurposed plotting code found here:
http://matplotlib.org/examples/pylab_examples/barchart_demo.html
'''

__author__ = 'Jonathan Simon'

from DataProcessing.LoadData import getTrainingData
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

train_word_list, train_pos_list, train_ne_list = getTrainingData(HMM=True)

ne_counter = Counter([ne for sentence in train_ne_list for ne in sentence])
ne_names, ne_counts = zip(*ne_counter.most_common())

ne_count_total = sum(ne_counts)
ne_pos_probs = [1.0*x/ne_count_total for x in ne_counts]

# fig, ax = plt.subplots()
fig = plt.figure(figsize=(14,6))

index = np.arange(len(ne_counter)-1)
bar_width = 0.80

opacity = 0.4

# rects1 = plt.bar(index, train_pos_counts, bar_width,
# rects1 = plt.bar(index, ne_pos_probs[1:], bar_width,
rects1 = plt.bar(index, ne_counts[1:], bar_width,
                 alpha=opacity,
                 color='b')

# ax.set_yscale('log', basey=10)
plt.xlabel('Named Entity')
# plt.ylabel('Count')
# plt.ylabel("Probability Of Occurrence ('O' not shown)")
plt.ylabel("Frequency Of Occurrence ('O' not shown)")
plt.title("Named Entity Frequencies in Training Data ('O' not shown)")
plt.xticks(index + bar_width/2, ne_names[1:])

plt.tight_layout()
# plt.show()

dir_path = "/Users/Macbook/Documents/Cornell/CS 5740 - Natural Language Processing/Project 3/Named-Entity-Recognition/AnalysisAndVisualization/"
filename = "NE_count_barplot.png"
plt.savefig(dir_path+filename)