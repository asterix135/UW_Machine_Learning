"""
Code to answers first programming quiz week 2
"""

import json
import os
import numpy as np
import pandas as pd
import string

os.chdir("..")

products = pd.read_csv('data/amazon_baby_subset.csv')
# columns = ['name', 'review', 'rating', 'sentiment']

with open('data/important_words.json') as f:
    important_words = json.load(f)

# fill Nas and remove punctuation
products = products.fillna({'review': ''})


def remove_punctuation(text):
    remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
    return text.translate(remove_punct_map)


products['review_clean'] = products['review'].apply(remove_punctuation)

# count occurances of important words in clean reviews

for word in important_words:
    products[word] = products['review_clean'].apply(
        lambda s: s.split().count(word)
    )

# calculate number of reviews containing 'perfect'

print('========================')
print('Question 1')
print('----------\n')
print('Number of reviews contining "perfect": ' + str(
    sum(products['perfect'] > 0)
))
print('========================\n\n')


# Convert to arrays
def get_numpy_data(dataframe, features, label):
    """
    Convert dataframe to 2 numpy arrays
    One containing features, one with labels
    """
    dataframe['constant'] = 1
    features = ['constant'] + features
    features_frame = dataframe[features]
    feature_matrix = features_frame.as_matrix()
    label_sarray = dataframe[label]
    label_array = label_sarray.as_matrix()
    return feature_matrix, label_array


feature_matrix, sentiment = get_numpy_data(products,
                                           important_words,
                                           'sentiment')

print('========================')
print('Question 2')
print('----------\n')
print('Number of features in matrix: ' + str(feature_matrix.shape[1]))
print('========================\n\n')

print('========================')
print('Question 3')
print('----------\n')
print('Same')
print('========================\n\n')

