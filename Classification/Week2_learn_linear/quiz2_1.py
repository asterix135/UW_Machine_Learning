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


# estimate conditional probability with link function
def predict_probability(feature_matrix, coefficients):
    # take dot product of feature matrix and coefficients
    score = np.dot(feature_matrix, coefficients)
    # Compute P(y_i = +1 | x_i, w) using the link function
    predictions = 1 / (1 + np.exp(-score))
    return predictions


# compute derivative of log likelihood wrt a single coefficient
def feature_derivative(errors, feature):
    derivative = np.dot(errors, feature)
    return derivative


# compute log likelihood
def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment == +1)
    scores = np.dot(feature_matrix, coefficients)
    lp = np.sum((indicator-1) * scores - np.log(1. + np.exp(-scores)))
    return lp


# implement logistic regression
def logistic_regression(feature_matrix, sentiment, initial_coefficients,
                        step_size, max_iter):
    coefficients = np.array(initial_coefficients)
    for itr in range(max_iter):
        predictions = predict_probability(feature_matrix, coefficients)
        indicator = sentiment == +1
        errors = indicator - predictions

        for j in range(len(coefficients)):
            derivative = feature_derivative(errors, feature_matrix[:, j])
            coefficients[j] += step_size * derivative

        #Check whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or \
            (itr <= 1000 and itr % 100 == 0) or \
                (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print('iteration %*d: log likelihood of observed labels = %.8f' %
                  (int(np.ceil(np.log10(max_iter))), itr, lp))
    return coefficients


coefficients = logistic_regression(feature_matrix, sentiment,
                                   np.zeros((194)), 1e-7, 301)

print('========================')
print('Question 4')
print('----------\n')
print('Increase')
print('========================\n\n')

# Predict sentiments

scores = np.dot(feature_matrix, coefficients)
predicted_sentiment = np.array([+1 if s > 0 else -1 for s in scores])
q5 = sum(predicted_sentiment == +1)

print('========================')
print('Question 5')
print('----------\n')
print('Reviews with predicted positive sentiment: ' + str(q5))
print('========================\n\n')

# Measure accuracy

accuracy = sum(predicted_sentiment == sentiment) / len(sentiment)

print('========================')
print('Question 6')
print('----------\n')
print('Accuracy: ' + str(accuracy))
print('========================\n\n')

# Identify most positive & negative words

coefficients = list(coefficients[1:])  # exclude intercept
word_coefficient_tuples = [(word, coefficient) for word, coefficient in
                           zip(important_words, coefficients)]
word_coefficient_tuples = sorted(word_coefficient_tuples,
                                 key=lambda x:x[1],
                                 reverse=True)

# Ten most positive words

print('========================')
print('Question 7')
print('----------\n')
print('Top Ten Positive Words')
for word in word_coefficient_tuples[:10]:
    print(word)
print('========================\n\n')

# Ten most negative words


print('Question 7')
print('----------\n')
print('Top Ten Negative Words')
for word in word_coefficient_tuples[-11: -1]:
    print(word)
print('========================\n\n')