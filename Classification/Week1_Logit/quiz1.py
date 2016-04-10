"""
Code to answer questions for week 1
"""

import json
import numpy as np
import os
import pandas as pd
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
os.chdir('..')
products = pd.read_csv('data/amazon_baby.csv')


def remove_punctuation(text):
    remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
    return text.translate(remove_punct_map)


# replace NA values with empty string
products = products.fillna({'review': ''})

# remove punctuation
products['review_clean'] = products['review'].apply(remove_punctuation)

# remove neutral ratings
products = products[products['rating'] != 3]

# apply -1/1 sentiment value
products['sentiment'] = products['rating'].apply(
    lambda rating: +1 if rating > 3 else -1
)

# Split test/train with provided indices
with open('data/module-2-assignment-test-idx.json') as f:
    test_idx = json.load(f)

with open('data/module-2-assignment-train-idx.json') as f:
    train_idx = json.load(f)

train_data = products.iloc[train_idx, :]
test_data = products.iloc[test_idx, :]

# build bag of words
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')

train_matrix = vectorizer.fit_transform(train_data['review_clean'])
test_matrix = vectorizer.transform(test_data['review_clean'])

# Train classifier
sentiment_model = LogisticRegression()
sentiment_model.fit(train_matrix, train_data['sentiment'])

print('========================')
print('Question 1')
print('----------\n')
print('Num weights >= 0: ' + str(sum(sentiment_model.coef_[0] >= 0)))
print('========================\n\n')

# Make predictions

# Review some sample data
sample_test_data = test_data[10:13]

# transform data
sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
scores = sentiment_model.decision_function(sample_test_matrix)


# function to determine class
def predict_class(score):
    return -1 if score <= 0 else 1


predict_class = np.vectorize(predict_class)
sample_class_preds = predict_class(scores)


# function to calculate probability
def predict_prob_pos(score):
    return 1 / (1 + np.exp(-score))


print('========================')
print('Question 2')
print('----------\n')
print('review with lowest probability of being positive: ' +
      str(np.argmin(predict_prob_pos(scores)) + 1))
print('========================\n\n')

# Calculate top 20 most positive reviews in test set

test_data['predicted_pos_prob'] = sentiment_model.predict_proba(
    test_matrix)[:, 1]

top20 = test_data.sort_values('predicted_pos_prob', ascending=False)[:20]

print('===========================')
print('Question 3 - Top 20 reviews')
print('---------------------------\n')
print(top20['name'])
print('===========================\n\n')

# Calculate top 20 most negative reviews in test set

bottom20 = test_data.sort_values('predicted_pos_prob', ascending=True)[:20]

print('==============================')
print('Question 4 - Bottom 20 reviews')
print('------------------------------\n')
print(bottom20['name'])
print('==============================\n\n')


# Calculate accuracy of predictor

sentiment_preds = sentiment_model.predict(test_matrix)
num_agree = sum(sentiment_preds == test_data['sentiment'])

print('========================')
print('Question 5')
print('----------\n')
print('Accuracy: ' + str(num_agree / len(test_data)))
print('========================\n\n')

print('========================')
print('Question 6')
print('----------\n')
print('No')
print('========================\n\n')

# Fewer word classifier
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect',
                     'loves',
                     'well', 'able', 'car', 'broke', 'less', 'even', 'waste',
                     'disappointed',
                     'work', 'product', 'money', 'would', 'return']
vectorizer_word_subset = CountVectorizer(vocabulary=significant_words)
train_matrix_word_subset = vectorizer_word_subset.fit_transform(
    train_data['review_clean']
)
test_matrix_word_subset = vectorizer_word_subset.transform(
    test_data['review_clean']
)

simple_model = LogisticRegression()
simple_model.fit(train_matrix_word_subset, train_data['sentiment'])
simple_model_coef_table = pd.DataFrame(
    {'word': significant_words, 'coefficient': simple_model.coef_.flatten()}
)
print(simple_model_coef_table.sort_values('coefficient', ascending=False))

print('========================')
print('Question 7')
print('----------\n')
print('Number of postive coefficients: ' + str(
    sum(simple_model_coef_table['coefficient'] > 0)
))
print('========================\n\n')

all_words = vectorizer.get_feature_names()
not_both_positive = 0

for word in significant_words:
    vectorizer_idx = all_words.index(word)
    big_list_score = sentiment_model.coef_[0][vectorizer_idx]
    small_list_score = simple_model_coef_table.loc[
        simple_model_coef_table['word'] == word
    ].iloc[0]['coefficient']
    if small_list_score > 0 and (big_list_score * small_list_score < 0):
        print(word + ': score signs do not match')
        not_both_positive += 1

print('========================')
print('Question 8')
print('----------\n')
print(str(not_both_positive) + ' words are positive in small but not in big')
print('========================\n\n')

# Compare models

big_train_preds = sentiment_model.predict(train_matrix)
big_train_agree = sum(big_train_preds == train_data['sentiment'])

simple_train_preds = simple_model.predict(train_matrix_word_subset)
simple_train_agree = sum(simple_train_preds == train_data['sentiment'])

print('========================')
print('Question 9')
print('----------\n')
print('Sentiment model accuracy on train data: ' + \
      str(big_train_agree / len(train_data)))
print('Simple model accuracy on train data: ' + \
      str(simple_train_agree / len(train_data)))
print('========================\n\n')

simple_test_preds = simple_model.predict(test_matrix_word_subset)
simple_test_agree = sum(simple_test_preds == test_data['sentiment'])

print('========================')
print('Question 10')
print('-----------\n')
print('Sentiment model accuracy on test data: ' +
      str(num_agree / len(test_data)))
print('Simple model accuracy on train data: ' +
      str(simple_test_agree / len(test_data)))
print('========================\n\n')

# Compare with majority class predictor

majority_class = 1 if sum(test_data['sentiment'] == 1) > \
                      len(test_data) / 2 else -1
majority_preds = np.empty(len(test_data))
majority_preds.fill(majority_class)
majority_class_agree = sum(majority_preds == test_data['sentiment'])

print('========================')
print('Question 11')
print('-----------\n')
print('Accuracy of majority class prediction: ' + str(
    round(majority_class_agree / len(test_data), 2)
))
print('========================\n\n')
