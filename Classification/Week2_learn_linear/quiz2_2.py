"""
Code to answers second programming quiz week 2
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

for word in important_words:
    products[word] = products['review_clean'].apply(
        lambda s: s.split().count(word)
    )

# Split test/train with provided indices
with open('data/module-4-assignment-validation-idx.json') as f:
    validation_idx = json.load(f)

with open('data/module-4-assignment-train-idx.json') as f:
    train_idx = json.load(f)

train_data = products.iloc[train_idx, :]
validation_data = products.iloc[validation_idx, :]


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


feature_matrix_train, sentiment_train = get_numpy_data(
    train_data, important_words, 'sentiment'
)
feature_matrix_valid, sentiment_valid = get_numpy_data(
    validation_data, important_words, 'sentiment'
)


def predict_probability(feature_matrix, coefficients):
    # take dot product of feature matrix and coefficients
    score = np.dot(feature_matrix, coefficients)
    # Compute P(y_i = +1 | x_i, w) using the link function
    predictions = 1 / (1 + np.exp(-score))
    return predictions


def feature_derivative_with_L2(errors, feature, coefficient,
                               l2_penalty, feature_is_constant):
    # Compute the dot product of errors and feature
    derivative = np.dot(errors, feature)

    # add L2 penalty term for any feature that isn't the intercept.
    if not feature_is_constant:
        derivative -= 2 * l2_penalty * coefficient

    return derivative


print('========================')
print('Question 1')
print('----------\n')
print('Intercept is not regularized')
print('========================\n\n')


def compute_log_likelihood_with_L2(feature_matrix, sentiment,
                                   coefficients, l2_penalty):
    indicator = (sentiment == +1)
    scores = np.dot(feature_matrix, coefficients)

    lp = np.sum((indicator-1)*scores - np.log(1. + np.exp(-scores))) - \
        l2_penalty * np.sum(coefficients[1:] ** 2)

    return lp

print('========================')
print('Question 2')
print('----------\n')
print('Decreases ll(w)')
print('========================\n\n')


def logistic_regression_with_L2(feature_matrix, sentiment,
                                initial_coefficients, step_size,
                                l2_penalty, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    for itr in range(max_iter):
        # Predict P(y_i = +1|x_i,w) using your predict_probability() function
        predictions = predict_probability(feature_matrix, coefficients)

        # Compute indicator value for (y_i = +1)
        indicator = (sentiment == +1)

        # Compute the errors as indicator - predictions
        errors = indicator - predictions

        for j in range(len(coefficients)): # loop over each coefficient
            is_intercept = (j == 0)
            # Compute the derivative for coefficients[j].
            # Save it in a variable called derivative
            derivative = feature_derivative_with_L2(
                errors, feature_matrix[:, j], coefficients[j],
                l2_penalty, is_intercept
            )

            # add the step size times the derivative to the current coefficient
            coefficients[j] += step_size * derivative

        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or \
                (itr <= 1000 and itr % 100 == 0) or \
                (itr <= 10000 and itr % 1000 == 0) or \
                (itr % 10000 == 0):
            lp = compute_log_likelihood_with_L2(
                feature_matrix, sentiment, coefficients, l2_penalty
            )
            print('iteration %*d: log likelihood of observed labels = %.8f' %
                  (int(np.ceil(np.log10(max_iter))), itr, lp))
    return coefficients

# run with L2 = 0
coefficients_0_penalty = logistic_regression_with_L2(
    feature_matrix_train, sentiment_train, initial_coefficients=np.zeros(194),
    step_size=5e-6, l2_penalty=0, max_iter=501
)

# run with L2 = 4
coefficients_4_penalty = logistic_regression_with_L2(
    feature_matrix_train, sentiment_train, initial_coefficients=np.zeros(194),
    step_size=5e-6, l2_penalty=4, max_iter=501
)

# run with L2 = 10
coefficients_10_penalty = logistic_regression_with_L2(
    feature_matrix_train, sentiment_train, initial_coefficients=np.zeros(194),
    step_size=5e-6, l2_penalty=10, max_iter=501
)

# run with L2 = 1e2
coefficients_1e2_penalty = logistic_regression_with_L2(
    feature_matrix_train, sentiment_train, initial_coefficients=np.zeros(194),
    step_size=5e-6, l2_penalty=1e2, max_iter=501
)

# run with L2 = 1e3
coefficients_1e3_penalty = logistic_regression_with_L2(
    feature_matrix_train, sentiment_train, initial_coefficients=np.zeros(194),
    step_size=5e-6, l2_penalty=1e3, max_iter=501
)

# run with L2 = 1e5
coefficients_1e5_penalty = logistic_regression_with_L2(
    feature_matrix_train, sentiment_train, initial_coefficients=np.zeros(194),
    step_size=5e-6, l2_penalty=1e5, max_iter=501
)
