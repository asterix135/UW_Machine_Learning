"""
Implmentation of Decision Tree algorithm
"""

import json
import numpy as np
import os
import pandas as pd

####################
# Data preparation
####################

# Load lending club dataset
os.chdir('..')
loans = pd.read_csv('data/lending-club-data.csv')

# create column to indicate safe/unsafe loan coded as +1/-1
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x == 0 else -1)

# Extract features and target value from full data frame
features = ['grade',  # grade of the loan
            'emp_length',  # number of years of employment
            'home_ownership',  # home_ownership status: own, mortgage or rent
            'term',  # the term of the loan
            ]
target = 'safe_loans'  # prediction target (y) (+1 means safe, -1 is risky)
loans = loans[features + [target]]

# Unpack categorical variables (One-hot encoding)
loans = pd.get_dummies(loans)

# rename values in features list
features = loans.columns
features = features.drop('safe_loans')  # Remove the response variable

# Split test/train with provided indices
with open('data/module-5-assignment-2-test-idx.json') as f:
    test_idx = json.load(f)
with open('data/module-5-assignment-2-train-idx.json') as f:
    train_idx = json.load(f)
train_data = loans.iloc[train_idx]
test_data = loans.iloc[test_idx]

####################
# Tree implementation
####################


def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: if labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0
    # count number of 1s (safe loans)
    safe_loans = int(labels_in_node[labels_in_node == 1].count()[0])
    # count number of -1s (risky loans)
    risky_loans = int(labels_in_node[labels_in_node == -1].count()[0])
    # return number of mistakes
    if safe_loans > risky_loans:
        return risky_loans
    else:
        return safe_loans


# Test case 1
example_labels = pd.DataFrame([-1, -1, 1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print('Test passed!')
else:
    print('Test 1 failed... try again!')

# Test case 2
example_labels = pd.DataFrame([-1, -1, 1, 1, 1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print('Test passed!')
else:
    print('Test 2 failed... try again!')

# Test case 3
example_labels = pd.DataFrame([-1, -1, -1, -1, -1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print('Test passed!')
else:
    print('Test 3 failed... try again!')


def best_splitting_feature(data, features, target):

    target_values = data[target]
    best_feature = None  #  Keep track of the best feature
    best_error = 10      #  Keep track of the best error so far
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))

    # Loop through each feature to consider splitting on that feature
    for feature in features:

        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]

        # The right split will have all data points where the feature value is 1
        right_split =  data[data[feature] == 1]

        # Calculate the number of misclassified examples in the left split.
        left_mistakes = intermediate_node_num_mistakes(left_split)

        # Calculate the number of misclassified examples in the right split.
        right_mistakes = intermediate_node_num_mistakes(right_split)

        # Compute the classification error of this split.
        # Error = (# mistakes (left) + # mistakes (right)) / (# of data points)
        error = (left_mistakes + right_mistakes) / num_data_points

        # If this is the best error we have found so far,
        # store the feature as best_feature and the error as best_error
        if error < best_error:
            best_feature = feature
            best_error = error

    return best_feature  # Return the best feature we found

foo = best_splitting_feature(train_data, features, 'safe_loans')
if best_splitting_feature(train_data, features, 'safe_loans') == \
        'term_ 36 months':
    print('Test passed!')
else:
    print('Test failed... try again!')
    print('best splitting feature: ' + str(
        best_splitting_feature(train_data, features, 'safe_loans')
    ))


# Building the tree

def create_leaf(target_values):
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True}

    # Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])

    # For the leaf node, set the prediction to be the majority class.
    # Store the predicted class (1 or -1) in leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] =  1
    else:
        leaf['prediction'] =  -1

    # Return the leaf node
    return leaf


