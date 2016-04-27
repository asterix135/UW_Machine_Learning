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
loans = loans.fillna(0)
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
    safe_loans = labels_in_node[labels_in_node == 1].count()
    # count number of -1s (risky loans)
    risky_loans = labels_in_node[labels_in_node == -1].count()
    # return number of mistakes
    if safe_loans > risky_loans:
        return risky_loans
    else:
        return safe_loans


# Test case 1
example_labels = pd.Series([-1, -1, 1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print('Test passed!')
else:
    print('Test 1 failed... try again!')

# Test case 2
example_labels = pd.Series([-1, -1, 1, 1, 1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print('Test passed!')
else:
    print('Test 2 failed... try again!')

# Test case 3
example_labels = pd.Series([-1, -1, -1, -1, -1, 1, 1])
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
        left_split = target_values[data[feature] == 0]

        # The right split will have all data points where the feature value is 1
        right_split = target_values[data[feature] == 1]

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


def decision_tree_create(data, features, target, current_depth=0, max_depth=10):
    remaining_features = features[:] # Make a copy of the features.
    target_values = data[target]
    print("-------------------------------------------------------------------")
    print("Subtree, depth = %s (%s data points)." %
          (current_depth, len(target_values)))

    # Stopping condition 1
    # (Check if there are mistakes at current node.
    if intermediate_node_num_mistakes(target_values) == 0:
        print("Stopping condition 1 reached.")
        # If not mistakes at current node, make current node a leaf node
        return create_leaf(target_values)

    # Stopping condition 2
    # (check if there are remaining features to consider splitting on)
    if remaining_features == []:
        print("Stopping condition 2 reached.")
        # If no remaining features to consider, make current node a leaf node
        return create_leaf(target_values)

    # Additional stopping condition (limit tree depth)
    if current_depth >= max_depth:
        print("Reached maximum depth. Stopping for now.")
        # If the max tree depth has been reached, make current node a leaf node
        return create_leaf(target_values)

    # Find the best splitting feature
    splitting_feature = best_splitting_feature(data, features, target)

    # Split on the best feature that we found.
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    remaining_features = remaining_features.drop(splitting_feature)
    print("Split on feature %s. (%s, %s)" % (
                      splitting_feature, len(left_split), len(right_split)))

    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print("Creating leaf node.")
        return create_leaf(left_split[target])
    if len(right_split) == len(data):
        print("Creating leaf node.")
        return create_leaf(right_split[target])


    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(
        left_split, remaining_features, target, current_depth + 1, max_depth)
    right_tree = decision_tree_create(
        right_split, remaining_features, target, current_depth + 1, max_depth)

    return {'is_leaf'          : False,
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree,
            'right'            : right_tree}


# Checkpoint
def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])


small_data_decision_tree = decision_tree_create(train_data, features,
                                                'safe_loans', max_depth = 3)
if count_nodes(small_data_decision_tree) == 13:
    print('Test passed!')
else:
    print('Test failed... try again!')
    print('Number of nodes found                :',
          count_nodes(small_data_decision_tree))
    print('Number of nodes that should be there : 13')

# Build the Tree!!

my_decision_tree = decision_tree_create(train_data, features,
                                        'safe_loans', max_depth = 6)


# Classify test data
def classify(tree, x, annotate=False):
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate:
            print(
                "At leaf, predicting %s" % tree['prediction'])
        return tree['prediction']
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate:
            print(
                "Split on %s = %s" % (
                tree['splitting_feature'], split_feature_value))
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)

print('\n\n============================================================='
      '\nClassification starts here'
      '============================================================\n\n')
print(test_data.iloc[0])
print('\n\nPredicted class: %s ' %
      classify(my_decision_tree, test_data.iloc[0]))
print('\n\n')

classify(my_decision_tree, test_data.iloc[0], annotate=True)


# Evaluate Tree
def evaluate_classification_error(tree, data, target):
    # Apply the classify(tree, x) to each row in your data
    # prediction = data.apply(lambda x: classify(tree, x))
    prediction = pd.Series()
    for i in range(len(data)):
        prediction = prediction.set_value(i, classify(tree, data.iloc[i]))

    # calculate the classification error and return it
    errors = sum(prediction != data[target])
    return errors / len(data)


print('\n================================\nQuiz Answer')
print(round(evaluate_classification_error(my_decision_tree,
                                          test_data, 'safe_loans'), 2))
print('================================\n')


def print_stump(tree, name = 'root'):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print("(leaf, label: %s)" % tree['prediction'])
        return None
    split_feature, split_value = split_name.split('_')
    print('                       %s' % name)
    print('         |---------------|----------------|')
    print('         |                                |')
    print('         |                                |')
    print('         |                                |')
    print('  [{0} == 0]               [{0} == 1]    '.format(split_name))
    print('         |                                |')
    print('         |                                |')
    print('         |                                |')
    print('    (%s)                         (%s)'
        % (('leaf, label: ' + str(tree['left']['prediction']) if
            tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if
            tree['right']['is_leaf'] else 'subtree')))


print_stump(my_decision_tree)

print_stump(my_decision_tree['left'], my_decision_tree['splitting_feature'])

print_stump(my_decision_tree['left']['left'],
            my_decision_tree['left']['splitting_feature'])
