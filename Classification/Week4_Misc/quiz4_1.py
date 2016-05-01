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
loans = loans.drop('bad_loans', axis=1)

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
with open('data/module-6-assignment-validation-idx.json') as f:
    validation_idx = json.load(f)
with open('data/module-6-assignment-train-idx.json') as f:
    train_idx = json.load(f)
train_data = loans.iloc[train_idx]

validation_data = loans.iloc[validation_idx]


def reached_minimum_node_size(data, min_node_size):
    return True if len(data) <= min_node_size else False


def error_reduction(error_before_split, error_after_split):
    return error_before_split - error_after_split


def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: if labels_in_node is empty, return 0
    # print(labels_in_node.head(2))
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
        leaf['prediction'] = 1
    else:
        leaf['prediction'] = -1

    # Return the leaf node
    return leaf


def decision_tree_create(data, features, target, current_depth=0,
                         max_depth=10, min_node_size=1,
                         min_error_reduction=0.0):

    remaining_features = features[:]  # Make a copy of the features.

    target_values = data[target]
    # print("-------------------------------------------------------------------")
    # print("Subtree, depth = %s (%s data points)." %
    #       (current_depth, len(target_values)))

    # Stopping condition 1
    # (Check if there are mistakes at current node.
    if intermediate_node_num_mistakes(target_values) == 0:
        # print("Stopping condition 1 reached.  All data points have same value")
        # If not mistakes at current node, make current node a leaf node
        return create_leaf(target_values)

    # Stopping condition 2
    # (check if there are remaining features to consider splitting on)
    if remaining_features == []:
        # print("Stopping condition 2 reached. No remaining features")
        # If no remaining features to consider, make current node a leaf node
        return create_leaf(target_values)

    # Early stopping condition 1: Reached max depth limit.
    if current_depth >= max_depth:
        # print("Early stopping condition 1 reached. Reached maximum depth.")
        return create_leaf(target_values)

    # Early stopping condition 2: Reached the minimum node size.
    # If the number of data points is less than or equal to the minimum size, return a leaf.
    if len(data) <= min_node_size:
        # print("Early stopping condition 2 reached. Reached minimum node size.")
        return  create_leaf(target_values)

    # Find the best splitting feature
    splitting_feature = best_splitting_feature(data, features, target)

    # Split on the best feature that we found.
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]


    # Early stopping condition 3: Minimum error reduction
    # Calculate the error before splitting (number of misclassified examples
    # divided by the total number of examples)
    error_before_split = intermediate_node_num_mistakes(target_values) / float(
        len(data))

    # Calculate the error after splitting (number of misclassified examples
    # in both groups divided by the total number of examples)
    left_mistakes = intermediate_node_num_mistakes(left_split[target])
    right_mistakes = intermediate_node_num_mistakes(right_split[target])
    error_after_split = (left_mistakes + right_mistakes) / float(len(data))

    # If the error reduction is LESS THAN OR EQUAL TO min_error_reduction, return a leaf.
    if error_reduction(error_before_split, error_after_split) <= \
            min_error_reduction:
        # print("Early stopping condition 3 reached. Minimum error reduction.")
        return create_leaf(target_values)

    remaining_features.drop(splitting_feature)
    # print(
    #     "Split on feature %s. (%s, %s)" % (
    #         splitting_feature, len(left_split), len(right_split)))

    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target,
                                     current_depth + 1, max_depth,
                                     min_node_size,
                                     min_error_reduction
    )

    right_tree = decision_tree_create(
        right_split, remaining_features, target, current_depth + 1, max_depth,
        min_node_size, min_error_reduction
    )

    return {'is_leaf': False,
            'prediction': None,
            'splitting_feature': splitting_feature,
            'left': left_tree,
            'right': right_tree}


my_decision_tree_new = decision_tree_create(train_data, features, 'safe_loans',
                                            max_depth=6,
                                            min_node_size=100,
                                            min_error_reduction=0.0)

my_decision_tree_old = decision_tree_create(train_data, features, 'safe_loans',
                                            max_depth = 6,
                                            min_node_size = 0,
                                            min_error_reduction=-1)


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

print('=======================')
print('One sample data point')
print('=======================\n')


# print(validation_data.iloc[0])
print('Predicted class: %s ' % classify(my_decision_tree_new,
                                        validation_data.iloc[0]))

print("\nnew Tree")
print('----------')
classify(my_decision_tree_new, validation_data.iloc[0], annotate = True)
print('\nOld Tree')
print('------------')
classify(my_decision_tree_old, validation_data.iloc[0], annotate = True)
print()

def evaluate_classification_error(tree, data, target):
    # Apply the classify(tree, x) to each row in your data
    # prediction = data.apply(lambda x: classify(tree, x))
    prediction = pd.Series()
    for i in range(len(data)):
        prediction = prediction.set_value(i, classify(tree, data.iloc[i]))

    # calculate the classification error and return it
    errors = sum(prediction != data[target])
    return errors / len(data)


print('new tree classification error on validation data: %s' %
      evaluate_classification_error(my_decision_tree_new, validation_data,
                                    'safe_loans'))
print('old tree classification error on validation data: %s' %
      evaluate_classification_error(my_decision_tree_old, validation_data,
                                    'safe_loans'))

model_1 = decision_tree_create(train_data, features, 'safe_loans',
                               max_depth=2,
                               min_node_size=0,
                               min_error_reduction=-1)
model_2 = decision_tree_create(train_data, features, 'safe_loans',
                               max_depth=6,
                               min_node_size=0,
                               min_error_reduction=-1)
model_3 = decision_tree_create(train_data, features, 'safe_loans',
                               max_depth=14,
                               min_node_size=0,
                               min_error_reduction=-1)
print()
print("Training data, classification error (model 1):",
      evaluate_classification_error(model_1, train_data, 'safe_loans'))
print("Training data, classification error (model 2):",
      evaluate_classification_error(model_2, train_data, 'safe_loans'))
print("Training data, classification error (model 3):",
      evaluate_classification_error(model_3, train_data, 'safe_loans'))


print()
print("Validation data, classification error (model 1):",
      evaluate_classification_error(model_1, validation_data, 'safe_loans'))
print("Validation data, classification error (model 2):",
      evaluate_classification_error(model_2, validation_data, 'safe_loans'))
print("Validation data, classification error (model 3):",
      evaluate_classification_error(model_3, validation_data, 'safe_loans'))


def count_leaves(tree):
    if tree['is_leaf']:
        return 1
    return count_leaves(tree['left']) + count_leaves(tree['right'])

print()
print("model 1 leaves: %s" % count_leaves(model_1))
print("model 2 leaves: %s" % count_leaves(model_2))
print("model 3 leaves: %s" % count_leaves(model_3))

model_4 = decision_tree_create(train_data, features, 'safe_loans',
                               max_depth=6,
                               min_node_size=0,
                               min_error_reduction=-1)
model_5 = decision_tree_create(train_data, features, 'safe_loans',
                               max_depth=6,
                               min_node_size=0,
                               min_error_reduction=0)
model_6 = decision_tree_create(train_data, features, 'safe_loans',
                               max_depth=6,
                               min_node_size=0,
                               min_error_reduction=5)

print()
print("Validation data, classification error (model 4):",
      evaluate_classification_error(model_4, validation_data, 'safe_loans'))
print("Validation data, classification error (model 5):",
      evaluate_classification_error(model_5, validation_data, 'safe_loans'))
print("Validation data, classification error (model 6):",
      evaluate_classification_error(model_6, validation_data, 'safe_loans'))

print()
print("model 4 leaves: %s" % count_leaves(model_4))
print("model 5 leaves: %s" % count_leaves(model_5))
print("model 6 leaves: %s" % count_leaves(model_6))


model_7 = decision_tree_create(train_data, features, 'safe_loans',
                               max_depth=6,
                               min_node_size=0,
                               min_error_reduction=-1)
model_8 = decision_tree_create(train_data, features, 'safe_loans',
                               max_depth=6,
                               min_node_size=2000,
                               min_error_reduction=-1)
model_9 = decision_tree_create(train_data, features, 'safe_loans',
                               max_depth=6,
                               min_node_size=50000,
                               min_error_reduction=-1)


print()
print("Validation data, classification error (model 7):",
      evaluate_classification_error(model_7, validation_data, 'safe_loans'))
print("Validation data, classification error (model 8):",
      evaluate_classification_error(model_8, validation_data, 'safe_loans'))
print("Validation data, classification error (model 9):",
      evaluate_classification_error(model_9, validation_data, 'safe_loans'))

print()
print("model 7 leaves: %s" % count_leaves(model_7))
print("model 8 leaves: %s" % count_leaves(model_8))
print("model 9 leaves: %s" % count_leaves(model_9))