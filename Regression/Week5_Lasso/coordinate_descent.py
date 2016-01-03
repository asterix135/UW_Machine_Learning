"""
Implement coordinate descent for Lasso Regression
"""

import os
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from math import log, sqrt


def get_numpy_data(df, features, output):
    """
    A function that takes a data set, a list of features (e.g. [‘sqft_living’,
    ‘bedrooms’]), to be used as inputs, and a name of the output (e.g.
    ‘price’). This function should return a features_matrix (2D array)
    consisting of first a column of ones followed by columns containing the
    values of the input features in the data set in the same order as the input
    list. It should also return an output_array which is an array of the values
    of the output in the data set (e.g. ‘price’)
    :param df: Pandas data frame (data set)
    :param features: list of features (column names)
    :param output: name of output column
    :return: 2 numpy arrays
    """
    df['constant'] = 1  # add a constant column to data frame
    # prepend variable 'constant' to the features list
    features = ['constant'] + features
    features_df = df[features]
    features_matrix = features_df.values
    output_array = df[output].values  # df[[output]].values
    return features_matrix, output_array


def predict_outcome(feature_matrix, weights):
    """
    Calculates predicted outcomes based on regression values
    :param feature_matrix: 2D array of features
    :param weights: 1D array of weightes
    :return: 1D array of predicted outcomes
    """
    return np.dot(feature_matrix, weights)


def normalize_features(features):
    """
    Normalizes columns of a features matrix
    :param features: numpy array
    :return features_normalized: numpy array
    """
    norms = np.linalg.norm(features, axis=0)
    features_normalized = features/norms
    return features_normalized, norms


def normalize_weights(weights, norms):
    """
    Takes weights from normalized regression and normalizes them for prediction
    :param weights: numpy array
    :param norms: numpy array
    :return weights_normalized: numpy array:
    """
    weights_normalized = weights/norms
    return weights_normalized


def lasso_coordinate_descent_step(i, feature_matrix, output,
                                  weights, l1_penalty):
    """
    Calculates new weight for a parameter based on Lasso model
    :param i: integer identifying column
    :param feature_matrix: numpy array
    :param output: numpy array
    :param weights: list
    :param l1_penalty: number - penalty value
    :return new_weight_i: new weight for parameter
    """
    # compute prediction
    prediction = predict_outcome(feature_matrix, weights)
    # compute ro[i] =
    # SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = sum(feature_matrix[:, i] *
               (output - prediction + weights[i] * feature_matrix[:, i]))

    if i == 0:  # intercept -- do not regularize
        new_weight_i = ro_i
    elif ro_i < -l1_penalty/2.:
        new_weight_i = ro_i + l1_penalty/2
    elif ro_i > l1_penalty/2.:
        new_weight_i = ro_i - l1_penalty/2
    else:
        new_weight_i = 0.

    return new_weight_i


def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights,
                                      l1_penalty, tolerance):
    weights = np.array(initial_weights)
    max_change = float('inf')
    while max_change > tolerance:
        max_change = 0
        for i in range(len(initial_weights)):
            new_weight_i = lasso_coordinate_descent_step(i, feature_matrix,
                                                         output, weights,
                                                         l1_penalty)
            change = abs(weights[i] - new_weight_i)
            max_change = change if change > max_change else max_change
            weights[i] = new_weight_i
    return weights


def question1(df):
    """
    Find ro values for 2-feature regression
    :param df: numpy data frame
    :return:
    """
    features_list = ['sqft_living', 'bedrooms']
    q1_features, q1_output = get_numpy_data(df, features_list, 'price')
    norms, dummy = normalize_features(q1_features)
    initial_weights = [1, 4, 1]
    # initial_weights = np.array([1, 4, 1])
    preds = predict_outcome(norms, initial_weights)
    ro_vals = []
    print('ro values')
    for i in range(1, len(features_list) + 1):
        ro = sum(norms[:, i] *
                 (q1_output - preds + initial_weights[i] * norms[:, i]))
        # (initial_weights[i] * norms[i])
        print('{:.2E}'.format(ro * 2))
        ro_vals.append(ro)
    print()
    return ro_vals


def question2(df):
    features_list = ['sqft_living', 'bedrooms']
    q2_features, q2_output = get_numpy_data(df, features_list, 'price')
    norms, dummy = normalize_features(q2_features)
    initial_weights = np.zeros(3)
    l1_penalty = 1e7
    tolerance = 1.0
    new_weights = lasso_cyclical_coordinate_descent(norms, q2_output,
                                                    initial_weights,
                                                    l1_penalty, tolerance)
    preds = predict_outcome(norms, new_weights)
    rss = sum((preds - q2_output)**2)
    print('RSS = ' + str(rss))
    print('Weights: ' + str(new_weights) + '\n')


def question3(train, test):
    """
    Evaluate Lasso with more features
    :param train: pandas dataframe
    :param test: pandas dataframe
    :return:
    """
    features_list = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                     'floors', 'waterfront', 'view', 'condition', 'grade',
                     'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']
    # create normalized feature matrix from training data
    train_features, train_output = get_numpy_data(train, features_list, 'price')
    train_norm, norm_vals = normalize_features(train_features)

    # learn weights with penalty = 1e7
    initial_weights = np.zeros(len(features_list) + 1)
    l1_penalty = 1e7
    tolerance = 1.0
    weights1e7 = lasso_cyclical_coordinate_descent(train_norm, train_output,
                                                    initial_weights, l1_penalty,
                                                    tolerance)
    # Show coefficients with non-zero weights
    print('non zero coefficients with lambda1 = 1e7')
    print(np.array(['intercept'] + features_list)[weights1e7 != 0])

    # learn weights with penalty = 1e8
    initial_weights = np.zeros(len(features_list) + 1)
    l1_penalty = 1e8
    tolerance = 1.0
    weights1e8 = lasso_cyclical_coordinate_descent(train_norm, train_output,
                                                    initial_weights, l1_penalty,
                                                    tolerance)
    # Show coefficients with non-zero weights
    print('\nnon zero coefficients with lambda1 = 1e8')
    print(np.array(['intercept'] + features_list)[weights1e8 != 0])

    # learn weights with penalty = 1e4
    initial_weights = np.zeros(len(features_list) + 1)
    l1_penalty = 1e4
    tolerance = 5e5
    weights1e4 = lasso_cyclical_coordinate_descent(train_norm, train_output,
                                                    initial_weights, l1_penalty,
                                                    tolerance)
    # Show coefficients with non-zero weights
    print('\nnon zero coefficients with lambda1 = 1e4')
    print(np.array(['intercept'] + features_list)[weights1e4 != 0])

    # Normalize weights
    norm_1e7wts = normalize_weights(weights1e7, norm_vals)
    norm_1e8wts = normalize_weights(weights1e8, norm_vals)
    norm_1e4wts = normalize_weights(weights1e4, norm_vals)

    # get predicted values for each model
    test_features, test_output = get_numpy_data(test, features_list, 'price')
    preds_1e7 = predict_outcome(test_features, norm_1e7wts)
    preds_1e8 = predict_outcome(test_features, norm_1e8wts)
    preds_1e4 = predict_outcome(test_features, norm_1e4wts)

    # calculate rss for each model
    rss_1e7 = sum((preds_1e7 - test_output)**2)
    rss_1e8 = sum((preds_1e8 - test_output)**2)
    rss_1e4 = sum((preds_1e4 - test_output)**2)

    ## print results to console
    print('\nRSS for 1e7 model: ' + str(rss_1e7))
    print('RSS for 1e8 model: ' + str(rss_1e8))
    print('RSS for 1e4 model: ' + str(rss_1e4))




def wrapper():
    dtype_dict = {'bathrooms': float,
                  'waterfront': int,
                  'sqft_above': int,
                  'sqft_living15': float,
                  'grade': int,
                  'yr_renovated': int,
                  'price': float,
                  'bedrooms': float,
                  'zipcode': str,
                  'long': float,
                  'sqft_lot15': float,
                  'sqft_living': float,
                  'floors': float,
                  'condition': int,
                  'lat': float,
                  'date': str,
                  'sqft_basement': int,
                  'yr_built': int,
                  'id': str,
                  'sqft_lot': int,
                  'view': int}
    os.chdir('..')
    sales = pd.read_csv('kc_house_data.csv')
    testing = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)
    training = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)
    question1(sales)
    question2(sales)
    question3(training, testing)

    # os.chdir('..')
    # select_l2(dtype_dict)


if __name__ == '__main__':
    wrapper()
