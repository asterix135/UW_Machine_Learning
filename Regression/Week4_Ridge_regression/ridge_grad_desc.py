import os
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
# TODO: Step 5 in instructionsS


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


def feature_derivative_ridge(errors, feature, weight, l2_penalty,
                             feature_is_constant):
    if feature_is_constant:
        derivative = 2 * np.dot(feature, errors)
    else:
        derivative = 2 * np.dot(feature, errors) + 2 * l2_penalty * weight
    return derivative


def ridge_regression_gradient_descent(feature_matrix, output, initial_weights,
                                      step_size, l2_penalty,
                                      max_iterations=100):
    weights = np.array(initial_weights)  # make sure it's a numpy array
    for j in range(max_iterations):
        # while not reached maximum number of iterations:
        # compute the predictions using your predict_output() function
        preds = predict_outcome(feature_matrix, weights)

        # compute the errors as predictions - output
        errors = preds - output

        for i in range(len(weights)):  # loop over each weight
            feature_is_constant = True if i == 0 else False
            feature = feature_matrix[:,i]
            # compute the derivative for weight[i].
            derivative = feature_derivative_ridge(errors, feature, weights[i],
                                                  l2_penalty,
                                                  feature_is_constant)
            # subtract step size times the derivative from the current weight
            weights[i] -= step_size * derivative
    return weights


def test1(sales):
    (example_features, example_output) = \
        get_numpy_data(sales, ['sqft_living'], 'price')
    my_weights = np.array([1., 10.])
    test_predictions = predict_outcome(example_features, my_weights)
    errors = test_predictions - example_output  # prediction errors

    # next two lines should print the same values
    print('test1')
    print(feature_derivative_ridge(errors, example_features[:, 1],
                                   my_weights[1], 1, False))
    print(np.sum(errors*example_features[:, 1])*2+20.)
    print('')

    # next two lines should print the same values
    print('test2')
    print(feature_derivative_ridge(errors, example_features[:, 0],
                                   my_weights[0], 1, True))
    print(np.sum(errors)*2.)


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
    sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
    sales = sales.sort_values(by=['sqft_living', 'price'])
    #test1(sales)
    test_data = pd.read_csv('kc_house_test_data.csv')
    train_data = pd.read_csv('kc_house_train_data.csv')
    os.chdir('Week4_Ridge_regression')
    simple_features = ['sqft_living']
    my_output = 'price'
    (simple_feature_matrix, output) = get_numpy_data(train_data,
                                                     simple_features, my_output)
    (simple_test_feature_matrix, test_output) = get_numpy_data(test_data,
                                                               simple_features,
                                                               my_output)
    # Run Ridge regression with zero penalty
    l2_penalty = 0
    initial_weights = np.zeros(2)
    step_size = 1e-12
    max_iterations = 1000
    simple_weights_0_penalty = \
        ridge_regression_gradient_descent(simple_feature_matrix, output,
                                          initial_weights, step_size,
                                          l2_penalty, max_iterations)
    print("zero weight coefs: " + str(simple_weights_0_penalty))

    # Run Ridge regression with high penalty
    l2_penalty = 1e11
    simple_weights_high_penalty = \
        ridge_regression_gradient_descent(simple_feature_matrix, output,
                                          initial_weights, step_size,
                                          l2_penalty, max_iterations)
    print('high weight coefs: ' + str(simple_weights_high_penalty) + '\n')

    # compare lines
    plt.plot(simple_feature_matrix, output, 'k.', simple_feature_matrix,
             predict_outcome(simple_feature_matrix, simple_weights_0_penalty),
             'b-', simple_feature_matrix,
             predict_outcome(simple_feature_matrix,
                             simple_weights_high_penalty),'r-')
    plt.savefig('simple_comparison.png')

    # Compute RSS
    test_pred_zero = predict_outcome(simple_test_feature_matrix,
                                     simple_weights_0_penalty)
    test_pred_high = predict_outcome(simple_test_feature_matrix,
                                     simple_weights_high_penalty)
    rss_zero = sum((test_pred_zero - test_output)**2)
    rss_high = sum((test_pred_high - test_output)**2)
    print('zero weight rss: ' + str('%.3e' % rss_zero))
    print('high weight rss: ' + str('%.3e' % rss_high) + '\n')

    # Two feature model
    model_features = ['sqft_living', 'sqft_living15']
    my_output = 'price'
    (feature_matrix, output) = get_numpy_data(train_data,
                                              model_features, my_output)
    (test_feature_matrix, test_output) = get_numpy_data(test_data,
                                                        model_features,
                                                        my_output)

    # Run Ridge regression with zero penalty
    l2_penalty = 0
    initial_weights = np.zeros(3)
    step_size = 1e-12
    max_iterations = 1000
    multiple_weights_0_penalty = \
        ridge_regression_gradient_descent(feature_matrix, output,
                                          initial_weights, step_size,
                                          l2_penalty, max_iterations)
    print("multiple 0 weight coefs: " + str(multiple_weights_0_penalty))

    # with high penalty
    l2_penalty = 1e11
    multiple_weights_high_penalty = \
        ridge_regression_gradient_descent(feature_matrix, output,
                                          initial_weights, step_size,
                                          l2_penalty, max_iterations)
    print('high weight coefs: ' + str(multiple_weights_high_penalty) + '\n')

    # Compute RSS
    test_pred_m_zero = predict_outcome(test_feature_matrix,
                                       multiple_weights_0_penalty)
    test_pred_m_high = predict_outcome(test_feature_matrix,
                                       multiple_weights_high_penalty)
    rss_m_zero = sum((test_pred_m_zero - test_output)**2)
    rss_m_high = sum((test_pred_m_high - test_output)**2)
    print('multiple zero weight rss: ' + str('%.3e' % rss_m_zero))
    print('multiple high weight rss: ' + str('%.3e' % rss_m_high) + '\n')

    # Get first house value
    print('first house no regularization: ' + str(test_pred_m_zero[0]))
    print('first house regularization: ' + str(test_pred_m_high[0]))
    print('actual house price: ' + str(test_data['price'][0]))
    print('no reg difference: ' + str(test_data['price'][0] -
                                      test_pred_m_zero[0]))
    print('reg difference: ' + str(test_data['price'][0] -
                                   test_pred_m_high[0]))


if __name__ == '__main__':
    wrapper()
