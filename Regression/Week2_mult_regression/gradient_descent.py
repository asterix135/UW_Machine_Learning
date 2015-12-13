import numpy as np
import pandas as pd
import os


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


def feature_derivative(errors, feature):
    """
    Calculates derivative of the regression cost function with respect to the
    weight of ‘feature’
    :param errors: 1D array
    :param feature: 1D array
    :return: number
    """
    return 2 * np.dot(feature, errors)


def regression_gradient_descent(feature_matrix, output, initial_weights,
                                step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        predictions = predict_outcome(feature_matrix, weights)
        # compute the errors as predictions - output:
        errors = predictions - output

        gradient_sum_squares = 0  # initialize the gradient
        # while not converged, update each weight individually:
        for i in range(len(weights)):
            # feature_matrix[:, i] is the feature column for weights[i]
            # compute the derivative for weight[i]:
            feature = feature_matrix[:, i]
            derivative = feature_derivative(errors, feature)
            # add the squared derivative to the gradient magnitude

            gradient_sum_squares += derivative ** 2

            # update the weight based on step size and derivative:
            weights[i] -= step_size * derivative

        gradient_magnitude = np.sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)


def model1(test_data, train_data):
    # Build model & get coefficients
    simple_features = ['sqft_living']
    my_output= 'price'
    simple_feature_matrix, output = get_numpy_data(train_data, simple_features,
                                                   my_output)
    initial_weights = np.array([-47000., 1.])
    step_size = 7e-12
    tolerance = 2.5e7
    simple_weights = regression_gradient_descent(simple_feature_matrix, output,
                                                 initial_weights, step_size,
                                                 tolerance)
    print(simple_weights)

    # Predict on test set
    simple_test_matrix, test_output = get_numpy_data(test_data, simple_features,
                                                     my_output)
    test_pred = predict_outcome(simple_test_matrix, simple_weights)
    print(test_pred[0])
    rss = np.mean((test_pred - test_output)**2)
    print('model 1 RSS: ' + str(rss))


def model2(test_data, train_data):
    # Build model & get coefficients
    model_features = ['sqft_living', 'sqft_living15']
    my_output= 'price'
    feature_matrix, output = get_numpy_data(train_data, model_features,
                                                   my_output)
    initial_weights = np.array([-100000., 1., 1.])
    step_size = 4e-12
    tolerance = 1e9
    mod2_weights = regression_gradient_descent(feature_matrix, output,
                                               initial_weights, step_size,
                                               tolerance)
    # print(mod2_weights)

    # Predict on test set
    simple_test_matrix, test_output = get_numpy_data(test_data, model_features,
                                                     my_output)
    test_pred = predict_outcome(simple_test_matrix, mod2_weights)
    print(test_pred[0])
    rss = np.mean((test_pred - test_output)**2)
    print('model 2 RSS: ' + str(rss))


def wrapper():
    os.chdir('..')
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
                  'floors': str,
                  'condition': int,
                  'lat': float,
                  'date': str,
                  'sqft_basement': int,
                  'yr_built': int,
                  'id': str,
                  'sqft_lot': int,
                  'view': int}
    train_data = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)
    test_data = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)
    # all = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
    model1(test_data, train_data)
    model2(test_data, train_data)
    print(test_data['price'][0])



if __name__ == '__main__':
    wrapper()
