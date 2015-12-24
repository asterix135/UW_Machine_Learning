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


if __name__ == '__main__':
    wrapper()
