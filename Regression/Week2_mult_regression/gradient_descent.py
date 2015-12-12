import numpy as np
import pandas as pd
import os


def get_numpy_data(df, features, output):
    df['constant'] = 1  # add a constant column to data frame
    # prepend variable 'constant' to the features list
    features = ['constant'] + features
    features_df = df[features]
    features_matrix = features_df.values
    output_array = df[[output]].values
    return features_matrix, output_array


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
    train = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)
    test = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)
    all = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)


if __name__ == '__main__':
    wrapper()
