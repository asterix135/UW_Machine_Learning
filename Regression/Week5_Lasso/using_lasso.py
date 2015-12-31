import os
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from math import log, sqrt


def add_features(df):
    """
    Adds manipulated features to data frame
    :param df:
    :return:
    """
    df['sqft_living_sqrt'] = df['sqft_living'].apply(sqrt)
    df['sqft_lot_sqrt'] = df['sqft_lot'].apply(sqrt)
    df['bedrooms_square'] = df['bedrooms'] * df['bedrooms']
    df['floors_square'] = df['floors'] * df['floors']


def question1(df):
    """
    prints features selected by simple lasso model
    :param df: pandas data frame
    :return: model
    """
    df_x = df.drop(['price', 'date'], axis=1)
    model_all = linear_model.Lasso(alpha=5e2, normalize=True)  # set parameters
    model_all.fit(df_x, df['price'])  # learn weights
    # model_all.fit(df, df['price'])
    print('Question 1 - selected columns')
    print(df_x.columns.values[model_all.coef_ != 0])
    print()
    return model_all


def question2(train, valid, test):
    best_rss = float('inf')
    train_x = train.drop(['price', 'date'], axis=1)
    valid_x = valid.drop(['price', 'date'], axis=1)
    test_x = test.drop(['price', 'date'], axis=1)

    # figure out best penalty for Lasso
    for penalty in np.logspace(3, 9, num=13):
        model = linear_model.Lasso(alpha=penalty, normalize=True)
        model.fit(train_x, train['price'])
        rss = sum((model.predict(valid_x) - valid['price'])**2)
        if rss < best_rss:
            best_rss, best_penalty = rss, penalty
            best_model = model
    print('best L1 on validation set: ' + str(best_penalty) + '\n')

    # calculate RSS on test data

    print('RSS on test data:')
    print(sum((best_model.predict(test_x) - test['price'])**2))

    print('\nNonzero Weights: ' + str(np.count_nonzero(best_model.coef_) +
                                      np.count_nonzero(best_model.intercept_)))


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
    add_features(sales)
    question1(sales)

    testing = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
    add_features(testing)
    training = pd.read_csv('wk3_kc_house_train_data.csv', dtype=dtype_dict)
    add_features(training)
    validation = pd.read_csv('wk3_kc_house_valid_data.csv', dtype=dtype_dict)
    add_features(validation)

    question2(training, validation, testing)
    # os.chdir('..')
    # select_l2(dtype_dict)


if __name__ == '__main__':
    wrapper()
