import os
import numpy as np
import pandas as pd
from sklearn import linear_model
from math import sqrt


ALL_FEATURES = ['bedrooms', 'bedrooms_square',
                'bathrooms',
                'sqft_living', 'sqft_living_sqrt',
                'sqft_lot', 'sqft_lot_sqrt',
                'floors', 'floors_square',
                'waterfront', 'view', 'condition', 'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built', 'yr_renovated']


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
    model_all = linear_model.Lasso(alpha=5e2, normalize=True)  # set parameters
    model_all.fit(df[ALL_FEATURES], df['price'])  # learn weights
    # model_all.fit(df, df['price'])
    print('Question 1 - selected columns')

    # print(df.columns.values[model_all.coef_ != 0])
    print(np.array(ALL_FEATURES)[model_all.coef_ != 0])
    print()
    return model_all


def question2(train, valid, test):
    """
    Find Best l1 value for Lasso regression
    :param train: pandas dataframe
    :param valid: pandas dataframe
    :param test: pandas dataframe
    :return:
    """
    best_rss = float('inf')

    # figure out best penalty for Lasso
    for penalty in np.logspace(1, 7, num=13):
        model = linear_model.Lasso(alpha=penalty, normalize=True)
        model.fit(train[ALL_FEATURES], train['price'])
        rss = sum((model.predict(valid[ALL_FEATURES]) - valid['price'])**2)
        if rss < best_rss:
            best_rss, best_penalty = rss, penalty
            best_model = model
    print('best L1 on validation set: ' + str(best_penalty) + '\n')

    # Calculate non-zero coefficient in model
    print('\nNonzero Weights: ' + str(np.count_nonzero(best_model.coef_) +
                                      np.count_nonzero(best_model.intercept_)))

    # calculate RSS on test data
    print('RSS on test data:')
    print('{:f}'.format(
            sum((best_model.predict(test[ALL_FEATURES]) - test['price'])**2)))
    print()


def question3(train, valid, test):
    """
    Fit Lasso model with max 7 features
    :param train: pandas dataframe
    :param valid: pandas dataframe
    :param test: pandas dataframe
    :return:
    """
    # find range that has appx the right number of non-zero coefficients
    max_nonzeros = 7
    non_zeros = []
    l1_penalty_max = 0
    for l1_penalty in np.logspace(1, 4, num=20):
        model = linear_model.Lasso(alpha=l1_penalty, normalize=True)
        model.fit(train[ALL_FEATURES], train['price'])
        num_coefs = np.count_nonzero(model.coef_) + \
                    np.count_nonzero(model.intercept_)
        if num_coefs > max_nonzeros:
            l1_penalty_min = l1_penalty
        elif num_coefs < max_nonzeros and num_coefs > l1_penalty_max:
            l1_penalty_max = l1_penalty
        non_zeros.append([l1_penalty, num_coefs])
    print('l1_penalty_min: ' + str(l1_penalty_min))
    print('l1_penalty_max: ' + str(l1_penalty_max))

    # find best model within smaller range
    best_rss = float('inf')

    for l1_penalty in np.linspace(l1_penalty_min,l1_penalty_max,20):
        model = linear_model.Lasso(alpha=l1_penalty, normalize=True)
        model.fit(train[ALL_FEATURES], train['price'])
        sparsity = np.count_nonzero(model.coef_) + \
                   np.count_nonzero(model.intercept_)
        if sparsity == max_nonzeros:
            rss = sum((model.predict(valid[ALL_FEATURES]) - valid['price'])**2)
            # print(l1_penalty, rss)
            if rss < best_rss:
                best_rss, best_penalty = rss, l1_penalty
                best_model = model
    print('best l1 penalty: ' + str(best_penalty) + '\n')

    # find RSS prediction
    print('RSS on test data for 7 variables:')
    print('{:f}'.format(
            sum((best_model.predict(test[ALL_FEATURES]) - test['price'])**2)))
    print('\n7 Feature Model Columns')
    print(np.array(ALL_FEATURES)[best_model.coef_ != 0])


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

    question3(training, validation, testing)
    # os.chdir('..')
    # select_l2(dtype_dict)


if __name__ == '__main__':
    wrapper()
