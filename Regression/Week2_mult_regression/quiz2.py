import os
import pandas as pd
import numpy as np
from sklearn import linear_model


def add_variables(df):
    df['bedrooms_squared'] = df['bedrooms']**2
    df['bed_bath_rooms'] = df['bedrooms'] * df['bathrooms']
    df['log_sqft_living'] = np.log(df['sqft_living'])
    df['lat_plus_long'] = df['lat'] + df['long']


def get_new_means(df):
    print('Means of New Variables')
    print('bedrooms_squared: ' + str(np.mean(df['bedrooms_squared'])))
    print('bed_bath_rooms: ' + str(np.mean(df['bed_bath_rooms'])))
    print('log_sqft_living: ' + str(np.mean(df['log_sqft_living'])))
    print('lat_plus_long: ' + str(np.mean(df['lat_plus_long'])) + '\n')


def get_regression_coefficients(df_train, df_test):
    mod1 = linear_model.LinearRegression()
    mod2 = linear_model.LinearRegression()
    mod3 = linear_model.LinearRegression()
    mod1.fit(df_train[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']],
             df_train['price'])
    mod2.fit(df_train[['sqft_living',
                       'bedrooms',
                       'bathrooms',
                       'lat',
                       'long',
                       'bed_bath_rooms']],
             df_train['price'])
    mod3.fit(df_train[['sqft_living',
                       'bedrooms',
                       'bathrooms',
                       'lat',
                       'long',
                       'bed_bath_rooms',
                       'bedrooms_squared',
                       'log_sqft_living',
                       'lat_plus_long']],
             df_train['price'])
    print('bathroom coef, model 1: ', str(mod1.coef_[2]))
    print('bathroom coef, model 2: ', str(mod2.coef_[2]))
    mod1_train_pred = mod1.predict(df_train[['sqft_living',
                                             'bedrooms',
                                             'bathrooms',
                                             'lat',
                                             'long']])
    mod2_train_pred = mod2.predict(df_train[['sqft_living',
                                             'bedrooms',
                                             'bathrooms',
                                             'lat',
                                             'long',
                                             'bed_bath_rooms']])
    mod3_train_pred = mod3.predict(df_train[['sqft_living',
                                             'bedrooms',
                                             'bathrooms',
                                             'lat',
                                             'long',
                                             'bed_bath_rooms',
                                             'bedrooms_squared',
                                             'log_sqft_living',
                                             'lat_plus_long']])
    print('mod1 train RSS: ' + str(np.mean((mod1_train_pred -
                                            df_train['price'])**2)))
    print('mod2 train RSS: ' + str(np.mean((mod2_train_pred -
                                            df_train['price'])**2)))
    print('mod3 train RSS: ' + str(np.mean((mod3_train_pred -
                                            df_train['price'])**2)))
    mod1_test_pred = mod1.predict(df_test[['sqft_living',
                                           'bedrooms',
                                           'bathrooms',
                                           'lat',
                                           'long']])
    mod2_test_pred = mod2.predict(df_test[['sqft_living',
                                           'bedrooms',
                                           'bathrooms',
                                           'lat',
                                           'long',
                                           'bed_bath_rooms']])
    mod3_test_pred = mod3.predict(df_test[['sqft_living',
                                           'bedrooms',
                                           'bathrooms',
                                           'lat',
                                           'long',
                                           'bed_bath_rooms',
                                           'bedrooms_squared',
                                           'log_sqft_living',
                                           'lat_plus_long']])
    print('mod1 test RSS: ' + str(np.mean((mod1_test_pred -
                                            df_test['price'])**2)))
    print('mod2 test RSS: ' + str(np.mean((mod2_test_pred -
                                            df_test['price'])**2)))
    print('mod3 test RSS: ' + str(np.mean((mod3_test_pred -
                                            df_test['price'])**2)))


def wrapper():
    os.chdir('..')
    train = pd.read_csv('kc_house_train_data.csv')
    test = pd.read_csv('kc_house_test_data.csv')
    add_variables(train)
    add_variables(test)
    get_new_means(test)
    get_regression_coefficients(train, test)


if __name__ == '__main__':
    wrapper()
