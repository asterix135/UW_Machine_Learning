# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import os


def polynomial_dataframe(feature, degree):  # feature is pandas.Series type
    # assume that degree >= 1
    # initialize the dataframe:
    poly_dataframe = pd.DataFrame()
    # and set poly_dataframe['power_1'] equal to the passed feature
    poly_dataframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # assign poly_dataframe[name] to be feature^power; use apply(*)
            poly_dataframe[name] = feature.apply(lambda x: x ** power)
    return poly_dataframe


def plot_regression(df, tag):
    mod = linear_model.LinearRegression()
    features = df.drop(['price'], axis=1)
    col_names = list(features.columns.values)
    mod.fit(df[col_names], df['price'])
    plot_title = str(len(features.columns)) + '-degree regression (' + tag + ')'
    file_name = tag + '_degree' + str(len(features.columns)) + '_plot.png'
    plt.plot(df['power_1'], df['price'], '.',
             df['power_1'], mod.predict(df[col_names]), '-')
    plt.title(plot_title)
    plt.savefig(file_name)
    # plt.show()
    plt.close()
    return mod.coef_


def check_15_degrees(test, train, valid):
    best_degree = 0
    best_rss = float('inf')
    for i in range(15):
        curr_train = polynomial_dataframe(train['sqft_living'], i+1)
        mod = linear_model.LinearRegression()
        mod.fit(curr_train, train['price'])
        curr_valid = polynomial_dataframe(valid['sqft_living'], i+1)
        curr_rss = sum((mod.predict(curr_valid) - valid['price'])**2)
        if curr_rss < best_rss:
            best_rss, best_degree = curr_rss, i + 1
            best_mod = mod
    print('best degree = ' + str(best_degree))
    test_poly = polynomial_dataframe(test['sqft_living'], best_degree)
    test_rss = sum((best_mod.predict(test_poly) - test['price'])**2)
    print('test set rss: ' + str(test_rss))


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
                  'floors': str,
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
    os.chdir('Week3_Assessing_performance')

    # EDA
    sales = sales.sort_values(by=['sqft_living', 'price'])
    poly1_data = polynomial_dataframe(sales['sqft_living'], 1)
    poly1_data['price'] = sales['price']
    print('1st degree - full data set')
    print(plot_regression(poly1_data, 'full'))

    poly2_data = polynomial_dataframe(sales['sqft_living'], 2)
    poly2_data['price'] = sales['price']
    print('2nd degree - full data set')
    print(plot_regression(poly2_data, 'full'))

    poly3_data = polynomial_dataframe(sales['sqft_living'], 3)
    poly3_data['price'] = sales['price']
    print('3rd degree - full data set')
    print(plot_regression(poly3_data, 'full'))

    poly15_data = polynomial_dataframe(sales['sqft_living'], 15)
    poly15_data['price'] = sales['price']
    print('15th degree - full data set')
    print(plot_regression(poly15_data, 'full'))

    # Compare Splits
    set1 = pd.read_csv('wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
    set1 = set1.sort_values(by=['sqft_living', 'price'])
    set1_15 = polynomial_dataframe(set1['sqft_living'], 15)
    set1_15['price'] = set1['price']
    print('set1 15th degree')
    print(plot_regression(set1_15, 'set1')[14])

    set2 = pd.read_csv('wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
    set2 = set2.sort_values(by=['sqft_living', 'price'])
    set2_15 = polynomial_dataframe(set2['sqft_living'], 15)
    set2_15['price'] = set2['price']
    print('set2 15th degree')
    print(plot_regression(set2_15, 'set2')[14])

    set3 = pd.read_csv('wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
    set3 = set3.sort_values(by=['sqft_living', 'price'])
    set3_15 = polynomial_dataframe(set3['sqft_living'], 15)
    set3_15['price'] = set3['price']
    print('set3 15th degree')
    print(plot_regression(set3_15, 'set3')[14])

    set4 = pd.read_csv('wk3_kc_house_set_4_data.csv', dtype=dtype_dict)
    set4 = set4.sort_values(by=['sqft_living', 'price'])
    set4_15 = polynomial_dataframe(set4['sqft_living'], 15)
    set4_15['price'] = set4['price']
    print('set4 15th degree')
    print(plot_regression(set4_15, 'set4')[14])

    test = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
    train = pd.read_csv('wk3_kc_house_train_data.csv', dtype=dtype_dict)
    valid = pd.read_csv('wk3_kc_house_valid_data.csv', dtype=dtype_dict)
    check_15_degrees(test, train, valid)


if __name__ == '__main__':
    wrapper()
