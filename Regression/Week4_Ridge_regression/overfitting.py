import os
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


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


def print_coefficients(model):
    # Get the degree of the polynomial
    deg = len(model.coef_)
    # Get learned parameters as a list
    w = list(model.coef_)

    # Numpy has a nifty function to print out polynomials in a pretty way
    # (We'll use it, but it needs the parameters in the reverse order)
    print('Learned polynomial for degree ' + str(deg) + ':')
    w.reverse()
    w += [model.intercept_]
    print(np.poly1d(w))


def plot_regression(mod, df, y_vals, tag):
    # features = df.drop(['price'], axis=1)
    col_names = list(df.columns.values)
    plot_title = str(len(col_names)) + '-degree regression (' + tag + ')'
    file_name = tag + '_degree' + str(len(col_names)) + '_plot.png'
    plt.plot(df['power_1'], y_vals, '.',
             df['power_1'], mod.predict(df[col_names]), '-')
    plt.title(plot_title)
    plt.savefig(file_name)
    # plt.show()
    plt.close()



def run_ridge_regression(df, penalty, tag):
    poly_15_data = polynomial_dataframe(df['sqft_living'], 15)
    mod = linear_model.Ridge(alpha=penalty, normalize=True)
    mod.fit(poly_15_data, df['price'])
    plot_regression(mod, poly_15_data, df['price'], tag)
    return mod


def k_fold_cross_validation(k, l2_penalty, data, output):
    n = len(data)
    rss = []
    for i in range(k):
        # Compute starting and ending indices of segment i
        start = (n * i) // k
        end = (n * (i + 1)) // k - 1
        # Form validation set by taking a slice (start:end+1) from the data.
        valid_set = data[start:end+1]
        valid_output = output[start:end+1]
        # Form training set by appending slice (end+1:n) to the end of slice (0:start).
        train_set = data[0:start].append(data[end+1:n])
        train_output = output[0:start].append(output[end+1:n])
        # Train a linear model using training set just formed, with a given l2_penalty
        mod = linear_model.Ridge(alpha=l2_penalty, normalize=True)
        mod.fit(train_set, train_output)
        # Compute validation error (RSS) using validation set just formed
        rss += [sum((mod.predict(valid_set) - valid_output)**2)]
    return np.mean(rss)


def question1(df):
    l2_small_penalty = 1.5e-5
    poly15_data = polynomial_dataframe(df['sqft_living'], 15)
    model = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
    model.fit(poly15_data, df['price'])
    print_coefficients(model)
    print()
    return model


def question2(dtype_dict):
    l2_small_penalty = 1e-9

    # working directory already correct for data sets
    set_1 = pd.read_csv('wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
    os.chdir('Week4_Ridge_regression')
    mod = run_ridge_regression(set_1, l2_small_penalty, 'Set 1 small')
    print('set 1, power 1 coefficient: ' + str(mod.coef_[0]) + '\n')

    os.chdir('..')
    set_2 = pd.read_csv('wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
    os.chdir('Week4_Ridge_regression')
    mod = run_ridge_regression(set_2, l2_small_penalty, 'Set 2 small')
    print('set 2, power 1 coefficient: ' + str(mod.coef_[0]) + '\n')

    os.chdir('..')
    set_3 = pd.read_csv('wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
    os.chdir('Week4_Ridge_regression')
    mod = run_ridge_regression(set_3, l2_small_penalty, 'Set 3 small')
    print('set 3, power 1 coefficient: ' + str(mod.coef_[0]) + '\n')

    os.chdir('..')
    set_4 = pd.read_csv('wk3_kc_house_set_4_data.csv', dtype=dtype_dict)
    os.chdir('Week4_Ridge_regression')
    mod = run_ridge_regression(set_4, l2_small_penalty, 'Set 4 small')
    print('set 4, power 1 coefficient: ' + str(mod.coef_[0]) + '\n')

    l2_large_penalty=1.23e2

    os.chdir('..')
    set_1 = pd.read_csv('wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
    os.chdir('Week4_Ridge_regression')
    mod = run_ridge_regression(set_1, l2_large_penalty, 'Set 1 large')
    print('set 1, power 1 coefficient large penalty: ' + str(mod.coef_[0]) +
          '\n')

    os.chdir('..')
    set_2 = pd.read_csv('wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
    os.chdir('Week4_Ridge_regression')
    mod = run_ridge_regression(set_2, l2_large_penalty, 'Set 2 large')
    print('set 2, power 1 coefficient large penalty: ' + str(mod.coef_[0]) +
          '\n')

    os.chdir('..')
    set_3 = pd.read_csv('wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
    os.chdir('Week4_Ridge_regression')
    mod = run_ridge_regression(set_3, l2_large_penalty, 'Set 3 large')
    print('set 3, power 1 coefficient large penalty: ' + str(mod.coef_[0]) +
          '\n')

    os.chdir('..')
    set_4 = pd.read_csv('wk3_kc_house_set_4_data.csv', dtype=dtype_dict)
    os.chdir('Week4_Ridge_regression')
    mod = run_ridge_regression(set_4, l2_large_penalty, 'Set 4 large')
    print('set 4, power 1 coefficient large penalty: ' + str(mod.coef_[0]) +
          '\n')


def select_l2(dtype_dict):
    train_valid_shuffled = pd.read_csv('wk3_kc_house_train_valid_shuffled.csv',
                                       dtype=dtype_dict)
    poly15_train = polynomial_dataframe(train_valid_shuffled['sqft_living'], 15)
    test = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
    poly15_test = polynomial_dataframe(test['sqft_living'], 15)

    k=10
    best_rss = float('inf')
    best_l2 = 0
    for penalty in np.logspace(3, 9, num = 13):
        rss = k_fold_cross_validation(k, penalty, poly15_train,
                                      train_valid_shuffled['price'])
        if rss < best_rss:
            best_rss = rss
            best_l2 = penalty
    print(best_l2)
    mod = linear_model.Ridge(alpha=best_l2, normalize=True)
    mod.fit(poly15_train, train_valid_shuffled['price'])
    test_rss = sum((mod.predict(poly15_test) - test['price']) ** 2)
    print('optimized ls test rss: ' + str(test_rss))


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
    question1(sales)
    question2(dtype_dict)
    os.chdir('..')
    select_l2(dtype_dict)


if __name__ == '__main__':
    wrapper()
