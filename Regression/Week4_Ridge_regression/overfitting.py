import os
import numpy as np
import pandas as pd
from sklearn import linear_model


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
    print(model.intercept_)
    print(type(model.intercept_))
    # Get learned parameters as a list
    w = list(model.coef_) + [model.intercept_]

    # Numpy has a nifty function to print out polynomials in a pretty way
    # (We'll use it, but it needs the parameters in the reverse order)
    print('Learned polynomial for degree ' + str(deg) + ':')
    w.reverse()
    print(np.poly1d(w))


# TODO: This needs to be fixed to take a model as input
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


def question1(df):
    l2_small_penalty = 1.5e-5
    poly15_data = polynomial_dataframe(df['sqft_living'], 15)
    model = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
    model.fit(poly15_data, df['price'])
    print_coefficients(model)
    return model


def question2(dtype_dict):
    # working directory already correct for data sets
    set_1 = pd.read_csv('wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
    set_2 = pd.read_csv('wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
    set_3 = pd.read_csv('wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
    set_4 = pd.read_csv('wk3_kc_house_set_4_data.csv', dtype=dtype_dict)
    l2_small_penalty = 1e-9
    os.chdir('Week4_Ridge_regression')



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


if __name__ == '__main__':
    wrapper()
