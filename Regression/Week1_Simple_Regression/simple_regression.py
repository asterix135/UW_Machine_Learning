import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt


def simple_linear_regression(input_feature, output):
    """
    Computes coefficients for simple regression
    :param input_feature: x-value for regression
    :param output: y-value for regression
    :return: intercept and slope of regression equation
    """
    y_sum = sum(output)
    x_sum = sum(input_feature)
    xy_sum = sum(input_feature * output)
    xsqr_sum = sum(input_feature ** 2)
    slope = (xy_sum - (y_sum * x_sum)/len(output)) / \
            (xsqr_sum - (x_sum**2 / len(output)))
    intercept = np.mean(output) - slope * np.mean(input_feature)
    return intercept, slope


def get_regression_predictions(input_feature, intercept, slope):
    return intercept + input_feature * slope


def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    preds = get_regression_predictions(input_feature, intercept, slope)
    rss = sum((preds-output) ** 2)
    return rss


def inverse_regression_predictions(output, intercept, slope):
    return (output - intercept) / slope


def wrapper():
    train = pd.read_csv('kc_house_train_data.csv')
    test = pd.read_csv('kc_house_test_data.csv')
    sqft_intercept, sqft_slope = simple_linear_regression(train['sqft_living'],
                                                          train['price'])
    # print(sqft_intercept, sqft_slope)
    print(get_regression_predictions(2650, sqft_intercept, sqft_slope))
    print(get_residual_sum_of_squares(train['sqft_living'], train['price'],
                                      sqft_intercept, sqft_slope))
    print(inverse_regression_predictions(800000, sqft_intercept, sqft_slope))
    bedroom_intercept, bedroom_slope = \
        simple_linear_regression(train['bedrooms'], train['price'])
    print('RSS test sqft')
    print(get_residual_sum_of_squares(test['sqft_living'], test['price'],
                                      sqft_intercept, sqft_slope))
    print('RSS test bedrrom')
    print(get_residual_sum_of_squares(test['sqft_living'], test['price'],
                                      bedroom_intercept, bedroom_slope))


if __name__ == '__main__':
    wrapper()
