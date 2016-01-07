import numpy as np
import pandas as pd
from sklearn import linear_model
from math import sqrt


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


def normalize_features(features):
    """
    Normalizes columns of a features matrix
    :param features: numpy array
    :return features_normalized: numpy array
    """
    norms = np.linalg.norm(features, axis=0)
    features_normalized = features/norms
    return features_normalized, norms


def compute_distances(features_instances, features_query):
    """
    computes the distances from a query point to all training point.
    :param features_instances: numpy matrix
    :param features_query: numpy vector
    :return:
    """
    diff = features_instances - features_query
    distances = np.sqrt(np.sum(diff**2, axis=1))
    return distances


def k_nearest_neighbors(k, features_train, features_query):
    """
    Finds k nearest neighbors to a given query instance
    :param k: integer
    :param features_train: numpy array
    :param features_query: numpy array
    :return neighbors: indices of k nearest neighbors
    """
    distances = compute_distances(features_train, features_query)
    neighbors = np.argsort(distances)
    return neighbors[:k]


def predict_output_of_query(k, features_train, output_train, features_query):
    """
    Returns average y-value of k-nearest neighbors to features_query instance
    :param k: integer
    :param features_train: numpy array
    :param output_train: numpy array
    :param features_query: numpy array
    :return prediction: mean (unweighted) of y-values for k-nearest neighbors
    """
    neighbors = k_nearest_neighbors(k, features_train, features_query)
    prediction = np.mean(output_train[neighbors])
    return prediction


def predict_output(k, features_train, output_train, features_query):
    """
    Uses knn to predict average output value for each element in training set
    :param k: integer
    :param features_train: numpy array
    :param output_train: numpy array
    :param features_query: numpy array
    :return predictions: numpy array
    """
    predictions = []
    for i in range(features_query.shape[0]):
        predictions.append(predict_output_of_query(k, features_train,
                                                   output_train,
                                                   features_query[i]))
    return np.array(predictions)


def question1(test, train):
    # find Euclidian distance between 2 houses
    print('distance between test[0] and train [9]')
    print(np.sqrt(np.sum((train[9] - test[0])**2)))


def question2(test, train):
    # find closest of first 10 houses
    min_distance = float('inf')
    closest_house = 0
    print('\nDistances between test[0] and first 9 train houses')
    for i in range(10):
        distance = np.sqrt(np.sum((train[i] - test[0])**2))
        if distance < min_distance:
            min_distance = distance
            closest_house = i
        print(i, distance)
    print('\nClosest house is index #: ' + str(closest_house))


def test_vectorization(test, train):
    for i in range(3):
        print(train[i] - test[0])
    # should print 3 vectors of length 18

    print(train[0:3] - test[0])

    # verify that vectorization works
    results = train[0:3] - test[0]
    print(results[0] - (train[0] - test[0]))
    # should print all 0's if results[0] == (features_train[0]-features_test[0])
    print(results[1] - (train[1] - test[0]))
    # should print all 0's if results[1] == (features_train[1]-features_test[0])
    print(results[2] - (train[2]-test[0]))
    # should print all 0's if results[2] == (features_train[2]-features_test[0])


def question3(test, train, train_output):
    # Find nearest house and predicted value to a sample
    print('\nNearest house to test[2]')
    differences = compute_distances(train, test[2])
    min_index = np.argmin(differences)
    print('Index of closest house: ' + str(min_index))
    print('predicted house price: ' + str(train_output[min_index]))


def question4(test, train, train_output):
    # knn regression
    nearest_4 = k_nearest_neighbors(4, train, test[2])
    print('\nNearest 4 neighbors to test[2]')
    print(nearest_4)
    print('Predicted value for test[2]')
    print(predict_output_of_query(4, train, train_output, test[2]))


def question5(test, train, train_output):
    # find index and value for lowest predicted price from first 10
    # houses in train set
    print('\nLowest predicted price and index for train[:10]')
    test10 = predict_output(10, train, train_output, test[:10,])
    min_index = np.argmin(test10)
    print('Index number: ' + str(min_index))
    print('Predicted value: ' + str(test10[min_index]))


def question6(test, test_output, train, train_output, valid, valid_output):
    # find optimal k value from validation set
    print('\nThis part is slower...')
    best_rss = float('inf')
    best_k = 0
    for k in range(1, 16):
        k_preds = predict_output(k, train, train_output, valid)
        rss = np.sum((valid_output - k_preds)**2)
        if rss < best_rss:
            best_rss, best_k = rss, k
    print('\nBest k value: ' + str(best_k))

    # find rss for test set based on best_k
    test_preds = predict_output(best_k, train, train_output, test)
    rss_test = np.sum((test_output - test_preds)**2)
    print('RSS for test data: ' + str(rss_test))


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

    feature_list = ['bedrooms',
                    'bathrooms',
                    'sqft_living',
                    'sqft_lot',
                    'floors',
                    'waterfront',
                    'view',
                    'condition',
                    'grade',
                    'sqft_above',
                    'sqft_basement',
                    'yr_built',
                    'yr_renovated',
                    'lat',
                    'long',
                    'sqft_living15',
                    'sqft_lot15']

    train = pd.read_csv('kc_house_data_small_train.csv', dtype=dtype_dict)
    test = pd.read_csv('kc_house_data_small_test.csv', dtype=dtype_dict)
    valid = pd.read_csv('kc_house_data_validation.csv', dtype=dtype_dict)
    train_features, train_output = get_numpy_data(train, feature_list, 'price')
    test_features, test_output = get_numpy_data(test, feature_list, 'price')
    valid_features, valid_output = get_numpy_data(valid, feature_list, 'price')
    train_features, norms = normalize_features(train_features)
    test_features = test_features / norms
    valid_features = valid_features / norms

    question1(test_features, train_features)
    question2(test_features, train_features)
    # test_vectorization(test_features, train_features)
    question3(test_features, train_features, train_output)
    question4(test_features, train_features, train_output)
    question5(test_features, train_features, train_output)
    question6(test_features, test_output, train_features, train_output,
              valid_features, valid_output)

if __name__ == '__main__':
    wrapper()