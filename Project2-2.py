import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# Sort the entire data by year column in ascending order
def sort_dataset(dataset_df):
    return dataset_df.sort_values(by='year')


# Split the entire data as train/test datasets
def split_dataset(dataset_df):
    dataset_df['salary'] *= 0.001
    xTrain = dataset_df.iloc[:1718].drop('salary', axis=1)
    yTrain = dataset_df.iloc[:1718]['salary']
    xTest = dataset_df.iloc[1718:].drop('salary', axis=1)
    yTest = dataset_df.iloc[1718:]['salary']
    return xTrain, xTest, yTrain, yTest


# Extract only numerical columns
def extract_numerical_cols(dataset_df):
    numericalCol = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP',
                    'fly', 'war']
    return dataset_df[numericalCol]


# Complete the train and predict functions for decision tree
def train_predict_decision_tree(X_train, Y_train, X_test):
    model = DecisionTreeRegressor()
    model.fit(X_train, Y_train)
    return model.predict(X_test)


# Complete the train and predict functions for random forest
def train_predict_random_forest(X_train, Y_train, X_test):
    model = RandomForestRegressor()
    model.fit(X_train, Y_train)
    return model.predict(X_test)


# Complete the train and predict functions for svm
def train_predict_svm(X_train, Y_train, X_test):
    model = make_pipeline(StandardScaler(), SVR())
    model.fit(X_train, Y_train)
    return model.predict(X_test)


# Calculate RMSE for given labels and predictions
def calculate_RMSE(Labels, predictions):
    return np.sqrt(mean_squared_error(Labels, predictions))


if __name__ == '__main__':
    # DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

    sorted_df = sort_dataset(data_df)
    X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)

    X_train = extract_numerical_cols(X_train)
    X_test = extract_numerical_cols(X_test)

    dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
    rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
    svm_predictions = train_predict_svm(X_train, Y_train, X_test)

    print("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))
    print("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))
    print("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))
