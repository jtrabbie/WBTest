import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error


def main():
    df = pd.read_csv("VesselData.csv")
    len_full_data = len(df)
    # Remove all rows for which all important variables are 0
    df = df[(df.discharge1 != 0) | (df.discharge2 != 0) | (df.discharge3 != 0) | (df.discharge4 != 0) |
            (df.load1 != 0) | (df.load2 != 0) | (df.load3 != 0) | (df.load4 != 0)]
    print("Percentage missing data:", 1 - len(df) / len_full_data)
    # Convert dates to datetime object
    for date in ['ata', 'eta', 'atd', 'earliesteta', 'latesteta']:
        df[date] = pd.to_datetime(df[date])
    df = df.sort_values(by='eta')
    # Plot the data to get some initial feeling
    # plot_discharge_over_time(df)
    # pie_chart_discharges(df)
    # Add some extra columns to the dataframe using the eta since we only know the eta before arrival
    df['quarter'] = pd.DatetimeIndex(df['eta']).quarter
    df['month'] = pd.DatetimeIndex(df['eta']).month
    df['weekday'] = pd.DatetimeIndex(df['eta']).weekday
    # Convert these to dummy variables for the regression
    df = pd.get_dummies(df, columns=['month'], drop_first=True, prefix='month')
    df = pd.get_dummies(df, columns=['weekday'], drop_first=True, prefix='wday')
    df = pd.get_dummies(df, columns=['quarter'], drop_first=True, prefix='qrtr')
    # Fix the seed for reproducibility and randomly split into test and training data
    np.random.seed(1234)
    mask = np.random.rand(len(df)) < 0.7
    train_df = df[mask]
    test_df = df[~mask]
    predictors = list(set(list(train_df.columns)) - {'ata', 'eta', 'atd', 'earliesteta', 'latesteta', 'stevedorenames',
                                                     'traveltype', 'isremarkable', 'hasnohamis'})
    # Create a decision tree, fit and predict data using a type of one cargo type
    decision_tree_regressor(train_df, test_df, cargo_type='4', variable='load', predictors=predictors, max_tree_depth=5)
    # Note that cargo type 3 is very well predictable, while 2 and 4 are not
    # linear_regression(train_df, test_df, cargo_type='4', predictors=predictors)


def decision_tree_regressor(train_df, test_df, cargo_type=None, variable='load', predictors=None, max_tree_depth=5):
    if variable not in ['load', 'discharge']:
        raise ValueError("Regression variable must be either 'load' or 'discharge'")
    target_column = variable + cargo_type
    # Split the dataframes into numpy arrays of test and training data
    all_discharges = {'discharge1', 'discharge2', 'discharge3', 'discharge4'}
    all_loads = {'load1', 'load2', 'load3', 'load4'}
    predictors = list(set(predictors) - all_discharges - all_loads)
    X_train = train_df[predictors].values
    y_train = train_df[target_column].values
    X_test = test_df[predictors].values
    y_test = test_df[target_column].values
    print(f"Decision Tree for variable {target_column}")
    # Create decision tree
    dtree = DecisionTreeRegressor(max_depth=max_tree_depth, random_state=1234)
    # Set NaN's to 0 for now
    X_train[np.isnan(X_train)] = 0
    X_test[np.isnan(X_test)] = 0
    # First fit the tree with the training data
    dtree.fit(X_train, y_train)
    # Then check the output statistics for the fit itself and the prediction
    pred_train_tree = dtree.predict(X_train)
    print("MSE Train:", mean_squared_error(y_train, pred_train_tree))
    print("R squared Train:", r2_score(y_train, pred_train_tree))
    pred_test_tree = dtree.predict(X_test)
    print("MSE Test:", mean_squared_error(y_test, pred_test_tree))
    print("R square Test:", r2_score(y_test, pred_test_tree))


def linear_regression(train_df, test_df, cargo_type='1', predictors=None):
    # Split the dataframes into numpy arrays of test and training data
    all_discharges = ['discharge1', 'discharge2', 'discharge3', 'discharge4']
    target_column = f"discharge{cargo_type}"
    predictors = list(set(predictors) - set(all_discharges) - {target_column})
    X_train = train_df[predictors].values
    X_train[np.isnan(X_train)] = 0
    y_train = train_df[target_column].values
    X_test = test_df[predictors].values
    y_test = test_df[target_column].values
    print(f"LASSO variable selection for variable discharge{cargo_type}")
    reg = linear_model.Lasso(alpha=0.1, normalize=True)
    reg.fit(X_train, y_train)
    lasso_out = reg.predict(X_test)
    print("MSE:", mean_squared_error(y_test, lasso_out))
    print("R squared:", r2_score(y_test, lasso_out))


def plot_discharge_over_time(df):
    df.plot(x='eta', y=['discharge1', 'discharge2', 'discharge3', 'discharge4'], marker='o', ls='')
    plt.legend(loc='best')
    plt.show()


def pie_chart_discharges(df):
    dfs = df.sum(axis=0)
    plt.pie([dfs.discharge1, dfs.discharge2, dfs.discharge3, dfs.discharge4], labels=['1', '2', '3', '4'])
    plt.show()


if __name__ == "__main__":
    main()
