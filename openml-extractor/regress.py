import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict


def remove_invalid_y(dataframe: pd.DataFrame, y: np.ndarray, y_ped: np.ndarray) -> None:
    new_dataset: bool = False
    for val1, val2 in zip(y, y_ped):
        diff = abs(val1 - val2)
        if diff > 1:
            new_dataset = True
            dataframe = dataframe[dataframe['f1_score_mean'] != val1]
            print(f"y  = {val1} y_pred = {val2} diff =  {diff}")
    if new_dataset:
        dataframe.to_csv('datasets/new_dataset.csv', index=False)


def estimate(dataframe: pd.DataFrame, estimator, cv: int) -> None:
    X = df.drop(columns=['f1_score_mean']).values
    y = df['f1_score_mean']

    y_pred = cross_val_predict(estimator, X, y, cv=cv)
    remove_invalid_y(dataframe, y, y_pred)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    print(f"{estimator} - Mean Squared error (MSE): {mse} ({mse * 100:.2f}%)")
    print(f"{estimator} - Coefficient of Determination (R²): {r2}")

    plot(y, y_pred, estimator)


def get_model(dataframe: pd.DataFrame, estimator):
    X = dataframe.drop(columns=['f1_score_mean']).values
    y = np.array(dataframe['f1_score_mean'])
    return estimator.fit(X, y)


def plot(x: np.ndarray, y: np.ndarray, label: str) -> None:
    plt.scatter(x, y, color='blue')
    plt.xlabel('y')
    plt.ylabel('y_pred')
    plt.title(label)
    plt.grid(True)
    plt.show()


def remove_duplicates_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    def get_first_name(name):
        try:
            return name.split("_")[0]
        except:
            return name

    dataframe['dataset_name'] = dataframe['dataset_name'].apply(get_first_name)
    dataframe = dataframe.drop_duplicates(subset=['dataset_name'])
    
    dataframe.to_csv('./datasets/metadatset_no_duplicates.csv', index=False)
    return dataframe


def evaluate(dataframe: pd.DataFrame, model) -> None:
    targets = dataframe['f1_score_mean'].values
    dataframe = dataframe.drop(columns=['f1_score_mean'])

    for i, row in dataframe.iterrows():
        X_test = row.values.reshape(1, -1)
        result = model.predict(X_test)
        print(f"Target: {targets[i]} Predicted: {result} Diff: {result - targets[i]}")


if __name__ == '__main__':
    df = pd.read_csv('./datasets/metadataset_26_06_2024.csv')
    estimate(df, RandomForestRegressor(), 10)
    model = get_model(df, RandomForestRegressor())
    df_test = pd.read_csv('./datasets/test_metadaset.csv')
    evaluate(df_test, model)
