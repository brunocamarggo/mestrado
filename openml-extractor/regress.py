import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsRegressor


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


def train(dataframe: pd.DataFrame, X: np.ndarray, y: np.ndarray, model, cv: int) -> None:
    model.fit(X, y)

    y_pred = cross_val_predict(model, X, y, cv=cv)
    remove_invalid_y(dataframe, y, y_pred)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    print(f"{model} - Mean Squared error (MSE): {mse} ({mse * 100:.2f}%)")
    print(f"{model} - Coefficient of Determination (R²): {r2}")

    plot(X, y, y_pred, model)


def plot(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, label: str):
    plt.scatter(X[:, 0], y, color='blue', label='Dados Reais')
    plt.plot(X[:, 0], y_pred, color='red', label=label)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(label)
    plt.grid(True)
    plt.show()


def remove_duplicates_df(dataframe: pd.DataFrame):
    def get_first_name(name):
        try:
            return name.split("_")[0]
        except:
            return name

    dataframe['dataset_name'] = dataframe['dataset_name'].apply(get_first_name)
    dataframe = dataframe.drop_duplicates(subset=['dataset_name'])
    
    dataframe.to_csv('./datasets/metadatset_no_duplicates.csv', index=False)
    return dataframe


if __name__ == '__main__':

    df = pd.read_csv('./datasets/metadataset_26_06_2024.csv')

    # df = remove_duplicates_df(df)
    # df = df.drop(columns=['dataset_name'])
    # df = df.drop(columns=['did'])
    
    X = df.drop(columns=['f1_score_mean']) .values.tolist()
    y = df['f1_score_mean'] 

    X = np.array(X)
    y = np.array(y)

    train(df, X, y, KNeighborsRegressor(n_neighbors=5), 10)
    train(df, X, y, LinearRegression(), 10)
