import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import r2_score


def mse(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Intercepto:", model.intercept_)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Erro médio quadrático:", mse)
    

def test(X, y):
    model = LinearRegression()
    r2_scores = cross_val_score(model, X, y, cv=10, scoring='r2')
    print("Média de r2_scores:", r2_scores.mean())


def r2(X, y):
    X = df.drop(columns=['f1_score_mean']) 
    y = df['f1_score_mean']  

    model = LinearRegression()
    predicted = cross_val_predict(model, X, y, cv=10)
    r2_ = r2_score(y, predicted)
    print("Média de R^2:", r2_)


def remove_duplicates_df(df):
    def get_first_name(name):
        try:
            return name.split("_")[0]
        except:
            return name

    df['dataset_name'] = df['dataset_name'].apply(get_first_name)
    df = df.drop_duplicates(subset=['dataset_name'])
    
    df.to_csv('./datasets/metadatset_no_duplicates.csv', index=False)
    return df


if __name__ == '__main__':

    df = pd.read_csv('./datasets/metadaset_nonan.csv')

    df = remove_duplicates_df(df)
    
    df = df.drop(columns=['dataset_name']) 
    df = df.drop(columns=['did'])
    
    X = df.drop(columns=['f1_score_mean']) 
    y = df['f1_score_mean'] 
    

    r2(X, y)
    test(X, y)
    mse(X, y)