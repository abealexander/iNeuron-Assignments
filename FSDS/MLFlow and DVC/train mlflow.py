# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV
from sklearn.metrics import r2_score
import pickle
import mlflow
import mlflow.sklearn
import warnings


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Loading Data
    df = pd.read_csv('train/ai4i2020.csv')

    # Dropping Columns
    df.drop(['UDI', 'Product ID'], axis=1, inplace=True)

    # Checking for NaN values
    print(df.isnull().values.any())

    # Label Encoding
    le = LabelEncoder()
    df['Type'] = le.fit_transform(df['Type'])

    # Normalize the data
    X = df.drop('Air temperature [K]', axis=1)
    y = df['Air temperature [K]']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=27)
    X_train_norm = MinMaxScaler().fit_transform(X_train)
    X_test_norm = MinMaxScaler().fit_transform(X_test)

    mlflow.sklearn.autolog()
    with mlflow.start_run():
        # Build a Model
        lr = LinearRegression()
        lr.fit(X_train_norm, y_train)

        # Model Accuracy
        y_pred = lr.predict(X_test_norm)
        print("R2 Score:", r2_score(y_test, y_pred))

        # Lasso Regularization
        # lassocv = LassoCV(alphas = None,cv =10, max_iter = 100000, normalize = True)
        # lassocv.fit(X_train_norm, y_train)
        # alpha = lassocv.alpha_
        # lasso_reg = Lasso(alpha)
        # lasso_reg.fit(X_train_norm, y_train)
        # print("R2 Score with Lasso Regularization", lasso_reg.score(X_test_norm, y_test))

        # Ridge Regularization
        # alphas = np.random.uniform(low=0, high=10, size=(50,))
        # ridgecv = RidgeCV(alphas = alphas,cv=10,normalize = True)
        # ridgecv.fit(X_train_norm, y_train)
        # ridge_model = Ridge(alpha=ridgecv.alpha_)
        # ridge_model.fit(X_train_norm, y_train)
        # print("R2 Score with Ridge Regularization", ridge_model.score(X_test_norm, y_test))

        # # Usigng Elastic Net
        # elasticCV = ElasticNetCV(alphas = None, cv =10)
        # elasticCV.fit(X_train_norm, y_train)
        # elasticnet_reg = ElasticNet(alpha = elasticCV.alpha_,l1_ratio=0.5)
        # elasticnet_reg.fit(X_train_norm, y_train)
        # print("R2 Score with Elastic Net", elasticnet_reg.score(X_test_norm, y_test))

        # Saving Model with best scores
        # filename = 'model.sav'
        # pickle.dump(elasticnet_reg, open(filename, 'wb'))

        # Test Case
        # Test_Case = [[0.5, 0.87654321, 0.37087912, 0.04743083, 0.02766798, 0, 0, 0, 0, 0, 0]]
        # print("Test Case Prediction:",elasticnet_reg.predict(Test_Case))
