from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

def linear_regression(self):
        """
        Simple Linear Regression
        Inputs:
        - X_train, y_train: training data for fitting model 
        - X_test, y_test: test data for evaluating performance 
        Output:
        - E_test: Test error (MSE/L2 error) (float)
        - model: (sklearn pipeline) the fitted model 
        """

