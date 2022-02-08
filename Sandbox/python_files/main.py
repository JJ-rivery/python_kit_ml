# Import necessary libraries.
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, x, y=None):
        return self  # not relevant here

    def transform(self, x):
        """
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        """
        output = x.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, x, y=None):
        return self.fit(x, y).transform(x)


def data_preprocessor_lin_regression(raw_dataset):
    """
    Function that preprocesses the raw data to conform to the format required of the linear regression ML algorithm.
    :param raw_dataset: pandas dataframe object.
    :return: pandas dataframe
    """
    # Drop NA values in dataset.
    preprocessed_data_lin_reg_drop_na = raw_dataset.replace(r'^\s*$', np.nan, regex=True)

    preprocessed_data_lin_reg = MultiColumnLabelEncoder \
        (columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                  'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                  'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                  'PaperlessBilling', 'PaymentMethod', 'Churn']).fit_transform(preprocessed_data_lin_reg_drop_na)

    preprocessed_data_lin_reg['MonthlyCharges'] = preprocessed_data_lin_reg['MonthlyCharges'].fillna(0.0)
    preprocessed_data_lin_reg['TotalCharges'] = preprocessed_data_lin_reg['TotalCharges'].fillna(0.0)

    return preprocessed_data_lin_reg


def data_splitter(data):
    """
    Function that splits data into 80% training and 20% testing datasets.
    :param data: pandas dataframe
    :return: series, series, series, series
    """
    data_length = len(data)

    # Split data into train/test sets (80:20 split).
    telco_train = data[0:int(data_length * .8)]
    telco_test = data[int(data_length * .8):data_length]

    # Split into X and Y train/test sets.
    telco_x_train = telco_train.iloc[:, 1:-1]
    telco_y_train = telco_train.iloc[:, -1]
    telco_x_test = telco_test.iloc[:, 1:-1]
    telco_y_test = telco_test.iloc[:, -1]
    print(telco_x_test)
    return telco_x_train, telco_y_train, telco_x_test, telco_y_test


def main():
    # Import in raw telco dataset.
    raw_dataset_telco = pd.read_csv(
        r"C:\Users\Ujjwal-Work\Desktop\Rivery\python_kits\Sandbox\WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Preprocess raw data for Linear Regression.
    preprocessed_data_lin_reg_telco = data_preprocessor_lin_regression(raw_dataset_telco)
    print(preprocessed_data_lin_reg_telco)

    # Split data into train/test sets.
    telco_x_train, telco_y_train, telco_x_test, telco_y_test = data_splitter(preprocessed_data_lin_reg_telco)

    # Build Linear Regression model.
    lin_regression_model = linear_model.LinearRegression()

    # Train Linear Regression model.
    lin_regression_model.fit(telco_x_train, telco_y_train)

    # Use model to predict testing data churn outcome.
    lin_reg_y_pred = lin_regression_model.predict(telco_x_test)

    # The coefficients
    print("Coefficients: \n", lin_regression_model.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(telco_y_test, lin_reg_y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(telco_y_test, lin_reg_y_pred))


if __name__ == "__main__":
    main()
