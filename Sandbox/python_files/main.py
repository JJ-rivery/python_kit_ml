# Import necessary libraries.
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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
    preprocessed_data_lin_reg_drop_na = raw_dataset.dropna()

    preprocessed_data_lin_reg = MultiColumnLabelEncoder \
        (columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                  'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                  'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                  'PaperlessBilling', 'PaymentMethod', 'Churn']).fit_transform(preprocessed_data_lin_reg_drop_na)

    return preprocessed_data_lin_reg


def main():
    # Import in raw telco dataset.
    raw_dataset_telco = pd.read_csv(
        r"C:\Users\Ujjwal-Work\Desktop\Rivery\python_kits\Sandbox\WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Preprocess raw data for Linear Regression.
    preprocessed_data_lin_reg_telco = data_preprocessor_lin_regression(raw_dataset_telco)
    print(preprocessed_data_lin_reg_telco)


if __name__ == "__main__":
    main()
