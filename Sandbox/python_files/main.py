# Import necessary libraries.
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

import pdb


def data_inverse_transformer(transform_dict):
    inverse_transform_dict = {}
    for col, d in transform_dict.items():
        inverse_transform_dict[col] = {v: k for k, v in d.items()}


class MultiColumnLabelEncoder:
    def __init__(self, data):
        self.data = data

    def data_transformer(self, cat_columns):
        transform_dict = {}
        for col in cat_columns:
            cats = pd.Categorical(self[col]).categories
            d = {}
            for i, cat in enumerate(cats):
                d[cat] = i
            transform_dict[col] = d

        return self.replace(transform_dict), transform_dict

    def replace(self, transform_dict):
        pass


def data_preprocessor_lin_regression(raw_dataset, cat_columns):
    """
    Function that preprocesses the raw data to conform to the format required of the linear regression ML algorithm.
    :param raw_dataset: pandas dataframe object.
    :return: pandas dataframe
    """
    # Drop NA values in dataset.
    preprocessed_data_lin_reg_drop_na = raw_dataset.replace(r'^\s*$', np.nan, regex=True)

    preprocessed_data_lin_reg, transform_dict = MultiColumnLabelEncoder.data_transformer(
        preprocessed_data_lin_reg_drop_na,
        cat_columns=cat_columns)
    preprocessed_data_lin_reg['MonthlyCharges'] = preprocessed_data_lin_reg['MonthlyCharges'].fillna(0.0)
    preprocessed_data_lin_reg['TotalCharges'] = preprocessed_data_lin_reg['TotalCharges'].fillna(0.0)

    return preprocessed_data_lin_reg


def data_splitter(data):
    """
    Function that splits data into 80% training and 20% testing datasets.
    :param data: pandas dataframe
    :return: series, series, series, series
    """
    from sklearn.model_selection import train_test_split

    # Split data into train/test sets (80:20 split).
    telco_x_train, telco_x_test, telco_y_train, telco_y_test = train_test_split(data.iloc[:, 1:-1], data.iloc[:, -1],
                                                                                test_size=.2, random_state=42)

    return telco_x_train, telco_y_train, telco_x_test, telco_y_test


def model_predictor(model, data_to_predict):
    return model.predict(data_to_predict)


def main():
    # Import in raw telco dataset.
    raw_dataset_telco = pd.read_csv(
        r"C:\Users\Ujjwal-Work\Desktop\Rivery\python_kits\Sandbox\WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Preprocess raw data for Regressions.
    preprocessed_data_lin_reg_telco = data_preprocessor_lin_regression(raw_dataset_telco, cat_columns=
    ['gender', 'Partner', 'Dependents',
     'PhoneService',
     'MultipleLines',
     'InternetService',
     'OnlineSecurity', 'OnlineBackup',
     'DeviceProtection', 'TechSupport',
     'StreamingTV',
     'StreamingMovies', 'Contract',
     'PaperlessBilling',
     'PaymentMethod', 'Churn'])

    # Split data into train/test sets.
    telco_x_train, telco_y_train, telco_x_test, telco_y_test = data_splitter(preprocessed_data_lin_reg_telco)

    # Build Linear Regression model.
    lin_regression_model = linear_model.LinearRegression()

    # Train Linear Regression model.
    lin_regression_model.fit(telco_x_train, telco_y_train)

    # Use model to predict testing data churn outcome.
    lin_reg_y_pred = lin_regression_model.predict(telco_x_test)

    # Determine Linear Regression Model score.
    lin_reg_score = r2_score(telco_y_test, lin_reg_y_pred)
    print("lin_reg_score: %.2f" % lin_reg_score)

    # Build and Train Logistic Regression model.
    log_reg = LogisticRegression(solver='liblinear', random_state=0, max_iter=1000).fit(telco_x_train, telco_y_train)

    # Use Logistic Regression model to predict.
    log_reg.predict(telco_x_test)

    # Evaluate Logistic Regression model.
    log_reg_score = log_reg.score(telco_x_test, telco_y_test)

    print("log_reg_score: %.2f" % log_reg_score)

    # Generate decision tree classifier.
    decision_tree = DecisionTreeClassifier()

    # Train Decision Tree classifier.
    clf = decision_tree.fit(telco_x_train, telco_y_train)

    # Predict the outcome on the test data.
    decision_tree_y_pred = clf.predict(telco_x_test)

    # Decision Tree Model Score
    desc_tree_score = metrics.accuracy_score(telco_y_test, decision_tree_y_pred)
    print("decs_tree_score: %.2f" % desc_tree_score)

    # Create scores dict from all three models.
    scores_dict = {'Linear Regression': lin_reg_score, 'Logistic Regression': log_reg_score,
                   'Decision Tree': desc_tree_score}

    best_model = max(scores_dict)

    print(f"The most accurate prediction algorithm is {best_model} with R squared value of "
          f"{scores_dict[best_model]}!")

    data_to_predict_ = data_preprocessor_lin_regression(raw_dataset_telco, cat_columns=
                                                                        ['gender', 'Partner', 'Dependents',
                                                                         'PhoneService',
                                                                         'MultipleLines',
                                                                         'InternetService',
                                                                         'OnlineSecurity', 'OnlineBackup',
                                                                         'DeviceProtection', 'TechSupport',
                                                                         'StreamingTV',
                                                                         'StreamingMovies', 'Contract',
                                                                         'PaperlessBilling',
                                                                         'PaymentMethod', 'Churn'])

    data_to_predict = data_to_predict_.iloc[0:10]
    print(f"data_to_predict = {data_to_predict.iloc[:,0]}")
    print(f"type(data_to_predict) = {type(data_to_predict)}")
    OutputDF = pd.DataFrame()
    OutputDF[str(data_to_predict.columns[0])] = data_to_predict[str(data_to_predict.columns[0])]
    if best_model == 'Linear Regression':
        prediction = model_predictor(lin_regression_model, data_to_predict[1:-1])
        print([data_to_predict.iloc[:, 0], prediction])
    elif best_model == 'Logistic Regression':
        print(f"data_to_predict.iloc[:, 1:-1] = {data_to_predict.iloc[:, 1:-1]}")
        prediction = model_predictor(log_reg, data_to_predict.iloc[:, 1:-1])
        prediction = np.array(prediction)
        OutputDF['prediction'] = prediction.tolist()
        print(f"OutputDF = {OutputDF}")
    elif best_model == 'Decision Tree':
        prediction = model_predictor(clf, data_to_predict[1:-1])
        print([data_to_predict.iloc[:, 0], prediction])


if __name__ == "__main__":
    main()
