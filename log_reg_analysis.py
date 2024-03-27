import os
import pandas as pd
import Graph

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def create_training_set(folder_of_data, t_list = [0, 1, 2, 33, 99]):
    # collect it from the simulated data
    X_list = [[] for _ in range(len(t_list))]
    y_list = [[] for _ in range(len(t_list))]
    for filename in os.listdir(folder_of_data):
        if os.path.isfile(os.path.join(folder_of_data, filename)):
            this_df = pd.read_csv(f"{folder_of_data}/{filename}")
            this_X = Graph.construct_X(this_df)
            list_of_t_indexed = [t-1 for t in t_list]
            this_C_for_all_t = this_df["C"].values[list_of_t_indexed]
            for idx, t in enumerate(t_list):
                X_list[idx].append(this_X)
                y_list[idx].append(this_C_for_all_t[idx])

    # nr samples
    nr_samples = len(X_list[0])
    for idx, t in enumerate(t_list):
        X_np = np.array(X_list[idx])
        print(X_np.shape)
        X_list[idx] = X_np.reshape(nr_samples, -1)
        print(X_list[idx].shape)

    #exit()
    return X_list, y_list

from sklearn.preprocessing import StandardScaler

def fit_log_reg(X, y):
    scaler = StandardScaler()
    X_data_standardized = scaler.fit_transform(X)   # forward looking bias (fitting standardization also on test data)
    X_train, X_test, y_train, y_test = train_test_split(X_data_standardized, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    return accuracy


sim_para_dict = {
    "gamma": 0.1,
    "beta": 0.2,
    "alpha": 0.9,
    "lambda_Z0": 1., 
    "lambda_Z1": 5.
}
if False:
    Graph.generate_training_datasets(T=100, n=10, sim_para_dict=sim_para_dict, nr_datasets=1000)

t_list = [7, 20, 77]    # t values for which to train a logistic regression
X_list, y_list = create_training_set(folder_of_data="simulated_datasets", t_list=t_list)
for idx, t in enumerate(t_list):
    print(f"For t={t} we fit logistic regression...")
    fit_log_reg(X_list[idx], y_list[idx])
    print("--------------------------")