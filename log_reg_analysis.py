import os
import pandas as pd
import numpy as np
import Graph

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


def create_training_set(folder_of_data, t_list):
    # collect training dataset from the simulated data
    # bring it into the right format to be used for classification tasks
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

    nr_samples = len(X_list[0])
    for idx, t in enumerate(t_list):
        X_np = np.array(X_list[idx])
        X_list[idx] = X_np.reshape(nr_samples, -1)
    return X_list, y_list


def fit_clf(X, y, model="log"):
    scaler = StandardScaler()
    X_data_standardized = scaler.fit_transform(X)   # forward looking bias (fitting standardization also on test data)
    X_train, X_test, y_train, y_test = train_test_split(X_data_standardized, y, test_size=0.2, random_state=42)

    y_pred = 0
    if model=="svm":
        svm_classifier = SVC(kernel='rbf') #'linear', 'poly', 'rbf'
        svm_classifier.fit(X_train, y_train)
        y_pred = svm_classifier.predict(X_test)
        #print(model.coef_.shape[1])
    elif model=="nn":
        model = MLPClassifier(hidden_layer_sizes=(20, 5), activation='relu', solver='adam', max_iter=100)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# might want to generate simulated training datasets, if not happened yet
if False:
    Graph.generate_training_datasets(T=100, n=10, sim_para_dict=sim_para_dict, nr_datasets=1000)

sim_para_dict = {
    "gamma": 0.1,
    "beta": 0.2,
    "alpha": 0.9,
    "lambda_Z0": 1., 
    "lambda_Z1": 5.
}


# fit a classification model on each "datapoint"
t_list = list(range(2,100))#[7, 20, 77]    # t values for which to train a logistic regression
X_list, y_list = create_training_set(folder_of_data="simulated_datasets", t_list=t_list)
accuracy_list = []
for idx, t in enumerate(t_list):
    this_acc = fit_clf(X_list[idx], y_list[idx], model="log")
    accuracy_list.append(this_acc)

# plot the resulting forecasting accuracy of the chosen model across all choices of t
plt.figure(figsize=(10, 4))
plt.plot(t_list, accuracy_list)
plt.xlabel("t")
plt.ylabel("accuracy of SVM with a linear kernel")
plt.show()
