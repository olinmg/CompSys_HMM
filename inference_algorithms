import os
import pandas as pd
import Graph
from Graph import HMM
from Graph import construct_X
import numpy as np

#-------------------------------------------------------------------------
#Insert path to csv files. 

folder_path = r"C:\Users\nicol\OneDrive\Skrivebord\ComplexSystems\proj_HMM"

#------------------------------------------------------------------------

def calculate_conditional_prob_C(hmm_model, forward_probs, backward_probs):
    T = hmm_model.T  
    num_states = 3  
    conditional_prob_C = np.zeros((T, num_states))

    for t in range(T):
        for state in range(num_states):
            conditional_prob_C[t, state] = forward_probs[t, state] * backward_probs[t, state]
        
        normalization_constant = np.sum(conditional_prob_C[t, :])
        if normalization_constant != 0:  
            conditional_prob_C[t, :] /= normalization_constant

    return conditional_prob_C

def calculate_conditional_prob_Z(hmm_model, forward_probs, backward_probs, X):
    T, n = hmm_model.T, hmm_model.n
    num_C_states = 3  
    num_Z_states = 2  
    
    conditional_prob_Z = np.zeros((T, n, num_Z_states))

    for t in range(T):
        for i in range(n):
            prob_sum_over_Z = np.zeros(num_Z_states)  

            for z_state in range(num_Z_states):
                prob_sum_over_C = np.zeros(num_C_states)

                for c_state in range(num_C_states):
                    emission_prob = hmm_model.C_node_list[t].Z_node_list[i].X_node.observation_probability(X[t])[z_state]
                    emission_prob_scalar = emission_prob if np.isscalar(emission_prob) else emission_prob[0]

                    joint_prob = forward_probs[t, c_state] * emission_prob_scalar * backward_probs[t, c_state]
                    prob_sum_over_C[c_state] = joint_prob

                prob_sum_over_Z[z_state] = np.sum(prob_sum_over_C)  

            normalization_factor = np.sum(prob_sum_over_Z)  
            if normalization_factor > 0:
                for z_state in range(num_Z_states):
                    conditional_prob_Z[t, i, z_state] = prob_sum_over_Z[z_state] / normalization_factor

    return conditional_prob_Z




def forward_pass(hmm_model, X):
    T = hmm_model.T
    num_states = 3  
    forward_probs = np.zeros((T, num_states))
    forward_probs[0] = hmm_model.initial_state_distribution

    for t in range(1, T):
        for s in range(num_states):
            sum_prob = 0
            for prev_s in range(num_states):
                transition_prob = hmm_model.C_node_list[t - 1].tranisition_matrix[prev_s, s]
                emission_probs = hmm_model.C_node_list[t].Z_node_list[s].X_node.observation_probability(X[t])
                emission_prob = np.sum(emission_probs, axis=1)  
                sum_prob += forward_probs[t - 1, prev_s] * transition_prob * emission_prob[s]
            forward_probs[t, s] = sum_prob
            if sum_prob == 0:
                print(f"Warning: Zero forward probability at time {t} and state {s}")
    return forward_probs


def backward_pass(hmm_model, X):
    T = hmm_model.T
    num_states = 3  
    backward_probs = np.zeros((T, num_states))
    backward_probs[T-1, :] = 1  

    for t in range(T - 2, -1, -1):
        for s in range(num_states):
            sum_prob = 0
            for next_s in range(num_states):
                emission_probs = hmm_model.C_node_list[t + 1].Z_node_list[next_s].X_node.observation_probability(X[t + 1])
                emission_prob = np.sum(emission_probs, axis=1)  
                transition_prob = hmm_model.C_node_list[t].tranisition_matrix[s, next_s]
                sum_prob += backward_probs[t + 1, next_s] * transition_prob * emission_prob[next_s]
            backward_probs[t, s] = sum_prob

    return backward_probs




def forward_backward_algorithm(hmm_model, X):
    forward_probs = forward_pass(hmm_model, X)
    backward_probs = backward_pass(hmm_model, X)
    return forward_probs, backward_probs





# Define HMM parameters
T = 100  
n = 5  


#---------------------------------------------------------------Simulation test-----------------------------------------
sim_para_dict = {
    "gamma": 0.1,
    "beta": 0.2,
    "alpha": 0.9,
    "lambda_Z0": 1.0, 
    "lambda_Z1": 5.0
}

my_hmm = HMM(T, n)

my_hmm.set_proba_paras(sim_para_dict)

simulation_results = my_hmm.start_simulation()
state_df = my_hmm.state_as_df()  

X = construct_X(state_df)

# Run forward-backward algorithm
forward_probs, backward_probs = forward_backward_algorithm(my_hmm, X)

# Calculate conditional probabilities
conditional_prob_C = calculate_conditional_prob_C(my_hmm, forward_probs, backward_probs)
# Make sure to pass 'X' here
conditional_prob_Z = calculate_conditional_prob_Z(my_hmm, forward_probs, backward_probs, X)

print("Conditional probabilities for variable C:")  
print(conditional_prob_C)

print("Conditional probabilities for variable Z:")
print(conditional_prob_Z)


#-----------------------------calculate the average for multiple runs to test for average = ca zero-----------------------------------------

def test_hmm_implementation(hmm_model, num_replications):
    differences_C = []
    differences_Z = []

    for _ in range(num_replications):
        hmm_model.reset_HMM_values()

        hmm_model.start_simulation()

        state_df = hmm_model.state_as_df()
        X = construct_X(state_df)

        # Runs the forward-backward algorithm on the observed data
        forward_probs, backward_probs = forward_backward_algorithm(hmm_model, X)

        # Calculate conditional probabilities for C and Z 
        conditional_prob_C = calculate_conditional_prob_C(hmm_model, forward_probs, backward_probs)
        conditional_prob_Z = calculate_conditional_prob_Z(hmm_model, forward_probs, backward_probs, X)

        C_simulated = [c_node.C_value for c_node in hmm_model.C_node_list]
        Z_simulated = [[z_node.Z_value for z_node in c_node.Z_node_list] for c_node in hmm_model.C_node_list]

        for t, c_value in enumerate(C_simulated):
            for state in range(3):
                indicator_C = int(c_value == state)
                difference_C = indicator_C - conditional_prob_C[t, state]
                differences_C.append(difference_C)

        for t, z_values in enumerate(Z_simulated):
            for i, z_value in enumerate(z_values):
                for z_state in [0, 1]:
                    indicator_Z = int(z_value == z_state)
                    difference_Z = indicator_Z - conditional_prob_Z[t, i, z_state]
                    differences_Z.append(difference_Z)

    average_difference_C = np.mean(differences_C)
    average_difference_Z = np.mean(differences_Z)

    print("Average difference for C across replications:", average_difference_C)
    print("Average difference for Z across replications:", average_difference_Z)



    return average_difference_C, average_difference_Z




number_of_replications = 10  

test_hmm_implementation(my_hmm, number_of_replications)

#---------------------------------------------------------------------------inference på data file-----------------------------


def process_data_in_folder(folder_of_data):
    X_list = []
    for filename in os.listdir(folder_of_data):
        if os.path.isfile(os.path.join(folder_of_data, filename)):
            this_df = pd.read_csv(os.path.join(folder_of_data, filename))
            print(f"Processing file: {filename}")
            print("DataFrame from CSV file:")
            X_list.append(Graph.construct_X(this_df))
    return X_list


def apply_inference_algorithm(data):
    results = []
    for dataset in data:
        T, n = dataset.shape[0], dataset.shape[1]
        
        sim_para_dict = {
            "gamma": 0.1,
            "beta": 0.2,
            "alpha": 0.9,
            "lambda_Z0": 1.0, 
            "lambda_Z1": 2.0
        }
        my_hmm = HMM(T, n)
        my_hmm.set_proba_paras(sim_para_dict)
        
        forward_probs, backward_probs = forward_backward_algorithm(my_hmm, dataset)  
        
        # Calculate conditional probabilities
        conditional_prob_C = calculate_conditional_prob_C(my_hmm, forward_probs, backward_probs)
        conditional_prob_Z = calculate_conditional_prob_Z(my_hmm, forward_probs, backward_probs, dataset)  
        
        results.append((conditional_prob_C, conditional_prob_Z))
    
    return results



data = process_data_in_folder(folder_path)

results = apply_inference_algorithm(data)

for i, result in enumerate(results):
    print(f"Results for Dataset {i+1}:")
    conditional_prob_C, conditional_prob_Z = result
    print("Conditional probabilities for variable C:")
    print(conditional_prob_C)
    print("Conditional probabilities for variable Z:")
    print(conditional_prob_Z)
    print("\n") 





