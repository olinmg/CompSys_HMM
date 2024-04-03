from Graph import experiment_result
from Graph import state_df
import numpy as np

##----------LEARNING OF THE PARAMETERS---------------------------------------##
##----------SIMULATION TEST--------------------------------------------------##
def return_lambda(T,n):
    placeholder_lambda_0 = 0
    placeholder_lambda_1 = 0 
    counter_0 = 0 
    counter_1 = 0
    
    for i in range(n):
        for j in range(2,n+2):
            if experiment_result.iloc[i,j] == 0:
                placeholder_lambda_0 += experiment_result.iloc[i,j+8]
                counter_0 += 1
            else:
                placeholder_lambda_1 += experiment_result.iloc[i,j+8]
                counter_1 += 1
                
    lambda_0 = placeholder_lambda_0 / counter_0
    lambda_1 = placeholder_lambda_1 / counter_1
    return lambda_0, lambda_1



def relative_frequency(): 
    par = 0
    for i in range(100):
        if experiment_result['C'][i] == experiment_result['Z1'][i]:
            par += 1
        if experiment_result['C'][i] == experiment_result['Z2'][i]:
            par += 1
        if experiment_result['C'][i] == experiment_result['Z3'][i]:
            par += 1
        if experiment_result['C'][i] == experiment_result['Z4'][i]:
            par += 1
        if experiment_result['C'][i] == experiment_result['Z5'][i]:
            par += 1
        if experiment_result['C'][i] == experiment_result['Z6'][i]:
            par += 1
        if experiment_result['C'][i] == experiment_result['Z7'][i]:
            par += 1
        if experiment_result['C'][i] == experiment_result['Z8'][i]:
            par += 1
            
    alpha_hat = par
    return alpha_hat


def transition_frequency(n):
    beta_hat=0
    gamma_hat=0
    C_vector = experiment_result['C'].copy()
    
    for i in range(n-1):
        if (C_vector[i]==2 and C_vector[i+1] in [0, 1]) == True:
            beta_hat += 1
        if (C_vector[i] in [0, 1] and C_vector[i+1]==2) == True:
            gamma_hat += 1
    beta_hat = beta_hat / n 
    gamma_hat = gamma_hat / n
    return beta_hat , gamma_hat

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    