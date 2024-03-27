import numpy as np
import pandas as pd
import os
import re

class HMM():
    def __init__(self, T, n) -> None:
        self.T = T
        self.n = n
        self.proba_was_set = False
        self.first_node = C_node(0, n, None, self)
        self.C_node_list = [self.first_node]
        for t in range(1, T):
            self.C_node_list.append(C_node(t, n, self.C_node_list[-1], self))

    def set_proba_paras(self, para_dict):
        for t in range(self.T):
            self.C_node_list[t].set_proba_paras(para_dict)
        self.proba_was_set = True
        print("Probability parameters were set.")

    def set_transition_matrix_in_Cs(self, mat):
        for t in range(self.T):
            self.C_node_list[t].set_transition_matrix(mat)

    def start_simulation(self, first_C_value=2):
        assert self.proba_was_set
        self.C_node_list[0].set_C_value(first_C_value)
        results_dict = {"t=0": {"C_t": 2, "(Z, X)_t_i": self.C_node_list[0].simulate_children()}}
        for t, c in enumerate(self.C_node_list[1:]):
            this_c_val = c.simulate()
            results_dict[f"t={t+1}"] = {"C_t": this_c_val, f"(Z, X)t_i": c.simulate_children()}
        return results_dict

    def reset_HMM_values(self):
        for t in range(self.T):
            self.C_node_list[t].reset()

    def give_current_state(self):
        cur_state_dict = {}
        for t, c in enumerate(self.C_node_list):
            cur_state_dict[f"t={t}"] = {"C_t": c.C_value, f"(Z, X)t_i": [(z.Z_value, z.X_node.X_value) for z in c.Z_node_list]}
        return cur_state_dict
    
    def show_current_state(self):
        cur_state = self.give_current_state()
        for elem in cur_state.items():
            print(elem)
        return cur_state
    
    def load_df(self, df, load_Z=True, load_C=True):
        # expects a df that has fitting dimensions (self.n, self.T)
        self.reset_HMM_values()
        for idx, row in df.iterrows():
            this_t = row["t"]
            this_C = self.C_node_list[int(this_t)-1]
            # might want to set C_value
            if f"C" in df.columns and not pd.isna(row["C"]) and load_C:
                new_C_value = int(row["C"])
                this_C.set_C_value(new_C_value)
            # setting X_values in HMM
            for i, z in enumerate(this_C.Z_node_list):
                new_X_value = int(row[f"X{i+1}"])
                z.X_node.set_X_value(new_X_value)
                # might also want to set Z_values
                if f"Z{i+1}" in df.columns and not pd.isna(row[f"Z{i+1}"]) and load_Z:
                    new_Z_value = int(row[f"Z{i+1}"])
                    z.set_Z_value(new_Z_value)

    def state_as_df(self, state_dict=None):
        # t starts at 1 so our storage format aligns with given data sets
        col_names = ["t", "C"]
        col_names.extend([f"Z{t+1}" for t in range(len(self.C_node_list[0].Z_node_list))])
        col_names.extend([f"X{t+1}" for t in range(len(self.C_node_list[0].Z_node_list))])
        
        np_state = self.state_as_np(state_dict)
        new_df = pd.DataFrame(columns=col_names, data=np_state)
        return new_df

    def state_as_np(self, state_dict=None):
        # returns T many rows of shape: [t, C_value, Z_1, Z_2, Z_3, X_1, X_2, X3]
        # t starts at 1 so our storage format aligns with given data sets
        state_as_lists = []
        for t, C in enumerate(self.C_node_list):
            this_t_state = [t+1, C.C_value]
            this_t_state.extend([z.Z_value for z in C.Z_node_list])
            this_t_state.extend([z.X_node.X_value for z in C.Z_node_list])
            state_as_lists.append(this_t_state)
        return np.array(state_as_lists)


class C_node():
    # possible value: {0, 1, 2} -> serial processing (0 vs 1), or parallel (2) 
    def __init__(self, t, n, parent_node, parent_HMM) -> None:
        self.name=f"C_node_{t}"
        self.t = t
        self.n = n
        self.parent = parent_node
        self.parent_HMM = parent_HMM
        self.Z_node_list = [Z_node(t, i, self) for i in range(n)]
        
        if t==1:
            self.C_value = 2
        else:
            self.C_value = None # {0, 1, 2}

        # probability parameters
        self.gamma = None
        self.beta = None
        self.tranisition_matrix = None

    def set_proba_paras(self, para_dict):
        assert 0 < para_dict["gamma"] < 1
        assert 0 < para_dict["beta"] < 1

        self.gamma = para_dict["gamma"]
        self.beta = para_dict["beta"]
        self.set_transition_matrix()
        for i in range(self.n):
            self.Z_node_list[i].set_proba_paras(para_dict)

    def set_transition_matrix(self, mat=None):
        if mat is None:
            assert not self.gamma is None
            assert not self.beta is None
            mat = np.zeros((3, 3))
            mat[0, 0] = 1 - self.gamma
            mat[1, 1] = 1 - self.gamma
            mat[0, 2] = self.gamma
            mat[1, 2] = self.gamma
            mat[2, 0] = self.beta / 2
            mat[2, 1] = self.beta / 2
            mat[2, 2] = 1 - self.beta
        self.tranisition_matrix = mat

    def set_C_value(self, new_C_value):
        assert new_C_value in [None, 0, 1, 2]
        self.C_value = new_C_value
        
    def simulate(self):
        if not self.t == 0: # all but first node have parent
            assert not self.parent.C_value is None

        # take probability distribution from state of previous C_value:
        probas_for_next_state = list(self.tranisition_matrix[self.parent.C_value, :])
        sim_result = np.random.choice([0, 1, 2], p=probas_for_next_state)

        self.set_C_value(int(sim_result))
        return sim_result

    def simulate_children(self):
        assert not self.C_value is None
        results_list = []
        for z in self.Z_node_list:
            this_z = z.simulate()
            results_list.append((this_z, z.simulate_children()))
        return results_list

    def reset(self):
        self.set_C_value(None)
        for z in self.Z_node_list:
            z.reset()


class Z_node():
    # possible value: {0, 1} -> object of focus
    def __init__(self, t, i, parent) -> None:
        self.name=f"Z_node_{t}_{i}"
        self.t = t  # 1 to T
        self.i = i  # 1 to n
        self.parent = parent
        self.X_node = X_node(t, i, self)
        
        self.Z_value = None # {0, 1}

        # probability parameters
        self.alpha = None

    def set_proba_paras(self, para_dict):
        assert 0.5 < para_dict["alpha"] < 1
        self.alpha = para_dict["alpha"]
        self.X_node.set_proba_paras(para_dict)

    def set_Z_value(self, new_Z_value):
        if not new_Z_value in [None, 0, 1]:
            print(new_Z_value)
            assert new_Z_value in [None, 0, 1]
        self.Z_value = new_Z_value

    def simulate(self):
        # compute the Z_value according to given proba dist
        if self.parent.C_value == 0:
            sim_result = np.random.choice([0, 1], p=[self.alpha, 1-self.alpha])
        elif self.parent.C_value == 1:
            sim_result = np.random.choice([0, 1], p=[1-self.alpha, self.alpha])
        elif self.parent.C_value == 2:
            sim_result = np.random.choice([0, 1], p=[0.5, 0.5])
        else:
            raise ValueError
        self.set_Z_value(sim_result)
        return sim_result

    def simulate_children(self):
        assert not self.Z_value is None
        return self.X_node.simulate()

    def reset(self):
        self.set_Z_value(None)
        self.X_node.reset()


class X_node():
    # possible value: [0, ...) -> observations: poisson distributed
    def __init__(self, t, i, parent) -> None:
        self.name=f"X_node_{t}_{i}"
        self.t = t
        self.i = i
        self.parent = parent
        
        self.X_value = None # [0, ...)

        # probability parameters
        self.lambda_Z0 = None   # poisson mean for Z=0
        self.lambda_Z1 = None
    
    def set_proba_paras(self, para_dict):
        assert 0 < para_dict["lambda_Z0"]
        assert 0 < para_dict["lambda_Z1"]

        self.lambda_Z0 = para_dict["lambda_Z0"]
        self.lambda_Z1 = para_dict["lambda_Z1"]

    def set_X_value(self, new_X_value):
        assert isinstance(new_X_value, int) or new_X_value is None
        self.X_value = new_X_value

    def simulate(self):
        if self.parent.Z_value == 0:
            sim_result = int(np.random.poisson(self.lambda_Z0, size=1)[0])
        elif self.parent.Z_value == 1:
            sim_result = int(np.random.poisson(self.lambda_Z1, size=1)[0])
        else:
            raise ValueError
        self.set_X_value(sim_result)
        return sim_result
        
    def reset(self):
        self.set_X_value(None)


def load_csv_as_HMM(csv_file_path):
    # expects a path to a csv file.
    # Reads each row as: t,X1,X2,X3,...
    df = pd.read_csv(csv_file_path)
    this_T = len(df)
    this_n = len(df.columns) - 1    # "t" column is not an X value
    this_HMM = HMM(T=this_T, n=this_n)
    this_HMM.load_df(df)
    return this_HMM


def generate_training_datasets(sim_para_dict, T=100, n=10, nr_datasets=100):
    if not os.path.exists("simulated_datasets"):
        os.makedirs("simulated_datasets")
    sim_HMM = HMM(T=T, n=n)
    sim_HMM.set_proba_paras(sim_para_dict)
    for i in range(nr_datasets):
        sim_HMM.reset_HMM_values()
        sim_HMM.start_simulation()
        simulated_df = sim_HMM.state_as_df()
        simulated_df.to_csv(f"simulated_datasets/sim_nr{i}.csv", index=False)
    print(f"{nr_datasets} samples have been generated for T={T}, n={n}, proba paras:", sim_para_dict)


def construct_X(HMM_state_df):
    '''
    Constructs the "bold X" variable from the task sheet that should be used for inference tasks.
    '''
    X_col_names = [col for col in HMM_state_df.columns if bool(re.match(r'^X\d+$', col))]
    X = HMM_state_df[X_col_names].values
    return X


## HOW TO USE THE CODE

'''
# generate data from HMM with given dimensions and proba paras
sim_para_dict = {
    "gamma": 0.1,
    "beta": 0.2,
    "alpha": 0.9,
    "lambda_Z0": 1., 
    "lambda_Z1": 5.
}
generate_training_datasets(T=100, n=10, sim_para_dict=sim_para_dict, nr_datasets=100)
exit()
'''

'''
# loading a HMM from a given csv file (uses df in background)
loaded_HMM = load_csv_as_HMM("proj_HMM/Ex_10.csv")

# printing the state of the HMM to console
loaded_HMM.show_current_state()

# getting the HMM in form of a df
state_df = loaded_HMM.state_as_df()

# sets all values in the HMM to None
loaded_HMM.reset_HMM_values()
print(state_df)

# df state can be loaded back using this...
loaded_HMM.load_df(state_df)
loaded_HMM.show_current_state()


# initialize a new HMM with given dimensions T, n
new_HMM = HMM(T=7, n=4)

# sets probability parameters in the HMM
sim_para_dict = {
    "gamma": 0.1,
    "beta": 0.2,
    "alpha": 0.9,
    "lambda_Z0": 1., 
    "lambda_Z1": 5.
}
new_HMM.set_proba_paras(sim_para_dict)

# does a simulation using the given proba parameters. Needs to have set proba paras before!
new_HMM.start_simulation()
new_HMM.show_current_state()
experiment_result = new_HMM.state_as_df()
'''