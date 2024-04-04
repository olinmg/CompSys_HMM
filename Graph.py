import numpy as np
import pandas as pd
import os
import re
from scipy.stats import poisson
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

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
        results_dict = {"t=0": {"C_t": first_C_value, "(Z, X)_t_i": self.C_node_list[0].simulate_children()}}
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

    def trigger_upwards_pass(self, root):
        '''
        Triggers upwards pass function in all leaf nodes (all X nodes) to a chosen root node.
        '''
        # cant choose the last or first C as root for simplicity of implementation
        assert root.t != 1
        assert root.t != self.T

        for C in self.C_node_list:
            for Z in C.Z_node_list:
                Z.X_node.upward_message_passing(root)
                # actually works, since C node returns when it hasn't received enough messages
                # otherwise would never execute more than the first loop

    def trigger_downwards_pass(self, root):
        '''
        Triggers a downward message passing from a chosen root node.
        '''
        root.downward_message_passing(received_down_message=np.array([[1],[1],[1]]), root=root)
        #print("Completed downwards message passing.")


    def trigger_belief_comp(self):
        '''
        Makes all C and Z nodes compute their beliefs. 
        Can only be executed after trigger_upwards_pass() + trigger_downwards_pass().
        '''
        for C in self.C_node_list:
            for Z in C.Z_node_list:
                Z.compute_belief()
                #print(Z.belief)

class C_node():
    # possible value: {0, 1, 2} -> serial processing (0 vs 1), or parallel (2) 
    def __init__(self, t, n, parent_node, parent_HMM) -> None:
        self.name=f"C_node_{t}"
        self.t = t
        self.n = n
        self.parent = parent_node
        self.parent_HMM = parent_HMM
        self.Z_node_list = [Z_node(t, i, self) for i in range(n)]
        
        # for message passing
        self.initial_potential = None
        self.up_message = None
        self.down_message = None
        self.from_uppass_msg = []
        self.from_uppass_msg_C = None
        self.from_downpass_msg = None
        self.belief = None

        if t==1:
            self.C_value = 2
        else:
            self.C_value = None # {0, 1, 2}

        # probability parameters
        self.gamma = None
        self.beta = None
        self.transition_matrix = None

    def set_proba_paras(self, para_dict):
        assert 0 < para_dict["gamma"] < 1
        assert 0 < para_dict["beta"] < 1

        self.gamma = para_dict["gamma"]
        self.beta = para_dict["beta"]
        self.set_transition_matrix()
        # TODO: still need to check if transition matrix is defined the right way around!
        self.initial_potential = copy.deepcopy(self.transition_matrix)
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
        self.transition_matrix = mat
        self.initial_potential = self.transition_matrix


    def set_C_value(self, new_C_value):
        assert new_C_value in [None, 0, 1, 2]
        self.C_value = new_C_value
        
    def simulate(self):
        if not self.t == 0: 
            # make sure value in parent is set -> need for conditioning
            assert not self.parent.C_value is None

        # take probability distribution from state of previous C_value:
        probas_for_next_state = list(self.transition_matrix[self.parent.C_value, :])
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


    def upward_message_passing(self, received_up_message, root, from_C=False):
        '''
        Passes a message upwards towards the C-node.
        Representative for message: {Z(t,i), C(t)} -> {C(t), C(t-1)}
        '''
        # append received message
        self.set_transition_matrix()

        # when collected enough messages: do own upwards message
        if not from_C:
            self.from_uppass_msg.append(received_up_message)
        else:
            self.from_uppass_msg_C = received_up_message

        if self.t == 0 and len(self.from_uppass_msg) == self.n:
            # special message sent by C_0
            prod_Z_msgs = np.prod([np.reshape(vec, (3,1)) for vec in self.from_uppass_msg], axis=0)
            self.up_message = np.array([[0], [0], [1]]) * prod_Z_msgs #(3,1)
            self.up_message = self.up_message/np.sum(self.up_message)
            self.parent_HMM.C_node_list[self.t+1].upward_message_passing(self.up_message, root, from_C=True)

        if self.t==self.parent_HMM.T-1 and len(self.from_uppass_msg) == self.n:
            # special message sent by C_T
            prod_Z_msgs = np.prod([np.reshape(vec, (3,1)) for vec in self.from_uppass_msg], axis=0)
            self.up_message = np.dot(self.transition_matrix, prod_Z_msgs)
            self.up_message = self.up_message/np.sum(self.up_message)
            self.parent.upward_message_passing(self.up_message, root, from_C=True)

        if len(self.from_uppass_msg) == self.n and not self.from_uppass_msg_C is None:
            # all nodes that are not C_0 or C_T

            # received all needed messages from upwards pass -> construct own upwards message
            prod_Z_msgs = np.prod([np.reshape(vec, (3,1)) for vec in self.from_uppass_msg], axis=0)
            #print(prod_Z_msgs.shape) # (3,1)

            if root.t == self.t:
                # this node is root, thus the upwards pass is completed
                #print("Completed upwards message passing.")
                return

            # figure out which direction to do upwards pass to: parent C(t-1) or child C(t+1)
            if root.t < self.t:
                # send message to parent: C(t-1)
                self.up_message = np.dot(self.transition_matrix, prod_Z_msgs * self.from_uppass_msg_C)
                self.up_message = self.up_message/np.sum(self.up_message)
                self.parent.upward_message_passing(self.up_message, root, from_C=True)
            else:
                # send message to child: C(t+1)
                mod_tra_mat = copy.deepcopy(self.transition_matrix)    
                mod_tra_mat[:, 0] *= prod_Z_msgs[0]
                mod_tra_mat[:, 1] *= prod_Z_msgs[1]
                mod_tra_mat[:, 2] *= prod_Z_msgs[2]
                self.up_message = np.dot(mod_tra_mat.transpose(), self.from_uppass_msg_C)
                self.up_message = self.up_message/np.sum(self.up_message)
                self.parent_HMM.C_node_list[self.t+1].upward_message_passing(self.up_message, root, from_C=True)
        else:
            # returning after this statement will allow for the next X(t,i) to start its upwards pass in: trigger_upwards_pass
            return
        

    def downward_message_passing(self, received_down_message, root):
        '''
        Passes a message downwards towards the Z-nodes and a C-node.
        Representative for message: {C(t), C(t-1)} -> {Z(t,i), C(t)}, for multiple i
                                and ( {C(t), C(t-1)} -> {C(t+1), C(t)} or {C(t), C(t-1)} -> {C(t-1), C(t-2)} )
        Root is always one of the C nodes.
        '''

        self.from_downpass_msg = received_down_message  # this always comes from a C node
        all_uppass_msgs = [np.reshape(vec, (3,1)) for vec in self.from_uppass_msg]
        prod_all_up_msgs = np.prod(all_uppass_msgs, axis=0)
        ctmin_ctminmin = None
        ctplus_ct = None


        if self.t == 0:
            # only need to send messages to the Z children
            for Z_node in self.Z_node_list:
                # send product of all messages from uppass, besides the one that whas send by the Z_node we are sending this message to
                filtered_up_msgs = [arr for arr in all_uppass_msgs if not np.array_equal(arr, np.reshape(Z_node.up_message, (3,1)))]
                prod_filtered_up_msgs = np.prod(filtered_up_msgs, axis=0)
                this_down_message = np.array([[0],[0],[1]]) * prod_filtered_up_msgs * received_down_message
                if np.sum(this_down_message) != 0:
                    Z_node.downward_message_passing(this_down_message/np.sum(this_down_message))
                else:
                    Z_node.downward_message_passing(this_down_message)
            return
            

        if self.t == self.parent_HMM.T-1:
            for Z_node in self.Z_node_list:
            # send product of all messages from uppass, besides the one that whas send by the Z_node we are sending this message to
                filtered_up_msgs = [arr for arr in all_uppass_msgs if not np.array_equal(arr, np.reshape(Z_node.up_message, (3,1)))]
                prod_filtered_up_msgs = np.prod(filtered_up_msgs, axis=0)
                this_down_message = prod_filtered_up_msgs # [u, v, w] from script
                mod_tra_mat = copy.deepcopy(self.transition_matrix)
                mod_tra_mat[:, 0] *= this_down_message[0]
                mod_tra_mat[:, 1] *= this_down_message[1]
                mod_tra_mat[:, 2] *= this_down_message[2]
                this_down_message = np.dot(mod_tra_mat.transpose(), received_down_message)
                Z_node.downward_message_passing(this_down_message/np.sum(this_down_message))
            return

        if self.t <= root.t:
            # the message from Ct-1, Ct-2 is in from_uppass_msg_C
            # and msg from Ct+1, Ct in received_down_message
            ctmin_ctminmin = self.from_uppass_msg_C
            ctplus_ct = received_down_message
            mod_tra_mat = copy.deepcopy(self.transition_matrix)    
            mod_tra_mat[:, 0] *= prod_all_up_msgs[0]
            mod_tra_mat[:, 1] *= prod_all_up_msgs[1]
            mod_tra_mat[:, 2] *= prod_all_up_msgs[2]
            this_down_message = np.dot(mod_tra_mat.transpose(), ctmin_ctminmin)
            self.parent.downward_message_passing(this_down_message/np.sum(this_down_message), root)


        if self.t >= root.t:
            # the message from Ct-1, Ct-2 is in received_down_message
            # and msg from Ct+1, Ct in from_uppass_msg_C
            ctmin_ctminmin = received_down_message
            ctplus_ct = self.from_uppass_msg_C
            this_down_message = np.dot(self.transition_matrix, prod_all_up_msgs * ctplus_ct)
            self.parent_HMM.C_node_list[self.t+1].downward_message_passing(this_down_message/np.sum(this_down_message), root)
        
        # pass message downwards to all connected Z nodes: 
        for Z_node in self.Z_node_list:
            # all uppass msgs, but the one from this Z_node
            filtered_up_msgs = [arr for arr in all_uppass_msgs if not np.array_equal(arr, np.reshape(Z_node.up_message, (3,1)))]
            prod_filtered_up_msgs = np.prod(filtered_up_msgs, axis=0)
            # send product of all messages from uppass, besides the one that whas send by the Z_node we are sending this message to
            this_down_message = ctplus_ct * prod_filtered_up_msgs # [u, v, w] from script
            mod_tra_mat = copy.deepcopy(self.transition_matrix)
            mod_tra_mat[:, 0] *= this_down_message[0]
            mod_tra_mat[:, 1] *= this_down_message[1]
            mod_tra_mat[:, 2] *= this_down_message[2]
            this_down_message = np.dot(mod_tra_mat.transpose(), ctmin_ctminmin)
            if np.sum(this_down_message) != 0:
                Z_node.downward_message_passing(this_down_message/np.sum(this_down_message))
            else:
                Z_node.downward_message_passing(this_down_message)

        

    def compute_belief(self):
        # incoming messages
        pass     


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
        self.parent = parent    # an object of class "C_node"
        self.X_node = X_node(t, i, self)
        
        self.Z_value = None # {0, 1}

        # for message passing
        self.initial_potential = None
        self.up_message = None
        self.down_message = None
        self.from_uppass_msg = None
        self.from_downpass_msg = None
        self.belief = None

        # probability parameters
        self.alpha = None

    def set_proba_paras(self, para_dict):
        assert 0.5 < para_dict["alpha"] < 1
        self.alpha = para_dict["alpha"]
        self.initial_potential = np.array([[self.alpha, 1-self.alpha], [1-self.alpha, self.alpha], [0.5, 0.5]])
        self.X_node.set_proba_paras(para_dict)

    def set_Z_value(self, new_Z_value):
        if not new_Z_value in [None, 0, 1]:
            print(new_Z_value)
            assert new_Z_value in [None, 0, 1]
        self.Z_value = new_Z_value

    def simulate(self):
        # randomly sample the Z_value according to given proba dist
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


    def upward_message_passing(self, up_message_from_X, root):
        '''
        Passes a message upwards towards the C-node.
        Representative for message: {Z(t,i), C(t)} -> {C(t), C(t-1)}
        '''
        # store the message from X created in upwards pass
        self.from_uppass_msg = up_message_from_X
        #print(self.initial_potential.shape) # (3,2)
        #print(up_message_from_X.shape) #(2,)
        self.up_message = np.dot(self.initial_potential, up_message_from_X)
        #print(self.up_message.shape) # (3,)
        if np.sum(self.up_message) != 0:
            self.up_message = self.up_message/np.sum(self.up_message)
        
        # send the message to the parent C node
        self.parent.upward_message_passing(self.up_message, root)


    def downward_message_passing(self, down_message_from_C):
        '''
        Passes a message downwards towards the X-node.
        Representative for message: {Z(t,i), C(t)} -> {X(t,i), Z(t,i)}

        Don't actually need this downward pass, since X(t,i) is always observed -> no need for message passing to it.
        '''
        # store the message from C created in downwards pass
        self.from_downpass_msg = down_message_from_C

        self.down_message = None
        self.X_node.downward_message_passing(self.down_message)


    def compute_belief(self):
        # the result should be two dimensional: giving probability for Z=0 and for Z=1
        # TODO: this returns wrong results
        beta_belief = self.initial_potential 
        #print(beta_belief.shape) # (3,2)
        beta_belief[0, :] *= self.from_downpass_msg[0]
        beta_belief[1, :] *= self.from_downpass_msg[1]
        beta_belief[2, :] *= self.from_downpass_msg[2]
        beta_belief[:, 0] *= self.from_uppass_msg[0]
        beta_belief[:, 1] *= self.from_uppass_msg[1]
        self.belief = beta_belief
        self.belief_Z = np.sum(self.belief, axis=0)
        if np.sum(self.belief_Z) != 0:
            self.belief_Z /= np.sum(self.belief_Z)
        self.belief_C = np.sum(self.belief, axis=1)
        if np.sum(self.belief_C) != 0:
            self.belief_C /= np.sum(self.belief_C)
        #print(self.belief_Z)
        #print(self.belief_C)
        return self.belief_Z, self.belief_C
        

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
        
        # for message passing
        self.initial_potential = None
        self.up_message = None
        self.down_message = None
        self.from_downpass_msg = None
        # no uppass_msg, since is leaf

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
        self.set_X_value(sim_result)
        return sim_result
        
    def set_initial_potential(self):
        observation = self.X_value
        proba_Z0 = poisson.pmf(observation, self.lambda_Z0)
        proba_Z1 = poisson.pmf(observation, self.lambda_Z1)
        self.initial_potential = np.array([proba_Z0, proba_Z1])
        return self.initial_potential

    def upward_message_passing(self, root):
        '''
        Passes a message upwards towards the Z-node.
        Representative for message: {X(t,i), Z(t,i)} -> {Z(t,i), C(t)}
        '''
        self.set_initial_potential()
        self.up_message = self.initial_potential
        self.up_message = self.up_message/np.sum(self.up_message)

        # send the message to the parent Z node
        self.parent.upward_message_passing(self.up_message, root)


    def downward_message_passing(self, down_message_from_Z):
        '''
        Doesn't exist for X-node, as it's a leaf in the clique tree.
        '''
        self.from_downpass_msg = down_message_from_Z

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
    for i in tqdm(range(nr_datasets)):
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


def visualize_gen_data(shape, state_df):
    # Sample data for illustration purposes
    # shape = (T, n)

    T = shape[0]
    n = shape[1]

    time_intervals = range(T)

    cmap_colors = [(0.9, 0.7, 0.7), #red
                   (0.7, 0.9, 0.7), #green
                    (0.5, 0.6, 0.95), #blue
                    ]
    cust_cmap = LinearSegmentedColormap.from_list('custom_cmap', cmap_colors, N=3)


    # Line plot for individual neuron activities
    fig, axs = plt.subplots(n, 1, figsize=(10, 6), sharex=True, sharey=True)
    for i in range(n):
        values_this_Xi_over_time = state_df[f"X{i+1}"].values
        axs[i].plot(time_intervals, values_this_Xi_over_time, marker='o', color="black", markersize=2.5, linewidth=1)
        
        values_this_Zi_over_time = state_df[f"Z{i+1}"].values*15

        for j in range(len(values_this_Zi_over_time) - 1):
            this_C_t = state_df["C"].values[j]
            axs[i].plot([j, j + 1], [values_this_Zi_over_time[j], values_this_Zi_over_time[j + 1]], marker='o', color=cust_cmap(this_C_t), markersize=2.5, linewidth=1)

        axs[i].set_ylabel(f'X{i+1}')
        axs[i].grid(True)

    plt.xlabel("t")
    plt.legend()
    plt.show()


def test_C(shape):
    new_HMM = HMM(T=shape[0], n=shape[1])
    sim_para_dict = {
        "gamma": 0.1,
        "beta": 0.2,
        "alpha": 0.9,
        "lambda_Z0": 1., 
        "lambda_Z1": 5.
    }
    new_HMM.set_proba_paras(sim_para_dict)
    C_diff_0 = [[] for _ in range(shape[0])]
    C_diff_1 = [[] for _ in range(shape[0])]
    C_diff_2 = [[] for _ in range(shape[0])]

    for _ in range(10):
        new_HMM.reset_HMM_values()
        new_HMM.start_simulation()
        root_for_mp = new_HMM.C_node_list[2]
        new_HMM.trigger_upwards_pass(root_for_mp)
        new_HMM.trigger_downwards_pass(root_for_mp)
        new_HMM.trigger_belief_comp()
        
        #new_HMM.show_current_state()

        for idx, C_node in enumerate(new_HMM.C_node_list):
            some_Z = C_node.Z_node_list[3]
            if C_node.C_value == 0:
                C_diff_0[idx].append(1 - some_Z.belief_C[0])
            else:
                C_diff_0[idx].append(-some_Z.belief_C[0])
            
            if C_node.C_value == 1:
                C_diff_1[idx].append(1 - some_Z.belief_C[1])
            else:
                C_diff_1[idx].append(-some_Z.belief_C[1])
            if C_node.C_value == 2:
                C_diff_2[idx].append(1 - some_Z.belief_C[2])
            else:
                C_diff_2[idx].append(-some_Z.belief_C[2])
    print("Means for C==0: ", [sum(C_t) / len(C_t) for C_t in C_diff_0])
    print("Means for C==1: ", [sum(C_t) / len(C_t) for C_t in C_diff_1])
    print("Means for C==2: ", [sum(C_t) / len(C_t) for C_t in C_diff_2])


'''
## HOW TO USE THE CODE

# 0. probability parameters for a HMM
sim_para_dict = {
    "gamma": 0.1,
    "beta": 0.2,
    "alpha": 0.9,
    "lambda_Z0": 1., 
    "lambda_Z1": 5.
}

# 1. generate forward simulation as follows:
new_HMM = HMM(T=100, n=10)  # create a new instance of the HMM
new_HMM.set_proba_paras(sim_para_dict)  # set probability parameters to values from a dict
new_HMM.start_simulation()  # trigger the forward simulation in the leafs (here: X_nodes)
visualize_gen_data((100, 10), new_HMM.state_as_df())    # show the generated series visually

# 2. reset and load back values of HMM
state_df = new_HMM.state_as_df()    # get values of HMM as df
new_HMM.reset_HMM_values()  # reset values of HMM
new_HMM.load_df(state_df)   # load state_df back into the HMM

# 3. loading a HMM from a given csv file (uses df in background)
loaded_HMM = load_csv_as_HMM("proj_HMM/Ex_1.csv")   # initializes HMM and sets all available values
loaded_HMM.show_current_state()    # show the given HMM in text

# 4. doing message passing on the clique tree of HMM and compute C, Z distributions conditioned on X
root_for_mp = loaded_HMM.C_node_list[4] # choose a root node from which to execute message passing (not: C_0 or C_T)
loaded_HMM.trigger_upwards_pass(root_for_mp)    # does upward message passing
loaded_HMM.trigger_downwards_pass(root_for_mp)  # does downward message passing
loaded_HMM.trigger_belief_comp()  # computes the beliefs in Z_nodes and conditional distributions for C, Z

# 5. generate a whole training datasets, saved as csv to folder "./simulated_datasets"
generate_training_datasets(T=100, n=10, sim_para_dict=sim_para_dict, nr_datasets=100)

# 6. testing if computed conditional distributions of C_t are correct
test_C(shape=(10, 4))
'''
