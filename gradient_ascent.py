import pandas as pd
import random
import math
import numpy as np

# Read the CSV file
df = pd.read_csv("Ex_2.csv")

lst = df['X1'].tolist() + df['X2'].tolist() + df['X3'].tolist() + df['X4'].tolist() + df['X5'].tolist() + df['X6'].tolist() + df['X7'].tolist() + df['X8'].tolist()

#calculating Z with one gradient

def gradient(lst):
    final_len = 9999999999
    for j in range(100):
        num_lst = [0,0]
        while num_lst[0] == num_lst[1]:
            num_lst = lst[random.randint(0, len(lst)) - 1], lst[random.randint(0, len(lst) - 1)]

        d0 = []
        d1 = []

        for i in range(len(lst)):
            if math.sqrt((min(num_lst) - lst[i]) ** 2) <= math.sqrt((max(num_lst) - lst[i]) ** 2):
                d0.append(lst[i])
            else:
                d1.append(lst[i])

        total_len = 0

        for i in range(len(d0)):
            total_len += math.sqrt((np.mean(d0) - d0[i]) ** 2)

        for i in range(len(d1)):
            total_len += math.sqrt((np.mean(d1) - d1[i]) ** 2)

        if total_len < final_len:
            final_len = total_len
            final_mean_low = np.mean(d0)
            final_mean_high = np.mean(d1)
            final_d0 = d0
            final_d1 = d1

    return final_len, final_mean_low, final_mean_high, final_d0, final_d1

final_len, final_mean_low, final_mean_high, final_d0, final_d1 = gradient(lst)

#list for Z

z_lst = []

for i in range(len(lst)):
    if math.sqrt((final_mean_low - lst[i]) ** 2) <= math.sqrt((final_mean_high - lst[i]) ** 2):
        z_lst.append(0)
    else:
        z_lst.append(1)

z_lst = np.array(z_lst)
z_lst = z_lst.reshape(100,8)

sum_lst = []

for i in range(len(z_lst)):
    sum_lst.append(sum(z_lst[i]))

#calculating C with two gradients

def twogradient(lst):
    final_len = 9999999999
    for j in range(100):
        num_lst = [0,0,0]
        while num_lst[0] == num_lst[1] or num_lst[0] == num_lst[2] or num_lst[1] == num_lst[2]:
            num_lst = lst[random.randint(0, len(lst)) - 1], lst[random.randint(0, len(lst) - 1)], lst[random.randint(0, len(lst)) - 1]

        d0 = []
        d1 = []
        d2 = []

        for i in range(len(lst)):
            if math.sqrt((min(num_lst) - lst[i]) ** 2) < math.sqrt((max(num_lst) - lst[i]) ** 2) and math.sqrt((np.median(num_lst) - lst[i]) ** 2):
                d0.append(lst[i])
            elif math.sqrt((max(num_lst) - lst[i]) ** 2) < math.sqrt((min(num_lst) - lst[i]) ** 2) and math.sqrt((np.median(num_lst) - lst[i]) ** 2):
                d1.append(lst[i])
            else: 
                d2.append(lst[i])

        total_len = 0

        for i in range(len(d0)):
            total_len += math.sqrt((np.mean(d0) - d0[i]) ** 2)

        for i in range(len(d1)):
            total_len += math.sqrt((np.mean(d1) - d1[i]) ** 2)

        for i in range(len(d2)):
            total_len += math.sqrt((np.mean(d2) - d2[i]) ** 2)

        if total_len < final_len:
            final_len = total_len
            final_mean_low = np.mean(d0)
            final_mean_high = np.mean(d1)
            final_mean_median = np.mean(d2)

    return final_len, final_mean_low, final_mean_high, final_mean_median


final_len, final_mean_low, final_mean_high, final_mean_median = twogradient(lst)

#list of C

c_lst = []

for i in range(len(sum_lst)):
    if math.sqrt((final_mean_low - sum_lst[i]) ** 2) < math.sqrt((final_mean_high - lst[i]) ** 2) and math.sqrt((final_mean_median - lst[i]) ** 2):
        c_lst.append(0)
    elif math.sqrt((final_mean_high - lst[i]) ** 2) < math.sqrt((final_mean_low - lst[i]) ** 2) and math.sqrt((final_mean_median - lst[i]) ** 2):
        c_lst.append(1)
    else: 
        c_lst.append(2)

c_lst = np.array(c_lst)

#final table

z_df = pd.DataFrame(z_lst, columns=['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8']) 

c_df = pd.DataFrame(c_lst, columns=['C'])

cat = pd.concat([c_df, z_df, df], axis=1, join="inner")



par = 0

for i in range(len(c_df)):
    if c_df['C'][i] == z_df['Z1'][i]:
        par += 1
    if c_df['C'][i] == z_df['Z2'][i]:
        par += 1
    if c_df['C'][i] == z_df['Z3'][i]:
        par += 1
    if c_df['C'][i] == z_df['Z4'][i]:
        par += 1
    if c_df['C'][i] == z_df['Z5'][i]:
        par += 1
    if c_df['C'][i] == z_df['Z6'][i]:
        par += 1
    if c_df['C'][i] == z_df['Z7'][i]:
        par += 1
    if c_df['C'][i] == z_df['Z8'][i]:
        par += 1


beta = 0
gamma = 0
total_onezero = 0
total_two = 0

for i in range(len(c_lst) - 1):
    if c_df['C'][i] == 0 or 1:
        total_onezero +=1
        if c_df['C'][i+1] == 2:
            gamma += 1
    if c_df['C'][i] == 2:
        total_two +=1
        if c_df['C'][i+1] == 0 or 1:
            beta += 1

print(beta / total_onezero, gamma / total_two)
            




