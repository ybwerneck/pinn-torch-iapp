import numpy as np
import matplotlib.pyplot as plt
import os
import chaospy as cp
from FHNCUDAlib import FHNCUDA

# Constants
t_w = 100
nc = 12


def generateExactSolution(t, dt, x0, w0, rate, P, begin, end):
    n2 = int(t / dt) + 2
    n = int((end - begin) / (dt * rate))
    Sol = np.zeros((n, 3))
    Sol2 = np.zeros((n2, 2))
    Sol2[0] = x0, w0
    T = 0
    k = 0

    while k < n2 - 1:
        x, w = Sol2[k]
        Sol2[k + 1] = 10 * (x * (x - 0.4) * (1 - x) - w + P) * dt + x, 0.2 * (x * 0.2 - 0.8 * w) * dt + w

        if (k * dt == begin or ((k + 1) % rate == 0 and k * dt >= begin and k * dt <= end)) and T < n:
            Sol[T] = Sol2[k][0], Sol2[k][1], k * dt
            T += 1

        k += 1
        if k * dt > end:
            break
    return Sol

def unique(array):
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]

def getDatasetFromDistA(sample_set, data_folder="./extrapol/", ti=0, tf=50, r=1, deltaT=0.01):
    try:
        os.mkdir(data_folder)
    except:
        print("Overwriting")
    
    T = np.empty(0)
    K = np.empty(0)
    V = np.empty(0)
    U = np.empty(0)
    print("Generating dataset in", data_folder)

    rate = 10 * r
    num_samples = np.shape(sample_set)[1]
    print(np.shape(sample_set))

    u, v, t, p = FHNCUDA.run(sample_set.T, tf, deltaT, rate)
    SOLs = np.array(u).flatten()
    SOLw = np.array(v).flatten()
    t = t[0]
    T = np.tile(t, num_samples)
    K = np.repeat(unique(sample_set[2, :]), len(t))
    U = np.repeat(unique(sample_set[0, :]), len(t))
    V = np.repeat(unique(sample_set[1, :]), len(t))
    n = len(t)

    SOLs = SOLs[np.arange(len(SOLs)) % n != 0]
    SOLw = SOLw[np.arange(len(SOLw)) % n != 0]
    T = T[np.arange(len(T)) % n != 0]
    K = K[np.arange(len(K)) % n != 0]
    U = U[np.arange(len(U)) % n != 0]
    V = V[np.arange(len(V)) % n != 0]

    def min_max_scaling(arr, min_val, max_val):
        scaled_arr = (arr - min_val) / (max_val - min_val)
        return scaled_arr

    T = min_max_scaling(T, 0, 50)
    K = min_max_scaling(K, 0, 1)
    U = min_max_scaling(U, -0.1, 0.81)
    V = min_max_scaling(V, 0, 0.12)
    SOLs = min_max_scaling(SOLs, -0.1, 0.81)
    SOLw = min_max_scaling(SOLw, 0, 0.12)

    print("Generated set")

    np.save(data_folder + "T.npy", T)
    np.save(data_folder + "K.npy", K)
    np.save(data_folder + "U.npy", U)
    np.save(data_folder + "V.npy", V)
    np.save(data_folder + "SOLs.npy", SOLs)
    np.save(data_folder + "SOLw.npy", SOLw)
    nc = num_samples
    pred_u = np.reshape(SOLs, (nc, len(t) - 1))
    true_u = np.reshape(SOLs, (nc, len(t) - 1))

x_values = []
y_values = []
z_values = []

def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val

with open("data.txt", "r") as file:
    for line in file:
        U, V, K = line.strip().split(",")
        K = denormalize(float(K), 0, 1)
        U = denormalize(float(U), -0.1, 0.81)
        V = denormalize(float(V), 0.0, 0.12)
        print(U, V, K)
        x_values.append(float(U))
        y_values.append(float(V))
        z_values.append(float(K))

print(y_values)

sample_set = np.stack((x_values, y_values, z_values), axis=0)
print(np.shape(sample_set))
print(sample_set)
sample_set = np.tile(sample_set, (1, int(1e3)))

print(np.shape(sample_set))
epsilon = 0.001
perturbation = np.random.uniform(1 - epsilon, 1 + epsilon, sample_set.shape)
sample_set = sample_set * perturbation
print(sample_set)
print("Generating training data for region")
getDatasetFromDistA(sample_set, data_folder="../training_data/treinocr2/", ti=0, tf=50)
