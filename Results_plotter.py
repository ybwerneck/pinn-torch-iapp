import h5py
import matplotlib.pyplot as plt
import numpy as np
import operator

def plot_results(file_path):
    with h5py.File(file_path + "/val.h5", 'r') as hf:
        target = hf['target'][:]
        pred = hf['pred'][:]
            
    with h5py.File(file_path + "/val_err.h5", 'r') as hf:
        err = hf['error_stats'][:]

    window_size = 500
    absolute_error = np.abs(target - pred)
    max_error_index = np.argmax(absolute_error.T[0])
    print(max_error_index)
    print(err)

    start_index = max(0, max_error_index - window_size)
    end_index = start_index + window_size * 2

    # Plot error
    print("Val Error")
    plt.plot(err)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.yscale('log')
    plt.ylim(1e-4, 1e1)
    plt.savefig(dir+'/val_error.png')
    plt.clf()
        
    # Plot target and prediction for the first window size
    print("Last Validation Set")
    plt.plot(target[:window_size, 0])
    plt.plot(pred[:window_size, 0])
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend(['Target', 'Prediction'])
    plt.savefig(dir+'/last_validation_set.png')
    plt.clf()

    # Plot absolute error in the region around the maximum error
    print(np.mean(absolute_error))
    plt.plot(absolute_error[start_index:end_index])
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend(['Errw', 'Errv'])
    plt.savefig(dir+'/absolute_error.png')
    plt.clf()
  
    # Plot target and prediction in the region around the maximum error
    plt.plot(target[start_index:end_index, 0], label='Target')
    plt.plot(pred[start_index:end_index, 0], label='Prediction')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(dir+'/target_prediction_max_error.png')
    plt.clf()
    
    # Scatter plot of target vs prediction
    plt.scatter(target[:3000], pred[:3000])
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.legend(['Target', 'Prediction'])
    plt.savefig(dir+'/target_vs_prediction.png')
    plt.clf()

# Call the plot_results function with the path to your HDF5 file
import sys

file_path = "batch_results/simulation0"
file_path=sys.argv[1]

dir=file_path
plot_results(file_path)

colors = ["red", "blue", "green", "orange", "red", "blue", "green", "yellow", "orange"]

with h5py.File(file_path + "/val.h5", 'r') as hf:
    T, U, V, K = np.array(hf['input']).T
    TX, TW = np.array(hf['target']).T
    X, W = np.array(hf['pred']).T

E = ((X - TX)**2)**0.5
ae, m = max(enumerate(E), key=operator.itemgetter(1))
print("max error region: ", K[ae], U[ae], V[ae], T[ae])
print("max err", np.max(((X - TX)**2)**0.5))
print("mean err", (np.mean(X - TX)**2)**0.5)
print(np.unique(T))
e = lambda x: np.expand_dims(x, axis=1)

nc = len(U) // len(np.unique(T))
pred_u = np.reshape(X, (nc, len(X) // nc))
true_u = np.reshape(TX, (nc, len(X) // nc))
ks, us, vs = np.reshape(K, (nc, len(X) // nc)), np.reshape(U, (nc, len(X) // nc)), np.reshape(V, (nc, len(X) // nc))

k = 3

N = 0

max_error = float('-inf')
max_error_solution = None

for N in range(len(ks)):
    current_error = np.max(np.abs(pred_u[N] - true_u[N]))
    if current_error > max_error:
        max_error = current_error
        max_error_solution = N

    N = max_error_solution


u_k = np.unique(ks)
u_v = np.unique(vs)
u_u = np.unique(us)

Ns = [0, 1, 2, 3, 4, max_error_solution]
plt.figure(figsize=(20, 10))
for i, N in enumerate(Ns):
    print("IAPP", ks[N][1])
    print("W0", vs[N][1])
    print("U0", us[N][1])
    plt.plot(T[0:len(X) // nc], pred_u[N], label="W0" + str(us[N][1]), color=colors[i])
    plt.plot(T[0:len(X) // nc], true_u[N], "--", label="W0_T" + str(us[N][1]), color=colors[i])

plt.legend(loc="best")
plt.savefig(dir+'/plot_selected_Ns.png')
plt.clf()

Ns = [0, 163, 352]
data = pred_u[Ns, :]
datat = true_u[Ns, :]
datae = np.abs(true_u[Ns, :] - pred_u[Ns, :])

x = np.linspace(0, T[-1], np.shape(data)[1])
y = np.linspace(0, len(Ns), np.shape(data)[0])
X, Y = np.meshgrid(x, y)

fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})

surfs = []
data_arrays = [datat, data, datae]
plot_titles = ['True', 'Pred', 'Error']
mi, ma = 0, 1
for i, ax in enumerate(axs):
    if i == 2:
        ax.set_zlim(0, 0.1)
        mi, ma = 0, 0.1

    surf = ax.plot_surface(X, Y, data_arrays[i], cmap='viridis', vmin=mi, vmax=ma)
    surfs.append(surf)
    ax.set_title(plot_titles[i])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Value')
    fig.colorbar(surf, ax=ax)

plt.tight_layout()
plt.savefig(dir+'/3d_plots.png')
plt.clf()

Ns = range(0, nc)
data = pred_u[Ns, :]
datat = true_u[Ns, :]
datae = np.abs(true_u[Ns, :] - pred_u[Ns, :])

x = np.linspace(0, T[-1], np.shape(data)[1])
y = np.linspace(0, len(Ns), np.shape(data)[0])
X, Y = np.meshgrid(x, y)

fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})

surfs = []
data_arrays = [datat, data, datae]
plot_titles = ['True', 'Pred', 'Error']
mi, ma = 0, 1
for i, ax in enumerate(axs):
    if i == 2:
        ax.set_zlim(0, 0.1)
        mi, ma = 0, 0.1

    surf = ax.plot_surface(X, Y, data_arrays[i], cmap='viridis', vmin=mi, vmax=ma)
    surfs.append(surf)
    ax.set_title(plot_titles[i])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Value')
    fig.colorbar(surf, ax=ax)

plt.tight_layout()
plt.savefig(dir+'/3d_plots_all.png')
plt.clf()

print("abc")
