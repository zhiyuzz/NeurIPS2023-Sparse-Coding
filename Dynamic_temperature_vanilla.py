from matplotlib import pyplot as plt
from datetime import datetime
from Static_algorithm import *

##########
# Read from the dataset
##########

time_raw = np.loadtxt("WeatherJena.csv", dtype=str, delimiter=',', skiprows=1, usecols=0)
time = [datetime.strptime(t, '%d.%m.%Y %H:%M:%S') for t in time_raw]

temperature = np.loadtxt("WeatherJena.csv", delimiter=',', skiprows=1, usecols=2)

##########
# Define the Fourier dictionary
##########

omega = 2 * np.pi / 24 / 60 / 60    # base frequency
K_max = 10
K_class = range(K_max)  # different number of harmonics

##########
# Shared settings of the problem
##########

G = 1   # Lipschitz constant
T = 50000   # Length of the studied segment
temperature_subset = temperature[:T]

##########
# The loop over settings with different K
##########

total_loss = np.zeros(len(K_class))
for ind_K in range(len(K_class)):

    K = K_class[ind_K]

    ##########
    # Define 2K+1 static algorithms for the 2K+1 features
    ##########

    eps = 1 / (2 * K + 1)  # prior of the algorithm
    algs = []
    for ind in range(2 * K + 1):
        algs.append(StaticOneD(eps, G))

    ##########
    # The main loop
    ##########

    predictions = np.zeros(T)
    loss = 0

    for t in range(T):

        # Define the features
        ht = np.zeros(2 * K + 1)
        ht[0] = 1
        time_diff = time[t] - time[0]
        for k in range(K):
            ht[2 * k + 1] = np.cos((k + 1) * omega * time_diff.seconds)
            ht[2 * k + 2] = np.sin((k + 1) * omega * time_diff.seconds)

        # Send Lipschitz constants to the static subroutines, and query their predictions
        predictions_subroutines = np.zeros(2 * K + 1)
        for ind in range(2 * K + 1):
            predictions_subroutines[ind] = algs[ind].get_prediction() * ht[ind]
        if t > 0:
            predictions[t] = np.sum(predictions_subroutines)

        # Calculate the gradient
        loss += np.abs(predictions[t] - temperature_subset[t])
        if predictions[t] >= temperature_subset[t]:
            gt = 1
        else:
            gt = -1

        # Update the static subroutines
        for ind in range(2 * K + 1):
            algs[ind].update(gt * ht[ind])

    total_loss[ind_K] = loss

plt.figure()
plt.rcParams.update({'font.size': 14})
plt.rcParams['lines.linewidth'] = 3
plt.plot(np.arange(K_max, dtype=int) * 2 + 1, total_loss)
plt.xlabel('Number of features')
plt.xticks(range(1, 2 * K_max, 2))
plt.ylabel('Total loss')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.savefig("Figures/Dynamic_vanilla.pdf", bbox_inches='tight')
