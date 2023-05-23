from matplotlib import pyplot as plt
from datetime import datetime
from Dynamic_baseline_algorithm import *

##########
# Read from the dataset
##########

time_raw = np.loadtxt("WeatherJena.csv", dtype=str, delimiter=',', skiprows=1, usecols=0)
time = [datetime.strptime(t, '%d.%m.%Y %H:%M:%S') for t in time_raw]

temperature = np.loadtxt("WeatherJena.csv", delimiter=',', skiprows=1, usecols=2)

##########
# Setting of the problem
##########

G = 1   # Lipschitz constant
T = 50000   # Length of the studied segment
temperature_subset = temperature[:T]

##########
# The loop over settings with different eps
##########

eps_class = np.array([0.01, 0.1, 1, 10, 100])
total_loss = np.zeros(len(eps_class))
for ind_eps in range(len(eps_class)):

    eps = eps_class[ind_eps]
    alg = DynamicBaseline(eps, G, T)

    predictions = np.zeros(T)
    loss = 0

    for t in range(T):

        predictions[t] = alg.get_prediction()

        loss += np.abs(predictions[t] - temperature_subset[t])
        if predictions[t] >= temperature_subset[t]:
            gt = 1
        else:
            gt = -1

        alg.update(gt)

    total_loss[ind_eps] = loss

plt.figure()
plt.rcParams.update({'font.size': 14})
plt.rcParams['lines.linewidth'] = 3
plt.plot(eps_class, total_loss)
plt.xlabel(r'$\epsilon$')
plt.xscale('log')
plt.ylabel('Total loss')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.savefig("Figures/Dynamic_baseline.pdf", bbox_inches='tight')
