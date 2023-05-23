from datetime import datetime
from Algorithm_baseline import *

##########
# Read from the dataset
##########

time_raw = np.loadtxt("WeatherJena.csv", dtype=str, delimiter=',', skiprows=1, usecols=0)
time = [datetime.strptime(t, '%d.%m.%Y %H:%M:%S') for t in time_raw]

temperature = np.loadtxt("WeatherJena.csv", delimiter=',', skiprows=1, usecols=2)

##########
# Setting
##########

G = 1   # Lipschitz constant
T = 50000   # Length of the studied segment
temperature_subset = temperature[:T]
eps = 1

alg = DynamicBaseline(eps, G, T)
predictions = np.zeros(T)
loss = 0

for t in range(T):

    predictions[t] = alg.get_prediction() + temperature_subset[t - 1]

    loss += np.abs(predictions[t] - temperature_subset[t])
    if predictions[t] >= temperature_subset[t]:
        gt = 1
    else:
        gt = -1

    alg.update(gt)
