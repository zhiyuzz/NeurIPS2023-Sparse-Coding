from Algorithm_static_subroutine import *
from Algorithm_baseline import *


##########
# Problem settings shared across different random seeds
##########

K = 15
T = 2 ** K
G = 1

##########
# Loop over random seeds
##########

loss_ours = 0
loss_baseline = 0

for seed in range(2023, 2013, -1):

    ##########
    # Generate the data
    ##########

    rng = np.random.default_rng(seed)
    signal = np.empty(T)
    temp = 1
    for i in range(T):
        temp = temp * np.sign(rng.random()-0.0005) + 0.01 * (rng.random()-0.5)
        signal[i] = temp

    ##########
    # Initialize our algorithm
    ##########

    eps_ours = 1 / T  # prior of the algorithm
    subroutines = []
    for k in range(K + 1):
        subroutines.append(StaticOneD(eps_ours, G))

    Tau = np.zeros(K)   # length of the local time interval
    for k in range(K):
        Tau[k] = 2 ** (k + 1)
    tau = np.zeros(K)   # Track the elapsed time during local timer intervals

    ##########
    # Initialize the baselines
    ##########

    baseline = DynamicBaseline(1, G, T)

    ##########
    # The main loop
    ##########

    prediction_ours = 0
    prediction_baseline = 0

    for t in range(T):

        # For our algorithm, reinitialize the subroutines, if required; and define the features
        ht = np.zeros(K + 1)
        ht[-1] = 1
        for k in range(K):
            if t % Tau[k] == 0:
                subroutines[k] = StaticOneD(eps_ours, G)
                tau[k] = 0
            if 2 * tau[k] < Tau[k]:
                ht[k] = 1
            else:
                ht[k] = -1

        # Query the predictions of the static subroutines
        predictions_subroutines = np.zeros(K + 1)
        for k in range(K + 1):
            predictions_subroutines[k] = subroutines[k].get_prediction() * ht[k]
        if t > 0:
            prediction_ours = np.sum(predictions_subroutines)

        # Calculate the gradient
        loss_ours += np.abs(prediction_ours - signal[t])
        if prediction_ours >= signal[t]:
            gt_ours = 1
        else:
            gt_ours = -1

        # Update the static subroutines
        for k in range(K + 1):
            subroutines[k].update(gt_ours * ht[k])

        for k in range(K):
            tau[k] += 1

        # Query the baseline
        prediction_baseline = baseline.get_prediction()

        # Calculate the gradient
        loss_baseline += np.abs(prediction_baseline - signal[t])
        if prediction_baseline >= signal[t]:
            gt_baseline = 1
        else:
            gt_baseline = -1

        baseline.update(gt_baseline)
