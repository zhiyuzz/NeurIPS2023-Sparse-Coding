from matplotlib import pyplot as plt
import pywt
import numpy as np

##########
# Generate a random low switching signal
##########

rng = np.random.default_rng(2020)
T = 2**15
signal = np.empty(T)
temp = 1
for i in range(T):
    temp = temp * np.sign(rng.random()-0.0005) + 0.01 * (rng.random()-0.5)
    signal[i] = temp

##########
# Take the Discrete Haar Wavelet Transform
##########

coeffs = pywt.wavedec(signal, 'haar')
flat_coeffs = np.array([item for sublist in coeffs for item in sublist])
sorted_coeffs = -np.sort(-np.abs(flat_coeffs))

# We fit a linear model on the log-log scale, using the largest 100 wavelet coefficients
useful_indices = np.arange(100) + 1
useful_wavelet_coeff = sorted_coeffs[:100]
[k1, k0] = np.polyfit(np.log10(useful_indices), np.log10(useful_wavelet_coeff), 1)

##########
# Plot the results
##########

# The time domain signal
plt.figure()
plt.rcParams.update({'font.size': 14})
plt.rcParams['lines.linewidth'] = 3
plt.plot(signal)
plt.xlabel('Time')
plt.ylabel('Signal')
plt.savefig("Figures/PL_wavelet_time_domain_2020.pdf", bbox_inches='tight')

# The wavelet coefficients and the linear fit
transform_length = len(sorted_coeffs)
all_indices = np.arange(transform_length) + 1
fitted_wavelet = 10 ** (k1 * np.log10(all_indices) + k0)

plt.figure()
plt.rcParams.update({'font.size': 14})
plt.rcParams['lines.linewidth'] = 3
plt.loglog(all_indices, sorted_coeffs, '-', label='Data')
plt.loglog(all_indices, fitted_wavelet, '--', label=r'Fitted, $\alpha$='+"{:.2f}".format(-k1))
plt.legend()
plt.xlabel('Indices (log)')
plt.ylabel('Wavelet coefficients (abs, log)')
plt.savefig("Figures/PL_wavelet_tran_domain_2020.pdf", bbox_inches='tight')
