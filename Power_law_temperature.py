from matplotlib import pyplot as plt
import numpy as np

##########
# Read from the dataset
##########

temperature = np.loadtxt("WeatherJena.csv", delimiter=',', skiprows=1, usecols=2)

##########
# Validate the power law on the temperature sequence
##########

Fourier = np.fft.rfft(temperature)
Fourier_abs = np.abs(Fourier)
sorted_Fourier = - np.sort(-Fourier_abs)

# We fit a linear model on the log-log scale, using the largest 100 Fourier coefficients
useful_indices = np.arange(100) + 1
useful_Fourier_coeff = sorted_Fourier[:100]
[k1, k0] = np.polyfit(np.log10(useful_indices), np.log10(useful_Fourier_coeff), 1)

# Plot the magnitude of Fourier coefficients (on the log-log scale), together with the fitted line
transform_length = len(Fourier)
all_indices = np.arange(transform_length) + 1
fitted_Fourier = 10 ** (k1 * np.log10(all_indices) + k0)

plt.figure()
plt.rcParams.update({'font.size': 14})
plt.rcParams['lines.linewidth'] = 3
plt.loglog(all_indices, sorted_Fourier, '-', label='Data')
plt.loglog(all_indices, fitted_Fourier, '--', label=r'Fitted, $\alpha$='+"{:.2f}".format(-k1))
plt.legend()
plt.xlabel('Indices (log)')
plt.ylabel('Fourier coefficients (abs, log)')
plt.savefig("Figures/PL_temperature.pdf", bbox_inches='tight')

##########
# Validate the power law on the difference of the temperature sequence
##########

Fourier_diff = np.fft.rfft(np.diff(temperature))
Fourier_diff_abs = np.abs(Fourier_diff)
sorted_Fourier_diff = - np.sort(-Fourier_diff_abs)

# Similar to the above, we fit a linear model on the log-log scale, using the largest 100 coefficients
useful_Fourier_diff = sorted_Fourier_diff[:100]
[k1_diff, k0_diff] = np.polyfit(np.log10(useful_indices), np.log10(useful_Fourier_diff), 1)

# Plot the magnitude of Fourier coefficients (on the log-log scale), together with the fitted line
transform_length_diff = len(Fourier_diff)
all_indices_diff = np.arange(transform_length_diff) + 1
fitted_Fourier_diff = 10 ** (k1_diff * np.log10(all_indices_diff) + k0_diff)

plt.figure()
plt.rcParams.update({'font.size': 14})
plt.rcParams['lines.linewidth'] = 3
plt.loglog(all_indices_diff, sorted_Fourier_diff, '-', label='Data')
plt.loglog(all_indices_diff, fitted_Fourier_diff, '--', label=r'Fitted, $\alpha$='+"{:.2f}".format(-k1_diff))
plt.legend()
plt.xlabel('Indices (log)')
plt.ylabel('Fourier coefficients of diff (abs, log)')
plt.savefig("Figures/PL_temperature_diff.pdf", bbox_inches='tight')
