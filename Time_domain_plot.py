from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime

temperature = np.loadtxt("WeatherJena.csv", delimiter=',', skiprows=1, usecols=2)
humidity = np.loadtxt("WeatherJena.csv", delimiter=',', skiprows=1, usecols=5)
time_raw = np.loadtxt("WeatherJena.csv", dtype=str, delimiter=',', skiprows=1, usecols=0)
time = [datetime.strptime(t, '%d.%m.%Y %H:%M:%S') for t in time_raw]

plt.figure()
plt.rcParams.update({'font.size': 14})
plt.rcParams['lines.linewidth'] = 3
plt.plot(time, temperature)
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.savefig("Figures/Temperature.pdf", bbox_inches='tight')

plt.figure()
plt.rcParams.update({'font.size': 14})
plt.rcParams['lines.linewidth'] = 3
plt.plot(time, humidity)
plt.xlabel('Time')
plt.ylabel('Relative humidity (%)')
plt.savefig("Figures/Humidity.pdf", bbox_inches='tight')
