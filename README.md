# Code for the paper "Unconstrained Dynamic Regret via Sparse Coding"

Paper available at https://arxiv.org/abs/2301.13349

Python version 3.8

Requires numpy, matplotlib, datetime, pywt

The weather data is contained in WeatherJena.csv, retrived from the Jena Weather project. https://www.bgc-jena.mpg.de/wetter/

Instruction: 

- Algorithm_static_subroutine.py and Algorithm_baseline.py implement the algorithms we test.

- Power_law_wavelet.py produces Figure 3 (with different random seeds).

- Time_domain_plot.py produces Figure 4.

- Power_law_temperature.py produces Figure 5 (including Figure 2).

- Power_law_humidity.py produces Figure 6.

- Dynamic_wavelet.py performs the "wavelet dictionary experiment" in Appendix E.2.

- Dynamic_temperature_ours.py produces Figure 7. Dynamic_temperature_baseline.py tests the performance of the baseline ([JC22]) for the "Fourier dictionary experiment" in Appendix E.2.
