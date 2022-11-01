from simulation import run_sim

import numpy as np
import matplotlib.pyplot as plt

n_neurons = np.array([20, 10])
n_ext = np.array([8, 4])
dt = 0.01
t_ges = 1
[v_m, spikes, ext_input_neurons] = run_sim(n_neurons, n_ext, dt, t_ges)

print(sum(spikes))

plt.plot(v_m[:, np.where(ext_input_neurons)[0][0]])
plt.plot(spikes[:, np.where(ext_input_neurons)[0][0]])
plt.show()
