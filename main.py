from simulation import run_sim

import numpy as np
import matplotlib.pyplot as plt

test_factor = 1

n_neurons = np.array([100 * test_factor, 400 * test_factor])  # ratio 1:4
n_ext = np.array([15 * test_factor, 20 * test_factor])
dt = 0.1 * 1e-3  # 0.1 ms
t_ges = 100 * 1e-3  # 100 ms
[v_m, spikes, ext_input_neurons] = run_sim(n_neurons, n_ext, dt, t_ges)

print("Spikes total:\n", sum(spikes))
print("External input neurons:\n", ext_input_neurons.astype(int))

other_neurons = np.invert(ext_input_neurons)

# for i in range(sum(ext_input_neurons)):
#     plt.plot(v_m[:, np.where(ext_input_neurons)[0][i]])

plt.plot(v_m[:, np.where(ext_input_neurons)[0][0]])
plt.plot(v_m[:, np.where(ext_input_neurons)[0][-1]])

# for i in range(sum(other_neurons)):
#     plt.plot(v_m[:, np.where(other_neurons)[0][i]])

plt.plot(v_m[:, np.where(other_neurons)[0][0]])
plt.plot(v_m[:, np.where(other_neurons)[0][-1]])

plt.legend(['inh_stim', 'exc_stim', 'inh_nostim', 'exc_nostim'])
plt.title('Membrane potential')

plt.figure()

# for i in range(sum(ext_input_neurons)):
#     plt.plot(spikes[:, np.where(ext_input_neurons)[0][i]])

plt.plot(spikes[:, np.where(ext_input_neurons)[0][0]])
plt.plot(spikes[:, np.where(ext_input_neurons)[0][-1]])

# for i in range(sum(other_neurons)):
#     plt.plot(spikes[:, np.where(other_neurons)[0][i]])

plt.plot(spikes[:, np.where(other_neurons)[0][0]])
plt.plot(spikes[:, np.where(other_neurons)[0][-1]])

# plt.figure()
# for i in range(n_ext[0]):
#     plt.plot(v_m[:, np.where(ext_input_neurons)[0][i]])
plt.legend(['inh_stim', 'exc_stim', 'inh_nostim', 'exc_nostim'])
plt.title('Spikes')

plt.show()
