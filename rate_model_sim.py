from functions import f_function, forward_euler, cont_rect_pulses

import numpy as np
import matplotlib.pyplot as plt


def run_sim(i_t, dt, steps):
    ############################################################
    # Init
    ############################################################

    # neuron constants -> [exc, pv, sst]
    thresholds = [0.7, 1.0, 1.0]
    tau = [10 * 1e-3, 10 * 1e-3, 10 * 1e-3]  # s
    i_opt = [0.0, -2.0, -1.0]
    thal_flag = np.array([1, 1, 0])

    # init firing rates
    n_subtypes = len(tau)
    f_rates = np.zeros((steps + 1, n_subtypes))
    f_rates[0, :] = np.random.rand(n_subtypes)

    weights = [[1.1, -2, -1], [1, -2, -2], [6, -0, -0]]  # [[post_exc], [post_pv], [post_sst]]

    # thalamic input
    q = 5
    tau_d1 = 1500 * 1e-3  # s
    tau_d2 = 20 * 1e-3  # s
    g_0 = 1
    thal_input = np.zeros(steps + 1)
    thal_input[0] = g_0

    #########################################################################################
    # Simulation
    #########################################################################################

    for i in range(steps):
        # Update thalamic input
        dg_dt = (g_0 - thal_input[i]) / tau_d1 - thal_input[i] * i_t[i] / tau_d2
        thal_input[i + 1] = forward_euler(dg_dt, thal_input[i], dt)

        # Update Wilson-Cowan model
        f_arg = np.sum(weights * f_rates[i, :], axis=1) + i_opt + thal_flag * q * thal_input[i + 1] * i_t[i]
        d_dt = (-f_rates[i, :] + f_function(f_arg - thresholds)) / tau
        f_rates[i + 1, :] = forward_euler(d_dt, f_rates[i, :], dt)  # update firing rates

    return f_rates, thal_input


def exe_wilson_cowan():
    dt = 0.1 * 1e-3  # s
    t_ges = 1000 * 1e-3  # s
    steps = int(t_ges / dt)

    # input stimulus
    stim_dur = 100 * 1e-3
    inter_stim_dur = 300 * 1e-3
    i_t = cont_rect_pulses(stim_dur, inter_stim_dur, steps, dt)  # TODO: Amplitude?

    [f_rates, thal_input] = run_sim(i_t, dt, steps)

    for i in range(f_rates.shape[1]):
        plt.plot(np.arange(0, t_ges + dt, dt), f_rates[:, i])

    plt.legend(['exc', 'pv', 'sst'])
    plt.title("Firing rates")
    plt.xlabel("t / s")

    plt.figure()

    plt.plot(np.arange(0, t_ges + dt, dt), thal_input)
    plt.title("Thalamic input")
    plt.xlabel("t / s")

    plt.show()
