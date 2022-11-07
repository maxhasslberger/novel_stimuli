from functions import f_function, forward_euler, cont_rect_pulses

import numpy as np
import matplotlib.pyplot as plt


def run_sim(i_t, d_flag, dt, steps):
    ############################################################
    # Init
    ############################################################

    # neuron constants -> [exc, pv, sst]
    thresholds = [0.7, 1.0, 1.0, 0.0]
    tau = [10 * 1e-3, 10 * 1e-3, 10 * 1e-3, 10 * 1e-3]  # s
    i_opt = [0.0, -2.0, -1.0, 0.0]

    thal_flag = np.array([1, 1, 0, 0])
    weights = [[1.1, -2, -1, -0.0], [1, -2, -2, -0.0], [6, -0, -0, -0.0], [0, -0, -0, -0.0]]
    # [[post_exc], [post_pv], [post_sst]]

    # init firing rates
    n_subtypes = len(tau)
    f_rates = np.zeros((steps + 1, n_subtypes))
    f_rates[0, :] = np.random.rand(n_subtypes)

    # Depression constants
    D = 1.0
    a = 0.5

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

        # if np.mod(i, 500) == 0:
        #     deb = 1

        # Update depression term
        dD_dt = (1 - D) / tau_d1 - D * i_t[i] / tau_d2
        D = forward_euler(dD_dt, D, dt)

        # Update Wilson-Cowan model
        f_arg = np.sum((weights + d_flag * a * (1 - D)) * f_rates[i, :],
                       axis=1) + i_opt + thal_flag * q * thal_input[i + 1] * i_t[i]
        d_dt = (-f_rates[i, :] + f_function(f_arg - thresholds)) / tau
        f_rates[i + 1, :] = forward_euler(d_dt, f_rates[i, :], dt)  # update firing rates

    return f_rates, thal_input


def exe_wilson_cowan():
    dt = 0.1 * 1e-3  # s
    t_ges = 1000 * 1e-3  # s
    steps = int(t_ges / dt)

    d_flag = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    # input stimulus
    stim_dur = 100 * 1e-3
    inter_stim_dur = 300 * 1e-3
    i_t = cont_rect_pulses(stim_dur, inter_stim_dur, steps, dt)  # TODO: Amplitude?

    [f_rates, thal_input] = run_sim(i_t, d_flag, dt, steps)
    d_flag = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    [f_rates2, _] = run_sim(i_t, d_flag, dt, steps)

    for i in range(f_rates.shape[1]):
        plt.plot(np.arange(0, t_ges + dt, dt), f_rates[:, i])

    plt.legend(['exc', 'pv', 'sst', 'vip'])
    plt.title("Firing rates")
    plt.xlabel("t / s")

    plt.figure()

    plt.plot(np.arange(0, t_ges + dt, dt), f_rates[:, 0])
    plt.plot(np.arange(0, t_ges + dt, dt), f_rates2[:, 0])

    plt.legend(['No Depression', 'Depression'])
    plt.title("Excitatory Firing rates")
    plt.xlabel("t / s")

    plt.figure()

    plt.plot(np.arange(0, t_ges + dt, dt), thal_input)
    plt.title("Thalamic input")
    plt.xlabel("t / s")

    plt.figure()

    plt.plot(np.arange(0, t_ges, dt), i_t)
    plt.title("Input current")
    plt.xlabel("t / s")

    plt.show()
