from functions import f_function, forward_euler, cont_rect_pulses

import numpy as np
import matplotlib.pyplot as plt


def run_sim(i_t, vip_in, f_flag, d_flag, dt, steps):
    ############################################################
    # Init
    ############################################################

    # neuron constants -> [exc, pv, sst, vip]
    thresholds = [0.7, 1.0, 1.0, 0.0]
    tau = [10 * 1e-3, 10 * 1e-3, 10 * 1e-3, 10 * 1e-3]  # s
    i_opt = [0.0, -0.0, -0.0, 0.0]  # [0.0, -2.0, -1.0, 0.0]

    thal_flag = np.array([1, 1, 0, 0])
    vip_flag = np.array([0, 0, 0, 1])
    weights = [[1.1, -2, -1, -0], [1, -2, -2, -0], [6, -0, -0, -3], [0, -0, -3, -3]]
    # [[post_exc], [post_pv], [post_sst], [post_vip]]

    # init firing rates
    n_subtypes = len(tau)
    f_rates = np.zeros((steps + 1, n_subtypes))
    # f_rates[0, :] = np.random.rand(n_subtypes)

    # Depression and Facilitation constants - post exc.: Park2020; post inh. types: Campagnola2022
    D = 1.0
    a_dep = np.array([[-0.0, 0.5, 0.0, 0.0], [-0, 0.5, 0.11, 0.13], [-0, 0.35, 0.18, 0], [-0, 0.37, 0, 0]])

    F = 0.0
    a_fac = np.array([[0.0, -0.0, -1.0, -0.0], [0, -0, -0, -0], [0, -0, -0, -0.05], [0, -0, -0.28, -0.04]])

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

        # if i == 4110:  # debug
        #     deb = 1

        # Update depression and facilitation terms
        dD_dt = (1 - D) / tau_d1 - D * i_t[i] / tau_d2
        dF_dt = - F / tau_d1 + i_t[i] / tau_d2
        D = forward_euler(dD_dt, D, dt)
        F = forward_euler(dF_dt, F, dt)

        # Update Wilson-Cowan model
        f_arg = np.sum((weights + d_flag * a_dep * (1 - D) + f_flag * a_fac * F) * f_rates[i, :],
                       axis=1) + i_opt + thal_flag * q * thal_input[i + 1] * i_t[i] + vip_flag * vip_in[i]
        d_dt = (-f_rates[i, :] + f_function(f_arg - thresholds)) / tau
        f_rates[i + 1, :] = forward_euler(d_dt, f_rates[i, :], dt)  # update firing rates

    return f_rates, thal_input


def exe_wilson_cowan():
    dt = 0.1 * 1e-3  # s
    t_ges = 1000 * 1e-3  # s
    steps = int(t_ges / dt)

    # Switch on/off arbitrary no of facilitation and depression terms
    # d_flag = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    # f_flag = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    d_flag = np.ones((4, 4))
    f_flag = np.ones((4, 4))

    # input stimulus
    stim_dur = 100 * 1e-3
    inter_stim_dur = 300 * 1e-3
    i_t = cont_rect_pulses(stim_dur, inter_stim_dur, steps, dt)
    # i_t = np.ones(steps)

    # Higher order input
    stim_dur = 600 * 1e-3
    inter_stim_dur = 400 * 1e-3
    magnitude = 3.0
    vip_in = magnitude * cont_rect_pulses(stim_dur, inter_stim_dur, steps, dt)

    [f_rates_plas, thal_input] = run_sim(i_t, vip_in, f_flag, d_flag, dt, steps)

    d_flag = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    f_flag = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    [f_rates_noplas, _] = run_sim(i_t, vip_in, f_flag, d_flag, dt, steps)

    time = np.arange(0, t_ges + dt, dt)

    for i in range(f_rates_plas.shape[1]):
        plt.plot(time, f_rates_plas[:, i])

    plt.legend(['exc', 'pv', 'sst', 'vip'])
    plt.title("Firing rates")
    plt.xlabel("t / s")

    plt.figure()

    plt.plot(time, f_rates_plas[:, 0])
    plt.plot(time, f_rates_noplas[:, 0])

    plt.legend(['Plasticity', 'No Plasticity'])
    plt.title("Excitatory Firing rates")
    plt.xlabel("t / s")

    # plt.figure()
    #
    # plt.plot(time, thal_input)
    # plt.title("Thalamic input")
    # plt.xlabel("t / s")

    plt.figure()

    time = np.arange(0, t_ges, dt)

    plt.plot(time, i_t)
    plt.plot(time, vip_in)

    plt.legend(['Stimulus', 'Higher Order'])
    plt.title("Input signals")
    plt.xlabel("t / s")

    plt.show()
