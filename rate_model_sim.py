from functions import f_function, forward_euler, cont_pulse_trials

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
    weights = [[1.1, -2, -1, -0], [1, -2, -2, -0], [6, -0, -0, -3], [0, -1.5, -3, -3]]
    # [[post_exc], [post_pv], [post_sst], [post_vip]]

    # init firing rates
    n_subtypes = len(tau)
    f_rates = np.zeros((steps + 1, n_subtypes))
    # f_rates[0, :] = np.random.rand(n_subtypes)

    # Depression and Facilitation constants - post exc.: Park2020; post inh. types: Campagnola2022
    D = np.zeros(steps + 1)
    D[0] = 1.0
    a_dep = np.array([[-0.0, 0.5, 0.0, 0.0], [-0, 0.5, 0.11, 0.13], [-0, 0.35, 0.18, 0], [-0, 0.37, 0, 0]])

    F = np.zeros(steps + 1)
    F[0] = 0.0
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
    deb = 0
    for i in range(steps):
        # Update thalamic input
        dg_dt = (g_0 - thal_input[i]) / tau_d1 - thal_input[i] * i_t[i] / tau_d2
        thal_input[i + 1] = forward_euler(dg_dt, thal_input[i], dt)

        if f_rates[i, 0] >= 0.7:  # debug
            if deb != i - 1:
                deb = 0
            deb = i

        # Update depression and facilitation terms
        dD_dt = (1 - D[i]) / tau_d1 - D[i] * i_t[i] / tau_d2
        dF_dt = - F[i] / tau_d1 + i_t[i] / tau_d2
        D[i+1] = forward_euler(dD_dt, D[i], dt)
        F[i+1] = forward_euler(dF_dt, F[i], dt)

        # Update Wilson-Cowan model
        f_arg = np.sum((weights + d_flag * a_dep * (1 - D[i+1]) + f_flag * a_fac * F[i+1]) * f_rates[i, :],
                       axis=1) + i_opt + thal_flag * q * thal_input[i + 1] * i_t[i] + vip_flag * vip_in[i]
        d_dt = (-f_rates[i, :] + f_function(f_arg - thresholds)) / tau
        f_rates[i + 1, :] = forward_euler(d_dt, f_rates[i, :], dt)  # update firing rates

    return f_rates, thal_input, F, D


def exe_wilson_cowan():
    dt = 0.1 * 1e-3  # s
    t_ges = 10000 * 1e-3  # s
    steps = int(t_ges / dt)

    # Switch on/off arbitrary no of facilitation and depression terms
    d_flag = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]])
    d_flag = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    d_flag = np.zeros((4, 4))
    d_flag = np.ones((4, 4))
    f_flag = d_flag

    # input stimulus
    stim_dur = 100 * 1e-3
    inter_stim_dur = 300 * 1e-3
    inter_trial_dur = 2400 * 1e-3
    trial_pulses = 8
    i_t = cont_pulse_trials(stim_dur, inter_stim_dur, inter_trial_dur, trial_pulses, steps, dt)
    # i_t = np.ones(steps)

    # Higher order input
    # stim_dur = 5000 * 1e-3
    # inter_stim_dur = 5000 * 1e-3
    magnitude = 3.0
    vip_in = magnitude * cont_pulse_trials(stim_dur, inter_stim_dur, inter_trial_dur, trial_pulses, steps, dt)

    [f_rates, thal_input, F, D] = run_sim(i_t, vip_in, f_flag, d_flag, dt, steps)

    d_flag = np.ones((4, 4))
    d_flag = np.zeros((4, 4))
    f_flag = d_flag

    [f_rates2, _, _, _] = run_sim(i_t, vip_in, f_flag, d_flag, dt, steps)

    time = np.arange(0, t_ges + dt, dt)
    scatter = ['o', 'o', '.', '--']

    for i in range(f_rates.shape[1]):
        plt.plot(time, f_rates[:, i], scatter[i])

    plt.legend(['exc', 'pv', 'sst', 'vip'])
    plt.title("Firing rates Exp1")
    plt.xlabel("t / s")

    plt.figure()

    plt.plot(time, f_rates[:, 0], scatter[0])
    plt.plot(time, f_rates2[:, 0], scatter[2])

    plt.legend(['Exp 1', 'Exp 2'])
    plt.title("Excitatory Firing rates")
    plt.xlabel("t / s")

    plt.figure()

    plt.plot(time, F)
    plt.plot(time, D)

    plt.legend(['F', 'D'])
    plt.title("Short-term plasticity")
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
