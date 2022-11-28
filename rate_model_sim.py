from functions import f_function, forward_euler, cont_pulse_trials

import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt


def run_sim(n_units, i_t, vip_in, f_flag, d_flag, dt, steps):
    ############################################################
    # Init
    ############################################################

    # neuron constants -> [exc, pv, sst, vip]
    thresholds = unit_gen(np.array([0.7, 1.0, 1.0, 0.0]), n_units)
    tau = unit_gen(np.array([10 * 1e-3, 10 * 1e-3, 10 * 1e-3, 10 * 1e-3]), n_units)  # s
    # i_opt = [0.0, -0.0, -0.0, 0.0]  # [0.0, -2.0, -1.0, 0.0]

    thal_flag = unit_gen(np.array([1, 1, 0, 0]), n_units)
    vip_flag = unit_gen(np.array([0, 0, 0, 1]), n_units)

    w_star = np.array([0.667, 1.25, 0.125, 0.0])  # only with exc. pre synapses, Park2020
    w_star = w_star[:, np.newaxis]
    weights = np.array([[1.1, -2, -1, -0], [1, -2, -2, -0], [6, -0, -0, -3], [0, -1.5, -3, -3]])
    # [[post_exc], [post_pv], [post_sst], [post_vip]]

    # init firing rates
    n_subtypes = len(tau)
    f_rates = np.zeros((steps + 1, n_subtypes, n_units))
    # f_rates[0, :] = np.random.rand(n_subtypes)

    # Depression and Facilitation constants - post exc.: Park2020; post inh. types: Campagnola2022
    D = np.zeros((steps + 1, n_units))
    D[0, :] = 1.0
    a_dep = np.array([[-0.0, 0.5, 0.0, 0.0], [-0, 0.5, 0.11, 0.13], [-0, 0.35, 0.18, 0], [-0, 0.37, 0, 0]])

    F = np.zeros((steps + 1, n_units))
    a_fac = np.array([[0.0, -0.0, -1.0, -0.0], [0, -0, -0, -0], [0, -0, -0, -0.05], [0, -0, -0.28, -0.04]])

    # thalamic input
    q = 5
    alpha = 0.65
    tau_d1 = 1500 * 1e-3  # s
    tau_d2 = 20 * 1e-3  # s
    g_0 = 1
    thal_input = np.zeros((steps + 1, n_units))
    thal_input[0, :] = g_0

    #########################################################################################
    # Simulation
    #########################################################################################
    # deb = 0
    for i in range(steps):
        # Update thalamic input
        dg_dt = (g_0 - thal_input[i, :]) / tau_d1 - thal_input[i, :] * i_t[i, :n_units] / tau_d2
        thal_input[i + 1, :] = forward_euler(dg_dt, thal_input[i, :], dt)
        thal_arg = thal_input[i + 1, :] * i_t[i, :n_units]

        # if f_rates[i, 0, 0] >= 0.7:  # debug
        #     if deb != i - 1:
        #         deb = 0
        #     deb = i

        # Update depression and facilitation terms
        dD_dt = (1 - D[i, :]) / tau_d1 - D[i, :] * i_t[i, :n_units] / tau_d2
        dF_dt = - F[i, :] / tau_d1 + i_t[i, :n_units] / tau_d2
        D[i + 1, :] = forward_euler(dD_dt, D[i, :], dt)
        F[i + 1, :] = forward_euler(dF_dt, F[i, :], dt)

        # Reshaping for multiple unit processing
        F_tmp = np.tile(F[i + 1, :], (n_subtypes, n_subtypes, 1))
        F_tmp = np.swapaxes(F_tmp, 0, 2)
        D_tmp = np.tile(D[i + 1, :], (n_subtypes, n_subtypes, 1))
        D_tmp = np.swapaxes(D_tmp, 0, 2)

        if n_units > 1:
            # Lateral inter-unit excitatory connections
            exc_combos = list(combinations(f_rates[i, 0, :], n_units - 1))
            cross_exc_rates = np.sum(exc_combos, axis=1)
            cross_exc_rates = np.flip(cross_exc_rates)  # -> nth index without contribution of unit n
            cross_exc_in = w_star * cross_exc_rates / (n_units - 1)

            # Lateral inter-unit thalamic input
            thal_combos = list(combinations(thal_arg * alpha, n_units - 1))
            cross_thal = np.sum(thal_combos, axis=1)
            cross_thal = np.flip(cross_thal)
        else:
            cross_exc_in = np.zeros((n_subtypes, n_units))
            cross_thal = np.zeros((n_subtypes, n_units))

        # Update Wilson-Cowan model
        tmp = np.swapaxes(weights + d_flag * a_dep * (1 - D_tmp) + f_flag * a_fac * F_tmp, 0, 2)
        tmp = np.swapaxes(tmp, 0, 1)
        f_rates_tmp = np.tile(f_rates[i, :, :], (n_subtypes, 1, 1))

        f_arg = np.sum(tmp * f_rates_tmp, axis=1) + cross_exc_in \
                + thal_flag * q * (thal_arg + cross_thal) + vip_flag * vip_in[i, :n_units]
        d_dt = (-f_rates[i, :, :] + f_function(f_arg - thresholds)) / tau
        f_rates[i + 1, :, :] = forward_euler(d_dt, f_rates[i, :, :], dt)  # update firing rates

    return f_rates, thal_input, F, D


def unit_gen(arr, no_of_units):
    arr = np.tile(arr, (no_of_units, 1))
    arr = np.swapaxes(arr, 0, 1)
    return arr


def exe_wilson_cowan():
    dt = 0.1 * 1e-3  # s
    t_ges = 10000 * 1e-3  # s
    steps = int(t_ges / dt)
    no_of_units = 2

    # Switch on/off arbitrary no of facilitation and depression terms
    # d_flag = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]])
    # d_flag = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    # d_flag = np.zeros((4, 4))
    d_flag = np.ones((4, 4))
    f_flag = d_flag

    # input stimulus
    stim_dur = 100 * 1e-3
    inter_stim_dur = 300 * 1e-3
    inter_trial_dur = 2400 * 1e-3
    trial_pulses = 8
    i_t = cont_pulse_trials(stim_dur, inter_stim_dur, inter_trial_dur, trial_pulses, steps, dt)
    i_t = np.tile(i_t, (no_of_units, 1))
    i_t = np.swapaxes(i_t, 0, 1)
    # i_t = np.ones(steps)

    # Higher order input
    # stim_dur = 5000 * 1e-3
    # inter_stim_dur = 5000 * 1e-3
    magnitude = 3.0
    vip_in = magnitude * cont_pulse_trials(stim_dur, inter_stim_dur, inter_trial_dur, trial_pulses, steps, dt)
    vip_in = np.tile(vip_in, (no_of_units, 1))
    vip_in = np.swapaxes(vip_in, 0, 1)

    [f_rates, thal_input, F, D] = run_sim(no_of_units, i_t, vip_in, f_flag, d_flag, dt, steps)

    # d_flag = np.ones((4, 4))
    # d_flag = np.zeros((4, 4))
    # f_flag = d_flag
    no_of_units = 1

    [f_rates2, _, _, _] = run_sim(no_of_units, i_t, vip_in, f_flag, d_flag, dt, steps)

    time = np.arange(0, t_ges + dt, dt)
    scatter = ['o', '.', '.', '--']

    for i in range(f_rates.shape[1]):
        plt.plot(time, f_rates[:, i, 0], scatter[i])

    plt.legend(['exc', 'pv', 'sst', 'vip'])
    plt.title("Firing rates Exp1")
    plt.xlabel("t / s")

    plt.figure()

    plt.plot(time, f_rates[:, 0, 0], scatter[0])
    plt.plot(time, f_rates2[:, 0, 0], scatter[2])

    plt.legend(['Exp 1', 'Exp 2'])
    plt.title("Comp. Firing rates")
    plt.xlabel("t / s")

    plt.figure()

    plt.plot(time, F[:, 0])
    plt.plot(time, D[:, 0])

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

    plt.plot(time, i_t[:, 0])
    plt.plot(time, vip_in[:, 0])

    plt.legend(['Stimulus', 'Higher Order'])
    plt.title("Input signals")
    plt.xlabel("t / s")

    plt.show()
