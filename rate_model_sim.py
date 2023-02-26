from functions import f_function, forward_euler, cont_pulse_trials

import numpy as np
# from itertools import combinations
import matplotlib.pyplot as plt


def run_sim(i_t, vip_in, q_thal, q_vip, f_flag, d_flag, dt, steps, v_flag):
    ############################################################
    # Init
    ############################################################

    # neuron constants -> [exc, pv, sst, vip]
    thresholds = np.array([-0.1, -0.0, 0.0, 0.0])  # -> baseline
    tau = np.array([10 * 1e-3, 10 * 1e-3, 10 * 1e-3, 10 * 1e-3])  # s
    # i_opt = [0.0, -0.0, -0.0, 0.0]  # [0.0, -2.0, -1.0, 0.0]

    thal_flag = np.array([1, 1, 0, 0])
    vip_flag = np.array([0, 0, 0, 1])

    # w_star = np.array([0.667, 1.25, 0.125, 0.0])  # only with exc. pre synapses, Park2020
    # w_star = w_star[:, np.newaxis]

    w_amp = 1
    # weights = w_amp * np.array([[1.1, -2, -1, -0.01], [1, -2, -2, -0.01], [6, -0, -0, -10], [0, -1.5, -0.5, -5]])
    weights = w_amp * np.array([[1.1, -2, -1, -0], [1, -2, -2, -0], [6, -0, 0, -3], [0, -0, -0.1, -5]])
    # weights = w_amp * np.array([[0.8, -1, -1, -0.0], [1, -1, -0.5, -0.0], [1, -0, -0, -0.25], [1, -0.0, -0.6, -0.0]])
    # [[post_exc], [post_pv], [post_sst], [post_vip]]

    # init firing rates
    n_subtypes = len(tau)
    f_rates = np.zeros((steps + 1, n_subtypes))
    # f_rates[0, :] = np.random.rand(n_subtypes)

    # Depression and Facilitation constants - Campagnola2022
    tau_df1 = 1500 * 1e-3  # s
    tau_df2 = 20 * 1e-3  # s
    stp_amp = 0.5

    D = np.zeros((steps + 1))
    D[0] = 1.0
    V = np.zeros((steps + 1))
    V[0] = 1.0
    # a_dep = np.array([[-0.19, 0.49, 0.12, 0.14], [-0.04, 0.5, 0.11, 0.13], [-0, 0.35, 0.18, 0], [-0, 0.37, 0, 0]])
    a_dep = stp_amp * np.array([[-0.19, 0.49, 0.12, 0], [-0.04, 0.5, 0.11, 0], [-0, 0.35, 0.18, 0], [-0, 0.0, 0, 0]])

    F = np.zeros((steps + 1))
    V2 = np.zeros((steps + 1))
    # a_fac = np.array([[0, -0, -0, -0], [0, -0, -0, -0], [0.18, -0, -0, -0.05], [0.03, -0, -0.28, -0.04]])
    a_fac = stp_amp * np.array([[0, -0, -0, -0], [0, -0, -0, -0], [0.18, -0, -0, -0.05], [0, -0, -0.28, -0.04]])

    # thalamic input
    # q = 5
    # alpha = 0.65
    tau_d1 = 1500 * 1e-3  # s
    tau_d2 = 20 * 1e-3  # s
    # tau_d2 = tau_df2  # s
    g_0 = 1
    thal_input = np.zeros((steps + 1))
    thal_arg = np.zeros((steps + 1))
    thal_input[0] = g_0
    thal_arg[0] = g_0

    #########################################################################################
    # Simulation
    #########################################################################################
    # deb = 0
    for i in range(steps):
        # Update thalamic input
        dg_dt = (g_0 - thal_input[i]) / tau_d1 - thal_input[i] * i_t[i] / tau_d2
        thal_input[i + 1] = forward_euler(dg_dt, thal_input[i], dt)
        # thal_arg[i + 1] = thal_input[i + 1] * i_t[i]

        # if f_rates[i, 0, 0] >= 0.7:  # debug
        #     if deb != i - 1:
        #         deb = 0
        #     deb = i

        # Update depression and facilitation terms
        dD_dt = (1 - D[i]) / tau_df1 - D[i] * i_t[i] / tau_df2
        # dD_dt = (1 - D[i]) / tau_df1 - D[i] * vip_in[i] / tau_df2
        dV_dt = (1 - V[i]) / tau_df1 - V[i] * vip_in[i] / tau_df2
        dF_dt = - F[i] / tau_df1 + (1 - F[i]) * i_t[i] / tau_df2
        # dF_dt = - F[i] / tau_df1 + (1 - F[i]) * vip_in[i] / tau_df2
        dV2_dt = - V2[i] / tau_df1 + (1 - V2[i]) * vip_in[i] / tau_df2
        D[i + 1] = forward_euler(dD_dt, D[i], dt)
        V[i + 1] = forward_euler(dV_dt, V[i], dt)
        F[i + 1] = forward_euler(dF_dt, F[i], dt)
        V2[i + 1] = forward_euler(dV2_dt, V2[i], dt)

        # if 0.0*steps < i < 0.5*steps:  # stp on
        #     D[i + 1] = 0
        #     V[i + 1] = 0
        #     F[i + 1] = 1
        #     V2[i + 1] = 1
        # else:
        #     D[i + 1] = 1
        #     V[i + 1] = 1
        #     F[i + 1] = 0
        #     V2[i + 1] = 0

        # # Reshaping for multiple unit processing
        # F_tmp = np.tile(F[i + 1, :], (n_subtypes, n_subtypes, 1))
        # F_tmp = np.swapaxes(F_tmp, 0, 2)
        # D_tmp = np.tile(D[i + 1, :], (n_subtypes, n_subtypes, 1))
        # D_tmp = np.swapaxes(D_tmp, 0, 2)
        #
        # if n_units > 1:
        #     # Lateral inter-unit excitatory connections
        #     exc_combos = list(combinations(f_rates[i, 0, :], n_units - 1))
        #     cross_exc_rates = np.sum(exc_combos, axis=1)
        #     cross_exc_rates = np.flip(cross_exc_rates)  # -> nth index without contribution of unit n
        #     cross_exc_in = w_star * cross_exc_rates / (n_units - 1)
        #
        #     # Lateral inter-unit thalamic input
        #     thal_combos = list(combinations(thal_arg[i + 1, :] * alpha, n_units - 1))
        #     cross_thal = np.sum(thal_combos, axis=1)
        #     cross_thal = np.flip(cross_thal)
        # else:
        #     cross_exc_in = np.zeros((n_subtypes, n_units))
        #     cross_thal = np.zeros((n_subtypes, n_units))

        # Update Wilson-Cowan model
        tmp = weights + d_flag * a_dep * (1 - D[i + 1]) + v_flag * a_dep * (1 - V[i + 1]) \
              + f_flag * a_fac * F[i + 1] + v_flag * a_fac * V2[i+1]
        # tmp = np.swapaxes(tmp, 0, 1)
        # f_rates_tmp = np.tile(f_rates[i, :], (n_subtypes, 1, 1))

        # thal_arg[i + 1, :] = 1
        f_arg = np.sum(tmp * f_rates[i, :], axis=1) \
                + thal_flag * q_thal * thal_input[i + 1] * i_t[i] + vip_flag * q_vip * vip_in[i]
        d_dt = (-f_rates[i, :] + f_function(f_arg - thresholds)) / tau
        f_rates[i + 1, :] = forward_euler(d_dt, f_rates[i, :], dt)  # update firing rates
        # f_rates[i + 1, 3, :] = vip_in[i, :n_units]

    return f_rates, thal_input, F, D


def unit_gen(arr, no_of_units):
    arr = np.tile(arr, (no_of_units, 1))
    arr = np.swapaxes(arr, 0, 1)
    return arr


def exe_wilson_cowan():
    dt = 0.025 * 1e-3  # s
    t_ges = 10000 * 1e-3  # s
    steps = int(t_ges / dt)

    # Switch on/off arbitrary no of facilitation and depression terms
    # d_flag = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]])
    d_flag = np.array([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]])
    # d_flag = np.zeros((4, 4))
    # d_flag = np.ones((4, 4))
    f_flag = d_flag
    v_flag = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]])
    # v_flag = np.zeros((4, 4))
    # v_flag = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    # input stimulus
    stim_dur = 250 * 1e-3
    inter_stim_dur = 500 * 1e-3
    # inter_trial_dur = 2400 * 1e-3
    inter_trial_dur = 1250 * 1e-3

    # stim_dur = 200 * 1e-3
    # inter_stim_dur = 600 * 1e-3
    # inter_trial_dur = 1400 * 1e-3
    trial_pulses = 7
    q_thal = 0.1
    i_t = cont_pulse_trials(0, 0, stim_dur, inter_stim_dur, inter_trial_dur, trial_pulses, steps, dt)
    # i_t = i_t - 0.5 * magnitude * cont_pulse_trials(0, stim_dur, inter_stim_dur, 8300 * 1e-3, 1, steps, dt)
    # i_t[84000:85000] = 0.5 * magnitude  # higher order impacting input plasticity (depressive)
    # i_t = np.tile(i_t, (no_of_units, 1))
    # i_t = np.swapaxes(i_t, 0, 1)
    # i_t = np.ones(steps)

    # Higher order input
    stim_dur = 20 * 1e-3
    inter_stim_dur = 750 * 1e-3 - stim_dur
    inter_trial_dur = 1500 * 1e-3 - stim_dur
    # off_frac = (inter_stim_dur + stim_dur * 2) / t_ges
    # trial_pulses = trial_pulses - 1
    q_vip = 0.5
    # q_vip = 1
    vip_in = cont_pulse_trials(0, 0, stim_dur, inter_stim_dur, inter_trial_dur, trial_pulses, steps, dt)
    # vip_in[int(steps / 2):] = vip_in[int(steps / 2):] / 1.5
    stim_dur2 = 750 * 1e-3
    # vip_in[int(0.6*steps):int((0.6+stim_dur*2/10)*steps)] = 1
    vip_amp_2 = 2
    vip_in = vip_in + vip_amp_2 * cont_pulse_trials(1, 0.525, stim_dur2, inter_stim_dur, t_ges, 1, steps, dt)
    vip_in = vip_in + (vip_amp_2 - 1) * cont_pulse_trials(2, 0.6, stim_dur, inter_stim_dur, t_ges, 1, steps, dt)
    vip_in = vip_in + cont_pulse_trials(0, 0.6 + stim_dur/10, stim_dur, inter_stim_dur, t_ges, 1, steps, dt)
    # vip_in[84000:84200] = vip_in[84000:84200] - q_vip
    # vip_in = vip_in + 0.5 * cont_pulse_trials(0, 350 * 1e-3, inter_stim_dur, 2900 * 1e-3 + 100 * 1e-3, 1, steps, dt)

    # vip_in = np.tile(vip_in, (no_of_units, 1))
    # vip_in = np.swapaxes(vip_in, 0, 1)

    [f_rates, thal_input, F, D] = run_sim(i_t, vip_in, q_thal, q_vip,
                                          f_flag, d_flag, dt, steps, v_flag)

    # d_flag = np.ones((4, 4))
    # d_flag = np.zeros((4, 4))
    # f_flag = d_flag

    # no_of_units = 1
    #
    # [f_rates2, _, _, _] = run_sim(no_of_units, i_t, baseline, vip_in, f_flag, d_flag, dt, steps)

    time = np.arange(0, t_ges + dt, dt)
    scatter = ['o', '^', '.', '-']

    for i in range(f_rates.shape[1]):
        plt.plot(time, f_rates[:, i], scatter[i])

    #plt.legend(['exc', 'pv', 'sst', 'vip'])
    plt.title("Firing rates Exp1")
    plt.xlabel("t / s")

    # plt.figure()
    #
    # plt.plot(time, f_rates[:, 0, 0], scatter[0])
    # plt.plot(time, f_rates2[:, 0, 0], scatter[2])
    #
    # plt.legend(['Exp 1', 'Exp 2'])
    # plt.title("Comp. Firing rates")
    # plt.xlabel("t / s")

    plt.figure()

    plt.plot(time, F)
    plt.plot(time, D)
    plt.plot(time, thal_input)

    #plt.legend(['F', 'D', 'Thalamic input'])
    plt.title("Short-term plasticity")
    plt.xlabel("t / s")

    # plt.figure()
    #
    # plt.plot(time, thal_input)
    # plt.title("Thalamic input")
    # plt.xlabel("t / s")

    plt.figure()

    time = np.arange(0, t_ges, dt)

    plt.plot(time, q_thal * i_t)
    plt.plot(time, q_vip * vip_in)

    #plt.legend(['Stimulus', 'Higher Order'])
    plt.title("Input signals")
    plt.xlabel("t / s")

    plt.show()
