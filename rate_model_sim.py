from functions import f_function, cont_pulse_trials, forward_euler_rates, forward_euler

import numpy as np
# from itertools import combinations
import matplotlib.pyplot as plt


def run_sim(mode, i_t, vip_in, q_thal, q_vip, f_flag, d_flag, dt, steps, v_flag):
    ############################################################
    # Init
    ############################################################

    # neuron constants -> [exc, pv, sst, vip]
    thresholds = np.array([0.0, 0.1, 0.3, 0.])
    tau = np.array([10 * 1e-3, 10 * 1e-3, 10 * 1e-3, 10 * 1e-3])  # s
    # i_opt = [0.0, -0.0, -0.0, 0.0]  # [0.0, -2.0, -1.0, 0.0]

    thal_flag = np.array([1, 1, 0, 0])
    vip_flag = np.array([0, 0, 0, 1])

    # w_star = np.array([0.667, 1.25, 0.125, 0.0])  # only with exc. pre synapses, Park2020
    # w_star = w_star[:, np.newaxis]

    w_amp = 1
    # weights = w_amp * np.array([[1.1, -2, -1, -0.01], [1, -2, -2, -0.01], [6, -0, -0, -10], [0, -1.5, -0.5, -5]])
    weights = w_amp * np.array([[1.1, -3, -1, -0], [1, -2, -2, -0], [6, -0, -0, -3], [0, -0, -0.1, -3]])
    # weights = w_amp * np.array([[0.8, -1, -1, -0.0], [1, -1, -0.5, -0.0], [1, -0, -0, -0.25], [1, -0.0, -0.6, -0.0]])
    # [[post_exc], [post_pv], [post_sst], [post_vip]]

    # init firing rates
    n_subtypes = len(tau)
    f_rates = np.zeros((steps + 1, n_subtypes))
    # f_rates[0, :] = np.random.rand(n_subtypes)

    # Depression and Facilitation constants - Campagnola2022
    tau_df1 = 1500 * 1e-3  # s
    tau_df2 = 20 * 1e-3  # s
    stp_amp = 1

    D = np.zeros((steps + 1))
    D[0:2] = 1.0
    V = np.zeros((steps + 1))
    V[0:2] = 1.0
    a_dep = stp_amp * np.array([[-0.19, 0.49, 0.12, 0], [-0.04, 0.5, 0.11, 0], [-0, 0.35, 0.0, 0], [-0, 0.0, 0, 0]])
    # a_dep = np.array([[-0.19, 0.49, 0.12, 0.14], [-0.04, 0.5, 0.11, 0.13], [-0, 0.35, 0.18, 0], [-0, 0.37, 0, 0]])

    F = np.zeros((steps + 1))
    V2 = np.zeros((steps + 1))
    a_fac = stp_amp * np.array([[0, -0, -0, -0], [0, -0, -0, -0], [0.18, -0, -0, -0.9], [0.0, -0, -0.28, -0.04]])
    # a_fac = stp_amp * np.array([[0, -0, -0, -0], [0, -0, -0, -0], [0.18, -0, -0, -0.05], [0.03, -0, -0.28, -0.04]])

    v_flag = v_flag * 15

    # thalamic input
    # q = 5
    # alpha = 0.65
    tau_d1 = 1500 * 1e-3  # s
    tau_d2 = 20 * 1e-3  # s
    # tau_df2 = tau_d2  # s
    g_0 = 1
    thal_input = np.zeros((steps + 1))
    # thal_in = np.zeros((steps + 1))
    # thal_arg = np.zeros((steps + 1))
    thal_input[0:2] = g_0
    # thal_in[0:2] = g_0
    # thal_arg[0] = g_0
    baseline = [0.05, 0.0, 0.0, 0.0]

    # model functions
    dep_fcn = lambda arg, arg_input, tau1, tau2: (1 - arg) / tau1 - arg * arg_input / tau2
    fac_fcn = lambda arg, arg_input, tau1, tau2: - arg / tau1 + (1 - arg) * arg_input / tau2
    wc_fcn = lambda rates_arg, d_arg, v_arg, f_arg, v2_arg, thal_arg, in_arg, vip_in_arg: \
        (-rates_arg + f_function(
            np.sum((weights + d_flag * a_dep * (1 - d_arg) + v_flag * a_dep * (1 - v_arg) + f_flag * a_fac * f_arg +
                    v_flag * a_fac * v2_arg) * rates_arg, axis=1) +
            thal_flag * q_thal * thal_arg * in_arg + vip_flag * q_vip * vip_in_arg
            - thresholds) + baseline) / tau
    # wc_fcn = lambda rates_arg, d_arg, v_arg, f_arg, v2_arg, thal_arg, in_arg, vip_in_arg: \
    #     (-rates_arg + f_function(
    #         (np.sum(weights*rates_arg, axis=1) + thal_flag * q_thal * thal_arg * in_arg)
    #         - thresholds)) / tau
    # wc_fcn = lambda rates_arg, d_arg, v_arg, f_arg, v2_arg, thal_arg, in_arg, vip_in_arg: (1 - rates_arg) / tau_d1 - \
    # rates_arg * in_arg / tau_d2

    #########################################################################################
    # Simulation
    #########################################################################################
    # deb = 0
    for i in range(0, steps):
        # Update thalamic input
        # dg_dt = (g_0 - thal_input[i]) / tau_d1 - thal_input[i] * i_t[i] / tau_d2
        # thal_input[i + 1] = forward_euler(dg_dt, thal_input[i], dt)
        thal_input[i + 1] = forward_euler(dep_fcn, tau_d1, tau_d2, thal_input[i], i_t[i], dt)
        # thal_in[i + 1] = thal_input[i + 1] * i_t[i] * (1 - baseline) + baseline

        # thal_arg[i + 1] = thal_input[i + 1] * i_t[i]

        # if f_rates[i, 0, 0] >= 0.7:  # debug
        #     if deb != i - 1:
        #         deb = 0
        #     deb = i

        # Update depression and facilitation terms
        # dD_dt = (1 - D[i]) / tau_df1 - D[i] * i_t[i] / tau_df2
        # # dD_dt = (1 - D[i]) / tau_df1 - D[i] * vip_in[i] / tau_df2
        # dV_dt = (1 - V[i]) / tau_df1 - V[i] * vip_in[i] / tau_df2
        # dF_dt = - F[i] / tau_df1 + (1 - F[i]) * i_t[i] / tau_df2
        # # dF_dt = - F[i] / tau_df1 + (1 - F[i]) * vip_in[i] / tau_df2
        # dV2_dt = - V2[i] / tau_df1 + (1 - V2[i]) * vip_in[i] / tau_df2
        # D[i + 1] = forward_euler(dD_dt, D[i], dt)
        # V[i + 1] = forward_euler(dV_dt, V[i], dt)
        # F[i + 1] = forward_euler(dF_dt, F[i], dt)
        # V2[i + 1] = forward_euler(dV2_dt, V2[i], dt)

        D[i + 1] = forward_euler(dep_fcn, tau_df1, tau_df2, D[i], i_t[i], dt)
        V[i + 1] = forward_euler(dep_fcn, tau_df1, tau_df2, V[i], vip_in[i], dt)
        F[i + 1] = forward_euler(fac_fcn, tau_df1, tau_df2, F[i], i_t[i], dt)
        V2[i + 1] = forward_euler(fac_fcn, tau_df1, tau_df2, V2[i], vip_in[i], dt)

        if 0.4495 * steps < i < 0.4505 * steps and i_t[i] > 0 and i_t[i-1] == 0 and mode == 1:  # novel stim following
            thal_input[i + 1] = 0.45

        # if 0.0*steps < i < 0.5*steps:  # stp on
        #     #D[i + 1] = 0
        #     V[i + 1] = 0.5
        #     #F[i + 1] = 1
        #     V2[i + 1] = 0.5
        # else:
        #     #D[i + 1] = 0.5
        #     V[i + 1] = 0
        #     #F[i + 1] = 0.5
        #     V2[i + 1] = 1

        # Update Wilson-Cowan model
        # if i*dt > 5.9:
        #     tmp = weights + d_flag * a_dep * (1 - D[i + 1]) + v_flag * a_dep * (1 - V[i + 1]) \
        #           + f_flag * a_fac * F[i + 1] + v_flag * a_fac * V2[i+1]
        #     # tmp = np.swapaxes(tmp, 0, 1)
        #     # f_rates_tmp = np.tile(f_rates[i, :], (n_subtypes, 1, 1))
        #
        #     # thal_arg[i + 1, :] = 1
        #     f_arg = np.sum(tmp * f_rates[i, :], axis=1) \
        #             + thal_flag * q_thal * thal_input[i + 1] * i_t[i] + vip_flag * q_vip * vip_in[i]
        #     d_dt = (-f_rates[i, :] + f_function(f_arg - thresholds) + baseline) / tau

        d_dt = wc_fcn(f_rates[i, :], D[i+1], V[i+1], F[i+1], V2[i+1], thal_input[i+1], i_t[i], vip_in[i])
        f_rates[i+1, :] = forward_euler_rates(d_dt, f_rates[i, :], dt)

    return f_rates, thal_input, V2, V


def unit_gen(arr, no_of_units):
    arr = np.tile(arr, (no_of_units, 1))
    arr = np.swapaxes(arr, 0, 1)
    return arr


def exe_wilson_cowan():
    for dt_i in range(1, 3):
        if dt_i == 2:
            break

        # Mode config
        mode = 0

        mode_str = ["Image Omission - Familiar", "Image Change - Novel"]
        xlims = [[4.4, 6.3], [3.7, 4.8]]

        # dt = 0.05 * 1e-3 / dt_i  # s
        dt = 0.05 * 1e-3  # s
        t_ges = 10  # s
        steps = int(np.ceil(t_ges / dt))

        # Switch on/off arbitrary no of facilitation and depression terms
        # d_flag = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]])
        d_flag = np.array([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]])
        # d_flag = np.zeros((4, 4))
        # d_flag = np.ones((4, 4))
        f_flag = d_flag
        v_flag = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]])
        # v_flag = np.zeros((4, 4))
        # v_flag = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        # input stimulus (Bottom up)
        stim_dur = 250 * 1e-3
        inter_stim_dur = 500 * 1e-3
        # inter_trial_dur = 2400 * 1e-3
        inter_trial_dur = 1250 * 1e-3

        # stim_dur = 200 * 1e-3
        # inter_stim_dur = 600 * 1e-3
        # inter_trial_dur = 1400 * 1e-3
        trial_pulses = 7
        q_thal = 2.0
        i_t = cont_pulse_trials(0, 0, stim_dur, inter_stim_dur, inter_trial_dur, trial_pulses, steps, dt)
        # i_t = i_t - 0.5 * magnitude * cont_pulse_trials(0, stim_dur, inter_stim_dur, 8300 * 1e-3, 1, steps, dt)
        # i_t[84000:85000] = 0.5 * magnitude  # higher order impacting input plasticity (depressive)
        # i_t = np.tile(i_t, (no_of_units, 1))
        # i_t = np.swapaxes(i_t, 0, 1)
        # i_t = np.ones(steps)

        # Higher order input (Top down)
        if mode == 0:
            q_vip, vip_in = img_omission_fam(dt, steps, t_ges, trial_pulses, stim_dur)
        else:
            q_vip, vip_in = img_change_nov(dt, steps, t_ges, trial_pulses)

        [f_rates, thal_input, F, D] = run_sim(mode, i_t, vip_in, q_thal, q_vip, f_flag, d_flag, dt, steps, v_flag)

        # Presentation plots
        time = np.arange(0, t_ges, dt)
        population = ["Excitatory", "PV", "SST", "VIP"]

        for i in range(f_rates.shape[1]):
            plt.figure()
            scale_fac = max(f_rates[:-1, i]) / max(q_thal, q_vip) / 2
            plt.plot(time, i_t * q_thal * scale_fac)
            plt.plot(time, vip_in * q_vip * scale_fac)
            plt.plot(time, f_rates[:-1, i])

            plt.legend(["Bottom-up input (x" + str(round(scale_fac, 2)) + ")",
                        "Top-down input (x" + str(round(scale_fac, 2)) + ")",
                        population[i] + " Activity (normalized)"])
            plt.title(mode_str[mode])
            plt.xlabel("t / s")
            plt.xlim(xlims[mode])

        # # Std plots
        # plt.figure()
        #
        # time = np.arange(0, t_ges + dt, dt)
        # scatter = ['o', '^', '.', '-']
        #
        # for i in range(f_rates.shape[1]):
        #     plt.plot(time, f_rates[:, i], scatter[i])
        #
        # plt.legend(['exc', 'pv', 'sst', 'vip'])
        # plt.title("Firing rates Exp1")
        # plt.xlabel("t / s")
        # # plt.ylim(0, 0.6)

        plt.figure()
        time = np.arange(0, t_ges + dt, dt)

        plt.plot(time, F)
        plt.plot(time, D)
        plt.plot(time, thal_input)

        plt.legend(['F', 'D', 'Thalamic input'])
        plt.title("Short-term plasticity")
        plt.xlabel("t / s")

        plt.figure()

        time = np.arange(0, t_ges, dt)

        plt.plot(time, q_thal * i_t)
        plt.plot(time, q_vip * vip_in)

        plt.legend(['Stimulus', 'Higher Order'])
        plt.title("Input signals")
        plt.xlabel("t / s")

    plt.show()


def img_omission_fam(dt, steps, t_ref, trial_pulses, bot_up_dur):
    stim_dur = 20 * 1e-3
    stim_dur2 = 750 * 1e-3
    inter_stim_dur = 750 * 1e-3 - stim_dur
    bot_up_inter_dur = inter_stim_dur + stim_dur - bot_up_dur
    inter_trial_dur = 1500 * 1e-3 - stim_dur
    # off_frac = (inter_stim_dur + stim_dur * 0.5) / t_ges
    # trial_pulses = trial_pulses - 1
    q_vip = 0.25
    # q_vip = 1
    vip_in = cont_pulse_trials(1, bot_up_dur / t_ref, bot_up_inter_dur - 2*dt, t_ref, bot_up_dur, 1, steps, dt)
    vip_in[int((0.45 + stim_dur / t_ref) * steps):] = 0.0
    vip_in = vip_in + cont_pulse_trials(1, 0.625, bot_up_inter_dur - 2*dt, t_ref, bot_up_dur, 1, steps, dt)
    vip_in = vip_in + cont_pulse_trials(0, 0, stim_dur, inter_stim_dur, inter_trial_dur, trial_pulses, steps, dt)
    # vip_in[int(steps / 2):] = vip_in[int(steps / 2):] / 1.5
    # vip_in[int(0.6*steps):int((0.6+stim_dur*2/10)*steps)] = 1
    vip_amp_2 = 1 / q_vip
    vip_decay_amp = vip_amp_2 * 0.5
    rev_fac = 1.5

    vip_in = vip_in + cont_pulse_trials(1, 0.45 + bot_up_dur / t_ref, bot_up_inter_dur - dt, t_ref, t_ref, 1, steps, dt)
    vip_in = vip_in + cont_pulse_trials(0, 0.6 - stim_dur2 / t_ref, stim_dur2, inter_stim_dur, t_ref, 1, steps, dt)
    vip_in = vip_in + (vip_amp_2 - 1) * cont_pulse_trials(1, 0.6 - stim_dur2 / t_ref, stim_dur2, inter_stim_dur,
                                                          t_ref, 1, steps, dt)  # big ramp

    vip_in = vip_in + (vip_amp_2 - vip_decay_amp) * cont_pulse_trials(2, 0.6, stim_dur, inter_stim_dur, t_ref, 1, steps,
                                                                      dt)  # ramp decay
    vip_in = vip_in + vip_decay_amp * cont_pulse_trials(0, 0.6, stim_dur * rev_fac, inter_stim_dur, t_ref, 1, steps, dt)
    vip_in = vip_in - cont_pulse_trials(0, 0.6, stim_dur, inter_stim_dur, t_ref, 1, steps, dt)
    # vip_in[84000:84200] = vip_in[84000:84200] - q_vip
    # vip_in = vip_in + 0.5 * cont_pulse_trials(0, 350 * 1e-3, inter_stim_dur,
    # 2900 * 1e-3 + 100 * 1e-3, 1, steps, dt)
    return q_vip, vip_in


def img_change_nov(dt, steps, t_ref, trial_pulses):
    stim_dur = 20 * 1e-3
    inter_stim_dur = 750 * 1e-3 - stim_dur
    inter_trial_dur = 1500 * 1e-3 - stim_dur

    q_vip = 0.25
    vip_in = cont_pulse_trials(0, 0, stim_dur, inter_stim_dur, inter_trial_dur, trial_pulses, steps, dt)

    nov_amp = 1.25
    vip_in = vip_in + nov_amp * cont_pulse_trials(0, 0.45, stim_dur, inter_stim_dur, t_ref, 1, steps, dt)
    # vip_in = vip_in + nov_amp * cont_pulse_trials(0, 0.6, stim_dur, inter_stim_dur, t_ges, trial_pulses, steps, dt)

    return q_vip, vip_in
