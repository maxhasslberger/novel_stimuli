from functions import f_function, cont_pulse_trials, forward_euler_rates, forward_euler
import higher_order as ho

import numpy as np
# from itertools import combinations
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import VideoClip


def run_sim(mode, i_t, vip_in, q_thal, q_vip, f_flag, d_flag, dt, steps, v_flag, nov_plus):
    ############################################################
    # Init
    ############################################################

    # neuron constants -> [exc, pv, sst, vip]
    thresholds = np.array([0.0, 0.1, 0.3, 0.])
    tau = np.array([10 * 1e-3, 10 * 1e-3, 10 * 1e-3, 10 * 1e-3])  # s

    thal_flag = np.array([1, 1, 0, 0])
    vip_flag = np.array([0, 0, 0, 1])

    w_amp = 1
    # weights = w_amp * np.array([[1.1, -2, -1, -0.01], [1, -2, -2, -0.01], [6, -0, -0, -10], [0, -1.5, -0.5, -5]])
    weights = w_amp * np.array([[1.1, -3, -1, -0], [1, -2, -2, -0], [6, -2, -0, -13.2], [0, -0, -0.1, -3]])
    # [[post_exc], [post_pv], [post_sst], [post_vip]]

    # init firing rates
    n_subtypes = len(tau)
    f_rates = np.zeros((steps + 1, n_subtypes))
    # f_rates[0, :] = np.random.rand(n_subtypes)

    # Depression and Facilitation constants - Campagnola2022
    tau_df1 = 1500 * 1e-3  # s
    tau_df2 = 20 * 1e-3  # s
    stp_amp = 1

    D = np.ones((steps + 1))
    V = np.ones((steps + 1))
    a_dep = stp_amp * np.array([[-0.19, 0.49, 0.12, 0], [-0.04, 0.5, 0.11, 0], [-0, 0.35, 0.0, 0], [-0, 0.0, 0, 0]])
    # a_dep = np.array([[-0.19, 0.49, 0.12, 0.14], [-0.04, 0.5, 0.11, 0.13], [-0, 0.35, 0.18, 0], [-0, 0.37, 0, 0]])

    F = np.zeros((steps + 1))
    V2 = np.zeros((steps + 1))
    a_fac = stp_amp * np.array([[0, -0, -0, -0], [0, -0, -0, -0], [0.18, -0, -0, -0.05], [0.0, -0, -0.0, -0.04]])
    # a_fac = stp_amp * np.array([[0, -0, -0, -0], [0, -0, -0, -0], [0.18, -0, -0, -0.05], [0.03, -0, -0.28, -0.04]])

    # thalamic input
    tau_d1 = 1500 * 1e-3  # s
    d1_thal_frac_novel = 0.1
    tau_d2 = 20 * 1e-3  # s

    thal_fac = 1.0
    thal_fac_novel = 0.4

    thal_change = 0.4
    thal_nov_change_fac = 0.8 / thal_change

    exc_r = 1.5
    exc_nov_r_fac = 2.1 / exc_r
    if 2 <= mode <= 3:  # Novel case
        tau_d1 = d1_thal_frac_novel * tau_d1
        thal_fac = thal_fac_novel * thal_fac
        thal_change = thal_nov_change_fac * thal_change
        exc_r = exc_nov_r_fac * exc_r
    elif mode >= 4:  # Novel+ case
        tau_d1 = (d1_thal_frac_novel + nov_plus * (1 - d1_thal_frac_novel)) * tau_d1
        thal_fac = (thal_fac_novel + nov_plus * (1 - thal_fac_novel)) * thal_fac
        thal_change = (thal_nov_change_fac + nov_plus * (1 - thal_nov_change_fac)) * thal_change
        exc_r = (exc_nov_r_fac + nov_plus * (1 - exc_nov_r_fac)) * exc_r

    g_0 = 1
    thal_input = np.ones((steps + 1)) * g_0
    baseline = [0.05, 0.0, 0.0, 0.0]

    # model functions
    dep_fcn = lambda arg, arg_input, tau1, tau2: (1 - arg) / tau1 - arg * arg_input / tau2
    fac_fcn = lambda arg, arg_input, tau1, tau2: - arg / tau1 + (1 - arg) * arg_input / tau2
    wc_fcn = lambda rates_arg, d_arg, v_arg, f_arg, v2_arg, thal_arg, in_arg, vip_in_arg: \
        (-rates_arg + f_function(
            np.sum((weights + d_flag * a_dep * (1 - d_arg) + v_flag * a_dep * (1 - v_arg) + f_flag * a_fac * f_arg +
                    v_flag * a_fac * v2_arg) * rates_arg, axis=1) +
            thal_flag * q_thal * thal_arg * thal_fac * in_arg + vip_flag * q_vip * vip_in_arg
            - thresholds, exc_r) + baseline) / tau
    # wc_fcn = lambda rates_arg, d_arg, v_arg, f_arg, v2_arg, thal_arg, in_arg, vip_in_arg: (1 - rates_arg) / tau_d1 - \
    # rates_arg * in_arg / tau_d2

    #########################################################################################
    # Simulation
    #########################################################################################
    # deb = 0
    for i in range(0, steps):
        # Update thalamic input
        thal_input[i + 1] = forward_euler(dep_fcn, tau_d1, tau_d2, thal_input[i], i_t[i], dt)

        # if f_rates[i, 0, 0] >= 0.7:  # debug
        #     if deb != i - 1:
        #         deb = 0
        #     deb = i

        D[i + 1] = forward_euler(dep_fcn, tau_df1, tau_df2, D[i], i_t[i], dt)
        V[i + 1] = forward_euler(dep_fcn, tau_df1, tau_df2, V[i], vip_in[i], dt)
        F[i + 1] = forward_euler(fac_fcn, tau_df1, tau_df2, F[i], i_t[i], dt)
        V2[i + 1] = forward_euler(fac_fcn, tau_df1, tau_df2, V2[i], vip_in[i], dt)

        # image change following
        if 0.4495 * steps < i < 0.4505 * steps and i_t[i] > 0 and i_t[i-1] == 0 and np.mod(mode, 2):
            thal_input[i + 1] = thal_change / thal_fac

        d_dt = wc_fcn(f_rates[i, :], D[i+1], V[i+1], F[i+1], V2[i+1], thal_input[i+1], i_t[i], vip_in[i])
        f_rates[i+1, :] = forward_euler_rates(d_dt, f_rates[i, :], dt)

    return f_rates, thal_input*thal_fac, F, D


def exe_wilson_cowan(mode=4, nov_plus=0.3, dt=0.05 * 1e-3):
    # # Mode config
    # mode = 4
    # nov_plus = 0.3  # Only for novel+ cases: 1.0 -> familiar; 0.0 -> novel

    mode_str = ["Image Omission - Familiar", "Image Change - Familiar", "Image Omission - Novel",
                "Image Change - Novel",
                "Image Omission - Novel+ - " + "%.1f" % (nov_plus*100) + "%",
                "Image Change - Novel+ - " + "%.1f" % (nov_plus*100) + "%"]
    xlims = [[4.4, 6.3], [3.7, 4.8], [4.4, 6.3], [3.7, 4.8], [4.4, 6.3], [3.7, 4.8]]

    # # dt = 0.05 * 1e-3 / dt_i  # s
    # dt = 0.05 * 1e-3  # s
    t_ges = 10  # s
    steps = int(np.ceil(t_ges / dt))

    # Switch on/off arbitrary no of facilitation and depression terms
    d_flag = np.array([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]])
    # d_flag = np.zeros((4, 4))
    # d_flag = np.ones((4, 4))
    f_flag = d_flag
    v_flag = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]])
    # v_flag = np.zeros((4, 4))

    # input param
    stim_dur = 250 * 1e-3
    inter_stim_dur = 500 * 1e-3
    inter_trial_dur = 1250 * 1e-3

    trial_pulses = 7 + 8 * np.mod(mode, 2)  # 15 for omissions, 7 for changes

    # Higher order input (Top down)
    if mode == 0:
        q_vip, vip_in = ho.img_omission_fam(dt, steps, t_ges, stim_dur)
    elif mode == 1:
        q_vip, vip_in = ho.img_change_fam(dt, steps, t_ges, stim_dur)
    elif mode == 2:
        q_vip, vip_in = ho.img_omission_nov(dt, steps, t_ges, trial_pulses)
    elif mode == 3:
        q_vip, vip_in = ho.img_change_nov(dt, steps, t_ges, trial_pulses)
    elif mode == 4:
        q_vip, vip_in = ho.img_omission_mix(dt, steps, t_ges, stim_dur, trial_pulses, nov_plus)
    else:
        q_vip, vip_in = ho.img_change_mix(dt, steps, t_ges, stim_dur, trial_pulses, nov_plus)

    # Input stimulus (Bottom up)
    q_thal = 2.0
    i_t = cont_pulse_trials(0, 0, stim_dur, inter_stim_dur, inter_trial_dur, trial_pulses, steps, dt)

    [f_rates, thal_input, F, D] = run_sim(mode, i_t, vip_in, q_thal, q_vip, f_flag, d_flag, dt, steps, v_flag, nov_plus)

    obj = plot_rates(D, F, dt, f_rates, i_t, mode, mode_str, q_thal, q_vip, t_ges, thal_input, vip_in, xlims,
                     plot=False)

    return obj


def create_video_from_plots(plots, filename, fps):
    # Define a function to convert a plot to an image
    def plot_to_npimage(plot):
        plot.set_size_inches(6, 4)
        return mplfig_to_npimage(plot)

    # Create a video clip from the list of plots
    clip = VideoClip(lambda t: plot_to_npimage(plots[int(t * fps)]), duration=len(plots) / fps)

    # Write the video file
    clip.write_videofile(filename, fps=fps, codec='libx264')


def plot_rates(dep, fac, dt, f_rates, i_t, mode, mode_str, q_thal, q_vip, t_ges, thal_input, vip_in, xlims, plot=True):

    # Std plots
    plt.figure()

    time = np.arange(0, t_ges + dt, dt)
    scatter = ['o', '^', '.', '-']

    for i in range(f_rates.shape[1]):
        plt.plot(time, f_rates[:, i], scatter[i])

    plt.legend(['exc', 'pv', 'sst', 'vip'])
    plt.title("Firing rates Exp1")
    plt.xlabel("t / s")
    # plt.ylim(0, 0.6)

    plt.figure()
    time = np.arange(0, t_ges + dt, dt)

    plt.plot(time, fac)
    plt.plot(time, dep)
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

    # Presentation plots
    time = np.arange(0, t_ges, dt)
    population = ["Excitatory", "PV", "SST", "VIP"]

    for i in range(f_rates.shape[1]):
        plt.figure()
        scale_fac = max(f_rates[:-1, i]) / max(q_thal, q_vip) / 2
        plt.plot(time, i_t * q_thal * scale_fac)
        plt.plot(time, vip_in * q_vip * scale_fac)
        plt.plot(time, f_rates[:-1, i])

        plt.legend(["Bottom-up input (x" + str(round(scale_fac, 2)) + ")",# Bug: scales not always the same!
                    "Top-down input (x" + str(round(scale_fac, 2)) + ")",
                    population[i] + " Activity (normalized)"])
    plt.title(mode_str[mode])
    plt.xlabel("t / s")
    plt.xlim(xlims[mode])
    # plt.ylim(-0.05, 1.05)

    obj = plt.gcf()

    if plot:
        plt.show()

    return obj
