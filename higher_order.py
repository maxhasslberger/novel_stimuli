from functions import cont_pulse_trials


def img_omission_fam(dt, steps, t_ref, bot_up_dur):
    stim_dur = 20 * 1e-3
    stim_dur2 = 750 * 1e-3
    inter_stim_dur = 750 * 1e-3 - stim_dur
    bot_up_inter_dur = inter_stim_dur + stim_dur - bot_up_dur
    # inter_trial_dur = 1500 * 1e-3 - stim_dur
    # off_frac = (inter_stim_dur + stim_dur * 0.5) / t_ges
    # trial_pulses = trial_pulses - 1
    q_vip = 0.25
    # q_vip = 1
    vip_in = cont_pulse_trials(1, bot_up_dur / t_ref, bot_up_inter_dur - 2*dt, t_ref, bot_up_dur, 1, steps, dt)
    vip_in[int((0.45 + stim_dur / t_ref) * steps):] = 0.0
    vip_in = vip_in + cont_pulse_trials(1, 0.625, bot_up_inter_dur - 2*dt, t_ref, bot_up_dur, 1, steps, dt)
    # vip_in = vip_in + cont_pulse_trials(0, 0, stim_dur, inter_stim_dur, inter_trial_dur, trial_pulses, steps, dt)
    # vip_in[int(steps / 2):] = vip_in[int(steps / 2):] / 1.5
    # vip_in[int(0.6*steps):int((0.6+stim_dur*2/10)*steps)] = 1
    vip_amp_2 = 1 / q_vip
    vip_decay_amp = vip_amp_2 * 0.5
    rev_fac = 1.5

    vip_in = vip_in + cont_pulse_trials(1, 0.45 + bot_up_dur / t_ref, bot_up_inter_dur, t_ref, t_ref, 1, steps, dt)
    vip_in = vip_in + cont_pulse_trials(0, 0.6 - stim_dur2 / t_ref, stim_dur2, inter_stim_dur, t_ref, 1, steps, dt)
    vip_in = vip_in + (vip_amp_2 - 1) * cont_pulse_trials(1, 0.6 - stim_dur2 / t_ref, stim_dur2, inter_stim_dur,
                                                          t_ref, 1, steps, dt)  # big ramp

    vip_in = vip_in + (vip_amp_2 - vip_decay_amp) * cont_pulse_trials(2, 0.6, stim_dur, inter_stim_dur, t_ref, 1, steps,
                                                                      dt)  # ramp decay
    vip_in = vip_in + vip_decay_amp * cont_pulse_trials(0, 0.6, stim_dur * rev_fac, inter_stim_dur, t_ref, 1, steps, dt)
    # vip_in = vip_in - cont_pulse_trials(0, 0.6, stim_dur, inter_stim_dur, t_ref, 1, steps, dt)
    # vip_in[84000:84200] = vip_in[84000:84200] - q_vip
    # vip_in = vip_in + 0.5 * cont_pulse_trials(0, 350 * 1e-3, inter_stim_dur,
    # 2900 * 1e-3 + 100 * 1e-3, 1, steps, dt)
    return q_vip, vip_in


def img_change_fam(dt, steps, t_ref, bot_up_dur):
    stim_dur = 20 * 1e-3
    stim_dur2 = 750 * 1e-3
    inter_stim_dur = 750 * 1e-3 - stim_dur
    bot_up_inter_dur = inter_stim_dur + stim_dur - bot_up_dur
    # inter_trial_dur = 1500 * 1e-3 - stim_dur
    # off_frac = (inter_stim_dur + stim_dur * 0.5) / t_ges
    # trial_pulses = trial_pulses - 1
    q_vip = 0.25
    # q_vip = 1
    vip_in = cont_pulse_trials(1, bot_up_dur / t_ref, bot_up_inter_dur - 2*dt, t_ref, bot_up_dur, 1, steps, dt)
    vip_in[int((0.45 + stim_dur / t_ref) * steps):] = 0.0
    vip_in = vip_in + cont_pulse_trials(1, 0.625, bot_up_inter_dur - 2*dt, t_ref, bot_up_dur, 1, steps, dt)
    # vip_in = vip_in + cont_pulse_trials(0, 0, stim_dur, inter_stim_dur, inter_trial_dur, trial_pulses, steps, dt)
    # vip_in[int(steps / 2):] = vip_in[int(steps / 2):] / 1.5
    # vip_in[int(0.6*steps):int((0.6+stim_dur*2/10)*steps)] = 1
    vip_amp_2 = 1 / q_vip
    vip_decay_amp = vip_amp_2 * 0.5
    rev_fac = 1.5

    vip_in = vip_in + cont_pulse_trials(1, 0.45 + bot_up_dur / t_ref, bot_up_inter_dur, t_ref, t_ref, 1, steps, dt)
    vip_in = vip_in + cont_pulse_trials(0, 0.6 - stim_dur2 / t_ref, stim_dur2, inter_stim_dur, t_ref, 1, steps, dt)
    vip_in = vip_in + (vip_amp_2 - 1) * cont_pulse_trials(1, 0.6 - stim_dur2 / t_ref, stim_dur2, inter_stim_dur,
                                                          t_ref, 1, steps, dt)  # big ramp

    vip_in = vip_in + (vip_amp_2 - vip_decay_amp) * cont_pulse_trials(2, 0.6, stim_dur, inter_stim_dur, t_ref, 1, steps,
                                                                      dt)  # ramp decay
    vip_in = vip_in + vip_decay_amp * cont_pulse_trials(0, 0.6, stim_dur * rev_fac, inter_stim_dur, t_ref, 1, steps, dt)
    # vip_in = vip_in - cont_pulse_trials(0, 0.6, stim_dur, inter_stim_dur, t_ref, 1, steps, dt)
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
