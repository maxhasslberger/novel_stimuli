from functions import cont_pulse_trials


def img_omission_fam(dt, steps, t_ref, bot_up_dur):
    # Cont.
    stim_dur = 20 * 1e-3
    stim_dur2 = 750 * 1e-3
    inter_stim_dur = 750 * 1e-3 - stim_dur
    bot_up_inter_dur = inter_stim_dur + stim_dur - bot_up_dur

    q_vip = 0.125
    vip_in = cont_pulse_trials(1, bot_up_dur / t_ref, bot_up_inter_dur, t_ref, bot_up_dur, 1, steps, dt)
    vip_in[int((0.45 + stim_dur / t_ref) * steps):] = 0.0
    vip_in = vip_in + cont_pulse_trials(1, 0.625, bot_up_inter_dur, t_ref, bot_up_dur, 1, steps, dt)

    # Omission
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
    # Cont.
    stim_dur = 20 * 1e-3
    inter_stim_dur = 750 * 1e-3 - stim_dur
    bot_up_inter_dur = inter_stim_dur + stim_dur - bot_up_dur

    q_vip = 0.125
    vip_in = cont_pulse_trials(1, bot_up_dur / t_ref, bot_up_inter_dur, t_ref, bot_up_dur, 1, steps, dt)

    # Novel stim
    nov_amp = 1.5
    vip_in = vip_in + nov_amp * cont_pulse_trials(0, 0.45, stim_dur, inter_stim_dur, t_ref, 1, steps, dt)

    return q_vip, vip_in


def img_omission_nov(dt, steps, t_ref, trial_pulses):
    # Cont.
    stim_dur = 20 * 1e-3
    dur2 = 750 * 1e-3
    inter_stim_dur = 750 * 1e-3 - stim_dur
    inter_trial_dur = 2250 * 1e-3 - stim_dur

    q_vip = 0.7
    vip_in = cont_pulse_trials(0, 0, stim_dur, inter_stim_dur, inter_trial_dur, trial_pulses, steps, dt)

    # Omission
    om_amp = 0.2
    rev_fac = 1
    aft_stim_add = 1.5
    vip_in = vip_in + om_amp * cont_pulse_trials(1, 0.6 - dur2 / t_ref, dur2, inter_stim_dur, t_ref, 1, steps, dt)
    vip_in = vip_in + aft_stim_add * cont_pulse_trials(0, 0.6, stim_dur * rev_fac, inter_stim_dur, t_ref, 1, steps, dt)

    return q_vip, vip_in


def img_change_nov(dt, steps, t_ref, trial_pulses):
    # Cont.
    stim_dur = 20 * 1e-3
    inter_stim_dur = 750 * 1e-3 - stim_dur + dt
    inter_trial_dur = 1500 * 1e-3 - stim_dur

    q_vip = 0.7
    vip_in = cont_pulse_trials(0, 0, stim_dur, inter_stim_dur, inter_trial_dur, trial_pulses, steps, dt)

    # Novel stim
    nov_amp = 1.5
    vip_in = vip_in + nov_amp * cont_pulse_trials(0, 0.45, stim_dur, inter_stim_dur, t_ref, 1, steps, dt)

    return q_vip, vip_in


def img_omission_mix(dt, steps, t_ref, bot_up_dur, trial_pulses, fam_prc):
    # Familiar part
    q_vip_fam, vip_in_fam = img_omission_fam(dt, steps, t_ref, bot_up_dur)

    # Novel part
    q_vip_nov, vip_in_nov = img_omission_nov(dt, steps, t_ref, trial_pulses)

    # Scale and merge
    q_vip, vip_in = merge_novel_familiar(fam_prc, q_vip_fam, q_vip_nov, vip_in_fam, vip_in_nov)

    return q_vip, vip_in


def img_change_mix(dt, steps, t_ref, bot_up_dur, trial_pulses, fam_prc):
    # Familiar part
    q_vip_fam, vip_in_fam = img_change_fam(dt, steps, t_ref, bot_up_dur)

    # Novel part
    q_vip_nov, vip_in_nov = img_change_nov(dt, steps, t_ref, trial_pulses)

    # Scale and merge
    q_vip, vip_in = merge_novel_familiar(fam_prc, q_vip_fam, q_vip_nov, vip_in_fam, vip_in_nov)

    return q_vip, vip_in


def merge_novel_familiar(fam_prc, q_vip_fam, q_vip_nov, vip_in_fam, vip_in_nov):
    q_vip = fam_prc * q_vip_fam + (1 - fam_prc) * q_vip_nov

    vip_in_fam *= fam_prc * q_vip_fam / q_vip
    vip_in_nov *= (1 - fam_prc) * q_vip_nov / q_vip
    vip_in = vip_in_fam + vip_in_nov

    return q_vip, vip_in
