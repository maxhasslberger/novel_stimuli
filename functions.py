import numpy as np


def f_function(x):
    r = 3.0
    f = np.zeros(x.shape)

    f[x > 0] = r * x[x > 0]
    f[x > 1/r] = 1.0
    return f


def forward_euler(dx_dt, x0, dt):
    return x0 + dx_dt * dt


def cont_pulse_trials(mode, off_steps, stim_dur, inter_stim_dur, inter_trial_dur, trial_pulses, steps, dt):
    trial_dur = (stim_dur + inter_stim_dur) * trial_pulses - inter_stim_dur
    trial_steps = trial_dur / dt
    inter_trial_steps = inter_trial_dur / dt
    signal = np.zeros(off_steps)

    for i in range(int(np.ceil((steps - off_steps) / (trial_steps + inter_trial_steps)))):
        signal = np.append(signal, cont_rect_pulses(mode, stim_dur, inter_stim_dur, trial_steps, dt))  # one pulse
        signal = np.append(signal, np.zeros(int(inter_trial_steps)))

    return signal[0:steps]


def cont_rect_pulses(mode, stim_dur, inter_stim_dur, steps, dt):
    stim_steps = int(stim_dur / dt)
    inter_stim_steps = int(inter_stim_dur / dt)
    pulse_times = np.arange(0, steps-1, stim_steps + inter_stim_steps)
    if not mode:
        signal = pulse_train(np.arange(steps), pulse_times, rect(stim_steps))
    else:
        signal = pulse_train(np.arange(steps), pulse_times, ramp(steps))
    return signal


def rect(T):
    return lambda t: (0 <= t) & (t < T)


def ramp(a):
    return lambda t: t / a  # if (0 <= t < T) else 0


def pulse_train(t, pulse_times, fun):
    return np.sum(fun(t - pulse_times[:, np.newaxis]), axis=0)


def step_LIF(v_m, i_input, v_rest, v_th, tau, c_m, t_refr, t_refr_ref, dt):
    # spike determination
    spikes = np.zeros(v_m.shape[0], dtype=bool)
    spikes[v_m >= v_th] = True
    v_m[spikes] = v_rest[spikes]
    t_refr[spikes] = t_refr_ref[spikes] / dt + 1  # +1 subtracted in next section

    # refractory period update
    modify = t_refr <= 0
    t_refr[t_refr > 0] = t_refr[t_refr > 0] - 1

    # Execute LIF
    # v_m[modify] = v_m[modify] + (-(v_m[modify] - v_rest[modify]) * g_l[modify] + i_input[modify]) * (dt / c_m[modify])
    v_m[modify] = v_m[modify] + dt * (i_input[modify] / c_m[modify] + (v_rest[modify] - v_m[modify]) / tau[modify])

    return v_m, t_refr, spikes
