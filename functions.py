import numpy as np


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
