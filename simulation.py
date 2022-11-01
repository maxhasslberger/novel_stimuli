from functions import step_LIF

import numpy as np


# TODO: plasticity adjustments, normalize weights

def run_sim(n_neurons, n_ext, dt, t_ges):

    #########################################################################################
    # Init
    #########################################################################################

    # membrane dynamics
    tau_init = [20 * 1e-3, 20 * 1e-3]  # membrane time constant s [inh. neuron, exc. neuron] type
    C_init = [300 * 1e-9, 300 * 1e-9]  # membrane capacitance F
    v_rest_init = [-70 * 1e-3, -62 * 1e-3]  # resting potential V [inh. neuron, exc. neuron] type
    v_th_init = [20 * 1e-3, 10 * 1e-3]  # cutoff for voltage. when crossed, record a spike and reset V
    # v_th_add = [10 * 1e-3, 10 * 1e-3]  # increase in threshold post spike V
    t_refr_ref_init = [1 * 1e-3, 1 * 1e-3]  # absolute refractory period s

    n_ges = sum(n_neurons)
    weights_init = np.array([[-1.0, -48.7], [1.0, 1.78]])

    ext_input_rates_init = [2.5 * 1e3, 4.5 * 1e3]  # external input rate (Hz) to [inh. neuron, exc. neuron] type
    ext_connectivity_init = [1.27, 1.78]  # external input weights to [inh. neuron, exc. neuron] type

    # Init different neuron type params
    weights = np.zeros((n_ges, n_ges))  # weight matrix
    next_ext = np.zeros(n_ges)  # Next external input time step
    ext_input_rates = np.zeros(n_ges)
    ext_connectivity = np.zeros(n_ges)

    tau = np.zeros(n_ges)
    C = np.zeros(n_ges)
    v_rest = np.zeros(n_ges)
    v_th = np.zeros(n_ges)
    t_refr_ref = np.zeros(n_ges)

    ext_input_neurons = np.zeros(n_ges, dtype=bool)  # selected input neurons for external stimulus

    n_neuron_types = n_neurons.shape[0]
    n_types_i = np.append(0, n_neurons)
    for i in range(n_neuron_types):
        arg = np.arange(n_types_i[i], n_types_i[1:i + 2].sum())

        tau[arg] = tau_init[i]
        C[arg] = C_init[i]
        v_rest[arg] = v_rest_init[i]
        v_th[arg] = v_th_init[i]
        t_refr_ref[arg] = t_refr_ref_init[i]

        ext_connectivity[arg] = ext_connectivity_init[i]
        ext_input_rates[arg] = ext_input_rates_init[i]
        next_ext[arg] = 1.0 / ext_input_rates[arg]  # add random offset

        tmp = np.copy(arg)
        np.random.shuffle(tmp)
        ext_input_neurons[tmp[:n_ext[i]]] = True  # set n random neurons from type i to external input neurons

        for j in range(n_neuron_types):
            weights[arg, n_types_i[j]:n_types_i[1:j + 2].sum()] = weights_init[i, j] * \
                                                          (1 + np.random.rand(n_neurons[i], n_neurons[j]))

    curr_neuron_inputs = np.zeros(n_ges)  # summed up inputs at current time step
    t_refr = np.zeros(n_ges)  # current absolute refractory period

    #########################################################################################
    # Simulation
    #########################################################################################

    n_steps = round(t_ges / dt)

    v_m = np.zeros((n_steps+1, n_ges))
    v_m[0, :] = v_rest + (v_th - v_rest) * np.random.rand(n_ges)  # initial membrane potential
    spikes = np.zeros((n_steps+1, n_ges), dtype=bool)  # spike of neuron during last step?

    for i in range(n_steps):
        timestep = i * dt
        curr_neuron_inputs[:] = 0

        # Determine and update external input
        input_now = next_ext >= timestep
        if sum(input_now) > 0:
            a = 1
        next_ext[input_now] += 1.0 / ext_input_rates[input_now]  # add random offset  # update next input step

        # Sum up input on each neuron
        curr_neuron_inputs[ext_input_neurons] += ext_connectivity[ext_input_neurons] * input_now[ext_input_neurons]
        curr_neuron_inputs += np.sum(weights[spikes[i, :], :], 0)

        # Update LIF model
        [v_m[i+1, :], t_refr, spikes[i+1, :]] = \
            step_LIF(v_m[i, :], curr_neuron_inputs, v_rest, v_th, tau, C, t_refr, t_refr_ref, dt)

    return v_m, spikes, ext_input_neurons
