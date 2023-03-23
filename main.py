from neuron_model_sim import exe_neuron_model
from rate_model_sim import exe_wilson_cowan, create_video_from_plots


# exe_neuron_model()
# exe_wilson_cowan()

mode = 4
# n_subtypes = 4
#
# dt = 0.05 * 1e-3  # s
# t_ges = 10  # s
# steps = int(np.ceil(t_ges / dt))

vid_frames = 10
rates = []

for i in range(vid_frames):
    # rates[i, :, :], _, _, _ = exe_wilson_cowan(mode, i / vid_frames, dt)
    rates.append(exe_wilson_cowan(mode, i / (vid_frames - 1)))

create_video_from_plots(rates, 'D:\\riseo\\Documents\\TUM\\2 Sem\\Julijana\\Plots\\NovelP_vids\\omission_vip.mp4', 1)
