# from neuron_model_sim import exe_neuron_model
from rate_model_sim import exe_wilson_cowan, create_video_from_plots

import numpy as np


# exe_neuron_model()
# exe_wilson_cowan()

mode = 4
ylims = [[-0.05, 1.0], [-0.05, 0.3], [-0.05, 1.0], [-0.05, 0.3]]  # mode 4 ylims
# ylims = [[-0.05, 1.0], [-0.05, 0.4], [-0.05, 1.0], [-0.05, 0.5]]  # mode 5 ylims

vid_frames = 101
clip_dur = 10  # s
fps = int(vid_frames / clip_dur)

rates = []
extra_frames = int(vid_frames/10)

# lower initial step size
for i in range(0, extra_frames):
    nov_plus = i / (vid_frames - 1) / extra_frames
    rates.append(exe_wilson_cowan(mode=mode, nov_plus=nov_plus, ylims=ylims))
    print("Simulation " + "%.1f" % (nov_plus*100) + "% done.")

# Middle part
for i in range(1, vid_frames-1):
    nov_plus = i / (vid_frames - 1)
    rates.append(exe_wilson_cowan(mode=mode, nov_plus=nov_plus, ylims=ylims))
    print("Simulation " + "%.1f" % (nov_plus*100) + "% done.")

# lower step size at the end
last_orig = (vid_frames - 2) / (vid_frames - 1)
for i in range(1, extra_frames+1):
    nov_plus = last_orig + i / (vid_frames - 1) / extra_frames
    rates.append(exe_wilson_cowan(mode=mode, nov_plus=nov_plus, ylims=ylims))
    print("Simulation " + "%.1f" % (nov_plus*100) + "% done.")

rates = np.array(rates)
rates = rates.T

for i in range(rates.shape[0]):
    create_video_from_plots(rates[i, :], "D:\\riseo\\Documents\\TUM\\2 Sem\\Julijana\\Plots\\NovelP_vids\\"
                            + str(fps) + "FR_" + str(vid_frames) + "F_#" + str(mode) + str(i) + ".mp4", fps)
