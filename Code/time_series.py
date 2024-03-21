import numpy as np
import matplotlib.pyplot as plt
import os

# load HIS alpha data
doy = []
speed = []
estimation_method = []
directory = "HIS speed timeseries data 2022/"
filenames = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
for file in filenames:
    path = directory + file
    doy_file, speed_file, estimation_method_file = np.loadtxt(path, delimiter=' ', skiprows=0, unpack=True)
    doy = np.append(doy, doy_file)
    speed = np.append(speed, speed_file)
    estimation_method = np.append(estimation_method, estimation_method_file)
    
# write his data into new txt file
quality_flag = np.zeros_like(doy)
bad_ind = np.where(estimation_method==2)
quality_flag[bad_ind] = 1

path = 'his_he2+_timeseries_2022'
formats = ['%.5f', '%.2f', '%.0f']
np.savetxt(path+'.txt', np.column_stack((doy, speed, quality_flag)), delimiter=' ', header='time,v,quality flag\ndoy km/s 0:good/1:close to edge\n', comments='', fmt=formats)




# load PAS proton data
doy_pas = []
speed_pas = []
directory = "speed data pas 2022/"
filenames = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
for file in filenames:
    path = directory + file
    doy_file,v_file,vr,vt,vn = np.loadtxt(path, delimiter=' ', skiprows=3, unpack=True)
    doy_pas = np.append(doy_pas, doy_file)
    speed_pas = np.append(speed_pas, v_file)




doy_pas = doy_pas[:-17]
speed_pas = speed_pas[:-17]

# resample pas data

doy_pas_resampl = np.mean(doy_pas[:-7].reshape(-1, 8), axis=1)
speed_pas_resampl = np.mean(speed_pas[:-7].reshape(-1, 8), axis=1)

# write pas data into new txt file
path = 'pas_proton_timeseries_2022'
formats = ['%.5f', '%.2f']
np.savetxt(path+'.txt', np.column_stack((doy_pas_resampl, speed_pas_resampl)), delimiter=' ', header='time,v\ndoy km/s\n', comments='', fmt=formats)

# load MAG data
path = 'MAG_timeseries_2022.txt'
doy_MAG, Br, Bt, Bn, B = np.loadtxt(path, delimiter=' ', skiprows=3, unpack=True)

doy_MAG = doy_MAG[:-1]
Br = Br[:-1]
Bt = Bt[:-1]
Bn = Bn[:-1]
B = B[:-1]

doy_start = 0
doy_stop = 370

# look at specific time interval

# doy_start = 180
# doy_stop = 240

# doy_start = 250
# doy_stop = 275

ind_start = np.argmin(np.abs(doy-doy_start))
ind_stop = np.argmin(np.abs(doy-doy_stop))

doy = doy[ind_start:ind_stop]
speed = speed[ind_start:ind_stop]
estimation_method = estimation_method[ind_start:ind_stop]

ind_pas_start = np.argmin(np.abs(doy_pas-doy_start))
ind_pas_stop = np.argmin(np.abs(doy_pas-doy_stop))

doy_pas = doy_pas[ind_pas_start:ind_pas_stop]
speed_pas = speed_pas[ind_pas_start:ind_pas_stop]

ind_pas_start_r = np.argmin(np.abs(doy_pas_resampl-doy_start))
ind_pas_stop_r = np.argmin(np.abs(doy_pas_resampl-doy_stop))

doy_pas_resampl = doy_pas_resampl[ind_pas_start_r:ind_pas_stop_r]
speed_pas_resampl = speed_pas_resampl[ind_pas_start_r:ind_pas_stop_r]

dt = doy[1:] - doy[:-1]
ind_data_gap = np.array(np.where(dt > 1))
ind_data_gap_end = ind_data_gap + 1
start_array = doy[ind_data_gap]
stop_array = doy[ind_data_gap_end]


# fig, ax = plt.subplots(1,1, figsize=(10,4))
# ax.plot(doy, speed, color='darkviolet', label=r'$\langle$v$\rangle$ He$^{2+}$ (HIS)', zorder=1, alpha=0.7)
# # ax.plot(doy_pas, speed_pas, color='tab:blue', label=r'$\langle$v$\rangle$ H$^{+}$ (PAS)', zorder=0)
# ax.plot(doy_pas_resampl, speed_pas_resampl, color='tab:blue', label=r'$\langle$v$\rangle$ H$^{+}$ (PAS)', zorder=0, alpha=0.7)
# # ax.set_title("HIS speed timeseries 2022")
# ax.set_xlabel("DOY 2022")
# ax.set_ylabel("speed [km/s]")

# ax.set_ylim(200, 1200)
# ax.legend(loc='upper right', markerscale=8.)
# for i in range(len(start_array[0])):
#     ax.axvspan(start_array[0][i], stop_array[0][i], color='gray', alpha=0.2, linestyle='')
# # fig.savefig("HIS_timeseries_2022.png", dpi=400)
# # fig.savefig("HIS_PAS_timeseries_2022.png", dpi=400)



# make a plot with the MAG data

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.suptitle("HIS He2+ timeseries 2022")
ax1.plot(doy, speed, color='darkviolet', label=r'$\langle$v$\rangle$ He$^{2+}$ (HIS)', zorder=1, alpha=0.7)
ax1.plot(doy_pas_resampl, speed_pas_resampl, color='tab:blue', label=r'$\langle$v$\rangle$ H$^{+}$ (PAS)', zorder=0, alpha=0.7)
ax2.plot(doy_MAG, B, color='black', label=r'|B|', alpha=0.7)
ax2.plot(doy_MAG, Br, color='tab:green', label=r'B_r', alpha=0.7)
ax2.plot(doy_MAG, Bt, color='tab:orange', label=r'B_t', alpha=0.7)
ax2.plot(doy_MAG, Bn, color='tab:gray', label=r'B_n', alpha=0.7)
ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax1.set_ylabel("speed [km/s]")
ax1.legend(loc='upper right', markerscale=8.)
ax2.set_ylabel("B [nT]")
ax2.set_xlabel("DOY 2022")
ax2.legend(loc='upper right', markerscale=8.)
