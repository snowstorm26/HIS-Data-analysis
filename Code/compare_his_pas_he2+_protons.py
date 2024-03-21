import numpy as np
import matplotlib.pyplot as plt


pas_filename = 'speed data/1day sample alpha kinetic features.txt'

pas_proton_filename = 'speed data/2022-09-16_PAS_proton_data.txt'
pas_proton_filename2 = 'speed data/2022-09-17_PAS_proton_data.txt'

his_speed_fit = 'speed data/2022-09-16_HIS_He2+_speed_edge_corrected.txt'
doy_fit, speed_fit, v_thermal = np.loadtxt(his_speed_fit, delimiter=' ', skiprows=2, unpack=True)

his_speed_fit_2 = 'speed data/2022-09-17_HIS_He2+_speed_edge_corrected.txt'
doy_fit_2, speed_fit_2, v_thermal_2 = np.loadtxt(his_speed_fit_2, delimiter=' ', skiprows=2, unpack=True)





# get pas data
pas_data = np.genfromtxt(pas_filename, skip_header=2) 
daydec = np.array(pas_data[:, 0])
Nalpha = pas_data[:, 1]
VR = np.array(pas_data[:, 2])
VT = np.array(pas_data[:, 3])
VN = np.array(pas_data[:, 4])

Tpar = pas_data[:, 5]
Tperp = pas_data[:, 6]
speed_he2_pas = np.sqrt(VR**2 + VT**2 + VN**2)

# add offset to get doy
daydec = daydec + 243


# get pas proton data
doy_p, v_p, vr_p, vt_p, vn_p,  = np.loadtxt(pas_proton_filename, delimiter=' ', skiprows=3, unpack=True)
doy_p2, v_p2, vr_p2, vt_p2, vn_p2,  = np.loadtxt(pas_proton_filename2, delimiter=' ', skiprows=3, unpack=True)

append_2nd_date = 1 # change to 1 to append 17/9/2022 date, 0 for only the 16/9/2022
if append_2nd_date == True:
   
    doy_p = np.append(doy_p, doy_p2)
    v_p = np.append(v_p, v_p2)
    vr_p = np.append(vr_p, vr_p2)
    vt_p = np.append(vt_p, vt_p2)
    vn_p = np.append(vn_p, vn_p2)
    
    speed_fit = np.append(speed_fit, speed_fit_2)
    doy_fit = np.append(doy_fit, doy_fit_2)

# resample VR (PAS)
VR_reshaped = VR[1:-5].reshape(-1,2)
VR_rescaled = np.mean(VR_reshaped, axis=1)

speed_he2_pas_reshaped = speed_he2_pas[1:-5].reshape(-1,2)
speed_he2_pas_rescaled = np.mean(speed_he2_pas_reshaped, axis=1)

daydec_reshaped = daydec[1:-5].reshape(-1,2)
daydec_rescaled = np.mean(daydec_reshaped, axis=1)

plt_start = 259.5
plt_end = 260.5 #doy_fit[-1] # 260.0

# there is an offset for the doy

offset = 0.042
doy_fit -= offset




alpha = 0.8
plt.figure(figsize=(8,5))

plt.xlim((plt_start, plt_end))
plt.plot(doy_fit,speed_fit, color='darkviolet', label=r'$\langle$v$\rangle$ He$^{2+}$ (HIS)', zorder=4, alpha=alpha)
plt.plot(daydec_rescaled,speed_he2_pas_rescaled, label=r"$\langle$v$\rangle$ He$^{2+}$ (PAS)", color='seagreen', zorder=3, alpha=alpha)
plt.plot(doy_p, v_p, color='tab:blue', label=r'$\langle$v$\rangle$ H$^{+}$ (PAS)', zorder=5, alpha=alpha)
plt.xlabel("time [DOY 2022]")
plt.ylabel("speed [km/s]")
plt.legend(loc='upper right')
plt.ylim((500,900))
plt.yticks([500, 600, 700, 800, 900])
# plt.title("Speed comparison HIS/PAS")
# plt.savefig('python plots/speed_comparison_he2_his_pas.png', dpi=400)
# plt.savefig('python plots/speed_comparison_he2_protons.png', dpi=400)


