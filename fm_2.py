import numpy as np
import matplotlib.pyplot as plt

step=0.001
t=np.arange(0,1,step)
fc=50
kf=10

signal=np.sin(2*np.pi*2*t)
int_sig=np.cumsum(signal)*step #누적합

fm_sig=np.cos(2*np.pi*fc*t+2*np.pi*kf*int_sig)

f_fm_sig=np.fft.fftshift(np.fft.fft(fm_sig))

diff_fm_sig=np.abs(np.diff(fm_sig))

f_diff_fm_sig=np.fft.fftshift(np.fft.fft(diff_fm_sig))



lpf=np.zeros((999,))
lpf[480:520]=1
f_diff_fm_sig=f_diff_fm_sig*lpf

dem_sig=np.fft.ifft(np.fft.fftshift(f_diff_fm_sig))

plt.subplot(5,1,1)
plt.plot(np.abs(f_fm_sig))

#적분했으니까 sin의 적분은 -cos
plt.subplot(5,1,2)
plt.plot(t,fm_sig)

plt.subplot(5,1,3)
plt.plot(diff_fm_sig)

plt.subplot(5,1,4)
plt.plot(abs(f_diff_fm_sig))

plt.subplot(5,1,5)
plt.plot(dem_sig.real)


plt.show()