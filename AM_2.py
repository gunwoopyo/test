import numpy as np
import matplotlib.pyplot as plt

time_step = 0.002
t=np.arange(0,1,time_step)
sig = 3*np.sin(2*np.pi*2*t)+1*np.sin(2*np.pi*4*t)+2*np.sin(2*np.pi*6*t)
plt.subplot(911)
plt.plot(t,sig)

carrier = np.cos(2*np.pi*100*t)
plt.subplot(912)
plt.plot(t,carrier)

am_sig=sig*carrier
plt.subplot(913)
plt.plot(t,am_sig) #주석

f_sig=np.fft.fftshift(np.fft.fft(sig))
plt.subplot(914)
plt.plot(np.abs(f_sig))

f_am_sig=np.fft.fftshift(np.fft.fft(am_sig))
plt.subplot(915)
plt.plot(t,abs(f_am_sig))

dem_am_sig=am_sig*carrier
plt.subplot(916)
plt.plot(t,dem_am_sig)

f_dem_am_sig=np.fft.fftshift(np.fft.fft(dem_am_sig))
plt.subplot(917)
plt.plot(np.abs(f_dem_am_sig))

lpf=np.zeros((500))
lpf[200:300]=1

lpf_f_sig=f_dem_am_sig*lpf
plt.subplot(918)
plt.plot(np.abs(lpf_f_sig))

dem_am_sig=np.fft.ifft(np.fft.fftshift(lpf_f_sig))
plt.subplot(919)
plt.plot(dem_am_sig)



plt.show()