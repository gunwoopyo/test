import numpy as np
import matplotlib.pyplot as plt

time_step = 0.002    #2/1000 초 간격   500개 샘플링
t = np.arange(0,1,time_step)
sig = 3*np.sin(2*np.pi*2*t) + 1*np.sin(2*np.pi*4*t)+2*np.sin(2*np.pi*6*t)   #주파수 2 주파수4  주파수6
plt.subplot(911)   #9행 1열
plt.plot(t,sig)    #가로 t

carrier = np.cos(2*np.pi*100*t)    #주파수 100짜리 캐리어 신호
plt.subplot(912)    #9행1열 두번째
plt.plot(t,carrier)

am_sig = sig*carrier    #am신호 만듦
plt.subplot(913)
plt.plot(t,am_sig)


f_sig = np.fft.fftshift(np.fft.fft(sig))   #주파수 성분의 시그널
plt.plot(np.abs(f_sig))   #절대값
plt.subplot(914)
plt.plot(np.abs(f_sig))


f_am_sig=np.fft.fftshift(np.fft.fft(am_sig))
plt.subplot(915)
plt.plot(np.abs(f_am_sig))

dem_am_sig=am_sig*carrier
plt.subplot(916)
plt.plot(np.abs(dem_am_sig))

f_dem_am_sig=np.fft.fftshift(np.fft.fft(dem_am_sig))
plt.subplot(917)
plt.plot(np.abs(f_dem_am_sig))


lpf=np.zeros((500))
lpf[48:450]=1

lpf=np.zeros((500))
lpf[200:300]=1

lpf_f_Sig=f_dem_am_sig*lpf
plt.subplot(918)
plt.plot(np.abs(lpf_f_Sig))


dem_am_sig=np.fft.ifft(np.fft.fftshift(lpf_f_Sig))
plt.subplot(919)
plt.plot(dem_am_sig)




plt.show()



