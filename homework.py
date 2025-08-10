import numpy as np
import matplotlib.pyplot as plt

time_step = 0.002
t=np.arange(0,1,time_step)

# 1번 그림 m(t)
sig = 3*np.sin(2*np.pi*2*t)+1*np.sin(2*np.pi*4*t)+2*np.sin(2*np.pi*6*t)
sig = sig/6
plt.subplot(911)
plt.plot(t,sig)

# 2번 그림 캐리어 신호
carrier = np.cos(2*np.pi*100*t)  #주파수 100cos
plt.subplot(912)
plt.plot(t,carrier)

# lc
lc = 1*np.cos(2*np.pi*100*t)   #주파수 100 cos

# 3번 그림 AM
am_sig= sig*carrier + lc
plt.subplot(913)
plt.plot(t,am_sig)

# 4번그림 - x(t) 의 FT
f_sig = np.fft.fftshift(np.fft.fft(sig))
plt.subplot(914)
plt.plot(np.abs(f_sig))



# 5번째 그림 - AM의 주파수 영역
f_am_sig = np.fft.fftshift(np.fft.fft(am_sig))
plt.subplot(915)
plt.plot(t,abs(f_am_sig))

# 6번째 그림 - AM의 절댓값
dem_am_sig=np.abs(am_sig)
plt.subplot(916)
plt.plot(t,dem_am_sig)

#복조 현상- AM 절댓값의 FT - 수신된 신호에 cos을 곱해준 거랑 같음. 주파수영역
f_dem_am_sig = np.fft.fftshift(np.fft.fft(dem_am_sig))
plt.subplot(917)
plt.plot(np.abs(f_dem_am_sig))


#lpf
lpf=np.zeros((500))
lpf[200:300] = 1

#8번째 그림 lpf*복조된 신호(f_dem_am_sig) in 주파수 영역. lpf 곱해서 원래 x(t)복원하기
lpf_f_sig =  f_dem_am_sig * lpf
plt.subplot(918)
plt.plot(np.abs(lpf_f_sig))

#9번째 그림 복조된 신호 Inverse FT
dem_am_sig=np.fft.ifft(np.fft.fftshift(lpf_f_sig))
plt.subplot(919)
plt.plot(dem_am_sig)
plt.show()




