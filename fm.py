import numpy as np
import matplotlib.pyplot as plt
step=0.001
t=np.arange(0,1,step)
fc=50
kf=10
#1 원 신호
signal=np.sin(2*np.pi*2*t)
plt.subplot(6,1,1)
plt.plot(t, signal)
int_sig=np.cumsum(signal)*step #누적합

#2 FM신호 생성
fm_sig=np.cos(2*np.pi*fc*t+2*np.pi*kf*int_sig)
plt.subplot(6,1,2)
plt.plot(t, fm_sig)

#3 FM신호 푸리에변환
f_fm_sig=np.fft.fftshift(np.fft.fft(fm_sig))
plt.subplot(6,1,3)
plt.plot(np.abs(f_fm_sig))
#4 미분된 FM
diff_fm_sig=np.abs(np.diff(fm_sig))
plt.subplot(6,1,4)
plt.plot(diff_fm_sig)

# 미분된 신호의 주파수 변환
f_diff_fm_sig=np.fft.fftshift(np.fft.fft(diff_fm_sig))
plt.subplot(6,1,5)
plt.plot(np.abs(f_diff_fm_sig))
#저역통과 필터 적용
lpf=np.zeros((999,))
lpf[480:520]=1
f_diff_fm_sig=f_diff_fm_sig*lpf
#필터링된 신호를 시간영역으로 변환
dem_sig=np.fft.ifft(np.fft.fftshift(f_diff_fm_sig))
plt.subplot(6,1,6)
plt.plot(np.abs(dem_sig))
plt.show()




#마지막으로 LPF를 사용한 후에 ifft 해서 시간 영역에서 복원된 신호를 그릴 때

#ifft 후 np.abs()를 해서 특정 값을 빼면 복원할 수 있을 겁니다.

#plt.subplot(5,1,1)
#plt.plot(np.abs(f_fm_sig))

#적분했으니까 sin의 적분은 -cos
#plt.subplot(5,1,2)
#plt.plot(t,fm_sig)

#plt.subplot(5,1,3)
#plt.plot(diff_fm_sig)

#plt.subplot(5,1,4)
#plt.plot(abs(f_diff_fm_sig))

#plt.subplot(5,1,5)
#plt.plot(dem_sig.real)







#plt.plot(np.abs(f_sig))



#미분  앞신호 -뒤신호
#d_fm_sig = np.diff(fm_sig)

#plt.plot(d_fm_sig)
#현재 dsb-lc
# 절댓값 취한다음 fft. 가운데 필터링
#숙제 - 미분된 신호로부터 원 신호를 복원.





