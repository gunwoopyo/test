import numpy as np
import matplotlib.pyplot as plt

size = 1000000
real_noise = np.random.randn(size)       #노이즈파워 1인 노이즈 발생시켯음.   // 노이즈 발생시키기 -- 실수 노이즈와 허수 노이즈 둘다 만들기 -- cos sin 필터
noise_power = np.mean(np.abs(real_noise)**2)     #복소수 신호의  절댓값 후 제곱의 평균 == 노이즈 파워
print(noise_power)
print(np.mean(real_noise))   # 평균은 0, 파워1인 노이즈 생성

imag_noise = np.random.randn(size)
awgn=(real_noise+imag_noise*1j)/np.sqrt(2)   #복수수의 파워가 1인 awgn 생성.
print(awgn)

awgn_power = np.mean(np.abs(awgn)**2)  #awgn의 파워
print(awgn_power)
plt.figure(figsize=(3,3))
plt.plot(awgn.real, awgn.imag,'.')
plt.show()
#실수노이즈와 허수노이즈의 파워
