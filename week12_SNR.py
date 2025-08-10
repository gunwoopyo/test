import numpy as np
import matplotlib.pyplot as plt

# monte carlo 시뮬레이션
size = 1000000
sig = np.exp(1j*np.pi/4)
print(sig)
sig=np.sqrt(100)*sig   #기존신호에서 파워를 10으로 만듦 --> 루트10을 곱했음


real_noise = np.random.randn(size)
imag_noise = np.random.randn(size)
awgn=(real_noise+imag_noise*1j)/np.sqrt(2)
rcv_sig=sig*awgn
plt.figure(figsize=(3,3))
plt.plot(awgn.real, awgn.imag,'.')
plt.show()
print(np.mean(rcv_sig.real))
print(np.mean(rcv_sig.imag))

sig_power = np.abs(sig.real)**2
noise_power=np.mean(np.abs(awgn)**2)
print("snr : ", 10*np.log10(sig_power/noise_power),"db")
#파워가 1짜리인 신호 생성
'''
size = 10000
sig = np.exp(1j*np.pi/4)
print(sig)
angle = np.random.rand(size)*2*np.pi
sig=np.exp(1j*angle)
plt.figure(figsize=(3,3))
plt.plot(sig.real, sig.imag,'.')
plt.show()
'''

