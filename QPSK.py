import numpy as np
import matplotlib.pyplot as plt

data_size=1000000
max_snr=13
ber=[]

for snr_db in range(0,max_snr): # 0~12dB
    real_sig=np.random.randint(0,2,data_size) * 2 - 1
    imag_sig = np.random.randint(0, 2, data_size) * 2 - 1

qpsk_sym=(real_sig+1j*imag_sig)/np.sqrt(2) #power가 1인 신호
noise_std=10**(-snr_db/20)
real_noise=np.random.randn(data_size) * noise_std
imag_noise=np.random.randn(data_size) * noise_std
noise=(real_noise+1j*imag_noise)/np.sqrt(2)
rcv_sig=qpsk_sym+noise
real_rcv_data=(rcv_sig.real>0+0)*2-1
imag_rcv_data = (rcv_sig.imag > 0 + 0) * 2 - 1
real_err_num=np.sum(np.abs(real_rcv_data-real_sig)/2)
imag_err_num = np.sum(np.abs(imag_rcv_data - imag_sig) / 2)
num_err=real_err_num+imag_err_num
ber.append(num_err/(data_size*2))

snr=np.arange(0,max_snr)
plt.semilogy(snr,ber)
plt.show()
#plt.plot(rcv_sig.real,rcv_sig.imag,'o')
#plt.show()