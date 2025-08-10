import matplotlib.pyplot as plt
import numpy as np

data_size = 1000000
max_snr = 10
ber = [] # bit error rate

for snr_db in range(0,max_snr):
    signal = np.random.randint(0,2,data_size)*2-1
    noise_std = 10**(-snr_db/20)
    noise = np.random.randn(data_size)*noise_std /np.sqrt(2)
    rcv_signal = signal + noise
    detected_signal=((rcv_signal>0)+0)*2-1
    num_error =np.sum(np.abs(detected_signal - signal)) / 2
    ber.append(num_error / data_size)

snr = np.arange(0,max_snr)
plt.semilogy(snr,ber)
plt.show()





