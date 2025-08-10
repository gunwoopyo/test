import matplotlib.pyplot as plt
import numpy as np

data_size = 500
max_snr = 15
ber = []

for snr_db in range(max_snr-1, max_snr):
    real_signal = np.random.randint(0, 2, data_size) * 2 - 1
    imag_signal = np.random.randint(0, 2, data_size) * 2 - 1

    qpsk_sym = (real_signal+1j*imag_signal)/np.sqrt(2)
    noise_std = 10**(-snr_db/20)
    noise = np.random.randn(data_size)*noise_std/np.sqrt(2) + 1j*np.random.randn(data_size)*noise_std/np.sqrt(2)
    rcv_signal = qpsk_sym + noise

    real_detected_signal = ((rcv_signal.real>0)+0)*2 - 1
    imag_detected_signal = ((rcv_signal.imag>0)+0)*2 - 1
    num_error = np.sum(np.abs(real_signal-real_detected_signal))/2 + np.sum(np.abs(imag_signal-imag_detected_signal))/2
    ber.append(num_error/(data_size*2))

plt.scatter(rcv_signal.real, rcv_signal.imag)
plt.show()


