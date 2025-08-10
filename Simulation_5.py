import matplotlib.pyplot as plt
import numpy as np
import ConvCodec as cc
totalExperiments = 5000
bitsPerOnetime = 200
max_snr_db = 10
ber = []
totalBits = totalExperiments * bitsPerOnetime
totalErrors = 0
for snr_db in range(0, max_snr_db + 1):
    for a in range(totalExperiments):
        data = np.random.randint(0, 2, bitsPerOnetime)
        encoded_bit = cc.Encoder(data)

        real_signal = encoded_bit[0, :] * 2 - 1
        imag_signal = encoded_bit[1, :] * 2 - 1

        qpsk_sym = (real_signal + 1j * imag_signal) / np.sqrt(2)
        ofdm_sym = np.fft.ifft(qpsk_sym) * np.sqrt(bitsPerOnetime)

        noise_std = 10 ** (-snr_db / 20)
        noise = np.random.randn(bitsPerOnetime + 3) * noise_std / np.sqrt(2) + 1j * np.random.randn(bitsPerOnetime + 3) * noise_std / np.sqrt(2)
        rcv_signal = np.fft.fft(ofdm_sym) / np.sqrt(bitsPerOnetime) + noise

        real_detected_signal = np.array(((rcv_signal.real > 0) + 0)).reshape(1, bitsPerOnetime + 3)
        imag_detected_signal = np.array(((rcv_signal.imag > 0) + 0)).reshape(1, bitsPerOnetime + 3)

        dec_input = np.vstack([real_detected_signal, imag_detected_signal])
        decoded_bit = cc.ViterbiDecoder(dec_input)  # 디코딩
        totalErrors += np.sum(np.abs(data - decoded_bit))

    ber_value = totalErrors / totalBits
    ber.append(ber_value)
    print(f"SNR : {snr_db}dB, BER = {ber_value}")

snr = np.arange(0, max_snr_db + 1)
plt.semilogy(snr, ber)
plt.show()

