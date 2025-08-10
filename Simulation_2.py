import numpy as np
import ConvCodec as cc

data_size = 1000  #백만개 데이터 생성
max_snr = 5
ber = []

for snr_db in range(max_snr-1, max_snr):
    data = np.random.randint(0, 2, data_size)
    encoded_bit = cc.Encoder(data)

    real_signal = encoded_bit[0, :] * 2 - 1  # -1과 1을 발생
    imag_signal = encoded_bit[1, :] * 2 - 1  # -1과 1을 발생

    qpsk_sym = (real_signal + 1j * imag_signal) / np.sqrt(2)
    ofdm_sym = np.fft.ifft(qpsk_sym) * np.sqrt(data_size)

    noise_std = 10 ** (-snr_db / 20)
    noise = np.random.randn(data_size+3) * noise_std / np.sqrt(2) + 1j * np.random.randn(data_size+3) * noise_std / np.sqrt(2)
    rcv_signal = np.fft.fft(ofdm_sym) / np.sqrt(data_size) + noise

    real_detected_signal = np.array(((rcv_signal.real > 0) + 0)).reshape(1, data_size+3)
    imag_detected_signal = np.array(((rcv_signal.imag > 0) + 0)).reshape(1, data_size+3)

    dec_input = np.vstack([real_detected_signal, imag_detected_signal])
    decoded_bit = cc.ViterbiDecoder(dec_input)

    print(np.sum(np.abs(dec_input - encoded_bit)))   # 오류 정정 전 결과
    print(np.sum(np.abs(data - decoded_bit)))        # 오류 정정 후 결과
