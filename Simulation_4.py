import numpy as np
import ConvCodec as cc

data_size = 1000000  # 1024개의 data bit
max_snr = 10  # 최대 SNR 10db까지 실행
ber = []

for snr_db in range(max_snr-1, max_snr):
    data = np.random.randint(0, 2, data_size)  # 0 or 1 데이터 1024개 (1024, )
    encoded_bit = cc.Encoder(data)  # (2,1024 + 3)   2행 1024열 0과 1로 구성된 데이터

    real_signal = encoded_bit[0, :] * 2 - 1  # -1과 1을 발생
    imag_signal = encoded_bit[1, :] * 2 - 1  # -1과 1을 발생

    qpsk_sym = (real_signal + 1j * imag_signal) / np.sqrt(2)
    ofdm_sym = np.fft.ifft(qpsk_sym) * np.sqrt(data_size)  # 평균파워 1이 되도록

    # 노이즈 섞기
    noise_std = 10 ** (-snr_db / 20)
    noise = np.random.randn(data_size+3) * noise_std / np.sqrt(2) + 1j * np.random.randn(data_size+3) * noise_std / np.sqrt(2)
    rcv_signal = np.fft.fft(ofdm_sym) / np.sqrt(data_size) + noise
    # 크기가 맞아짐    # np.fft.fft(ofdm_sym)의 크기와 noise의 크기를 맞춤

    real_detected_signal = np.array(((rcv_signal.real > 0) + 0)).reshape(1, data_size+3)
    imag_detected_signal = np.array(((rcv_signal.imag > 0) + 0)).reshape(1, data_size+3)


    dec_input = np.vstack([real_detected_signal, imag_detected_signal])
    # np.shape(dec_input)

    decoded_bit = cc.ViterbiDecoder(dec_input)

    print(np.sum(np.abs(dec_input - encoded_bit)))   # 오류 정정 전 결과
    print(np.sum(np.abs(data - decoded_bit)))        # 오류 정정 후 결과
