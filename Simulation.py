import numpy as np
import ConvCodec as cc

num_experiments = 5000  # 실험 횟수
bitsPerOnetime = 200  # 한 실험에서 생성할 비트 수
max_snr = 10
ber = []  # 비트 오류율을 저장할 리스트

# 8데시벨 값만 사용함.
for snr_db in range(max_snr, max_snr+1):
    totalErrors = 0  # 오류 발생 비트 수 누적시킬 변수.
    total_bits = num_experiments * bitsPerOnetime  # 총 1000000개 비트

    for a in range(num_experiments):  #5000번 반복
        data = np.random.randint(0, 2, bitsPerOnetime) # 0 or 1 200비트 생성

        encoded_bit = cc.Encoder(data) # 인코딩

        real_signal = encoded_bit[0, :] * 2 - 1  # -1과 1을 발생
        imag_signal = encoded_bit[1, :] * 2 - 1  # -1과 1을 발생

        qpsk_sym = (real_signal + 1j * imag_signal) / np.sqrt(2)
        ofdm_sym = np.fft.ifft(qpsk_sym) * np.sqrt(bitsPerOnetime)

        noise_std = 10 ** (-snr_db / 20)
        noise = np.random.randn(bitsPerOnetime + 3) * noise_std / np.sqrt(2) + 1j * np.random.randn(bitsPerOnetime + 3) * noise_std / np.sqrt(2)
        rcv_signal = np.fft.fft(ofdm_sym) / np.sqrt(bitsPerOnetime) + noise

        real_detected_signal = np.array(((rcv_signal.real > 0) + 0)).reshape(1, bitsPerOnetime + 3)
        imag_detected_signal = np.array(((rcv_signal.imag > 0) + 0)).reshape(1, bitsPerOnetime + 3)

        dec_input = np.vstack([real_detected_signal, imag_detected_signal])
        decoded_bit = cc.ViterbiDecoder(dec_input)

        totalErrors += np.sum(np.abs(data - decoded_bit))

    # ber
    ber_value = int(totalErrors / total_bits)
    ber.append(ber_value)

    print(f"SNR = {snr_db} dB, BER = {ber_value}")
