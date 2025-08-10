import numpy as np
import matplotlib.pyplot as plt
data_size = 100002
QAM = [
    -7-7j, -7-5j, -7-3j, -7-1j, -7+1j, -7+3j, -7+5j, -7+7j, -5-7j, -5-5j, -5-3j, -5-1j, -5+1j, -5+3j, -5+5j, -5+7j,
    -3-7j, -3-5j, -3-3j, -3-1j, -3+1j, -3+3j, -3+5j, -3+7j, -1-7j, -1-5j, -1-3j, -1-1j, -1+1j, -1+3j, -1+5j, -1+7j,
     1-7j,  1-5j,  1-3j,  1-1j,  1+1j,  1+3j,  1+5j,  1+7j, 3-7j,  3-5j,  3-3j,  3-1j,  3+1j,  3+3j,  3+5j,  3+7j,
     5-7j,  5-5j,  5-3j,  5-1j,  5+1j,  5+3j,  5+5j,  5+7j, 7-7j,  7-5j,  7-3j,  7-1j,  7+1j,  7+3j,  7+5j,  7+7j] / np.sqrt(42)

Q2B = [
    [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1], [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1],
    [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 0], [0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 1, 1],
    [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 1, 1], [0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 0, 1], [0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 1, 1],
    [1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 1],[1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1],
    [1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 1, 0], [1, 0, 1, 1, 1, 1],[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1], [1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 1],
    [1, 1, 0, 1, 0, 0], [1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 1, 0], [1, 1, 0, 1, 1, 1],[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 1], [1, 1, 1, 0, 1, 0], [1, 1, 1, 0, 1, 1],
[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]
]
ber = []
snr = 30

for snr_db in range(snr):
    binary_data = np.random.randint(0, 2, data_size)
    binary_data = binary_data.reshape(-1, 6)
    binary_to_10num = np.array([32, 16, 8, 4, 2, 1]).reshape(6, 1)
    tmp = np.dot(binary_data, binary_to_10num).transpose()
    sym_size = np.shape(tmp)[1]
    QAM64_sym = []

    for a in range(sym_size):
        idx = tmp[0, a]
        QAM64_sym.append(QAM[idx])

    noise_std = 10**(-snr_db / 20)
    noise = np.random.randn(sym_size) * noise_std / np.sqrt(2) + 1j * np.random.randn(sym_size) * noise_std / np.sqrt(2)

    rcv_QAM64 = QAM64_sym + noise
    rcv_binary = []

    for a in range(sym_size):
        tmp_min = np.argmin(np.abs(QAM - rcv_QAM64[a]))
        rcv_binary.append(Q2B[tmp_min][:])

    num_error = np.sum(np.abs(np.array(rcv_binary) - binary_data))
    tmp_ber = num_error / data_size
    ber.append(tmp_ber)

print(ber)
snr = range(20)
plt.semilogy(snr, ber)
plt.show()