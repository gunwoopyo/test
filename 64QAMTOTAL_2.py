import numpy as np
import matplotlib.pyplot as plt

# Parameters
DATA_SIZE_16QAM = 160000
DATA_SIZE_64QAM = 160002

# QAM Definitions
QAM16 = np.array([1+1j, 1+3j, 3+1j, 3+3j, 1-1j, 1-3j, 3-1j, 3-3j, -1+1j, -1+3j, -3+1j, -3+3j, -1-1j, -1-3j, -3-1j, -3-3j]) / np.sqrt(10)
QAM64 = np.array([
    -7-7j, -7-5j, -7-3j, -7-1j, -7+1j, -7+3j, -7+5j, -7+7j,
    -5-7j, -5-5j, -5-3j, -5-1j, -5+1j, -5+3j, -5+5j, -5+7j,
    -3-7j, -3-5j, -3-3j, -3-1j, -3+1j, -3+3j, -3+5j, -3+7j,
    -1-7j, -1-5j, -1-3j, -1-1j, -1+1j, -1+3j, -1+5j, -1+7j,
     1-7j,  1-5j,  1-3j,  1-1j,  1+1j,  1+3j,  1+5j,  1+7j,
     3-7j,  3-5j,  3-3j,  3-1j,  3+1j,  3+3j,  3+5j,  3+7j,
     5-7j,  5-5j,  5-3j,  5-1j,  5+1j,  5+3j,  5+5j,  5+7j,
     7-7j,  7-5j,  7-3j,  7-1j,  7+1j,  7+3j,  7+5j,  7+7j
]) / np.sqrt(42)

# Binary Mappings
Q2B_16QAM = [
    [0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1],
    [0,1,0,0], [0,1,0,1], [0,1,1,0], [0,1,1,1],
    [1,0,0,0], [1,0,0,1], [1,0,1,0], [1,0,1,1],
    [1,1,0,0], [1,1,0,1], [1,1,1,0], [1,1,1,1]
]

Q2B_64QAM = [[int(b) for b in f'{i:06b}'] for i in range(64)]

def simulate_qam(data_size, QAM, Q2B, bits_per_symbol, snr_max):
    ber = []
    for snr_db in range(snr_max):
        binary_data = np.random.randint(0, 2, data_size)
        binary_data = binary_data.reshape(-1, bits_per_symbol)
        binary_to_10num = np.array([2**i for i in range(bits_per_symbol - 1, -1, -1)]).reshape(bits_per_symbol, 1)
        symbols = np.dot(binary_data, binary_to_10num).flatten()
        transmitted_symbols = QAM[symbols]

        noise_std = 10**(-snr_db / 20)
        noise = (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols))) * noise_std / np.sqrt(2)
        received_symbols = transmitted_symbols + noise

        detected_binary = [Q2B[np.argmin(np.abs(QAM - r))] for r in received_symbols]
        num_errors = np.sum(np.abs(np.array(detected_binary) - binary_data))
        ber.append(num_errors / data_size)

    return ber

# Simulation and Plot
ber_16qam = simulate_qam(DATA_SIZE_16QAM, QAM16, Q2B_16QAM, 4, 18)
ber_64qam = simulate_qam(DATA_SIZE_64QAM, QAM64, Q2B_64QAM, 6, 20)

plt.semilogy(range(18), ber_16qam, label='16-QAM')
plt.semilogy(range(20), ber_64qam, label='64-QAM')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('BER Performance of 16-QAM and 64-QAM')
plt.legend()
plt.grid(True, which="both")
plt.show()
