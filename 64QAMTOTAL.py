import numpy as np
import matplotlib.pyplot as plt
data_size_16QAM = 160000
QAM16=[1+1j, 1+3j, 3+1j, 3+3j, 1-1j, 1-3j,3-1j,3-3j, -1+1j, -1+3j, -3+1j, -3+3j,-1-1j, -1-3j, -3-1j, -3-3j]/np.sqrt(10)

Q2B_16QAM=[
[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],
[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],
[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],
[1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]
]


data_size_64QAM = 160002
QAM64 = [
    -7-7j, -7-5j, -7-3j, -7-1j, -7+1j, -7+3j, -7+5j, -7+7j, -5-7j, -5-5j, -5-3j, -5-1j, -5+1j, -5+3j, -5+5j, -5+7j,
    -3-7j, -3-5j, -3-3j, -3-1j, -3+1j, -3+3j, -3+5j, -3+7j, -1-7j, -1-5j, -1-3j, -1-1j, -1+1j, -1+3j, -1+5j, -1+7j,
     1-7j,  1-5j,  1-3j,  1-1j,  1+1j,  1+3j,  1+5j,  1+7j, 3-7j,  3-5j,  3-3j,  3-1j,  3+1j,  3+3j,  3+5j,  3+7j,
     5-7j,  5-5j,  5-3j,  5-1j,  5+1j,  5+3j,  5+5j,  5+7j, 7-7j,  7-5j,  7-3j,  7-1j,  7+1j,  7+3j,  7+5j,  7+7j] / np.sqrt(42)



Q2B_64QAM = [
    [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1],
    [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1],
    [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 0], [0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 1, 1],
    [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 1, 1],
    [0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 0, 1], [0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 1, 1],
    [1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 1],
    [1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1],
    [1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 1, 0], [1, 0, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1], [1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 1],
    [1, 1, 0, 1, 0, 0], [1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 1, 0], [1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 1], [1, 1, 1, 0, 1, 0], [1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]
]

ber_16QAM=[]
snr_16QAM = 18

ber_64QAM = []
snr_64QAM = 20

#16QAM
for snr_db in range(snr_16QAM):
    binary_data_16QAM = np.random.randint(0, 2, data_size_16QAM)
    binary_data_16QAM = binary_data_16QAM.reshape(-1, 4)
    binary_to_10num_16QAM = np.array([8, 4, 2, 1]).reshape(4, 1)
    tmp_16QAM = np.dot(binary_data_16QAM, binary_to_10num_16QAM).transpose()
    sym_size_16QAM = np.shape(tmp_16QAM)[1]
    QAM16_sym = []

    for a in range(sym_size_16QAM):
        idx_16QAM = tmp_16QAM[0, a]
        QAM16_sym.append(QAM16[idx_16QAM])



    noise_std_16QAM = 10**(-snr_db/20)
    noise_16QAM=np.random.randn(sym_size_16QAM)*noise_std_16QAM/np.sqrt(2) + 1j * np.random.randn(sym_size_16QAM)*noise_std_16QAM/np.sqrt(2)

    rcv_QAM16 =  QAM16_sym + noise_16QAM
    rcv_binary_16QAM=[]

    for a in range(sym_size_16QAM):
        tmp_min_QAM16=np.argmin(np.abs(QAM16-rcv_QAM16[a]))
        rcv_binary_16QAM.append(Q2B_16QAM[tmp_min_QAM16][:])


    num_error_16QAM=(np.sum(np.abs(rcv_binary_16QAM-binary_data_16QAM)))    #노이즈 개수
    tmp_ber_16QAM=num_error_16QAM/data_size_16QAM
    ber_16QAM.append(tmp_ber_16QAM)









#64QAM
for snr_db in range(snr_64QAM):
    binary_data_64QAM = np.random.randint(0, 2, data_size_64QAM)
    binary_data_64QAM = binary_data_64QAM.reshape(-1, 6)
    binary_to_10num_64QAM = np.array([32, 16, 8, 4, 2, 1]).reshape(6, 1)
    tmp_64QAM = np.dot(binary_data_64QAM, binary_to_10num_64QAM).transpose()
    sym_size_64QAM = np.shape(tmp_64QAM)[1]
    QAM64_sym_64QAM = []

    for a in range(sym_size_64QAM):
        idx_QAM64 = tmp_64QAM[0, a]
        QAM64_sym_64QAM.append(QAM64[idx_QAM64])

    noise_std_QAM64 = 10**(-snr_db / 20)
    noise_QAM64 = np.random.randn(sym_size_64QAM) * noise_std_QAM64 / np.sqrt(2) + 1j * np.random.randn(sym_size_64QAM) * noise_std_QAM64 / np.sqrt(2)

    rcv_QAM64 = QAM64_sym_64QAM + noise_QAM64
    rcv_binary_QAM64 = []

    for a in range(sym_size_64QAM):
        tmp_min_QAM64 = np.argmin(np.abs(QAM64 - rcv_QAM64[a]))
        rcv_binary_QAM64.append(Q2B_64QAM[tmp_min_QAM64][:])

    num_error_QAM64 = np.sum(np.abs(np.array(rcv_binary_QAM64) - binary_data_64QAM))
    tmp_ber_QAM64 = num_error_QAM64 / data_size_64QAM
    ber_64QAM.append(tmp_ber_QAM64)

    # 16QAM
print(ber_16QAM)
snr_16QAM = range(snr_16QAM)
plt.semilogy(snr_16QAM, ber_16QAM)

print(ber_64QAM)
snr_64QAM = range(snr_64QAM)
plt.semilogy(snr_64QAM, ber_64QAM)
plt.show()

