#16qam
import numpy as np
import matplotlib.pyplot as plt


data_size = 160000
QAM16=[1+1j, 1+3j, 3+1j, 3+3j, 1-1j, 1-3j,3-1j,3-3j, -1+1j, -1+3j, -3+1j, -3+3j,-1-1j, -1-3j, -3-1j, -3-3j]/np.sqrt(10)

Q2B=[
[0,0,0,0],
[0,0,0,1],
[0,0,1,0],
[0,0,1,1],
[0,1,0,0],
[0,1,0,1],
[0,1,1,0],
[0,1,1,1],
[1,0,0,0],
[1,0,0,1],
[1,0,1,0],
[1,0,1,1],
[1,1,0,0],
[1,1,0,1],
[1,1,1,0],
[1,1,1,1]
]
ber=[]
snr = 18
for snr_db in range(snr):
    binary_data = np.random.randint(0,2,data_size)
    binary_data = binary_data.reshape(-1,4)                   #열의 개수는 4개 --> 4비트씩 묶음
    binary_to_10num = np.array([8,4,2,1]).reshape(4,1)        #10진수로 바꾸기
    tmp=np.dot(binary_data, binary_to_10num).transpose()
    sym_size = np.shape(tmp)[1]
    QAM16_sym = []


    for a in range(sym_size) :
        idx=tmp[0,a]
        QAM16_sym.append(QAM16[idx])


    noise_std = 10**(-snr_db/20)
    noise=np.random.randn(sym_size)*noise_std/np.sqrt(2) + 1j * np.random.randn(sym_size)*noise_std/np.sqrt(2)

    rcv_QAM16 =  QAM16_sym + noise
    rcv_binary=[]

    for a in range(sym_size):
        tmp_min=np.argmin(np.abs(QAM16-rcv_QAM16[a]))
        rcv_binary.append(Q2B[tmp_min][:])


    num_error=(np.sum(np.abs(rcv_binary-binary_data)))    #노이즈 개수
    tmp_ber=num_error/data_size
    ber.append(tmp_ber)

print(ber)
snr=range(18)
plt.semilogy(snr,ber)
plt.show()




#plt.scatter(rcv_QAM16.real , rcv_QAM16.imag)
#plt.show()