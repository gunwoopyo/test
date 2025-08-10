# 16 QAM

import numpy as np
import matplotlib.pyplot as plt

data_size = 16

QAM_16 = [1+1j, 1+3j, 3+1j , 3+3j, #1사분면
          1-1j, 1-3j, 3-1j, 3-3j,  #4사분면
          -1+1j, -1+3j,-3+1j,-3+3j, #2
          -1-1j, -1-3j, -3-1j, -3-3j]  / np.sqrt(10)  #신호의 종류   비트정보와 매핑 일대일로.  3사분면

QAM_to_bit = [
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

print(np.sum(np.abs(QAM_16)**2/16))

snr_db=5
binary_data = np.random.randint(0,2,data_size)  #0101로 1600개
binary_data=binary_data.reshape(-1,4)
binary_to_10num =np.array([8,4,2,1]).reshape(4,1)
tmp = np.dot(binary_data,binary_to_10num).transpose()
symbol_size=np.shape(tmp)[1]

sym=[]

for a in range(symbol_size):
    idx=tmp[0,a]
    sym.append(QAM_16[idx])

noise_std=10**(-snr_db/20)
real_noise = np.random.randn(symbol_size) * noise_std  # std를 곱했으니 snr에 맞게  노이즈 생성함
imag_noise = np.random.randn(symbol_size)  # sig쪽에 섞이는 노이즈 섞임
noise=(real_noise+1j*imag_noise)/np.sqrt(2)

rcv_sig=sym+noise
for a in range(symbol_size):
    tmp_min=np.argmin(np.abs(QAM_16-rcv_sig[a]))
    rcv_sig.append(QAM_to_bit[tmp_min][:])

#print(np.shape(rcv_data))




print(binary_data)
print(symbol_size)

print()



#print(tmp)




