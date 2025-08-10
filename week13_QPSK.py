import numpy as np
import matplotlib.pyplot as plt

data_size = 1000000
max_snr = 13    #13db 10^-6승
#신호대잡음비가 늘어남
ber = []
#qpsk는 신호가 두 가지로 나눠서 감
for snr_db in range(0,max_snr): #노이즈 조절   # 0~12db ,
    real_sig = np.random.randint(0, 2, data_size) * 2 - 1
    imag_sig = np.random.randint(0, 2, data_size) * 2 - 1

    qpsk_sym = (real_sig + 1j * imag_sig) / np.sqrt(2)  # power가 1인 신호
    noise_std = 10 ** (-snr_db / 20)
    # 노이즈는 실수와 허수로 구분 해줘야함
    real_noise = np.random.randn(data_size) * noise_std    # std를 곱했으니 snr에 맞게  노이즈 생성함
    imag_noise  = np.random.randn(data_size) * noise_std #sig쪽에 섞이는 노잊
    noise=(real_noise+1j*imag_noise)/np.sqrt(2)
    rcv_sig=qpsk_sym+noise
    real_rcv_data=(rcv_sig.real>0+0)*2-1
    imag_rcv_data = (rcv_sig.imag > 0 + 0) * 2 - 1
    real_err_num=np.sum(np.abs(real_rcv_data-real_sig)/2)
    imag_err_num = np.sum(np.abs(imag_rcv_data - imag_sig) / 2)
    num_err=real_err_num+imag_err_num
    ber.append(num_err/(data_size*2))  #총 데이터 수는 2백만개



snr= np.arange(0,max_snr)
plt.semilogy(snr,ber)
plt.show()





#신호대잡음비를 설정(선호) --> 신호는 1로 고정하고 상대적으로 노이즈방법을 줄임

#물리적으로는  --> 노이즈는 고정이고 신호를 조절




