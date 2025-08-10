import numpy as np
import matplotlib.pyplot as plt
#신호대잡음비 데시벨
min=-10
max=10
step=0.01    #음수 1000개   양수 1000개 총 2000개
x=np.arange(min,max,step)
sig=np.sqrt(2)   #파워10짜리 신호
#3데시벨은 신호의 크기를 루트 2로 만들고,
#신호의 파워가 노이즈 파워의 2배
#이론 그래프
g_pr = 1/np.sqrt(np.pi*2)*np.exp(-((x-sig)**2/2)) #가우스 확률공식
plt.figure(figsize=(20,3))
plt.plot(x,g_pr)
plt.show()


print(x)
print(sig)   #3.16

#전체 확률운 1
sum_prob=np.sum(g_pr*step)
print(sum_prob)    #0.9999

theory_error_prob = np.sum(g_pr[:1000]*step)
print("이론BER : ",theory_error_prob)

size = 100000
awgn=np.random.randn(size) #randn이므로 노이즈파워1
rcv_sig = sig+awgn   #루트10으로 만들었으니 10데시벨임
num_err = np.sum(rcv_sig < 0 + 0)
bit_error_rate = num_err/size
print("실험 BER : ", bit_error_rate)