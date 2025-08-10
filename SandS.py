import numpy as np
import matplotlib.pyplot as plt

signal=[0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0]
f_sig=np.fft.fft(signal)
f_sig=np.fft.fftshift(f_sig)
plt.plot(np.abs(f_sig))
plt.show()
