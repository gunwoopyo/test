import matplotlib.pyplot as plt
import numpy as np

from  scipy.stats import norm   #정규분포, normal distribution

mu = 0  # 평균
sigma = 1   # 표준편차

num_sample = 10000000  #랜덤 변수 생성


sample = np.random.normal(mu, sigma, num_sample)   #랜덤 변수들을 생성
#평균 mu , 표준편차 sigma인 정규 분포를 따르는 랜덤 변수 num_sample 개를 생성하시오.

# 이 랜덤 변수는 정말 정규분포를 따르고 있을까?
cnt, bins, ignored = plt.hist(sample,100, density=True)  #수식 없이 pdf를 그리는 방법

plt.plot(bins, 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-((bins-mu)**2)/(2*sigma**2)))
# Normal distribution의 pdf(확률밀도함수) 를 그려본 것.
#plt.show()

#PDF   x=1인 확률 구하기
print(1/(np.sqrt(2*np.pi)*sigma)*np.exp(-((1-mu)**2)/(2*sigma**2)))

#PDF   x=5인 확률 구하기
print(1/(np.sqrt(2*np.pi)*sigma)*np.exp(-((5-mu)**2)/(2*sigma**2)))

#CDF의 계산
print(norm.cdf(0))   #수학적 계산
print(norm.cdf(2))
print(norm.cdf(10))

print(np.sum(sample<2)/num_sample)   # 실험적 계산