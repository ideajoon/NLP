
written by ideajoon<br/>
※ 참고 : 딥 러닝을 이용한 자연어 처리 입문 (https://wikidocs.net/book/2155) 자료를 공부하고 정리함

# 07. 머신 러닝(Machine Learning) 개요

## 목차
1. 머신 러닝이란(What is Machine Learning?)
2. 머신 러닝 훑어보기
3. 선형 회귀(Linear Regression)
4. 로지스틱 회귀(Logistic Regression) - 이진 분류
5. 다중 입력에 대한 실습
6. 벡터와 행렬 연산
7. 소프트맥스 회귀(Softmax Regression) - 다중 클래스 분류

## 1. 머신 러닝이란(What is Machine Learning?)

### 1) 머신 러닝(Machine Learning)이 아닌 접근 방법의 한계

### 2) 머신 러닝은 기존 프로그래밍의 한계에 대한 해결책이 될 수 있다

## 2. 머신 러닝 훑어보기

### 1) 머신 러닝 모델의 평가

![](https://wikidocs.net/images/page/24987/%EB%8D%B0%EC%9D%B4%ED%84%B0.PNG)

검증용 데이터는 모델의 성능을 평가하기 위한 용도가 아니라, 모델의 성능을 조정하기 위한 용도입니다. 더 정확히는 과적합이 되고 있는지 판단하거나 하이퍼파라미터의 조정을 위한 용도입니다.

가중치와 편향과 같은 학습을 통해 바뀌어져가는 변수를 이 책에서는 매개변수라고 부릅니다.

하이퍼파라미터는 보통 사용자가 직접 정해줄 수 있는 변수라는 점입니다. <br/>
경사 하강법에서 학습률(learning rate)이 이에 해당되며 딥 러닝에서는 은닉층의 수, 뉴런의 수, 드롭아웃 비율 등이 이에 해당됩니다.

 매개변수는 사용자가 결정해주는 값이 아니라 모델이 학습하는 과정에서 얻어지는 값입니다. 

만약, 검증 데이터와 테스트 데이터를 나눌 만큼 데이터가 충분하지 않다면 k-폴드 교차 검증이라는 또 다른 방법을 사용하기도 합니다.

### 2) 분류(Classification)와 회귀(Regression)

1. 분류(Classification)
  - 이진 분류(Binary Classification)
  - 다중 클래스 분류(Multi-Class Classification)
  - 다중 레이블 분류(Multi-lable Classification)
2. 회귀(Regression)
  - 선형 회귀(Lineare Regression)
  - 로지스틱 회귀(Logistic Rgression)

회귀 문제는 분류 문제처럼 0 또는 1이나 과학 책장, IT 책장 등과 같이 분리된(비연속적인) 답이 결과가 아니라 연속된 값을 결과로 가집니다. 예를 들어 시험 성적을 예측하는데 5시간 공부하였을 때 80점, 5시간 1분 공부하였을 때는 80.5점, 7시간 공부하였을 때는 90점 등이 나오는 것과 같은 문제가 있습니다. 그 외에도 시계열 데이터를 이용한 주가 예측, 생산량 예측, 지수 예측 등이 이에 속합니다.

### 3) 지도 학습(Supervised Learning)과 비지도 학습(Unsupervised Learning)

#### (1) 지도 학습

지도 학습이란 레이블(Label)이라는 정답과 함께 학습하는 것을 말합니다. 

#### (2) 비지도 학습

비지도 학습은 레이블이 없이 학습하는 것을 말합니다.

예를 들어 토픽 모델링의 LDA는 비지도 학습에 속하며, 뒤에서 배우게 되는 워드투벡터(Word2Vec)는 마치 지도 학습을 닮았지만, 비지도 학습에 속합니다.

### 4) 샘플(Sample)과 특성(Feature)

![](https://wikidocs.net/images/page/35821/n_x_m.PNG)

### 5) 혼동 행렬(Confusion Matrix)

혼동 행렬(Confusion Matrix)

|구분|참(Positive)|거짓(Negative)|
|---|---|---|
|참|TP|FN|
|거짓|FP|TN|

이를 각각 TP(True Positive), TN(True Negative), FP(False Postivie), FN(False Negative)라고 하는데 True는 정답을 맞춘 경우고 False는 정답을 맞추지 못한 경우입니다. 

그리고 Positive와 Negative는 각각 제시했던 정답입니다.

즉, TP는 양성(Postive)이라고 대답하였고 실제로 양성이라서 정답을 맞춘 경우입니다. TN은 음성(Negative)이라고 대답하였는데 실제로 음성이라서 정답을 맞춘 경우입니다.

#### (1) 정밀도(Precision)

정밀도은 양성이라고 대답한 전체 케이스에 대한 TP의 비율입니다.

$정밀도 = \frac{TP}{TP + FP}$

#### (2) 재현률(Recall)

재현률은 실제값이 양성인 데이터의 전체 개수에 대해서 TP의 비율입니다. 즉, 양성인 데이터 중에서 얼마나 양성인지를 예측(재현)했는지를 나타냅니다.

$재현률 = \frac{TP}{TP + FN}$

### 6) 과적합(Overfitting)과 과소 적합(Underfitting)

- 과적합(Overfitting)

과적합(Overfitting)이란 훈련 데이터를 과하게 학습한 경우를 말합니다.

![](https://wikidocs.net/images/page/32012/%EC%8A%A4%ED%8C%B8_%EB%A9%94%EC%9D%BC_%EC%98%A4%EC%B0%A8.png)

스팸 메일 분류하기 실습에서 훈련 데이터에 대한 훈련 횟수를 30회로 주어서 의도적으로 과적합을 발생시켰을 때의 그래프입니다.

X축의 에포크(epoch)는 전체 훈련 데이터에 대한 훈련 횟수를 의미합니다.

스팸 메일 분류하기 실습은 에포크가 3~4를 넘어가게 되면 과적합이 발생합니다. 

- 과소적합(Underfitting)

테스트 데이터의 성능이 올라갈 여지가 있음에도 훈련을 덜 한 상태를 반대로 과소적합(Underfitting)이라고 합니다.

과소 적합은 훈련 자체가 부족한 상태이므로 과대 적합과는 달리 훈련 데이터에 대해서도 보통 정확도가 낮다는 특징이 있습니다.

딥 러닝을 할 때는 과적합을 막을 수 있는 드롭 아웃(Drop out), 조기 종료(Early Stopping)과 같은 몇 가지 방법이 존재

## 3. 선형 회귀(Linear Regression)

### 1) 선형 회귀(Linear Regression)

만약, 독립 변수 x가 1개라면 단순 선형 회귀라고 합니다.

#### (1) 단순 선형 회귀 분석(Simple Linear Regression Analysis)

$y = {Wx + b}$

W를 머신 러닝에서는 가중치(Weight), 별도로 더해지는 값 b를 편향(Bias)이라고 합니다

#### (2) 다중 선형 회귀 분석(Multiple Linear Regression Analysis)

$y = {W_1x_1 + W_2x_2 + ... W_nx_n + b}$

y 는 여전히 1개이지만 이제 x는 1개가 아니라 여러 개가 되었습니다. 이제 이를 다중 선형 회귀 분석이라고 합니다.

### 2) 가설(Hypothesis) 세우기

머신 러닝에서는 y와 x간의 관계를 유추한 식을 가설(Hypothesis)이라고 합니다.

$H(x) = {Wx + b}$

![](https://wikidocs.net/images/page/21670/W%EC%99%80_b%EA%B0%80_%EB%8B%A4%EB%A6%84.PNG)

위의 그림은 W와 b의 값에 따라서 천차만별로 그려지는 직선의 모습을 보여줍니다.

어떻게 적절한 W와 b를 찾을 수 있을까요? 지금부터 W와 b를 구하는 방법에 대해서 배워보도록 하겠습니다.

### 3) 비용 함수(Cost function) : 평균 제곱 오차(MSE)

실제값과 예측값에 대한 오차에 대한 식을 목적 함수(Objective function) 또는 비용 함수(Cost function) 또는 손실 함수(Loss function)라고 합니다.

함수의 값을 최소화하거나, 최대화하거나 하는 목적을 가진 함수를 목적 함수(Objective function)라고 합니다. 그리고 값을 최소화하려고 하면 이를 비용 함수(Cost function) 또는 손실 함수(Loss function)라고 합니다.

회귀 문제의 경우에는 주로 평균 제곱 오차(Mean Squered Error, MSE)가 사용됩니다.

$\frac{1}{n} \sum_i^{n} \left[y_{i} - H(x_{i})\right]^2$

평균 제곱 오차를 W와 b에 의한 비용 함수(Cost function)로 재정의해보면 다음과 같습니다.

$cost(W, b) = \frac{1}{n} \sum_i^{n} \left[y_{i} - H(x_{i})\right]^2$

이 평균 최곱 오차. 즉, Cost(W,b)를 최소가 되게 만드는 W와 b를 구하면 결과적으로 y와 x의 관계를 가장 잘 나타내는 직선을 그릴 수 있게 됩니다.

$W, b → minimize\ cost(W, b)$

### 4) 옵티마이저(Optimizer) : 경사하강법(Gradient Descent)

비용 함수(Cost Function)의 값을 최소로 하는 W와 b를 찾는 방법

가장 기본적인 옵티마이저 알고리즘인 경사 하강법(Gradient Descent)에 대해서 배웁니다.

W 와 cost의 관계를 그래프로 표현하면 다음과 같습니다.

$y=Wx$

![](https://wikidocs.net/images/page/21670/%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95%EB%B2%95.PNG)

 비용 함수(Cost function)를 미분하여 현재 W에서의 접선의 기울기를 구하고, 접선의 기울기가 낮은 방향으로 W의 값을 변경하고 다시 미분하고 이 과정을 접선의 기울기가 0인 곳을 향해 W의 값을 변경하는 작업을 반복하는 것

$cost(W) = \frac{1}{n} \sum_i^{n} \left[y_{i} - H(x_{i})\right]^2$

$W := W - α\frac{∂}{∂W}cost(W)$

α는 여기서 학습률(learning rate)

![](https://wikidocs.net/images/page/21670/%EB%AF%B8%EB%B6%84.PNG)

학습률 α은 W의 값을 변경할 때, 얼마나 크게 변경할지를 결정합니다. 또는 W를 그래프의 한 점으로보고 접선의 기울기가 0일 때까지 경사를 따라 내려간다는 관점에서는 얼마나 큰 폭으로 이동할지를 결정합니다.

![](https://wikidocs.net/images/page/21670/%EA%B8%B0%EC%9A%B8%EA%B8%B0%EB%B0%9C%EC%82%B0.PNG)

 학습률 α가 지나치게 높은 값을 가질 때, 접선의 기울기가 0이 되는 W를 찾아가는 것이 아니라 W의 값이 발산하는 상황을 보여줍니다. 

### 5) 케라스로 구현하는 선형 회귀


```python
from keras.models import Sequential # 케라스의 Sequential()을 임포트
from keras.layers import Dense # 케라스의 Dense()를 임포트
from keras import optimizers # 케라스의 옵티마이저를 임포트
import numpy as np # Numpy를 임포트

X=np.array([1,2,3,4,5,6,7,8,9]) # 공부하는 시간
y=np.array([11,22,33,44,53,66,77,87,95]) # 각 공부하는 시간에 맵핑되는 성적

model=Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))
sgd=optimizers.SGD(lr=0.01)
# 학습률(learning rate, lr)은 0.01로 합니다.
model.compile(optimizer=sgd ,loss='mse',metrics=['mse'])
# 옵티마이저는 경사하강법의 일종인 확률적 경사 하강법 sgd를 사용합니다.
# 손실 함수(Loss function)은 평균제곱오차 mse를 사용합니다.
model.fit(X,y, batch_size=1, epochs=300, shuffle=False)
# 주어진 X와 y데이터에 대해서 오차를 최소화하는 작업을 300번 시도합니다.
```

    Epoch 1/300
    9/9 [==============================] - 0s 7ms/step - loss: 383.2430 - mean_squared_error: 383.2430
    Epoch 2/300
    9/9 [==============================] - 0s 553us/step - loss: 2.3026 - mean_squared_error: 2.3026
    Epoch 3/300
    9/9 [==============================] - 0s 663us/step - loss: 2.2969 - mean_squared_error: 2.2969
    Epoch 4/300
    9/9 [==============================] - 0s 662us/step - loss: 2.2913 - mean_squared_error: 2.2913
    Epoch 5/300
    9/9 [==============================] - 0s 669us/step - loss: 2.2860 - mean_squared_error: 2.2860
    Epoch 6/300
    9/9 [==============================] - 0s 771us/step - loss: 2.2809 - mean_squared_error: 2.2809
    Epoch 7/300
    9/9 [==============================] - 0s 664us/step - loss: 2.2760 - mean_squared_error: 2.2760
    Epoch 8/300
    9/9 [==============================] - 0s 665us/step - loss: 2.2713 - mean_squared_error: 2.2713
    Epoch 9/300
    9/9 [==============================] - 0s 665us/step - loss: 2.2667 - mean_squared_error: 2.2667
    Epoch 10/300
    9/9 [==============================] - 0s 662us/step - loss: 2.2624 - mean_squared_error: 2.2624
    Epoch 11/300
    9/9 [==============================] - 0s 665us/step - loss: 2.2582 - mean_squared_error: 2.2582
    Epoch 12/300
    9/9 [==============================] - 0s 662us/step - loss: 2.2541 - mean_squared_error: 2.2541
    Epoch 13/300
    9/9 [==============================] - 0s 553us/step - loss: 2.2503 - mean_squared_error: 2.2503
    Epoch 14/300
    9/9 [==============================] - 0s 554us/step - loss: 2.2465 - mean_squared_error: 2.2465
    Epoch 15/300
    9/9 [==============================] - 0s 554us/step - loss: 2.2430 - mean_squared_error: 2.2430
    Epoch 16/300
    9/9 [==============================] - 0s 554us/step - loss: 2.2395 - mean_squared_error: 2.2395
    Epoch 17/300
    9/9 [==============================] - 0s 443us/step - loss: 2.2362 - mean_squared_error: 2.2362
    Epoch 18/300
    9/9 [==============================] - 0s 554us/step - loss: 2.2330 - mean_squared_error: 2.2330
    Epoch 19/300
    9/9 [==============================] - 0s 554us/step - loss: 2.2299 - mean_squared_error: 2.2299
    Epoch 20/300
    9/9 [==============================] - 0s 554us/step - loss: 2.2270 - mean_squared_error: 2.2270
    Epoch 21/300
    9/9 [==============================] - 0s 554us/step - loss: 2.2241 - mean_squared_error: 2.2241
    Epoch 22/300
    9/9 [==============================] - 0s 554us/step - loss: 2.2214 - mean_squared_error: 2.2214
    Epoch 23/300
    9/9 [==============================] - 0s 443us/step - loss: 2.2187 - mean_squared_error: 2.2187
    Epoch 24/300
    9/9 [==============================] - 0s 554us/step - loss: 2.2162 - mean_squared_error: 2.2162
    Epoch 25/300
    9/9 [==============================] - 0s 613us/step - loss: 2.2137 - mean_squared_error: 2.2137
    Epoch 26/300
    9/9 [==============================] - 0s 554us/step - loss: 2.2114 - mean_squared_error: 2.2114
    Epoch 27/300
    9/9 [==============================] - 0s 554us/step - loss: 2.2091 - mean_squared_error: 2.2091
    Epoch 28/300
    9/9 [==============================] - 0s 557us/step - loss: 2.2069 - mean_squared_error: 2.2069
    Epoch 29/300
    9/9 [==============================] - 0s 444us/step - loss: 2.2048 - mean_squared_error: 2.2048
    Epoch 30/300
    9/9 [==============================] - 0s 554us/step - loss: 2.2028 - mean_squared_error: 2.2028
    Epoch 31/300
    9/9 [==============================] - 0s 554us/step - loss: 2.2008 - mean_squared_error: 2.2008
    Epoch 32/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1989 - mean_squared_error: 2.1989
    Epoch 33/300
    9/9 [==============================] - 0s 440us/step - loss: 2.1971 - mean_squared_error: 2.1971
    Epoch 34/300
    9/9 [==============================] - 0s 447us/step - loss: 2.1953 - mean_squared_error: 2.1953
    Epoch 35/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1936 - mean_squared_error: 2.1936
    Epoch 36/300
    9/9 [==============================] - 0s 440us/step - loss: 2.1920 - mean_squared_error: 2.1920
    Epoch 37/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1904 - mean_squared_error: 2.1904
    Epoch 38/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1889 - mean_squared_error: 2.1889
    Epoch 39/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1874 - mean_squared_error: 2.1874
    Epoch 40/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1860 - mean_squared_error: 2.1860
    Epoch 41/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1846 - mean_squared_error: 2.1846
    Epoch 42/300
    9/9 [==============================] - 0s 558us/step - loss: 2.1833 - mean_squared_error: 2.1833
    Epoch 43/300
    9/9 [==============================] - 0s 446us/step - loss: 2.1820 - mean_squared_error: 2.1820
    Epoch 44/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1808 - mean_squared_error: 2.1808
    Epoch 45/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1796 - mean_squared_error: 2.1796
    Epoch 46/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1785 - mean_squared_error: 2.1785
    Epoch 47/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1774 - mean_squared_error: 2.1774
    Epoch 48/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1763 - mean_squared_error: 2.1763
    Epoch 49/300
    9/9 [==============================] - 0s 447us/step - loss: 2.1753 - mean_squared_error: 2.1753
    Epoch 50/300
    9/9 [==============================] - 0s 558us/step - loss: 2.1743 - mean_squared_error: 2.1743
    Epoch 51/300
    9/9 [==============================] - 0s 439us/step - loss: 2.1734 - mean_squared_error: 2.1734
    Epoch 52/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1724 - mean_squared_error: 2.1724
    Epoch 53/300
    9/9 [==============================] - 0s 557us/step - loss: 2.1716 - mean_squared_error: 2.1716
    Epoch 54/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1707 - mean_squared_error: 2.1707
    Epoch 55/300
    9/9 [==============================] - 0s 666us/step - loss: 2.1699 - mean_squared_error: 2.1699
    Epoch 56/300
    9/9 [==============================] - 0s 550us/step - loss: 2.1690 - mean_squared_error: 2.1690
    Epoch 57/300
    9/9 [==============================] - 0s 557us/step - loss: 2.1683 - mean_squared_error: 2.1683
    Epoch 58/300
    9/9 [==============================] - 0s 440us/step - loss: 2.1675 - mean_squared_error: 2.1675
    Epoch 59/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1668 - mean_squared_error: 2.1668
    Epoch 60/300
    9/9 [==============================] - 0s 557us/step - loss: 2.1661 - mean_squared_error: 2.1661
    Epoch 61/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1654 - mean_squared_error: 2.1654
    Epoch 62/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1648 - mean_squared_error: 2.1648
    Epoch 63/300
    9/9 [==============================] - 0s 557us/step - loss: 2.1642 - mean_squared_error: 2.1642
    Epoch 64/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1635 - mean_squared_error: 2.1635
    Epoch 65/300
    9/9 [==============================] - 0s 557us/step - loss: 2.1630 - mean_squared_error: 2.1630
    Epoch 66/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1624 - mean_squared_error: 2.1624
    Epoch 67/300
    9/9 [==============================] - 0s 440us/step - loss: 2.1618 - mean_squared_error: 2.1618
    Epoch 68/300
    9/9 [==============================] - 0s 440us/step - loss: 2.1613 - mean_squared_error: 2.1613
    Epoch 69/300
    9/9 [==============================] - 0s 447us/step - loss: 2.1608 - mean_squared_error: 2.1608
    Epoch 70/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1603 - mean_squared_error: 2.1603
    Epoch 71/300
    9/9 [==============================] - 0s 557us/step - loss: 2.1598 - mean_squared_error: 2.1598
    Epoch 72/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1594 - mean_squared_error: 2.1594
    Epoch 73/300
    9/9 [==============================] - 0s 446us/step - loss: 2.1589 - mean_squared_error: 2.1589
    Epoch 74/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1585 - mean_squared_error: 2.1585
    Epoch 75/300
    9/9 [==============================] - 0s 558us/step - loss: 2.1581 - mean_squared_error: 2.1581
    Epoch 76/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1577 - mean_squared_error: 2.1577
    Epoch 77/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1573 - mean_squared_error: 2.1573
    Epoch 78/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1569 - mean_squared_error: 2.1569
    Epoch 79/300
    9/9 [==============================] - 0s 444us/step - loss: 2.1565 - mean_squared_error: 2.1565
    Epoch 80/300
    9/9 [==============================] - 0s 439us/step - loss: 2.1562 - mean_squared_error: 2.1562
    Epoch 81/300
    9/9 [==============================] - 0s 447us/step - loss: 2.1559 - mean_squared_error: 2.1559
    Epoch 82/300
    9/9 [==============================] - 0s 557us/step - loss: 2.1555 - mean_squared_error: 2.1555
    Epoch 83/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1552 - mean_squared_error: 2.1552
    Epoch 84/300
    9/9 [==============================] - 0s 440us/step - loss: 2.1549 - mean_squared_error: 2.1549
    Epoch 85/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1546 - mean_squared_error: 2.1546
    Epoch 86/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1543 - mean_squared_error: 2.1543
    Epoch 87/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1541 - mean_squared_error: 2.1541
    Epoch 88/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1538 - mean_squared_error: 2.1538
    Epoch 89/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1535 - mean_squared_error: 2.1535
    Epoch 90/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1533 - mean_squared_error: 2.1533
    Epoch 91/300
    9/9 [==============================] - 0s 446us/step - loss: 2.1530 - mean_squared_error: 2.1530
    Epoch 92/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1528 - mean_squared_error: 2.1528
    Epoch 93/300
    9/9 [==============================] - 0s 440us/step - loss: 2.1526 - mean_squared_error: 2.1526
    Epoch 94/300
    9/9 [==============================] - 0s 446us/step - loss: 2.1524 - mean_squared_error: 2.1524
    Epoch 95/300
    9/9 [==============================] - 0s 447us/step - loss: 2.1522 - mean_squared_error: 2.1522
    Epoch 96/300
    9/9 [==============================] - 0s 446us/step - loss: 2.1520 - mean_squared_error: 2.1520
    Epoch 97/300
    9/9 [==============================] - 0s 440us/step - loss: 2.1518 - mean_squared_error: 2.1518
    Epoch 98/300
    9/9 [==============================] - 0s 440us/step - loss: 2.1516 - mean_squared_error: 2.1516
    Epoch 99/300
    9/9 [==============================] - 0s 553us/step - loss: 2.1514 - mean_squared_error: 2.1514
    Epoch 100/300
    9/9 [==============================] - 0s 555us/step - loss: 2.1512 - mean_squared_error: 2.1512
    Epoch 101/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1510 - mean_squared_error: 2.1510
    Epoch 102/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1509 - mean_squared_error: 2.1509
    Epoch 103/300
    9/9 [==============================] - 0s 446us/step - loss: 2.1507 - mean_squared_error: 2.1507
    Epoch 104/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1506 - mean_squared_error: 2.1506
    Epoch 105/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1504 - mean_squared_error: 2.1504
    Epoch 106/300
    9/9 [==============================] - 0s 555us/step - loss: 2.1503 - mean_squared_error: 2.1503
    Epoch 107/300
    9/9 [==============================] - 0s 555us/step - loss: 2.1501 - mean_squared_error: 2.1501
    Epoch 108/300
    9/9 [==============================] - 0s 550us/step - loss: 2.1500 - mean_squared_error: 2.1500
    Epoch 109/300
    9/9 [==============================] - 0s 445us/step - loss: 2.1498 - mean_squared_error: 2.1498
    Epoch 110/300
    9/9 [==============================] - 0s 549us/step - loss: 2.1497 - mean_squared_error: 2.1497
    Epoch 111/300
    9/9 [==============================] - 0s 557us/step - loss: 2.1496 - mean_squared_error: 2.1496
    Epoch 112/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1495 - mean_squared_error: 2.1495
    Epoch 113/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1494 - mean_squared_error: 2.1494
    Epoch 114/300
    9/9 [==============================] - 0s 558us/step - loss: 2.1493 - mean_squared_error: 2.1493
    Epoch 115/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1492 - mean_squared_error: 2.1492
    Epoch 116/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1490 - mean_squared_error: 2.1490
    Epoch 117/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1489 - mean_squared_error: 2.1489
    Epoch 118/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1489 - mean_squared_error: 2.1489
    Epoch 119/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1488 - mean_squared_error: 2.1488
    Epoch 120/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1487 - mean_squared_error: 2.1487
    Epoch 121/300
    9/9 [==============================] - 0s 440us/step - loss: 2.1486 - mean_squared_error: 2.1486
    Epoch 122/300
    9/9 [==============================] - 0s 446us/step - loss: 2.1485 - mean_squared_error: 2.1485
    Epoch 123/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1484 - mean_squared_error: 2.1484
    Epoch 124/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1483 - mean_squared_error: 2.1483
    Epoch 125/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1483 - mean_squared_error: 2.1483
    Epoch 126/300
    9/9 [==============================] - 0s 558us/step - loss: 2.1482 - mean_squared_error: 2.1482
    Epoch 127/300
    9/9 [==============================] - 0s 550us/step - loss: 2.1481 - mean_squared_error: 2.1481
    Epoch 128/300
    9/9 [==============================] - 0s 441us/step - loss: 2.1480 - mean_squared_error: 2.1480
    Epoch 129/300
    9/9 [==============================] - 0s 557us/step - loss: 2.1480 - mean_squared_error: 2.1480
    Epoch 130/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1479 - mean_squared_error: 2.1479
    Epoch 131/300
    9/9 [==============================] - 0s 440us/step - loss: 2.1479 - mean_squared_error: 2.1479
    Epoch 132/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1478 - mean_squared_error: 2.1478
    Epoch 133/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1477 - mean_squared_error: 2.1477
    Epoch 134/300
    9/9 [==============================] - 0s 447us/step - loss: 2.1477 - mean_squared_error: 2.1477
    Epoch 135/300
    9/9 [==============================] - 0s 446us/step - loss: 2.1476 - mean_squared_error: 2.1476
    Epoch 136/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1476 - mean_squared_error: 2.1476
    Epoch 137/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1475 - mean_squared_error: 2.1475
    Epoch 138/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1475 - mean_squared_error: 2.1475
    Epoch 139/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1474 - mean_squared_error: 2.1474
    Epoch 140/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1474 - mean_squared_error: 2.1474
    Epoch 141/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1473 - mean_squared_error: 2.1473
    Epoch 142/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1473 - mean_squared_error: 2.1473
    Epoch 143/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1472 - mean_squared_error: 2.1472
    Epoch 144/300
    9/9 [==============================] - 0s 447us/step - loss: 2.1472 - mean_squared_error: 2.1472
    Epoch 145/300
    9/9 [==============================] - 0s 445us/step - loss: 2.1472 - mean_squared_error: 2.1472
    Epoch 146/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1471 - mean_squared_error: 2.1471
    Epoch 147/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1471 - mean_squared_error: 2.1471
    Epoch 148/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1471 - mean_squared_error: 2.1471
    Epoch 149/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1470 - mean_squared_error: 2.1470
    Epoch 150/300
    9/9 [==============================] - 0s 444us/step - loss: 2.1470 - mean_squared_error: 2.1470
    Epoch 151/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1470 - mean_squared_error: 2.1470
    Epoch 152/300
    9/9 [==============================] - 0s 442us/step - loss: 2.1469 - mean_squared_error: 2.1469
    Epoch 153/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1469 - mean_squared_error: 2.1469
    Epoch 154/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1469 - mean_squared_error: 2.1469
    Epoch 155/300
    9/9 [==============================] - 0s 550us/step - loss: 2.1468 - mean_squared_error: 2.1468
    Epoch 156/300
    9/9 [==============================] - 0s 557us/step - loss: 2.1468 - mean_squared_error: 2.1468
    Epoch 157/300
    9/9 [==============================] - 0s 556us/step - loss: 2.1468 - mean_squared_error: 2.1468
    Epoch 158/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1468 - mean_squared_error: 2.1468
    Epoch 159/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1467 - mean_squared_error: 2.1467
    Epoch 160/300
    9/9 [==============================] - 0s 440us/step - loss: 2.1467 - mean_squared_error: 2.1467
    Epoch 161/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1467 - mean_squared_error: 2.1467
    Epoch 162/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1467 - mean_squared_error: 2.1467
    Epoch 163/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1467 - mean_squared_error: 2.1467
    Epoch 164/300
    9/9 [==============================] - 0s 440us/step - loss: 2.1466 - mean_squared_error: 2.1466
    Epoch 165/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1466 - mean_squared_error: 2.1466
    Epoch 166/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1466 - mean_squared_error: 2.1466
    Epoch 167/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1466 - mean_squared_error: 2.1466
    Epoch 168/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1466 - mean_squared_error: 2.1466
    Epoch 169/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1465 - mean_squared_error: 2.1465
    Epoch 170/300
    9/9 [==============================] - 0s 557us/step - loss: 2.1465 - mean_squared_error: 2.1465
    Epoch 171/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1465 - mean_squared_error: 2.1465
    Epoch 172/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1465 - mean_squared_error: 2.1465
    Epoch 173/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1465 - mean_squared_error: 2.1465
    Epoch 174/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1465 - mean_squared_error: 2.1465
    Epoch 175/300
    9/9 [==============================] - 0s 439us/step - loss: 2.1464 - mean_squared_error: 2.1464
    Epoch 176/300
    9/9 [==============================] - 0s 446us/step - loss: 2.1464 - mean_squared_error: 2.1464
    Epoch 177/300
    9/9 [==============================] - 0s 440us/step - loss: 2.1464 - mean_squared_error: 2.1464
    Epoch 178/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1464 - mean_squared_error: 2.1464
    Epoch 179/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1464 - mean_squared_error: 2.1464
    Epoch 180/300
    9/9 [==============================] - 0s 558us/step - loss: 2.1464 - mean_squared_error: 2.1464
    Epoch 181/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1464 - mean_squared_error: 2.1464
    Epoch 182/300
    9/9 [==============================] - 0s 447us/step - loss: 2.1464 - mean_squared_error: 2.1464
    Epoch 183/300
    9/9 [==============================] - 0s 444us/step - loss: 2.1463 - mean_squared_error: 2.1463
    Epoch 184/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1463 - mean_squared_error: 2.1463
    Epoch 185/300
    9/9 [==============================] - 0s 557us/step - loss: 2.1463 - mean_squared_error: 2.1463
    Epoch 186/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1463 - mean_squared_error: 2.1463
    Epoch 187/300
    9/9 [==============================] - 0s 558us/step - loss: 2.1463 - mean_squared_error: 2.1463
    Epoch 188/300
    9/9 [==============================] - 0s 445us/step - loss: 2.1463 - mean_squared_error: 2.1463
    Epoch 189/300
    9/9 [==============================] - 0s 557us/step - loss: 2.1463 - mean_squared_error: 2.1463
    Epoch 190/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1463 - mean_squared_error: 2.1463
    Epoch 191/300
    9/9 [==============================] - 0s 558us/step - loss: 2.1463 - mean_squared_error: 2.1463
    Epoch 192/300
    9/9 [==============================] - 0s 553us/step - loss: 2.1463 - mean_squared_error: 2.1463
    Epoch 193/300
    9/9 [==============================] - 0s 557us/step - loss: 2.1463 - mean_squared_error: 2.1463
    Epoch 194/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1462 - mean_squared_error: 2.1462
    Epoch 195/300
    9/9 [==============================] - 0s 440us/step - loss: 2.1462 - mean_squared_error: 2.1462
    Epoch 196/300
    9/9 [==============================] - 0s 445us/step - loss: 2.1462 - mean_squared_error: 2.1462
    Epoch 197/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1462 - mean_squared_error: 2.1462
    Epoch 198/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1462 - mean_squared_error: 2.1462
    Epoch 199/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1462 - mean_squared_error: 2.1462
    Epoch 200/300
    9/9 [==============================] - 0s 557us/step - loss: 2.1462 - mean_squared_error: 2.1462
    Epoch 201/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1462 - mean_squared_error: 2.1462
    Epoch 202/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1462 - mean_squared_error: 2.1462
    Epoch 203/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1462 - mean_squared_error: 2.1462
    Epoch 204/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1462 - mean_squared_error: 2.1462
    Epoch 205/300
    9/9 [==============================] - 0s 439us/step - loss: 2.1462 - mean_squared_error: 2.1462
    Epoch 206/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1462 - mean_squared_error: 2.1462
    Epoch 207/300
    9/9 [==============================] - 0s 556us/step - loss: 2.1462 - mean_squared_error: 2.1462
    Epoch 208/300
    9/9 [==============================] - 0s 553us/step - loss: 2.1462 - mean_squared_error: 2.1462
    Epoch 209/300
    9/9 [==============================] - 0s 446us/step - loss: 2.1462 - mean_squared_error: 2.1462
    Epoch 210/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1462 - mean_squared_error: 2.1462
    Epoch 211/300
    9/9 [==============================] - 0s 440us/step - loss: 2.1462 - mean_squared_error: 2.1462
    Epoch 212/300
    9/9 [==============================] - 0s 557us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 213/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 214/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 215/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 216/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 217/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 218/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 219/300
    9/9 [==============================] - 0s 447us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 220/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 221/300
    9/9 [==============================] - 0s 439us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 222/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 223/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 224/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 225/300
    9/9 [==============================] - 0s 444us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 226/300
    9/9 [==============================] - 0s 446us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 227/300
    9/9 [==============================] - 0s 546us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 228/300
    9/9 [==============================] - 0s 555us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 229/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 230/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 231/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 232/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 233/300
    9/9 [==============================] - 0s 440us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 234/300
    9/9 [==============================] - 0s 440us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 235/300
    9/9 [==============================] - 0s 550us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 236/300
    9/9 [==============================] - 0s 446us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 237/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 238/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 239/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 240/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 241/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 242/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 243/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 244/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 245/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 246/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 247/300
    9/9 [==============================] - 0s 448us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 248/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 249/300
    9/9 [==============================] - 0s 440us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 250/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 251/300
    9/9 [==============================] - 0s 557us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 252/300
    9/9 [==============================] - 0s 447us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 253/300
    9/9 [==============================] - 0s 442us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 254/300
    9/9 [==============================] - 0s 446us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 255/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 256/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 257/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 258/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1461 - mean_squared_error: 2.1461
    Epoch 259/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 260/300
    9/9 [==============================] - 0s 447us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 261/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 262/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 263/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 264/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 265/300
    9/9 [==============================] - 0s 444us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 266/300
    9/9 [==============================] - 0s 568us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 267/300
    9/9 [==============================] - 0s 448us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 268/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 269/300
    9/9 [==============================] - 0s 558us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 270/300
    9/9 [==============================] - 0s 550us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 271/300
    9/9 [==============================] - 0s 446us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 272/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 273/300
    9/9 [==============================] - 0s 444us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 274/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 275/300
    9/9 [==============================] - 0s 558us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 276/300
    9/9 [==============================] - 0s 550us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 277/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 278/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 279/300
    9/9 [==============================] - 0s 558us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 280/300
    9/9 [==============================] - 0s 550us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 281/300
    9/9 [==============================] - 0s 447us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 282/300
    9/9 [==============================] - 0s 439us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 283/300
    9/9 [==============================] - 0s 553us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 284/300
    9/9 [==============================] - 0s 557us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 285/300
    9/9 [==============================] - 0s 556us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 286/300
    9/9 [==============================] - 0s 557us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 287/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 288/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 289/300
    9/9 [==============================] - 0s 557us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 290/300
    9/9 [==============================] - 0s 553us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 291/300
    9/9 [==============================] - 0s 443us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 292/300
    9/9 [==============================] - 0s 551us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 293/300
    9/9 [==============================] - 0s 445us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 294/300
    9/9 [==============================] - 0s 444us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 295/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 296/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 297/300
    9/9 [==============================] - 0s 554us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 298/300
    9/9 [==============================] - 0s 555us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 299/300
    9/9 [==============================] - 0s 446us/step - loss: 2.1460 - mean_squared_error: 2.1460
    Epoch 300/300
    9/9 [==============================] - 0s 439us/step - loss: 2.1460 - mean_squared_error: 2.1460
    




    <keras.callbacks.History at 0x215e0614c18>



activation은 어떤 함수를 사용할 것인지를 의미하는데 선형 회귀를 사용할 경우에는 linear라고 기재합니다.

옵티마이저로는 경사 하강법의 일종인 확률적 경사 하강법을 사용하였으며, 학습률은 0.01로 정하였습니다. 손실 함수로는 평균 제곱 오차를 사용합니다. 그리고 전체 데이터에 대한 훈련 횟수는 300으로 합니다.

전체 데이터에 대한 훈련 횟수는 300으로 하였지만, 어느 순간 오차가 더 이상 줄어들지 않는데 이는 오차를 최소화하는 가중치 W와 b를 찾았기 때문으로 추정이 가능합니다. 이제 최종적으로 선택된 오차를 최소화하는 직선을 그래프로 그려보겠습니다.


```python
import matplotlib.pyplot as plt
plt.plot(X, model.predict(X), 'b', X,y, 'k.')
```




    [<matplotlib.lines.Line2D at 0x215e134d5c0>,
     <matplotlib.lines.Line2D at 0x215e134d710>]




![png](NLP_basic_07_Machine_Learning_files/NLP_basic_07_Machine_Learning_91_1.png)


model.predict()은 학습이 완료된 모델이 입력된 데이터에 대해서 어떤 값을 예측하는지를 보여줍니다.


```python
print(model.predict([9.5]))
```

    [[98.55647]]
    

9시간 30분을 공부하면 약 98.5점을 얻는다고 예측하고 있습니다.

## 4. 로지스틱 회귀(Logistic Regression) - 이진 분류

### 1) 이진 분류(Binary Classification)

위의 데이터에서 합격을 1, 불합격을 0이라고 하였을 때 그래프를 그려보면 아래와 같습니다. 

![](https://wikidocs.net/images/page/22881/%EB%A1%9C%EC%A7%80%EC%8A%A4%ED%8B%B1%ED%9A%8C%EA%B7%80.PNG)

실제값 y가 0 또는 1이라는 두 가지 값밖에 가지지 않는데, 이 문제를 풀기 위해서는 예측값이 0과 1사이의 값을 가지도록 하는 것이 보편적입니다.

0과 1사이의 값을 가지면서, S자 형태로 그려지는 이러한 조건을 충족하는 유명한 함수가 존재하는데, 바로 시그모이드 함수(Sigmoid function)입니다.

### 2) 시그모이드 함수(Sigmoid function)

$H(X) = \frac{1}{1 + e^{-(Wx + b)}} = sigmoid(Wx + b) = σ(Wx + b)$

여기서 구해야할 것은 여전히 주어진 데이터에 가장 적합한 가중치 W(weight)와 편향 b(bias)입니다.

시그모이드 함수를 파이썬의 Matplotlib을 통해서 그래프로 표현하면 다음과 같습니다. 아래의 그래프는 W는 1, b는 0임을 가정한 그래프입니다.


```python
import numpy as np # 넘파이 사용
import matplotlib.pyplot as plt # 맷플롯립 사용

def sigmoid(x):
    return 1/(1+np.exp(-x))
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y, 'g')
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()
```


![png](NLP_basic_07_Machine_Learning_files/NLP_basic_07_Machine_Learning_105_0.png)


우선 W의 값을 변화시키고 이에 따른 그래프를 확인해보겠습니다.


```python
import numpy as np # 넘파이 사용
import matplotlib.pyplot as plt # 맷플롯립 사용

def sigmoid(x):
    return 1/(1+np.exp(-x))
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5*x)
y2 = sigmoid(x)
y3 = sigmoid(2*x)

plt.plot(x, y1, 'r', linestyle='--') # W의 값이 0.5일때
plt.plot(x, y2, 'g') # W의 값이 1일때
plt.plot(x, y3, 'b', linestyle='--') # W의 값이 2일때
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()
```


![png](NLP_basic_07_Machine_Learning_files/NLP_basic_07_Machine_Learning_107_0.png)


위의 그래프는 W의 값이 0.5일때 빨간색선, W의 값이 1일때는 초록색선, W의 값이 2일때 파란색선이 나오도록 하였습니다.

앞서 선형 회귀에서 가중치 W는 직선의 기울기를 의미했지만, 여기서는 그래프의 경사도를 결정합니다.

이제 b의 값에 따라서 그래프가 어떻게 변하는지 확인해보도록 하겠습니다.


```python
import numpy as np # 넘파이 사용
import matplotlib.pyplot as plt # 맷플롯립 사용

def sigmoid(x):
    return 1/(1+np.exp(-x))
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x+0.5)
y2 = sigmoid(x+1)
y3 = sigmoid(x+1.5)

plt.plot(x, y1, 'r', linestyle='--') # x + 0.5
plt.plot(x, y2, 'g') # x + 1
plt.plot(x, y3, 'b', linestyle='--') # x + 1.5
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()
```


![png](NLP_basic_07_Machine_Learning_files/NLP_basic_07_Machine_Learning_111_0.png)


위의 그래프는 b의 값에 따라서 그래프가 이동하는 것을 보여줍니다.

시그모이드 함수는 입력값이 커지면 1에 수렴하고, 입력값이 작아지면 0에 수렴합니다. 0부터의 1까지의 값을 가지는데 출력값이 0.5 이상이면 1(True), 0.5이하면 0(False)로 만들면 이진 분류 문제로 사용할 수 있습니다.

이를 확률이라고 생각하면 해당 범주에 속할 확률이 50%가 넘으면, 해당 범주라고 판단하고 50%보다 낮으면 아니라고 판단한다고도 볼 수 있습니다.

### 3) 비용 함수(Cost function)

로지스틱 회귀 또한 경사 하강법을 사용하여 가중치 W를 찾아내지만, 비용 함수로는 평균 제곱 오차를 사용하지 않습니다.

그 이유는 시그모이드 함수에 비용 함수를 평균 제곱 오차로 하여 그래프를 그리면 다음과 같이 되기 때문입니다.

![](https://wikidocs.net/images/page/22881/%EB%A1%9C%EC%BB%AC%EB%AF%B8%EB%8B%88%EB%A9%88.PNG)

로지스틱 회귀에서 평균 제곱 오차를 비용 함수로 사용하면, 경사 하강법을 사용하였을때 자칫 잘못하면 찾고자 하는 최소값이 아닌 잘못된 최소값에 빠집니다.

이를 전체 함수에 걸쳐 최소값인 글로벌 미니멈(Global Minimum)이 아닌 특정 구역에서의 최소값인 로컬 미니멈(Local Minimum)에 도달했다고 합니다. 이는 cost가 최소가 되는 가중치 W를 찾는다는 비용 함수의 목적에 맞지 않습니다.

그렇다면 가중치 W를 최소로 만드는 새로운 비용 함수를 찾아야 합니다. 가중치를 최소화하는 아래의 어떤 함수를 목적 함수라고 합시다.

비용 함수와 목적 함수를 최적의 가중치를 찾기 위해 함수의 값을 최소화하는 함수라는 의미에서 같은 의미의 용어로 사용합니다.

J는 목적 함수(objective function)를 의미합니다.

$J(W) = \frac{1}{n} \sum_{i=1}^{n} f\left(H(X_{i}), y_{i})\right)$

여기서 이 f는 비용 함수(cost function)라고 하겠습니다.

$J(W) = \frac{1}{n} \sum_{i=1}^{n} cost\left(H(X_{i}), y_{i})\right)$

시그모이드 함수는 0과 1사이의 y값을 반환합니다. 이는 실제값이 0일 때 y값이 1에 가까워지면 오차가 커지며 실제값이 1일 때 y값이 0에 가까워지면 오차가 커짐을 의미합니다. 그리고 이를 반영할 수 있는 함수는 로그 함수를 통해 표현이 가능합니다.

$\text{if } y=1 → \text{cost}\left( H(X), y \right) = -\log(H(X))$<br/>
$\text{if } y=0 → \text{cost}\left( H(X), y \right) = -\log(1-H(x))$

위의 두 식을 그래프 상으로 표현하면 아래와 같습니다.

![](https://wikidocs.net/images/page/22881/%EC%86%90%EC%8B%A4%ED%95%A8%EC%88%98.PNG)

실제값이 1일 때의 그래프를 파란색 선으로 표현하였으며, 실제값이 0일 때의 그래프를 빨간색 선으로 표현하였습니다.

위의 그래프를 간략히 설명하면, 실제값이 1일 때, 예측값인 H(X)의 값이 1이면 오차가 0이므로 당연히 cost는 0이 됩니다. 반면, 실제값이 1일 때, H(X)가 0으로 수렴하면 cost는 무한대로 발산합니다. 

실제값이 0인 경우는 그 반대로 이해하면 됩니다. 이는 다음과 같이 하나의 식으로 표현할 수 있습니다.

$\text{cost}\left( H(X), y \right) = -[ylogH(X) + (1-y)log(1-H(X))]$

결과적으로 로지스틱 회귀의 목적 함수는 아래와 같습니다.

$J(W) = -\frac{1}{n} \sum_{i=1}^{n} [y_{i}logH(X_{i}) + (1-y_{i})log(1-H(X_{i}))]$

이때 로지스틱 회귀에서 찾아낸 비용 함수를 크로스 엔트로피(Cross Entropy)함수라고 합니다.

즉, 결론적으로 로지스틱 회귀는 비용 함수로 크로스 엔트로피 함수를 사용하며, 가중치를 찾기 위해서 크로스 엔트로피 함수의 평균을 취한 함수를 사용합니다. 

### 4) 케라스로 구현하는 로지스틱 회귀


```python
from keras.models import Sequential # 케라스의 Sequential()을 임포트
from keras.layers import Dense # 케라스의 Dense()를 임포트
from keras import optimizers # 케라스의 옵티마이저를 임포트
import numpy as np # Numpy를 임포트

X=np.array([-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50])
y=np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) #숫자 10부터 1

model=Sequential()
model.add(Dense(1, input_dim=1, activation='sigmoid'))
sgd=optimizers.SGD(lr=0.01)
model.compile(optimizer=sgd ,loss='binary_crossentropy',metrics=['binary_accuracy'])
# 옵티마이저는 경사하강법의 일종인 확률적 경사 하강법 sgd를 사용합니다.
# 손실 함수(Loss function)는 binary_crossentropy(이진 크로스 엔트로피)를 사용합니다.
model.fit(X,y, batch_size=1, epochs=200, shuffle=False)
# 주어진 X와 y데이터에 대해서 오차를 최소화하는 작업을 200번 시도합니다.
```

    Epoch 1/200
    13/13 [==============================] - 0s 9ms/step - loss: 0.2069 - binary_accuracy: 0.9231
    Epoch 2/200
    13/13 [==============================] - 0s 767us/step - loss: 0.2051 - binary_accuracy: 0.9231
    Epoch 3/200
    13/13 [==============================] - 0s 767us/step - loss: 0.2034 - binary_accuracy: 0.9231
    Epoch 4/200
    13/13 [==============================] - 0s 691us/step - loss: 0.2017 - binary_accuracy: 0.9231
    Epoch 5/200
    13/13 [==============================] - 0s 689us/step - loss: 0.2000 - binary_accuracy: 0.9231
    Epoch 6/200
    13/13 [==============================] - 0s 690us/step - loss: 0.1984 - binary_accuracy: 0.9231
    Epoch 7/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1968 - binary_accuracy: 0.9231
    Epoch 8/200
    13/13 [==============================] - 0s 703us/step - loss: 0.1952 - binary_accuracy: 0.9231
    Epoch 9/200
    13/13 [==============================] - 0s 612us/step - loss: 0.1936 - binary_accuracy: 0.9231
    Epoch 10/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1921 - binary_accuracy: 0.9231
    Epoch 11/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1906 - binary_accuracy: 0.9231
    Epoch 12/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1891 - binary_accuracy: 0.9231
    Epoch 13/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1877 - binary_accuracy: 0.9231
    Epoch 14/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1862 - binary_accuracy: 0.9231
    Epoch 15/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1848 - binary_accuracy: 0.9231
    Epoch 16/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1834 - binary_accuracy: 0.9231
    Epoch 17/200
    13/13 [==============================] - 0s 616us/step - loss: 0.1821 - binary_accuracy: 0.9231
    Epoch 18/200
    13/13 [==============================] - 0s 539us/step - loss: 0.1807 - binary_accuracy: 0.9231
    Epoch 19/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1794 - binary_accuracy: 0.9231
    Epoch 20/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1781 - binary_accuracy: 0.9231
    Epoch 21/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1768 - binary_accuracy: 0.9231
    Epoch 22/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1756 - binary_accuracy: 0.9231
    Epoch 23/200
    13/13 [==============================] - 0s 460us/step - loss: 0.1743 - binary_accuracy: 0.9231
    Epoch 24/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1731 - binary_accuracy: 0.9231
    Epoch 25/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1719 - binary_accuracy: 0.9231
    Epoch 26/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1707 - binary_accuracy: 0.9231
    Epoch 27/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1696 - binary_accuracy: 0.9231
    Epoch 28/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1684 - binary_accuracy: 0.9231
    Epoch 29/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1673 - binary_accuracy: 0.9231
    Epoch 30/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1662 - binary_accuracy: 0.9231
    Epoch 31/200
    13/13 [==============================] - 0s 510us/step - loss: 0.1651 - binary_accuracy: 0.9231
    Epoch 32/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1640 - binary_accuracy: 0.9231
    Epoch 33/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1629 - binary_accuracy: 0.9231
    Epoch 34/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1619 - binary_accuracy: 0.9231
    Epoch 35/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1609 - binary_accuracy: 0.9231
    Epoch 36/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1599 - binary_accuracy: 0.9231
    Epoch 37/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1589 - binary_accuracy: 0.9231
    Epoch 38/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1579 - binary_accuracy: 0.9231
    Epoch 39/200
    13/13 [==============================] - 0s 616us/step - loss: 0.1569 - binary_accuracy: 0.9231
    Epoch 40/200
    13/13 [==============================] - 0s 612us/step - loss: 0.1560 - binary_accuracy: 0.9231
    Epoch 41/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1550 - binary_accuracy: 0.9231
    Epoch 42/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1541 - binary_accuracy: 0.9231
    Epoch 43/200
    13/13 [==============================] - 0s 582us/step - loss: 0.1532 - binary_accuracy: 0.9231
    Epoch 44/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1523 - binary_accuracy: 0.9231
    Epoch 45/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1514 - binary_accuracy: 0.9231
    Epoch 46/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1505 - binary_accuracy: 0.9231
    Epoch 47/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1497 - binary_accuracy: 0.9231
    Epoch 48/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1488 - binary_accuracy: 0.9231
    Epoch 49/200
    13/13 [==============================] - 0s 460us/step - loss: 0.1480 - binary_accuracy: 0.9231
    Epoch 50/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1472 - binary_accuracy: 0.9231
    Epoch 51/200
    13/13 [==============================] - 0s 460us/step - loss: 0.1464 - binary_accuracy: 0.9231
    Epoch 52/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1456 - binary_accuracy: 0.9231
    Epoch 53/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1448 - binary_accuracy: 0.9231
    Epoch 54/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1440 - binary_accuracy: 0.9231
    Epoch 55/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1432 - binary_accuracy: 0.9231
    Epoch 56/200
    13/13 [==============================] - 0s 616us/step - loss: 0.1425 - binary_accuracy: 0.9231
    Epoch 57/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1417 - binary_accuracy: 0.9231
    Epoch 58/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1410 - binary_accuracy: 0.9231
    Epoch 59/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1403 - binary_accuracy: 0.9231
    Epoch 60/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1396 - binary_accuracy: 0.9231
    Epoch 61/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1389 - binary_accuracy: 0.9231
    Epoch 62/200
    13/13 [==============================] - 0s 460us/step - loss: 0.1382 - binary_accuracy: 0.9231
    Epoch 63/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1375 - binary_accuracy: 0.9231
    Epoch 64/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1368 - binary_accuracy: 0.9231
    Epoch 65/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1361 - binary_accuracy: 0.9231
    Epoch 66/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1355 - binary_accuracy: 0.9231
    Epoch 67/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1348 - binary_accuracy: 0.9231
    Epoch 68/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1342 - binary_accuracy: 0.9231
    Epoch 69/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1335 - binary_accuracy: 0.9231
    Epoch 70/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1329 - binary_accuracy: 0.9231
    Epoch 71/200
    13/13 [==============================] - 0s 460us/step - loss: 0.1323 - binary_accuracy: 0.9231
    Epoch 72/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1317 - binary_accuracy: 0.9231
    Epoch 73/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1311 - binary_accuracy: 0.9231
    Epoch 74/200
    13/13 [==============================] - 0s 460us/step - loss: 0.1305 - binary_accuracy: 0.9231
    Epoch 75/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1299 - binary_accuracy: 0.9231
    Epoch 76/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1293 - binary_accuracy: 0.9231
    Epoch 77/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1287 - binary_accuracy: 0.9231
    Epoch 78/200
    13/13 [==============================] - 0s 460us/step - loss: 0.1281 - binary_accuracy: 0.9231
    Epoch 79/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1276 - binary_accuracy: 0.9231
    Epoch 80/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1270 - binary_accuracy: 0.9231
    Epoch 81/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1265 - binary_accuracy: 0.9231
    Epoch 82/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1259 - binary_accuracy: 0.9231
    Epoch 83/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1254 - binary_accuracy: 0.9231
    Epoch 84/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1249 - binary_accuracy: 0.9231
    Epoch 85/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1243 - binary_accuracy: 0.9231
    Epoch 86/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1238 - binary_accuracy: 0.9231
    Epoch 87/200
    13/13 [==============================] - 0s 535us/step - loss: 0.1233 - binary_accuracy: 0.9231
    Epoch 88/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1228 - binary_accuracy: 0.9231
    Epoch 89/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1223 - binary_accuracy: 0.9231
    Epoch 90/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1218 - binary_accuracy: 0.9231
    Epoch 91/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1213 - binary_accuracy: 0.9231
    Epoch 92/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1208 - binary_accuracy: 0.9231
    Epoch 93/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1204 - binary_accuracy: 0.9231
    Epoch 94/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1199 - binary_accuracy: 0.9231
    Epoch 95/200
    13/13 [==============================] - 0s 535us/step - loss: 0.1194 - binary_accuracy: 0.9231
    Epoch 96/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1190 - binary_accuracy: 0.9231
    Epoch 97/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1185 - binary_accuracy: 0.9231
    Epoch 98/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1181 - binary_accuracy: 0.9231
    Epoch 99/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1176 - binary_accuracy: 0.9231
    Epoch 100/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1172 - binary_accuracy: 0.9231
    Epoch 101/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1167 - binary_accuracy: 0.9231
    Epoch 102/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1163 - binary_accuracy: 0.9231
    Epoch 103/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1159 - binary_accuracy: 0.9231
    Epoch 104/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1155 - binary_accuracy: 0.9231
    Epoch 105/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1150 - binary_accuracy: 0.9231
    Epoch 106/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1146 - binary_accuracy: 0.9231
    Epoch 107/200
    13/13 [==============================] - 0s 535us/step - loss: 0.1142 - binary_accuracy: 0.9231
    Epoch 108/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1138 - binary_accuracy: 0.9231
    Epoch 109/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1134 - binary_accuracy: 0.9231
    Epoch 110/200
    13/13 [==============================] - 0s 536us/step - loss: 0.1130 - binary_accuracy: 0.9231
    Epoch 111/200
    13/13 [==============================] - 0s 613us/step - loss: 0.1126 - binary_accuracy: 0.9231
    Epoch 112/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1122 - binary_accuracy: 0.9231
    Epoch 113/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1118 - binary_accuracy: 0.9231
    Epoch 114/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1114 - binary_accuracy: 0.9231
    Epoch 115/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1111 - binary_accuracy: 0.9231
    Epoch 116/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1107 - binary_accuracy: 0.9231
    Epoch 117/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1103 - binary_accuracy: 0.9231
    Epoch 118/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1100 - binary_accuracy: 0.9231
    Epoch 119/200
    13/13 [==============================] - 0s 536us/step - loss: 0.1096 - binary_accuracy: 0.9231
    Epoch 120/200
    13/13 [==============================] - 0s 539us/step - loss: 0.1092 - binary_accuracy: 0.9231
    Epoch 121/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1089 - binary_accuracy: 0.9231
    Epoch 122/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1085 - binary_accuracy: 0.9231
    Epoch 123/200
    13/13 [==============================] - 0s 535us/step - loss: 0.1082 - binary_accuracy: 0.9231
    Epoch 124/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1078 - binary_accuracy: 0.9231
    Epoch 125/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1075 - binary_accuracy: 0.9231
    Epoch 126/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1071 - binary_accuracy: 0.9231
    Epoch 127/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1068 - binary_accuracy: 0.9231
    Epoch 128/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1065 - binary_accuracy: 0.9231
    Epoch 129/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1061 - binary_accuracy: 0.9231
    Epoch 130/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1058 - binary_accuracy: 0.9231
    Epoch 131/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1055 - binary_accuracy: 0.9231
    Epoch 132/200
    13/13 [==============================] - 0s 614us/step - loss: 0.1052 - binary_accuracy: 0.9231
    Epoch 133/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1048 - binary_accuracy: 0.9231
    Epoch 134/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1045 - binary_accuracy: 0.9231
    Epoch 135/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1042 - binary_accuracy: 0.9231
    Epoch 136/200
    13/13 [==============================] - 0s 539us/step - loss: 0.1039 - binary_accuracy: 0.9231
    Epoch 137/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1036 - binary_accuracy: 0.9231
    Epoch 138/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1033 - binary_accuracy: 0.9231
    Epoch 139/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1030 - binary_accuracy: 0.9231
    Epoch 140/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1027 - binary_accuracy: 0.9231
    Epoch 141/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1024 - binary_accuracy: 0.9231
    Epoch 142/200
    13/13 [==============================] - 0s 460us/step - loss: 0.1021 - binary_accuracy: 0.9231
    Epoch 143/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1018 - binary_accuracy: 0.9231
    Epoch 144/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1015 - binary_accuracy: 0.9231
    Epoch 145/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1012 - binary_accuracy: 0.9231
    Epoch 146/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1009 - binary_accuracy: 0.9231
    Epoch 147/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1006 - binary_accuracy: 0.9231
    Epoch 148/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1004 - binary_accuracy: 0.9231
    Epoch 149/200
    13/13 [==============================] - 0s 537us/step - loss: 0.1001 - binary_accuracy: 0.9231
    Epoch 150/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0998 - binary_accuracy: 0.9231
    Epoch 151/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0995 - binary_accuracy: 0.9231
    Epoch 152/200
    13/13 [==============================] - 0s 614us/step - loss: 0.0993 - binary_accuracy: 0.9231
    Epoch 153/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0990 - binary_accuracy: 0.9231
    Epoch 154/200
    13/13 [==============================] - 0s 614us/step - loss: 0.0987 - binary_accuracy: 0.9231
    Epoch 155/200
    13/13 [==============================] - 0s 614us/step - loss: 0.0985 - binary_accuracy: 0.9231
    Epoch 156/200
    13/13 [==============================] - 0s 614us/step - loss: 0.0982 - binary_accuracy: 0.9231
    Epoch 157/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0979 - binary_accuracy: 0.9231
    Epoch 158/200
    13/13 [==============================] - 0s 614us/step - loss: 0.0977 - binary_accuracy: 0.9231
    Epoch 159/200
    13/13 [==============================] - 0s 460us/step - loss: 0.0974 - binary_accuracy: 0.9231
    Epoch 160/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0972 - binary_accuracy: 0.9231
    Epoch 161/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0969 - binary_accuracy: 0.9231
    Epoch 162/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0967 - binary_accuracy: 0.9231
    Epoch 163/200
    13/13 [==============================] - 0s 460us/step - loss: 0.0964 - binary_accuracy: 0.9231
    Epoch 164/200
    13/13 [==============================] - 0s 460us/step - loss: 0.0962 - binary_accuracy: 0.9231
    Epoch 165/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0959 - binary_accuracy: 0.9231
    Epoch 166/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0957 - binary_accuracy: 0.9231
    Epoch 167/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0954 - binary_accuracy: 0.9231
    Epoch 168/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0952 - binary_accuracy: 0.9231
    Epoch 169/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0950 - binary_accuracy: 0.9231
    Epoch 170/200
    13/13 [==============================] - 0s 614us/step - loss: 0.0947 - binary_accuracy: 0.9231
    Epoch 171/200
    13/13 [==============================] - 0s 614us/step - loss: 0.0945 - binary_accuracy: 0.9231
    Epoch 172/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0942 - binary_accuracy: 0.9231
    Epoch 173/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0940 - binary_accuracy: 0.9231
    Epoch 174/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0938 - binary_accuracy: 0.9231
    Epoch 175/200
    13/13 [==============================] - 0s 614us/step - loss: 0.0936 - binary_accuracy: 0.9231
    Epoch 176/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0933 - binary_accuracy: 0.9231
    Epoch 177/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0931 - binary_accuracy: 0.9231
    Epoch 178/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0929 - binary_accuracy: 0.9231
    Epoch 179/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0927 - binary_accuracy: 0.9231
    Epoch 180/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0924 - binary_accuracy: 0.9231
    Epoch 181/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0922 - binary_accuracy: 0.9231
    Epoch 182/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0920 - binary_accuracy: 0.9231
    Epoch 183/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0918 - binary_accuracy: 0.9231
    Epoch 184/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0916 - binary_accuracy: 0.9231
    Epoch 185/200
    13/13 [==============================] - 0s 614us/step - loss: 0.0914 - binary_accuracy: 0.9231
    Epoch 186/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0912 - binary_accuracy: 0.9231
    Epoch 187/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0910 - binary_accuracy: 0.9231
    Epoch 188/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0907 - binary_accuracy: 0.9231
    Epoch 189/200
    13/13 [==============================] - 0s 614us/step - loss: 0.0905 - binary_accuracy: 0.9231
    Epoch 190/200
    13/13 [==============================] - 0s 614us/step - loss: 0.0903 - binary_accuracy: 0.9231
    Epoch 191/200
    13/13 [==============================] - 0s 539us/step - loss: 0.0901 - binary_accuracy: 0.9231
    Epoch 192/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0899 - binary_accuracy: 0.9231
    Epoch 193/200
    13/13 [==============================] - 0s 535us/step - loss: 0.0897 - binary_accuracy: 1.0000
    Epoch 194/200
    13/13 [==============================] - 0s 614us/step - loss: 0.0895 - binary_accuracy: 1.0000
    Epoch 195/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0893 - binary_accuracy: 1.0000
    Epoch 196/200
    13/13 [==============================] - 0s 535us/step - loss: 0.0891 - binary_accuracy: 1.0000
    Epoch 197/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0889 - binary_accuracy: 1.0000
    Epoch 198/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0887 - binary_accuracy: 1.0000
    Epoch 199/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0885 - binary_accuracy: 1.0000
    Epoch 200/200
    13/13 [==============================] - 0s 537us/step - loss: 0.0884 - binary_accuracy: 1.0000
    




    <keras.callbacks.History at 0x215e13272e8>



임의의 숫자들의 나열을 X라고 하였을 때, 숫자 10 이상인 경우에는 1, 미만인 경우에는 0을 부여한 레이블 데이터를 y라고 해봅시다.

단순 선형 회귀때와 마찬가지로 1개의 실수인 X로부터 1개의 실수인 y를 예측하는 맵핑 관계를 가지므로 각각 1을 기재합니다. 또한 시그모이드 함수를 사용할 것이므로 activation에 sigmoid를 기재해줍니다.

옵티마이저로는 경사 하강법의 일종인 확률적 경사 하강법을 사용하였으며, 손실 함수로는 크로스 엔트로피 함수를 사용합니다. 이진 분류 문제에 크로스 엔트로피 함수를 사용할 경우에는 binary_crossentropy를 기재해주면 됩니다. 전체 데이터에 대한 훈련 횟수는 200으로 합니다.

총 200회에 걸쳐 전체 데이터에 대한 오차를 최소화하는 W와 b를 찾아내는 작업을 합니다. 약 190회부터 정확도가 100%가 나오기 시작했습니다.

실제값과 오차를 최소화하는 W와 b의 값을 가진 시그모이드 함수 그래프를 그려보겠습니다.


```python
import matplotlib.pyplot as plt
plt.plot(X, model.predict(X), 'b', X,y, 'k.')
```




    [<matplotlib.lines.Line2D at 0x215e2880da0>,
     <matplotlib.lines.Line2D at 0x215e2880ef0>]




![png](NLP_basic_07_Machine_Learning_files/NLP_basic_07_Machine_Learning_146_1.png)


이제 X값이 5보다 작은 값일 때와 X값이 10보다 클 때에 대해서 y값을 출력해봅시다.


```python
print(model.predict([1, 2, 3, 4, 4.5]))
print(model.predict([11, 21, 31, 41, 500]))
```

    [[0.21112327]
     [0.26951855]
     [0.33716246]
     [0.4122035 ]
     [0.4515741 ]]
    [[0.86907977]
     [0.9939634 ]
     [0.9997552 ]
     [0.9999901 ]
     [1.        ]]
    

X값이 5보다 작을 때는 0.5보다 작은 값을, X값이 10보다 클 때는 0.5보다 큰 값을 출력하는 것을 볼 수 있습니다.

## 5. 다중 입력에 대한 실습

독립 변수 x가 2개 이상인 선형 회귀와 로지스틱 회귀

### 1) 다중 선형 회귀

 독립 변수가 2개 이상

입력 벡터의 차원이 2이상이라고 할 수 있습니다.

$H(X) = {W_1x_1 + W_2x_2 + W_3x_3 + b}$

y를 결정하는데 있어 독립 변수가 3개인 선형 회귀를 풀어봅시다. 중간 고사, 기말 고사, 그리고 추가 점수를 어떤 공식을 통해 최종 점수를 계산한 데이터가 있습니다.

데이터 중 상위 5개의 데이터만 훈련에 사용하고, 나머지 2개는 테스트에 사용해보겠습니다.


```python
from keras.models import Sequential # 케라스의 Sequential()을 임포트
from keras.layers import Dense # 케라스의 Dense()를 임포트
from keras import optimizers # 케라스의 옵티마이저를 임포트
import numpy as np # Numpy를 임포트

X=np.array([[70,85,11],[71,89,18],[50,80,20],[99,20,10],[50,10,10]]) # 중간, 기말, 가산점
# 입력 벡터의 차원은 3입니다. 즉, input_dim은 3입니다.
y=np.array([73,82,72,57,34]) # 최종 성적
# 출력 벡터의 차원은 1입니다. 즉, output_dim은 1입니다.

model=Sequential()
model.add(Dense(1, input_dim=3, activation='linear'))
sgd=optimizers.SGD(lr=0.00001)
# 학습률(learning rate, lr)은 0.01로 합니다.
model.compile(optimizer=sgd ,loss='mse',metrics=['mse'])
# 옵티마이저는 경사하강법의 변형인 확률적 경사 하강법 sgd를 사용합니다.
# 손실 함수(Loss function)은 평균제곱오차 mse를 사용합니다.
model.fit(X,y, batch_size=1, epochs=2000, shuffle=False)
# 주어진 X와 y데이터에 대해서 오차를 최소화하는 작업을 2,000번 시도합니다.
```

    Epoch 1/2000
    5/5 [==============================] - 0s 20ms/step - loss: 14961.5208 - mean_squared_error: 14961.5208
    Epoch 2/2000
    5/5 [==============================] - 0s 591us/step - loss: 2074.0689 - mean_squared_error: 2074.0689
    Epoch 3/2000
    5/5 [==============================] - 0s 599us/step - loss: 333.5370 - mean_squared_error: 333.5370
    Epoch 4/2000
    5/5 [==============================] - 0s 791us/step - loss: 105.8379 - mean_squared_error: 105.8379
    Epoch 5/2000
    5/5 [==============================] - 0s 997us/step - loss: 72.5138 - mean_squared_error: 72.5138
    Epoch 6/2000
    5/5 [==============================] - 0s 900us/step - loss: 61.3420 - mean_squared_error: 61.3420
    Epoch 7/2000
    5/5 [==============================] - 0s 801us/step - loss: 52.5591 - mean_squared_error: 52.5591
    Epoch 8/2000
    5/5 [==============================] - 0s 798us/step - loss: 44.7519 - mean_squared_error: 44.7519
    Epoch 9/2000
    5/5 [==============================] - 0s 998us/step - loss: 38.0364 - mean_squared_error: 38.0364
    Epoch 10/2000
    5/5 [==============================] - 0s 797us/step - loss: 32.4059 - mean_squared_error: 32.4059
    Epoch 11/2000
    5/5 [==============================] - 0s 798us/step - loss: 27.7429 - mean_squared_error: 27.7429
    Epoch 12/2000
    5/5 [==============================] - 0s 798us/step - loss: 23.9013 - mean_squared_error: 23.9013
    Epoch 13/2000
    5/5 [==============================] - 0s 798us/step - loss: 20.7425 - mean_squared_error: 20.7425
    Epoch 14/2000
    5/5 [==============================] - 0s 798us/step - loss: 18.1465 - mean_squared_error: 18.1465
    Epoch 15/2000
    5/5 [==============================] - 0s 798us/step - loss: 16.0126 - mean_squared_error: 16.0126
    Epoch 16/2000
    5/5 [==============================] - 0s 997us/step - loss: 14.2576 - mean_squared_error: 14.2576
    Epoch 17/2000
    5/5 [==============================] - 0s 1ms/step - loss: 12.8131 - mean_squared_error: 12.8131
    Epoch 18/2000
    5/5 [==============================] - 0s 997us/step - loss: 11.6231 - mean_squared_error: 11.6231
    Epoch 19/2000
    5/5 [==============================] - 0s 796us/step - loss: 10.6415 - mean_squared_error: 10.6415
    Epoch 20/2000
    5/5 [==============================] - 0s 604us/step - loss: 9.8309 - mean_squared_error: 9.8309
    Epoch 21/2000
    5/5 [==============================] - 0s 598us/step - loss: 9.1604 - mean_squared_error: 9.1604
    Epoch 22/2000
    5/5 [==============================] - 0s 599us/step - loss: 8.6049 - mean_squared_error: 8.6049
    Epoch 23/2000
    5/5 [==============================] - 0s 598us/step - loss: 8.1436 - mean_squared_error: 8.1436
    Epoch 24/2000
    5/5 [==============================] - 0s 598us/step - loss: 7.7598 - mean_squared_error: 7.7598
    Epoch 25/2000
    5/5 [==============================] - 0s 598us/step - loss: 7.4396 - mean_squared_error: 7.4396
    Epoch 26/2000
    5/5 [==============================] - 0s 599us/step - loss: 7.1716 - mean_squared_error: 7.1716
    Epoch 27/2000
    5/5 [==============================] - 0s 598us/step - loss: 6.9464 - mean_squared_error: 6.9464
    Epoch 28/2000
    5/5 [==============================] - 0s 605us/step - loss: 6.7567 - mean_squared_error: 6.7567
    Epoch 29/2000
    5/5 [==============================] - 0s 598us/step - loss: 6.5959 - mean_squared_error: 6.5959
    Epoch 30/2000
    5/5 [==============================] - 0s 399us/step - loss: 6.4591 - mean_squared_error: 6.4591
    Epoch 31/2000
    5/5 [==============================] - 0s 399us/step - loss: 6.3421 - mean_squared_error: 6.3421
    Epoch 32/2000
    5/5 [==============================] - 0s 598us/step - loss: 6.2413 - mean_squared_error: 6.2413
    Epoch 33/2000
    5/5 [==============================] - 0s 599us/step - loss: 6.1539 - mean_squared_error: 6.1539
    Epoch 34/2000
    5/5 [==============================] - 0s 599us/step - loss: 6.0777 - mean_squared_error: 6.0777
    Epoch 35/2000
    5/5 [==============================] - 0s 598us/step - loss: 6.0107 - mean_squared_error: 6.0107
    Epoch 36/2000
    5/5 [==============================] - 0s 399us/step - loss: 5.9512 - mean_squared_error: 5.9512
    Epoch 37/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.8981 - mean_squared_error: 5.8981
    Epoch 38/2000
    5/5 [==============================] - 0s 599us/step - loss: 5.8503 - mean_squared_error: 5.8503
    Epoch 39/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.8069 - mean_squared_error: 5.8069
    Epoch 40/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.7671 - mean_squared_error: 5.7671
    Epoch 41/2000
    5/5 [==============================] - 0s 599us/step - loss: 5.7303 - mean_squared_error: 5.7303
    Epoch 42/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.6961 - mean_squared_error: 5.6961
    Epoch 43/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.6641 - mean_squared_error: 5.6641
    Epoch 44/2000
    5/5 [==============================] - 0s 399us/step - loss: 5.6339 - mean_squared_error: 5.6339
    Epoch 45/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.6052 - mean_squared_error: 5.6052
    Epoch 46/2000
    5/5 [==============================] - 0s 599us/step - loss: 5.5778 - mean_squared_error: 5.5778
    Epoch 47/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.5515 - mean_squared_error: 5.5515
    Epoch 48/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.5261 - mean_squared_error: 5.5261
    Epoch 49/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.5016 - mean_squared_error: 5.5016
    Epoch 50/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.4777 - mean_squared_error: 5.4777
    Epoch 51/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.4544 - mean_squared_error: 5.4544
    Epoch 52/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.4317 - mean_squared_error: 5.4317
    Epoch 53/2000
    5/5 [==============================] - 0s 596us/step - loss: 5.4094 - mean_squared_error: 5.4094
    Epoch 54/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.3875 - mean_squared_error: 5.3875
    Epoch 55/2000
    5/5 [==============================] - 0s 798us/step - loss: 5.3660 - mean_squared_error: 5.3660
    Epoch 56/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.3447 - mean_squared_error: 5.3447
    Epoch 57/2000
    5/5 [==============================] - 0s 599us/step - loss: 5.3238 - mean_squared_error: 5.3238
    Epoch 58/2000
    5/5 [==============================] - 0s 599us/step - loss: 5.3031 - mean_squared_error: 5.3031
    Epoch 59/2000
    5/5 [==============================] - 0s 597us/step - loss: 5.2826 - mean_squared_error: 5.2826
    Epoch 60/2000
    5/5 [==============================] - 0s 599us/step - loss: 5.2624 - mean_squared_error: 5.2624
    Epoch 61/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.2423 - mean_squared_error: 5.2423
    Epoch 62/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.2224 - mean_squared_error: 5.2224
    Epoch 63/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.2026 - mean_squared_error: 5.2026
    Epoch 64/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.1830 - mean_squared_error: 5.1830
    Epoch 65/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.1636 - mean_squared_error: 5.1636
    Epoch 66/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.1442 - mean_squared_error: 5.1442
    Epoch 67/2000
    5/5 [==============================] - 0s 798us/step - loss: 5.1250 - mean_squared_error: 5.1250
    Epoch 68/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.1059 - mean_squared_error: 5.1059
    Epoch 69/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.0869 - mean_squared_error: 5.0869
    Epoch 70/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.0680 - mean_squared_error: 5.0680
    Epoch 71/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.0492 - mean_squared_error: 5.0492
    Epoch 72/2000
    5/5 [==============================] - 0s 798us/step - loss: 5.0306 - mean_squared_error: 5.0306
    Epoch 73/2000
    5/5 [==============================] - 0s 598us/step - loss: 5.0120 - mean_squared_error: 5.0120
    Epoch 74/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.9935 - mean_squared_error: 4.9935
    Epoch 75/2000
    5/5 [==============================] - 0s 599us/step - loss: 4.9750 - mean_squared_error: 4.9750
    Epoch 76/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.9567 - mean_squared_error: 4.9567
    Epoch 77/2000
    5/5 [==============================] - 0s 593us/step - loss: 4.9385 - mean_squared_error: 4.9385
    Epoch 78/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.9203 - mean_squared_error: 4.9203
    Epoch 79/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.9022 - mean_squared_error: 4.9022
    Epoch 80/2000
    5/5 [==============================] - 0s 599us/step - loss: 4.8842 - mean_squared_error: 4.8842
    Epoch 81/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.8662 - mean_squared_error: 4.8662
    Epoch 82/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.8484 - mean_squared_error: 4.8484
    Epoch 83/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.8306 - mean_squared_error: 4.8306
    Epoch 84/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.8129 - mean_squared_error: 4.8129
    Epoch 85/2000
    5/5 [==============================] - 0s 599us/step - loss: 4.7952 - mean_squared_error: 4.7952
    Epoch 86/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.7776 - mean_squared_error: 4.7776
    Epoch 87/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.7601 - mean_squared_error: 4.7601
    Epoch 88/2000
    5/5 [==============================] - 0s 399us/step - loss: 4.7427 - mean_squared_error: 4.7427
    Epoch 89/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.7253 - mean_squared_error: 4.7253
    Epoch 90/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.7080 - mean_squared_error: 4.7080
    Epoch 91/2000
    5/5 [==============================] - 0s 399us/step - loss: 4.6908 - mean_squared_error: 4.6908
    Epoch 92/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.6736 - mean_squared_error: 4.6736
    Epoch 93/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.6565 - mean_squared_error: 4.6565
    Epoch 94/2000
    5/5 [==============================] - 0s 599us/step - loss: 4.6395 - mean_squared_error: 4.6395
    Epoch 95/2000
    5/5 [==============================] - 0s 599us/step - loss: 4.6225 - mean_squared_error: 4.6225
    Epoch 96/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.6056 - mean_squared_error: 4.6056
    Epoch 97/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.5887 - mean_squared_error: 4.5887
    Epoch 98/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.5719 - mean_squared_error: 4.5719
    Epoch 99/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.5552 - mean_squared_error: 4.5552
    Epoch 100/2000
    5/5 [==============================] - 0s 798us/step - loss: 4.5385 - mean_squared_error: 4.5385
    Epoch 101/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.5220 - mean_squared_error: 4.5220
    Epoch 102/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.5054 - mean_squared_error: 4.5054
    Epoch 103/2000
    5/5 [==============================] - 0s 599us/step - loss: 4.4889 - mean_squared_error: 4.4889
    Epoch 104/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.4726 - mean_squared_error: 4.4726
    Epoch 105/2000
    5/5 [==============================] - 0s 599us/step - loss: 4.4562 - mean_squared_error: 4.4562
    Epoch 106/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.4399 - mean_squared_error: 4.4399
    Epoch 107/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.4237 - mean_squared_error: 4.4237
    Epoch 108/2000
    5/5 [==============================] - 0s 798us/step - loss: 4.4075 - mean_squared_error: 4.4075
    Epoch 109/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.3914 - mean_squared_error: 4.3914
    Epoch 110/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.3753 - mean_squared_error: 4.3753
    Epoch 111/2000
    5/5 [==============================] - 0s 599us/step - loss: 4.3594 - mean_squared_error: 4.3594
    Epoch 112/2000
    5/5 [==============================] - 0s 798us/step - loss: 4.3434 - mean_squared_error: 4.3434
    Epoch 113/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.3275 - mean_squared_error: 4.3275
    Epoch 114/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.3117 - mean_squared_error: 4.3117
    Epoch 115/2000
    5/5 [==============================] - 0s 638us/step - loss: 4.2960 - mean_squared_error: 4.2960
    Epoch 116/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.2803 - mean_squared_error: 4.2803
    Epoch 117/2000
    5/5 [==============================] - 0s 599us/step - loss: 4.2646 - mean_squared_error: 4.2646
    Epoch 118/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.2490 - mean_squared_error: 4.2490
    Epoch 119/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.2335 - mean_squared_error: 4.2335
    Epoch 120/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.2181 - mean_squared_error: 4.2181
    Epoch 121/2000
    5/5 [==============================] - 0s 599us/step - loss: 4.2026 - mean_squared_error: 4.2026
    Epoch 122/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.1873 - mean_squared_error: 4.1873
    Epoch 123/2000
    5/5 [==============================] - 0s 599us/step - loss: 4.1720 - mean_squared_error: 4.1720
    Epoch 124/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.1567 - mean_squared_error: 4.1567
    Epoch 125/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.1415 - mean_squared_error: 4.1415
    Epoch 126/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.1264 - mean_squared_error: 4.1264
    Epoch 127/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.1113 - mean_squared_error: 4.1113
    Epoch 128/2000
    5/5 [==============================] - 0s 399us/step - loss: 4.0963 - mean_squared_error: 4.0963
    Epoch 129/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.0813 - mean_squared_error: 4.0813
    Epoch 130/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.0664 - mean_squared_error: 4.0664
    Epoch 131/2000
    5/5 [==============================] - 0s 598us/step - loss: 4.0516 - mean_squared_error: 4.0516
    Epoch 132/2000
    5/5 [==============================] - 0s 599us/step - loss: 4.0368 - mean_squared_error: 4.0368
    Epoch 133/2000
    5/5 [==============================] - 0s 399us/step - loss: 4.0220 - mean_squared_error: 4.0220
    Epoch 134/2000
    5/5 [==============================] - 0s 599us/step - loss: 4.0073 - mean_squared_error: 4.0073
    Epoch 135/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.9927 - mean_squared_error: 3.9927
    Epoch 136/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.9781 - mean_squared_error: 3.9781
    Epoch 137/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.9636 - mean_squared_error: 3.9636
    Epoch 138/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.9491 - mean_squared_error: 3.9491
    Epoch 139/2000
    5/5 [==============================] - 0s 798us/step - loss: 3.9347 - mean_squared_error: 3.9347
    Epoch 140/2000
    5/5 [==============================] - 0s 599us/step - loss: 3.9203 - mean_squared_error: 3.9203
    Epoch 141/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.9060 - mean_squared_error: 3.9060
    Epoch 142/2000
    5/5 [==============================] - 0s 599us/step - loss: 3.8917 - mean_squared_error: 3.8917
    Epoch 143/2000
    5/5 [==============================] - 0s 797us/step - loss: 3.8775 - mean_squared_error: 3.8775
    Epoch 144/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.8633 - mean_squared_error: 3.8633
    Epoch 145/2000
    5/5 [==============================] - 0s 599us/step - loss: 3.8492 - mean_squared_error: 3.8492
    Epoch 146/2000
    5/5 [==============================] - 0s 798us/step - loss: 3.8351 - mean_squared_error: 3.8351
    Epoch 147/2000
    5/5 [==============================] - 0s 694us/step - loss: 3.8211 - mean_squared_error: 3.8211
    Epoch 148/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.8072 - mean_squared_error: 3.8072
    Epoch 149/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.7933 - mean_squared_error: 3.7933
    Epoch 150/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.7794 - mean_squared_error: 3.7794
    Epoch 151/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.7656 - mean_squared_error: 3.7656
    Epoch 152/2000
    5/5 [==============================] - 0s 599us/step - loss: 3.7519 - mean_squared_error: 3.7519
    Epoch 153/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.7382 - mean_squared_error: 3.7382
    Epoch 154/2000
    5/5 [==============================] - 0s 599us/step - loss: 3.7245 - mean_squared_error: 3.7245
    Epoch 155/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.7109 - mean_squared_error: 3.7109
    Epoch 156/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.6974 - mean_squared_error: 3.6974
    Epoch 157/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.6839 - mean_squared_error: 3.6839
    Epoch 158/2000
    5/5 [==============================] - 0s 599us/step - loss: 3.6704 - mean_squared_error: 3.6704
    Epoch 159/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.6570 - mean_squared_error: 3.6570
    Epoch 160/2000
    5/5 [==============================] - 0s 399us/step - loss: 3.6437 - mean_squared_error: 3.6437
    Epoch 161/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.6303 - mean_squared_error: 3.6303
    Epoch 162/2000
    5/5 [==============================] - 0s 599us/step - loss: 3.6171 - mean_squared_error: 3.6171
    Epoch 163/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.6039 - mean_squared_error: 3.6039
    Epoch 164/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.5907 - mean_squared_error: 3.5907
    Epoch 165/2000
    5/5 [==============================] - 0s 599us/step - loss: 3.5776 - mean_squared_error: 3.5776
    Epoch 166/2000
    5/5 [==============================] - 0s 599us/step - loss: 3.5645 - mean_squared_error: 3.5645
    Epoch 167/2000
    5/5 [==============================] - 0s 599us/step - loss: 3.5515 - mean_squared_error: 3.5515
    Epoch 168/2000
    5/5 [==============================] - 0s 395us/step - loss: 3.5385 - mean_squared_error: 3.5385
    Epoch 169/2000
    5/5 [==============================] - 0s 599us/step - loss: 3.5256 - mean_squared_error: 3.5256
    Epoch 170/2000
    5/5 [==============================] - 0s 599us/step - loss: 3.5127 - mean_squared_error: 3.5127
    Epoch 171/2000
    5/5 [==============================] - 0s 399us/step - loss: 3.4999 - mean_squared_error: 3.4999
    Epoch 172/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.4871 - mean_squared_error: 3.4871
    Epoch 173/2000
    5/5 [==============================] - 0s 798us/step - loss: 3.4744 - mean_squared_error: 3.4744
    Epoch 174/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.4617 - mean_squared_error: 3.4617
    Epoch 175/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.4491 - mean_squared_error: 3.4491
    Epoch 176/2000
    5/5 [==============================] - 0s 798us/step - loss: 3.4365 - mean_squared_error: 3.4365
    Epoch 177/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.4239 - mean_squared_error: 3.4239
    Epoch 178/2000
    5/5 [==============================] - 0s 599us/step - loss: 3.4114 - mean_squared_error: 3.4114
    Epoch 179/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.3990 - mean_squared_error: 3.3990
    Epoch 180/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.3866 - mean_squared_error: 3.3866
    Epoch 181/2000
    5/5 [==============================] - 0s 399us/step - loss: 3.3742 - mean_squared_error: 3.3742
    Epoch 182/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.3619 - mean_squared_error: 3.3619
    Epoch 183/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.3496 - mean_squared_error: 3.3496
    Epoch 184/2000
    5/5 [==============================] - 0s 599us/step - loss: 3.3374 - mean_squared_error: 3.3374
    Epoch 185/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.3252 - mean_squared_error: 3.3252
    Epoch 186/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.3130 - mean_squared_error: 3.3130
    Epoch 187/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.3010 - mean_squared_error: 3.3010
    Epoch 188/2000
    5/5 [==============================] - 0s 599us/step - loss: 3.2889 - mean_squared_error: 3.2889
    Epoch 189/2000
    5/5 [==============================] - 0s 399us/step - loss: 3.2769 - mean_squared_error: 3.2769
    Epoch 190/2000
    5/5 [==============================] - 0s 599us/step - loss: 3.2649 - mean_squared_error: 3.2649
    Epoch 191/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.2530 - mean_squared_error: 3.2530
    Epoch 192/2000
    5/5 [==============================] - 0s 599us/step - loss: 3.2411 - mean_squared_error: 3.2411
    Epoch 193/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.2293 - mean_squared_error: 3.2293
    Epoch 194/2000
    5/5 [==============================] - 0s 399us/step - loss: 3.2175 - mean_squared_error: 3.2175
    Epoch 195/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.2058 - mean_squared_error: 3.2058
    Epoch 196/2000
    5/5 [==============================] - 0s 399us/step - loss: 3.1941 - mean_squared_error: 3.1941
    Epoch 197/2000
    5/5 [==============================] - 0s 599us/step - loss: 3.1824 - mean_squared_error: 3.1824
    Epoch 198/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.1708 - mean_squared_error: 3.1708
    Epoch 199/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.1592 - mean_squared_error: 3.1592
    Epoch 200/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.1477 - mean_squared_error: 3.1477
    Epoch 201/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.1362 - mean_squared_error: 3.1362
    Epoch 202/2000
    5/5 [==============================] - 0s 399us/step - loss: 3.1248 - mean_squared_error: 3.1248
    Epoch 203/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.1133 - mean_squared_error: 3.1133
    Epoch 204/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.1020 - mean_squared_error: 3.1020
    Epoch 205/2000
    5/5 [==============================] - 0s 798us/step - loss: 3.0907 - mean_squared_error: 3.0907
    Epoch 206/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.0794 - mean_squared_error: 3.0794
    Epoch 207/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.0681 - mean_squared_error: 3.0681
    Epoch 208/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.0569 - mean_squared_error: 3.0569
    Epoch 209/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.0458 - mean_squared_error: 3.0458
    Epoch 210/2000
    5/5 [==============================] - 0s 399us/step - loss: 3.0347 - mean_squared_error: 3.0347
    Epoch 211/2000
    5/5 [==============================] - 0s 598us/step - loss: 3.0236 - mean_squared_error: 3.0236
    Epoch 212/2000
    5/5 [==============================] - 0s 557us/step - loss: 3.0126 - mean_squared_error: 3.0126
    Epoch 213/2000
    5/5 [==============================] - 0s 604us/step - loss: 3.0016 - mean_squared_error: 3.0016
    Epoch 214/2000
    5/5 [==============================] - 0s 593us/step - loss: 2.9906 - mean_squared_error: 2.9906
    Epoch 215/2000
    5/5 [==============================] - 0s 599us/step - loss: 2.9797 - mean_squared_error: 2.9797
    Epoch 216/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.9688 - mean_squared_error: 2.9688
    Epoch 217/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.9580 - mean_squared_error: 2.9580
    Epoch 218/2000
    5/5 [==============================] - 0s 599us/step - loss: 2.9472 - mean_squared_error: 2.9472
    Epoch 219/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.9364 - mean_squared_error: 2.9364
    Epoch 220/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.9257 - mean_squared_error: 2.9257
    Epoch 221/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.9151 - mean_squared_error: 2.9151
    Epoch 222/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.9044 - mean_squared_error: 2.9044
    Epoch 223/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.8938 - mean_squared_error: 2.8938
    Epoch 224/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.8833 - mean_squared_error: 2.8833
    Epoch 225/2000
    5/5 [==============================] - 0s 599us/step - loss: 2.8727 - mean_squared_error: 2.8727
    Epoch 226/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.8623 - mean_squared_error: 2.8623
    Epoch 227/2000
    5/5 [==============================] - 0s 599us/step - loss: 2.8518 - mean_squared_error: 2.8518
    Epoch 228/2000
    5/5 [==============================] - 0s 599us/step - loss: 2.8414 - mean_squared_error: 2.8414
    Epoch 229/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.8310 - mean_squared_error: 2.8310
    Epoch 230/2000
    5/5 [==============================] - 0s 798us/step - loss: 2.8207 - mean_squared_error: 2.8207
    Epoch 231/2000
    5/5 [==============================] - 0s 798us/step - loss: 2.8104 - mean_squared_error: 2.8104
    Epoch 232/2000
    5/5 [==============================] - 0s 599us/step - loss: 2.8002 - mean_squared_error: 2.8002
    Epoch 233/2000
    5/5 [==============================] - 0s 798us/step - loss: 2.7899 - mean_squared_error: 2.7899
    Epoch 234/2000
    5/5 [==============================] - 0s 798us/step - loss: 2.7798 - mean_squared_error: 2.7798
    Epoch 235/2000
    5/5 [==============================] - 0s 599us/step - loss: 2.7696 - mean_squared_error: 2.7696
    Epoch 236/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.7595 - mean_squared_error: 2.7595
    Epoch 237/2000
    5/5 [==============================] - 0s 599us/step - loss: 2.7494 - mean_squared_error: 2.7494
    Epoch 238/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.7394 - mean_squared_error: 2.7394
    Epoch 239/2000
    5/5 [==============================] - 0s 798us/step - loss: 2.7294 - mean_squared_error: 2.7294
    Epoch 240/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.7195 - mean_squared_error: 2.7195
    Epoch 241/2000
    5/5 [==============================] - 0s 599us/step - loss: 2.7096 - mean_squared_error: 2.7096
    Epoch 242/2000
    5/5 [==============================] - 0s 399us/step - loss: 2.6997 - mean_squared_error: 2.6997
    Epoch 243/2000
    5/5 [==============================] - 0s 599us/step - loss: 2.6898 - mean_squared_error: 2.6898
    Epoch 244/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.6800 - mean_squared_error: 2.6800
    Epoch 245/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.6702 - mean_squared_error: 2.6702
    Epoch 246/2000
    5/5 [==============================] - 0s 399us/step - loss: 2.6605 - mean_squared_error: 2.6605
    Epoch 247/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.6508 - mean_squared_error: 2.6508
    Epoch 248/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.6411 - mean_squared_error: 2.6411
    Epoch 249/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.6315 - mean_squared_error: 2.6315
    Epoch 250/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.6219 - mean_squared_error: 2.6219
    Epoch 251/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.6123 - mean_squared_error: 2.6123
    Epoch 252/2000
    5/5 [==============================] - 0s 798us/step - loss: 2.6028 - mean_squared_error: 2.6028
    Epoch 253/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.5933 - mean_squared_error: 2.5933
    Epoch 254/2000
    5/5 [==============================] - 0s 798us/step - loss: 2.5838 - mean_squared_error: 2.5838
    Epoch 255/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.5744 - mean_squared_error: 2.5744
    Epoch 256/2000
    5/5 [==============================] - 0s 599us/step - loss: 2.5650 - mean_squared_error: 2.5650
    Epoch 257/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.5557 - mean_squared_error: 2.5557
    Epoch 258/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.5464 - mean_squared_error: 2.5464
    Epoch 259/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.5371 - mean_squared_error: 2.5371
    Epoch 260/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.5278 - mean_squared_error: 2.5278
    Epoch 261/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.5186 - mean_squared_error: 2.5186
    Epoch 262/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.5094 - mean_squared_error: 2.5094
    Epoch 263/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.5003 - mean_squared_error: 2.5003
    Epoch 264/2000
    5/5 [==============================] - 0s 399us/step - loss: 2.4911 - mean_squared_error: 2.4911
    Epoch 265/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.4821 - mean_squared_error: 2.4821
    Epoch 266/2000
    5/5 [==============================] - 0s 399us/step - loss: 2.4730 - mean_squared_error: 2.4730
    Epoch 267/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.4640 - mean_squared_error: 2.4640
    Epoch 268/2000
    5/5 [==============================] - 0s 599us/step - loss: 2.4550 - mean_squared_error: 2.4550
    Epoch 269/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.4461 - mean_squared_error: 2.4461
    Epoch 270/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.4371 - mean_squared_error: 2.4371
    Epoch 271/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.4283 - mean_squared_error: 2.4283
    Epoch 272/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.4194 - mean_squared_error: 2.4194
    Epoch 273/2000
    5/5 [==============================] - 0s 399us/step - loss: 2.4106 - mean_squared_error: 2.4106
    Epoch 274/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.4018 - mean_squared_error: 2.4018
    Epoch 275/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.3930 - mean_squared_error: 2.3930
    Epoch 276/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.3843 - mean_squared_error: 2.3843
    Epoch 277/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.3756 - mean_squared_error: 2.3756
    Epoch 278/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.3670 - mean_squared_error: 2.3670
    Epoch 279/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.3583 - mean_squared_error: 2.3583
    Epoch 280/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.3497 - mean_squared_error: 2.3497
    Epoch 281/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.3412 - mean_squared_error: 2.3412
    Epoch 282/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.3326 - mean_squared_error: 2.3326
    Epoch 283/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.3241 - mean_squared_error: 2.3241
    Epoch 284/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.3157 - mean_squared_error: 2.3157
    Epoch 285/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.3072 - mean_squared_error: 2.3072
    Epoch 286/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.2988 - mean_squared_error: 2.2988
    Epoch 287/2000
    5/5 [==============================] - 0s 399us/step - loss: 2.2904 - mean_squared_error: 2.2904
    Epoch 288/2000
    5/5 [==============================] - 0s 599us/step - loss: 2.2821 - mean_squared_error: 2.2821
    Epoch 289/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.2738 - mean_squared_error: 2.2738
    Epoch 290/2000
    5/5 [==============================] - 0s 599us/step - loss: 2.2655 - mean_squared_error: 2.2655
    Epoch 291/2000
    5/5 [==============================] - 0s 399us/step - loss: 2.2572 - mean_squared_error: 2.2572
    Epoch 292/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.2490 - mean_squared_error: 2.2490
    Epoch 293/2000
    5/5 [==============================] - 0s 599us/step - loss: 2.2408 - mean_squared_error: 2.2408
    Epoch 294/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.2326 - mean_squared_error: 2.2326
    Epoch 295/2000
    5/5 [==============================] - 0s 599us/step - loss: 2.2245 - mean_squared_error: 2.2245
    Epoch 296/2000
    5/5 [==============================] - 0s 798us/step - loss: 2.2164 - mean_squared_error: 2.2164
    Epoch 297/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.2083 - mean_squared_error: 2.2083
    Epoch 298/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.2003 - mean_squared_error: 2.2003
    Epoch 299/2000
    5/5 [==============================] - 0s 399us/step - loss: 2.1922 - mean_squared_error: 2.1922
    Epoch 300/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.1842 - mean_squared_error: 2.1842
    Epoch 301/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.1763 - mean_squared_error: 2.1763
    Epoch 302/2000
    5/5 [==============================] - 0s 599us/step - loss: 2.1684 - mean_squared_error: 2.1684
    Epoch 303/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.1605 - mean_squared_error: 2.1605
    Epoch 304/2000
    5/5 [==============================] - 0s 599us/step - loss: 2.1526 - mean_squared_error: 2.1526
    Epoch 305/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.1447 - mean_squared_error: 2.1447
    Epoch 306/2000
    5/5 [==============================] - 0s 587us/step - loss: 2.1369 - mean_squared_error: 2.1369
    Epoch 307/2000
    5/5 [==============================] - 0s 593us/step - loss: 2.1292 - mean_squared_error: 2.1292
    Epoch 308/2000
    5/5 [==============================] - 0s 798us/step - loss: 2.1214 - mean_squared_error: 2.1214
    Epoch 309/2000
    5/5 [==============================] - 0s 599us/step - loss: 2.1137 - mean_squared_error: 2.1137
    Epoch 310/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.1060 - mean_squared_error: 2.1060
    Epoch 311/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.0983 - mean_squared_error: 2.0983
    Epoch 312/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.0906 - mean_squared_error: 2.0906
    Epoch 313/2000
    5/5 [==============================] - 0s 599us/step - loss: 2.0830 - mean_squared_error: 2.0830
    Epoch 314/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.0754 - mean_squared_error: 2.0754
    Epoch 315/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.0679 - mean_squared_error: 2.0679
    Epoch 316/2000
    5/5 [==============================] - 0s 798us/step - loss: 2.0603 - mean_squared_error: 2.0603
    Epoch 317/2000
    5/5 [==============================] - 0s 798us/step - loss: 2.0528 - mean_squared_error: 2.0528
    Epoch 318/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.0454 - mean_squared_error: 2.0454
    Epoch 319/2000
    5/5 [==============================] - 0s 591us/step - loss: 2.0379 - mean_squared_error: 2.0379
    Epoch 320/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.0305 - mean_squared_error: 2.0305
    Epoch 321/2000
    5/5 [==============================] - 0s 598us/step - loss: 2.0231 - mean_squared_error: 2.0231
    Epoch 322/2000
    5/5 [==============================] - 0s 399us/step - loss: 2.0157 - mean_squared_error: 2.0157
    Epoch 323/2000
    5/5 [==============================] - 0s 599us/step - loss: 2.0084 - mean_squared_error: 2.0084
    Epoch 324/2000
    5/5 [==============================] - 0s 599us/step - loss: 2.0011 - mean_squared_error: 2.0011
    Epoch 325/2000
    5/5 [==============================] - 0s 798us/step - loss: 1.9938 - mean_squared_error: 1.9938
    Epoch 326/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.9865 - mean_squared_error: 1.9865
    Epoch 327/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.9793 - mean_squared_error: 1.9793
    Epoch 328/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.9721 - mean_squared_error: 1.9721
    Epoch 329/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.9649 - mean_squared_error: 1.9649
    Epoch 330/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.9577 - mean_squared_error: 1.9577
    Epoch 331/2000
    5/5 [==============================] - 0s 798us/step - loss: 1.9506 - mean_squared_error: 1.9506
    Epoch 332/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.9435 - mean_squared_error: 1.9435
    Epoch 333/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.9364 - mean_squared_error: 1.9364
    Epoch 334/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.9294 - mean_squared_error: 1.9294
    Epoch 335/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.9224 - mean_squared_error: 1.9224
    Epoch 336/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.9154 - mean_squared_error: 1.9154
    Epoch 337/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.9084 - mean_squared_error: 1.9084
    Epoch 338/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.9014 - mean_squared_error: 1.9014
    Epoch 339/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.8945 - mean_squared_error: 1.8945
    Epoch 340/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.8876 - mean_squared_error: 1.8876
    Epoch 341/2000
    5/5 [==============================] - 0s 798us/step - loss: 1.8808 - mean_squared_error: 1.8808
    Epoch 342/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.8739 - mean_squared_error: 1.8739
    Epoch 343/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.8671 - mean_squared_error: 1.8671
    Epoch 344/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.8603 - mean_squared_error: 1.8603
    Epoch 345/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.8535 - mean_squared_error: 1.8535
    Epoch 346/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.8468 - mean_squared_error: 1.8468
    Epoch 347/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.8400 - mean_squared_error: 1.8400
    Epoch 348/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.8333 - mean_squared_error: 1.8333
    Epoch 349/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.8267 - mean_squared_error: 1.8267
    Epoch 350/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.8200 - mean_squared_error: 1.8200
    Epoch 351/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.8134 - mean_squared_error: 1.8134
    Epoch 352/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.8068 - mean_squared_error: 1.8068
    Epoch 353/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.8002 - mean_squared_error: 1.8002
    Epoch 354/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.7937 - mean_squared_error: 1.7937
    Epoch 355/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.7872 - mean_squared_error: 1.7872
    Epoch 356/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.7806 - mean_squared_error: 1.7806
    Epoch 357/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.7742 - mean_squared_error: 1.7742
    Epoch 358/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.7677 - mean_squared_error: 1.7677
    Epoch 359/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.7613 - mean_squared_error: 1.7613
    Epoch 360/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.7549 - mean_squared_error: 1.7549
    Epoch 361/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.7485 - mean_squared_error: 1.7485
    Epoch 362/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.7421 - mean_squared_error: 1.7421
    Epoch 363/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.7358 - mean_squared_error: 1.7358
    Epoch 364/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.7295 - mean_squared_error: 1.7295
    Epoch 365/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.7232 - mean_squared_error: 1.7232
    Epoch 366/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.7169 - mean_squared_error: 1.7169
    Epoch 367/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.7107 - mean_squared_error: 1.7107
    Epoch 368/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.7044 - mean_squared_error: 1.7044
    Epoch 369/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.6982 - mean_squared_error: 1.6982
    Epoch 370/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.6921 - mean_squared_error: 1.6921
    Epoch 371/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.6859 - mean_squared_error: 1.6859
    Epoch 372/2000
    5/5 [==============================] - 0s 798us/step - loss: 1.6798 - mean_squared_error: 1.6798
    Epoch 373/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.6737 - mean_squared_error: 1.6737
    Epoch 374/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.6676 - mean_squared_error: 1.6676
    Epoch 375/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.6615 - mean_squared_error: 1.6615
    Epoch 376/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.6555 - mean_squared_error: 1.6555
    Epoch 377/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.6494 - mean_squared_error: 1.6494
    Epoch 378/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.6435 - mean_squared_error: 1.6435
    Epoch 379/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.6375 - mean_squared_error: 1.6375
    Epoch 380/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.6315 - mean_squared_error: 1.6315
    Epoch 381/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.6256 - mean_squared_error: 1.6256
    Epoch 382/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.6197 - mean_squared_error: 1.6197
    Epoch 383/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.6138 - mean_squared_error: 1.6138
    Epoch 384/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.6079 - mean_squared_error: 1.6079
    Epoch 385/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.6021 - mean_squared_error: 1.6021
    Epoch 386/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.5962 - mean_squared_error: 1.5962
    Epoch 387/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.5904 - mean_squared_error: 1.5904
    Epoch 388/2000
    5/5 [==============================] - 0s 798us/step - loss: 1.5847 - mean_squared_error: 1.5847
    Epoch 389/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.5789 - mean_squared_error: 1.5789
    Epoch 390/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.5732 - mean_squared_error: 1.5732
    Epoch 391/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.5674 - mean_squared_error: 1.5674
    Epoch 392/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.5617 - mean_squared_error: 1.5617
    Epoch 393/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.5561 - mean_squared_error: 1.5561
    Epoch 394/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.5504 - mean_squared_error: 1.5504
    Epoch 395/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.5448 - mean_squared_error: 1.5448
    Epoch 396/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.5392 - mean_squared_error: 1.5392
    Epoch 397/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.5336 - mean_squared_error: 1.5336
    Epoch 398/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.5280 - mean_squared_error: 1.5280
    Epoch 399/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.5224 - mean_squared_error: 1.5224
    Epoch 400/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.5169 - mean_squared_error: 1.5169
    Epoch 401/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.5114 - mean_squared_error: 1.5114
    Epoch 402/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.5059 - mean_squared_error: 1.5059
    Epoch 403/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.5004 - mean_squared_error: 1.5004
    Epoch 404/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.4950 - mean_squared_error: 1.4950
    Epoch 405/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.4895 - mean_squared_error: 1.4895
    Epoch 406/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.4841 - mean_squared_error: 1.4841
    Epoch 407/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.4787 - mean_squared_error: 1.4787
    Epoch 408/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.4734 - mean_squared_error: 1.4734
    Epoch 409/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.4680 - mean_squared_error: 1.4680
    Epoch 410/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.4627 - mean_squared_error: 1.4627
    Epoch 411/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.4574 - mean_squared_error: 1.4574
    Epoch 412/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.4521 - mean_squared_error: 1.4521
    Epoch 413/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.4468 - mean_squared_error: 1.4468
    Epoch 414/2000
    5/5 [==============================] - 0s 604us/step - loss: 1.4415 - mean_squared_error: 1.4415
    Epoch 415/2000
    5/5 [==============================] - 0s 404us/step - loss: 1.4363 - mean_squared_error: 1.4363
    Epoch 416/2000
    5/5 [==============================] - 0s 605us/step - loss: 1.4311 - mean_squared_error: 1.4311
    Epoch 417/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.4259 - mean_squared_error: 1.4259
    Epoch 418/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.4207 - mean_squared_error: 1.4207
    Epoch 419/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.4155 - mean_squared_error: 1.4155
    Epoch 420/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.4104 - mean_squared_error: 1.4104
    Epoch 421/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.4053 - mean_squared_error: 1.4053
    Epoch 422/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.4002 - mean_squared_error: 1.4002
    Epoch 423/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.3951 - mean_squared_error: 1.3951
    Epoch 424/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.3900 - mean_squared_error: 1.3900
    Epoch 425/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.3850 - mean_squared_error: 1.3850
    Epoch 426/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.3799 - mean_squared_error: 1.3799
    Epoch 427/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.3749 - mean_squared_error: 1.3749
    Epoch 428/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.3699 - mean_squared_error: 1.3699
    Epoch 429/2000
    5/5 [==============================] - 0s 453us/step - loss: 1.3650 - mean_squared_error: 1.3650
    Epoch 430/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.3600 - mean_squared_error: 1.3600
    Epoch 431/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.3551 - mean_squared_error: 1.3551
    Epoch 432/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.3501 - mean_squared_error: 1.3501
    Epoch 433/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.3452 - mean_squared_error: 1.3452
    Epoch 434/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.3404 - mean_squared_error: 1.3404
    Epoch 435/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.3355 - mean_squared_error: 1.3355
    Epoch 436/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.3307 - mean_squared_error: 1.3307
    Epoch 437/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.3258 - mean_squared_error: 1.3258
    Epoch 438/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.3210 - mean_squared_error: 1.3210
    Epoch 439/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.3162 - mean_squared_error: 1.3162
    Epoch 440/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.3114 - mean_squared_error: 1.3114
    Epoch 441/2000
    5/5 [==============================] - 0s 798us/step - loss: 1.3067 - mean_squared_error: 1.3067
    Epoch 442/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.3019 - mean_squared_error: 1.3019
    Epoch 443/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.2972 - mean_squared_error: 1.2972
    Epoch 444/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.2925 - mean_squared_error: 1.2925
    Epoch 445/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.2878 - mean_squared_error: 1.2878
    Epoch 446/2000
    5/5 [==============================] - 0s 798us/step - loss: 1.2831 - mean_squared_error: 1.2831
    Epoch 447/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.2785 - mean_squared_error: 1.2785
    Epoch 448/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.2738 - mean_squared_error: 1.2738
    Epoch 449/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.2692 - mean_squared_error: 1.2692
    Epoch 450/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.2646 - mean_squared_error: 1.2646
    Epoch 451/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.2600 - mean_squared_error: 1.2600
    Epoch 452/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.2555 - mean_squared_error: 1.2555
    Epoch 453/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.2509 - mean_squared_error: 1.2509
    Epoch 454/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.2464 - mean_squared_error: 1.2464
    Epoch 455/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.2418 - mean_squared_error: 1.2418
    Epoch 456/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.2373 - mean_squared_error: 1.2373
    Epoch 457/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.2329 - mean_squared_error: 1.2329
    Epoch 458/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.2284 - mean_squared_error: 1.2284
    Epoch 459/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.2239 - mean_squared_error: 1.2239
    Epoch 460/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.2195 - mean_squared_error: 1.2195
    Epoch 461/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.2151 - mean_squared_error: 1.2151
    Epoch 462/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.2107 - mean_squared_error: 1.2107
    Epoch 463/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.2063 - mean_squared_error: 1.2063
    Epoch 464/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.2019 - mean_squared_error: 1.2019
    Epoch 465/2000
    5/5 [==============================] - 0s 798us/step - loss: 1.1975 - mean_squared_error: 1.1975
    Epoch 466/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.1932 - mean_squared_error: 1.1932
    Epoch 467/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.1889 - mean_squared_error: 1.1889
    Epoch 468/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.1846 - mean_squared_error: 1.1846
    Epoch 469/2000
    5/5 [==============================] - 0s 398us/step - loss: 1.1803 - mean_squared_error: 1.1803
    Epoch 470/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.1760 - mean_squared_error: 1.1760
    Epoch 471/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.1717 - mean_squared_error: 1.1717
    Epoch 472/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.1675 - mean_squared_error: 1.1675
    Epoch 473/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.1632 - mean_squared_error: 1.1632
    Epoch 474/2000
    5/5 [==============================] - 0s 798us/step - loss: 1.1590 - mean_squared_error: 1.1590
    Epoch 475/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.1548 - mean_squared_error: 1.1548
    Epoch 476/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.1506 - mean_squared_error: 1.1506
    Epoch 477/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.1465 - mean_squared_error: 1.1465
    Epoch 478/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.1423 - mean_squared_error: 1.1423
    Epoch 479/2000
    5/5 [==============================] - 0s 798us/step - loss: 1.1382 - mean_squared_error: 1.1382
    Epoch 480/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.1340 - mean_squared_error: 1.1340
    Epoch 481/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.1299 - mean_squared_error: 1.1299
    Epoch 482/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.1258 - mean_squared_error: 1.1258
    Epoch 483/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.1218 - mean_squared_error: 1.1218
    Epoch 484/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.1177 - mean_squared_error: 1.1177
    Epoch 485/2000
    5/5 [==============================] - 0s 798us/step - loss: 1.1136 - mean_squared_error: 1.1136
    Epoch 486/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.1096 - mean_squared_error: 1.1096
    Epoch 487/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.1056 - mean_squared_error: 1.1056
    Epoch 488/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.1016 - mean_squared_error: 1.1016
    Epoch 489/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.0976 - mean_squared_error: 1.0976
    Epoch 490/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.0936 - mean_squared_error: 1.0936
    Epoch 491/2000
    5/5 [==============================] - 0s 798us/step - loss: 1.0897 - mean_squared_error: 1.0897
    Epoch 492/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.0857 - mean_squared_error: 1.0857
    Epoch 493/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.0818 - mean_squared_error: 1.0818
    Epoch 494/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.0779 - mean_squared_error: 1.0779
    Epoch 495/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.0740 - mean_squared_error: 1.0740
    Epoch 496/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.0701 - mean_squared_error: 1.0701
    Epoch 497/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.0662 - mean_squared_error: 1.0662
    Epoch 498/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.0623 - mean_squared_error: 1.0623
    Epoch 499/2000
    5/5 [==============================] - 0s 798us/step - loss: 1.0585 - mean_squared_error: 1.0585
    Epoch 500/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.0547 - mean_squared_error: 1.0547
    Epoch 501/2000
    5/5 [==============================] - 0s 798us/step - loss: 1.0508 - mean_squared_error: 1.0508
    Epoch 502/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.0470 - mean_squared_error: 1.0470
    Epoch 503/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.0432 - mean_squared_error: 1.0432
    Epoch 504/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.0395 - mean_squared_error: 1.0395
    Epoch 505/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.0357 - mean_squared_error: 1.0357
    Epoch 506/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.0320 - mean_squared_error: 1.0320
    Epoch 507/2000
    5/5 [==============================] - 0s 599us/step - loss: 1.0282 - mean_squared_error: 1.0282
    Epoch 508/2000
    5/5 [==============================] - 0s 798us/step - loss: 1.0245 - mean_squared_error: 1.0245
    Epoch 509/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.0208 - mean_squared_error: 1.0208
    Epoch 510/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.0171 - mean_squared_error: 1.0171
    Epoch 511/2000
    5/5 [==============================] - 0s 598us/step - loss: 1.0134 - mean_squared_error: 1.0134
    Epoch 512/2000
    5/5 [==============================] - 0s 798us/step - loss: 1.0098 - mean_squared_error: 1.0098
    Epoch 513/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.0061 - mean_squared_error: 1.0061
    Epoch 514/2000
    5/5 [==============================] - 0s 399us/step - loss: 1.0025 - mean_squared_error: 1.0025
    Epoch 515/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.9988 - mean_squared_error: 0.9988
    Epoch 516/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.9952 - mean_squared_error: 0.9952
    Epoch 517/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.9916 - mean_squared_error: 0.9916
    Epoch 518/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.9880 - mean_squared_error: 0.9880
    Epoch 519/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.9845 - mean_squared_error: 0.9845
    Epoch 520/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.9809 - mean_squared_error: 0.9809
    Epoch 521/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.9773 - mean_squared_error: 0.9773
    Epoch 522/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.9738 - mean_squared_error: 0.9738
    Epoch 523/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.9703 - mean_squared_error: 0.9703
    Epoch 524/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.9668 - mean_squared_error: 0.9668
    Epoch 525/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.9633 - mean_squared_error: 0.9633
    Epoch 526/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.9598 - mean_squared_error: 0.9598
    Epoch 527/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.9563 - mean_squared_error: 0.9563
    Epoch 528/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.9529 - mean_squared_error: 0.9529
    Epoch 529/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.9494 - mean_squared_error: 0.9494
    Epoch 530/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.9460 - mean_squared_error: 0.9460
    Epoch 531/2000
    5/5 [==============================] - 0s 613us/step - loss: 0.9426 - mean_squared_error: 0.9426
    Epoch 532/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.9392 - mean_squared_error: 0.9392
    Epoch 533/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.9358 - mean_squared_error: 0.9358
    Epoch 534/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.9324 - mean_squared_error: 0.9324
    Epoch 535/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.9290 - mean_squared_error: 0.9290
    Epoch 536/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.9257 - mean_squared_error: 0.9257
    Epoch 537/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.9223 - mean_squared_error: 0.9223
    Epoch 538/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.9190 - mean_squared_error: 0.9190
    Epoch 539/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.9157 - mean_squared_error: 0.9157
    Epoch 540/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.9124 - mean_squared_error: 0.9124
    Epoch 541/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.9091 - mean_squared_error: 0.9091
    Epoch 542/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.9058 - mean_squared_error: 0.9058
    Epoch 543/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.9025 - mean_squared_error: 0.9025
    Epoch 544/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.8992 - mean_squared_error: 0.8992
    Epoch 545/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.8960 - mean_squared_error: 0.8960
    Epoch 546/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.8928 - mean_squared_error: 0.8928
    Epoch 547/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.8895 - mean_squared_error: 0.8895
    Epoch 548/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.8863 - mean_squared_error: 0.8863
    Epoch 549/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.8831 - mean_squared_error: 0.8831
    Epoch 550/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.8799 - mean_squared_error: 0.8799
    Epoch 551/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.8768 - mean_squared_error: 0.8768
    Epoch 552/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.8736 - mean_squared_error: 0.8736
    Epoch 553/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.8704 - mean_squared_error: 0.8704
    Epoch 554/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.8673 - mean_squared_error: 0.8673
    Epoch 555/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.8642 - mean_squared_error: 0.8642
    Epoch 556/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.8610 - mean_squared_error: 0.8610
    Epoch 557/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.8579 - mean_squared_error: 0.8579
    Epoch 558/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.8548 - mean_squared_error: 0.8548
    Epoch 559/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.8518 - mean_squared_error: 0.8518
    Epoch 560/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.8487 - mean_squared_error: 0.8487
    Epoch 561/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.8456 - mean_squared_error: 0.8456
    Epoch 562/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.8426 - mean_squared_error: 0.8426
    Epoch 563/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.8395 - mean_squared_error: 0.8395
    Epoch 564/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.8365 - mean_squared_error: 0.8365
    Epoch 565/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.8335 - mean_squared_error: 0.8335
    Epoch 566/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.8305 - mean_squared_error: 0.8305
    Epoch 567/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.8275 - mean_squared_error: 0.8275
    Epoch 568/2000
    5/5 [==============================] - 0s 534us/step - loss: 0.8245 - mean_squared_error: 0.8245
    Epoch 569/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.8215 - mean_squared_error: 0.8215
    Epoch 570/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.8186 - mean_squared_error: 0.8186
    Epoch 571/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.8156 - mean_squared_error: 0.8156
    Epoch 572/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.8127 - mean_squared_error: 0.8127
    Epoch 573/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.8097 - mean_squared_error: 0.8097
    Epoch 574/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.8068 - mean_squared_error: 0.8068
    Epoch 575/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.8039 - mean_squared_error: 0.8039
    Epoch 576/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.8010 - mean_squared_error: 0.8010
    Epoch 577/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.7981 - mean_squared_error: 0.7981
    Epoch 578/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.7952 - mean_squared_error: 0.7952
    Epoch 579/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.7924 - mean_squared_error: 0.7924
    Epoch 580/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.7895 - mean_squared_error: 0.7895
    Epoch 581/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.7867 - mean_squared_error: 0.7867
    Epoch 582/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.7838 - mean_squared_error: 0.7838
    Epoch 583/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.7810 - mean_squared_error: 0.7810
    Epoch 584/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.7782 - mean_squared_error: 0.7782
    Epoch 585/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.7754 - mean_squared_error: 0.7754
    Epoch 586/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.7726 - mean_squared_error: 0.7726
    Epoch 587/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.7698 - mean_squared_error: 0.7698
    Epoch 588/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.7670 - mean_squared_error: 0.7670
    Epoch 589/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.7643 - mean_squared_error: 0.7643
    Epoch 590/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.7615 - mean_squared_error: 0.7615
    Epoch 591/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.7588 - mean_squared_error: 0.7588
    Epoch 592/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.7560 - mean_squared_error: 0.7560
    Epoch 593/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.7533 - mean_squared_error: 0.7533
    Epoch 594/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.7506 - mean_squared_error: 0.7506
    Epoch 595/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.7479 - mean_squared_error: 0.7479
    Epoch 596/2000
    5/5 [==============================] - 0s 597us/step - loss: 0.7452 - mean_squared_error: 0.7452
    Epoch 597/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.7425 - mean_squared_error: 0.7425
    Epoch 598/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.7399 - mean_squared_error: 0.7399
    Epoch 599/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.7372 - mean_squared_error: 0.7372
    Epoch 600/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.7345 - mean_squared_error: 0.7345
    Epoch 601/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.7319 - mean_squared_error: 0.7319
    Epoch 602/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.7293 - mean_squared_error: 0.7293
    Epoch 603/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.7266 - mean_squared_error: 0.7266
    Epoch 604/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.7240 - mean_squared_error: 0.7240
    Epoch 605/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.7214 - mean_squared_error: 0.7214
    Epoch 606/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.7188 - mean_squared_error: 0.7188
    Epoch 607/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.7162 - mean_squared_error: 0.7162
    Epoch 608/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.7137 - mean_squared_error: 0.7137
    Epoch 609/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.7111 - mean_squared_error: 0.7111
    Epoch 610/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.7085 - mean_squared_error: 0.7085
    Epoch 611/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.7060 - mean_squared_error: 0.7060
    Epoch 612/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.7034 - mean_squared_error: 0.7034
    Epoch 613/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.7009 - mean_squared_error: 0.7009
    Epoch 614/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.6984 - mean_squared_error: 0.6984
    Epoch 615/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.6959 - mean_squared_error: 0.6959
    Epoch 616/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.6934 - mean_squared_error: 0.6934
    Epoch 617/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.6909 - mean_squared_error: 0.6909
    Epoch 618/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.6884 - mean_squared_error: 0.6884
    Epoch 619/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.6859 - mean_squared_error: 0.6859
    Epoch 620/2000
    5/5 [==============================] - 0s 595us/step - loss: 0.6835 - mean_squared_error: 0.6835
    Epoch 621/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.6810 - mean_squared_error: 0.6810
    Epoch 622/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.6785 - mean_squared_error: 0.6785
    Epoch 623/2000
    5/5 [==============================] - 0s 700us/step - loss: 0.6761 - mean_squared_error: 0.6761
    Epoch 624/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.6737 - mean_squared_error: 0.6737
    Epoch 625/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.6713 - mean_squared_error: 0.6713
    Epoch 626/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.6688 - mean_squared_error: 0.6688
    Epoch 627/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.6664 - mean_squared_error: 0.6664
    Epoch 628/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.6640 - mean_squared_error: 0.6640
    Epoch 629/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.6617 - mean_squared_error: 0.6617
    Epoch 630/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.6593 - mean_squared_error: 0.6593
    Epoch 631/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.6569 - mean_squared_error: 0.6569
    Epoch 632/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.6546 - mean_squared_error: 0.6546
    Epoch 633/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.6522 - mean_squared_error: 0.6522
    Epoch 634/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.6499 - mean_squared_error: 0.6499
    Epoch 635/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.6475 - mean_squared_error: 0.6475
    Epoch 636/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.6452 - mean_squared_error: 0.6452
    Epoch 637/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.6429 - mean_squared_error: 0.6429
    Epoch 638/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.6406 - mean_squared_error: 0.6406
    Epoch 639/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.6383 - mean_squared_error: 0.6383
    Epoch 640/2000
    5/5 [==============================] - 0s 665us/step - loss: 0.6360 - mean_squared_error: 0.6360
    Epoch 641/2000
    5/5 [==============================] - 0s 616us/step - loss: 0.6337 - mean_squared_error: 0.6337
    Epoch 642/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.6314 - mean_squared_error: 0.6314
    Epoch 643/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.6292 - mean_squared_error: 0.6292
    Epoch 644/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.6269 - mean_squared_error: 0.6269
    Epoch 645/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.6247 - mean_squared_error: 0.6247
    Epoch 646/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.6224 - mean_squared_error: 0.6224
    Epoch 647/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.6202 - mean_squared_error: 0.6202
    Epoch 648/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.6180 - mean_squared_error: 0.6180
    Epoch 649/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.6157 - mean_squared_error: 0.6157
    Epoch 650/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.6135 - mean_squared_error: 0.6135
    Epoch 651/2000
    5/5 [==============================] - 0s 597us/step - loss: 0.6113 - mean_squared_error: 0.6113
    Epoch 652/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.6091 - mean_squared_error: 0.6091
    Epoch 653/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.6070 - mean_squared_error: 0.6070
    Epoch 654/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.6048 - mean_squared_error: 0.6048
    Epoch 655/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.6026 - mean_squared_error: 0.6026
    Epoch 656/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.6005 - mean_squared_error: 0.6005
    Epoch 657/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.5983 - mean_squared_error: 0.5983
    Epoch 658/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5962 - mean_squared_error: 0.5962
    Epoch 659/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.5940 - mean_squared_error: 0.5940
    Epoch 660/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.5919 - mean_squared_error: 0.5919
    Epoch 661/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5898 - mean_squared_error: 0.5898
    Epoch 662/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5877 - mean_squared_error: 0.5877
    Epoch 663/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5856 - mean_squared_error: 0.5856
    Epoch 664/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.5835 - mean_squared_error: 0.5835
    Epoch 665/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5814 - mean_squared_error: 0.5814
    Epoch 666/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.5793 - mean_squared_error: 0.5793
    Epoch 667/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5772 - mean_squared_error: 0.5772
    Epoch 668/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.5751 - mean_squared_error: 0.5751
    Epoch 669/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5731 - mean_squared_error: 0.5731
    Epoch 670/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5710 - mean_squared_error: 0.5710
    Epoch 671/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5690 - mean_squared_error: 0.5690
    Epoch 672/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.5669 - mean_squared_error: 0.5669
    Epoch 673/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5649 - mean_squared_error: 0.5649
    Epoch 674/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5629 - mean_squared_error: 0.5629
    Epoch 675/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.5609 - mean_squared_error: 0.5609
    Epoch 676/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5589 - mean_squared_error: 0.5589
    Epoch 677/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.5569 - mean_squared_error: 0.5569
    Epoch 678/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5549 - mean_squared_error: 0.5549
    Epoch 679/2000
    5/5 [==============================] - 0s 700us/step - loss: 0.5529 - mean_squared_error: 0.5529
    Epoch 680/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.5509 - mean_squared_error: 0.5509
    Epoch 681/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5490 - mean_squared_error: 0.5490
    Epoch 682/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5470 - mean_squared_error: 0.5470
    Epoch 683/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5450 - mean_squared_error: 0.5450
    Epoch 684/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5431 - mean_squared_error: 0.5431
    Epoch 685/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.5411 - mean_squared_error: 0.5411
    Epoch 686/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.5392 - mean_squared_error: 0.5392
    Epoch 687/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5373 - mean_squared_error: 0.5373
    Epoch 688/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5354 - mean_squared_error: 0.5354
    Epoch 689/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.5334 - mean_squared_error: 0.5334
    Epoch 690/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.5315 - mean_squared_error: 0.5315
    Epoch 691/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.5296 - mean_squared_error: 0.5296
    Epoch 692/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5277 - mean_squared_error: 0.5277
    Epoch 693/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5259 - mean_squared_error: 0.5259
    Epoch 694/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5240 - mean_squared_error: 0.5240
    Epoch 695/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5221 - mean_squared_error: 0.5221
    Epoch 696/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5202 - mean_squared_error: 0.5202
    Epoch 697/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.5184 - mean_squared_error: 0.5184
    Epoch 698/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5165 - mean_squared_error: 0.5165
    Epoch 699/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5147 - mean_squared_error: 0.5147
    Epoch 700/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5129 - mean_squared_error: 0.5129
    Epoch 701/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5110 - mean_squared_error: 0.5110
    Epoch 702/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.5092 - mean_squared_error: 0.5092
    Epoch 703/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5074 - mean_squared_error: 0.5074
    Epoch 704/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5056 - mean_squared_error: 0.5056
    Epoch 705/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.5038 - mean_squared_error: 0.5038
    Epoch 706/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.5020 - mean_squared_error: 0.5020
    Epoch 707/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.5002 - mean_squared_error: 0.5002
    Epoch 708/2000
    5/5 [==============================] - 0s 393us/step - loss: 0.4984 - mean_squared_error: 0.4984
    Epoch 709/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4966 - mean_squared_error: 0.4966
    Epoch 710/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.4948 - mean_squared_error: 0.4948
    Epoch 711/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.4931 - mean_squared_error: 0.4931
    Epoch 712/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4913 - mean_squared_error: 0.4913
    Epoch 713/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.4896 - mean_squared_error: 0.4896
    Epoch 714/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4878 - mean_squared_error: 0.4878
    Epoch 715/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4861 - mean_squared_error: 0.4861
    Epoch 716/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4844 - mean_squared_error: 0.4844
    Epoch 717/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4826 - mean_squared_error: 0.4826
    Epoch 718/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.4809 - mean_squared_error: 0.4809
    Epoch 719/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.4792 - mean_squared_error: 0.4792
    Epoch 720/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4775 - mean_squared_error: 0.4775
    Epoch 721/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.4758 - mean_squared_error: 0.4758
    Epoch 722/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4741 - mean_squared_error: 0.4741
    Epoch 723/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4724 - mean_squared_error: 0.4724
    Epoch 724/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4707 - mean_squared_error: 0.4707
    Epoch 725/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4690 - mean_squared_error: 0.4690
    Epoch 726/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.4674 - mean_squared_error: 0.4674
    Epoch 727/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.4657 - mean_squared_error: 0.4657
    Epoch 728/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4640 - mean_squared_error: 0.4640
    Epoch 729/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4624 - mean_squared_error: 0.4624
    Epoch 730/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4607 - mean_squared_error: 0.4607
    Epoch 731/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4591 - mean_squared_error: 0.4591
    Epoch 732/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4575 - mean_squared_error: 0.4575
    Epoch 733/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4558 - mean_squared_error: 0.4558
    Epoch 734/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4542 - mean_squared_error: 0.4542
    Epoch 735/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4526 - mean_squared_error: 0.4526
    Epoch 736/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.4510 - mean_squared_error: 0.4510
    Epoch 737/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4494 - mean_squared_error: 0.4494
    Epoch 738/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4478 - mean_squared_error: 0.4478
    Epoch 739/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4462 - mean_squared_error: 0.4462
    Epoch 740/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.4446 - mean_squared_error: 0.4446
    Epoch 741/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.4430 - mean_squared_error: 0.4430
    Epoch 742/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.4415 - mean_squared_error: 0.4415
    Epoch 743/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4399 - mean_squared_error: 0.4399
    Epoch 744/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4383 - mean_squared_error: 0.4383
    Epoch 745/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4368 - mean_squared_error: 0.4368
    Epoch 746/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.4352 - mean_squared_error: 0.4352
    Epoch 747/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.4337 - mean_squared_error: 0.4337
    Epoch 748/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4321 - mean_squared_error: 0.4321
    Epoch 749/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.4306 - mean_squared_error: 0.4306
    Epoch 750/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4291 - mean_squared_error: 0.4291
    Epoch 751/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4276 - mean_squared_error: 0.4276
    Epoch 752/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4260 - mean_squared_error: 0.4260
    Epoch 753/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4245 - mean_squared_error: 0.4245
    Epoch 754/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4230 - mean_squared_error: 0.4230
    Epoch 755/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4215 - mean_squared_error: 0.4215
    Epoch 756/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4200 - mean_squared_error: 0.4200
    Epoch 757/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.4185 - mean_squared_error: 0.4185
    Epoch 758/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4170 - mean_squared_error: 0.4170
    Epoch 759/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4156 - mean_squared_error: 0.4156
    Epoch 760/2000
    5/5 [==============================] - 0s 602us/step - loss: 0.4141 - mean_squared_error: 0.4141
    Epoch 761/2000
    5/5 [==============================] - 0s 595us/step - loss: 0.4126 - mean_squared_error: 0.4126
    Epoch 762/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.4112 - mean_squared_error: 0.4112
    Epoch 763/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.4097 - mean_squared_error: 0.4097
    Epoch 764/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4082 - mean_squared_error: 0.4082
    Epoch 765/2000
    5/5 [==============================] - 0s 606us/step - loss: 0.4068 - mean_squared_error: 0.4068
    Epoch 766/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4054 - mean_squared_error: 0.4054
    Epoch 767/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.4039 - mean_squared_error: 0.4039
    Epoch 768/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.4025 - mean_squared_error: 0.4025
    Epoch 769/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.4011 - mean_squared_error: 0.4011
    Epoch 770/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3996 - mean_squared_error: 0.3996
    Epoch 771/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3982 - mean_squared_error: 0.3982
    Epoch 772/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3968 - mean_squared_error: 0.3968
    Epoch 773/2000
    5/5 [==============================] - 0s 606us/step - loss: 0.3954 - mean_squared_error: 0.3954
    Epoch 774/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.3940 - mean_squared_error: 0.3940
    Epoch 775/2000
    5/5 [==============================] - 0s 997us/step - loss: 0.3926 - mean_squared_error: 0.3926
    Epoch 776/2000
    5/5 [==============================] - 0s 869us/step - loss: 0.3912 - mean_squared_error: 0.3912
    Epoch 777/2000
    5/5 [==============================] - 0s 996us/step - loss: 0.3898 - mean_squared_error: 0.3898
    Epoch 778/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.3885 - mean_squared_error: 0.3885
    Epoch 779/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3871 - mean_squared_error: 0.3871
    Epoch 780/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.3857 - mean_squared_error: 0.3857
    Epoch 781/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3844 - mean_squared_error: 0.3844
    Epoch 782/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.3830 - mean_squared_error: 0.3830
    Epoch 783/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.3816 - mean_squared_error: 0.3816
    Epoch 784/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3803 - mean_squared_error: 0.3803
    Epoch 785/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.3790 - mean_squared_error: 0.3790
    Epoch 786/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.3776 - mean_squared_error: 0.3776
    Epoch 787/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3763 - mean_squared_error: 0.3763
    Epoch 788/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3749 - mean_squared_error: 0.3749
    Epoch 789/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.3736 - mean_squared_error: 0.3736
    Epoch 790/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.3723 - mean_squared_error: 0.3723
    Epoch 791/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3710 - mean_squared_error: 0.3710
    Epoch 792/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3697 - mean_squared_error: 0.3697
    Epoch 793/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.3684 - mean_squared_error: 0.3684
    Epoch 794/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.3671 - mean_squared_error: 0.3671
    Epoch 795/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.3658 - mean_squared_error: 0.3658
    Epoch 796/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3645 - mean_squared_error: 0.3645
    Epoch 797/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3632 - mean_squared_error: 0.3632
    Epoch 798/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3619 - mean_squared_error: 0.3619
    Epoch 799/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.3606 - mean_squared_error: 0.3606
    Epoch 800/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.3594 - mean_squared_error: 0.3594
    Epoch 801/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3581 - mean_squared_error: 0.3581
    Epoch 802/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3568 - mean_squared_error: 0.3568
    Epoch 803/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.3556 - mean_squared_error: 0.3556
    Epoch 804/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3543 - mean_squared_error: 0.3543
    Epoch 805/2000
    5/5 [==============================] - 0s 398us/step - loss: 0.3531 - mean_squared_error: 0.3531
    Epoch 806/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3518 - mean_squared_error: 0.3518
    Epoch 807/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3506 - mean_squared_error: 0.3506
    Epoch 808/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3494 - mean_squared_error: 0.3494
    Epoch 809/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3481 - mean_squared_error: 0.3481
    Epoch 810/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.3469 - mean_squared_error: 0.3469
    Epoch 811/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3457 - mean_squared_error: 0.3457
    Epoch 812/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3445 - mean_squared_error: 0.3445
    Epoch 813/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3432 - mean_squared_error: 0.3432
    Epoch 814/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3420 - mean_squared_error: 0.3420
    Epoch 815/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.3408 - mean_squared_error: 0.3408
    Epoch 816/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3396 - mean_squared_error: 0.3396
    Epoch 817/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3384 - mean_squared_error: 0.3384
    Epoch 818/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3372 - mean_squared_error: 0.3372
    Epoch 819/2000
    5/5 [==============================] - 0s 595us/step - loss: 0.3361 - mean_squared_error: 0.3361
    Epoch 820/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3349 - mean_squared_error: 0.3349
    Epoch 821/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3337 - mean_squared_error: 0.3337
    Epoch 822/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.3325 - mean_squared_error: 0.3325
    Epoch 823/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.3314 - mean_squared_error: 0.3314
    Epoch 824/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.3302 - mean_squared_error: 0.3302
    Epoch 825/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3290 - mean_squared_error: 0.3290
    Epoch 826/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.3279 - mean_squared_error: 0.3279
    Epoch 827/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3267 - mean_squared_error: 0.3267
    Epoch 828/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3256 - mean_squared_error: 0.3256
    Epoch 829/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3244 - mean_squared_error: 0.3244
    Epoch 830/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.3233 - mean_squared_error: 0.3233
    Epoch 831/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3222 - mean_squared_error: 0.3222
    Epoch 832/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.3210 - mean_squared_error: 0.3210
    Epoch 833/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3199 - mean_squared_error: 0.3199
    Epoch 834/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3188 - mean_squared_error: 0.3188
    Epoch 835/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.3177 - mean_squared_error: 0.3177
    Epoch 836/2000
    5/5 [==============================] - 0s 604us/step - loss: 0.3165 - mean_squared_error: 0.3165
    Epoch 837/2000
    5/5 [==============================] - 0s 592us/step - loss: 0.3154 - mean_squared_error: 0.3154
    Epoch 838/2000
    5/5 [==============================] - 0s 683us/step - loss: 0.3143 - mean_squared_error: 0.3143
    Epoch 839/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.3132 - mean_squared_error: 0.3132
    Epoch 840/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.3121 - mean_squared_error: 0.3121
    Epoch 841/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3110 - mean_squared_error: 0.3110
    Epoch 842/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3099 - mean_squared_error: 0.3099
    Epoch 843/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.3089 - mean_squared_error: 0.3089
    Epoch 844/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3078 - mean_squared_error: 0.3078
    Epoch 845/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.3067 - mean_squared_error: 0.3067
    Epoch 846/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3056 - mean_squared_error: 0.3056
    Epoch 847/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3045 - mean_squared_error: 0.3045
    Epoch 848/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3035 - mean_squared_error: 0.3035
    Epoch 849/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.3024 - mean_squared_error: 0.3024
    Epoch 850/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.3014 - mean_squared_error: 0.3014
    Epoch 851/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.3003 - mean_squared_error: 0.3003
    Epoch 852/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2993 - mean_squared_error: 0.2993
    Epoch 853/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2982 - mean_squared_error: 0.2982
    Epoch 854/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2972 - mean_squared_error: 0.2972
    Epoch 855/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2961 - mean_squared_error: 0.2961
    Epoch 856/2000
    5/5 [==============================] - 0s 398us/step - loss: 0.2951 - mean_squared_error: 0.2951
    Epoch 857/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2941 - mean_squared_error: 0.2941
    Epoch 858/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.2930 - mean_squared_error: 0.2930
    Epoch 859/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2920 - mean_squared_error: 0.2920
    Epoch 860/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2910 - mean_squared_error: 0.2910
    Epoch 861/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.2900 - mean_squared_error: 0.2900
    Epoch 862/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2890 - mean_squared_error: 0.2890
    Epoch 863/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2879 - mean_squared_error: 0.2879
    Epoch 864/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.2869 - mean_squared_error: 0.2869
    Epoch 865/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2859 - mean_squared_error: 0.2859
    Epoch 866/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2849 - mean_squared_error: 0.2849
    Epoch 867/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2839 - mean_squared_error: 0.2839
    Epoch 868/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.2830 - mean_squared_error: 0.2830
    Epoch 869/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2820 - mean_squared_error: 0.2820
    Epoch 870/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2810 - mean_squared_error: 0.2810
    Epoch 871/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.2800 - mean_squared_error: 0.2800
    Epoch 872/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2790 - mean_squared_error: 0.2790
    Epoch 873/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2781 - mean_squared_error: 0.2781
    Epoch 874/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2771 - mean_squared_error: 0.2771
    Epoch 875/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.2761 - mean_squared_error: 0.2761
    Epoch 876/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.2752 - mean_squared_error: 0.2752
    Epoch 877/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.2742 - mean_squared_error: 0.2742
    Epoch 878/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2732 - mean_squared_error: 0.2732
    Epoch 879/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.2723 - mean_squared_error: 0.2723
    Epoch 880/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.2714 - mean_squared_error: 0.2714
    Epoch 881/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2704 - mean_squared_error: 0.2704
    Epoch 882/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2695 - mean_squared_error: 0.2695
    Epoch 883/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2685 - mean_squared_error: 0.2685
    Epoch 884/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.2676 - mean_squared_error: 0.2676
    Epoch 885/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.2667 - mean_squared_error: 0.2667
    Epoch 886/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2657 - mean_squared_error: 0.2657
    Epoch 887/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2648 - mean_squared_error: 0.2648
    Epoch 888/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2639 - mean_squared_error: 0.2639
    Epoch 889/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2630 - mean_squared_error: 0.2630
    Epoch 890/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.2621 - mean_squared_error: 0.2621
    Epoch 891/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.2611 - mean_squared_error: 0.2611
    Epoch 892/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2602 - mean_squared_error: 0.2602
    Epoch 893/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2593 - mean_squared_error: 0.2593
    Epoch 894/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.2584 - mean_squared_error: 0.2584
    Epoch 895/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2575 - mean_squared_error: 0.2575
    Epoch 896/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.2566 - mean_squared_error: 0.2566
    Epoch 897/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.2558 - mean_squared_error: 0.2558
    Epoch 898/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.2549 - mean_squared_error: 0.2549
    Epoch 899/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.2540 - mean_squared_error: 0.2540
    Epoch 900/2000
    5/5 [==============================] - 0s 604us/step - loss: 0.2531 - mean_squared_error: 0.2531
    Epoch 901/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2522 - mean_squared_error: 0.2522
    Epoch 902/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2514 - mean_squared_error: 0.2514
    Epoch 903/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2505 - mean_squared_error: 0.2505
    Epoch 904/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2496 - mean_squared_error: 0.2496
    Epoch 905/2000
    5/5 [==============================] - 0s 606us/step - loss: 0.2487 - mean_squared_error: 0.2487
    Epoch 906/2000
    5/5 [==============================] - 0s 591us/step - loss: 0.2479 - mean_squared_error: 0.2479
    Epoch 907/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.2470 - mean_squared_error: 0.2470
    Epoch 908/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.2462 - mean_squared_error: 0.2462
    Epoch 909/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2453 - mean_squared_error: 0.2453
    Epoch 910/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.2445 - mean_squared_error: 0.2445
    Epoch 911/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2436 - mean_squared_error: 0.2436
    Epoch 912/2000
    5/5 [==============================] - 0s 594us/step - loss: 0.2428 - mean_squared_error: 0.2428
    Epoch 913/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2419 - mean_squared_error: 0.2419
    Epoch 914/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.2411 - mean_squared_error: 0.2411
    Epoch 915/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2403 - mean_squared_error: 0.2403
    Epoch 916/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.2394 - mean_squared_error: 0.2394
    Epoch 917/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.2386 - mean_squared_error: 0.2386
    Epoch 918/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2378 - mean_squared_error: 0.2378
    Epoch 919/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.2370 - mean_squared_error: 0.2370
    Epoch 920/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2362 - mean_squared_error: 0.2362
    Epoch 921/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.2353 - mean_squared_error: 0.2353
    Epoch 922/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2345 - mean_squared_error: 0.2345
    Epoch 923/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2337 - mean_squared_error: 0.2337
    Epoch 924/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2329 - mean_squared_error: 0.2329
    Epoch 925/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.2321 - mean_squared_error: 0.2321
    Epoch 926/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2313 - mean_squared_error: 0.2313
    Epoch 927/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.2305 - mean_squared_error: 0.2305
    Epoch 928/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.2297 - mean_squared_error: 0.2297
    Epoch 929/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2289 - mean_squared_error: 0.2289
    Epoch 930/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.2281 - mean_squared_error: 0.2281
    Epoch 931/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2274 - mean_squared_error: 0.2274
    Epoch 932/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2266 - mean_squared_error: 0.2266
    Epoch 933/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.2258 - mean_squared_error: 0.2258
    Epoch 934/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2250 - mean_squared_error: 0.2250
    Epoch 935/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2242 - mean_squared_error: 0.2242
    Epoch 936/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.2235 - mean_squared_error: 0.2235
    Epoch 937/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2227 - mean_squared_error: 0.2227
    Epoch 938/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2219 - mean_squared_error: 0.2219
    Epoch 939/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2212 - mean_squared_error: 0.2212
    Epoch 940/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2204 - mean_squared_error: 0.2204
    Epoch 941/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2197 - mean_squared_error: 0.2197
    Epoch 942/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.2189 - mean_squared_error: 0.2189
    Epoch 943/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2181 - mean_squared_error: 0.2181
    Epoch 944/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2174 - mean_squared_error: 0.2174
    Epoch 945/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.2166 - mean_squared_error: 0.2166
    Epoch 946/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2159 - mean_squared_error: 0.2159
    Epoch 947/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2152 - mean_squared_error: 0.2152
    Epoch 948/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2144 - mean_squared_error: 0.2144
    Epoch 949/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2137 - mean_squared_error: 0.2137
    Epoch 950/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.2130 - mean_squared_error: 0.2130
    Epoch 951/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.2122 - mean_squared_error: 0.2122
    Epoch 952/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2115 - mean_squared_error: 0.2115
    Epoch 953/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.2108 - mean_squared_error: 0.2108
    Epoch 954/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2101 - mean_squared_error: 0.2101
    Epoch 955/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2093 - mean_squared_error: 0.2093
    Epoch 956/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.2086 - mean_squared_error: 0.2086
    Epoch 957/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2079 - mean_squared_error: 0.2079
    Epoch 958/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.2072 - mean_squared_error: 0.2072
    Epoch 959/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2065 - mean_squared_error: 0.2065
    Epoch 960/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2058 - mean_squared_error: 0.2058
    Epoch 961/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2051 - mean_squared_error: 0.2051
    Epoch 962/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2044 - mean_squared_error: 0.2044
    Epoch 963/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2037 - mean_squared_error: 0.2037
    Epoch 964/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2030 - mean_squared_error: 0.2030
    Epoch 965/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2023 - mean_squared_error: 0.2023
    Epoch 966/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.2016 - mean_squared_error: 0.2016
    Epoch 967/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2009 - mean_squared_error: 0.2009
    Epoch 968/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.2002 - mean_squared_error: 0.2002
    Epoch 969/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.1995 - mean_squared_error: 0.1995
    Epoch 970/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1989 - mean_squared_error: 0.1989
    Epoch 971/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.1982 - mean_squared_error: 0.1982
    Epoch 972/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1975 - mean_squared_error: 0.1975
    Epoch 973/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1968 - mean_squared_error: 0.1968
    Epoch 974/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1962 - mean_squared_error: 0.1962
    Epoch 975/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1955 - mean_squared_error: 0.1955
    Epoch 976/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1948 - mean_squared_error: 0.1948
    Epoch 977/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1942 - mean_squared_error: 0.1942
    Epoch 978/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1935 - mean_squared_error: 0.1935
    Epoch 979/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1928 - mean_squared_error: 0.1928
    Epoch 980/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1922 - mean_squared_error: 0.1922
    Epoch 981/2000
    5/5 [==============================] - 0s 600us/step - loss: 0.1915 - mean_squared_error: 0.1915
    Epoch 982/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1909 - mean_squared_error: 0.1909
    Epoch 983/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1902 - mean_squared_error: 0.1902
    Epoch 984/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1896 - mean_squared_error: 0.1896
    Epoch 985/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1889 - mean_squared_error: 0.1889
    Epoch 986/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1883 - mean_squared_error: 0.1883
    Epoch 987/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1877 - mean_squared_error: 0.1877
    Epoch 988/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1870 - mean_squared_error: 0.1870
    Epoch 989/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1864 - mean_squared_error: 0.1864
    Epoch 990/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1858 - mean_squared_error: 0.1858
    Epoch 991/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1851 - mean_squared_error: 0.1851
    Epoch 992/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1845 - mean_squared_error: 0.1845
    Epoch 993/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1839 - mean_squared_error: 0.1839
    Epoch 994/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.1833 - mean_squared_error: 0.1833
    Epoch 995/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1826 - mean_squared_error: 0.1826
    Epoch 996/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1820 - mean_squared_error: 0.1820
    Epoch 997/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1814 - mean_squared_error: 0.1814
    Epoch 998/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1808 - mean_squared_error: 0.1808
    Epoch 999/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1802 - mean_squared_error: 0.1802
    Epoch 1000/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1796 - mean_squared_error: 0.1796
    Epoch 1001/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1790 - mean_squared_error: 0.1790
    Epoch 1002/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1783 - mean_squared_error: 0.1783
    Epoch 1003/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1777 - mean_squared_error: 0.1777
    Epoch 1004/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1771 - mean_squared_error: 0.1771
    Epoch 1005/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1765 - mean_squared_error: 0.1765
    Epoch 1006/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1760 - mean_squared_error: 0.1760
    Epoch 1007/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1754 - mean_squared_error: 0.1754
    Epoch 1008/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1748 - mean_squared_error: 0.1748
    Epoch 1009/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1742 - mean_squared_error: 0.1742
    Epoch 1010/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1736 - mean_squared_error: 0.1736
    Epoch 1011/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1730 - mean_squared_error: 0.1730
    Epoch 1012/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1724 - mean_squared_error: 0.1724
    Epoch 1013/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1718 - mean_squared_error: 0.1718
    Epoch 1014/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1713 - mean_squared_error: 0.1713
    Epoch 1015/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1707 - mean_squared_error: 0.1707
    Epoch 1016/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1701 - mean_squared_error: 0.1701
    Epoch 1017/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1695 - mean_squared_error: 0.1695
    Epoch 1018/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1690 - mean_squared_error: 0.1690
    Epoch 1019/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.1684 - mean_squared_error: 0.1684
    Epoch 1020/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1678 - mean_squared_error: 0.1678
    Epoch 1021/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1673 - mean_squared_error: 0.1673
    Epoch 1022/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1667 - mean_squared_error: 0.1667
    Epoch 1023/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1661 - mean_squared_error: 0.1661
    Epoch 1024/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1656 - mean_squared_error: 0.1656
    Epoch 1025/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1650 - mean_squared_error: 0.1650
    Epoch 1026/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1645 - mean_squared_error: 0.1645
    Epoch 1027/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1639 - mean_squared_error: 0.1639
    Epoch 1028/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1634 - mean_squared_error: 0.1634
    Epoch 1029/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1628 - mean_squared_error: 0.1628
    Epoch 1030/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1623 - mean_squared_error: 0.1623
    Epoch 1031/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.1617 - mean_squared_error: 0.1617
    Epoch 1032/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.1612 - mean_squared_error: 0.1612
    Epoch 1033/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.1607 - mean_squared_error: 0.1607
    Epoch 1034/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1601 - mean_squared_error: 0.1601
    Epoch 1035/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1596 - mean_squared_error: 0.1596
    Epoch 1036/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.1591 - mean_squared_error: 0.1591
    Epoch 1037/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1585 - mean_squared_error: 0.1585
    Epoch 1038/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1580 - mean_squared_error: 0.1580
    Epoch 1039/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1575 - mean_squared_error: 0.1575
    Epoch 1040/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1569 - mean_squared_error: 0.1569
    Epoch 1041/2000
    5/5 [==============================] - 0s 398us/step - loss: 0.1564 - mean_squared_error: 0.1564
    Epoch 1042/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1559 - mean_squared_error: 0.1559
    Epoch 1043/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1554 - mean_squared_error: 0.1554
    Epoch 1044/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1549 - mean_squared_error: 0.1549
    Epoch 1045/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1543 - mean_squared_error: 0.1543
    Epoch 1046/2000
    5/5 [==============================] - 0s 605us/step - loss: 0.1538 - mean_squared_error: 0.1538
    Epoch 1047/2000
    5/5 [==============================] - 0s 592us/step - loss: 0.1533 - mean_squared_error: 0.1533
    Epoch 1048/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1528 - mean_squared_error: 0.1528
    Epoch 1049/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1523 - mean_squared_error: 0.1523
    Epoch 1050/2000
    5/5 [==============================] - 0s 593us/step - loss: 0.1518 - mean_squared_error: 0.1518
    Epoch 1051/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1513 - mean_squared_error: 0.1513
    Epoch 1052/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1508 - mean_squared_error: 0.1508
    Epoch 1053/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1503 - mean_squared_error: 0.1503
    Epoch 1054/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1498 - mean_squared_error: 0.1498
    Epoch 1055/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1493 - mean_squared_error: 0.1493
    Epoch 1056/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1488 - mean_squared_error: 0.1488
    Epoch 1057/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1483 - mean_squared_error: 0.1483
    Epoch 1058/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1478 - mean_squared_error: 0.1478
    Epoch 1059/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1473 - mean_squared_error: 0.1473
    Epoch 1060/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1468 - mean_squared_error: 0.1468
    Epoch 1061/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1463 - mean_squared_error: 0.1463
    Epoch 1062/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1458 - mean_squared_error: 0.1458
    Epoch 1063/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1454 - mean_squared_error: 0.1454
    Epoch 1064/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1449 - mean_squared_error: 0.1449
    Epoch 1065/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1444 - mean_squared_error: 0.1444
    Epoch 1066/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1439 - mean_squared_error: 0.1439
    Epoch 1067/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1434 - mean_squared_error: 0.1434
    Epoch 1068/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1430 - mean_squared_error: 0.1430
    Epoch 1069/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1425 - mean_squared_error: 0.1425
    Epoch 1070/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1420 - mean_squared_error: 0.1420
    Epoch 1071/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1416 - mean_squared_error: 0.1416
    Epoch 1072/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1411 - mean_squared_error: 0.1411
    Epoch 1073/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.1406 - mean_squared_error: 0.1406
    Epoch 1074/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1402 - mean_squared_error: 0.1402
    Epoch 1075/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1397 - mean_squared_error: 0.1397
    Epoch 1076/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1392 - mean_squared_error: 0.1392
    Epoch 1077/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1388 - mean_squared_error: 0.1388
    Epoch 1078/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1383 - mean_squared_error: 0.1383
    Epoch 1079/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1379 - mean_squared_error: 0.1379
    Epoch 1080/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1374 - mean_squared_error: 0.1374
    Epoch 1081/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1370 - mean_squared_error: 0.1370
    Epoch 1082/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1365 - mean_squared_error: 0.1365
    Epoch 1083/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1361 - mean_squared_error: 0.1361
    Epoch 1084/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1356 - mean_squared_error: 0.1356
    Epoch 1085/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1352 - mean_squared_error: 0.1352
    Epoch 1086/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1347 - mean_squared_error: 0.1347
    Epoch 1087/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1343 - mean_squared_error: 0.1343
    Epoch 1088/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1338 - mean_squared_error: 0.1338
    Epoch 1089/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1334 - mean_squared_error: 0.1334
    Epoch 1090/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1330 - mean_squared_error: 0.1330
    Epoch 1091/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1325 - mean_squared_error: 0.1325
    Epoch 1092/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1321 - mean_squared_error: 0.1321
    Epoch 1093/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1317 - mean_squared_error: 0.1317
    Epoch 1094/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1312 - mean_squared_error: 0.1312
    Epoch 1095/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1308 - mean_squared_error: 0.1308
    Epoch 1096/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1304 - mean_squared_error: 0.1304
    Epoch 1097/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1299 - mean_squared_error: 0.1299
    Epoch 1098/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1295 - mean_squared_error: 0.1295
    Epoch 1099/2000
    5/5 [==============================] - 0s 596us/step - loss: 0.1291 - mean_squared_error: 0.1291
    Epoch 1100/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1287 - mean_squared_error: 0.1287
    Epoch 1101/2000
    5/5 [==============================] - 0s 760us/step - loss: 0.1283 - mean_squared_error: 0.1283
    Epoch 1102/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1278 - mean_squared_error: 0.1278
    Epoch 1103/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1274 - mean_squared_error: 0.1274
    Epoch 1104/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1270 - mean_squared_error: 0.1270
    Epoch 1105/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.1266 - mean_squared_error: 0.1266
    Epoch 1106/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.1262 - mean_squared_error: 0.1262
    Epoch 1107/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1258 - mean_squared_error: 0.1258
    Epoch 1108/2000
    5/5 [==============================] - 0s 604us/step - loss: 0.1254 - mean_squared_error: 0.1254
    Epoch 1109/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1249 - mean_squared_error: 0.1249
    Epoch 1110/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.1245 - mean_squared_error: 0.1245
    Epoch 1111/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1241 - mean_squared_error: 0.1241
    Epoch 1112/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1237 - mean_squared_error: 0.1237
    Epoch 1113/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1233 - mean_squared_error: 0.1233
    Epoch 1114/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1229 - mean_squared_error: 0.1229
    Epoch 1115/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1225 - mean_squared_error: 0.1225
    Epoch 1116/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1221 - mean_squared_error: 0.1221
    Epoch 1117/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1217 - mean_squared_error: 0.1217
    Epoch 1118/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1213 - mean_squared_error: 0.1213
    Epoch 1119/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1209 - mean_squared_error: 0.1209
    Epoch 1120/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1206 - mean_squared_error: 0.1206
    Epoch 1121/2000
    5/5 [==============================] - 0s 394us/step - loss: 0.1202 - mean_squared_error: 0.1202
    Epoch 1122/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1198 - mean_squared_error: 0.1198
    Epoch 1123/2000
    5/5 [==============================] - 0s 404us/step - loss: 0.1194 - mean_squared_error: 0.1194
    Epoch 1124/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1190 - mean_squared_error: 0.1190
    Epoch 1125/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1186 - mean_squared_error: 0.1186
    Epoch 1126/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1182 - mean_squared_error: 0.1182
    Epoch 1127/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1179 - mean_squared_error: 0.1179
    Epoch 1128/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1175 - mean_squared_error: 0.1175
    Epoch 1129/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1171 - mean_squared_error: 0.1171
    Epoch 1130/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1167 - mean_squared_error: 0.1167
    Epoch 1131/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1163 - mean_squared_error: 0.1163
    Epoch 1132/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1160 - mean_squared_error: 0.1160
    Epoch 1133/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1156 - mean_squared_error: 0.1156
    Epoch 1134/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1152 - mean_squared_error: 0.1152
    Epoch 1135/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1149 - mean_squared_error: 0.1149
    Epoch 1136/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1145 - mean_squared_error: 0.1145
    Epoch 1137/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1141 - mean_squared_error: 0.1141
    Epoch 1138/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1137 - mean_squared_error: 0.1137
    Epoch 1139/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1134 - mean_squared_error: 0.1134
    Epoch 1140/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1130 - mean_squared_error: 0.1130
    Epoch 1141/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1127 - mean_squared_error: 0.1127
    Epoch 1142/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1123 - mean_squared_error: 0.1123
    Epoch 1143/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.1119 - mean_squared_error: 0.1119
    Epoch 1144/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1116 - mean_squared_error: 0.1116
    Epoch 1145/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1112 - mean_squared_error: 0.1112
    Epoch 1146/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1109 - mean_squared_error: 0.1109
    Epoch 1147/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1105 - mean_squared_error: 0.1105
    Epoch 1148/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1102 - mean_squared_error: 0.1102
    Epoch 1149/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1098 - mean_squared_error: 0.1098
    Epoch 1150/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1095 - mean_squared_error: 0.1095
    Epoch 1151/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1091 - mean_squared_error: 0.1091
    Epoch 1152/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1088 - mean_squared_error: 0.1088
    Epoch 1153/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1084 - mean_squared_error: 0.1084
    Epoch 1154/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1081 - mean_squared_error: 0.1081
    Epoch 1155/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1077 - mean_squared_error: 0.1077
    Epoch 1156/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1074 - mean_squared_error: 0.1074
    Epoch 1157/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1070 - mean_squared_error: 0.1070
    Epoch 1158/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1067 - mean_squared_error: 0.1067
    Epoch 1159/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1064 - mean_squared_error: 0.1064
    Epoch 1160/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1060 - mean_squared_error: 0.1060
    Epoch 1161/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1057 - mean_squared_error: 0.1057
    Epoch 1162/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1053 - mean_squared_error: 0.1053
    Epoch 1163/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1050 - mean_squared_error: 0.1050
    Epoch 1164/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1047 - mean_squared_error: 0.1047
    Epoch 1165/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1043 - mean_squared_error: 0.1043
    Epoch 1166/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1040 - mean_squared_error: 0.1040
    Epoch 1167/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1037 - mean_squared_error: 0.1037
    Epoch 1168/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1034 - mean_squared_error: 0.1034
    Epoch 1169/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1030 - mean_squared_error: 0.1030
    Epoch 1170/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1027 - mean_squared_error: 0.1027
    Epoch 1171/2000
    5/5 [==============================] - 0s 491us/step - loss: 0.1024 - mean_squared_error: 0.1024
    Epoch 1172/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.1021 - mean_squared_error: 0.1021
    Epoch 1173/2000
    5/5 [==============================] - 0s 501us/step - loss: 0.1017 - mean_squared_error: 0.1017
    Epoch 1174/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1014 - mean_squared_error: 0.1014
    Epoch 1175/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1011 - mean_squared_error: 0.1011
    Epoch 1176/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1008 - mean_squared_error: 0.1008
    Epoch 1177/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.1005 - mean_squared_error: 0.1005
    Epoch 1178/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.1001 - mean_squared_error: 0.1001
    Epoch 1179/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0998 - mean_squared_error: 0.0998
    Epoch 1180/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0995 - mean_squared_error: 0.0995
    Epoch 1181/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0992 - mean_squared_error: 0.0992
    Epoch 1182/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0989 - mean_squared_error: 0.0989
    Epoch 1183/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0986 - mean_squared_error: 0.0986
    Epoch 1184/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0983 - mean_squared_error: 0.0983
    Epoch 1185/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0980 - mean_squared_error: 0.0980
    Epoch 1186/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0977 - mean_squared_error: 0.0977
    Epoch 1187/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0973 - mean_squared_error: 0.0973
    Epoch 1188/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0970 - mean_squared_error: 0.0970
    Epoch 1189/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0967 - mean_squared_error: 0.0967
    Epoch 1190/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0964 - mean_squared_error: 0.0964
    Epoch 1191/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0961 - mean_squared_error: 0.0961
    Epoch 1192/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0958 - mean_squared_error: 0.0958
    Epoch 1193/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0955 - mean_squared_error: 0.0955
    Epoch 1194/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0952 - mean_squared_error: 0.0952
    Epoch 1195/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0949 - mean_squared_error: 0.0949
    Epoch 1196/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0946 - mean_squared_error: 0.0946
    Epoch 1197/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0943 - mean_squared_error: 0.0943
    Epoch 1198/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0941 - mean_squared_error: 0.0941
    Epoch 1199/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0938 - mean_squared_error: 0.0938
    Epoch 1200/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0935 - mean_squared_error: 0.0935
    Epoch 1201/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0932 - mean_squared_error: 0.0932
    Epoch 1202/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0929 - mean_squared_error: 0.0929
    Epoch 1203/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0926 - mean_squared_error: 0.0926
    Epoch 1204/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0923 - mean_squared_error: 0.0923
    Epoch 1205/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0920 - mean_squared_error: 0.0920
    Epoch 1206/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0917 - mean_squared_error: 0.0917
    Epoch 1207/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0915 - mean_squared_error: 0.0915
    Epoch 1208/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0912 - mean_squared_error: 0.0912
    Epoch 1209/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0909 - mean_squared_error: 0.0909
    Epoch 1210/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0906 - mean_squared_error: 0.0906
    Epoch 1211/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0903 - mean_squared_error: 0.0903
    Epoch 1212/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0901 - mean_squared_error: 0.0901
    Epoch 1213/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0898 - mean_squared_error: 0.0898
    Epoch 1214/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0895 - mean_squared_error: 0.0895
    Epoch 1215/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0892 - mean_squared_error: 0.0892
    Epoch 1216/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0889 - mean_squared_error: 0.0889
    Epoch 1217/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0887 - mean_squared_error: 0.0887
    Epoch 1218/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0884 - mean_squared_error: 0.0884
    Epoch 1219/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0881 - mean_squared_error: 0.0881
    Epoch 1220/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0879 - mean_squared_error: 0.0879
    Epoch 1221/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0876 - mean_squared_error: 0.0876
    Epoch 1222/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0873 - mean_squared_error: 0.0873
    Epoch 1223/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0871 - mean_squared_error: 0.0871
    Epoch 1224/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0868 - mean_squared_error: 0.0868
    Epoch 1225/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0865 - mean_squared_error: 0.0865
    Epoch 1226/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0863 - mean_squared_error: 0.0863
    Epoch 1227/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0860 - mean_squared_error: 0.0860
    Epoch 1228/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0857 - mean_squared_error: 0.0857
    Epoch 1229/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0855 - mean_squared_error: 0.0855
    Epoch 1230/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0852 - mean_squared_error: 0.0852
    Epoch 1231/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0849 - mean_squared_error: 0.0849
    Epoch 1232/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0847 - mean_squared_error: 0.0847
    Epoch 1233/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0844 - mean_squared_error: 0.0844
    Epoch 1234/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0842 - mean_squared_error: 0.0842
    Epoch 1235/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0839 - mean_squared_error: 0.0839
    Epoch 1236/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0837 - mean_squared_error: 0.0837
    Epoch 1237/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0834 - mean_squared_error: 0.0834
    Epoch 1238/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0831 - mean_squared_error: 0.0831
    Epoch 1239/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0829 - mean_squared_error: 0.0829
    Epoch 1240/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0826 - mean_squared_error: 0.0826
    Epoch 1241/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0824 - mean_squared_error: 0.0824
    Epoch 1242/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0821 - mean_squared_error: 0.0821
    Epoch 1243/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0819 - mean_squared_error: 0.0819
    Epoch 1244/2000
    5/5 [==============================] - 0s 398us/step - loss: 0.0816 - mean_squared_error: 0.0816
    Epoch 1245/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0814 - mean_squared_error: 0.0814
    Epoch 1246/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0812 - mean_squared_error: 0.0812
    Epoch 1247/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0809 - mean_squared_error: 0.0809
    Epoch 1248/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0807 - mean_squared_error: 0.0807
    Epoch 1249/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0804 - mean_squared_error: 0.0804
    Epoch 1250/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0802 - mean_squared_error: 0.0802
    Epoch 1251/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0799 - mean_squared_error: 0.0799
    Epoch 1252/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0797 - mean_squared_error: 0.0797
    Epoch 1253/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0795 - mean_squared_error: 0.0795
    Epoch 1254/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0792 - mean_squared_error: 0.0792
    Epoch 1255/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0790 - mean_squared_error: 0.0790
    Epoch 1256/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0787 - mean_squared_error: 0.0787
    Epoch 1257/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0785 - mean_squared_error: 0.0785
    Epoch 1258/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0783 - mean_squared_error: 0.0783
    Epoch 1259/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0780 - mean_squared_error: 0.0780
    Epoch 1260/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0778 - mean_squared_error: 0.0778
    Epoch 1261/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0776 - mean_squared_error: 0.0776
    Epoch 1262/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0773 - mean_squared_error: 0.0773
    Epoch 1263/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0771 - mean_squared_error: 0.0771
    Epoch 1264/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0769 - mean_squared_error: 0.0769
    Epoch 1265/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0766 - mean_squared_error: 0.0766
    Epoch 1266/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0764 - mean_squared_error: 0.0764
    Epoch 1267/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0762 - mean_squared_error: 0.0762
    Epoch 1268/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0760 - mean_squared_error: 0.0760
    Epoch 1269/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0757 - mean_squared_error: 0.0757
    Epoch 1270/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0755 - mean_squared_error: 0.0755
    Epoch 1271/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0753 - mean_squared_error: 0.0753
    Epoch 1272/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0751 - mean_squared_error: 0.0751
    Epoch 1273/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0748 - mean_squared_error: 0.0748
    Epoch 1274/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0746 - mean_squared_error: 0.0746
    Epoch 1275/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0744 - mean_squared_error: 0.0744
    Epoch 1276/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0742 - mean_squared_error: 0.0742
    Epoch 1277/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0740 - mean_squared_error: 0.0740
    Epoch 1278/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0737 - mean_squared_error: 0.0737
    Epoch 1279/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0735 - mean_squared_error: 0.0735
    Epoch 1280/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0733 - mean_squared_error: 0.0733
    Epoch 1281/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0731 - mean_squared_error: 0.0731
    Epoch 1282/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0729 - mean_squared_error: 0.0729
    Epoch 1283/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0727 - mean_squared_error: 0.0727
    Epoch 1284/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0724 - mean_squared_error: 0.0724
    Epoch 1285/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0722 - mean_squared_error: 0.0722
    Epoch 1286/2000
    5/5 [==============================] - 0s 609us/step - loss: 0.0720 - mean_squared_error: 0.0720
    Epoch 1287/2000
    5/5 [==============================] - 0s 597us/step - loss: 0.0718 - mean_squared_error: 0.0718
    Epoch 1288/2000
    5/5 [==============================] - 0s 593us/step - loss: 0.0716 - mean_squared_error: 0.0716
    Epoch 1289/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0714 - mean_squared_error: 0.0714
    Epoch 1290/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0712 - mean_squared_error: 0.0712
    Epoch 1291/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0710 - mean_squared_error: 0.0710
    Epoch 1292/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0708 - mean_squared_error: 0.0708
    Epoch 1293/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0706 - mean_squared_error: 0.0706
    Epoch 1294/2000
    5/5 [==============================] - 0s 472us/step - loss: 0.0703 - mean_squared_error: 0.0703
    Epoch 1295/2000
    5/5 [==============================] - 0s 644us/step - loss: 0.0701 - mean_squared_error: 0.0701
    Epoch 1296/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0699 - mean_squared_error: 0.0699
    Epoch 1297/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0697 - mean_squared_error: 0.0697
    Epoch 1298/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0695 - mean_squared_error: 0.0695
    Epoch 1299/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0693 - mean_squared_error: 0.0693
    Epoch 1300/2000
    5/5 [==============================] - 0s 398us/step - loss: 0.0691 - mean_squared_error: 0.0691
    Epoch 1301/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0689 - mean_squared_error: 0.0689
    Epoch 1302/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0687 - mean_squared_error: 0.0687
    Epoch 1303/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0685 - mean_squared_error: 0.0685
    Epoch 1304/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0683 - mean_squared_error: 0.0683
    Epoch 1305/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0681 - mean_squared_error: 0.0681
    Epoch 1306/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0679 - mean_squared_error: 0.0679
    Epoch 1307/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0677 - mean_squared_error: 0.0677
    Epoch 1308/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0675 - mean_squared_error: 0.0675
    Epoch 1309/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0673 - mean_squared_error: 0.0673
    Epoch 1310/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0672 - mean_squared_error: 0.0672
    Epoch 1311/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0670 - mean_squared_error: 0.0670
    Epoch 1312/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0668 - mean_squared_error: 0.0668
    Epoch 1313/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0666 - mean_squared_error: 0.0666
    Epoch 1314/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0664 - mean_squared_error: 0.0664
    Epoch 1315/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0662 - mean_squared_error: 0.0662
    Epoch 1316/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0660 - mean_squared_error: 0.0660
    Epoch 1317/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0658 - mean_squared_error: 0.0658
    Epoch 1318/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0656 - mean_squared_error: 0.0656
    Epoch 1319/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0654 - mean_squared_error: 0.0654
    Epoch 1320/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0652 - mean_squared_error: 0.0652
    Epoch 1321/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0651 - mean_squared_error: 0.0651
    Epoch 1322/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0649 - mean_squared_error: 0.0649
    Epoch 1323/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0647 - mean_squared_error: 0.0647
    Epoch 1324/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0645 - mean_squared_error: 0.0645
    Epoch 1325/2000
    5/5 [==============================] - 0s 791us/step - loss: 0.0643 - mean_squared_error: 0.0643
    Epoch 1326/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0641 - mean_squared_error: 0.0641
    Epoch 1327/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0640 - mean_squared_error: 0.0640
    Epoch 1328/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0638 - mean_squared_error: 0.0638
    Epoch 1329/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0636 - mean_squared_error: 0.0636
    Epoch 1330/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0634 - mean_squared_error: 0.0634
    Epoch 1331/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0632 - mean_squared_error: 0.0632
    Epoch 1332/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0631 - mean_squared_error: 0.0631
    Epoch 1333/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0629 - mean_squared_error: 0.0629
    Epoch 1334/2000
    5/5 [==============================] - 0s 600us/step - loss: 0.0627 - mean_squared_error: 0.0627
    Epoch 1335/2000
    5/5 [==============================] - 0s 605us/step - loss: 0.0625 - mean_squared_error: 0.0625
    Epoch 1336/2000
    5/5 [==============================] - 0s 592us/step - loss: 0.0623 - mean_squared_error: 0.0623
    Epoch 1337/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0622 - mean_squared_error: 0.0622
    Epoch 1338/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0620 - mean_squared_error: 0.0620
    Epoch 1339/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0618 - mean_squared_error: 0.0618
    Epoch 1340/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0616 - mean_squared_error: 0.0616
    Epoch 1341/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0615 - mean_squared_error: 0.0615
    Epoch 1342/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0613 - mean_squared_error: 0.0613
    Epoch 1343/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0611 - mean_squared_error: 0.0611
    Epoch 1344/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0610 - mean_squared_error: 0.0610
    Epoch 1345/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0608 - mean_squared_error: 0.0608
    Epoch 1346/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0606 - mean_squared_error: 0.0606
    Epoch 1347/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0604 - mean_squared_error: 0.0604
    Epoch 1348/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0603 - mean_squared_error: 0.0603
    Epoch 1349/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0601 - mean_squared_error: 0.0601
    Epoch 1350/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0599 - mean_squared_error: 0.0599
    Epoch 1351/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0598 - mean_squared_error: 0.0598
    Epoch 1352/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0596 - mean_squared_error: 0.0596
    Epoch 1353/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0594 - mean_squared_error: 0.0594
    Epoch 1354/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0593 - mean_squared_error: 0.0593
    Epoch 1355/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0591 - mean_squared_error: 0.0591
    Epoch 1356/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0589 - mean_squared_error: 0.0589
    Epoch 1357/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0588 - mean_squared_error: 0.0588
    Epoch 1358/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0586 - mean_squared_error: 0.0586
    Epoch 1359/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0585 - mean_squared_error: 0.0585
    Epoch 1360/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0583 - mean_squared_error: 0.0583
    Epoch 1361/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0581 - mean_squared_error: 0.0581
    Epoch 1362/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0580 - mean_squared_error: 0.0580
    Epoch 1363/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0578 - mean_squared_error: 0.0578
    Epoch 1364/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0577 - mean_squared_error: 0.0577
    Epoch 1365/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0575 - mean_squared_error: 0.0575
    Epoch 1366/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0573 - mean_squared_error: 0.0573
    Epoch 1367/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0572 - mean_squared_error: 0.0572
    Epoch 1368/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0570 - mean_squared_error: 0.0570
    Epoch 1369/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0569 - mean_squared_error: 0.0569
    Epoch 1370/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0567 - mean_squared_error: 0.0567
    Epoch 1371/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0566 - mean_squared_error: 0.0566
    Epoch 1372/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0564 - mean_squared_error: 0.0564
    Epoch 1373/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0562 - mean_squared_error: 0.0562
    Epoch 1374/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0561 - mean_squared_error: 0.0561
    Epoch 1375/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0559 - mean_squared_error: 0.0559
    Epoch 1376/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0558 - mean_squared_error: 0.0558
    Epoch 1377/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0556 - mean_squared_error: 0.0556
    Epoch 1378/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0555 - mean_squared_error: 0.0555
    Epoch 1379/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0553 - mean_squared_error: 0.0553
    Epoch 1380/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0552 - mean_squared_error: 0.0552
    Epoch 1381/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0550 - mean_squared_error: 0.0550
    Epoch 1382/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0549 - mean_squared_error: 0.0549
    Epoch 1383/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0547 - mean_squared_error: 0.0547
    Epoch 1384/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0546 - mean_squared_error: 0.0546
    Epoch 1385/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0544 - mean_squared_error: 0.0544
    Epoch 1386/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0543 - mean_squared_error: 0.0543
    Epoch 1387/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0542 - mean_squared_error: 0.0542
    Epoch 1388/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0540 - mean_squared_error: 0.0540
    Epoch 1389/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0539 - mean_squared_error: 0.0539
    Epoch 1390/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0537 - mean_squared_error: 0.0537
    Epoch 1391/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0536 - mean_squared_error: 0.0536
    Epoch 1392/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0534 - mean_squared_error: 0.0534
    Epoch 1393/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0533 - mean_squared_error: 0.0533
    Epoch 1394/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0531 - mean_squared_error: 0.0531
    Epoch 1395/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0530 - mean_squared_error: 0.0530
    Epoch 1396/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0529 - mean_squared_error: 0.0529
    Epoch 1397/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0527 - mean_squared_error: 0.0527
    Epoch 1398/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0526 - mean_squared_error: 0.0526
    Epoch 1399/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0524 - mean_squared_error: 0.0524
    Epoch 1400/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0523 - mean_squared_error: 0.0523
    Epoch 1401/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0522 - mean_squared_error: 0.0522
    Epoch 1402/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0520 - mean_squared_error: 0.0520
    Epoch 1403/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0519 - mean_squared_error: 0.0519
    Epoch 1404/2000
    5/5 [==============================] - 0s 398us/step - loss: 0.0517 - mean_squared_error: 0.0517
    Epoch 1405/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0516 - mean_squared_error: 0.0516
    Epoch 1406/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0515 - mean_squared_error: 0.0515
    Epoch 1407/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0513 - mean_squared_error: 0.0513
    Epoch 1408/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0512 - mean_squared_error: 0.0512
    Epoch 1409/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0511 - mean_squared_error: 0.0511
    Epoch 1410/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0509 - mean_squared_error: 0.0509
    Epoch 1411/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0508 - mean_squared_error: 0.0508
    Epoch 1412/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0507 - mean_squared_error: 0.0507
    Epoch 1413/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0505 - mean_squared_error: 0.0505
    Epoch 1414/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0504 - mean_squared_error: 0.0504
    Epoch 1415/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0503 - mean_squared_error: 0.0503
    Epoch 1416/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0501 - mean_squared_error: 0.0501
    Epoch 1417/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0500 - mean_squared_error: 0.0500
    Epoch 1418/2000
    5/5 [==============================] - 0s 595us/step - loss: 0.0499 - mean_squared_error: 0.0499
    Epoch 1419/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0497 - mean_squared_error: 0.0497
    Epoch 1420/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0496 - mean_squared_error: 0.0496
    Epoch 1421/2000
    5/5 [==============================] - 0s 597us/step - loss: 0.0495 - mean_squared_error: 0.0495
    Epoch 1422/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0494 - mean_squared_error: 0.0494
    Epoch 1423/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0492 - mean_squared_error: 0.0492
    Epoch 1424/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0491 - mean_squared_error: 0.0491
    Epoch 1425/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0490 - mean_squared_error: 0.0490
    Epoch 1426/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0489 - mean_squared_error: 0.0489
    Epoch 1427/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0487 - mean_squared_error: 0.0487
    Epoch 1428/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0486 - mean_squared_error: 0.0486
    Epoch 1429/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0485 - mean_squared_error: 0.0485
    Epoch 1430/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0484 - mean_squared_error: 0.0484
    Epoch 1431/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0482 - mean_squared_error: 0.0482
    Epoch 1432/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0481 - mean_squared_error: 0.0481
    Epoch 1433/2000
    5/5 [==============================] - ETA: 0s - loss: 0.0396 - mean_squared_error: 0.03 - 0s 598us/step - loss: 0.0480 - mean_squared_error: 0.0480
    Epoch 1434/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0479 - mean_squared_error: 0.0479
    Epoch 1435/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0477 - mean_squared_error: 0.0477
    Epoch 1436/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0476 - mean_squared_error: 0.0476
    Epoch 1437/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0475 - mean_squared_error: 0.0475
    Epoch 1438/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0474 - mean_squared_error: 0.0474
    Epoch 1439/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0472 - mean_squared_error: 0.0472
    Epoch 1440/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0471 - mean_squared_error: 0.0471
    Epoch 1441/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0470 - mean_squared_error: 0.0470
    Epoch 1442/2000
    5/5 [==============================] - 0s 643us/step - loss: 0.0469 - mean_squared_error: 0.0469
    Epoch 1443/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0468 - mean_squared_error: 0.0468
    Epoch 1444/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0467 - mean_squared_error: 0.0467
    Epoch 1445/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0465 - mean_squared_error: 0.0465
    Epoch 1446/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0464 - mean_squared_error: 0.0464
    Epoch 1447/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0463 - mean_squared_error: 0.0463
    Epoch 1448/2000
    5/5 [==============================] - 0s 398us/step - loss: 0.0462 - mean_squared_error: 0.0462
    Epoch 1449/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0461 - mean_squared_error: 0.0461
    Epoch 1450/2000
    5/5 [==============================] - 0s 591us/step - loss: 0.0460 - mean_squared_error: 0.0460
    Epoch 1451/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0458 - mean_squared_error: 0.0458
    Epoch 1452/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0457 - mean_squared_error: 0.0457
    Epoch 1453/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0456 - mean_squared_error: 0.0456
    Epoch 1454/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0455 - mean_squared_error: 0.0455
    Epoch 1455/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0454 - mean_squared_error: 0.0454
    Epoch 1456/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0453 - mean_squared_error: 0.0453
    Epoch 1457/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0452 - mean_squared_error: 0.0452
    Epoch 1458/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0450 - mean_squared_error: 0.0450
    Epoch 1459/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0449 - mean_squared_error: 0.0449
    Epoch 1460/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0448 - mean_squared_error: 0.0448
    Epoch 1461/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0447 - mean_squared_error: 0.0447
    Epoch 1462/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0446 - mean_squared_error: 0.0446
    Epoch 1463/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0445 - mean_squared_error: 0.0445
    Epoch 1464/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0444 - mean_squared_error: 0.0444
    Epoch 1465/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0443 - mean_squared_error: 0.0443
    Epoch 1466/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0442 - mean_squared_error: 0.0442
    Epoch 1467/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0440 - mean_squared_error: 0.0440
    Epoch 1468/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0439 - mean_squared_error: 0.0439
    Epoch 1469/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0438 - mean_squared_error: 0.0438
    Epoch 1470/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0437 - mean_squared_error: 0.0437
    Epoch 1471/2000
    5/5 [==============================] - 0s 766us/step - loss: 0.0436 - mean_squared_error: 0.0436
    Epoch 1472/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0435 - mean_squared_error: 0.0435
    Epoch 1473/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0434 - mean_squared_error: 0.0434
    Epoch 1474/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0433 - mean_squared_error: 0.0433
    Epoch 1475/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0432 - mean_squared_error: 0.0432
    Epoch 1476/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0431 - mean_squared_error: 0.0431
    Epoch 1477/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0430 - mean_squared_error: 0.0430
    Epoch 1478/2000
    5/5 [==============================] - 0s 597us/step - loss: 0.0429 - mean_squared_error: 0.0429
    Epoch 1479/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0428 - mean_squared_error: 0.0428
    Epoch 1480/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0427 - mean_squared_error: 0.0427
    Epoch 1481/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0426 - mean_squared_error: 0.0426
    Epoch 1482/2000
    5/5 [==============================] - 0s 604us/step - loss: 0.0425 - mean_squared_error: 0.0425
    Epoch 1483/2000
    5/5 [==============================] - 0s 593us/step - loss: 0.0424 - mean_squared_error: 0.0424
    Epoch 1484/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0423 - mean_squared_error: 0.0423
    Epoch 1485/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0422 - mean_squared_error: 0.0422
    Epoch 1486/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0421 - mean_squared_error: 0.0421
    Epoch 1487/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0420 - mean_squared_error: 0.0420
    Epoch 1488/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0419 - mean_squared_error: 0.0419
    Epoch 1489/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0418 - mean_squared_error: 0.0418
    Epoch 1490/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0417 - mean_squared_error: 0.0417
    Epoch 1491/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0416 - mean_squared_error: 0.0416
    Epoch 1492/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0415 - mean_squared_error: 0.0415
    Epoch 1493/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0414 - mean_squared_error: 0.0414
    Epoch 1494/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0413 - mean_squared_error: 0.0413
    Epoch 1495/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0412 - mean_squared_error: 0.0412
    Epoch 1496/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0411 - mean_squared_error: 0.0411
    Epoch 1497/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0410 - mean_squared_error: 0.0410
    Epoch 1498/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0409 - mean_squared_error: 0.0409
    Epoch 1499/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0408 - mean_squared_error: 0.0408
    Epoch 1500/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0407 - mean_squared_error: 0.0407
    Epoch 1501/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0406 - mean_squared_error: 0.0406
    Epoch 1502/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0405 - mean_squared_error: 0.0405
    Epoch 1503/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0404 - mean_squared_error: 0.0404
    Epoch 1504/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0403 - mean_squared_error: 0.0403
    Epoch 1505/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0402 - mean_squared_error: 0.0402
    Epoch 1506/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0401 - mean_squared_error: 0.0401
    Epoch 1507/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0400 - mean_squared_error: 0.0400
    Epoch 1508/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0399 - mean_squared_error: 0.0399
    Epoch 1509/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0398 - mean_squared_error: 0.0398
    Epoch 1510/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0397 - mean_squared_error: 0.0397
    Epoch 1511/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0396 - mean_squared_error: 0.0396
    Epoch 1512/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0395 - mean_squared_error: 0.0395
    Epoch 1513/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0395 - mean_squared_error: 0.0395
    Epoch 1514/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0394 - mean_squared_error: 0.0394
    Epoch 1515/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0393 - mean_squared_error: 0.0393
    Epoch 1516/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0392 - mean_squared_error: 0.0392
    Epoch 1517/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0391 - mean_squared_error: 0.0391
    Epoch 1518/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0390 - mean_squared_error: 0.0390
    Epoch 1519/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0389 - mean_squared_error: 0.0389
    Epoch 1520/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0388 - mean_squared_error: 0.0388
    Epoch 1521/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0387 - mean_squared_error: 0.0387
    Epoch 1522/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0386 - mean_squared_error: 0.0386
    Epoch 1523/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0386 - mean_squared_error: 0.0386
    Epoch 1524/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0385 - mean_squared_error: 0.0385
    Epoch 1525/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0384 - mean_squared_error: 0.0384
    Epoch 1526/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0383 - mean_squared_error: 0.0383
    Epoch 1527/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0382 - mean_squared_error: 0.0382
    Epoch 1528/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0381 - mean_squared_error: 0.0381
    Epoch 1529/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0380 - mean_squared_error: 0.0380
    Epoch 1530/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0379 - mean_squared_error: 0.0379
    Epoch 1531/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0379 - mean_squared_error: 0.0379
    Epoch 1532/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0378 - mean_squared_error: 0.0378
    Epoch 1533/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0377 - mean_squared_error: 0.0377
    Epoch 1534/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0376 - mean_squared_error: 0.0376
    Epoch 1535/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0375 - mean_squared_error: 0.0375
    Epoch 1536/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0374 - mean_squared_error: 0.0374
    Epoch 1537/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0374 - mean_squared_error: 0.0374
    Epoch 1538/2000
    5/5 [==============================] - 0s 797us/step - loss: 0.0373 - mean_squared_error: 0.0373
    Epoch 1539/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0372 - mean_squared_error: 0.0372
    Epoch 1540/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0371 - mean_squared_error: 0.0371
    Epoch 1541/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0370 - mean_squared_error: 0.0370
    Epoch 1542/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0369 - mean_squared_error: 0.0369
    Epoch 1543/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0369 - mean_squared_error: 0.0369
    Epoch 1544/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0368 - mean_squared_error: 0.0368
    Epoch 1545/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0367 - mean_squared_error: 0.0367
    Epoch 1546/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0366 - mean_squared_error: 0.0366
    Epoch 1547/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0365 - mean_squared_error: 0.0365
    Epoch 1548/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0365 - mean_squared_error: 0.0365
    Epoch 1549/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0364 - mean_squared_error: 0.0364
    Epoch 1550/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0363 - mean_squared_error: 0.0363
    Epoch 1551/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0362 - mean_squared_error: 0.0362
    Epoch 1552/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0361 - mean_squared_error: 0.0361
    Epoch 1553/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0361 - mean_squared_error: 0.0361
    Epoch 1554/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0360 - mean_squared_error: 0.0360
    Epoch 1555/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0359 - mean_squared_error: 0.0359
    Epoch 1556/2000
    5/5 [==============================] - 0s 597us/step - loss: 0.0358 - mean_squared_error: 0.0358
    Epoch 1557/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0357 - mean_squared_error: 0.0357
    Epoch 1558/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0357 - mean_squared_error: 0.0357
    Epoch 1559/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0356 - mean_squared_error: 0.0356
    Epoch 1560/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0355 - mean_squared_error: 0.0355
    Epoch 1561/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0354 - mean_squared_error: 0.0354
    Epoch 1562/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0354 - mean_squared_error: 0.0354
    Epoch 1563/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0353 - mean_squared_error: 0.0353
    Epoch 1564/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0352 - mean_squared_error: 0.0352
    Epoch 1565/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0351 - mean_squared_error: 0.0351
    Epoch 1566/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0351 - mean_squared_error: 0.0351
    Epoch 1567/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0350 - mean_squared_error: 0.0350
    Epoch 1568/2000
    5/5 [==============================] - 0s 606us/step - loss: 0.0349 - mean_squared_error: 0.0349
    Epoch 1569/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0348 - mean_squared_error: 0.0348
    Epoch 1570/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0348 - mean_squared_error: 0.0348
    Epoch 1571/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0347 - mean_squared_error: 0.0347
    Epoch 1572/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0346 - mean_squared_error: 0.0346
    Epoch 1573/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0345 - mean_squared_error: 0.0345
    Epoch 1574/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0345 - mean_squared_error: 0.0345
    Epoch 1575/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0344 - mean_squared_error: 0.0344
    Epoch 1576/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0343 - mean_squared_error: 0.0343
    Epoch 1577/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0342 - mean_squared_error: 0.0342
    Epoch 1578/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0342 - mean_squared_error: 0.0342
    Epoch 1579/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0341 - mean_squared_error: 0.0341
    Epoch 1580/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0340 - mean_squared_error: 0.0340
    Epoch 1581/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0340 - mean_squared_error: 0.0340
    Epoch 1582/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0339 - mean_squared_error: 0.0339
    Epoch 1583/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0338 - mean_squared_error: 0.0338
    Epoch 1584/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0337 - mean_squared_error: 0.0337
    Epoch 1585/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0337 - mean_squared_error: 0.0337
    Epoch 1586/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0336 - mean_squared_error: 0.0336
    Epoch 1587/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0335 - mean_squared_error: 0.0335
    Epoch 1588/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0335 - mean_squared_error: 0.0335
    Epoch 1589/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0334 - mean_squared_error: 0.0334
    Epoch 1590/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0333 - mean_squared_error: 0.0333
    Epoch 1591/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0333 - mean_squared_error: 0.0333
    Epoch 1592/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0332 - mean_squared_error: 0.0332
    Epoch 1593/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0331 - mean_squared_error: 0.0331
    Epoch 1594/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0331 - mean_squared_error: 0.0331
    Epoch 1595/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0330 - mean_squared_error: 0.0330
    Epoch 1596/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0329 - mean_squared_error: 0.0329
    Epoch 1597/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0328 - mean_squared_error: 0.0328
    Epoch 1598/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0328 - mean_squared_error: 0.0328
    Epoch 1599/2000
    5/5 [==============================] - 0s 804us/step - loss: 0.0327 - mean_squared_error: 0.0327
    Epoch 1600/2000
    5/5 [==============================] - 0s 593us/step - loss: 0.0326 - mean_squared_error: 0.0326
    Epoch 1601/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0326 - mean_squared_error: 0.0326
    Epoch 1602/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0325 - mean_squared_error: 0.0325
    Epoch 1603/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0325 - mean_squared_error: 0.0325
    Epoch 1604/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0324 - mean_squared_error: 0.0324
    Epoch 1605/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0323 - mean_squared_error: 0.0323
    Epoch 1606/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0323 - mean_squared_error: 0.0323
    Epoch 1607/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0322 - mean_squared_error: 0.0322
    Epoch 1608/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0321 - mean_squared_error: 0.0321
    Epoch 1609/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0321 - mean_squared_error: 0.0321
    Epoch 1610/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0320 - mean_squared_error: 0.0320
    Epoch 1611/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0319 - mean_squared_error: 0.0319
    Epoch 1612/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0319 - mean_squared_error: 0.0319
    Epoch 1613/2000
    5/5 [==============================] - 0s 596us/step - loss: 0.0318 - mean_squared_error: 0.0318
    Epoch 1614/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0317 - mean_squared_error: 0.0317
    Epoch 1615/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0317 - mean_squared_error: 0.0317
    Epoch 1616/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0316 - mean_squared_error: 0.0316
    Epoch 1617/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0316 - mean_squared_error: 0.0316
    Epoch 1618/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0315 - mean_squared_error: 0.0315
    Epoch 1619/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0314 - mean_squared_error: 0.0314
    Epoch 1620/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0314 - mean_squared_error: 0.0314
    Epoch 1621/2000
    5/5 [==============================] - 0s 604us/step - loss: 0.0313 - mean_squared_error: 0.0313
    Epoch 1622/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0312 - mean_squared_error: 0.0312
    Epoch 1623/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0312 - mean_squared_error: 0.0312
    Epoch 1624/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0311 - mean_squared_error: 0.0311
    Epoch 1625/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0311 - mean_squared_error: 0.0311
    Epoch 1626/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0310 - mean_squared_error: 0.0310
    Epoch 1627/2000
    5/5 [==============================] - 0s 593us/step - loss: 0.0309 - mean_squared_error: 0.0309
    Epoch 1628/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0309 - mean_squared_error: 0.0309
    Epoch 1629/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0308 - mean_squared_error: 0.0308
    Epoch 1630/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0308 - mean_squared_error: 0.0308
    Epoch 1631/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0307 - mean_squared_error: 0.0307
    Epoch 1632/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0306 - mean_squared_error: 0.0306
    Epoch 1633/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0306 - mean_squared_error: 0.0306
    Epoch 1634/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0305 - mean_squared_error: 0.0305
    Epoch 1635/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0305 - mean_squared_error: 0.0305
    Epoch 1636/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0304 - mean_squared_error: 0.0304
    Epoch 1637/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0304 - mean_squared_error: 0.0304
    Epoch 1638/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0303 - mean_squared_error: 0.0303
    Epoch 1639/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0302 - mean_squared_error: 0.0302
    Epoch 1640/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0302 - mean_squared_error: 0.0302
    Epoch 1641/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0301 - mean_squared_error: 0.0301
    Epoch 1642/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0301 - mean_squared_error: 0.0301
    Epoch 1643/2000
    5/5 [==============================] - 0s 591us/step - loss: 0.0300 - mean_squared_error: 0.0300
    Epoch 1644/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0300 - mean_squared_error: 0.0300
    Epoch 1645/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0299 - mean_squared_error: 0.0299
    Epoch 1646/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0298 - mean_squared_error: 0.0298
    Epoch 1647/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0298 - mean_squared_error: 0.0298
    Epoch 1648/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0297 - mean_squared_error: 0.0297
    Epoch 1649/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0297 - mean_squared_error: 0.0297
    Epoch 1650/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0296 - mean_squared_error: 0.0296
    Epoch 1651/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0296 - mean_squared_error: 0.0296
    Epoch 1652/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0295 - mean_squared_error: 0.0295
    Epoch 1653/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0295 - mean_squared_error: 0.0295
    Epoch 1654/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0294 - mean_squared_error: 0.0294
    Epoch 1655/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0293 - mean_squared_error: 0.0293
    Epoch 1656/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0293 - mean_squared_error: 0.0293
    Epoch 1657/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0292 - mean_squared_error: 0.0292
    Epoch 1658/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0292 - mean_squared_error: 0.0292
    Epoch 1659/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0291 - mean_squared_error: 0.0291
    Epoch 1660/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0291 - mean_squared_error: 0.0291
    Epoch 1661/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0290 - mean_squared_error: 0.0290
    Epoch 1662/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0290 - mean_squared_error: 0.0290
    Epoch 1663/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0289 - mean_squared_error: 0.0289
    Epoch 1664/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0289 - mean_squared_error: 0.0289
    Epoch 1665/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0288 - mean_squared_error: 0.0288
    Epoch 1666/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0288 - mean_squared_error: 0.0288
    Epoch 1667/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0287 - mean_squared_error: 0.0287
    Epoch 1668/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0287 - mean_squared_error: 0.0287
    Epoch 1669/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0286 - mean_squared_error: 0.0286
    Epoch 1670/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0286 - mean_squared_error: 0.0286
    Epoch 1671/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0285 - mean_squared_error: 0.0285
    Epoch 1672/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0285 - mean_squared_error: 0.0285
    Epoch 1673/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0284 - mean_squared_error: 0.0284
    Epoch 1674/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0284 - mean_squared_error: 0.0284
    Epoch 1675/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0283 - mean_squared_error: 0.0283
    Epoch 1676/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0283 - mean_squared_error: 0.0283
    Epoch 1677/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0282 - mean_squared_error: 0.0282
    Epoch 1678/2000
    5/5 [==============================] - 0s 592us/step - loss: 0.0282 - mean_squared_error: 0.0282
    Epoch 1679/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0281 - mean_squared_error: 0.0281
    Epoch 1680/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0281 - mean_squared_error: 0.0281
    Epoch 1681/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0280 - mean_squared_error: 0.0280
    Epoch 1682/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0280 - mean_squared_error: 0.0280
    Epoch 1683/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0279 - mean_squared_error: 0.0279
    Epoch 1684/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0279 - mean_squared_error: 0.0279
    Epoch 1685/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0278 - mean_squared_error: 0.0278
    Epoch 1686/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0278 - mean_squared_error: 0.0278
    Epoch 1687/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0277 - mean_squared_error: 0.0277
    Epoch 1688/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0277 - mean_squared_error: 0.0277
    Epoch 1689/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0276 - mean_squared_error: 0.0276
    Epoch 1690/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0276 - mean_squared_error: 0.0276
    Epoch 1691/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0275 - mean_squared_error: 0.0275
    Epoch 1692/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0275 - mean_squared_error: 0.0275
    Epoch 1693/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0274 - mean_squared_error: 0.0274
    Epoch 1694/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0274 - mean_squared_error: 0.0274
    Epoch 1695/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0273 - mean_squared_error: 0.0273
    Epoch 1696/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0273 - mean_squared_error: 0.0273
    Epoch 1697/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0272 - mean_squared_error: 0.0272
    Epoch 1698/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0272 - mean_squared_error: 0.0272
    Epoch 1699/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0272 - mean_squared_error: 0.0272
    Epoch 1700/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0271 - mean_squared_error: 0.0271
    Epoch 1701/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0271 - mean_squared_error: 0.0271
    Epoch 1702/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0270 - mean_squared_error: 0.0270
    Epoch 1703/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0270 - mean_squared_error: 0.0270
    Epoch 1704/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0269 - mean_squared_error: 0.0269
    Epoch 1705/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0269 - mean_squared_error: 0.0269
    Epoch 1706/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0268 - mean_squared_error: 0.0268
    Epoch 1707/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0268 - mean_squared_error: 0.0268
    Epoch 1708/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0267 - mean_squared_error: 0.0267
    Epoch 1709/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0267 - mean_squared_error: 0.0267
    Epoch 1710/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0267 - mean_squared_error: 0.0267
    Epoch 1711/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0266 - mean_squared_error: 0.0266
    Epoch 1712/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0266 - mean_squared_error: 0.0266
    Epoch 1713/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0265 - mean_squared_error: 0.0265
    Epoch 1714/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0265 - mean_squared_error: 0.0265
    Epoch 1715/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0264 - mean_squared_error: 0.0264
    Epoch 1716/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0264 - mean_squared_error: 0.0264
    Epoch 1717/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0264 - mean_squared_error: 0.0264
    Epoch 1718/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0263 - mean_squared_error: 0.0263
    Epoch 1719/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0263 - mean_squared_error: 0.0263
    Epoch 1720/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0262 - mean_squared_error: 0.0262
    Epoch 1721/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0262 - mean_squared_error: 0.0262
    Epoch 1722/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0261 - mean_squared_error: 0.0261
    Epoch 1723/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0261 - mean_squared_error: 0.0261
    Epoch 1724/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0261 - mean_squared_error: 0.0261
    Epoch 1725/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0260 - mean_squared_error: 0.0260
    Epoch 1726/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0260 - mean_squared_error: 0.0260
    Epoch 1727/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0259 - mean_squared_error: 0.0259
    Epoch 1728/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0259 - mean_squared_error: 0.0259
    Epoch 1729/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0259 - mean_squared_error: 0.0259
    Epoch 1730/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0258 - mean_squared_error: 0.0258
    Epoch 1731/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0258 - mean_squared_error: 0.0258
    Epoch 1732/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0257 - mean_squared_error: 0.0257
    Epoch 1733/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0257 - mean_squared_error: 0.0257
    Epoch 1734/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0257 - mean_squared_error: 0.0257
    Epoch 1735/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0256 - mean_squared_error: 0.0256
    Epoch 1736/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0256 - mean_squared_error: 0.0256
    Epoch 1737/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0255 - mean_squared_error: 0.0255
    Epoch 1738/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0255 - mean_squared_error: 0.0255
    Epoch 1739/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0254 - mean_squared_error: 0.0254
    Epoch 1740/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0254 - mean_squared_error: 0.0254
    Epoch 1741/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0254 - mean_squared_error: 0.0254
    Epoch 1742/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0253 - mean_squared_error: 0.0253
    Epoch 1743/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0253 - mean_squared_error: 0.0253
    Epoch 1744/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0253 - mean_squared_error: 0.0253
    Epoch 1745/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0252 - mean_squared_error: 0.0252
    Epoch 1746/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0252 - mean_squared_error: 0.0252
    Epoch 1747/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0251 - mean_squared_error: 0.0251
    Epoch 1748/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0251 - mean_squared_error: 0.0251
    Epoch 1749/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0251 - mean_squared_error: 0.0251
    Epoch 1750/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0250 - mean_squared_error: 0.0250
    Epoch 1751/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0250 - mean_squared_error: 0.0250
    Epoch 1752/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0249 - mean_squared_error: 0.0249
    Epoch 1753/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0249 - mean_squared_error: 0.0249
    Epoch 1754/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0249 - mean_squared_error: 0.0249
    Epoch 1755/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0248 - mean_squared_error: 0.0248
    Epoch 1756/2000
    5/5 [==============================] - 0s 604us/step - loss: 0.0248 - mean_squared_error: 0.0248
    Epoch 1757/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0248 - mean_squared_error: 0.0248
    Epoch 1758/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0247 - mean_squared_error: 0.0247
    Epoch 1759/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0247 - mean_squared_error: 0.0247
    Epoch 1760/2000
    5/5 [==============================] - ETA: 0s - loss: 0.0051 - mean_squared_error: 0.00 - 0s 604us/step - loss: 0.0247 - mean_squared_error: 0.0247
    Epoch 1761/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0246 - mean_squared_error: 0.0246
    Epoch 1762/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0246 - mean_squared_error: 0.0246
    Epoch 1763/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0245 - mean_squared_error: 0.0245
    Epoch 1764/2000
    5/5 [==============================] - 0s 604us/step - loss: 0.0245 - mean_squared_error: 0.0245
    Epoch 1765/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0245 - mean_squared_error: 0.0245
    Epoch 1766/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0244 - mean_squared_error: 0.0244
    Epoch 1767/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0244 - mean_squared_error: 0.0244
    Epoch 1768/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0244 - mean_squared_error: 0.0244
    Epoch 1769/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0243 - mean_squared_error: 0.0243
    Epoch 1770/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0243 - mean_squared_error: 0.0243
    Epoch 1771/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0243 - mean_squared_error: 0.0243
    Epoch 1772/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0242 - mean_squared_error: 0.0242
    Epoch 1773/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0242 - mean_squared_error: 0.0242
    Epoch 1774/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0242 - mean_squared_error: 0.0242
    Epoch 1775/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0241 - mean_squared_error: 0.0241
    Epoch 1776/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0241 - mean_squared_error: 0.0241
    Epoch 1777/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0240 - mean_squared_error: 0.0240
    Epoch 1778/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0240 - mean_squared_error: 0.0240
    Epoch 1779/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0240 - mean_squared_error: 0.0240
    Epoch 1780/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0239 - mean_squared_error: 0.0239
    Epoch 1781/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0239 - mean_squared_error: 0.0239
    Epoch 1782/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0239 - mean_squared_error: 0.0239
    Epoch 1783/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0238 - mean_squared_error: 0.0238
    Epoch 1784/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0238 - mean_squared_error: 0.0238
    Epoch 1785/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0238 - mean_squared_error: 0.0238
    Epoch 1786/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0237 - mean_squared_error: 0.0237
    Epoch 1787/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0237 - mean_squared_error: 0.0237
    Epoch 1788/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0237 - mean_squared_error: 0.0237
    Epoch 1789/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0236 - mean_squared_error: 0.0236
    Epoch 1790/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0236 - mean_squared_error: 0.0236
    Epoch 1791/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0236 - mean_squared_error: 0.0236
    Epoch 1792/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0235 - mean_squared_error: 0.0235
    Epoch 1793/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0235 - mean_squared_error: 0.0235
    Epoch 1794/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0235 - mean_squared_error: 0.0235
    Epoch 1795/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0235 - mean_squared_error: 0.0235
    Epoch 1796/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0234 - mean_squared_error: 0.0234
    Epoch 1797/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0234 - mean_squared_error: 0.0234
    Epoch 1798/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0234 - mean_squared_error: 0.0234
    Epoch 1799/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0233 - mean_squared_error: 0.0233
    Epoch 1800/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0233 - mean_squared_error: 0.0233
    Epoch 1801/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0233 - mean_squared_error: 0.0233
    Epoch 1802/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0232 - mean_squared_error: 0.0232
    Epoch 1803/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0232 - mean_squared_error: 0.0232
    Epoch 1804/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0232 - mean_squared_error: 0.0232
    Epoch 1805/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0231 - mean_squared_error: 0.0231
    Epoch 1806/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0231 - mean_squared_error: 0.0231
    Epoch 1807/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0231 - mean_squared_error: 0.0231
    Epoch 1808/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0230 - mean_squared_error: 0.0230
    Epoch 1809/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0230 - mean_squared_error: 0.0230
    Epoch 1810/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0230 - mean_squared_error: 0.0230
    Epoch 1811/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0230 - mean_squared_error: 0.0230
    Epoch 1812/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0229 - mean_squared_error: 0.0229
    Epoch 1813/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0229 - mean_squared_error: 0.0229
    Epoch 1814/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0229 - mean_squared_error: 0.0229
    Epoch 1815/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0228 - mean_squared_error: 0.0228
    Epoch 1816/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0228 - mean_squared_error: 0.0228
    Epoch 1817/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0228 - mean_squared_error: 0.0228
    Epoch 1818/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0227 - mean_squared_error: 0.0227
    Epoch 1819/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0227 - mean_squared_error: 0.0227
    Epoch 1820/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0227 - mean_squared_error: 0.0227
    Epoch 1821/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0227 - mean_squared_error: 0.0227
    Epoch 1822/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0226 - mean_squared_error: 0.0226
    Epoch 1823/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0226 - mean_squared_error: 0.0226
    Epoch 1824/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0226 - mean_squared_error: 0.0226
    Epoch 1825/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0225 - mean_squared_error: 0.0225
    Epoch 1826/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0225 - mean_squared_error: 0.0225
    Epoch 1827/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0225 - mean_squared_error: 0.0225
    Epoch 1828/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0225 - mean_squared_error: 0.0225
    Epoch 1829/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0224 - mean_squared_error: 0.0224
    Epoch 1830/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0224 - mean_squared_error: 0.0224
    Epoch 1831/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0224 - mean_squared_error: 0.0224
    Epoch 1832/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0223 - mean_squared_error: 0.0223
    Epoch 1833/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0223 - mean_squared_error: 0.0223
    Epoch 1834/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0223 - mean_squared_error: 0.0223
    Epoch 1835/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0223 - mean_squared_error: 0.0223
    Epoch 1836/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0222 - mean_squared_error: 0.0222
    Epoch 1837/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0222 - mean_squared_error: 0.0222
    Epoch 1838/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0222 - mean_squared_error: 0.0222
    Epoch 1839/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0221 - mean_squared_error: 0.0221
    Epoch 1840/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0221 - mean_squared_error: 0.0221
    Epoch 1841/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0221 - mean_squared_error: 0.0221
    Epoch 1842/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0221 - mean_squared_error: 0.0221
    Epoch 1843/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0220 - mean_squared_error: 0.0220
    Epoch 1844/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0220 - mean_squared_error: 0.0220
    Epoch 1845/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0220 - mean_squared_error: 0.0220
    Epoch 1846/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0220 - mean_squared_error: 0.0220
    Epoch 1847/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0219 - mean_squared_error: 0.0219
    Epoch 1848/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0219 - mean_squared_error: 0.0219
    Epoch 1849/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0219 - mean_squared_error: 0.0219
    Epoch 1850/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0219 - mean_squared_error: 0.0219
    Epoch 1851/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0218 - mean_squared_error: 0.0218
    Epoch 1852/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0218 - mean_squared_error: 0.0218
    Epoch 1853/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0218 - mean_squared_error: 0.0218
    Epoch 1854/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0218 - mean_squared_error: 0.0218
    Epoch 1855/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0217 - mean_squared_error: 0.0217
    Epoch 1856/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0217 - mean_squared_error: 0.0217
    Epoch 1857/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0217 - mean_squared_error: 0.0217
    Epoch 1858/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0216 - mean_squared_error: 0.0216
    Epoch 1859/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0216 - mean_squared_error: 0.0216
    Epoch 1860/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0216 - mean_squared_error: 0.0216
    Epoch 1861/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0216 - mean_squared_error: 0.0216
    Epoch 1862/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0215 - mean_squared_error: 0.0215
    Epoch 1863/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0215 - mean_squared_error: 0.0215
    Epoch 1864/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0215 - mean_squared_error: 0.0215
    Epoch 1865/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0215 - mean_squared_error: 0.0215
    Epoch 1866/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0214 - mean_squared_error: 0.0214
    Epoch 1867/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0214 - mean_squared_error: 0.0214
    Epoch 1868/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0214 - mean_squared_error: 0.0214
    Epoch 1869/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0214 - mean_squared_error: 0.0214
    Epoch 1870/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0213 - mean_squared_error: 0.0213
    Epoch 1871/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0213 - mean_squared_error: 0.0213
    Epoch 1872/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0213 - mean_squared_error: 0.0213
    Epoch 1873/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0213 - mean_squared_error: 0.0213
    Epoch 1874/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0213 - mean_squared_error: 0.0213
    Epoch 1875/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0212 - mean_squared_error: 0.0212
    Epoch 1876/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0212 - mean_squared_error: 0.0212
    Epoch 1877/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0212 - mean_squared_error: 0.0212
    Epoch 1878/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0212 - mean_squared_error: 0.0212
    Epoch 1879/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0211 - mean_squared_error: 0.0211
    Epoch 1880/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0211 - mean_squared_error: 0.0211
    Epoch 1881/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0211 - mean_squared_error: 0.0211
    Epoch 1882/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0211 - mean_squared_error: 0.0211
    Epoch 1883/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0210 - mean_squared_error: 0.0210
    Epoch 1884/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0210 - mean_squared_error: 0.0210
    Epoch 1885/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0210 - mean_squared_error: 0.0210
    Epoch 1886/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0210 - mean_squared_error: 0.0210
    Epoch 1887/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0210 - mean_squared_error: 0.0210
    Epoch 1888/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0209 - mean_squared_error: 0.0209
    Epoch 1889/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0209 - mean_squared_error: 0.0209
    Epoch 1890/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0209 - mean_squared_error: 0.0209
    Epoch 1891/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0209 - mean_squared_error: 0.0209
    Epoch 1892/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0208 - mean_squared_error: 0.0208
    Epoch 1893/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0208 - mean_squared_error: 0.0208
    Epoch 1894/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0208 - mean_squared_error: 0.0208
    Epoch 1895/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0208 - mean_squared_error: 0.0208
    Epoch 1896/2000
    5/5 [==============================] - 0s 600us/step - loss: 0.0207 - mean_squared_error: 0.0207
    Epoch 1897/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0207 - mean_squared_error: 0.0207
    Epoch 1898/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0207 - mean_squared_error: 0.0207
    Epoch 1899/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0207 - mean_squared_error: 0.0207
    Epoch 1900/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0207 - mean_squared_error: 0.0207
    Epoch 1901/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0206 - mean_squared_error: 0.0206
    Epoch 1902/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0206 - mean_squared_error: 0.0206
    Epoch 1903/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0206 - mean_squared_error: 0.0206
    Epoch 1904/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0206 - mean_squared_error: 0.0206
    Epoch 1905/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0206 - mean_squared_error: 0.0206
    Epoch 1906/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0205 - mean_squared_error: 0.0205
    Epoch 1907/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0205 - mean_squared_error: 0.0205
    Epoch 1908/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0205 - mean_squared_error: 0.0205
    Epoch 1909/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0205 - mean_squared_error: 0.0205
    Epoch 1910/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0204 - mean_squared_error: 0.0204
    Epoch 1911/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0204 - mean_squared_error: 0.0204
    Epoch 1912/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0204 - mean_squared_error: 0.0204
    Epoch 1913/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0204 - mean_squared_error: 0.0204
    Epoch 1914/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0204 - mean_squared_error: 0.0204
    Epoch 1915/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0203 - mean_squared_error: 0.0203
    Epoch 1916/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0203 - mean_squared_error: 0.0203
    Epoch 1917/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0203 - mean_squared_error: 0.0203
    Epoch 1918/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0203 - mean_squared_error: 0.0203
    Epoch 1919/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0203 - mean_squared_error: 0.0203
    Epoch 1920/2000
    5/5 [==============================] - 0s 604us/step - loss: 0.0202 - mean_squared_error: 0.0202
    Epoch 1921/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0202 - mean_squared_error: 0.0202
    Epoch 1922/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0202 - mean_squared_error: 0.0202
    Epoch 1923/2000
    5/5 [==============================] - 0s 569us/step - loss: 0.0202 - mean_squared_error: 0.0202  
    Epoch 1924/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0202 - mean_squared_error: 0.0202  
    Epoch 1925/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0201 - mean_squared_error: 0.0201  
    Epoch 1926/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0201 - mean_squared_error: 0.0201  
    Epoch 1927/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0201 - mean_squared_error: 0.0201  
    Epoch 1928/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0201 - mean_squared_error: 0.0201  
    Epoch 1929/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0201 - mean_squared_error: 0.0201  
    Epoch 1930/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0200 - mean_squared_error: 0.0200  
    Epoch 1931/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0200 - mean_squared_error: 0.0200  
    Epoch 1932/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0200 - mean_squared_error: 0.0200  
    Epoch 1933/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0200 - mean_squared_error: 0.0200  
    Epoch 1934/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0200 - mean_squared_error: 0.0200  
    Epoch 1935/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0200 - mean_squared_error: 0.0200  
    Epoch 1936/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0199 - mean_squared_error: 0.0199  
    Epoch 1937/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0199 - mean_squared_error: 0.0199  
    Epoch 1938/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0199 - mean_squared_error: 0.0199  
    Epoch 1939/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0199 - mean_squared_error: 0.0199  
    Epoch 1940/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0199 - mean_squared_error: 0.0199  
    Epoch 1941/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0198 - mean_squared_error: 0.0198  
    Epoch 1942/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0198 - mean_squared_error: 0.0198  
    Epoch 1943/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0198 - mean_squared_error: 0.0198  
    Epoch 1944/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0198 - mean_squared_error: 0.0198  
    Epoch 1945/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0198 - mean_squared_error: 0.0198  
    Epoch 1946/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0197 - mean_squared_error: 0.0197  
    Epoch 1947/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0197 - mean_squared_error: 0.0197  
    Epoch 1948/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0197 - mean_squared_error: 0.0197  
    Epoch 1949/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0197 - mean_squared_error: 0.0197  
    Epoch 1950/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0197 - mean_squared_error: 0.0197  
    Epoch 1951/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0197 - mean_squared_error: 0.0197  
    Epoch 1952/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0196 - mean_squared_error: 0.0196  
    Epoch 1953/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0196 - mean_squared_error: 0.0196  
    Epoch 1954/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0196 - mean_squared_error: 0.0196  
    Epoch 1955/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0196 - mean_squared_error: 0.0196  
    Epoch 1956/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0196 - mean_squared_error: 0.0196  
    Epoch 1957/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0195 - mean_squared_error: 0.0195  
    Epoch 1958/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0195 - mean_squared_error: 0.0195  
    Epoch 1959/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0195 - mean_squared_error: 0.0195  
    Epoch 1960/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0195 - mean_squared_error: 0.0195  
    Epoch 1961/2000
    5/5 [==============================] - 0s 605us/step - loss: 0.0195 - mean_squared_error: 0.0195  
    Epoch 1962/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0195 - mean_squared_error: 0.0195  
    Epoch 1963/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0194 - mean_squared_error: 0.0194  
    Epoch 1964/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0194 - mean_squared_error: 0.0194  
    Epoch 1965/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0194 - mean_squared_error: 0.0194  
    Epoch 1966/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0194 - mean_squared_error: 0.0194  
    Epoch 1967/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0194 - mean_squared_error: 0.0194  
    Epoch 1968/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0194 - mean_squared_error: 0.0194  
    Epoch 1969/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0193 - mean_squared_error: 0.0193  
    Epoch 1970/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0193 - mean_squared_error: 0.0193  
    Epoch 1971/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0193 - mean_squared_error: 0.0193  
    Epoch 1972/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0193 - mean_squared_error: 0.0193  
    Epoch 1973/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0193 - mean_squared_error: 0.0193  
    Epoch 1974/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0193 - mean_squared_error: 0.0193  
    Epoch 1975/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0192 - mean_squared_error: 0.0192  
    Epoch 1976/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0192 - mean_squared_error: 0.0192  
    Epoch 1977/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0192 - mean_squared_error: 0.0192  
    Epoch 1978/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0192 - mean_squared_error: 0.0192  
    Epoch 1979/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0192 - mean_squared_error: 0.0192  
    Epoch 1980/2000
    5/5 [==============================] - 0s 719us/step - loss: 0.0192 - mean_squared_error: 0.0192  
    Epoch 1981/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0191 - mean_squared_error: 0.0191  
    Epoch 1982/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0191 - mean_squared_error: 0.0191  
    Epoch 1983/2000
    5/5 [==============================] - 0s 604us/step - loss: 0.0191 - mean_squared_error: 0.0191  
    Epoch 1984/2000
    5/5 [==============================] - 0s 593us/step - loss: 0.0191 - mean_squared_error: 0.0191  
    Epoch 1985/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0191 - mean_squared_error: 0.0191  
    Epoch 1986/2000
    5/5 [==============================] - 0s 605us/step - loss: 0.0191 - mean_squared_error: 0.0191  
    Epoch 1987/2000
    5/5 [==============================] - 0s 591us/step - loss: 0.0191 - mean_squared_error: 0.0191  
    Epoch 1988/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0190 - mean_squared_error: 0.0190  
    Epoch 1989/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0190 - mean_squared_error: 0.0190  
    Epoch 1990/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0190 - mean_squared_error: 0.0190  
    Epoch 1991/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0190 - mean_squared_error: 0.0190  
    Epoch 1992/2000
    5/5 [==============================] - 0s 798us/step - loss: 0.0190 - mean_squared_error: 0.0190  
    Epoch 1993/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0190 - mean_squared_error: 0.0190  
    Epoch 1994/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0189 - mean_squared_error: 0.0189  
    Epoch 1995/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0189 - mean_squared_error: 0.0189  
    Epoch 1996/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0189 - mean_squared_error: 0.0189  
    Epoch 1997/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0189 - mean_squared_error: 0.0189  
    Epoch 1998/2000
    5/5 [==============================] - 0s 598us/step - loss: 0.0189 - mean_squared_error: 0.0189  
    Epoch 1999/2000
    5/5 [==============================] - 0s 399us/step - loss: 0.0189 - mean_squared_error: 0.0189  
    Epoch 2000/2000
    5/5 [==============================] - 0s 599us/step - loss: 0.0189 - mean_squared_error: 0.0189  
    




    <keras.callbacks.History at 0x215e14c5550>



앞서 실습한 선형 회귀 코드와 거의 동일한데 달라진 점은 입력의 차원이 3으로 바뀌면서, input_dim의 인자값이 3으로 바뀌었다는 점입니다.

아직 오차(loss)가 줄어들 여지가 있지만, 여기서는 이 정도에서 예측 작업을 해보겠습니다.


```python
print(model.predict(X))
```

    [[73.01625 ]
     [81.97073 ]
     [72.01715 ]
     [57.13904 ]
     [33.744396]]
    

이제 훈련할 때 사용하지 않았던 데이터를 가지고 예측 작업을 수행해보겠습니다.


```python
X_test=np.array([[20,99,10],[40,50,20]]) # 각각 58점과 56점을 예측해야 합니다.
print(model.predict(X_test))
```

    [[57.958828]
     [55.91542 ]]
    

### 2) 다중 로지스틱 회귀

y 를 결정하는데 있어 독립 변수 x가 2개인 로지스틱 회귀를 풀어봅시다.

이 경우 가설은 다음과 같습니다.

$H(X) = sigmoid({W_1x_1 + W_2x_2 + b})$

OR 게이트는 0 또는 1의 값을 입력으로 받는데, 두 개의 입력 x1, x2 중 하나라도 1이면 출력값 y가 1이 되고 두 개의 입력이 0인 경우에만 출력값이 0이 되는 게이트입니다. 로지스틱 회귀를 통해 OR 게이트를 구현해봅시다.


```python
import numpy as np
X=np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# 입력 벡터의 차원은 2입니다. 즉, input_dim은 2입니다.
y=np.array([0, 1, 1, 1])
# 출력 벡터의 차원은 1입니다. 즉, output_dim은 1입니다.
```


```python
from keras.models import Sequential # 케라스의 Sequential()을 임포트
from keras.layers import Dense # 케라스의 Dense()를 임포트
from keras import optimizers # 케라스의 옵티마이저를 임포트


model=Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid')) # 이제 입력의 차원은 2입니다.
model.compile(optimizer='sgd' ,loss='binary_crossentropy',metrics=['binary_accuracy'])
# 옵티마이저는 경사하강법의 변형인 확률적 경사 하강법 sgd를 사용합니다.
# 손실 함수(Loss function)는 binary_crossentropy(이진 크로스 엔트로피)를 사용합니다.
model.fit(X,y, batch_size=1, epochs=800, shuffle=False)
# 주어진 X와 y데이터에 대해서 오차를 최소화하는 작업을 800번 시도합니다.
```

    Epoch 1/800
    4/4 [==============================] - 0s 35ms/step - loss: 0.3774 - binary_accuracy: 1.0000
    Epoch 2/800
    4/4 [==============================] - 0s 998us/step - loss: 0.3764 - binary_accuracy: 0.7500
    Epoch 3/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3755 - binary_accuracy: 0.7500
    Epoch 4/800
    4/4 [==============================] - 0s 997us/step - loss: 0.3745 - binary_accuracy: 0.7500
    Epoch 5/800
    4/4 [==============================] - 0s 499us/step - loss: 0.3736 - binary_accuracy: 0.7500
    Epoch 6/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3727 - binary_accuracy: 0.7500
    Epoch 7/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3717 - binary_accuracy: 0.7500
    Epoch 8/800
    4/4 [==============================] - 0s 997us/step - loss: 0.3708 - binary_accuracy: 0.7500
    Epoch 9/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3700 - binary_accuracy: 0.7500
    Epoch 10/800
    4/4 [==============================] - 0s 744us/step - loss: 0.3691 - binary_accuracy: 0.7500
    Epoch 11/800
    4/4 [==============================] - 0s 998us/step - loss: 0.3682 - binary_accuracy: 0.7500
    Epoch 12/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3674 - binary_accuracy: 0.7500
    Epoch 13/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3665 - binary_accuracy: 0.7500
    Epoch 14/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3657 - binary_accuracy: 0.7500
    Epoch 15/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3649 - binary_accuracy: 0.7500
    Epoch 16/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3641 - binary_accuracy: 0.7500
    Epoch 17/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3633 - binary_accuracy: 0.7500
    Epoch 18/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3625 - binary_accuracy: 0.7500
    Epoch 19/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3617 - binary_accuracy: 0.7500
    Epoch 20/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3609 - binary_accuracy: 0.7500
    Epoch 21/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3602 - binary_accuracy: 0.7500
    Epoch 22/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3594 - binary_accuracy: 0.7500
    Epoch 23/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3587 - binary_accuracy: 0.7500
    Epoch 24/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3580 - binary_accuracy: 0.7500
    Epoch 25/800
    4/4 [==============================] - 0s 498us/step - loss: 0.3572 - binary_accuracy: 0.7500
    Epoch 26/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3565 - binary_accuracy: 0.7500
    Epoch 27/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3558 - binary_accuracy: 0.7500
    Epoch 28/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3551 - binary_accuracy: 0.7500
    Epoch 29/800
    4/4 [==============================] - 0s 499us/step - loss: 0.3544 - binary_accuracy: 0.7500
    Epoch 30/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3538 - binary_accuracy: 0.7500
    Epoch 31/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3531 - binary_accuracy: 0.7500
    Epoch 32/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3524 - binary_accuracy: 0.7500
    Epoch 33/800
    4/4 [==============================] - 0s 499us/step - loss: 0.3518 - binary_accuracy: 0.7500
    Epoch 34/800
    4/4 [==============================] - 0s 499us/step - loss: 0.3511 - binary_accuracy: 0.7500
    Epoch 35/800
    4/4 [==============================] - 0s 499us/step - loss: 0.3505 - binary_accuracy: 0.7500
    Epoch 36/800
    4/4 [==============================] - 0s 499us/step - loss: 0.3498 - binary_accuracy: 0.7500
    Epoch 37/800
    4/4 [==============================] - 0s 499us/step - loss: 0.3492 - binary_accuracy: 0.7500
    Epoch 38/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3486 - binary_accuracy: 0.7500
    Epoch 39/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3479 - binary_accuracy: 0.7500
    Epoch 40/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3473 - binary_accuracy: 0.7500
    Epoch 41/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3467 - binary_accuracy: 0.7500
    Epoch 42/800
    4/4 [==============================] - 0s 499us/step - loss: 0.3461 - binary_accuracy: 0.7500
    Epoch 43/800
    4/4 [==============================] - 0s 499us/step - loss: 0.3455 - binary_accuracy: 0.7500
    Epoch 44/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3449 - binary_accuracy: 0.7500
    Epoch 45/800
    4/4 [==============================] - 0s 749us/step - loss: 0.3443 - binary_accuracy: 0.7500
    Epoch 46/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3438 - binary_accuracy: 0.7500
    Epoch 47/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3432 - binary_accuracy: 0.7500
    Epoch 48/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3426 - binary_accuracy: 0.7500
    Epoch 49/800
    4/4 [==============================] - 0s 759us/step - loss: 0.3420 - binary_accuracy: 0.7500
    Epoch 50/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3415 - binary_accuracy: 0.7500
    Epoch 51/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3409 - binary_accuracy: 0.7500
    Epoch 52/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3404 - binary_accuracy: 0.7500
    Epoch 53/800
    4/4 [==============================] - 0s 751us/step - loss: 0.3398 - binary_accuracy: 0.7500
    Epoch 54/800
    4/4 [==============================] - 0s 745us/step - loss: 0.3393 - binary_accuracy: 0.7500
    Epoch 55/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3388 - binary_accuracy: 0.7500
    Epoch 56/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3382 - binary_accuracy: 0.7500
    Epoch 57/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3377 - binary_accuracy: 0.7500
    Epoch 58/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3372 - binary_accuracy: 0.7500
    Epoch 59/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3367 - binary_accuracy: 0.7500
    Epoch 60/800
    4/4 [==============================] - 0s 752us/step - loss: 0.3361 - binary_accuracy: 0.7500
    Epoch 61/800
    4/4 [==============================] - 0s 747us/step - loss: 0.3356 - binary_accuracy: 0.7500
    Epoch 62/800
    4/4 [==============================] - 0s 499us/step - loss: 0.3351 - binary_accuracy: 0.7500
    Epoch 63/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3346 - binary_accuracy: 0.7500
    Epoch 64/800
    4/4 [==============================] - 0s 499us/step - loss: 0.3341 - binary_accuracy: 0.7500
    Epoch 65/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3336 - binary_accuracy: 0.7500
    Epoch 66/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3331 - binary_accuracy: 0.7500
    Epoch 67/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3327 - binary_accuracy: 0.7500
    Epoch 68/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3322 - binary_accuracy: 0.7500
    Epoch 69/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3317 - binary_accuracy: 0.7500
    Epoch 70/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3312 - binary_accuracy: 0.7500
    Epoch 71/800
    4/4 [==============================] - 0s 749us/step - loss: 0.3307 - binary_accuracy: 0.7500
    Epoch 72/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3303 - binary_accuracy: 0.7500
    Epoch 73/800
    4/4 [==============================] - 0s 752us/step - loss: 0.3298 - binary_accuracy: 0.7500
    Epoch 74/800
    4/4 [==============================] - 0s 744us/step - loss: 0.3293 - binary_accuracy: 0.7500
    Epoch 75/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3289 - binary_accuracy: 0.7500
    Epoch 76/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3284 - binary_accuracy: 0.7500
    Epoch 77/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3280 - binary_accuracy: 0.7500
    Epoch 78/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3275 - binary_accuracy: 0.7500
    Epoch 79/800
    4/4 [==============================] - 0s 746us/step - loss: 0.3271 - binary_accuracy: 0.7500
    Epoch 80/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3266 - binary_accuracy: 0.7500
    Epoch 81/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3262 - binary_accuracy: 0.7500
    Epoch 82/800
    4/4 [==============================] - 0s 499us/step - loss: 0.3257 - binary_accuracy: 0.7500
    Epoch 83/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3253 - binary_accuracy: 0.7500
    Epoch 84/800
    4/4 [==============================] - 0s 499us/step - loss: 0.3249 - binary_accuracy: 0.7500
    Epoch 85/800
    4/4 [==============================] - 0s 760us/step - loss: 0.3244 - binary_accuracy: 0.7500
    Epoch 86/800
    4/4 [==============================] - 0s 736us/step - loss: 0.3240 - binary_accuracy: 0.7500
    Epoch 87/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3236 - binary_accuracy: 0.7500
    Epoch 88/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3232 - binary_accuracy: 0.7500
    Epoch 89/800
    4/4 [==============================] - 0s 997us/step - loss: 0.3227 - binary_accuracy: 0.7500
    Epoch 90/800
    4/4 [==============================] - 0s 774us/step - loss: 0.3223 - binary_accuracy: 0.7500
    Epoch 91/800
    4/4 [==============================] - 0s 997us/step - loss: 0.3219 - binary_accuracy: 0.7500
    Epoch 92/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3215 - binary_accuracy: 0.7500
    Epoch 93/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3211 - binary_accuracy: 0.7500
    Epoch 94/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3207 - binary_accuracy: 0.7500
    Epoch 95/800
    4/4 [==============================] - 0s 755us/step - loss: 0.3202 - binary_accuracy: 0.7500
    Epoch 96/800
    4/4 [==============================] - 0s 498us/step - loss: 0.3198 - binary_accuracy: 0.7500
    Epoch 97/800
    4/4 [==============================] - 0s 998us/step - loss: 0.3194 - binary_accuracy: 0.7500
    Epoch 98/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3190 - binary_accuracy: 0.7500
    Epoch 99/800
    4/4 [==============================] - 0s 499us/step - loss: 0.3186 - binary_accuracy: 0.7500
    Epoch 100/800
    4/4 [==============================] - 0s 997us/step - loss: 0.3182 - binary_accuracy: 0.7500
    Epoch 101/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3178 - binary_accuracy: 0.7500
    Epoch 102/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3174 - binary_accuracy: 0.7500
    Epoch 103/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3171 - binary_accuracy: 0.7500
    Epoch 104/800
    4/4 [==============================] - 0s 745us/step - loss: 0.3167 - binary_accuracy: 0.7500
    Epoch 105/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3163 - binary_accuracy: 0.7500
    Epoch 106/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3159 - binary_accuracy: 0.7500
    Epoch 107/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3155 - binary_accuracy: 0.7500
    Epoch 108/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3151 - binary_accuracy: 0.7500
    Epoch 109/800
    4/4 [==============================] - 0s 499us/step - loss: 0.3147 - binary_accuracy: 0.7500
    Epoch 110/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3144 - binary_accuracy: 0.7500
    Epoch 111/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3140 - binary_accuracy: 0.7500
    Epoch 112/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3136 - binary_accuracy: 0.7500
    Epoch 113/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3132 - binary_accuracy: 0.7500
    Epoch 114/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3129 - binary_accuracy: 0.7500
    Epoch 115/800
    4/4 [==============================] - 0s 499us/step - loss: 0.3125 - binary_accuracy: 0.7500
    Epoch 116/800
    4/4 [==============================] - 0s 499us/step - loss: 0.3121 - binary_accuracy: 0.7500
    Epoch 117/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3118 - binary_accuracy: 0.7500
    Epoch 118/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3114 - binary_accuracy: 0.7500
    Epoch 119/800
    4/4 [==============================] - 0s 997us/step - loss: 0.3110 - binary_accuracy: 0.7500
    Epoch 120/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3107 - binary_accuracy: 0.7500
    Epoch 121/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3103 - binary_accuracy: 0.7500
    Epoch 122/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3100 - binary_accuracy: 0.7500
    Epoch 123/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3096 - binary_accuracy: 0.7500
    Epoch 124/800
    4/4 [==============================] - 0s 499us/step - loss: 0.3092 - binary_accuracy: 0.7500
    Epoch 125/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3089 - binary_accuracy: 0.7500
    Epoch 126/800
    4/4 [==============================] - 0s 997us/step - loss: 0.3085 - binary_accuracy: 0.7500
    Epoch 127/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3082 - binary_accuracy: 0.7500
    Epoch 128/800
    4/4 [==============================] - 0s 997us/step - loss: 0.3078 - binary_accuracy: 0.7500
    Epoch 129/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3075 - binary_accuracy: 0.7500
    Epoch 130/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3071 - binary_accuracy: 0.7500
    Epoch 131/800
    4/4 [==============================] - 0s 747us/step - loss: 0.3068 - binary_accuracy: 0.7500
    Epoch 132/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3064 - binary_accuracy: 0.7500
    Epoch 133/800
    4/4 [==============================] - 0s 499us/step - loss: 0.3061 - binary_accuracy: 0.7500
    Epoch 134/800
    4/4 [==============================] - 0s 499us/step - loss: 0.3058 - binary_accuracy: 0.7500
    Epoch 135/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3054 - binary_accuracy: 0.7500
    Epoch 136/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3051 - binary_accuracy: 0.7500
    Epoch 137/800
    4/4 [==============================] - 0s 997us/step - loss: 0.3047 - binary_accuracy: 0.7500
    Epoch 138/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3044 - binary_accuracy: 0.7500
    Epoch 139/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3041 - binary_accuracy: 0.7500
    Epoch 140/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3037 - binary_accuracy: 0.7500
    Epoch 141/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3034 - binary_accuracy: 0.7500
    Epoch 142/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3031 - binary_accuracy: 0.7500
    Epoch 143/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3027 - binary_accuracy: 0.7500
    Epoch 144/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3024 - binary_accuracy: 0.7500
    Epoch 145/800
    4/4 [==============================] - 0s 499us/step - loss: 0.3021 - binary_accuracy: 0.7500
    Epoch 146/800
    4/4 [==============================] - 0s 499us/step - loss: 0.3018 - binary_accuracy: 0.7500
    Epoch 147/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3014 - binary_accuracy: 0.7500
    Epoch 148/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3011 - binary_accuracy: 0.7500
    Epoch 149/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3008 - binary_accuracy: 0.7500
    Epoch 150/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3005 - binary_accuracy: 0.7500
    Epoch 151/800
    4/4 [==============================] - 0s 748us/step - loss: 0.3001 - binary_accuracy: 0.7500
    Epoch 152/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2998 - binary_accuracy: 0.7500
    Epoch 153/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2995 - binary_accuracy: 0.7500
    Epoch 154/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2992 - binary_accuracy: 0.7500
    Epoch 155/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2989 - binary_accuracy: 0.7500
    Epoch 156/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2985 - binary_accuracy: 0.7500
    Epoch 157/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2982 - binary_accuracy: 0.7500
    Epoch 158/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2979 - binary_accuracy: 0.7500
    Epoch 159/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2976 - binary_accuracy: 0.7500
    Epoch 160/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2973 - binary_accuracy: 0.7500
    Epoch 161/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2970 - binary_accuracy: 0.7500
    Epoch 162/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2967 - binary_accuracy: 0.7500
    Epoch 163/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2963 - binary_accuracy: 0.7500
    Epoch 164/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2960 - binary_accuracy: 0.7500
    Epoch 165/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2957 - binary_accuracy: 0.7500
    Epoch 166/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2954 - binary_accuracy: 0.7500
    Epoch 167/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2951 - binary_accuracy: 0.7500
    Epoch 168/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2948 - binary_accuracy: 0.7500
    Epoch 169/800
    4/4 [==============================] - 0s 755us/step - loss: 0.2945 - binary_accuracy: 0.7500
    Epoch 170/800
    4/4 [==============================] - 0s 741us/step - loss: 0.2942 - binary_accuracy: 0.7500
    Epoch 171/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2939 - binary_accuracy: 0.7500
    Epoch 172/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2936 - binary_accuracy: 0.7500
    Epoch 173/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2933 - binary_accuracy: 0.7500
    Epoch 174/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2930 - binary_accuracy: 0.7500
    Epoch 175/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2927 - binary_accuracy: 0.7500
    Epoch 176/800
    4/4 [==============================] - 0s 999us/step - loss: 0.2924 - binary_accuracy: 0.7500
    Epoch 177/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2921 - binary_accuracy: 0.7500
    Epoch 178/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2918 - binary_accuracy: 0.7500
    Epoch 179/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2915 - binary_accuracy: 0.7500
    Epoch 180/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2912 - binary_accuracy: 0.7500
    Epoch 181/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2909 - binary_accuracy: 0.7500
    Epoch 182/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2906 - binary_accuracy: 0.7500
    Epoch 183/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2903 - binary_accuracy: 0.7500
    Epoch 184/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2900 - binary_accuracy: 0.7500
    Epoch 185/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2897 - binary_accuracy: 0.7500
    Epoch 186/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2894 - binary_accuracy: 0.7500
    Epoch 187/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2892 - binary_accuracy: 0.7500
    Epoch 188/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2889 - binary_accuracy: 0.7500
    Epoch 189/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2886 - binary_accuracy: 0.7500
    Epoch 190/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2883 - binary_accuracy: 0.7500
    Epoch 191/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2880 - binary_accuracy: 0.7500
    Epoch 192/800
    4/4 [==============================] - 0s 752us/step - loss: 0.2877 - binary_accuracy: 0.7500
    Epoch 193/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2874 - binary_accuracy: 0.7500
    Epoch 194/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2871 - binary_accuracy: 0.7500
    Epoch 195/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2869 - binary_accuracy: 0.7500
    Epoch 196/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2866 - binary_accuracy: 0.7500
    Epoch 197/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2863 - binary_accuracy: 0.7500
    Epoch 198/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2860 - binary_accuracy: 0.7500
    Epoch 199/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2857 - binary_accuracy: 0.7500
    Epoch 200/800
    4/4 [==============================] - 0s 999us/step - loss: 0.2854 - binary_accuracy: 0.7500
    Epoch 201/800
    4/4 [==============================] - 0s 498us/step - loss: 0.2852 - binary_accuracy: 0.7500
    Epoch 202/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2849 - binary_accuracy: 1.0000
    Epoch 203/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2846 - binary_accuracy: 1.0000
    Epoch 204/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2843 - binary_accuracy: 1.0000
    Epoch 205/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2840 - binary_accuracy: 1.0000
    Epoch 206/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2838 - binary_accuracy: 1.0000
    Epoch 207/800
    4/4 [==============================] - 0s 496us/step - loss: 0.2835 - binary_accuracy: 1.0000
    Epoch 208/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2832 - binary_accuracy: 1.0000
    Epoch 209/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2829 - binary_accuracy: 1.0000
    Epoch 210/800
    4/4 [==============================] - 0s 747us/step - loss: 0.2827 - binary_accuracy: 1.0000
    Epoch 211/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2824 - binary_accuracy: 1.0000
    Epoch 212/800
    4/4 [==============================] - 0s 997us/step - loss: 0.2821 - binary_accuracy: 1.0000
    Epoch 213/800
    4/4 [==============================] - 0s 757us/step - loss: 0.2819 - binary_accuracy: 1.0000
    Epoch 214/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2816 - binary_accuracy: 1.0000
    Epoch 215/800
    4/4 [==============================] - 0s 997us/step - loss: 0.2813 - binary_accuracy: 1.0000
    Epoch 216/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2810 - binary_accuracy: 1.0000
    Epoch 217/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2808 - binary_accuracy: 1.0000
    Epoch 218/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2805 - binary_accuracy: 1.0000
    Epoch 219/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2802 - binary_accuracy: 1.0000
    Epoch 220/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2800 - binary_accuracy: 1.0000
    Epoch 221/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2797 - binary_accuracy: 1.0000
    Epoch 222/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2794 - binary_accuracy: 1.0000
    Epoch 223/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2792 - binary_accuracy: 1.0000
    Epoch 224/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2789 - binary_accuracy: 1.0000
    Epoch 225/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2786 - binary_accuracy: 1.0000
    Epoch 226/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2784 - binary_accuracy: 1.0000
    Epoch 227/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2781 - binary_accuracy: 1.0000
    Epoch 228/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2778 - binary_accuracy: 1.0000
    Epoch 229/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2776 - binary_accuracy: 1.0000
    Epoch 230/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2773 - binary_accuracy: 1.0000
    Epoch 231/800
    4/4 [==============================] - 0s 749us/step - loss: 0.2770 - binary_accuracy: 1.0000
    Epoch 232/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2768 - binary_accuracy: 1.0000
    Epoch 233/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2765 - binary_accuracy: 1.0000
    Epoch 234/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2763 - binary_accuracy: 1.0000
    Epoch 235/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2760 - binary_accuracy: 1.0000
    Epoch 236/800
    4/4 [==============================] - 0s 817us/step - loss: 0.2757 - binary_accuracy: 1.0000
    Epoch 237/800
    4/4 [==============================] - 0s 997us/step - loss: 0.2755 - binary_accuracy: 1.0000
    Epoch 238/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2752 - binary_accuracy: 1.0000
    Epoch 239/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2750 - binary_accuracy: 1.0000
    Epoch 240/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2747 - binary_accuracy: 1.0000
    Epoch 241/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2745 - binary_accuracy: 1.0000
    Epoch 242/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2742 - binary_accuracy: 1.0000
    Epoch 243/800
    4/4 [==============================] - 0s 749us/step - loss: 0.2739 - binary_accuracy: 1.0000
    Epoch 244/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2737 - binary_accuracy: 1.0000
    Epoch 245/800
    4/4 [==============================] - 0s 997us/step - loss: 0.2734 - binary_accuracy: 1.0000
    Epoch 246/800
    4/4 [==============================] - 0s 757us/step - loss: 0.2732 - binary_accuracy: 1.0000
    Epoch 247/800
    4/4 [==============================] - 0s 988us/step - loss: 0.2729 - binary_accuracy: 1.0000
    Epoch 248/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2727 - binary_accuracy: 1.0000
    Epoch 249/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2724 - binary_accuracy: 1.0000
    Epoch 250/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2722 - binary_accuracy: 1.0000
    Epoch 251/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2719 - binary_accuracy: 1.0000
    Epoch 252/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2717 - binary_accuracy: 1.0000
    Epoch 253/800
    4/4 [==============================] - 0s 997us/step - loss: 0.2714 - binary_accuracy: 1.0000
    Epoch 254/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2712 - binary_accuracy: 1.0000
    Epoch 255/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2709 - binary_accuracy: 1.0000
    Epoch 256/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2707 - binary_accuracy: 1.0000
    Epoch 257/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2704 - binary_accuracy: 1.0000
    Epoch 258/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2702 - binary_accuracy: 1.0000
    Epoch 259/800
    4/4 [==============================] - 0s 749us/step - loss: 0.2699 - binary_accuracy: 1.0000
    Epoch 260/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2697 - binary_accuracy: 1.0000
    Epoch 261/800
    4/4 [==============================] - 0s 997us/step - loss: 0.2694 - binary_accuracy: 1.0000
    Epoch 262/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2692 - binary_accuracy: 1.0000
    Epoch 263/800
    4/4 [==============================] - 0s 756us/step - loss: 0.2689 - binary_accuracy: 1.0000
    Epoch 264/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2687 - binary_accuracy: 1.0000
    Epoch 265/800
    4/4 [==============================] - 0s 1ms/step - loss: 0.2684 - binary_accuracy: 1.0000
    Epoch 266/800
    4/4 [==============================] - 0s 741us/step - loss: 0.2682 - binary_accuracy: 1.0000
    Epoch 267/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2680 - binary_accuracy: 1.0000
    Epoch 268/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2677 - binary_accuracy: 1.0000
    Epoch 269/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2675 - binary_accuracy: 1.0000
    Epoch 270/800
    4/4 [==============================] - 0s 998us/step - loss: 0.2672 - binary_accuracy: 1.0000
    Epoch 271/800
    4/4 [==============================] - 0s 498us/step - loss: 0.2670 - binary_accuracy: 1.0000
    Epoch 272/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2667 - binary_accuracy: 1.0000
    Epoch 273/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2665 - binary_accuracy: 1.0000
    Epoch 274/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2663 - binary_accuracy: 1.0000
    Epoch 275/800
    4/4 [==============================] - 0s 997us/step - loss: 0.2660 - binary_accuracy: 1.0000
    Epoch 276/800
    4/4 [==============================] - 0s 749us/step - loss: 0.2658 - binary_accuracy: 1.0000
    Epoch 277/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2655 - binary_accuracy: 1.0000
    Epoch 278/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2653 - binary_accuracy: 1.0000
    Epoch 279/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2651 - binary_accuracy: 1.0000
    Epoch 280/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2648 - binary_accuracy: 1.0000
    Epoch 281/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2646 - binary_accuracy: 1.0000
    Epoch 282/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2643 - binary_accuracy: 1.0000
    Epoch 283/800
    4/4 [==============================] - 0s 494us/step - loss: 0.2641 - binary_accuracy: 1.0000
    Epoch 284/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2639 - binary_accuracy: 1.0000
    Epoch 285/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2636 - binary_accuracy: 1.0000
    Epoch 286/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2634 - binary_accuracy: 1.0000
    Epoch 287/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2632 - binary_accuracy: 1.0000
    Epoch 288/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2629 - binary_accuracy: 1.0000
    Epoch 289/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2627 - binary_accuracy: 1.0000
    Epoch 290/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2625 - binary_accuracy: 1.0000
    Epoch 291/800
    4/4 [==============================] - 0s 997us/step - loss: 0.2622 - binary_accuracy: 1.0000
    Epoch 292/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2620 - binary_accuracy: 1.0000
    Epoch 293/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2618 - binary_accuracy: 1.0000
    Epoch 294/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2615 - binary_accuracy: 1.0000
    Epoch 295/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2613 - binary_accuracy: 1.0000
    Epoch 296/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2611 - binary_accuracy: 1.0000
    Epoch 297/800
    4/4 [==============================] - 0s 849us/step - loss: 0.2608 - binary_accuracy: 1.0000
    Epoch 298/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2606 - binary_accuracy: 1.0000
    Epoch 299/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2604 - binary_accuracy: 1.0000
    Epoch 300/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2601 - binary_accuracy: 1.0000
    Epoch 301/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2599 - binary_accuracy: 1.0000
    Epoch 302/800
    4/4 [==============================] - 0s 739us/step - loss: 0.2597 - binary_accuracy: 1.0000
    Epoch 303/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2595 - binary_accuracy: 1.0000
    Epoch 304/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2592 - binary_accuracy: 1.0000
    Epoch 305/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2590 - binary_accuracy: 1.0000
    Epoch 306/800
    4/4 [==============================] - 0s 747us/step - loss: 0.2588 - binary_accuracy: 1.0000
    Epoch 307/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2585 - binary_accuracy: 1.0000
    Epoch 308/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2583 - binary_accuracy: 1.0000
    Epoch 309/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2581 - binary_accuracy: 1.0000
    Epoch 310/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2579 - binary_accuracy: 1.0000
    Epoch 311/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2576 - binary_accuracy: 1.0000
    Epoch 312/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2574 - binary_accuracy: 1.0000
    Epoch 313/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2572 - binary_accuracy: 1.0000
    Epoch 314/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2570 - binary_accuracy: 1.0000
    Epoch 315/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2567 - binary_accuracy: 1.0000
    Epoch 316/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2565 - binary_accuracy: 1.0000
    Epoch 317/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2563 - binary_accuracy: 1.0000
    Epoch 318/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2561 - binary_accuracy: 1.0000
    Epoch 319/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2559 - binary_accuracy: 1.0000
    Epoch 320/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2556 - binary_accuracy: 1.0000
    Epoch 321/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2554 - binary_accuracy: 1.0000
    Epoch 322/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2552 - binary_accuracy: 1.0000
    Epoch 323/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2550 - binary_accuracy: 1.0000
    Epoch 324/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2547 - binary_accuracy: 1.0000
    Epoch 325/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2545 - binary_accuracy: 1.0000
    Epoch 326/800
    4/4 [==============================] - 0s 497us/step - loss: 0.2543 - binary_accuracy: 1.0000
    Epoch 327/800
    4/4 [==============================] - 0s 749us/step - loss: 0.2541 - binary_accuracy: 1.0000
    Epoch 328/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2539 - binary_accuracy: 1.0000
    Epoch 329/800
    4/4 [==============================] - 0s 749us/step - loss: 0.2536 - binary_accuracy: 1.0000
    Epoch 330/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2534 - binary_accuracy: 1.0000
    Epoch 331/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2532 - binary_accuracy: 1.0000
    Epoch 332/800
    4/4 [==============================] - 0s 997us/step - loss: 0.2530 - binary_accuracy: 1.0000
    Epoch 333/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2528 - binary_accuracy: 1.0000
    Epoch 334/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2526 - binary_accuracy: 1.0000
    Epoch 335/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2523 - binary_accuracy: 1.0000
    Epoch 336/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2521 - binary_accuracy: 1.0000
    Epoch 337/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2519 - binary_accuracy: 1.0000
    Epoch 338/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2517 - binary_accuracy: 1.0000
    Epoch 339/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2515 - binary_accuracy: 1.0000
    Epoch 340/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2513 - binary_accuracy: 1.0000
    Epoch 341/800
    4/4 [==============================] - 0s 998us/step - loss: 0.2510 - binary_accuracy: 1.0000
    Epoch 342/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2508 - binary_accuracy: 1.0000
    Epoch 343/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2506 - binary_accuracy: 1.0000
    Epoch 344/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2504 - binary_accuracy: 1.0000
    Epoch 345/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2502 - binary_accuracy: 1.0000
    Epoch 346/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2500 - binary_accuracy: 1.0000
    Epoch 347/800
    4/4 [==============================] - 0s 747us/step - loss: 0.2498 - binary_accuracy: 1.0000
    Epoch 348/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2496 - binary_accuracy: 1.0000
    Epoch 349/800
    4/4 [==============================] - 0s 755us/step - loss: 0.2493 - binary_accuracy: 1.0000
    Epoch 350/800
    4/4 [==============================] - 0s 741us/step - loss: 0.2491 - binary_accuracy: 1.0000
    Epoch 351/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2489 - binary_accuracy: 1.0000
    Epoch 352/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2487 - binary_accuracy: 1.0000
    Epoch 353/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2485 - binary_accuracy: 1.0000
    Epoch 354/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2483 - binary_accuracy: 1.0000
    Epoch 355/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2481 - binary_accuracy: 1.0000
    Epoch 356/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2479 - binary_accuracy: 1.0000
    Epoch 357/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2477 - binary_accuracy: 1.0000
    Epoch 358/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2475 - binary_accuracy: 1.0000
    Epoch 359/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2472 - binary_accuracy: 1.0000
    Epoch 360/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2470 - binary_accuracy: 1.0000
    Epoch 361/800
    4/4 [==============================] - 0s 755us/step - loss: 0.2468 - binary_accuracy: 1.0000
    Epoch 362/800
    4/4 [==============================] - 0s 990us/step - loss: 0.2466 - binary_accuracy: 1.0000
    Epoch 363/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2464 - binary_accuracy: 1.0000
    Epoch 364/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2462 - binary_accuracy: 1.0000
    Epoch 365/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2460 - binary_accuracy: 1.0000
    Epoch 366/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2458 - binary_accuracy: 1.0000
    Epoch 367/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2456 - binary_accuracy: 1.0000
    Epoch 368/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2454 - binary_accuracy: 1.0000
    Epoch 369/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2452 - binary_accuracy: 1.0000
    Epoch 370/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2450 - binary_accuracy: 1.0000
    Epoch 371/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2448 - binary_accuracy: 1.0000
    Epoch 372/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2446 - binary_accuracy: 1.0000
    Epoch 373/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2444 - binary_accuracy: 1.0000
    Epoch 374/800
    4/4 [==============================] - 0s 997us/step - loss: 0.2442 - binary_accuracy: 1.0000
    Epoch 375/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2440 - binary_accuracy: 1.0000
    Epoch 376/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2438 - binary_accuracy: 1.0000
    Epoch 377/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2436 - binary_accuracy: 1.0000
    Epoch 378/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2434 - binary_accuracy: 1.0000
    Epoch 379/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2431 - binary_accuracy: 1.0000
    Epoch 380/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2429 - binary_accuracy: 1.0000
    Epoch 381/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2427 - binary_accuracy: 1.0000
    Epoch 382/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2425 - binary_accuracy: 1.0000
    Epoch 383/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2423 - binary_accuracy: 1.0000
    Epoch 384/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2421 - binary_accuracy: 1.0000
    Epoch 385/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2419 - binary_accuracy: 1.0000
    Epoch 386/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2417 - binary_accuracy: 1.0000
    Epoch 387/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2415 - binary_accuracy: 1.0000
    Epoch 388/800
    4/4 [==============================] - 0s 498us/step - loss: 0.2413 - binary_accuracy: 1.0000
    Epoch 389/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2411 - binary_accuracy: 1.0000
    Epoch 390/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2409 - binary_accuracy: 1.0000
    Epoch 391/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2407 - binary_accuracy: 1.0000
    Epoch 392/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2406 - binary_accuracy: 1.0000
    Epoch 393/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2404 - binary_accuracy: 1.0000
    Epoch 394/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2402 - binary_accuracy: 1.0000
    Epoch 395/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2400 - binary_accuracy: 1.0000
    Epoch 396/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2398 - binary_accuracy: 1.0000
    Epoch 397/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2396 - binary_accuracy: 1.0000
    Epoch 398/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2394 - binary_accuracy: 1.0000
    Epoch 399/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2392 - binary_accuracy: 1.0000
    Epoch 400/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2390 - binary_accuracy: 1.0000
    Epoch 401/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2388 - binary_accuracy: 1.0000
    Epoch 402/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2386 - binary_accuracy: 1.0000
    Epoch 403/800
    4/4 [==============================] - 0s 743us/step - loss: 0.2384 - binary_accuracy: 1.0000
    Epoch 404/800
    4/4 [==============================] - 0s 997us/step - loss: 0.2382 - binary_accuracy: 1.0000
    Epoch 405/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2380 - binary_accuracy: 1.0000
    Epoch 406/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2378 - binary_accuracy: 1.0000
    Epoch 407/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2376 - binary_accuracy: 1.0000
    Epoch 408/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2374 - binary_accuracy: 1.0000
    Epoch 409/800
    4/4 [==============================] - 0s 749us/step - loss: 0.2372 - binary_accuracy: 1.0000
    Epoch 410/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2370 - binary_accuracy: 1.0000
    Epoch 411/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2368 - binary_accuracy: 1.0000
    Epoch 412/800
    4/4 [==============================] - 0s 1ms/step - loss: 0.2367 - binary_accuracy: 1.0000
    Epoch 413/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2365 - binary_accuracy: 1.0000
    Epoch 414/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2363 - binary_accuracy: 1.0000
    Epoch 415/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2361 - binary_accuracy: 1.0000
    Epoch 416/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2359 - binary_accuracy: 1.0000
    Epoch 417/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2357 - binary_accuracy: 1.0000
    Epoch 418/800
    4/4 [==============================] - 0s 997us/step - loss: 0.2355 - binary_accuracy: 1.0000
    Epoch 419/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2353 - binary_accuracy: 1.0000
    Epoch 420/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2351 - binary_accuracy: 1.0000
    Epoch 421/800
    4/4 [==============================] - 0s 1ms/step - loss: 0.2349 - binary_accuracy: 1.0000
    Epoch 422/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2348 - binary_accuracy: 1.0000
    Epoch 423/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2346 - binary_accuracy: 1.0000
    Epoch 424/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2344 - binary_accuracy: 1.0000
    Epoch 425/800
    4/4 [==============================] - 0s 999us/step - loss: 0.2342 - binary_accuracy: 1.0000
    Epoch 426/800
    4/4 [==============================] - 0s 747us/step - loss: 0.2340 - binary_accuracy: 1.0000
    Epoch 427/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2338 - binary_accuracy: 1.0000
    Epoch 428/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2336 - binary_accuracy: 1.0000
    Epoch 429/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2334 - binary_accuracy: 1.0000
    Epoch 430/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2333 - binary_accuracy: 1.0000
    Epoch 431/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2331 - binary_accuracy: 1.0000
    Epoch 432/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2329 - binary_accuracy: 1.0000
    Epoch 433/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2327 - binary_accuracy: 1.0000
    Epoch 434/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2325 - binary_accuracy: 1.0000
    Epoch 435/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2323 - binary_accuracy: 1.0000
    Epoch 436/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2321 - binary_accuracy: 1.0000
    Epoch 437/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2320 - binary_accuracy: 1.0000
    Epoch 438/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2318 - binary_accuracy: 1.0000
    Epoch 439/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2316 - binary_accuracy: 1.0000
    Epoch 440/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2314 - binary_accuracy: 1.0000
    Epoch 441/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2312 - binary_accuracy: 1.0000
    Epoch 442/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2310 - binary_accuracy: 1.0000
    Epoch 443/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2308 - binary_accuracy: 1.0000
    Epoch 444/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2307 - binary_accuracy: 1.0000
    Epoch 445/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2305 - binary_accuracy: 1.0000
    Epoch 446/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2303 - binary_accuracy: 1.0000
    Epoch 447/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2301 - binary_accuracy: 1.0000
    Epoch 448/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2299 - binary_accuracy: 1.0000
    Epoch 449/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2298 - binary_accuracy: 1.0000
    Epoch 450/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2296 - binary_accuracy: 1.0000
    Epoch 451/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2294 - binary_accuracy: 1.0000
    Epoch 452/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2292 - binary_accuracy: 1.0000
    Epoch 453/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2290 - binary_accuracy: 1.0000
    Epoch 454/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2288 - binary_accuracy: 1.0000
    Epoch 455/800
    4/4 [==============================] - 0s 509us/step - loss: 0.2287 - binary_accuracy: 1.0000
    Epoch 456/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2285 - binary_accuracy: 1.0000
    Epoch 457/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2283 - binary_accuracy: 1.0000
    Epoch 458/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2281 - binary_accuracy: 1.0000
    Epoch 459/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2280 - binary_accuracy: 1.0000
    Epoch 460/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2278 - binary_accuracy: 1.0000
    Epoch 461/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2276 - binary_accuracy: 1.0000
    Epoch 462/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2274 - binary_accuracy: 1.0000
    Epoch 463/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2272 - binary_accuracy: 1.0000
    Epoch 464/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2271 - binary_accuracy: 1.0000
    Epoch 465/800
    4/4 [==============================] - 0s 997us/step - loss: 0.2269 - binary_accuracy: 1.0000
    Epoch 466/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2267 - binary_accuracy: 1.0000
    Epoch 467/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2265 - binary_accuracy: 1.0000
    Epoch 468/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2264 - binary_accuracy: 1.0000
    Epoch 469/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2262 - binary_accuracy: 1.0000
    Epoch 470/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2260 - binary_accuracy: 1.0000
    Epoch 471/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2258 - binary_accuracy: 1.0000
    Epoch 472/800
    4/4 [==============================] - 0s 997us/step - loss: 0.2256 - binary_accuracy: 1.0000
    Epoch 473/800
    4/4 [==============================] - 0s 997us/step - loss: 0.2255 - binary_accuracy: 1.0000
    Epoch 474/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2253 - binary_accuracy: 1.0000
    Epoch 475/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2251 - binary_accuracy: 1.0000
    Epoch 476/800
    4/4 [==============================] - 0s 997us/step - loss: 0.2249 - binary_accuracy: 1.0000
    Epoch 477/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2248 - binary_accuracy: 1.0000
    Epoch 478/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2246 - binary_accuracy: 1.0000
    Epoch 479/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2244 - binary_accuracy: 1.0000
    Epoch 480/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2243 - binary_accuracy: 1.0000
    Epoch 481/800
    4/4 [==============================] - 0s 497us/step - loss: 0.2241 - binary_accuracy: 1.0000
    Epoch 482/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2239 - binary_accuracy: 1.0000
    Epoch 483/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2237 - binary_accuracy: 1.0000
    Epoch 484/800
    4/4 [==============================] - 0s 997us/step - loss: 0.2236 - binary_accuracy: 1.0000
    Epoch 485/800
    4/4 [==============================] - 0s 752us/step - loss: 0.2234 - binary_accuracy: 1.0000
    Epoch 486/800
    4/4 [==============================] - 0s 743us/step - loss: 0.2232 - binary_accuracy: 1.0000
    Epoch 487/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2230 - binary_accuracy: 1.0000
    Epoch 488/800
    4/4 [==============================] - 0s 997us/step - loss: 0.2229 - binary_accuracy: 1.0000
    Epoch 489/800
    4/4 [==============================] - 0s 759us/step - loss: 0.2227 - binary_accuracy: 1.0000
    Epoch 490/800
    4/4 [==============================] - 0s 737us/step - loss: 0.2225 - binary_accuracy: 1.0000
    Epoch 491/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2224 - binary_accuracy: 1.0000
    Epoch 492/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2222 - binary_accuracy: 1.0000
    Epoch 493/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2220 - binary_accuracy: 1.0000
    Epoch 494/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2218 - binary_accuracy: 1.0000
    Epoch 495/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2217 - binary_accuracy: 1.0000
    Epoch 496/800
    4/4 [==============================] - 0s 749us/step - loss: 0.2215 - binary_accuracy: 1.0000
    Epoch 497/800
    4/4 [==============================] - 0s 498us/step - loss: 0.2213 - binary_accuracy: 1.0000
    Epoch 498/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2212 - binary_accuracy: 1.0000
    Epoch 499/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2210 - binary_accuracy: 1.0000
    Epoch 500/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2208 - binary_accuracy: 1.0000
    Epoch 501/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2207 - binary_accuracy: 1.0000
    Epoch 502/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2205 - binary_accuracy: 1.0000
    Epoch 503/800
    4/4 [==============================] - 0s 997us/step - loss: 0.2203 - binary_accuracy: 1.0000
    Epoch 504/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2202 - binary_accuracy: 1.0000
    Epoch 505/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2200 - binary_accuracy: 1.0000
    Epoch 506/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2198 - binary_accuracy: 1.0000
    Epoch 507/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2197 - binary_accuracy: 1.0000
    Epoch 508/800
    4/4 [==============================] - 0s 498us/step - loss: 0.2195 - binary_accuracy: 1.0000
    Epoch 509/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2193 - binary_accuracy: 1.0000
    Epoch 510/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2192 - binary_accuracy: 1.0000
    Epoch 511/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2190 - binary_accuracy: 1.0000
    Epoch 512/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2188 - binary_accuracy: 1.0000
    Epoch 513/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2187 - binary_accuracy: 1.0000
    Epoch 514/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2185 - binary_accuracy: 1.0000
    Epoch 515/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2183 - binary_accuracy: 1.0000
    Epoch 516/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2182 - binary_accuracy: 1.0000
    Epoch 517/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2180 - binary_accuracy: 1.0000
    Epoch 518/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2178 - binary_accuracy: 1.0000
    Epoch 519/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2177 - binary_accuracy: 1.0000
    Epoch 520/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2175 - binary_accuracy: 1.0000
    Epoch 521/800
    4/4 [==============================] - 0s 747us/step - loss: 0.2173 - binary_accuracy: 1.0000
    Epoch 522/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2172 - binary_accuracy: 1.0000
    Epoch 523/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2170 - binary_accuracy: 1.0000
    Epoch 524/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2169 - binary_accuracy: 1.0000
    Epoch 525/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2167 - binary_accuracy: 1.0000
    Epoch 526/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2165 - binary_accuracy: 1.0000
    Epoch 527/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2164 - binary_accuracy: 1.0000
    Epoch 528/800
    4/4 [==============================] - 0s 757us/step - loss: 0.2162 - binary_accuracy: 1.0000
    Epoch 529/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2160 - binary_accuracy: 1.0000
    Epoch 530/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2159 - binary_accuracy: 1.0000
    Epoch 531/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2157 - binary_accuracy: 1.0000
    Epoch 532/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2156 - binary_accuracy: 1.0000
    Epoch 533/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2154 - binary_accuracy: 1.0000
    Epoch 534/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2152 - binary_accuracy: 1.0000
    Epoch 535/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2151 - binary_accuracy: 1.0000
    Epoch 536/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2149 - binary_accuracy: 1.0000
    Epoch 537/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2147 - binary_accuracy: 1.0000
    Epoch 538/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2146 - binary_accuracy: 1.0000
    Epoch 539/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2144 - binary_accuracy: 1.0000
    Epoch 540/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2143 - binary_accuracy: 1.0000
    Epoch 541/800
    4/4 [==============================] - 0s 998us/step - loss: 0.2141 - binary_accuracy: 1.0000
    Epoch 542/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2140 - binary_accuracy: 1.0000
    Epoch 543/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2138 - binary_accuracy: 1.0000
    Epoch 544/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2136 - binary_accuracy: 1.0000
    Epoch 545/800
    4/4 [==============================] - 0s 997us/step - loss: 0.2135 - binary_accuracy: 1.0000
    Epoch 546/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2133 - binary_accuracy: 1.0000
    Epoch 547/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2132 - binary_accuracy: 1.0000
    Epoch 548/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2130 - binary_accuracy: 1.0000
    Epoch 549/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2128 - binary_accuracy: 1.0000
    Epoch 550/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2127 - binary_accuracy: 1.0000
    Epoch 551/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2125 - binary_accuracy: 1.0000
    Epoch 552/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2124 - binary_accuracy: 1.0000
    Epoch 553/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2122 - binary_accuracy: 1.0000
    Epoch 554/800
    4/4 [==============================] - 0s 998us/step - loss: 0.2121 - binary_accuracy: 1.0000
    Epoch 555/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2119 - binary_accuracy: 1.0000
    Epoch 556/800
    4/4 [==============================] - 0s 498us/step - loss: 0.2117 - binary_accuracy: 1.0000
    Epoch 557/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2116 - binary_accuracy: 1.0000
    Epoch 558/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2114 - binary_accuracy: 1.0000
    Epoch 559/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2113 - binary_accuracy: 1.0000
    Epoch 560/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2111 - binary_accuracy: 1.0000
    Epoch 561/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2110 - binary_accuracy: 1.0000
    Epoch 562/800
    4/4 [==============================] - 0s 498us/step - loss: 0.2108 - binary_accuracy: 1.0000
    Epoch 563/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2107 - binary_accuracy: 1.0000
    Epoch 564/800
    4/4 [==============================] - 0s 498us/step - loss: 0.2105 - binary_accuracy: 1.0000
    Epoch 565/800
    4/4 [==============================] - 0s 577us/step - loss: 0.2103 - binary_accuracy: 1.0000
    Epoch 566/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2102 - binary_accuracy: 1.0000
    Epoch 567/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2100 - binary_accuracy: 1.0000
    Epoch 568/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2099 - binary_accuracy: 1.0000
    Epoch 569/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2097 - binary_accuracy: 1.0000
    Epoch 570/800
    4/4 [==============================] - 0s 738us/step - loss: 0.2096 - binary_accuracy: 1.0000
    Epoch 571/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2094 - binary_accuracy: 1.0000
    Epoch 572/800
    4/4 [==============================] - 0s 498us/step - loss: 0.2093 - binary_accuracy: 1.0000
    Epoch 573/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2091 - binary_accuracy: 1.0000
    Epoch 574/800
    4/4 [==============================] - 0s 747us/step - loss: 0.2090 - binary_accuracy: 1.0000
    Epoch 575/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2088 - binary_accuracy: 1.0000
    Epoch 576/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2087 - binary_accuracy: 1.0000
    Epoch 577/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2085 - binary_accuracy: 1.0000
    Epoch 578/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2084 - binary_accuracy: 1.0000
    Epoch 579/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2082 - binary_accuracy: 1.0000
    Epoch 580/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2081 - binary_accuracy: 1.0000
    Epoch 581/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2079 - binary_accuracy: 1.0000
    Epoch 582/800
    4/4 [==============================] - 0s 997us/step - loss: 0.2078 - binary_accuracy: 1.0000
    Epoch 583/800
    4/4 [==============================] - 0s 755us/step - loss: 0.2076 - binary_accuracy: 1.0000
    Epoch 584/800
    4/4 [==============================] - 0s 741us/step - loss: 0.2075 - binary_accuracy: 1.0000
    Epoch 585/800
    4/4 [==============================] - 0s 505us/step - loss: 0.2073 - binary_accuracy: 1.0000
    Epoch 586/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2072 - binary_accuracy: 1.0000
    Epoch 587/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2070 - binary_accuracy: 1.0000
    Epoch 588/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2069 - binary_accuracy: 1.0000
    Epoch 589/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2067 - binary_accuracy: 1.0000
    Epoch 590/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2066 - binary_accuracy: 1.0000
    Epoch 591/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2064 - binary_accuracy: 1.0000
    Epoch 592/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2063 - binary_accuracy: 1.0000
    Epoch 593/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2061 - binary_accuracy: 1.0000
    Epoch 594/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2060 - binary_accuracy: 1.0000
    Epoch 595/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2058 - binary_accuracy: 1.0000
    Epoch 596/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2057 - binary_accuracy: 1.0000
    Epoch 597/800
    4/4 [==============================] - 0s 747us/step - loss: 0.2055 - binary_accuracy: 1.0000
    Epoch 598/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2054 - binary_accuracy: 1.0000
    Epoch 599/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2052 - binary_accuracy: 1.0000
    Epoch 600/800
    4/4 [==============================] - 0s 497us/step - loss: 0.2051 - binary_accuracy: 1.0000
    Epoch 601/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2049 - binary_accuracy: 1.0000
    Epoch 602/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2048 - binary_accuracy: 1.0000
    Epoch 603/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2046 - binary_accuracy: 1.0000
    Epoch 604/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2045 - binary_accuracy: 1.0000
    Epoch 605/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2043 - binary_accuracy: 1.0000
    Epoch 606/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2042 - binary_accuracy: 1.0000
    Epoch 607/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2041 - binary_accuracy: 1.0000
    Epoch 608/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2039 - binary_accuracy: 1.0000
    Epoch 609/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2038 - binary_accuracy: 1.0000
    Epoch 610/800
    4/4 [==============================] - 0s 498us/step - loss: 0.2036 - binary_accuracy: 1.0000
    Epoch 611/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2035 - binary_accuracy: 1.0000
    Epoch 612/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2033 - binary_accuracy: 1.0000
    Epoch 613/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2032 - binary_accuracy: 1.0000
    Epoch 614/800
    4/4 [==============================] - 0s 747us/step - loss: 0.2030 - binary_accuracy: 1.0000
    Epoch 615/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2029 - binary_accuracy: 1.0000
    Epoch 616/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2028 - binary_accuracy: 1.0000
    Epoch 617/800
    4/4 [==============================] - 0s 908us/step - loss: 0.2026 - binary_accuracy: 1.0000
    Epoch 618/800
    4/4 [==============================] - 0s 498us/step - loss: 0.2025 - binary_accuracy: 1.0000
    Epoch 619/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2023 - binary_accuracy: 1.0000
    Epoch 620/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2022 - binary_accuracy: 1.0000
    Epoch 621/800
    4/4 [==============================] - 0s 757us/step - loss: 0.2020 - binary_accuracy: 1.0000
    Epoch 622/800
    4/4 [==============================] - 0s 498us/step - loss: 0.2019 - binary_accuracy: 1.0000
    Epoch 623/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2017 - binary_accuracy: 1.0000
    Epoch 624/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2016 - binary_accuracy: 1.0000
    Epoch 625/800
    4/4 [==============================] - 0s 499us/step - loss: 0.2015 - binary_accuracy: 1.0000
    Epoch 626/800
    4/4 [==============================] - 0s 686us/step - loss: 0.2013 - binary_accuracy: 1.0000
    Epoch 627/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2012 - binary_accuracy: 1.0000
    Epoch 628/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2010 - binary_accuracy: 1.0000
    Epoch 629/800
    4/4 [==============================] - 0s 749us/step - loss: 0.2009 - binary_accuracy: 1.0000
    Epoch 630/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2008 - binary_accuracy: 1.0000
    Epoch 631/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2006 - binary_accuracy: 1.0000
    Epoch 632/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2005 - binary_accuracy: 1.0000
    Epoch 633/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2003 - binary_accuracy: 1.0000
    Epoch 634/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2002 - binary_accuracy: 1.0000
    Epoch 635/800
    4/4 [==============================] - 0s 748us/step - loss: 0.2000 - binary_accuracy: 1.0000
    Epoch 636/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1999 - binary_accuracy: 1.0000
    Epoch 637/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1998 - binary_accuracy: 1.0000
    Epoch 638/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1996 - binary_accuracy: 1.0000
    Epoch 639/800
    4/4 [==============================] - 0s 997us/step - loss: 0.1995 - binary_accuracy: 1.0000
    Epoch 640/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1993 - binary_accuracy: 1.0000
    Epoch 641/800
    4/4 [==============================] - 0s 741us/step - loss: 0.1992 - binary_accuracy: 1.0000
    Epoch 642/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1991 - binary_accuracy: 1.0000
    Epoch 643/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1989 - binary_accuracy: 1.0000
    Epoch 644/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1988 - binary_accuracy: 1.0000
    Epoch 645/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1987 - binary_accuracy: 1.0000
    Epoch 646/800
    4/4 [==============================] - 0s 747us/step - loss: 0.1985 - binary_accuracy: 1.0000
    Epoch 647/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1984 - binary_accuracy: 1.0000
    Epoch 648/800
    4/4 [==============================] - 0s 747us/step - loss: 0.1982 - binary_accuracy: 1.0000
    Epoch 649/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1981 - binary_accuracy: 1.0000
    Epoch 650/800
    4/4 [==============================] - 0s 497us/step - loss: 0.1980 - binary_accuracy: 1.0000
    Epoch 651/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1978 - binary_accuracy: 1.0000
    Epoch 652/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1977 - binary_accuracy: 1.0000
    Epoch 653/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1975 - binary_accuracy: 1.0000
    Epoch 654/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1974 - binary_accuracy: 1.0000
    Epoch 655/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1973 - binary_accuracy: 1.0000
    Epoch 656/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1971 - binary_accuracy: 1.0000
    Epoch 657/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1970 - binary_accuracy: 1.0000
    Epoch 658/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1969 - binary_accuracy: 1.0000
    Epoch 659/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1967 - binary_accuracy: 1.0000
    Epoch 660/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1966 - binary_accuracy: 1.0000
    Epoch 661/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1965 - binary_accuracy: 1.0000
    Epoch 662/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1963 - binary_accuracy: 1.0000
    Epoch 663/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1962 - binary_accuracy: 1.0000
    Epoch 664/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1960 - binary_accuracy: 1.0000
    Epoch 665/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1959 - binary_accuracy: 1.0000
    Epoch 666/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1958 - binary_accuracy: 1.0000
    Epoch 667/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1956 - binary_accuracy: 1.0000
    Epoch 668/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1955 - binary_accuracy: 1.0000
    Epoch 669/800
    4/4 [==============================] - 0s 997us/step - loss: 0.1954 - binary_accuracy: 1.0000
    Epoch 670/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1952 - binary_accuracy: 1.0000
    Epoch 671/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1951 - binary_accuracy: 1.0000
    Epoch 672/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1950 - binary_accuracy: 1.0000
    Epoch 673/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1948 - binary_accuracy: 1.0000
    Epoch 674/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1947 - binary_accuracy: 1.0000
    Epoch 675/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1946 - binary_accuracy: 1.0000
    Epoch 676/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1944 - binary_accuracy: 1.0000
    Epoch 677/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1943 - binary_accuracy: 1.0000
    Epoch 678/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1942 - binary_accuracy: 1.0000
    Epoch 679/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1940 - binary_accuracy: 1.0000
    Epoch 680/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1939 - binary_accuracy: 1.0000
    Epoch 681/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1938 - binary_accuracy: 1.0000
    Epoch 682/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1936 - binary_accuracy: 1.0000
    Epoch 683/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1935 - binary_accuracy: 1.0000
    Epoch 684/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1934 - binary_accuracy: 1.0000
    Epoch 685/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1932 - binary_accuracy: 1.0000
    Epoch 686/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1931 - binary_accuracy: 1.0000
    Epoch 687/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1930 - binary_accuracy: 1.0000
    Epoch 688/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1929 - binary_accuracy: 1.0000
    Epoch 689/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1927 - binary_accuracy: 1.0000
    Epoch 690/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1926 - binary_accuracy: 1.0000
    Epoch 691/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1925 - binary_accuracy: 1.0000
    Epoch 692/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1923 - binary_accuracy: 1.0000
    Epoch 693/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1922 - binary_accuracy: 1.0000
    Epoch 694/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1921 - binary_accuracy: 1.0000
    Epoch 695/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1919 - binary_accuracy: 1.0000
    Epoch 696/800
    4/4 [==============================] - 0s 749us/step - loss: 0.1918 - binary_accuracy: 1.0000
    Epoch 697/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1917 - binary_accuracy: 1.0000
    Epoch 698/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1915 - binary_accuracy: 1.0000
    Epoch 699/800
    4/4 [==============================] - 0s 495us/step - loss: 0.1914 - binary_accuracy: 1.0000
    Epoch 700/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1913 - binary_accuracy: 1.0000
    Epoch 701/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1912 - binary_accuracy: 1.0000
    Epoch 702/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1910 - binary_accuracy: 1.0000
    Epoch 703/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1909 - binary_accuracy: 1.0000
    Epoch 704/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1908 - binary_accuracy: 1.0000
    Epoch 705/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1906 - binary_accuracy: 1.0000
    Epoch 706/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1905 - binary_accuracy: 1.0000
    Epoch 707/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1904 - binary_accuracy: 1.0000
    Epoch 708/800
    4/4 [==============================] - 0s 646us/step - loss: 0.1903 - binary_accuracy: 1.0000
    Epoch 709/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1901 - binary_accuracy: 1.0000
    Epoch 710/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1900 - binary_accuracy: 1.0000
    Epoch 711/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1899 - binary_accuracy: 1.0000
    Epoch 712/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1897 - binary_accuracy: 1.0000
    Epoch 713/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1896 - binary_accuracy: 1.0000
    Epoch 714/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1895 - binary_accuracy: 1.0000
    Epoch 715/800
    4/4 [==============================] - 0s 997us/step - loss: 0.1894 - binary_accuracy: 1.0000
    Epoch 716/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1892 - binary_accuracy: 1.0000
    Epoch 717/800
    4/4 [==============================] - 0s 998us/step - loss: 0.1891 - binary_accuracy: 1.0000
    Epoch 718/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1890 - binary_accuracy: 1.0000
    Epoch 719/800
    4/4 [==============================] - 0s 756us/step - loss: 0.1889 - binary_accuracy: 1.0000
    Epoch 720/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1887 - binary_accuracy: 1.0000
    Epoch 721/800
    4/4 [==============================] - 0s 997us/step - loss: 0.1886 - binary_accuracy: 1.0000
    Epoch 722/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1885 - binary_accuracy: 1.0000
    Epoch 723/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1884 - binary_accuracy: 1.0000
    Epoch 724/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1882 - binary_accuracy: 1.0000
    Epoch 725/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1881 - binary_accuracy: 1.0000
    Epoch 726/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1880 - binary_accuracy: 1.0000
    Epoch 727/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1879 - binary_accuracy: 1.0000
    Epoch 728/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1877 - binary_accuracy: 1.0000
    Epoch 729/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1876 - binary_accuracy: 1.0000
    Epoch 730/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1875 - binary_accuracy: 1.0000
    Epoch 731/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1874 - binary_accuracy: 1.0000
    Epoch 732/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1872 - binary_accuracy: 1.0000
    Epoch 733/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1871 - binary_accuracy: 1.0000
    Epoch 734/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1870 - binary_accuracy: 1.0000
    Epoch 735/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1869 - binary_accuracy: 1.0000
    Epoch 736/800
    4/4 [==============================] - 0s 755us/step - loss: 0.1867 - binary_accuracy: 1.0000
    Epoch 737/800
    4/4 [==============================] - 0s 688us/step - loss: 0.1866 - binary_accuracy: 1.0000
    Epoch 738/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1865 - binary_accuracy: 1.0000
    Epoch 739/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1864 - binary_accuracy: 1.0000
    Epoch 740/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1862 - binary_accuracy: 1.0000
    Epoch 741/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1861 - binary_accuracy: 1.0000
    Epoch 742/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1860 - binary_accuracy: 1.0000
    Epoch 743/800
    4/4 [==============================] - 0s 997us/step - loss: 0.1859 - binary_accuracy: 1.0000
    Epoch 744/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1857 - binary_accuracy: 1.0000
    Epoch 745/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1856 - binary_accuracy: 1.0000
    Epoch 746/800
    4/4 [==============================] - 0s 498us/step - loss: 0.1855 - binary_accuracy: 1.0000
    Epoch 747/800
    4/4 [==============================] - 0s 749us/step - loss: 0.1854 - binary_accuracy: 1.0000
    Epoch 748/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1853 - binary_accuracy: 1.0000
    Epoch 749/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1851 - binary_accuracy: 1.0000
    Epoch 750/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1850 - binary_accuracy: 1.0000
    Epoch 751/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1849 - binary_accuracy: 1.0000
    Epoch 752/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1848 - binary_accuracy: 1.0000
    Epoch 753/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1847 - binary_accuracy: 1.0000
    Epoch 754/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1845 - binary_accuracy: 1.0000
    Epoch 755/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1844 - binary_accuracy: 1.0000
    Epoch 756/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1843 - binary_accuracy: 1.0000
    Epoch 757/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1842 - binary_accuracy: 1.0000
    Epoch 758/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1840 - binary_accuracy: 1.0000
    Epoch 759/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1839 - binary_accuracy: 1.0000
    Epoch 760/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1838 - binary_accuracy: 1.0000
    Epoch 761/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1837 - binary_accuracy: 1.0000
    Epoch 762/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1836 - binary_accuracy: 1.0000
    Epoch 763/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1834 - binary_accuracy: 1.0000
    Epoch 764/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1833 - binary_accuracy: 1.0000
    Epoch 765/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1832 - binary_accuracy: 1.0000
    Epoch 766/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1831 - binary_accuracy: 1.0000
    Epoch 767/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1830 - binary_accuracy: 1.0000
    Epoch 768/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1829 - binary_accuracy: 1.0000
    Epoch 769/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1827 - binary_accuracy: 1.0000
    Epoch 770/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1826 - binary_accuracy: 1.0000
    Epoch 771/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1825 - binary_accuracy: 1.0000
    Epoch 772/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1824 - binary_accuracy: 1.0000
    Epoch 773/800
    4/4 [==============================] - 0s 739us/step - loss: 0.1823 - binary_accuracy: 1.0000
    Epoch 774/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1821 - binary_accuracy: 1.0000
    Epoch 775/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1820 - binary_accuracy: 1.0000
    Epoch 776/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1819 - binary_accuracy: 1.0000
    Epoch 777/800
    4/4 [==============================] - 0s 997us/step - loss: 0.1818 - binary_accuracy: 1.0000
    Epoch 778/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1817 - binary_accuracy: 1.0000
    Epoch 779/800
    4/4 [==============================] - 0s 749us/step - loss: 0.1816 - binary_accuracy: 1.0000
    Epoch 780/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1814 - binary_accuracy: 1.0000
    Epoch 781/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1813 - binary_accuracy: 1.0000
    Epoch 782/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1812 - binary_accuracy: 1.0000
    Epoch 783/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1811 - binary_accuracy: 1.0000
    Epoch 784/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1810 - binary_accuracy: 1.0000
    Epoch 785/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1809 - binary_accuracy: 1.0000
    Epoch 786/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1807 - binary_accuracy: 1.0000
    Epoch 787/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1806 - binary_accuracy: 1.0000
    Epoch 788/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1805 - binary_accuracy: 1.0000
    Epoch 789/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1804 - binary_accuracy: 1.0000
    Epoch 790/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1803 - binary_accuracy: 1.0000
    Epoch 791/800
    4/4 [==============================] - 0s 499us/step - loss: 0.1802 - binary_accuracy: 1.0000
    Epoch 792/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1800 - binary_accuracy: 1.0000
    Epoch 793/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1799 - binary_accuracy: 1.0000
    Epoch 794/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1798 - binary_accuracy: 1.0000
    Epoch 795/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1797 - binary_accuracy: 1.0000
    Epoch 796/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1796 - binary_accuracy: 1.0000
    Epoch 797/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1795 - binary_accuracy: 1.0000
    Epoch 798/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1793 - binary_accuracy: 1.0000
    Epoch 799/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1792 - binary_accuracy: 1.0000
    Epoch 800/800
    4/4 [==============================] - 0s 748us/step - loss: 0.1791 - binary_accuracy: 1.0000
    




    <keras.callbacks.History at 0x215e28a6470>



앞서 실습한 로지스틱 회귀 코드와 거의 동일한데 달라진 점은 입력의 차원이 2로 바뀌면서, input_dim의 인자값이 2로 바뀌었다는 점입니다.

정확도는 100%가 나오고 있으므로 800회 정도로 학습을 멈추고 시그모이드 함수의 각 입력값에 대해서 출력값이 0.5보다 크고 작은지를 확인해보겠습니다.


```python
print(model.predict(X))
```

    [[0.35163167]
     [0.8766777 ]
     [0.8701636 ]
     [0.9887449 ]]
    

입력이 둘 다 0, 0인 경우를 제외하고 나머지 3개의 입력 쌍(pair)에 대해서는 전부 값이 0.5를 넘는 것을 볼 수 있습니다.

### 3) 인공 신경망 다이어그램

다중 로지스틱 회귀를 뒤에서 배우게 되는 인공 신경망의 형태로 표현하면 다음과 같습니다. 아직 인공 신경망을 배우지 않았음에도 이렇게 다이어그램으로 표현해보는 이유는 로지스틱 회귀를 일종의 인공 신경망 구조로 해석해도 무방함을 보여주기 위함입니다.

$y = sigmoid(W_1x_1 + W_2x_2 + W_3x_3 + ... + W_nx_n + b) = σ(W_1x_1 + W_2x_2 + W_3x_3 + ... + W_nx_n + b)$

![](https://wikidocs.net/images/page/35821/multiplelogistic_regression.PNG)

## 6. 벡터와 행렬 연산

다음 챕터의 소프트맥스 회귀에서는 종속 변수 y의 종류도 3개 이상이 되면서 더욱 복잡해집니다. 그리고 여기서 이러한 식들이 겹겹이 누적되는 층(layer) 개념까지 들어가기 시작하면 인공 신경망의 개념이 됩니다.

문제로부터 사용자가 데이터와 변수의 개수로부터 행렬의 크기, 더 나아가서는 텐서의 크기를 산정할 수 있어야 합니다.

인공 신경망의 각종 모델들에서 벡터와 행렬에 대한 개념이 지속적으로 등장하므로 이를 이해하고 있어야 합니다. 

### 1) 벡터와 행렬과 텐서

1. 벡터

벡터는 크기와 방향을 가진 양입니다.

숫자가 나열된 형상이며 파이썬에서는 1차원 배열 또는 리스트로 표현합니다.

2. 행렬

행렬은 행과 열을 가지는 2차원 형상을 가진 구조입니다.

파이썬에서는 2차원 배열로 표현합니다.

3. 텐서

3차원부터는 주로 텐서라고 부릅니다.

텐서는 파이썬에서는 3차원 이상의 배열로 표현합니다.

### 2) 텐서(Tensor)

인공 신경망은 복잡한 모델 내의 연산을 주로 행렬 연산을 통해 해결합니다.

머신 러닝의 입, 출력이 복잡해지면 3차원 텐서에 대한 이해가 필수로 요구됩니다.

텐서를 설명하기 위한 아래의 모든 코드는 Numpy를 임포트했다고 가정합니다.


```python
import numpy as np
```

#### (1) 0차원 텐서

스칼라는 하나의 실수값으로 이루어진 데이터를 말합니다. 

또한 스칼라값을 0차원 텐서라고 합니다. 차원을 영어로 Dimensionality라고 하므로 0D 텐서라고도 합니다.


```python
d=np.array(5)
print(d.ndim) # 차원수 출력
print(d.shape) # 텐서의 크기 출력 
```

    0
    ()
    

Numpy의 ndim은 축의 개수를 출력하는데, 이는 텐서에서의 차원수와 동일합니다.

#### (2) 1차원 텐서

숫자를 특정 순서대로 배열한 것을 벡터라고합니다. 또한 벡터를 1차원 텐서라고 합니다. 

주의할 점은 벡터의 차원과 텐서의 차원은 다른 개념이라는 점입니다. 아래의 예제는 4차원 벡터이지만, 1차원 텐서입니다. 1D 텐서라고도 합니다.


```python
d=np.array([1, 2, 3, 4])
print(d.ndim)
print(d.shape)
```

    1
    (4,)
    

벡터의 차원과 텐서의 차원의 정의로 인해 혼동할 수 있는데 벡터에서의 차원(Dimensionality)은 하나의 축에 차원들이 존재하는 것이고, 텐서에서의 차원(Dimensionality)은 축의 개수를 의미합니다.

#### (3) 2차원 텐서

행과 열이 존재하는 벡터의 배열. 즉, 행렬(matrix)을 2차원 텐서라고 합니다. 2D 텐서라고도 합니다.


```python
d=np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(d.ndim)
print(d.shape)
```

    2
    (3, 4)
    

#### (4) 3차원 텐서

행렬 또는 2차원 텐서를 단위로 한 번 더 배열하면 3차원 텐서라고 부릅니다. 3D 텐서라고도 합니다.

사실 위에서 언급한 0차원 ~ 2차원 텐서는 각각 스칼라, 벡터, 행렬이라고 해도 무방하므로 3차원 이상의 텐서부터 본격적으로 텐서라고 부릅니다. 

조금 쉽게 말하면 데이터 사이언스 분야 한정으로 주로 3차원 이상의 배열을 텐서라고 부릅니다. 


```python
d=np.array([
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [10, 11, 12, 13, 14]],
            [[15, 16, 17, 18, 19], [19, 20, 21, 22, 23], [23, 24, 25, 26, 27]]
            ])
print(d.ndim)
print(d.shape)
```

    3
    (2, 3, 5)
    

2개의 큰 데이터가 있는데, 그 각각은 3개의 더 작은 데이터로 구성되며, 그 3개의 데이터는 또한 더 작은 5개의 데이터로 구성되어져 있습니다.

자연어 처리에서 특히 자주 보게 되는 것이 이 3D 텐서입니다.

이 경우 3D 텐서는 (samples, timesteps, word_dim)이 됩니다. 또는 일괄로 처리하기 위해 데이터를 묶는 단위인 배치의 개념에 대해서 뒤에서 배울텐데 (batch_size, timesteps, word_dim)이라고도 볼 수 있습니다.

samples/batch_size는 데이터의 개수, timesteps는 시퀀스의 길이, word_dim은 단어를 표현하는 벡터의 차원을 의미합니다.

자연어 처리에서 왜 3D 텐서의 개념이 사용되는지 간단한 예를 들어봅시다.

- 문서1 : I like NLP
- 문서2 : I like DL
- 문서3 : DL is AI

인공 신경망의 모델의 입력으로 사용하기 위해서는 각 단어를 벡터화해야 합니다.

단어를 벡터화하는 방법으로는 원-핫 인코딩이나 워드 임베딩이라는 방법이 대표적이나 워드 임베딩은 아직 배우지 않았으므로 원-핫 인코딩으로 모든 단어를 벡터화 해보겠습니다.

|단어|One-hot vector|
|---|---|
|I|[1 0 0 0 0 0]|
|like|[0 1 0 0 0 0]|
|NLP|[0 0 1 0 0 0]|
|DL|[0 0 0 1 0 0]|
|is|[0 0 0 0 1 0]|
|AI|[0 0 0 0 0 1]|

 원-핫 벡터로 바꿔서 인공 신경망의 입력으로 한 꺼번에 사용한다고 하면 다음과 같습니다. (이렇게 훈련 데이터를 여러개 묶어서 한 꺼번에 입력으로 사용하는 것을 배치(Batch)라고 합니다.)

[[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]],<br/>
[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]],<br/>
[[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]]

이는 (3, 3, 6)의 크기를 가지는 3D 텐서입니다.

#### (5) 그 이상의 텐서

3차원 텐서를 배열로 합치면 4차원 텐서가 됩니다. 4차원 텐서를 배열로 합치면 5차원 텐서가 됩니다. 이런 식으로 텐서는 배열로서 계속해서 확장될 수 있습니다.

![](https://wikidocs.net/images/page/37001/tensor.png)

위의 그림은 각 텐서를 도형으로 시각화한 모습을 보여줍니다.

#### (6) 케라스에서의 텐서

케라스에서는 입력의 크기(shape)를 인자로 줄 때 input_shape라는 인자를 사용합니다.

input_shape는 배치 크기를 제외하고 차원을 지정하는데, <br/>예를 들어 input_shape(6, 5)라는 인자값을 사용하고 배치 크기를 32라고 지정한다면 이 텐서의 크기는 (32, 6, 5)을 의미합니다.

만약 배치 크기까지 지정해주고 싶다면 batch_input_shape=(8, 2, 10)와 같이 인자를 주면 이 텐서의 크기는 (8, 2, 10)을 의미합니다.

그 외에도 입력의 속성 수를 의미하는 input_dim, 시퀀스 데이터의 길이를 의미하는 input_length 등의 인자도 사용합니다. 사실 input_shape의 두 개의 인자는 (input_length, input_dim)라고 볼 수 있습니다.

### 3) 벡터와 행렬의 연산

여기서는 벡터와 행렬의 기본적인 연산에 대해서 알아보겠습니다. 아래의 모든 실습은 Numpy를 아래와 같이 임포트 해야 합니다.


```python
import numpy as np
```

#### (1) 벡터와 행렬의 덧셈과 뺄셈

같은 크기의 두 개의 벡터나 행렬은 덧셈과 뺄셈을 할 수 있습니다.

$a + b = \left[
    \begin{array}{c}
      8 \\
      4 \\
      5 \\
    \end{array}
  \right]
+ \left[
    \begin{array}{c}
      1 \\
      2 \\
      3 \\
    \end{array}
  \right]
= \left[
    \begin{array}{c}
      9 \\
      6 \\
      8 \\
    \end{array}
  \right]$

$a - b = \left[
    \begin{array}{c}
      8 \\
      4 \\
      5 \\
    \end{array}
  \right]
- \left[
    \begin{array}{c}
      1 \\
      2 \\
      3 \\
    \end{array}
  \right]
= \left[
    \begin{array}{c}
      7 \\
      2 \\
      2 \\
    \end{array}
  \right]$

Numpy를 이용하여 이를 구현할 수 있습니다.


```python
a = np.array([8, 4, 5])
b = np.array([1, 2, 3])
print(a+b)
print(a-b)
```

    [9 6 8]
    [7 2 2]
    

다음과 같이 a와 b라는 두 개의 행렬이 있다고 하였을 때, 두 행렬 a와 b의 덧셈과 뺄셈은 아래와 같습니다.

$a + b = \left[
    \begin{array}{c}
      10\ 20\ 30\ 40\\
      50\ 60\ 70\ 80\\
    \end{array}
  \right] 
+ \left[
    \begin{array}{c}
      5\ 6\ 7\ 8\\
      1\ 2\ 3\ 4\\
    \end{array}
  \right]
= \left[
    \begin{array}{c}
      15\ 26\ 37\ 48\\
      51\ 62\ 73\ 84\\
    \end{array}
  \right]$

$a - b = \left[
    \begin{array}{c}
      10\ 20\ 30\ 40\\
      50\ 60\ 70\ 80\\
    \end{array}
  \right] 
- \left[
    \begin{array}{c}
      5\ 6\ 7\ 8\\
      1\ 2\ 3\ 4\\
    \end{array}
  \right]
= \left[
    \begin{array}{c}
      5\ 14\ 23\ 32\\
      49\ 58\ 67\ 76\\
    \end{array}
  \right]$


```python
import numpy as np
a = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
b = np.array([[5, 6, 7, 8],[1, 2, 3, 4]])
print(a+b)
print(a-b)
```

    [[15 26 37 48]
     [51 62 73 84]]
    [[ 5 14 23 32]
     [49 58 67 76]]
    

#### (2) 벡터의 내적과 행렬의 곱셈

벡터의 내적은 연산을 점(dot)으로 표현하여 a⋅b와 같이 표현하기도 합니다.

내적이 성립하기 위해서는 두 벡터의 차원이 같아야 하며,

두 벡터 중 앞의 벡터가 행벡터(가로 방향 벡터)이고 뒤의 벡터가 열벡터(세로 방향 벡터)여야 합니다.

벡터의 내적의 결과는 스칼라가 된다는 특징이 있습니다.

$a \cdot b =
\left[
    \begin{array}{c}
      1\ 2\ 3
    \end{array}
  \right]
\left[
    \begin{array}{c}
      4 \\
      5 \\
      6 \\
    \end{array}
  \right]
= 1 × 4 + 2 × 5 + 3 × 6 = 32\text{(스칼라)}$


```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a,b))
```

    32
    

행렬의 곱셈을 이해하기 위해서는 벡터의 내적을 이해해야 합니다.

행렬의 곱셈은 왼쪽 행렬의 행벡터(가로 방향 벡터)와 오른쪽 행렬의 열벡터(세로 방향 벡터)의 내적(대응하는 원소들의 곱의 합)이 결과 행렬의 원소가 되는 것으로 이루어집니다. 

$ab =
\left[
    \begin{array}{c}
      1\ 3\\
      2\ 4\\
    \end{array}
  \right]
\left[
    \begin{array}{c}
      5\ 7\\
      6\ 8\\
    \end{array}
  \right]
= \left[
    \begin{array}{c}
      1 × 5 + 3 × 6\ \ \ 1 × 7 + 3 × 8\\
      2 × 5 + 4 × 6\ \ \ 2 × 7 + 4 × 8\\
    \end{array}
  \right]
=\left[
    \begin{array}{c}
      23\ 31\\
      34\ 46\\
    \end{array}
  \right]$


```python
import numpy as np
a = np.array([[1, 3],[2, 4]])
b = np.array([[5, 7],[6, 8]])
print(np.matmul(a,b))
```

    [[23 31]
     [34 46]]
    

### 3) 다중 선형 회귀 행렬 연산으로 이해하기

독립 변수가 2개 이상일 때, 1개의 종속 변수를 예측하는 문제를 행렬의 연산으로 표현한다면 어떻게 될까요?

다중 선형 회귀나 다중 로지스틱 회귀가 이러한 연산의 예인데, 여기서는 다중 선형 회귀를 통해 예를 들어보겠습니다.

$y = W_1x_1 + W_2x_2 + W_3x_3 + ... + W_nx_n + b$

$y = 
\left[
    \begin{array}{c}
      x_{1}\ x_{2}\ x_{3}\ \cdot\cdot\cdot\ x_{n}
    \end{array}
  \right]
\left[
    \begin{array}{c}
      W_{1} \\
      W_{2} \\
      W_{3} \\
      \cdot\cdot\cdot \\
      W_{n}
    \end{array}
  \right]
+
b
= x_1W_1 + x_2W_2 + x_3W_3 + ... + x_nW_n + b$

데이터의 개수가 많을 경우에는 벡터의 내적이 아니라 행렬의 곱셈으로 표현이 가능합니다.

$\left[
    \begin{array}{c}
      x_{11}\ x_{12}\ x_{13}\ x_{14} \\
      x_{21}\ x_{22}\ x_{23}\ x_{24} \\
      x_{31}\ x_{32}\ x_{33}\ x_{34} \\
      x_{41}\ x_{42}\ x_{43}\ x_{44} \\
      x_{51}\ x_{52}\ x_{53}\ x_{54} \\
    \end{array}
  \right]
\left[
    \begin{array}{c}
      W_{1} \\
      W_{2} \\
      W_{3} \\
      W_{4} \\
    \end{array}
  \right]
  =
\left[
    \begin{array}{c}
      x_{11}W_{1}\ x_{12}W_{2}\ x_{13}W_{3}\ x_{14}W_{4} \\
      x_{21}W_{1}\ x_{22}W_{2}\ x_{23}W_{3}\ x_{24}W_{4} \\
      x_{31}W_{1}\ x_{32}W_{2}\ x_{33}W_{3}\ x_{34}W_{4} \\
      x_{41}W_{1}\ x_{42}W_{2}\ x_{43}W_{3}\ x_{44}W_{4} \\
      x_{51}W_{1}\ x_{52}W_{2}\ x_{53}W_{3}\ x_{54}W_{4} \\
    \end{array}
  \right]$

여기에 편향 b를 더 해주면 독립 변수 값에 따라 달라지는 y의 값을 구하게 됩니다.

$\left[
    \begin{array}{c}
      x_{11}W_{1}\ x_{12}W_{2}\ x_{13}W_{3}\ x_{14}W_{4} \\
      x_{21}W_{1}\ x_{22}W_{2}\ x_{23}W_{3}\ x_{24}W_{4} \\
      x_{31}W_{1}\ x_{32}W_{2}\ x_{33}W_{3}\ x_{34}W_{4} \\
      x_{41}W_{1}\ x_{42}W_{2}\ x_{43}W_{3}\ x_{44}W_{4} \\
      x_{51}W_{1}\ x_{52}W_{2}\ x_{53}W_{3}\ x_{54}W_{4} \\
    \end{array}
  \right]
+
\left[
    \begin{array}{c}
      b \\
      b \\
      b \\
      b \\
      b \\
    \end{array}
  \right]
= 
\left[
    \begin{array}{c}
      y_{1}\\ y_{2}\\ y_{3}\\ y_{4}\\ y_{5} \\
    \end{array}
  \right]$

또는 가중치 벡터를 앞에 두고 입력 행렬을 뒤에 두고 행렬 연산을 할 수도 있는데, 이는 아래와 같습니다.

$\left[
    \begin{array}{c}
      W_{1}\ W_{2}\ W_{3}\ W_{4} \\
    \end{array}
  \right]
\left[
    \begin{array}{c}
      x_{11}\ x_{21}\ x_{31}\ x_{41}\ x_{51}\\
      x_{12}\ x_{22}\ x_{32}\ x_{42}\ x_{52}\\
      x_{13}\ x_{23}\ x_{33}\ x_{43}\ x_{53}\\
      x_{14}\ x_{24}\ x_{34}\ x_{44}\ x_{54}\\
    \end{array}
  \right]
+
\left[
    \begin{array}{c}
      b\ b\ b\ b\ b \\
    \end{array}
  \right]
=
\left[
    \begin{array}{c}
      y_{1}\ y_{2}\ y_{3}\ y_{4}\ y_{5} \\
    \end{array}
  \right]$

인공 신경망 자료들을 공부하게 되면 가중치 행렬에 입력 행렬을 곱하는 경우와 입력 행렬에 가중치 행렬을 곱하는 두 가지 경우를 모두 보게되므로 행렬의 곱셈이 익숙하지 않다면 이를 염두해두는 것이 좋습니다.

### 4) 샘플(Sample)과 특성(Feature)

![](https://wikidocs.net/images/page/35821/n_x_m.PNG)

머신 러닝에서는 하나의 데이터 각각을 샘플(Sample)이라고 부르며, 종속 변수 y를 예측하기 위한 각각의 독립 변수 x를 특성(Feature)이라고 부릅니다.

### 5) 가중치와 편향 행렬의 크기 결정

특성을 행렬의 열로 보는 경우를 가정하여 행렬의 크기가 어떻게 결정되는지 정리합니다.

행렬곱은 두 가지 정의를 가지는데, 두 개의 행렬 J와 K의 곱은 다음과 같은 조건을 충족해야 합니다.<br/>
1) 두 행렬의 곱 J x K에 대하여 행렬 J의 열의 수와 행렬 K의 행의 수는 같아야 한다.<br/>
2) 두 행렬의 곱 J x K의 결과로 나온 행렬 JK의 크기는 J의 행의 크기와 K의 열의 크기를 가진다.

이로부터 주어진 데이터가 입력과 출력의 행렬의 크기를 어떻게 가지느냐에 따라서 가중치 W의 행렬과 편향 b의 행렬의 크기를 찾아낼 수 있습니다. 

![](https://wikidocs.net/images/page/37001/matrix4.PNG)

- 배치 크기(Batch size)

전체 샘플 데이터 중 1개씩 불러와서 처리하고자한다면 m은 1이 됩니다. 또는 전체 데이터를 임의의 m개씩 묶인 작은 그룹들로 분할하여 여러번 처리할 수도 있는데 이렇게 처리하면서 기계가 학습하는 것을 미니배치 학습이라고 합니다.

예를 들어 전체 데이터가 1,024개가 있을 때 m을 64로 잡는다면 전체 데이터는 16개의 그룹으로 분할됩니다. 각 그룹은 총 64개의 샘플로 구성됩니다. 그리고 위에서 설명한 행렬 연산을 총 16번 반복하게되고 그제서야 전체 데이터에 대한 학습이 완료됩니다. 이때 64를 배치 크기(Batch size)라고 합니다.

## 7. 소프트맥스 회귀(Softmax Regression) - 다중 클래스 분류

로지스틱 회귀로 둘 중 하나의 범주로 분류하는 이진 분류 문제를 구현해보았습니다. 이번 챕터에서는 다중 클래스 분류 문제를 위한 소프트맥스 회귀(Softmax Regression)에 대해서 배웁니다.

### 1) 다중 클래스 분류(Multi-class Classification)

이진 분류가 두 개의 답 중 하나를 고르는 문제였다면, 세 개 이상의 답 중 하나를 고르는 문제를 다중 클래스 분류라고 합니다. 

예제에서 setosa, versicolor, virginica 3개 중에 하나의 정답을 골라야 하는데 이 샘플 데이터가 setosa일 확률이 0.58, versicolor일 확률 0.22, virginica일 확률이 0.2와 같이 총 합이 1인 확률 분포를 구할 수 있게 해보자는 것입니다. 이럴 때 사용할 수 있는 것이 소프트맥스 함수입니다.

### 2) 소프트맥스 함수(Softmax function)

소프트맥스 함수는 분류해야하는 정답지(클래스)의 총 개수를 k라고 할 때, k차원의 벡터를 입력받아 각 클래스에 대한 확률을 추정합니다.

#### (1) 소프트맥스 함수의 이해

k차원의 벡터에서 i번째 원소를 zi, i번째 클래스가 정답일 확률을 pi로 나타낸다고 하였을 때 소프트맥스 함수는 pi를 다음과 같이 정의합니다.

$p_{i}=\frac{e^{z_{i}}}{\sum_{j=1}^{k} e^{z_{j}}}\ \ for\ i=1, 2, ... k$

위에서 풀어야하는 문제의 경우 k=3이므로 3차원 벡터 z=[z1 z2 z3]의 입력을 받으면 소프트맥스 함수는 아래와 같은 출력을 리턴합니다.

$softmax(z)=[\frac{e^{z_{1}}}{\sum_{j=1}^{3} e^{z_{j}}}\ \frac{e^{z_{2}}}{\sum_{j=1}^{3} e^{z_{j}}}\ \frac{e^{z_{3}}}{\sum_{j=1}^{3} e^{z_{j}}}] = [p_{1}, p_{2}, p_{3}] = \hat{y} = \text{예측값}$

p1,p2,p3  각각은 1번 클래스가 정답일 확률, 2번 클래스가 정답일 확률, 3번 클래스가 정답일 확률을 나타내며 각각 0과 1사이의 값으로 총 합은 1이 됩니다.

이에따라 식을 문제에 맞게 다시 쓰면 아래와 같습니다.

$softmax(z)=[\frac{e^{z_{1}}}{\sum_{j=1}^{3} e^{z_{j}}}\ \frac{e^{z_{2}}}{\sum_{j=1}^{3} e^{z_{j}}}\ \frac{e^{z_{3}}}{\sum_{j=1}^{3} e^{z_{j}}}] = [p_{1}, p_{2}, p_{3}] = [p_{virginica}, p_{setosa}, p_{versicolor}]$

분류하고자 하는 클래스가 k개일 때, k차원의 벡터를 입력받아서 모든 벡터 원소의 값을 0과 1사이의 값으로 값을 변경하여 다시 k차원의 벡터를 리턴한다는 내용을 식으로 기재하였을 뿐입니다.

#### (2) 그림을 통한 이해

![](https://wikidocs.net/images/page/35476/softmax1_final_final.PNG)

여기서는 샘플 데이터를 1개씩 입력으로 받아 처리한다고 가정해봅시다. 즉, 배치 크기가 1입니다.

위의 그림에는 두 가지 질문이 있습니다. 첫번째 질문은 소프트맥스 함수의 입력에 대한 질문입니다. 하나의 샘플 데이터는 4개의 독립 변수 x를 가지는데 이는 모델이 4차원 벡터를 입력으로 받음을 의미합니다. 그런데 소프트맥스의 함수의 입력으로 사용되는 벡터는 벡터의 차원이 분류하고자 하는 클래스의 개수가 되어야 하므로 어떤 가중치 연산을 통해 3차원 벡터로 변환되어야 합니다. 위의 그림에서는 소프트맥스 함수의 입력으로 사용되는 3차원 벡터를 z로 표현하였습니다.

![](https://wikidocs.net/images/page/35476/softmaxbetween1and2.PNG)

샘플 데이터 벡터를 소프트맥스 함수의 입력 벡터로 차원을 축소하는 방법은 간단합니다. 소프트맥스 함수의 입력 벡터 z의 차원수만큼 결과값의 나오도록 가중치 곱을 진행합니다. 위의 그림에서 화살표는 총 (4 × 3 = 12) 12개이며 전부 다른 가중치를 가지고, 학습 과정에서 점차적으로 오차를 최소화하는 가중치로 값이 변경됩니다.

두번째 질문은 오차 계산 방법에 대한 질문입니다. 소프트맥스 함수의 출력은 분류하고자하는 클래스의 개수만큼 차원을 가지는 벡터로 각 원소는 0과 1사이의 값을 가집니다. 이 각각은 특정 클래스가 정답일 확률을 나타냅니다. 여기서는 첫번째 원소인 p1은 virginica가 정답일 확률, 두번째 원소인 p2는 setosa가 정답일 확률, 세번째 원소인 p3은 versicolor가 정답일 확률로 고려하고자 합니다. 그렇다면 이 예측값과 비교를 할 수 있는 실제값의 표현 방법이 있어야 합니다. 소프트맥스 회귀에서는 실제값을 원-핫 벡터로 표현합니다.

![](https://wikidocs.net/images/page/35476/softmax2_final.PNG)

위의 그림은 소프트맥스 함수의 출력 벡터의 첫번째 원소 p1가 virginica가 정답일 확률, 두번째 원소 p2가 setosa가 정답일 확률, 세번째 원소 p3가 versicolor가 정답일 확률을 의미한다고 하였을 때, 각 실제값의 정수 인코딩은 1, 2, 3이 되고 이에 원-핫 인코딩을 수행하여 실제값을 원-핫 벡터로 수치화한 것을 보여줍니다.

![](https://wikidocs.net/images/page/35476/softmax4.PNG)

예를 들어 현재 풀고 있는 샘플 데이터의 실제값이 setosa라면 setosa의 원-핫 벡터는 [0 1 0]입니다. 이 경우, 예측값과 실제값의 오차가 0이 되는 경우는 소프트맥스 함수의 결과가 [0 1 0]이 되는 경우입니다. 이 두 벡터의 오차를 계산하기 위해서 소프트맥스 회귀는 비용 함수로 크로스 엔트로피 함수를 사용하는데, 이는 뒤에서 비용 함수를 설명하는 부분에서 다시 언급하겠습니다.

![](https://wikidocs.net/images/page/35476/softmax5.PNG)

이제 앞서 배운 선형 회귀나 로지스틱 회귀와 마찬가지로 오차로부터 가중치를 업데이트 합니다.

![](https://wikidocs.net/images/page/35476/softmax6.PNG)

더 정확히는 선형 회귀나 로지스틱 회귀와 마찬가지로 편향 또한 업데이트의 대상이 되는 매개 변수입니다.

소프트맥스 회귀를 벡터와 행렬 연산으로 이해해봅시다. 입력을 특성(feature)의 수만큼의 차원을 가진 입력 벡터 x라고 하고, 가중치 행렬을 W, 편향을 b라고 하였을 때, 소프트맥스 회귀에서 예측값을 구하는 과정을 벡터와 행렬 연산으로 표현하면 아래와 같습니다.

![](https://wikidocs.net/images/page/35476/softmax7.PNG)

여기서 4는 특성의 수이며 3은 클래스의 개수에 해당됩니다.

### 3) 원-핫 벡터의 무작위성

꼭 실제값을 원-핫 벡터로 표현해야만 다중 클래스 분류 문제를 풀 수 있는 것은 아니지만, 대부분의 다중 클래스 분류 문제가 각 클래스 간의 관계가 균등하다는 점에서 원-핫 벡터는 이러한 점을 표현할 수 있는 적절한 표현 방법입니다.

- 정수 인코딩

직관적으로 생각해볼 수 있는 레이블링 방법은 분류해야 할 클래스 전체에 정수 인코딩을 하는 겁니다. 예를 들어서 분류해야 할 레이블이 {red, green, blue}와 같이 3개라면 각각 0, 1, 2로 레이블을 합니다. 또는 분류해야 할 클래스가 4개고 인덱스를 숫자 1부터 시작하고 싶다고 하면 {baby, child, adolescent, adult}라면 1, 2, 3, 4로 레이블을 해볼 수 있습니다. 

- 정수 인코딩 vs 원-핫 인코딩

그런데 일반적인 다중 클래스 분류 문제에서 레이블링 방법으로는 위와 같은 정수 인코딩이 아니라 원-핫 인코딩을 사용하는 것이 보다 클래스의 성질을 잘 표현하였다고 할 수 있습니다.

- 정수 인코딩보다 원-핫 인코딩이 더 좋은 이유?

Banana, Tomato, Apple라는 3개의 클래스가 존재하는 문제가 있다고 해봅시다. 레이블은 정수 인코딩을 사용하여 각각 1, 2, 3을 부여하였습니다. 손실 함수로 선형 회귀 챕터에서 배운 평균 제곱 오차 MSE를 사용하면 정수 인코딩이 어떤 오해를 불러일으킬 수 있는지 확인할 수 있습니다. 

아래는 세 개의 카테고리에 대해서 원-핫 인코딩을 통해서 레이블을 인코딩했을 때 각 클래스 간의 제곱 오차가 균등함을 보여줍니다.

$((1,0,0)-(0,1,0))^{2} = (1-0)^{2} + (0-1)^{2} + (0-0)^{2} = 2$

$((1,0,0)-(0,0,1))^{2} = (1-0)^{2} + (0-0)^{2} + (0-1)^{2} = 2$

다르게 표현하면 모든 클래스에 대해서 원-핫 인코딩을 통해 얻은 원-핫 벡터들은 모든 쌍에 대해서 유클리드 거리를 구해도 전부 유클리드 거리가 동일합니다. 원-핫 벡터는 이처럼 각 클래스의 표현 방법이 무작위성을 가진다는 점을 표현할 수 있습니다. 뒤에서 다시 언급되겠지만 이러한 원-핫 벡터의 관계의 무작위성은 때로는 단어의 유사성을 구할 수 없다는 단점으로 언급되기도 합니다.

### 4) 비용 함수(Cost function)

소프트맥스 회귀에서는 비용 함수로 크로스 엔트로피 함수를 사용합니다. 여기서는 소프트맥스 회귀에서의 크로스 엔트로피 함수뿐만 아니라, 다양한 표기 방법에 대해서 이해해보겠습니다.

#### (1) 크로스 엔트로피 함수

아래에서 y는 실제값을 나타내며, i와 k는 위에서 정의한 것과 같습니다. 즉, yi는 실제값 원-핫 벡터의 i번째 인덱스를 의미하며, pi는 샘플 데이터가 i번째 클래스일 확률을 나타냅니다. 표기에 따라서 $\hat{y}_{i}$로 표현하기도 합니다.

$cost(W) = -\sum_{i=1}^{k}y_{i}\ log(p_{i})$

이 함수가 왜 비용 함수로 적합한지 알아보겠습니다. c가 실제값 원-핫 벡터에서 1을 가진 원소의 인덱스라고 한다면, $p_{c}=1$은 $\hat{y}$가 y를 정확하게 예측한 경우가 됩니다. 이를 식에 대입해보면 −1log(1)=0이 되기 때문에, 결과적으로 $\hat{y}$가 y를 정확하게 예측한 경우의 크로스 엔트로피 함수의 값은 0이 됩니다. 즉, $-\sum_{i=1}^{k}y_{i}\ log(p_{i})$ 이 값을 최소화하는 방향으로 학습해야 합니다.

#### (2) 이진 분류에서의 크로스 엔트로피 함수

로지스틱 회귀에서 배운 크로스 엔트로피 함수식과 달라보이지만, 본질적으로는 동일한 함수식입니다. 로지스틱 회귀의 크로스 엔트로피 함수식으로부터 소프트맥스 회귀의 크로스 엔트로피 함수식을 도출해봅시다.

$cost(W) = -(y\ logH(X) + (1-y)\ log(1-H(X)))$

위의 식은 앞서 로지스틱 회귀에서 배웠던 크로스 엔트로피의 함수식을 보여줍니다. 위의 식에서 y를 y1, y−1을 y2로 치환하고 H(X)를 p1, 1−H(X)를 p2로 치환해봅시다. 결과적으로 아래의 식을 얻을 수 있습니다.

$-(y_{1}\ log(p_{1})+y_{2}\ log(p_{2}))$

이 식은 아래와 같이 표현할 수 있습니다.

$-(\sum_{i=1}^{2}y_{i}\ log\ p_{i})$

소프트맥스 회귀에서는 k의 값이 고정된 값이 아니므로 2를 k로 변경합니다.

$-(\sum_{i=1}^{k}y_{i}\ log\ p_{i})$

위의 식은 결과적으로 소프트맥스 회귀의 식과 동일합니다.

### 5) 소프트맥스 회귀(Softmax Regression)

다중 클래스 분류를 설명하기 위해 예를 들었던 품종 분류 문제입니다.

#### (1) 아이리스 품종 데이터에 대한 이해


```python
import pandas as pd
data = pd.read_csv('iris.csv',encoding='latin1')
```


```python
print(len(data)) # 총 샘플의 개수 출력
print(data[:5]) # 샘플 중 5개 출력
```

    150
       Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
    0   1            5.1           3.5            1.4           0.2  Iris-setosa
    1   2            4.9           3.0            1.4           0.2  Iris-setosa
    2   3            4.7           3.2            1.3           0.2  Iris-setosa
    3   4            4.6           3.1            1.5           0.2  Iris-setosa
    4   5            5.0           3.6            1.4           0.2  Iris-setosa
    

Species열은 몇 가지 품종으로 구성되어져 있는지 출력해보겠습니다.


```python
print("품종 종류:", data["Species"].unique(), sep="\n")
# 중복을 허용하지 않고, 있는 데이터의 모든 종류를 출력
```

    품종 종류:
    ['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']
    

즉, 이번 데이터를 가지고 푸는 문제는 주어진 샘플 데이터의 4개의 속성으로부터 3개 중 어떤 품종인지를 예측하는 문제가 되겠습니다.

3개의 품종이 4개의 속성에 대해서 어떤 분포를 가지고 있는지 시각화해봅시다.


```python
import seaborn as sns
#del data['Id'] # 인덱스 열 삭제
sns.set(style="ticks", color_codes=True)
g = sns.pairplot(data, hue="Species", palette="husl")
```


![png](NLP_basic_07_Machine_Learning_files/NLP_basic_07_Machine_Learning_354_0.png)


pairplot은 데이터프레임을 인수로 받아 데이터프레임의 각 열의 조합에 따라서 산점도(scatter plot)을 그립니다.


```python
sns.barplot(data['Species'], data['SepalWidthCm'], ci=None)
# 각 종과 특성에 대한 연관 관계
```




    <matplotlib.axes._subplots.AxesSubplot at 0x215e4885cf8>




![png](NLP_basic_07_Machine_Learning_files/NLP_basic_07_Machine_Learning_356_1.png)


barplot을 통해 종과 특성에 대한 연관관계를 출력할 수도 있습니다.

150개의 샘플 데이터 중에서 각 품종인 Iris-setosa, Iris-versicolor, Iris-virginica을 나타내는 데이터가 몇 개씩 있는지 확인해보겠습니다. 즉, Species열에서 각 품종이 몇 개있는지 확인합니다.


```python
import matplotlib.pyplot as plt
data['Species'].value_counts().plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x215e7388630>




![png](NLP_basic_07_Machine_Learning_files/NLP_basic_07_Machine_Learning_359_1.png)


데이터에 대한 구성을 파악하였다면, 이제 소프트맥스 회귀 모델을 구성하기 위해 전처리를 진행해야 합니다.

우선 Species열에 대해서 전부 수치화를 진행할 필요가 있습니다. 우선 원-핫 인코딩을 수행하기 전에 정수 인코딩을 수행합니다. 그리고 정상적으로 정수 인코딩이 수행되었는지 확인하기 위하여 다시 한 번 값의 분포를 출력합니다.


```python
data['Species'] = data['Species'].replace(['Iris-virginica','Iris-setosa','Iris-versicolor'],[0,1,2])
# Iris-virginica는 0, Iris-setosa는 1, Iris-versicolor는 2가 됨.
data['Species'].value_counts().plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x215e73e73c8>




![png](NLP_basic_07_Machine_Learning_files/NLP_basic_07_Machine_Learning_362_1.png)


이제 특성과 품종을 각각 종속 변수와 독립 변수 데이터로 분리하는 작업을 수행하고, 정확하게 분리가 되었는지 확인하기 위해 데이터 중 5개씩만 출력해보겠습니다.


```python
from sklearn.model_selection import train_test_split
data_X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values # X 데이터에 해당됩니다. X는 총 4개입니다.
data_y = data['Species'].values # Y 데이터에 해당됩니다. 예측해야하는 값입니다.

print(data_X[:5]) #X에 해당되는 데이터를 10개만 출력합니다.
print(data_y[:5]) #y에 해당되는 데이터를 10개만 출력합니다.
```

    [[5.1 3.5 1.4 0.2]
     [4.9 3.  1.4 0.2]
     [4.7 3.2 1.3 0.2]
     [4.6 3.1 1.5 0.2]
     [5.  3.6 1.4 0.2]]
    [1 1 1 1 1]
    

이제 훈련 데이터와 테스트 데이터의 분리와 원-핫 인코딩을 수행해보겠습니다.


```python
(X_train, X_test, y_train, y_test) = train_test_split(data_X, data_y, train_size=0.8, random_state=1)
# 훈련 데이터와 테스트 데이터를 8:2로 나눕니다. 또한 데이터의 순서를 섞습니다.
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# 훈련 데이터와 테스트 데이터에 대해서 원-핫 인코딩
print(y_train[:5])
print(y_test[:5])
```

    [[0. 0. 1.]
     [1. 0. 0.]
     [0. 0. 1.]
     [1. 0. 0.]
     [1. 0. 0.]]
    [[0. 1. 0.]
     [0. 0. 1.]
     [0. 0. 1.]
     [0. 1. 0.]
     [1. 0. 0.]]
    

이제 전처리 단계가 모두 끝이 났습니다.

#### (2) 소프트맥스 회귀


```python
from keras.models import Sequential # 케라스의 Sequential()을 임포트
from keras.layers import Dense # 케라스의 Dense()를 임포트
from keras import optimizers # 케라스의 옵티마이저를 임포트

model=Sequential()
model.add(Dense(3, input_dim=4, activation='softmax'))
sgd=optimizers.SGD(lr=0.01)
# 학습률(learning rate, lr)은 0.01로 합니다.
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
# 옵티마이저는 경사하강법의 일종인 adam을 사용합니다.
# 손실 함수(Loss function)은 평균제곱오차 크로스 엔트로피 함수를 사용합니다.
history=model.fit(X_train,y_train, batch_size=1, epochs=200, validation_data=(X_test, y_test))
# 주어진 X와 y데이터에 대해서 오차를 최소화하는 작업을 200번 시도합니다.
```

    Train on 120 samples, validate on 30 samples
    Epoch 1/200
    120/120 [==============================] - 0s 2ms/step - loss: 3.1604 - acc: 0.3667 - val_loss: 3.2442 - val_acc: 0.2000
    Epoch 2/200
    120/120 [==============================] - 0s 540us/step - loss: 2.0015 - acc: 0.3667 - val_loss: 1.8791 - val_acc: 0.2000
    Epoch 3/200
    120/120 [==============================] - 0s 532us/step - loss: 1.3139 - acc: 0.3667 - val_loss: 1.2945 - val_acc: 0.2000
    Epoch 4/200
    120/120 [==============================] - 0s 549us/step - loss: 1.0920 - acc: 0.3500 - val_loss: 1.0935 - val_acc: 0.2333
    Epoch 5/200
    120/120 [==============================] - 0s 524us/step - loss: 1.0113 - acc: 0.4083 - val_loss: 1.0069 - val_acc: 0.3333
    Epoch 6/200
    120/120 [==============================] - 0s 515us/step - loss: 0.9557 - acc: 0.5583 - val_loss: 0.9391 - val_acc: 0.4667
    Epoch 7/200
    120/120 [==============================] - 0s 515us/step - loss: 0.9076 - acc: 0.6417 - val_loss: 0.8918 - val_acc: 0.6000
    Epoch 8/200
    120/120 [==============================] - 0s 515us/step - loss: 0.8621 - acc: 0.6417 - val_loss: 0.8566 - val_acc: 0.6000
    Epoch 9/200
    120/120 [==============================] - 0s 525us/step - loss: 0.8270 - acc: 0.6750 - val_loss: 0.8221 - val_acc: 0.6333
    Epoch 10/200
    120/120 [==============================] - 0s 524us/step - loss: 0.7926 - acc: 0.7000 - val_loss: 0.7735 - val_acc: 0.6333
    Epoch 11/200
    120/120 [==============================] - 0s 532us/step - loss: 0.7648 - acc: 0.6250 - val_loss: 0.7665 - val_acc: 0.6000
    Epoch 12/200
    120/120 [==============================] - 0s 549us/step - loss: 0.7415 - acc: 0.6917 - val_loss: 0.7345 - val_acc: 0.6333
    Epoch 13/200
    120/120 [==============================] - 0s 544us/step - loss: 0.7170 - acc: 0.6917 - val_loss: 0.6921 - val_acc: 0.6333
    Epoch 14/200
    120/120 [==============================] - 0s 540us/step - loss: 0.6944 - acc: 0.6917 - val_loss: 0.6739 - val_acc: 0.6333
    Epoch 15/200
    120/120 [==============================] - 0s 524us/step - loss: 0.6773 - acc: 0.7083 - val_loss: 0.6493 - val_acc: 0.6667
    Epoch 16/200
    120/120 [==============================] - 0s 540us/step - loss: 0.6584 - acc: 0.7000 - val_loss: 0.6473 - val_acc: 0.6333
    Epoch 17/200
    120/120 [==============================] - 0s 532us/step - loss: 0.6466 - acc: 0.6667 - val_loss: 0.6035 - val_acc: 0.7667
    Epoch 18/200
    120/120 [==============================] - 0s 515us/step - loss: 0.6260 - acc: 0.7167 - val_loss: 0.6143 - val_acc: 0.6333
    Epoch 19/200
    120/120 [==============================] - 0s 524us/step - loss: 0.6149 - acc: 0.7250 - val_loss: 0.6019 - val_acc: 0.6333
    Epoch 20/200
    120/120 [==============================] - 0s 537us/step - loss: 0.6012 - acc: 0.7083 - val_loss: 0.5811 - val_acc: 0.6667
    Epoch 21/200
    120/120 [==============================] - 0s 532us/step - loss: 0.5901 - acc: 0.7167 - val_loss: 0.5623 - val_acc: 0.7000
    Epoch 22/200
    120/120 [==============================] - 0s 540us/step - loss: 0.5801 - acc: 0.7417 - val_loss: 0.5509 - val_acc: 0.7000
    Epoch 23/200
    120/120 [==============================] - 0s 529us/step - loss: 0.5712 - acc: 0.7167 - val_loss: 0.5583 - val_acc: 0.6333
    Epoch 24/200
    120/120 [==============================] - 0s 557us/step - loss: 0.5592 - acc: 0.7167 - val_loss: 0.5441 - val_acc: 0.6667
    Epoch 25/200
    120/120 [==============================] - 0s 549us/step - loss: 0.5534 - acc: 0.7417 - val_loss: 0.5361 - val_acc: 0.6667
    Epoch 26/200
    120/120 [==============================] - 0s 540us/step - loss: 0.5423 - acc: 0.7750 - val_loss: 0.5182 - val_acc: 0.7000
    Epoch 27/200
    120/120 [==============================] - 0s 532us/step - loss: 0.5345 - acc: 0.7500 - val_loss: 0.5210 - val_acc: 0.6667
    Epoch 28/200
    120/120 [==============================] - 0s 532us/step - loss: 0.5258 - acc: 0.7667 - val_loss: 0.5036 - val_acc: 0.7333
    Epoch 29/200
    120/120 [==============================] - 0s 557us/step - loss: 0.5179 - acc: 0.7750 - val_loss: 0.4955 - val_acc: 0.8000
    Epoch 30/200
    120/120 [==============================] - 0s 532us/step - loss: 0.5148 - acc: 0.7667 - val_loss: 0.5025 - val_acc: 0.6667
    Epoch 31/200
    120/120 [==============================] - 0s 540us/step - loss: 0.5083 - acc: 0.7833 - val_loss: 0.5054 - val_acc: 0.6333
    Epoch 32/200
    120/120 [==============================] - 0s 515us/step - loss: 0.4990 - acc: 0.7667 - val_loss: 0.4794 - val_acc: 0.8000
    Epoch 33/200
    120/120 [==============================] - 0s 524us/step - loss: 0.4923 - acc: 0.7833 - val_loss: 0.4662 - val_acc: 0.8333
    Epoch 34/200
    120/120 [==============================] - 0s 524us/step - loss: 0.4855 - acc: 0.7667 - val_loss: 0.4769 - val_acc: 0.6667
    Epoch 35/200
    120/120 [==============================] - 0s 524us/step - loss: 0.4845 - acc: 0.7667 - val_loss: 0.4557 - val_acc: 0.8333
    Epoch 36/200
    120/120 [==============================] - 0s 524us/step - loss: 0.4751 - acc: 0.8167 - val_loss: 0.4514 - val_acc: 0.8333
    Epoch 37/200
    120/120 [==============================] - 0s 520us/step - loss: 0.4701 - acc: 0.7833 - val_loss: 0.4507 - val_acc: 0.8333
    Epoch 38/200
    120/120 [==============================] - 0s 541us/step - loss: 0.4651 - acc: 0.7833 - val_loss: 0.4532 - val_acc: 0.8000
    Epoch 39/200
    120/120 [==============================] - 0s 540us/step - loss: 0.4591 - acc: 0.7667 - val_loss: 0.4362 - val_acc: 0.8667
    Epoch 40/200
    120/120 [==============================] - 0s 540us/step - loss: 0.4547 - acc: 0.8083 - val_loss: 0.4420 - val_acc: 0.8333
    Epoch 41/200
    120/120 [==============================] - 0s 565us/step - loss: 0.4489 - acc: 0.8167 - val_loss: 0.4237 - val_acc: 0.9000
    Epoch 42/200
    120/120 [==============================] - 0s 532us/step - loss: 0.4472 - acc: 0.8417 - val_loss: 0.4291 - val_acc: 0.8667
    Epoch 43/200
    120/120 [==============================] - 0s 559us/step - loss: 0.4400 - acc: 0.8083 - val_loss: 0.4276 - val_acc: 0.8333
    Epoch 44/200
    120/120 [==============================] - 0s 519us/step - loss: 0.4396 - acc: 0.8500 - val_loss: 0.4229 - val_acc: 0.8667
    Epoch 45/200
    120/120 [==============================] - 0s 524us/step - loss: 0.4326 - acc: 0.8500 - val_loss: 0.4143 - val_acc: 0.9000
    Epoch 46/200
    120/120 [==============================] - 0s 524us/step - loss: 0.4266 - acc: 0.8000 - val_loss: 0.4040 - val_acc: 0.9333
    Epoch 47/200
    120/120 [==============================] - 0s 549us/step - loss: 0.4241 - acc: 0.8750 - val_loss: 0.3990 - val_acc: 0.9333
    Epoch 48/200
    120/120 [==============================] - 0s 515us/step - loss: 0.4200 - acc: 0.8583 - val_loss: 0.4039 - val_acc: 0.9000
    Epoch 49/200
    120/120 [==============================] - 0s 524us/step - loss: 0.4172 - acc: 0.8583 - val_loss: 0.3946 - val_acc: 0.9333
    Epoch 50/200
    120/120 [==============================] - 0s 524us/step - loss: 0.4112 - acc: 0.9000 - val_loss: 0.4083 - val_acc: 0.8333
    Epoch 51/200
    120/120 [==============================] - 0s 529us/step - loss: 0.4099 - acc: 0.8417 - val_loss: 0.4025 - val_acc: 0.9000
    Epoch 52/200
    120/120 [==============================] - 0s 532us/step - loss: 0.4053 - acc: 0.8500 - val_loss: 0.3966 - val_acc: 0.9000
    Epoch 53/200
    120/120 [==============================] - 0s 515us/step - loss: 0.4025 - acc: 0.9000 - val_loss: 0.3763 - val_acc: 0.9667
    Epoch 54/200
    120/120 [==============================] - 0s 540us/step - loss: 0.3978 - acc: 0.8750 - val_loss: 0.3825 - val_acc: 0.9333
    Epoch 55/200
    120/120 [==============================] - 0s 541us/step - loss: 0.3934 - acc: 0.8667 - val_loss: 0.3962 - val_acc: 0.9000
    Epoch 56/200
    120/120 [==============================] - 0s 540us/step - loss: 0.3908 - acc: 0.8917 - val_loss: 0.3795 - val_acc: 0.9000
    Epoch 57/200
    120/120 [==============================] - 0s 524us/step - loss: 0.3860 - acc: 0.9000 - val_loss: 0.3775 - val_acc: 0.9000
    Epoch 58/200
    120/120 [==============================] - 0s 532us/step - loss: 0.3822 - acc: 0.8667 - val_loss: 0.3747 - val_acc: 0.9000
    Epoch 59/200
    120/120 [==============================] - ETA: 0s - loss: 0.3798 - acc: 0.911 - 0s 565us/step - loss: 0.3809 - acc: 0.9083 - val_loss: 0.3562 - val_acc: 0.9667
    Epoch 60/200
    120/120 [==============================] - 0s 524us/step - loss: 0.3778 - acc: 0.9083 - val_loss: 0.3513 - val_acc: 0.9667
    Epoch 61/200
    120/120 [==============================] - 0s 557us/step - loss: 0.3745 - acc: 0.8917 - val_loss: 0.3547 - val_acc: 0.9667
    Epoch 62/200
    120/120 [==============================] - 0s 545us/step - loss: 0.3705 - acc: 0.9250 - val_loss: 0.3527 - val_acc: 0.9667
    Epoch 63/200
    120/120 [==============================] - 0s 590us/step - loss: 0.3666 - acc: 0.9250 - val_loss: 0.3549 - val_acc: 0.9667
    Epoch 64/200
    120/120 [==============================] - 0s 582us/step - loss: 0.3648 - acc: 0.8917 - val_loss: 0.3506 - val_acc: 0.9667
    Epoch 65/200
    120/120 [==============================] - 0s 532us/step - loss: 0.3623 - acc: 0.9333 - val_loss: 0.3473 - val_acc: 0.9667
    Epoch 66/200
    120/120 [==============================] - 0s 515us/step - loss: 0.3599 - acc: 0.9250 - val_loss: 0.3442 - val_acc: 0.9667
    Epoch 67/200
    120/120 [==============================] - 0s 524us/step - loss: 0.3573 - acc: 0.9250 - val_loss: 0.3488 - val_acc: 0.9667
    Epoch 68/200
    120/120 [==============================] - 0s 571us/step - loss: 0.3548 - acc: 0.9250 - val_loss: 0.3488 - val_acc: 0.9667
    Epoch 69/200
    120/120 [==============================] - 0s 607us/step - loss: 0.3510 - acc: 0.9250 - val_loss: 0.3524 - val_acc: 0.9333
    Epoch 70/200
    120/120 [==============================] - 0s 615us/step - loss: 0.3475 - acc: 0.9250 - val_loss: 0.3450 - val_acc: 0.9667
    Epoch 71/200
    120/120 [==============================] - 0s 615us/step - loss: 0.3471 - acc: 0.9167 - val_loss: 0.3499 - val_acc: 0.9333
    Epoch 72/200
    120/120 [==============================] - 0s 607us/step - loss: 0.3450 - acc: 0.9083 - val_loss: 0.3479 - val_acc: 0.9333
    Epoch 73/200
    120/120 [==============================] - 0s 598us/step - loss: 0.3399 - acc: 0.9250 - val_loss: 0.3368 - val_acc: 0.9667
    Epoch 74/200
    120/120 [==============================] - 0s 582us/step - loss: 0.3377 - acc: 0.8833 - val_loss: 0.3202 - val_acc: 0.9667
    Epoch 75/200
    120/120 [==============================] - 0s 590us/step - loss: 0.3363 - acc: 0.9333 - val_loss: 0.3190 - val_acc: 0.9667
    Epoch 76/200
    120/120 [==============================] - 0s 590us/step - loss: 0.3318 - acc: 0.9333 - val_loss: 0.3152 - val_acc: 0.9667
    Epoch 77/200
    120/120 [==============================] - 0s 632us/step - loss: 0.3280 - acc: 0.9500 - val_loss: 0.3110 - val_acc: 0.9667
    Epoch 78/200
    120/120 [==============================] - 0s 615us/step - loss: 0.3265 - acc: 0.9417 - val_loss: 0.3204 - val_acc: 0.9667
    Epoch 79/200
    120/120 [==============================] - 0s 607us/step - loss: 0.3243 - acc: 0.9333 - val_loss: 0.3118 - val_acc: 0.9667
    Epoch 80/200
    120/120 [==============================] - 0s 623us/step - loss: 0.3205 - acc: 0.9333 - val_loss: 0.3182 - val_acc: 0.9667
    Epoch 81/200
    120/120 [==============================] - 0s 623us/step - loss: 0.3221 - acc: 0.9333 - val_loss: 0.2961 - val_acc: 0.9667
    Epoch 82/200
    120/120 [==============================] - 0s 598us/step - loss: 0.3166 - acc: 0.9500 - val_loss: 0.3081 - val_acc: 0.9667
    Epoch 83/200
    120/120 [==============================] - 0s 651us/step - loss: 0.3147 - acc: 0.9333 - val_loss: 0.3112 - val_acc: 0.9667
    Epoch 84/200
    120/120 [==============================] - 0s 677us/step - loss: 0.3127 - acc: 0.9333 - val_loss: 0.3214 - val_acc: 0.9667
    Epoch 85/200
    120/120 [==============================] - 0s 648us/step - loss: 0.3110 - acc: 0.9500 - val_loss: 0.3086 - val_acc: 0.9667
    Epoch 86/200
    120/120 [==============================] - 0s 640us/step - loss: 0.3087 - acc: 0.9333 - val_loss: 0.3090 - val_acc: 0.9667
    Epoch 87/200
    120/120 [==============================] - 0s 694us/step - loss: 0.3070 - acc: 0.9500 - val_loss: 0.3034 - val_acc: 0.9667
    Epoch 88/200
    120/120 [==============================] - 0s 573us/step - loss: 0.3038 - acc: 0.9500 - val_loss: 0.2969 - val_acc: 0.9667
    Epoch 89/200
    120/120 [==============================] - 0s 607us/step - loss: 0.3006 - acc: 0.9417 - val_loss: 0.3013 - val_acc: 0.9667
    Epoch 90/200
    120/120 [==============================] - 0s 621us/step - loss: 0.2986 - acc: 0.9417 - val_loss: 0.3046 - val_acc: 0.9667
    Epoch 91/200
    120/120 [==============================] - 0s 590us/step - loss: 0.2981 - acc: 0.9333 - val_loss: 0.3148 - val_acc: 0.9667
    Epoch 92/200
    120/120 [==============================] - 0s 573us/step - loss: 0.2981 - acc: 0.9417 - val_loss: 0.3080 - val_acc: 0.9667
    Epoch 93/200
    120/120 [==============================] - 0s 598us/step - loss: 0.2948 - acc: 0.9417 - val_loss: 0.2851 - val_acc: 0.9667
    Epoch 94/200
    120/120 [==============================] - 0s 557us/step - loss: 0.2922 - acc: 0.9500 - val_loss: 0.2987 - val_acc: 0.9667
    Epoch 95/200
    120/120 [==============================] - 0s 573us/step - loss: 0.2915 - acc: 0.9500 - val_loss: 0.3054 - val_acc: 0.9667
    Epoch 96/200
    120/120 [==============================] - 0s 657us/step - loss: 0.2880 - acc: 0.9250 - val_loss: 0.2836 - val_acc: 0.9667
    Epoch 97/200
    120/120 [==============================] - 0s 582us/step - loss: 0.2845 - acc: 0.9500 - val_loss: 0.2768 - val_acc: 0.9667
    Epoch 98/200
    120/120 [==============================] - 0s 573us/step - loss: 0.2832 - acc: 0.9500 - val_loss: 0.2824 - val_acc: 0.9667
    Epoch 99/200
    120/120 [==============================] - 0s 557us/step - loss: 0.2816 - acc: 0.9500 - val_loss: 0.2732 - val_acc: 0.9667
    Epoch 100/200
    120/120 [==============================] - 0s 557us/step - loss: 0.2815 - acc: 0.9417 - val_loss: 0.2729 - val_acc: 0.9667
    Epoch 101/200
    120/120 [==============================] - 0s 615us/step - loss: 0.2773 - acc: 0.9667 - val_loss: 0.2752 - val_acc: 0.9667
    Epoch 102/200
    120/120 [==============================] - 0s 598us/step - loss: 0.2767 - acc: 0.9500 - val_loss: 0.2682 - val_acc: 0.9667
    Epoch 103/200
    120/120 [==============================] - 0s 573us/step - loss: 0.2760 - acc: 0.9583 - val_loss: 0.2633 - val_acc: 0.9667
    Epoch 104/200
    120/120 [==============================] - 0s 565us/step - loss: 0.2721 - acc: 0.9583 - val_loss: 0.2636 - val_acc: 0.9667
    Epoch 105/200
    120/120 [==============================] - 0s 598us/step - loss: 0.2689 - acc: 0.9667 - val_loss: 0.2684 - val_acc: 0.9667
    Epoch 106/200
    120/120 [==============================] - 0s 598us/step - loss: 0.2697 - acc: 0.9583 - val_loss: 0.2668 - val_acc: 0.9667
    Epoch 107/200
    120/120 [==============================] - 0s 582us/step - loss: 0.2661 - acc: 0.9583 - val_loss: 0.2562 - val_acc: 0.9667
    Epoch 108/200
    120/120 [==============================] - 0s 623us/step - loss: 0.2657 - acc: 0.9583 - val_loss: 0.2610 - val_acc: 0.9667
    Epoch 109/200
    120/120 [==============================] - 0s 623us/step - loss: 0.2656 - acc: 0.9333 - val_loss: 0.2630 - val_acc: 0.9667
    Epoch 110/200
    120/120 [==============================] - 0s 590us/step - loss: 0.2620 - acc: 0.9583 - val_loss: 0.2537 - val_acc: 0.9667
    Epoch 111/200
    120/120 [==============================] - 0s 645us/step - loss: 0.2598 - acc: 0.9417 - val_loss: 0.2660 - val_acc: 0.9667
    Epoch 112/200
    120/120 [==============================] - 0s 595us/step - loss: 0.2580 - acc: 0.9583 - val_loss: 0.2554 - val_acc: 0.9667
    Epoch 113/200
    120/120 [==============================] - 0s 607us/step - loss: 0.2580 - acc: 0.9500 - val_loss: 0.2536 - val_acc: 0.9667
    Epoch 114/200
    120/120 [==============================] - 0s 611us/step - loss: 0.2558 - acc: 0.9500 - val_loss: 0.2594 - val_acc: 0.9667
    Epoch 115/200
    120/120 [==============================] - 0s 598us/step - loss: 0.2536 - acc: 0.9583 - val_loss: 0.2508 - val_acc: 0.9667
    Epoch 116/200
    120/120 [==============================] - 0s 573us/step - loss: 0.2550 - acc: 0.9500 - val_loss: 0.2587 - val_acc: 0.9667
    Epoch 117/200
    120/120 [==============================] - 0s 590us/step - loss: 0.2524 - acc: 0.9333 - val_loss: 0.2617 - val_acc: 0.9667
    Epoch 118/200
    120/120 [==============================] - 0s 590us/step - loss: 0.2487 - acc: 0.9583 - val_loss: 0.2571 - val_acc: 0.9667
    Epoch 119/200
    120/120 [==============================] - 0s 590us/step - loss: 0.2480 - acc: 0.9500 - val_loss: 0.2622 - val_acc: 0.9667
    Epoch 120/200
    120/120 [==============================] - 0s 615us/step - loss: 0.2458 - acc: 0.9583 - val_loss: 0.2531 - val_acc: 0.9667
    Epoch 121/200
    120/120 [==============================] - 0s 773us/step - loss: 0.2452 - acc: 0.9583 - val_loss: 0.2511 - val_acc: 0.9667
    Epoch 122/200
    120/120 [==============================] - 0s 731us/step - loss: 0.2443 - acc: 0.9583 - val_loss: 0.2529 - val_acc: 0.9667
    Epoch 123/200
    120/120 [==============================] - 0s 698us/step - loss: 0.2412 - acc: 0.9667 - val_loss: 0.2390 - val_acc: 0.9667
    Epoch 124/200
    120/120 [==============================] - 0s 706us/step - loss: 0.2407 - acc: 0.9583 - val_loss: 0.2352 - val_acc: 0.9667
    Epoch 125/200
    120/120 [==============================] - 0s 723us/step - loss: 0.2395 - acc: 0.9667 - val_loss: 0.2383 - val_acc: 0.9667
    Epoch 126/200
    120/120 [==============================] - 0s 727us/step - loss: 0.2383 - acc: 0.9583 - val_loss: 0.2438 - val_acc: 0.9667
    Epoch 127/200
    120/120 [==============================] - 0s 723us/step - loss: 0.2360 - acc: 0.9583 - val_loss: 0.2424 - val_acc: 0.9667
    Epoch 128/200
    120/120 [==============================] - 0s 765us/step - loss: 0.2353 - acc: 0.9667 - val_loss: 0.2299 - val_acc: 0.9667
    Epoch 129/200
    120/120 [==============================] - 0s 687us/step - loss: 0.2347 - acc: 0.9583 - val_loss: 0.2400 - val_acc: 0.9667
    Epoch 130/200
    120/120 [==============================] - 0s 731us/step - loss: 0.2327 - acc: 0.9500 - val_loss: 0.2315 - val_acc: 0.9667
    Epoch 131/200
    120/120 [==============================] - 0s 738us/step - loss: 0.2321 - acc: 0.9583 - val_loss: 0.2369 - val_acc: 0.9667
    Epoch 132/200
    120/120 [==============================] - 0s 774us/step - loss: 0.2312 - acc: 0.9500 - val_loss: 0.2426 - val_acc: 0.9667
    Epoch 133/200
    120/120 [==============================] - 0s 773us/step - loss: 0.2272 - acc: 0.9667 - val_loss: 0.2275 - val_acc: 0.9667
    Epoch 134/200
    120/120 [==============================] - 0s 773us/step - loss: 0.2286 - acc: 0.9583 - val_loss: 0.2230 - val_acc: 0.9667
    Epoch 135/200
    120/120 [==============================] - 0s 748us/step - loss: 0.2266 - acc: 0.9583 - val_loss: 0.2225 - val_acc: 0.9667
    Epoch 136/200
    120/120 [==============================] - 0s 682us/step - loss: 0.2264 - acc: 0.9500 - val_loss: 0.2315 - val_acc: 0.9667
    Epoch 137/200
    120/120 [==============================] - 0s 773us/step - loss: 0.2245 - acc: 0.9667 - val_loss: 0.2304 - val_acc: 0.9667
    Epoch 138/200
    120/120 [==============================] - 0s 715us/step - loss: 0.2234 - acc: 0.9583 - val_loss: 0.2146 - val_acc: 1.0000
    Epoch 139/200
    120/120 [==============================] - 0s 790us/step - loss: 0.2205 - acc: 0.9500 - val_loss: 0.2277 - val_acc: 0.9667
    Epoch 140/200
    120/120 [==============================] - 0s 723us/step - loss: 0.2197 - acc: 0.9417 - val_loss: 0.2344 - val_acc: 0.9667
    Epoch 141/200
    120/120 [==============================] - 0s 681us/step - loss: 0.2179 - acc: 0.9667 - val_loss: 0.2184 - val_acc: 0.9667
    Epoch 142/200
    120/120 [==============================] - 0s 632us/step - loss: 0.2195 - acc: 0.9583 - val_loss: 0.2110 - val_acc: 1.0000
    Epoch 143/200
    120/120 [==============================] - 0s 648us/step - loss: 0.2173 - acc: 0.9500 - val_loss: 0.2273 - val_acc: 0.9667
    Epoch 144/200
    120/120 [==============================] - 0s 690us/step - loss: 0.2156 - acc: 0.9583 - val_loss: 0.2256 - val_acc: 0.9667
    Epoch 145/200
    120/120 [==============================] - 0s 648us/step - loss: 0.2151 - acc: 0.9667 - val_loss: 0.2162 - val_acc: 0.9667
    Epoch 146/200
    120/120 [==============================] - 0s 640us/step - loss: 0.2140 - acc: 0.9500 - val_loss: 0.2201 - val_acc: 0.9667
    Epoch 147/200
    120/120 [==============================] - 0s 607us/step - loss: 0.2125 - acc: 0.9583 - val_loss: 0.2123 - val_acc: 0.9667
    Epoch 148/200
    120/120 [==============================] - 0s 623us/step - loss: 0.2130 - acc: 0.9583 - val_loss: 0.2189 - val_acc: 0.9667
    Epoch 149/200
    120/120 [==============================] - 0s 621us/step - loss: 0.2093 - acc: 0.9667 - val_loss: 0.2174 - val_acc: 0.9667
    Epoch 150/200
    120/120 [==============================] - 0s 661us/step - loss: 0.2080 - acc: 0.9583 - val_loss: 0.2192 - val_acc: 0.9667
    Epoch 151/200
    120/120 [==============================] - 0s 682us/step - loss: 0.2098 - acc: 0.9500 - val_loss: 0.2118 - val_acc: 0.9667
    Epoch 152/200
    120/120 [==============================] - 0s 673us/step - loss: 0.2061 - acc: 0.9583 - val_loss: 0.2098 - val_acc: 0.9667
    Epoch 153/200
    120/120 [==============================] - ETA: 0s - loss: 0.2148 - acc: 0.967 - 0s 640us/step - loss: 0.2060 - acc: 0.9667 - val_loss: 0.2196 - val_acc: 0.9667
    Epoch 154/200
    120/120 [==============================] - 0s 665us/step - loss: 0.2035 - acc: 0.9583 - val_loss: 0.2125 - val_acc: 0.9667
    Epoch 155/200
    120/120 [==============================] - 0s 640us/step - loss: 0.2027 - acc: 0.9583 - val_loss: 0.2051 - val_acc: 0.9667
    Epoch 156/200
    120/120 [==============================] - 0s 648us/step - loss: 0.2023 - acc: 0.9583 - val_loss: 0.2056 - val_acc: 0.9667
    Epoch 157/200
    120/120 [==============================] - 0s 623us/step - loss: 0.2010 - acc: 0.9500 - val_loss: 0.2195 - val_acc: 0.9667
    Epoch 158/200
    120/120 [==============================] - 0s 615us/step - loss: 0.2011 - acc: 0.9667 - val_loss: 0.2031 - val_acc: 0.9667
    Epoch 159/200
    120/120 [==============================] - 0s 615us/step - loss: 0.2003 - acc: 0.9417 - val_loss: 0.2056 - val_acc: 0.9667
    Epoch 160/200
    120/120 [==============================] - 0s 623us/step - loss: 0.1994 - acc: 0.9583 - val_loss: 0.2093 - val_acc: 0.9667
    Epoch 161/200
    120/120 [==============================] - 0s 615us/step - loss: 0.1974 - acc: 0.9583 - val_loss: 0.2010 - val_acc: 0.9667
    Epoch 162/200
    120/120 [==============================] - 0s 607us/step - loss: 0.1968 - acc: 0.9583 - val_loss: 0.2027 - val_acc: 0.9667
    Epoch 163/200
    120/120 [==============================] - 0s 623us/step - loss: 0.1972 - acc: 0.9583 - val_loss: 0.2035 - val_acc: 0.9667
    Epoch 164/200
    120/120 [==============================] - 0s 590us/step - loss: 0.1948 - acc: 0.9500 - val_loss: 0.2096 - val_acc: 0.9667
    Epoch 165/200
    120/120 [==============================] - 0s 595us/step - loss: 0.1947 - acc: 0.9583 - val_loss: 0.2007 - val_acc: 0.9667
    Epoch 166/200
    120/120 [==============================] - 0s 657us/step - loss: 0.1943 - acc: 0.9583 - val_loss: 0.1964 - val_acc: 0.9667
    Epoch 167/200
    120/120 [==============================] - 0s 657us/step - loss: 0.1918 - acc: 0.9583 - val_loss: 0.2002 - val_acc: 0.9667
    Epoch 168/200
    120/120 [==============================] - 0s 662us/step - loss: 0.1929 - acc: 0.9667 - val_loss: 0.1960 - val_acc: 0.9667
    Epoch 169/200
    120/120 [==============================] - 0s 673us/step - loss: 0.1923 - acc: 0.9500 - val_loss: 0.1917 - val_acc: 1.0000
    Epoch 170/200
    120/120 [==============================] - 0s 668us/step - loss: 0.1897 - acc: 0.9583 - val_loss: 0.1970 - val_acc: 0.9667
    Epoch 171/200
    120/120 [==============================] - 0s 661us/step - loss: 0.1887 - acc: 0.9583 - val_loss: 0.1973 - val_acc: 0.9667
    Epoch 172/200
    120/120 [==============================] - 0s 628us/step - loss: 0.1876 - acc: 0.9583 - val_loss: 0.1985 - val_acc: 0.9667
    Epoch 173/200
    120/120 [==============================] - 0s 665us/step - loss: 0.1888 - acc: 0.9500 - val_loss: 0.1998 - val_acc: 0.9667
    Epoch 174/200
    120/120 [==============================] - 0s 665us/step - loss: 0.1859 - acc: 0.9500 - val_loss: 0.1984 - val_acc: 0.9667
    Epoch 175/200
    120/120 [==============================] - 0s 648us/step - loss: 0.1873 - acc: 0.9583 - val_loss: 0.1929 - val_acc: 0.9667
    Epoch 176/200
    120/120 [==============================] - 0s 665us/step - loss: 0.1859 - acc: 0.9583 - val_loss: 0.1918 - val_acc: 0.9667
    Epoch 177/200
    120/120 [==============================] - 0s 631us/step - loss: 0.1837 - acc: 0.9500 - val_loss: 0.1959 - val_acc: 0.9667
    Epoch 178/200
    120/120 [==============================] - 0s 632us/step - loss: 0.1833 - acc: 0.9500 - val_loss: 0.1967 - val_acc: 0.9667
    Epoch 179/200
    120/120 [==============================] - 0s 623us/step - loss: 0.1836 - acc: 0.9583 - val_loss: 0.1919 - val_acc: 0.9667
    Epoch 180/200
    120/120 [==============================] - 0s 657us/step - loss: 0.1832 - acc: 0.9583 - val_loss: 0.1971 - val_acc: 0.9667
    Epoch 181/200
    120/120 [==============================] - 0s 648us/step - loss: 0.1836 - acc: 0.9500 - val_loss: 0.1916 - val_acc: 0.9667
    Epoch 182/200
    120/120 [==============================] - 0s 648us/step - loss: 0.1809 - acc: 0.9583 - val_loss: 0.1887 - val_acc: 0.9667
    Epoch 183/200
    120/120 [==============================] - 0s 657us/step - loss: 0.1812 - acc: 0.9583 - val_loss: 0.1830 - val_acc: 1.0000
    Epoch 184/200
    120/120 [==============================] - 0s 648us/step - loss: 0.1796 - acc: 0.9583 - val_loss: 0.1864 - val_acc: 0.9667
    Epoch 185/200
    120/120 [==============================] - 0s 631us/step - loss: 0.1802 - acc: 0.9500 - val_loss: 0.1806 - val_acc: 1.0000
    Epoch 186/200
    120/120 [==============================] - 0s 740us/step - loss: 0.1771 - acc: 0.9500 - val_loss: 0.1879 - val_acc: 0.9667
    Epoch 187/200
    120/120 [==============================] - 0s 657us/step - loss: 0.1763 - acc: 0.9583 - val_loss: 0.1889 - val_acc: 0.9667
    Epoch 188/200
    120/120 [==============================] - 0s 623us/step - loss: 0.1773 - acc: 0.9583 - val_loss: 0.1802 - val_acc: 1.0000
    Epoch 189/200
    120/120 [==============================] - 0s 623us/step - loss: 0.1757 - acc: 0.9583 - val_loss: 0.1791 - val_acc: 1.0000
    Epoch 190/200
    120/120 [==============================] - 0s 640us/step - loss: 0.1742 - acc: 0.9417 - val_loss: 0.1893 - val_acc: 0.9667
    Epoch 191/200
    120/120 [==============================] - 0s 610us/step - loss: 0.1738 - acc: 0.9583 - val_loss: 0.1839 - val_acc: 0.9667
    Epoch 192/200
    120/120 [==============================] - 0s 665us/step - loss: 0.1733 - acc: 0.9500 - val_loss: 0.1830 - val_acc: 0.9667
    Epoch 193/200
    120/120 [==============================] - 0s 665us/step - loss: 0.1728 - acc: 0.9500 - val_loss: 0.1875 - val_acc: 0.9667
    Epoch 194/200
    120/120 [==============================] - 0s 665us/step - loss: 0.1714 - acc: 0.9583 - val_loss: 0.1798 - val_acc: 0.9667
    Epoch 195/200
    120/120 [==============================] - 0s 657us/step - loss: 0.1728 - acc: 0.9583 - val_loss: 0.1872 - val_acc: 0.9667
    Epoch 196/200
    120/120 [==============================] - 0s 640us/step - loss: 0.1698 - acc: 0.9583 - val_loss: 0.1859 - val_acc: 0.9667
    Epoch 197/200
    120/120 [==============================] - 0s 640us/step - loss: 0.1706 - acc: 0.9667 - val_loss: 0.1726 - val_acc: 1.0000
    Epoch 198/200
    120/120 [==============================] - 0s 653us/step - loss: 0.1687 - acc: 0.9583 - val_loss: 0.1836 - val_acc: 0.9667
    Epoch 199/200
    120/120 [==============================] - 0s 620us/step - loss: 0.1686 - acc: 0.9583 - val_loss: 0.1734 - val_acc: 1.0000
    Epoch 200/200
    120/120 [==============================] - 0s 648us/step - loss: 0.1678 - acc: 0.9583 - val_loss: 0.1766 - val_acc: 0.9667
    

함수로는 소프트맥스 함수를 사용하므로 activation에는 softmax를 기재해줍니다.

오차 함수로는 크로스 엔트로피 함수를 사용합니다. 이진 분류 문제에서는 binary_crossentropy를 사용하였지만, 다중 클래스 분류 문제에서는 'categorical_crossentropy를 기재해주어야 합니다. 옵티마이저로는 경사 하강법의 일종인 adam을 사용합니다. 전체 데이터에 대한 훈련 횟수는 200회로 주었습니다. 

이번에는 테스트 데이터를 별도로 분리해서 평가에 사용하였는데, validation_data=()에 테스트 데이터를 기재해주면 실제로는 훈련에는 반영되지 않으면서 각 훈련 횟수마다 테스트 데이터에 대한 정확도를 출력합니다. 즉, 정확도가 전체 데이터에 대한 훈련 1회(1 에포크)마다 측정되고는 있지만 기계는 저 데이터를 가지고는 가중치를 업데이트하지 않습니다. 이해가 되지 않는다면 뒤의 로이터 뉴스 분류하기 챕터에서 다시 설명하므로 여기서는 넘어가도 좋습니다.

acc은 훈련 데이터에 대한 정확도이고, val_acc은 테스트 데이터에 대한 정확도를 의미합니다. 훈련 데이터에서는 95%의 정확도를 보이고, 테스트 데이터에 대해서는 100%의 정확도를 보입니다.

이번에는 각 에포크당 훈련 데이터와 테스트 데이터에 대한 정확도를 측정했으므로 한 번 에포크에 따른 정확도를 그래프로 출력해보겠습니다.


```python
epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
```


![png](NLP_basic_07_Machine_Learning_files/NLP_basic_07_Machine_Learning_375_0.png)


에포크가 증가함에 따라 정확도가 점차적으로 줄어드는 것을 볼 수 있습니다. 이미 테스트 데이터에 대한 정확도를 validation_data=()를 통해 알고는 있지만 케라스에서 테스트 데이터의 정확도를 측정하는 용도로 제공하고 있는 evaluate()를 통해 테스트 데이터에 대한 정확도를 다시 출력해보겠습니다.


```python
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))
```

    30/30 [==============================] - 0s 498us/step
    
     테스트 정확도: 0.9667
    

### 6) 인공 신경망 다이어그램

n개의 특성을 가지고 m개의 클래스를 분류하는 소프트맥스 회귀를 뒤에서 배우게 되는 인공 신경망의 형태로 표현하면 다음과 같습니다. 

![](https://wikidocs.net/images/page/35476/softmax_regression_nn.PNG)

사실 위의 그림은 앞서 소프트맥스 함수를 사용하기 위해 설명했던 아래의 그림에서 특성의 개수를 n으로 하고, 클래스의 개수를 m으로 일반화한 뒤에 그림을 좀 더 요약해서 표현한 것으로 봐도 무방합니다.

![](https://wikidocs.net/images/page/35476/softmax6.PNG)
