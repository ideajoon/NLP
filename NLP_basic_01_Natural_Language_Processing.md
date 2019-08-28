written by ideajoon<br/>
※ 참고 : 딥 러닝을 이용한 자연어 처리 입문
(https://wikidocs.net/book/2155) 자료를 공부하고 정리함

# 01. 자연어 처리(natural language processing)란?

## 목차
1. Pandas 사용
2. Numpy(넘파이) 사용
3. Matplotlib(맷플롯립)
4. 머신 러닝 워크플로우(Machine
Learning Workflow)

### nltk와 ntlk data 설치
- nltk는 자연어 처리를 위해 파이썬 패키지입니다.
- nltk의 기능을 제대로 사용하기 위해서는
nltk data라는 nltk의 여러 실습을 위한 각각의 데이터를 추가적으로 설치해야 합니다.

```
pip install nltk
import nltk
nltk.download()
```

### KoNLpy 설치
- 코엔엘파이(KoNLpy)는 한국어 형태소 분석기로, 한글 자연어 처리를 위해 만들어진 패키지입니다. 
- 한글에
대한 예제 실습을 위해 코엔엘파이를 설치합니다.

```
pip install konlpy
```

### 텐서플로우(Tensorflow) 설치
- 텐서플로우는 구글이 2015년에 공개한 기계 학습 오픈소스 라이브러리입니다. 
- 기계 학습과
딥 러닝을 직관적이고 손쉽게 할 수 있도록 설계되었습니다. 딥 러닝을 위해 텐서플로우를 설치해야 합니다.

```
pip install tensorflow
```

### 케라스(Keras) 설치
- 케라스(Keras)는 딥 러닝 프레임워크인 텐서플로우에 대한 추상화 된 API를 제공합니다. 
- 케라스는
백엔드로 텐서플로우를 사용하며, 좀 더 쉽게 딥 러닝을 사용할 수 있게 해줍니다. 
- 쉽게 말해, 텐서플로우의 코드를 훨씬 간단하게 작성할 수
있습니다.

```
pip install keras
```

### 넘파이(Numpy) 설치
- 넘파이(numpy)는 빠른 계산을 위해 지원되는 파이썬 라이브러리입니다.
- 넘파이 메뉴얼 :
http://docs.scipy.org/odc/numpy/

```
pip install numpy
conda install numpy
```

### 사이킷런(Scikit-learn) 설치
- 사이킷런은 파이썬 머신러닝 라이브러리입니다. 
- 사이킷런을 통해 나이브 베이즈 분류, 서포트
벡터 머신 등 다양한 머신 러닝 모듈을 불러올 수 있습니다. 
- 또한, 사이킷런에는 머신러닝을 연습하기 위한 아이리스 데이터, 당뇨병 데이터
등 자체 데이터 또한 제공하고 있습니다.
- 사이킷런은 넘파이(numpy)와 사이파이(scipy)라는 모듈이 먼저 설치되어야 하지만, 
-
아나콘다를 통해서 한꺼번에 설치하는 것도 가능합니다. 아나콘다로 설치 시에는 다음과 같은 명령어로 설치합니다.

```
conda install scikit-learn
```

## 1. Pandas 사용
Pandas는 총 세 가지의 데이터 구조를 사용합니다.
1. 시리즈(Series)
2.
데이터프레임(DataFrame)
3. 패널(Panel)

```python
import pandas as pd
```

### 1) 시리즈(Series)
- 시리즈 클래스는 1차원 배열의 값(values)에 각 값에 대응되는 인덱스(index)를 부여할 수 있는
구조를 갖고 있습니다.

```python
sr = pd.Series([17000, 18000, 1000, 5000],
       index=["피자", "치킨", "콜라", "맥주"])
print(sr)
```

```python
print(sr.values)
```

```python
print(sr.index)
```

### 2) 데이터프레임(DataFrame)
- 데이터프레임은 2차원 리스트를 매개변수로 전달합니다.
- 2차원이므로 행방향
인덱스(index)와 열방향 인덱스(column)가 존재합니다. 
- 즉, 행과 열을 가지는 자료구조입니다. 시리즈가 인덱스(index)와
값(values)으로 구성된다면, 
- 데이터프레임은 열(columns)까지 추가되어 열(columns), 인덱스(index),
값(values)으로 구성됩니다.

```python
values = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
index = ['one', 'two', 'three']
columns = ['A', 'B', 'C']
```

```python
df = pd.DataFrame(values, index = index, columns = columns)
print(df)
```

```python
print(df.index)
```

```python
print(df.columns)
```

```python
print(df.values)
```

### 3) 데이터프레임의 생성
- 데이터프레임은 리스트(List), 시리즈(Series), 딕셔너리(dict), Numpy의 ndarrays,
또 다른 데이터프레임으로 생성할 수 있습니다. 
- 여기서는 리스트와 딕셔너리를 통해서 데이터프레임을 생성 해보겠습니다.

```python
# 리스트로 생성하기
data = [
    ['1000', 'Steve', 90.72], 
    ['1001', 'James', 78.09], 
    ['1002', 'Doyeon', 98.43], 
    ['1003', 'Jane', 64.19], 
    ['1004', 'Pilwoong', 81.30],
    ['1005', 'Tony', 99.14],
]
```

```python
df = pd.DataFrame(data)
print(df)
```

```python
# 생성된 데이터프레임에 열(columns)을 지정해줄 수 있습니다.
df = pd.DataFrame(data, columns=['학번', '이름', '점수'])
print(df)
```

```python
# 딕셔너리로 생성하기
data = { '학번' : ['1000', '1001', '1002', '1003', '1004', '1005'],
         '이름' : [ 'Steve', 'James', 'Doyeon', 'Jane', 'Pilwoong', 'Tony'],
         '점수' : [90.72, 78.09, 98.43, 64.19, 81.30, 99.14]}
```

```python
df = pd.DataFrame(data)
print(df)
```

### 4) 데이터프레임 조회하기
아래의 명령어는 데이터프레임에서 원하는 구간만 확인하기 위한 명령어로서 유용하게 사용됩니다.
-
df.head(n) - 앞 부분을 n개만 보기
- df.tail(n) - 뒷 부분을 n개만 보기
- df['열이름'] - 해당되는 열을 확인

```python
print(df.head(3)) # 앞 부분을 3개만 보기
```

```python
print(df.tail(3)) # 뒷 부분을 3개만 보기
```

```python
print(df['학번']) # '학번'에 해당되는 열을 보기
```

### 5) 외부 데이터 읽기
- Pandas는 CSV, 텍스트, Excel, SQL, HTML, JSON 등 다양한 데이터 파일을 읽고 데이터
프레임을 생성할 수 있습니다.
- 예를 들어 csv 파일을 읽을 때는 pandas.read_csv()를 통해 읽을 수 있습니다.
- 다음과 같은
example.csv 파일이 있다고 합시다.

```python
df = pd.read_csv(r'C:\Users\joon2\example.csv')
```

```python
print(df)
```

```python
print(df.index)
```

## 2. Numpy(넘파이) 사용
- Numpy(보통 "넘파이"라고 발음.)는 수치 데이터를 다루는 Python 패키지입니다. 
-
Numpy의 핵심이라고 불리는 다차원 행렬 자료구조인 ndarray를 통해 벡터 및 행렬을 사용하는 선형 대수 계산에서 주로 사용됩니다. 
-
Numpy는 편의성뿐만 아니라, 속도면에서도 순수 파이썬에 비해 압도적으로 빠르다는 장점이 있습니다.

```python
import numpy as np
```

Numpy의 주요 모듈
- np.array()    => 리스트, 튜플, 배열로 부터 ndarray를 생성
- np.asarray()  =>
기존의 array로 부터 ndarray를 생성
- np.arange()   => range와 비슷
- np.linspace(start, end,
num)   => [start, end] 균일한 간격으로 num개 생성
- np.logspace(start, end, num)   =>
[start, end] log scale 간격으로 num개 생성

### 1) np.array()
- 앞서 Numpy의 핵심은 ndarray라고 언급한 바 있습니다. 
- np.array()는 리스트, 튜플,
배열로 부터 ndarray를 생성합니다. 
- 또한 인덱스가 항상 0으로 시작한다는 특징을 갖고 있습니다.

```python
a = np.array([1, 2, 3, 4, 5]) #리스트를 가지고 1차원 배열 생성
```

```python
print(type(a))
print(a)
```

```python
b = np.array([[10, 20, 30], [ 60, 70, 80]]) 
```

```python
print(b)
```

```python
print(b.ndim) #차원 출력
print(b.shape) #크기 출력
```

위의 결과는 각각 2차원이며, 2 X 3 행렬임을 보여줍니다. 가령, 앞서 만들었던 1차원 배열 a에 대해서 차원 및 크기를 출력하면 다음과
같습니다.

```python
print(a.ndim) #차원 출력
print(a.shape) #크기 출력
```

각각 1차원과 크기 5의 배열임을 보여줍니다.

### 2) ndarray의 초기화
- 위에서는 리스트를 가지고 ndarray를 생성했지만, ndarray를 만드는 다양한 다른 방법이
존재합니다. 
- zeros()는 해당 배열에 모두 0을 집어 넣고, ones()는 모두 1을 집어 넣는다. 
- full()은 배열에 사용자가
지정한 값을 넣는데 사용하고, 
- eye()는 대각선으로는 1이고 나머지는 0인 2차원 배열을 생성합니다.

```python
a = np.zeros((2,3)) # 모든값이 0인 2x3 배열 생성.
print(a)
```

```python
a = np.ones((2,3)) # 모든값이 1인 2x3 배열 생성.
print(a)
```

```python
a = np.full((2,2), 7) # 모든 값이 특정 상수인 배열 생성. 이 경우에는 7.
print(a)
```

```python
a = np.eye(3) # 대각선으로는 1이고 나머지는 0인 2차원 배열을 생성.
print(a)
```

```python
a = np.random.random((2,2)) # 임의의 값으로 채워진 배열 생성
print(a)
```

### 3) np.arange()
np.arange()는 지정해준 범위에 대해서 배열을 생성합니다. np.array()의 범위 지정 방법은
다음과 같습니다.
- numpy.arange(start, stop, step, dtype)
- a = np.arange(n) => 0, ...,
n-1까지 범위의 지정.
- a = np.arange(i, j, k) => i부터 j-1까지 k씩 증가하는 배열.

```python
a = np.arange(10) #0부터 9까지
print(a)
```

```python
a = np.arange(1, 10, 2) #1부터 9까지 +2씩 적용되는 범위
print(a)
```

### 4) reshape()

```python
a = np.array(np.arange(30)).reshape((5,6))
print(a)
```

위의 예제는 0부터 n-1 까지의 숫자를 생성하는 arange(n) 함수에 배열을 다차원으로 변형하는 reshape()를 통해 배열을
생성합니다.

### 5) Numpy 슬라이싱
- ndarray를 통해 만든 다차원 배열은 파이썬의 리스트처럼 슬라이스(Slice) 기능을 지원합니다. 
-
라이스 기능을 사용하면 원소들 중 복수 개에 접근할 수 있습니다.

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
```

```python
b = a[0:2, 0:1]
print(b)
```

다차원 배열을 슬라이싱하기 위해서는 각 차원 별로 슬라이스 범위를 지정해줘야 합니다.

```python
b=a[0, :] # 첫번째 행 출력
print(b)
```

```python
b=a[:, 1] # 두번째 열 출력
print(b)
```

### 6) Numpy 정수 인덱싱(integer indexing)
정수 인덱싱은 원본 배열로부터 부분 배열을 구합니다.

```python
a = np.array([[1,2], [4,5], [7,8]])
b = a[[2, 1],[1, 0]] # a[[row2, col1],[row1, col0]을 의미함.
```

```python
print(a)
```

```python
print(b)
```

### 7) Numpy 연산
- Numpy를 사용하면 배열간 연산을 손쉽게 수행할 수 있습니다. 
- +, -, *, /의 연산자를 사용할 수
있으며, 
- 또는 add(), substract(), multiply(), divide() 함수를 사용할 수도 있습니다.

```python
x = np.array([1,2,3])
y = np.array([4,5,6])
```

```python
b = x + y # 각 요소에 대해서 더함
print(b)
```

```python
b = np.add(x, y) # 각 요소에 대해서 더함
print(b)
```

```python
b = x - y # 각 요소에 대해서 빼기
print(b)
```

```python
b = np.subtract(x, y) # 각 요소에 대해서 빼기
print(b)
```

```python
b = b * x # 각 요소에 대해서 곱셈
print(b)
```

```python
b = np.multiply(b, x) # 각 요소에 대해서 곱셈
print(b)
```

```python
b = b / x # 각 요소에 대해서 나눗셈
print(b)
```

```python
b = np.divide(b, x) # 각 요소에 대해서 나눗셈
print(b)
```

위에서 *를 통해 수행한 것은 요소별 곱이었습니다. Numpy에서 벡터와 행렬의 곱 또는 행렬곱을 위해서는 dot()을 사용해야 합니다.

```python
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
```

```python
print(a)
```

```python
print(b)
```

```python
c = np.dot(a, b) #행렬 곱셈
print(c)
```

## 3. Matplotlib(맷플롯립)
- Matplotlib는 데이터를 차트(chart)나 플롯(plot)으로
시각화(visulaization)하는 패키지입니다. 
- 데이터 분석에서 Matplotlib은 데이터 분석 이전에 데이터 이해를 위한 시각화나,
데이터 분석 후에 결과를 시각화하기 위해서 사용됩니다.

```python
%matplotlib inline  
import matplotlib.pyplot as plt
```

주피터 노트북에 그림을 표시하도록 지정하는 '%matplotlib inline' 우선 수행해야 합니다.

### 1) 라인 플롯 그리기
- plot()은 라인 플롯을 그리는 기능을 수행합니다. 
- plot() X축과 Y축의 값을 기재하고 그림을
표시하는 show()를 통해서 시각화해봅시다. 
- 그래프에는 제목을 지정해줄 수 있는데 이 경우에는 title('원하는 제목')을 사용합니다.
- 여기서는 그래프에 'test'라는 제목을 넣어봅시다.

사실 주피터 노트북에서는 show()를 사용하지 않더라도 그래프가 자동으로 렌더링
되므로 그래프가 시각화가 되는 것을 확인할 수 있지만, 여기서는 다른 개발 환경에서 사용할 때 또한 가정하여 show()를 실습 코드에
삽입하였습니다.

```python
plt.title('test')
x = [1,2,3,4]
y = [2,4,8,6]
plt.plot(x, y)
plt.show()
```

### 2) 축 레이블 삽입하기
- 그래프에 제목을 넣기 위해서 title()을 사용하였다면, 
- X축과 Y축 각각에 축이름을 삽입하고 싶다면
xlabel('넣고 싶은 축이름')과 ylabel('넣고 싶은 축이름')을 사용하면 됩니다. 
- 위에서 시각화한 그래프에 hours와
score라는 축이름을 각각 X축과 Y축에 추가해봅시다.

```python
plt.title('test')
plt.plot([1,2,3,4],[2,4,8,6])
plt.xlabel('hours')
plt.ylabel('score')
plt.show()
```

### 3) 라인 추가와 범례 삽입하기
- 하나의 plot()뿐만 아니라 여러개의 plot()을 사용하여 하나의 그래프에 나타낼 수 있습니다.
- 여러개의 라인 플롯을 동시에 사용할 경우에는 각 선이 어떤 데이터를 나타내는지를 보여주기 위해 범례(legend)를 사용할 수 있습니다.

```python
plt.title('students')
plt.plot([1,2,3,4],[2,4,8,6])
plt.plot([1.5,2.5,3.5,4.5],[3,5,8,10]) #라인 새로 추가
plt.xlabel('hours')
plt.ylabel('score')
plt.legend(['A student', 'B student']) #범례 삽입
plt.show()
```

## 4. 머신 러닝 워크플로우(Machine Learning Workflow)

![](https://wikidocs.net/images/page/31947/%EB%A8%B8%EC%8B%A0_%EB%9F%AC%EB%8B%9D_%EC%9B%8C%ED%81%AC%ED%94%8C%EB%A1%9C%EC%9A%B0.PNG)

### 1) 수집(Acquisition)
- 머신 러닝을 하기 위해서는 기계에 학습시켜야 할 데이터가 필요합니다. 
- 자연어 처리의 경우,
자연어 데이터를 코퍼스(corpus)라고 부르는데 
- 코퍼스의 의미를 풀이하면, 조사나 연구 목적에 의해서 특정 도메인으로부터 수집된 텍스트
집합을 말합니다.
- 코퍼스. 즉, 텍스트 데이터의 파일 형식은 txt 파일, csv 파일, xml 파일 등 다양하며 
- 그 출처도 음성
데이터, 웹 수집기를 통해 수집된 데이터, 영화 리뷰 등 다양합니다.

### 2) 점검 및 탐색(Inspection and exploration)
- 데이터가 수집되었다면, 이제 데이터를 점검하고 탐색하는
단계입니다. 
- 여기서는 데이터의 구조, 노이즈 데이터, 머신 러닝 적용을 위해서 데이터를 어떻게 정제해야하는지 등을 파악해야 합니다.
- 이
단계를 탐색적 데이터 분석(Exploratory Data Analysis, EDA) 단계라고도 하는데 
- 이는 독립 변수, 종속 변수, 변수
유형, 변수의 데이터 타입 등을 점검하며 
- 데이터의 특징과 내재하는 구조적 관계를 알아내는 과정을 의미합니다. 
- 이 과정에서 시각화와
간단한 통계 테스트를 진행하기도 합니다.

### 3) 전처리 및 정제(Preprocessing and Cleaning)
- 데이터에 대한 파악이 끝났다면, 머신 러닝 워크플로우에서 가장
까다로운 작업 중 하나인 데이터 전처리 과정에 들어갑니다. 
- 이 단계는 많은 단계를 포함하고 있는데, 가령 자연어 처리라면 토큰화, 정제,
정규화, 불용어 제거 등의 단계를 포함합니다. 
- 빠르고 정확한 데이터 전처리를 하기 위해서는 사용하고 있는 툴(이 책에서는 파이썬)에 대한
다양한 라이브러리에 대한 지식이 필요합니다. 
- 정말 까다로운 전처리의 경우에는 전처리 과정에서 머신 러닝이 사용되기도 합니다.

### 4) 모델링 및 훈련(Modeling and Training)
- 데이터 전처리가 끝났다면, 머신 러닝에 대한 코드를 작성하는 단계인
모델링 단계에 들어갑니다. 
- 적절한 머신 러닝 알고리즘을 선택하여 모델링이 끝났다면, 
- 전처리가 완료 된 데이터를 머신 러닝 알고리즘을
통해 기계에게 학습(Training)시킵니다. 
- 기계가 데이터에 대한 학습을 마치고나서 훈련이 제대로 되었다면 
- 그 후에 기계는 우리가
원하는 태스크(Task)인 기계 번역, 음성 인식, 텍스트 분류 등의 자연어 처리 작업을 수행할 수 있게 됩니다.


- 여기서 주의해야할 점은
대부분의 경우에서 모든 데이터를 기계에게 학습시켜서는 안 된다는 점입니다. 
- 뒤의 머신 러닝, 딥 러닝 챕터에서 실제로 보게되겠지만 데이터
중 일부는 테스트용으로 남겨두고 훈련용 데이터만 훈련에 사용해야 합니다. 
- 그래야만 기계가 학습을 하고나서, 현재 성능이 얼마나 되는지를
측정할 수 있으며 과적합(Overfitting) 상황을 막을 수 있습니다. 
- 데이터의 양이 충분하여 더 세부적으로 나눌 수 있다면 훈련용,
검증용, 테스트용. 데이터를 이렇게 세 가지로 나누고 
- 훈련용 데이터만 훈련에 사용하기도 합니다.
![](https://wikidocs.net/images/page/31947/%EB%8D%B0%EC%9D%B4%ED%84%B0.PNG)
- 수능
시험에 비유하자면 훈련용은 학습지, 검증용은 모의고사, 테스트용은 수능 시험이라고 볼 수 있습니다. 
- 검증용 데이터는 현재 모델의 성능.
즉, 기계가 훈련용 데이터로 얼마나 제대로 학습이 되었는지를 판단하는 용으로 사용되며 
- 검증용 데이터를 사용하여 모델의 성능을 개선하는데
사용됩니다. 
- 테스트용 데이터는 모델의 최종 성능을 평가하는 데이터로 모델의 성능을 개선하는 일에 사용되는 것이 아니라, 
- 모델의 성능을
수치화하기 위해 사용됩니다. 이에 대한 자세한 내용은 뒤에서 다시 다룹니다.

### 5) 평가(Evaluation)
- 위에서 검증용 데이터와 테스트용 데이터의 차이를 설명하기 위해서 미리 언급하였는데, 
- 기계가 다
학습이 되었다면 테스트용 데이터로 성능을 평가하게 됩니다. 
- 평가 방법은 기계가 예측한 데이터가 테스트용 데이터의 실제 정답과 얼마나
가까운지를 측정합니다.

### 6) 배포(Deployment)
- 평가 단계에서 기계가 성공적으로 훈련이 된 것으로 판단된다면, 완성된 모델이 배포되는 단계가 됩니다.
- 다만, 여기서 완성된 모델에 대한 전체적인 피드백에 대해서 모델을 변경해야하는 상황이 온다면 
- 다시 처음부터 돌아가야 하는 상황이 올 수
있습니다.


- 그림에서는 언제든 다시 처음부터 돌아갈 수 있다는 의미에서 배포 단계에서 화살표를 수집 단계로 그렸지만, 
- 실제로는 거의
모든 단계에서 전 단계로 돌아가는 상황이 자주 발생합니다.

```python

```
