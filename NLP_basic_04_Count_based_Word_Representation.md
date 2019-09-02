
written by ideajoon<br/>
※ 참고 : 딥 러닝을 이용한 자연어 처리 입문 (https://wikidocs.net/book/2155) 자료를 공부하고 정리함

# 04. 카운트 기반의 단어 표현(Count based word Representation)

머신 러닝 등의 알고리즘이 적용된 본격적인 자연어 처리를 위해서는 문자를 숫자로 수치화할 필요가 있습니다.

## 목차
1. 다양한 단어의 표현 방법
2. Bag of Words(BoW)
3. 문서 단어 행렬(Document-Term Matrix, DTM)
4. TF-IDF(Term Frequency-Inverse Document Frequency)

## 1. 다양한 단어의 표현 방법

### 1) 단어의 표현 방법

(1) 국소 표현(Local representation) = 이산 표현(Discrete Representation)<br/>
: 해당 단어 그 자체만 보고, 특정값을 맵핑하여 단어를 표현하는 방법<br/>
예를 들어 puppy(강아지), cute(귀여운), lovely(사랑스러운)라는 단어가 있을 때 각 단어에 1번, 2번, 3번 등과 같은 숫자를 맵핑(Mapping)하여 부여


(2) 분산 표현(Distributed Representation) = 연속 표현(Continuous Represnetation)<br/>
: 그 단어를 표현하고자 주변을 참고하여 단어를 표현하는 방법<br/>
예를 들어 puppy(강아지)라는 단어 근처에는 주로 cute(귀여운), lovely(사랑스러운)이라는 단어가 자주 등장하므로, puppy라는 단어는 cute, lovely한 느낌이다로 단어를 정의

이 두 방법의 차이는 국소 표현 방법은 단어의 의미, 뉘앙스를 표현할 수 없지만, 분산 표현 방법은 단어의 뉘앙스를 표현할 수 있게 됩니다.

### 2) 단어 표현의 카테고리화

![](https://wikidocs.net/images/page/31767/%EB%8B%A8%EC%96%B4_%ED%91%9C%ED%98%84.PNG)

 Bag of Words는 국소 표현에(Local Representation)에 속하며, 단어의 빈도수를 카운트(Count)하여 단어를 수치화하는 단어 표현 방법입니다.

TF-IDF는 빈도수 기반 단어 표현에 단어의 중요도에 따른 가중치를 줄 수 있는 단어 표현 방법이다.

LSA는 단어의 뉘앙스를 반영하는 연속 표현(Continuous Representation)의 일종이다.

워드투벡터(Word2Vec)는 연속 표현(Continuous Representation)에 속하면서, 예측(Prediction)을 기반으로 단어의 뉘앙스를 표현하는 단어 표현 방법이다.

패스트텍스트(FastText)는 워드투벡터(Word2Vec)의 확장이다.

글로브(Glove)는 예측과 카운트라는 두 가지 방법이 모두 사용된 단어 표현 방법이다. 

## 2. Bag of Words(BoW)

### 1) Bag of Words란?

Bag of Words란 단어들의 순서는 전혀 고려하지 않고, 단어들의 출현 빈도(frequency)에만 집중하는 텍스트 데이터의 수치화 표현 방법입니다.

BoW를 만드는 과정을 이렇게 두 가지 과정으로 생각해보겠습니다.
- (1) 우선, 각 단어에 고유한 인덱스(Index)를 부여합니다.
- (2) 각 인덱스의 위치에 단어 토큰의 등장 횟수를 기록한 벡터(Vector)를 만듭니다.

한국어 예제를 통해서 BoW에 대해서 이해해보도록 하겠습니다.
- 문서1 : 정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.


```python
from konlpy.tag import Okt
import re  
okt = Okt()  

token = re.sub("(\.)","","정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.")  
# 정규 표현식을 통해 온점을 제거하는 정제 작업입니다.  
token = okt.morphs(token)  
# OKT 형태소 분석기를 통해 토큰화 작업을 수행한 뒤에, token에다가 넣습니다.  

word2index = {}  
bow = []

for voca in token:  
         if voca not in word2index.keys():  
             word2index[voca] = len(word2index)  
# token을 읽으면서, word2index에 없는 (not in) 단어는 새로 추가하고, 이미 있는 단어는 넘깁니다.   
             bow.insert(len(word2index) - 1,1)
# BoW 전체에 전부 기본값 1을 넣어줍니다. 단어의 개수는 최소 1개 이상이기 때문입니다.  
         else:
            index = word2index.get(voca)
# 재등장하는 단어의 인덱스를 받아옵니다.
            bow[index] = bow[index] + 1
# 재등장한 단어는 해당하는 인덱스의 위치에 1을 더해줍니다. (단어의 개수를 세는 것입니다.)  
print(word2index)
```

    {'정부': 0, '가': 1, '발표': 2, '하는': 3, '물가상승률': 4, '과': 5, '소비자': 6, '느끼는': 7, '은': 8, '다르다': 9}
    

문서1에 각 단어에 대해서 인덱스를 부여한 결과는 첫번째 출력 결과입니다. 첫번째 출력 결과를 단어 집합(Vocabulary)이라고 부릅니다. 여기서 단어 집합은 단어에 인덱스를 부여하는 일을 합니다. 


```python
bow 
```




    [1, 2, 1, 1, 2, 1, 1, 1, 1, 1]



단어 집합에 따른 문서1의 BoW는 두번째 출력 결과입니다. 두번째 출력 결과를 보면, 물가상승률의 index는 4이며, 문서1에서 물가상승률은 2번 언급되었기 때문에 index 4(다섯번째 값)에 해당하는 값이 2임을 알 수 있습니다. (여기서는 하지 않았으나, 원한다면 한국어에서 불용어에 해당되는 조사들 또한 제거하여 더 정제된 BoW를 만들 수도 있습니다.)

### 2) Bag of Words의 다른 예제들

앞서 언급했듯이, BoW에 있어서 중요한 것은 단어의 등장 빈도입니다. 단어의 순서. 즉, 인덱스의 순서는 전혀 상관없습니다.

- 문서2 : 소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.


```python
from konlpy.tag import Okt
import re  
okt = Okt()  

token = re.sub("(\.)","","소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.")  
# 정규 표현식을 통해 온점을 제거하는 정제 작업입니다.  
token = okt.morphs(token)  
# OKT 형태소 분석기를 통해 토큰화 작업을 수행한 뒤에, token에다가 넣습니다.  

word2index = {}  
bow = []

for voca in token:  
         if voca not in word2index.keys():  
             word2index[voca] = len(word2index)  
# token을 읽으면서, word2index에 없는 (not in) 단어는 새로 추가하고, 이미 있는 단어는 넘깁니다.   
             bow.insert(len(word2index) - 1,1)
# BoW 전체에 전부 기본값 1을 넣어줍니다. 단어의 개수는 최소 1개 이상이기 때문입니다.  
         else:
            index = word2index.get(voca)
# 재등장하는 단어의 인덱스를 받아옵니다.
            bow[index] = bow[index] + 1
# 재등장한 단어는 해당하는 인덱스의 위치에 1을 더해줍니다. (단어의 개수를 세는 것입니다.)  
print(word2index)
print(bow)
```

    {'소비자': 0, '는': 1, '주로': 2, '소비': 3, '하는': 4, '상품': 5, '을': 6, '기준': 7, '으로': 8, '물가상승률': 9, '느낀다': 10}
    [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]
    

- 문서3: 정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다. 소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.


```python
from konlpy.tag import Okt
import re  
okt = Okt()  

token = re.sub("(\.)","","정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다. 소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.")  
# 정규 표현식을 통해 온점을 제거하는 정제 작업입니다.  
token = okt.morphs(token)  
# OKT 형태소 분석기를 통해 토큰화 작업을 수행한 뒤에, token에다가 넣습니다.  

word2index = {}  
bow = []

for voca in token:  
         if voca not in word2index.keys():  
             word2index[voca] = len(word2index)  
# token을 읽으면서, word2index에 없는 (not in) 단어는 새로 추가하고, 이미 있는 단어는 넘깁니다.   
             bow.insert(len(word2index) - 1,1)
# BoW 전체에 전부 기본값 1을 넣어줍니다. 단어의 개수는 최소 1개 이상이기 때문입니다.  
         else:
            index = word2index.get(voca)
# 재등장하는 단어의 인덱스를 받아옵니다.
            bow[index] = bow[index] + 1
# 재등장한 단어는 해당하는 인덱스의 위치에 1을 더해줍니다. (단어의 개수를 세는 것입니다.)  
print(word2index)
print(bow)
```

    {'정부': 0, '가': 1, '발표': 2, '하는': 3, '물가상승률': 4, '과': 5, '소비자': 6, '느끼는': 7, '은': 8, '다르다': 9, '는': 10, '주로': 11, '소비': 12, '상품': 13, '을': 14, '기준': 15, '으로': 16, '느낀다': 17}
    [1, 2, 1, 2, 3, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]
    

### 3) CountVectorizer 클래스로 BoW 만들기

사이킷 런에서는 단어의 빈도를 Count하여 Vector로 만드는 CountVectorizer 클래스를 지원합니다. 


```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['you know I want your love. because I love you.']
vector = CountVectorizer()
print(vector.fit_transform(corpus).toarray()) # 코퍼스로부터 각 단어의 빈도 수를 기록한다.
print(vector.vocabulary_) # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.
```

    [[1 1 2 1 2 1]]
    {'you': 4, 'know': 1, 'want': 3, 'your': 5, 'love': 2, 'because': 0}
    

주의할 것은 CountVectorizer는 단지 띄어쓰기만을 기준으로 단어를 자르는 낮은 수준의 토큰화를 진행하고 BoW를 만든다는 점입니다. 이는 영어의 경우 띄어쓰기만으로 토큰화가 수행되기 때문에 문제가 없지만 한국어에 CountVectorizer를 적용하면, 조사 등의 이유로 제대로 BoW가 만들어지지 않음을 의미합니다.

'물가상승률과'와 '물가상승률은' 으로 조사를 포함해서 하나의 단어로 판단하기 때문에 서로 다른 두 단어로 인식합니다. 그리고 '물가상승률과'와 '물가상승률은'이 각자 다른 인덱스에서 1이라는 빈도의 값을 갖게 됩니다.

### 4) 불용어를 제거한 BoW 만들기

 BoW를 만들때 불용어를 제거하는 일은 자연어 처리의 정확도를 높이기 위해서 선택할 수 있는 전처리 기법입니다.

영어의 BoW를 만들기 위해 사용하는 CountVectorizer는 불용어를 지정하면, 불용어는 제외하고 BoW를 만들 수 있도록 불용어 제거 기능을 지원하고 있습니다.

#### (1) 사용자가 직접 정의한 불용어 사용


```python
from sklearn.feature_extraction.text import CountVectorizer
text=["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words=["the", "a", "an", "is", "not"])
print(vect.fit_transform(text).toarray()) 
print(vect.vocabulary_)
```

    [[1 1 1 1 1]]
    {'family': 1, 'important': 2, 'thing': 4, 'it': 3, 'everything': 0}
    

#### (2) CounterVectorizer에서 제공하는 자체 불용어 사용


```python
from sklearn.feature_extraction.text import CountVectorizer
text=["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words="english")
print(vect.fit_transform(text).toarray())
print(vect.vocabulary_)
```

    [[1 1 1]]
    {'family': 0, 'important': 1, 'thing': 2}
    

#### (3) NLTK에서 지원하는 불용어 사용


```python
from sklearn.feature_extraction.text import CountVectorizer
text=["Family is not an important thing. It's everything."]
from nltk.corpus import stopwords
sw = stopwords.words("english")
vect = CountVectorizer(stop_words =sw)
print(vect.fit_transform(text).toarray()) 
print(vect.vocabulary_)
```

    [[1 1 1 1]]
    {'family': 1, 'important': 2, 'thing': 3, 'everything': 0}
    

## 3. 문서 단어 행렬(Document-Term Matrix, DTM)

문서 단어 행렬(Document-Term Matrix, DTM) 표현 방법이란 각 문서에 대한 BoW 표현 방법을 그대로 갖고와서, 서로 다른 문서들의 BoW들을 결합한 표현 방법이다.

### 1) 문서 단어 행렬(Document-Term Matrix, DTM)의 표기법

문서 단어 행렬(Document-Term Matrix, DTM)이란 다수의 문서에서 등장하는 각 단어들의 빈도를 행렬로 표현한 것을 말합니다.

예를 들어서 이렇게 4개의 문서가 있다고 합시다.
- 문서1 : 먹고 싶은 사과
- 문서2 : 먹고 싶은 바나나
- 문서3 : 길고 노란 바나나 바나나
- 문서4 : 저는 과일이 좋아요

이를 문서 단어 행렬로 표현하면 다음과 같습니다.

![](https://lh3.googleusercontent.com/JGlXpGtESjJwWmcRLchla2VsV2bnIMPuiTC0nwFjHWl__DcwBtDak_FRyxofOIZcdvDIQy_D6a_N6SyTeA7QBVbR7U7FbF3mOqEZmVz-c9oRUlsKvg365uZPyy0Qy4gfKswsA38TdN17xdMCsrzdWclN99O2CP7vcSsA1_LUdjXCMy7fDd_jrrMbEMNyKElXMa7yj_3jt1_-2MeIjFPHUKO9CLqjtO6mjGqph4crV9p8ZT0ge_A-qCYezCCPAbCdWSBfSgCjg4X_H0yoxF2iZuB_KbsHtStCst7qQICQFZpIXf6Sa_eGOAXVFgua8SqQMmHV3JttWrWGWwB8T8Ncrv93umRCwviJ3HN5Mrbae0m9vGKFACiEMJpduB0ISibbWcAETQt9w9lP1y9pWrRHZokwH-Q_5K0ukei84xwxibnBVspHvtTKyEvEoDzeEStmhDspH25m0MNdUo4MEX66-XBj97MGSFYX3meD_4xBtf6LE4K0dHNPFQqT_-3HD5oYj_uT82BDAw1ZNjikKr-trcOzsCrdw57q5n4L7GxGYufv23m4dX67QaEcrZuppOYyLkGmGk5wr9ncyUhxgzXV207WEnKbrmeuV67qtuj3fuoUU0bPHlhyGDAMlqgVGZoK8m6BXOvOh7Soy4P-73BRMkupOo97Dqc=w948-h282-no)

각 문서에서 등장한 단어의 빈도를 행렬의 값으로 표기합니다. 문서 단어 행렬은 문서들을 서로 비교할 수 있도록 수치화할 수 있다는 점에서 의의를 갖습니다. (여기서는 하지 않았으나, 원한다면 한국어에서 불용어에 해당되는 조사들 또한 제거하여 더 정제된 DTM을 만들 수도 있을 것입니다.)

### 2) 문서 단어 행렬(Document-Term Matrix)의 한계

#### (1) 희소 표현(Sparse representation)

만약 가지고 있는 전체 코퍼스가 방대한 데이터라면 문서 벡터의 차원은 수백만의 차원을 가질 수도 있습니다. 또한 많은 문서 벡터가 대부분의 값이 0을 가질 수도 있습니다.

원-핫 벡터나 DTM과 같은 대부분의 값이 0인 표현을 희소 벡터(sparse vector) 또는 희소 행렬(sparse matrix)라고 부르는데, 희소 벡터는 방대한 양의 메모리와 계산을 위한 리소스를 필요로 합니다. 이러한 이유로 전처리를 통해 단어 집합의 크기를 줄이는 일은 BoW 표현을 사용하는 모델에서 중요합니다.

#### (2) 단순 빈도 수 기반 접근

영어에 대해서 DTM을 만들었을 때, 불용어인 the는 어떤 문서이든 자주 등장할 수 밖에 없습니다. 그런데 유사한 문서인지 비교하고 싶은 문서1, 문서2, 문서3에서 동일하게 the가 빈도수가 높다고 해서 이 문서들이 유사한 문서라고 판단해서는 안 됩니다.

## 4. TF-IDF(Term Frequency-Inverse Document Frequency)

TF-IDF이란 DTM에 불용어와 중요한 단어에 대해서 가중치를 줄 수 있다.

DTM 내에 있는 각 단어에 대한 중요도를 계산할 수 있다.

### 1) TF-IDF(단어 빈도-역 문서 빈도, Term Frequency-Inverse Document Frequency)

TF-IDF는 Term Frequency-Inverse Document Frequency의 줄임말로, 단어의 빈도와 역 문서 빈도(문서의 빈도에 특정 식을 취함)를 사용하여 DTM 내의 각 단어들마다 중요한 정도를 가중치로 주는 방법입니다. 사용 방법은 우선 DTM을 만든 후에, 거기에 TF-IDF 가중치를 주면됩니다.

문서를 d, 단어를 t, 문서의 총 개수를 n이라고 표현할 때 TF, DF, IDF는 각각 다음과 같이 정의할 수 있습니다.

#### (1) tf(d,t) : 특정 문서 d에서의 특정 단어 t의 등장 횟수.

TF는 앞에서 배운 DTM의 예제에서 각 단어들이 가진 값들입니다. DTM이 각 문서에서의 각 단어의 등장 빈도를 나타내는 값

#### (2) df(t) : 특정 단어 t가 등장한 문서의 수.

오직 특정 단어 t가 등장한 문서의 수에만 관심을 가집니다.

앞서 배운 DTM에서 바나나는 문서2와 문서3에서 등장했습니다. 즉, 바나나의 df는 2입니다. 심지어 바나나란 단어가 문서2에서 100번 등장했고, 문서3에서 200번 등장했다고 하더라도 바나나의 df는 2가 됩니다.

#### (3) idf(d, t) : df(t)에 반비례하는 수.

$idf(d, t) = log(\frac{n}{1+df(t)})$

TF-IDF는 모든 문서에서 자주 등장하는 단어는 중요도가 낮다고 판단하며, 특정 문서에서만 자주 등장하는 단어는 중요도가 높다고 판단합니다.

TF-IDF 값이 낮으면 중요도가 낮은 것이며, TF-IDF 값이 크면 중요도가 큰 것입니다.

![](https://lh3.googleusercontent.com/mO89Y3RWWEYDPjbr2moBG0BJwfmpbwdRZ_wi793eusiqa3OE9oSeLumvmRBDL8gdtqHOdgVDZdL0r2lD3PpcIPkBNMMbiT9C-e10Fn5ywKy_NxVVSUKv9U_TUGDmrgEINwz7MgyQKEyy8UiVrdE303O7CAkLWjeGw7xnbfKRNu_4BHDo5ddxsChr41pDiLA0IrQzCLXqXvzZkH_HIUwe27DWTB0mbxZR4z2LPrVpTQgHkwtRqRS9WsrUvpV9GUbAxTgmqNlReJrKMk-uGkhn_AbfYtkLh5-_hGaYq9VBEfBRhNZSs4dLK7r_nwJsOdtJzFGWAyNOSPgbEalfozEf2_cv49gqC8vaIUS1_IEoz8yaFhGVDKqzoLBcz6deoL8006GgYBF59Ub7Wt3IWypRCSo68rS5VxUx_htGADwJfpHQ5ye_IyYIG2dPAOqxVY3FBtBqkMqnRE7hj6gQ-M6o8Xk8DkvuCpBbjyBBaLtw6i0rZxPiYAvvuBRz9M7oMeRMiWHWVxpgj1n-Dcek-zqwBo_X8RMvmIFV5oO9djTrM1KMRKa6KS7blwNIlQ6vXrLBsVN0UCtOda-3mnkvY9zBe-ZIYlUN0YBf3FS60hBjNoNVdo94yVWUOdYNCzORF3anVVEB3I2CcL5g7F6jrXPnR3WIAN2WKIo=w948-h282-no)

위는 DTM의 예제 결과이고 다시 말하면 각 단어의 TF가 된다.

그렇다면 이제 구해야할 것은 TF와 곱해야할 값인 IDF입니다.

로그는 자연 로그를 사용하도록 하겠습니다. 자연 로그는 로그의 밑을 자연 상수 e(e=2.718281...)를 사용하는 로그를 말합니다. 자연 로그는 보통 log라고 표현하지 않고, ln이라고 표현합니다.

![](https://lh3.googleusercontent.com/NvSpxeq5EUyiXWo3NeMdj-242LHEbamus8Wql0ddyH1kk8SPtADqKYkz6Xtx8xb_qK7CpWutJz0G9oINpAFHXeQ0EO1brSR0v55hDeHWfSzas6TcF3uJ6hUuim_Rqw5zA69pZbfN4bItqXKi8qZs36GctVxBYjc_F-GSA8ychumi2F0hC-rg1GuNhBvxb43DTJOigf17cN0eGnWyE7JXgXcJXUcp0USN81LXGG21tp142VhrTDkaSWu2j00Armz9xs4lVFJGlcM6WjwJQkgoq-49UJ0vUmp3ckpV8RM0778K8jVYP5WCcIbFw6G5GqpfQ5Q0pGNwAfCPz36fxS42Ux9mTJwdBOgw9tYULu-cg6VZL777H_wlIp_BbMvh0IEr1w-V35XrZ9VYbLugh57bUGcZDpFDlpwiV1ESuIXVBTKalDLNXvnyXxt6UlneEXfoBr5lflFTXv3QhlMm7-rVKOC9MxM23gnZPJNSZ52PeClrpMSAYZ3b_qJJ229Ig_-MSnjZWViZ87-9Acu-suww2Gk42rZoHn2Fc97pTrrnbnb13DBwmuiUfijU40kZreHNrOGfWE67BjCp-9Wc7JxnmP12o0g6Zq6rXkS1hUYubj6UrKgCFNvBdDPfwMlTBEK5varlDILUTcM05iyIjmFNkXQR7NZu-N8=w386-h546-no)

문서의 총 수는 4이기 때문에 ln 안에서 분자는 늘 4으로 동일합니다.

분모의 경우에는 각 단어가 등장한 문서의 수(DF)를 의미하는데, 예를 들어서 '먹고'의 경우에는 총 2개의 문서(문서1, 문서2)에 등장했기 때문에 2라는 값을 가집니다.

앞서 사용한 DTM에서 단어 별로 위의 IDF값을 그대로 곱해주면 TF-IDF가 나오게 됩니다.

![](https://lh3.googleusercontent.com/uldZQxrfh6sn5abJ5GuIctIvJBCEhej0XRoXS-7M4jLver70wAJNe_u1js0NqU_VpY_4hbGcISa3iVV5otL_jq3TG2Xj73ZQ-ff57ViDfwMXx9ejk_KTdGhbZyIigPADX9Xwhe4Lsi5gQ121LmO__tK6HwI-XDzCBTYnJoqveHEVMsyLP_ZUEZZes0lORJaQNcH7ADETouYgrf7H5STmPNiXTw59rLRoTZVfgYyhGnJxf6P1y-p15FZYDYt9oH60rJMF_fSLZ9Qucel9CUXhE6T3A8KG_2G5W57hJ7LngIJXfscYN2LT4pecdBmzXCws0_dJH4eh9V-yPCm0p0EGhiiqw9dk-k0H46OSjxF39vWUisRtnFH565dkowfidQLP61mNdHRHWxsxKXw4bdVb6BgeAapOroq9moPl8DfUAsGTHcjVGOKR1uvDBXUDEZkDAbQgESEmnb3LlpepmDUdHyfz4awRMndIMwji99C-wXiCOrSK9PwFYeGlSDt2AzkBMRQKQTIBgCqPi73hjROdkcQOqDFYvW_L2YslV9GfKfTKtHo6p1kTwG9POiWfMTEQT8ds-2O4pwrTuwU4tqxDeKRQwUU6OUEuM6Vp2w_Zs8-0VBab7073u7oq1Cp-OHCzxf-GefrGGrkl60rhnx26oTrqd_6DxnY=w960-h209-no)

문서3에서의 바나나만 TF 값이 2이므로 IDF에 2를 곱해주고, 나머진 TF 값이 1이므로 그대로 IDF 값을 가져오면 됩니다.

### 2) 사이킷 런을 이용한 DTM과 TF-IDF 실습

DTM 또한 BoW 행렬이기 때문에, 앞서 BoW 챕터에서 배운 CountVectorizer를 사용하면 간단히 DTM을 만들 수 있습니다.


```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',    
]
vector = CountVectorizer()
print(vector.fit_transform(corpus).toarray()) # 코퍼스로부터 각 단어의 빈도 수를 기록한다.
print(vector.vocabulary_) # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.
```

    [[0 1 0 1 0 1 0 1 1]
     [0 0 1 0 0 0 0 1 0]
     [1 0 0 0 1 0 1 0 0]]
    {'you': 7, 'know': 1, 'want': 5, 'your': 8, 'love': 3, 'like': 2, 'what': 6, 'should': 4, 'do': 0}
    

사이킷런은 TF-IDF를 자동 계산해주는 TfidVectorizer 클래스를 제공합니다.


```python
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',    
]
tfidfv = TfidfVectorizer().fit(corpus)
print(tfidfv.transform(corpus).toarray())
print(tfidfv.vocabulary_)
```

    [[0.         0.46735098 0.         0.46735098 0.         0.46735098
      0.         0.35543247 0.46735098]
     [0.         0.         0.79596054 0.         0.         0.
      0.         0.60534851 0.        ]
     [0.57735027 0.         0.         0.         0.57735027 0.
      0.57735027 0.         0.        ]]
    {'you': 7, 'know': 1, 'want': 5, 'your': 8, 'love': 3, 'like': 2, 'what': 6, 'should': 4, 'do': 0}
    

이제 BoW, DTM, TF-IDF 가중치에 대해서 전부 학습했습니다. 그러면 문서들 간의 유사도를 구하기 위한 재료 손질하는 방법을 배운 것입니다.
