
written by ideajoon<br/>
※ 참고 : 딥 러닝을 이용한 자연어 처리 입문 (https://wikidocs.net/book/2155) 자료를 공부하고 정리함

# 05. 문서 유사도(Document Similarity)

사람들이 인식하는 문서의 유사도란 주로 문서들 간에 동일한 단어 또는 비슷한 단어가 얼마나 공통적으로 많이 사용되었는지에 달려있습니다. 기계도 마찬가지입니다.

## 목차
1. 코사인 유사도(Cosine Similarity)
2. 여러가지 유사도 기법

## 1. 코사인 유사도(Cosine Similarity)

BoW나 BoW에 기반한 단어 표현 방법인 DTM, TF-IDF, 또는 뒤에서 배우게 될 워드투벡터(Word2Vec) 등과 같이 단어를 수치화할 수 있는 방법을 이해했다면, 이러한 표현 방법에 대해서 코사인 유사도를 이용하여 문서의 유사도를 구하는 게 가능합니다.

### 1) 코사인 유사도(Cosine Similarity)

코사인 유사도는 두 벡터 간의 코사인 각도를 이용하여 구할 수 있는 두 벡터의 유사도를 의미합니다.

![](https://wikidocs.net/images/page/24603/%EC%BD%94%EC%82%AC%EC%9D%B8%EC%9C%A0%EC%82%AC%EB%8F%84.PNG)

즉, 결국 코사인 유사도는 -1 이상 1 이하의 값을 가지며 값이 1에 가까울수록 유사도가 높다고 판단할 수 있습니다.

두 벡터 A, B에 대해서 코사인 유사도는 식으로 표현하면 다음과 같습니다.

$similarity=cos(Θ)=\frac{A⋅B}{||A||\ ||B||}=\frac{\sum_{i=1}^{n}{A_{i}×B_{i}}}{\sqrt{\sum_{i=1}^{n}(A_{i})^2}×\sqrt{\sum_{i=1}^{n}(B_{i})^2}}$

문서의 유사도를 구하는 경우에는 문서 단어 행렬이나 TF-IDF 행렬이 각각의 특징 벡터 A, B가 됩니다.

- 문서1 : 저는 사과 좋아요
- 문서2 : 저는 바나나 좋아요
- 문서3 : 저는 바나나 좋아요 저는 바나나 좋아요

![](https://lh3.googleusercontent.com/Ai35nWImbzFCGYliggpFJRjJv6J7XY4pmXypazjuuI-RFD_TdgwiED0oNpiz9lCaYwXMWUQ5rWBFYntGMSfSeUjua82fTNFarIAZ2gj3ifGuBri98TRmdzrXrnCRYU7OL8-R5bMD79Ak8N5COBhBR0VHWkhx8IRH7Pe64nG-E5sEQf_VxWj_mdXHNaXgeZ9jwdcKsHsG--tofB-ES7nm-Dpz8PfhjNYppJlhsFhvbd69ZEIxAASM6SLptb7o6zA6-sEJFmJWpoj_nA_cU7SjuTv2vzzaTWzKEjdeKYAASxi4DELMeWHmgnDDXbg_thjRlg_hZlZqHHPJUKkpM3ApNr_fbRKofLIEtG9WvTUAxiYRApZtrNP_MUCuWSJQOOf4iNqjyWlfpe4mfp1NcDSCDGpeeFOK2BVh9scBqQPgwlzji4NV9pxtzsOzN6ik0vvSCJfS4q-uYoNpJuPw0RFOHV1R-73NGmKyWuf3ivAgajjHAqnxq4AMFxPFAK6_8PDQTJW7j2ZOnuRufqmv5Tf2DdAdyC6b0JKNNtyOsMPup-E7kdklFZ_fOHSQkJiGvGXC3yfHOOB7UHvIWGjWbXL6IgR-lxA-KTvRuv2cttcfD8SOX0AGSj1Lc6dtSx2S1DWDU2lOD-0IHtkFmYouBzvjFz2Jp7JJ2WM=w489-h222-no)

파이썬에서는 코사인 유사도를 구하는 방법은 여러가지가 있는데 여기서는 Numpy를 이용해서 계산해보겠습니다.


```python
from numpy import dot
from numpy.linalg import norm
import numpy as np
def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))
```


```python
doc1=np.array([0,1,1,1])
doc2=np.array([1,0,1,1])
doc3=np.array([2,0,2,2])
```


```python
print(cos_sim(doc1, doc2)) #문서1과 문서2의 코사인 유사도
print(cos_sim(doc1, doc3)) #문서1과 문서3의 코사인 유사도
print(cos_sim(doc2, doc3)) #문서2과 문서3의 코사인 유사도
```

    0.6666666666666667
    0.6666666666666667
    1.0000000000000002
    

한 문서 내의 모든 단어의 빈도수가 똑같이 증가하는 경우에는 기존의 문서와 코사인 유사도의 값이 1이라는 것입니다.

### 2) 유사도를 이용한 추천 시스템 구현하기

캐글에서 사용되었던 영화 데이터셋을 가지고 영화 추천 시스템을 만들어보겠습니다. TF-IDF와 코사인 유사도만으로 영화의 줄거리에 기반해서 영화를 추천하는 추천 시스템을 만들 수 있습니다.

* dataset = movies_metadata.csv 파일
- 총 24개의 열을 가진 45,466개의 샘플로 구성된 영화 정보 데이터


```python
import pandas as pd
data = pd.read_csv(r'movies_metadata.csv', low_memory=False)
data.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>adult</th>
      <th>belongs_to_collection</th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>imdb_id</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>...</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>video</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>{'id': 10194, 'name': 'Toy Story Collection', ...</td>
      <td>30000000</td>
      <td>[{'id': 16, 'name': 'Animation'}, {'id': 35, '...</td>
      <td>http://toystory.disney.com/toy-story</td>
      <td>862</td>
      <td>tt0114709</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>Led by Woody, Andy's toys live happily in his ...</td>
      <td>...</td>
      <td>1995-10-30</td>
      <td>373554033.0</td>
      <td>81.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Toy Story</td>
      <td>False</td>
      <td>7.7</td>
      <td>5415.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>NaN</td>
      <td>65000000</td>
      <td>[{'id': 12, 'name': 'Adventure'}, {'id': 14, '...</td>
      <td>NaN</td>
      <td>8844</td>
      <td>tt0113497</td>
      <td>en</td>
      <td>Jumanji</td>
      <td>When siblings Judy and Peter discover an encha...</td>
      <td>...</td>
      <td>1995-12-15</td>
      <td>262797249.0</td>
      <td>104.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}, {'iso...</td>
      <td>Released</td>
      <td>Roll the dice and unleash the excitement!</td>
      <td>Jumanji</td>
      <td>False</td>
      <td>6.9</td>
      <td>2413.0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 24 columns</p>
</div>



tf-idf의 대상이 되는 data의 overview 열에 Null 값이 있는지 확인합니다.


```python
data['overview'].isnull().sum()
```




    954



 pandas를 이용하면 Null 값을 처리하는 도구인 fillna()를 사용할 수 있습니다. 괄호 안에 Null 대신 넣고자하는 값을 넣으면 되는데, 이 경우에는 빈 값(empty value)으로 대체하여 Null 값을 제거합니다.


```python
data['overview'] = data['overview'].fillna('')
# overview에서 Null 값을 가진 경우에는 값 제거
```


```python
data['overview'].isnull().sum()
```




    0




```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['overview'])
# overview에 대해서 tf-idf 수행
print(tfidf_matrix.shape)
```

    (45466, 75827)
    

45,466개의 영화를 표현하기위해 총 75,827개의 단어가 사용되었음을 보여주고 있습니다. 이제 코사인 유사도를 사용하면 바로 문서의 유사도를 구할 수 있습니다.


```python
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
```

코사인 유사도를 구합니다.


```python
indices = pd.Series(data.index, index=data['title']).drop_duplicates()
print(indices.head())
```

    title
    Toy Story                      0
    Jumanji                        1
    Grumpier Old Men               2
    Waiting to Exhale              3
    Father of the Bride Part II    4
    dtype: int64
    


```python
idx = indices['Father of the Bride Part II']
print(idx)
```

    4
    

이제 선택한 영화에 대해서 코사인 유사도를 이용하여, 가장 overview가 유사한 10개의 영화를 찾아내는 함수를 만듭니다.


```python
def get_recommendations(title, cosine_sim=cosine_sim):
    # 선택한 영화의 타이틀로부터 해당되는 인덱스를 받아옵니다. 이제 선택한 영화를 가지고 연산할 수 있습니다.
    idx = indices[title]

    # 모든 영화에 대해서 해당 영화와의 유사도를 구합니다.
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 유사도에 따라 영화들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 10개의 영화를 받아옵니다.
    sim_scores = sim_scores[1:11]

    # 가장 유사한 10개의 영화의 인덱스를 받아옵니다.
    movie_indices = [i[0] for i in sim_scores]

    # 가장 유사한 10개의 영화의 제목을 리턴합니다.
    return data['title'].iloc[movie_indices]
```

영화 다크 나이트 라이즈와 overview가 유사한 영화들을 찾아보겠습니다.


```python
get_recommendations('The Dark Knight Rises')
```




    12481                                      The Dark Knight
    150                                         Batman Forever
    1328                                        Batman Returns
    15511                           Batman: Under the Red Hood
    585                                                 Batman
    21194    Batman Unmasked: The Psychology of the Dark Kn...
    9230                    Batman Beyond: Return of the Joker
    18035                                     Batman: Year One
    19792              Batman: The Dark Knight Returns, Part 1
    3095                          Batman: Mask of the Phantasm
    Name: title, dtype: object



가장 유사한 영화가 출력되는데, 영화 다크 나이트가 첫번째고, 그 외에도 전부 배트맨 영화를 찾아낸 것을 확인할 수 있습니다.

## 2. 여러가지 유사도 기법

### 1) 유클리드 거리(Euclidean distance)

유클리드 거리(euclidean distance)는 문서의 유사도를 구할 때 자카드 유사도나 코사인 유사도만큼, 유용한 방법은 아닙니다

다차원 공간에서 두개의 점 p와 q가 각각 $p=(p_{1}, p_{2}, p_{3}, ... , p_{n})$과 $q=(q_{1}, q_{2}, q_{3}, ..., q_{n})$ 의 좌표를 가질 때 두 점 사이의 거리를 계산하는 유클리드 거리 공식은 다음과 같습니다.

$\sqrt{(q_{1}-p_{1})^{2}+(q_{2}-p_{2})^{2}+\ ...\ +(q_{n}-p_{n})^{2}}=\sqrt{\sum_{i=1}^{n}(q_{i}-p_{i})^{2}}$

![](https://wikidocs.net/images/page/24654/2%EC%B0%A8%EC%9B%90_%ED%8F%89%EB%A9%B4.png)

예를 들어 아래와 같은 DTM이 있다고 합시다.

![](https://lh3.googleusercontent.com/DPHs2ZqWcrqSuTB9e_2YaAVDR9uWOtF7LooPaOof5xq6-ZuaGAAuoOeG4XFAa3WLjfQKPbuzvkOHv0aRrpepfZAkl4gCuwu1VDjbtZDlRgLVK3nhvlwFa0RvKGqG6EdiT9EC_jmg8Q8ORtjoVDUtCeM2S9sLidL0U6CcACmoWkiPbZXez6Au0qJ84WGhKwHGzUHfvzifg__EAWQG6B9Rxub2jitVAAF6zOInxQLYa7ocb35jNHCBHgOzmpgF6iGi-L70380d3a-yEIjGFx52XMkm3PoQ0hVxAcTottSrohf1e82zgOBTKZqsGnV_rjruAxGEo0iO4p3imlGzwa4MFYTHIART-A4kOOPeVUdprfMS7rdvXeBOolqfoPFAJB-MvEVunXDgq8M1fy_YMFH7L5HnaxS6kcdrNe0U0Csc4jWygTkYzmuhM27ieLM2GYxi9_xtST0lsvL6fTR5l1UOidX-QiWbdNiRnlAq2MDGEcqmBnjds_EvMFkKFYGNqbTUsdZM_8sZi1pjNwDCSNHoz41ktBqFUEYwnl93wNBLRdQXlQ5VJw3M3TVK4vKzJVkvv_hz-qpWrU-VN2ro5LwQgVhP9qIoQ8f5ljAvT8b_RdN3cCJmh8KiavJduzCFtGumC-pHwu5QwuXJ0nNSBLEy-qFfbx2hGiw=w489-h226-no)

단어의 개수가 4개이므로, 이는 4차원 공간에 문서1, 문서2, 문서3을 배치하는 것과 같습니다. 이때 다음과 같은 문서Q에 대해서 문서1, 문서2, 문서3 중 가장 유사한 문서를 찾아내고자 합니다.

![](https://lh3.googleusercontent.com/Xd87-ddIopd1eRZoGehG27z-hXqII64xbU81FHN9fKVYgtFdJjCEVaApgDf7EcGYnozr0X6f91HizcbPbFf3KxWGHtkYCX5aU5FU0h1DYENJ5LLOsS5ECX-f5lHfTVU0LPIwtbyt0P6ghKNoN2VDl4dzn80P8_yW311XtwGOtYXHpWrRJhVk8zM-Dh0qBWwD50Q53I_QwlsmYNz-9h5EUXD6IcRDXYDglmQP8DpQ4hryF1k-3EPwm3wO7zm2ZgKoy7Ym03BWzgJORO46ZxmJJwONJwBuBD8LSGJme_U6xdv5QOrVtyBfJAWvnD53bHI2EhOnc_LPX3SpqqHrr4IpkCWDdAj0atktD8Yo1LRz-y6DExrfKPkm1g1YKJIkg2m5IroBsGm0xC5vNOYUrJ8C-BxipXC6WAqywzp6ZDbZb4iFk9i7zejf6WT8WEA5AfCmEewtjaCc5WIL-7mqfVLRb1FyXzS9MDHqH96faiU85hX6NCqCs-Vo-9fAakWXqyNwDoWZuuB1JKJ0u2xgjZ9K1FpJxkXXkdtXP2xXhNswxHkRB7xnfDo2klHzlKm5l7qx7R0oYzoHT-Vqi5UNUQsVIN6p-zp6sd0lXGq5C0oGpup341oq76xARHlXvaCYyR79FMKH5Pfiw2eSrNG5w-vRpy8dWgmr-Dc=w489-h117-no)

이때 유클리드 거리를 통해 유사도를 구하려고 한다면, 문서Q 또한 다른 문서들처럼 4차원 공간에 배치시켰다는 관점에서 4차원 공간에서의 각각의 문서들과의 유클리드 거리를 구하면 됩니다. 


```python
import numpy as np
def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))

doc1 = np.array((2,3,0,1))
doc2 = np.array((1,2,3,1))
doc3 = np.array((2,1,2,2))
docQ = np.array((1,1,0,1))

print(dist(doc1,docQ))
print(dist(doc2,docQ))
print(dist(doc3,docQ))
```

    2.23606797749979
    3.1622776601683795
    2.449489742783178
    

유클리드 거리의 값이 가장 작다는 것은, 문서 간의 거리가 가장 가깝다는 것을 의미합니다. 즉, 문서1이 문서Q와 가장 유사하다고 볼 수 있습니다.

### 2) 자카드 유사도(Jaccard similarity)

합집합에서 교집합의 비율을 구한다면 두 집합 A와 B의 유사도를 구할 수 있다는 것이 자카드 유사도(jaccard similarity)의 아이디어입니다.

자카드 유사도는 0과 1사이의 값을 가지며, 만약 두 집합이 동일하다면 1의 값을 가지고, 두 집합의 공통 원소가 없다면 0의 값을 갖습니다. 자카드 유사도를 구하는 함수를 J라고 하였을 때, 자카드 유사도 함수 J는 아래와 같습니다.

$J(A,B)=\frac{|A∩B|}{|A∪B|}=\frac{|A∩B|}{|A|+|B|-|A∩B|}$

두 개의 비교할 문서를 각각 doc1, doc2라고 했을 때 doc1과 doc2의 문서의 유사도를 구하기 위한 자카드 유사도는 이와 같습니다.

$J(doc_{1},doc_{2})=\frac{doc_{1}∩doc_{2}}{doc_{1}∪doc_{2}}$


```python
# 다음과 같은 두 개의 문서가 있습니다.
# 두 문서 모두에서 등장한 단어는 apple과 banana 2개.
doc1 = "apple banana everyone like likey watch card holder"
doc2 = "apple banana coupon passport love you"

# 토큰화를 수행합니다.
tokenized_doc1 = doc1.split()
tokenized_doc2 = doc2.split()

# 토큰화 결과 출력
print(tokenized_doc1)
print(tokenized_doc2)
```

    ['apple', 'banana', 'everyone', 'like', 'likey', 'watch', 'card', 'holder']
    ['apple', 'banana', 'coupon', 'passport', 'love', 'you']
    

이 때 문서1과 문서2의 합집합을 구해보겠습니다.


```python
union = set(tokenized_doc1).union(set(tokenized_doc2))
print(union)
```

    {'passport', 'love', 'everyone', 'likey', 'watch', 'banana', 'like', 'you', 'apple', 'card', 'coupon', 'holder'}
    

문서1과 문서2의 교집합을 구해보겠습니다.


```python
intersection = set(tokenized_doc1).intersection(set(tokenized_doc2))
print(intersection)
```

    {'banana', 'apple'}
    

이제 교집합의 수를 합집합의 수로 나누면 자카드 유사도가 계산됩니다.


```python
print(len(intersection)/len(union)) # 2를 12로 나눔.
```

    0.16666666666666666
    

위의 값은 자카드 유사도이자, 두 문서의 총 단어 집합에서 두 문서에서 공통적으로 등장한 단어의 비율이기도 합니다.
