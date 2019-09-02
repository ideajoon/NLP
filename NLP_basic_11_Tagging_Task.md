
written by ideajoon<br/>
※ 참고 : 딥 러닝을 이용한 자연어 처리 입문 (https://wikidocs.net/book/2155) 자료를 공부하고 정리함

# 11. 태깅 작업(Tagging Task)

각 단어가 어떤 유형에 속해있는지를 알아내는 태깅 작업

- 개체명 인식(Named Entity Recognition)<br/>
각 단어의 유형이 사람, 장소, 단체 등 어떤 유형인지를 알아내는 작업
- 품사 태깅(Part-of-Speech Tagging)<br/>
 각 단어의 품사가 명사, 동사, 형용사 인지를 알아내는 작업

## 목차
1. 케라스를 이용한 태깅 작업 개요(Tagging Task using Keras)
2. 개체명 인식(Named Entity Recognition)
3. 양방향 LSTM을 이용한 개체명 인식(Named Entity Recognition using Bi-LSTM) 
4. 양방향 LSTM을 이용한 품사 태깅(Part-of-speech Tagging using Bi-LSTM)
5. 양방향 LSTM과 CRF(Bidirectional LSTM + CRF)

## 1. 케라스를 이용한 태깅 작업 개요(Tagging Task using Keras)

케라스(Keras)로 인공 신경망을 이용하여 태깅 작업을 하는 모델을 만듭니다. 즉, 개체명 인식기와 품사 태거를 만든다. 이러한 두 작업의 공통점은 RNN의 다-대-다(Many-to-Many) 작업이면서 또한 앞, 뒤 시점의 입력을 모두 참고하는 양방향 RNN(Bidirectional RNN)을 사용한다는 점입니다.

### 1) 훈련 데이터에 대한 이해

태깅 작업은 앞서 배운 텍스트 분류 작업과 동일하게 지도 학습(Supervised Learning)에 속합니다.

|구분|X_train|y_train|길이|
|---|---|---|---|
|0|['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb']|['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O']|8|
|1|['peter', 'blackburn']|['B-PER', 'I-PER']|2|
|2|['brussels', '1996-08-22' ]|['B-LOC', 'O']|2|
|3|['The', 'European', 'Commission']|['O', 'B-ORG', 'I-ORG']|3|

이렇게 병렬 관계를 가지는 각 데이터는 정수 인코딩 과정을 거친 후, 모든 데이터의 길이를 동일하게 맞춰주기위한 패딩(Padding) 작업을 거칩니다.

### 2) 시퀀스 레이블링(Sequence Labeling)

위와 같이 입력 시퀀스 X = [x1, x2, x3, ..., xn]에 대하여 레이블 시퀀스 y = [y1, y2, y3, ..., yn]를 각각 부여하는 작업을 시퀀스 레이블링 작업(Sequence Labeling Task)이라고 합니다. 

### 3) 양방향 LSTM(Bidirectional LSTM)

```
model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
```

양방향 LSTM을 사용합니다. 양방향은 기존의 단방향 LSTM()을 Bidirectional() 안에 넣으면 됩니다.

인자 정보는 단방향 LSTM을 사용할 때와 동일합니다. 즉, 인자값을 하나를 줄 경우에는 이는 hidden_size를 의미하며, 결과적으로 output_dim이 됩니다.

### 4) RNN의 다-대-다(Many-to-Many) 문제

![](https://wikidocs.net/images/page/24873/many_to_one.PNG)

- return_sequences=True<br/>
RNN의 다-대-다(Many-to-Many)로 설정한다.

- return_sequences=False<br/>
RNN의 다-대-일(Many-to-One)로 설정한다.

 예를 들어 위에서 설명한 데이터 중 첫번째 데이터에 해당되는 X_train[0]를 가지고 4번의 시점(timesteps)까지 RNN을 진행하였을 때의 그림은 다음과 같습니다.

![](https://wikidocs.net/images/page/33805/forwardrnn_ver2.PNG)

하지만 이번 실습에서는 양방향 RNN을 사용할 것이므로 아래의 그림과 같습니다.

![](https://wikidocs.net/images/page/33805/bidirectionalrnn_ver2.PNG)

## 2. 개체명 인식(Named Entity Recognition)

### 1) 개체명 인식(Named Entity Recognition)이란?

- 도연 - 사람  
- 2018년 - 시간  
- 골드만삭스 - 조직

개체명 모델이 토큰화와 품사 태깅(POS Tagging, Part-Of-Speech Tagging) 전처리를 끝내고 난 상태를 입력으로 요구하기도 합니다.

### 2) NLTK를 이용한 개체명 인식(Named Entity Recognition using NTLK)

NLTK에서는 개체명 인식기(NER chunker)를 지원한다.


```python
from nltk import word_tokenize, pos_tag, ne_chunk
sentence = "James is working at Disney in London"
sentence=pos_tag(word_tokenize(sentence))
print(sentence) # 토큰화와 품사 태깅을 동시 수행
```

    [('James', 'NNP'), ('is', 'VBZ'), ('working', 'VBG'), ('at', 'IN'), ('Disney', 'NNP'), ('in', 'IN'), ('London', 'NNP')]
    


```python
sentence=ne_chunk(sentence)
print(sentence) # 개체명 인식
```

    (S
      (PERSON James/NNP)
      is/VBZ
      working/VBG
      at/IN
      (ORGANIZATION Disney/NNP)
      in/IN
      (GPE London/NNP))
    

ne_chunk는 개체명을 태깅하기 위해서 앞서 품사 태깅(pos_tag)이 수행되어야 합니다. 위의 결과에서 James는 PERSON(사람), Disney는 조직(ORGANIZATION), London은 위치(GPE)라고 정상적으로 개체명 인식이 수행된 것을 볼 수 있습니다.

## 3. 양방향 LSTM을 이용한 개체명 인식(Named Entity Recognition using Bi-LSTM)

### 1) BIO 표현

B는 Begin의 약자로 개체명이 시작되는 부분, I는 Inside의 약자로 개체명의 내부 부분을 의미하며, O는 Outside의 약자로 개체명이 아닌 부분을 의미합니다.

예를 들어서 영화에 대한 코퍼스 중에서 영화 제목에 대한 개체명을 뽑아내고 싶다고 가정합시다.

해 B<br/>
리 I<br/>
포 I<br/>
터 I<br/>
보 O<br/>
러 O<br/>
가 O<br/>
자 O<br/>

영화 제목이 시작되는 글자인 '해'에서는 B가 사용되었고, 그리고 영화 제목이 끝나는 순간까지 I가 사용됩니다. 그리고 영화 제목이 아닌 부분에 대해서만 O가 사용됩니다. 

영화 제목에 대한 개체명과 극장에 대한 개체명이 있을 수 있습니다. 그럴 때는, 각 개체가 어떤 종류인지도 함께 태깅이 될 것입니다.

해 B-movie<br/>
리 I-movie<br/>
포 I-movie<br/>
터 I-movie<br/>
보 O<br/>
러 O<br/>
메 B-theater<br/>
가 I-theater<br/>
박 I-theater<br/>
스 I-theater<br/>
가 O<br/>
자 O<br/>

### 2) 양방향 LSTM(Bi-directional LSTM)으로 개체명 인식기 만들기

 CONLL2003은 개체명 인식을 위한 전통적인 영어 데이터 셋입니다.

다운로드 링크 : https://raw.githubusercontent.com/Franck-Dernoncourt/NeuroNER/master/neuroner/data/conll2003/en/train.txt<br/>
전체 데이터는 위 링크에서 train.txt 파일을 다운로드 받을 수 있습니다.

해당 데이터의 양식은 [단어] [품사 태깅] [청크 태깅] [개체명 태깅]의 형식으로 되어있습니다.

품사 태깅이 의미하는 바는 아래 링크 확인<br/>
https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

EU NNP B-NP B-ORG<br/>
rejects VBZ B-VP O<br/>
German JJ B-NP B-MISC<br/>
call NN I-NP O<br/>
to TO B-VP O<br/>
boycott VB I-VP O<br/>
British JJ B-NP B-MISC<br/>
lamb NN I-NP O<br/>
. . O O<br/>


Peter NNP B-NP B-PER<br/>
Blackburn NNP I-NP I-PER<br/>

예를 들어서 EU 옆에 붙어있는 NNP는 고유 명사 단수형을 의미하며, rejects 옆에 있는 VBZ는 3인칭 단수 동사 현재형을 의미합니다.

개체명 태깅의 경우에는 LOC는 location, ORG는 organization, PER은 person, MISC는 miscellaneous를 의미합니다.

다만, German 그 자체로 개체명 하나이기 때문에 거기서 개체명 인식은 종료되면서 뒤에 I가 별도로 붙는 단어가 나오지는 않았습니다. 

call은 개체명이 아니기 때문에 O가 태깅이 됩니다.

10번째 줄은 공란으로 되어 있는데, 이는 9번째 줄에서 문장이 끝나고 11번째 줄에서 새로운 문장이 시작됨을 의미합니다.

Peter는 개체명이 시작되면서 person에 해당되기 때문에 B-PER이라는 개체명 태깅이 붙습니다.

Blackburn에서는 I가 나오면서 I-PER이 개체명 태깅으로 붙게 됩니다.


```python
from collections import Counter
vocab=Counter()
import re
```

우선 훈련 데이터의 단어의 빈도수를 세기위해서 Counter, 데이터를 정제하기 위해서 re가 필요합니다.


```python
f = open('train.txt', 'r')
sentences = []
sentence = []
ner_set = set()
# 파이썬의 set은 중복을 허용하지 않는다. 개체명 태깅의 경우의 수. 즉, 종류를 알아내기 위함이다.  

for line in f:
    if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
        if len(sentence) > 0:
            sentences.append(sentence)
            sentence=[]
        continue
    splits = line.split(' ')
    # 공백을 기준으로 속성을 구분한다.
    splits[-1] = re.sub(r'\n', '', splits[-1])
    # 개체명 태깅 뒤에 붙어있는 줄바꿈 표시 \n을 제거한다.
    word=splits[0].lower()
    # 단어들은 소문자로 바꿔서 저장한다. 단어의 수를 줄이기 위해서이다.
    vocab[word]=vocab[word]+1
    # 단어마다 빈도 수가 몇 인지 기록한다.
    sentence.append([word, splits[-1]])
    # 단어와 개체명 태깅만 기록한다.
    ner_set.add(splits[-1])
    # set에다가 개체명 태깅을 집어 넣는다. 중복은 허용되지 않으므로
    # 나중에 개체명 태깅이 어떤 종류가 있는지 확인할 수 있다.
```

위의 코드를 거치면 단어와 개체명 태깅만이 남게되며, 문장의 구분이 유지됩니다.


```python
sentences[:3]
```




    [[['eu', 'B-ORG'],
      ['rejects', 'O'],
      ['german', 'B-MISC'],
      ['call', 'O'],
      ['to', 'O'],
      ['boycott', 'O'],
      ['british', 'B-MISC'],
      ['lamb', 'O'],
      ['.', 'O']],
     [['peter', 'B-PER'], ['blackburn', 'I-PER']],
     [['brussels', 'B-LOC'], ['1996-08-22', 'O']]]



위의 코드 중 'vocab[word]=vocab[word]+1' 해당 부분을 통해 현재 vocab은 각 단어에 대한 빈도수가 기록되어 있는 단어 집합이 되었습니다.

한 번 vocab에 저장된 데이터를 확인해보도록 하겠습니다.


```python
vocab
```




    Counter({'eu': 24,
             'rejects': 1,
             'german': 101,
             'call': 38,
             'to': 3424,
             'boycott': 5,
             'british': 96,
             'lamb': 3,
             '.': 7374,
             'peter': 31,
             'blackburn': 12,
             'brussels': 33,
             '1996-08-22': 125,
             'the': 8390,
             'european': 94,
             'commission': 67,
             'said': 1849,
             'on': 2092,
             'thursday': 292,
             'it': 762,
             'disagreed': 2,
             'with': 867,
             'advice': 5,
             'consumers': 9,
             'shun': 1,
             'until': 56,
             'scientists': 6,
             'determine': 11,
             'whether': 45,
             'mad': 13,
             'cow': 12,
             'disease': 37,
             'can': 72,
             'be': 525,
             'transmitted': 2,
             'sheep': 14,
             'germany': 144,
             "'s": 1566,
             'representative': 7,
             'union': 74,
             'veterinary': 8,
             'committee': 30,
             'werner': 2,
             'zwingmann': 2,
             'wednesday': 268,
             'should': 79,
             'buy': 26,
             'sheepmeat': 2,
             'from': 768,
             'countries': 44,
             'other': 142,
             'than': 144,
             'britain': 134,
             'scientific': 9,
             'was': 1095,
             'clearer': 1,
             '"': 2178,
             'we': 300,
             'do': 108,
             "n't": 98,
             'support': 63,
             'any': 94,
             'such': 42,
             'recommendation': 3,
             'because': 117,
             'see': 28,
             'grounds': 6,
             'for': 1465,
             ',': 7290,
             'chief': 55,
             'spokesman': 107,
             'nikolaus': 1,
             'van': 40,
             'der': 11,
             'pas': 1,
             'told': 199,
             'a': 3199,
             'news': 107,
             'briefing': 6,
             'he': 792,
             'further': 35,
             'study': 10,
             'required': 8,
             'and': 2872,
             'if': 131,
             'found': 62,
             'that': 797,
             'action': 31,
             'needed': 24,
             'taken': 41,
             'by': 726,
             'proposal': 10,
             'last': 305,
             'month': 90,
             'farm': 21,
             'commissioner': 9,
             'franz': 4,
             'fischler': 6,
             'ban': 40,
             'brains': 2,
             'spleens': 2,
             'spinal': 5,
             'cords': 2,
             'human': 41,
             'animal': 8,
             'food': 23,
             'chains': 6,
             'highly': 6,
             'specific': 7,
             'precautionary': 1,
             'move': 18,
             'protect': 4,
             'health': 38,
             'proposed': 15,
             'eu-wide': 1,
             'measures': 16,
             'after': 509,
             'reports': 32,
             'france': 128,
             'under': 171,
             'laboratory': 3,
             'conditions': 28,
             'could': 143,
             'contract': 32,
             'bovine': 4,
             'spongiform': 4,
             'encephalopathy': 4,
             '(': 2861,
             'bse': 11,
             ')': 2861,
             '--': 356,
             'but': 545,
             'agreed': 53,
             'review': 7,
             'his': 559,
             'standing': 9,
             'mational': 1,
             'officials': 134,
             'questioned': 12,
             'justified': 4,
             'as': 630,
             'there': 178,
             'only': 130,
             'slight': 14,
             'risk': 19,
             'spanish': 15,
             'minister': 176,
             'loyola': 1,
             'de': 61,
             'palacio': 1,
             'had': 561,
             'earlier': 90,
             'accused': 38,
             'at': 1146,
             'an': 542,
             'ministers': 15,
             "'": 146,
             'meeting': 88,
             'of': 3815,
             'causing': 4,
             'unjustified': 1,
             'alarm': 4,
             'through': 106,
             'dangerous': 13,
             'generalisation': 1,
             'backed': 7,
             'multidisciplinary': 1,
             'committees': 2,
             'are': 347,
             'due': 77,
             're-examine': 2,
             'issue': 60,
             'early': 77,
             'next': 112,
             'make': 58,
             'recommendations': 2,
             'senior': 30,
             'have': 502,
             'long': 40,
             'been': 368,
             'known': 31,
             'scrapie': 2,
             'brain-wasting': 3,
             'similar': 24,
             'which': 330,
             'is': 694,
             'believed': 28,
             'transferred': 3,
             'cattle': 12,
             'feed': 18,
             'containing': 5,
             'waste': 7,
             'farmers': 21,
             'denied': 24,
             'danger': 10,
             'their': 356,
             'expressed': 11,
             'concern': 19,
             'government': 271,
             'avoid': 15,
             'might': 36,
             'influence': 4,
             'across': 35,
             'europe': 39,
             'what': 71,
             'extremely': 8,
             'careful': 1,
             'how': 30,
             'going': 52,
             'take': 82,
             'lead': 81,
             'welsh': 2,
             'national': 129,
             'nfu': 1,
             'chairman': 32,
             'john': 50,
             'lloyd': 21,
             'jones': 17,
             'bbc': 2,
             'radio': 30,
             'bonn': 28,
             'has': 559,
             'led': 37,
             'efforts': 15,
             'public': 41,
             'consumer': 17,
             'confidence': 10,
             'collapsed': 3,
             'in': 3621,
             'march': 40,
             'report': 54,
             'suggested': 4,
             'humans': 3,
             'illness': 9,
             'eating': 8,
             'contaminated': 2,
             'beef': 10,
             'imported': 5,
             '47,600': 2,
             'year': 309,
             'nearly': 32,
             'half': 84,
             'total': 70,
             'imports': 15,
             'brought': 31,
             '4,275': 2,
             'tonnes': 82,
             'mutton': 3,
             'some': 179,
             '10': 197,
             'percent': 303,
             'overall': 23,
             'rare': 9,
             'hendrix': 5,
             'song': 2,
             'draft': 11,
             'sells': 5,
             'almost': 17,
             '$': 362,
             '17,000': 2,
             'london': 179,
             'handwritten': 1,
             'u.s.': 377,
             'guitar': 2,
             'legend': 1,
             'jimi': 1,
             'sold': 28,
             'auction': 15,
             'late': 55,
             'musician': 1,
             'favourite': 4,
             'possessions': 1,
             'florida': 35,
             'restaurant': 7,
             'paid': 18,
             '10,925': 1,
             'pounds': 23,
             '16,935': 1,
             'ai': 4,
             'no': 204,
             'telling': 6,
             'penned': 1,
             'piece': 5,
             'hotel': 11,
             'stationery': 2,
             '1966': 3,
             'end': 80,
             'january': 20,
             '1967': 4,
             'concert': 3,
             'english': 61,
             'city': 122,
             'nottingham': 6,
             'threw': 10,
             'sheet': 7,
             'paper': 24,
             'into': 161,
             'audience': 5,
             'where': 100,
             'retrieved': 1,
             'fan': 3,
             'buyers': 7,
             'also': 220,
             'snapped': 14,
             'up': 325,
             '16': 57,
             'items': 7,
             'were': 531,
             'put': 63,
             'former': 126,
             'girlfriend': 3,
             'kathy': 2,
             'etchingham': 1,
             'who': 440,
             'lived': 8,
             'him': 111,
             '1969': 2,
             'they': 392,
             'included': 20,
             'black': 34,
             'lacquer': 1,
             'mother': 34,
             'pearl': 1,
             'inlaid': 1,
             'box': 6,
             'used': 32,
             'store': 7,
             'drugs': 21,
             'anonymous': 3,
             'australian': 55,
             'purchaser': 1,
             'bought': 19,
             '5,060': 1,
             '7,845': 1,
             'guitarist': 1,
             'died': 37,
             'overdose': 1,
             '1970': 2,
             'aged': 8,
             '27': 37,
             'china': 98,
             'says': 66,
             'taiwan': 31,
             'spoils': 1,
             'atmosphere': 6,
             'talks': 85,
             'beijing': 24,
             'taipei': 8,
             'spoiling': 2,
             'resumption': 2,
             'strait': 3,
             'visit': 68,
             'ukraine': 39,
             'taiwanese': 3,
             'vice': 13,
             'president': 205,
             'lien': 2,
             'chan': 2,
             'this': 328,
             'week': 148,
             'infuriated': 3,
             'speaking': 8,
             'hours': 45,
             'chinese': 25,
             'state': 161,
             'media': 20,
             'time': 146,
             'right': 38,
             'engage': 2,
             'political': 74,
             'foreign': 121,
             'ministry': 75,
             'shen': 2,
             'guofang': 1,
             'reuters': 79,
             ':': 691,
             'necessary': 10,
             'opening': 31,
             'disrupted': 4,
             'authorities': 49,
             'quoted': 49,
             'top': 59,
             'negotiator': 5,
             'tang': 5,
             'shubei': 2,
             'visiting': 9,
             'group': 149,
             'rivals': 8,
             'hold': 38,
             'now': 89,
             'two': 388,
             'sides': 19,
             '...': 44,
             'hostility': 3,
             'overseas': 9,
             'edition': 5,
             'people': 193,
             'daily': 39,
             'saying': 75,
             'television': 50,
             'interview': 13,
             'read': 10,
             'comments': 8,
             'gave': 56,
             'details': 30,
             'why': 10,
             'considered': 11,
             'considers': 1,
             'renegade': 6,
             'province': 24,
             'opposed': 11,
             'all': 147,
             'gain': 6,
             'greater': 5,
             'international': 102,
             'recognition': 3,
             'rival': 18,
             'island': 35,
             'practical': 3,
             'steps': 4,
             'towards': 16,
             'goal': 25,
             'consultations': 2,
             'held': 81,
             'set': 83,
             'format': 1,
             'official': 115,
             'xinhua': 4,
             'agency': 60,
             'executive': 24,
             'association': 30,
             'relations': 23,
             'straits': 3,
             'july': 101,
             'car': 19,
             'registrations': 9,
             '14.2': 2,
             'pct': 93,
             'yr': 33,
             '/': 235,
             'frankfurt': 14,
             'first-time': 1,
             'motor': 15,
             'vehicles': 10,
             'jumped': 3,
             'year-earlier': 2,
             'period': 37,
             'federal': 31,
             'office': 57,
             '356,725': 1,
             'new': 466,
             'cars': 10,
             'registered': 9,
             '1996': 122,
             '304,850': 1,
             'passenger': 16,
             '15,613': 1,
             'trucks': 4,
             'figures': 32,
             'represent': 4,
             '13.6': 1,
             'increase': 22,
             '2.2': 3,
             'decline': 6,
             '1995': 63,
             'motor-bike': 1,
             'registration': 11,
             'rose': 45,
             '32.7': 1,
             'growth': 40,
             'partly': 6,
             'increased': 19,
             'number': 75,
             'germans': 2,
             'buying': 13,
             'abroad': 14,
             'while': 104,
             'manufacturers': 6,
             'domestic': 22,
             'demand': 50,
             'weak': 9,
             'posted': 11,
             'gains': 18,
             'numbers': 13,
             'volkswagen': 4,
             'ag': 14,
             'won': 163,
             '77,719': 1,
             'slightly': 15,
             'more': 171,
             'quarter': 45,
             'opel': 1,
             'together': 10,
             'general': 67,
             'motors': 3,
             'came': 45,
             'second': 242,
             'place': 51,
             '49,269': 1,
             '16.4': 1,
             'figure': 15,
             'third': 124,
             'ford': 12,
             '35,563': 1,
             'or': 188,
             '11.7': 2,
             'seat': 8,
             'porsche': 4,
             'fewer': 5,
             'compared': 24,
             '3,420': 1,
             '5522': 1,
             'fell': 46,
             '554': 1,
             '643': 1,
             'greek': 14,
             'socialists': 7,
             'give': 37,
             'green': 7,
             'light': 16,
             'pm': 7,
             'elections': 81,
             'athens': 22,
             'socialist': 16,
             'party': 127,
             'bureau': 12,
             'prime': 91,
             'costas': 9,
             'simitis': 8,
             'snap': 8,
             'its': 313,
             'secretary': 32,
             'skandalidis': 4,
             'reporters': 45,
             'announcement': 18,
             'cabinet': 22,
             'later': 49,
             'dimitris': 6,
             'kontogiannis': 5,
             'newsroom': 131,
             '+301': 7,
             '3311812-4': 7,
             'bayervb': 1,
             'sets': 19,
             'c$': 21,
             '100': 73,
             'million': 281,
             'six-year': 2,
             'bond': 29,
             'following': 56,
             'announced': 45,
             'manager': 39,
             'toronto': 46,
             'dominion': 2,
             'borrower': 3,
             'bayerische': 2,
             'vereinsbank': 2,
             'amt': 4,
             'mln': 15,
             'coupon': 8,
             '6.625': 1,
             'maturity': 7,
             '24.sep.02': 1,
             'type': 7,
             'straight': 30,
             'iss': 19,
             'price': 66,
             '100.92': 1,
             'pay': 38,
             'date': 51,
             '24.sep.96': 1,
             'full': 30,
             'fees': 3,
             '1.875': 1,
             'reoffer': 2,
             '99.32': 1,
             'spread': 4,
             '+20': 1,
             'bp': 6,
             'moody': 15,
             'aa1': 1,
             'listing': 5,
             'lux': 1,
             'freq': 1,
             '=': 11,
             's&p': 8,
             'denoms': 2,
             'k': 2,
             '1-10-100': 2,
             'sale': 31,
             'limits': 8,
             'us': 41,
             'uk': 11,
             'ca': 8,
             'neg': 2,
             'plg': 2,
             'crs': 2,
             'deflt': 2,
             'force': 32,
             'maj': 2,
             'gov': 10,
             'law': 39,
             'home': 124,
             'ctry': 2,
             'tax': 28,
             'provs': 2,
             'standard': 9,
             'mgt': 2,
             'und': 2,
             '0.275': 1,
             'sell': 27,
             'conc': 3,
             '1.60': 1,
             'praecip': 2,
             'underlying': 4,
             'govt': 3,
             '7.0': 6,
             'sept': 20,
             '2001': 7,
             'notes': 11,
             'joint': 10,
             '+44': 14,
             '171': 21,
             '542': 20,
             '7658': 3,
             'venantius': 2,
             '300': 24,
             '1999': 5,
             'frn': 2,
             'floating-rate': 1,
             'lehman': 7,
             'brothers': 6,
             'ab': 4,
             'swedish': 7,
             'mortgage': 6,
             '-': 1243,
             '12.5': 2,
             '21.jan.99': 1,
             'base': 11,
             '3m': 1,
             'libor': 2,
             's23.sep.96': 1,
             'aa3': 3,
             '99.956': 1,
             'aa+': 2,
             's': 4,
             'short': 35,
             'first': 469,
             'jp': 1,
             'fr': 1,
             'yes': 3,
             'ipma': 1,
             '2': 973,
             'sweden': 82,
             '5': 392,
             'issued': 14,
             'off': 122,
             'emtn': 1,
             'programme': 17,
             '8863': 1,
             'port': 28,
             'update': 2,
             'syria': 14,
             'lloyds': 2,
             'shipping': 3,
             'intelligence': 8,
             'service': 52,
             'lattakia': 2,
             'aug': 42,
             'waiting': 10,
             'tartous': 1,
             'presently': 2,
             '24': 63,
             'israel': 67,
             'plays': 8,
             'down': 119,
             'fears': 11,
             'war': 80,
             'colleen': 1,
             'siegel': 1,
             'jerusalem': 45,
             'outgoing': 2,
             'peace': 94,
             'current': 46,
             'tensions': 2,
             'between': 138,
             'appeared': 18,
             'storm': 10,
             'teacup': 2,
             'itamar': 2,
             'rabinovich': 3,
             'ambassador': 19,
             'washington': 61,
             'conducted': 8,
             'unfruitful': 1,
             'negotiations': 21,
             'looked': 17,
             'like': 44,
             'damascus': 5,
             'wanted': 19,
             'talk': 25,
             'rather': 12,
             'fight': 15,
             'appears': 6,
             'me': 35,
             'syrian': 4,
             'priority': 3,
             'still': 88,
             'negotiate': 2,
             'syrians': 2,
             'confused': 2,
             'definitely': 8,
             'tense': 5,
             'assessment': 7,
             'here': 27,
             'essentially': 1,
             'winding': 2,
             'term': 32,
             'will': 419,
             'replaced': 5,
             'eliahu': 1,
             'ben-elissar': 1,
             'israeli': 50,
             'envoy': 9,
             'egypt': 29,
             'right-wing': 9,
             'likud': 3,
             'politician': 6,
             'sent': 36,
             'message': 27,
             'via': 7,
             'committed': 10,
             'open': 120,
             'without': 38,
             'preconditions': 2,
             'slammed': 2,
             'creating': 3,
             'called': 61,
             'launching': 4,
             'hysterical': 1,
             'campaign': 40,
             'against': 244,
             'reported': 91,
             'recently': 16,
             'test': 60,
             'fired': 13,
             'missile': 2,
             'arms': 19,
             'purchases': 3,
             'defensive': 3,
             'purposes': 3,
             'hafez': 1,
             'al-': 1,
             'assad': 1,
             'ready': 9,
             'enter': 9,
             'david': 50,
             'levy': 12,
             'tension': 10,
             'mounted': 4,
             'since': 138,
             'benjamin': 10,
             'netanyahu': 25,
             'took': 76,
             'june': 90,
             'vowing': 1,
             'retain': 6,
             'golan': 3,
             'heights': 1,
             'captured': 9,
             'middle': 17,
             'east': 52,
             'israeli-syrian': 1,
             'deadlocked': 2,
             'over': 281,
             '1991': 23,
             'despite': 36,
             'previous': 42,
             'willingness': 1,
             'concessions': 2,
             'february': 20,
             'voices': 3,
             'coming': 28,
             'out': 302,
             'bad': 19,
             'not': 541,
             'good': 55,
             'expressions': 1,
             'declarations': 1,
             'must': 32,
             'worrying': 2,
             'artificial': 1,
             'very': 64,
             'those': 52,
             'become': 20,
             'prisoners': 6,
             'expect': 20,
             'face': 31,
             'answer': 8,
             'our': 79,
             'want': 51,
             'god': 5,
             'forbid': 1,
             'one': 291,
             'benefits': 3,
             'wars': 4,
             'channel': 2,
             'calming': 1,
             'signal': 6,
             'source': 25,
             'spokesmen': 2,
             'confirm': 7,
             'messages': 3,
             'reassure': 1,
             'cairo': 12,
             'united': 118,
             'states': 76,
             'moscow': 75,
             'polish': 26,
             'diplomat': 18,
             'denies': 2,
             'nurses': 5,
             'stranded': 4,
             'libya': 5,
             'tunis': 3,
             'tabloid': 3,
             'refusing': 4,
             'exit': 4,
             'visas': 3,
             'trying': 32,
             'return': 46,
             'working': 39,
             'north': 72,
             'african': 55,
             'country': 70,
             'true': 7,
             'today': 32,
             'knowledge': 3,
             'nurse': 4,
             'kept': 25,
             'her': 177,
             'received': 31,
             'complaint': 4,
             'embassy': 36,
             'charge': 19,
             "d'affaires": 1,
             'tripoli': 4,
             'tadeusz': 1,
             'awdankiewicz': 2,
             'telephone': 17,
             'poland': 32,
             'labour': 16,
             'would': 330,
             'send': 11,
             'team': 94,
             'investigate': 3,
             'probe': 6,
             'prompted': 9,
             'complaining': 3,
             'about': 189,
             'work': 46,
             'non-payment': 1,
             'salaries': 2,
             'estimated': 31,
             '800': 13,
             'iranian': 23,
             'opposition': 60,
             'leaders': 56,
             'meet': 60,
             'baghdad': 27,
             'hassan': 5,
             'hafidh': 1,
             'exile': 6,
             'based': 26,
             'iraq': 70,
             'vowed': 6,
             'extend': 3,
             'iran': 29,
             'kurdish': 34,
             'rebels': 55,
             'attacked': 7,
             'troops': 45,
             'deep': 2,
             'inside': 12,
             'mujahideen': 3,
             'khalq': 3,
             'statement': 108,
             'leader': 63,
             'massoud': 3,
             'rajavi': 2,
             'met': 26,
             'secretary-general': 8,
             'kurdistan': 18,
             'democratic': 57,
             'kdpi': 2,
             'rastegar': 1,
             'voiced': 3,
             'rebel': 26,
             'kurds': 7,
             'emphasised': 1,
             'resistance': 5,
             'continue': 26,
             'stand': 6,
             'side': 40,
             'compatriots': 1,
             'movement': 25,
             'signals': 2,
             'level': 29,
             'cooperation': 22,
             'oppositions': 1,
             'heavily': 5,
             'bombarded': 2,
             'targets': 6,
             'northern': 61,
             'pursuit': 6,
             'guerrillas': 31,
             'iraqi': 54,
             'areas': 25,
             'outside': 25,
             'control': 39,
             'bordering': 2,
             'patriotic': 7,
             'puk': 25,
             'kdp': 17,
             'main': 42,
             'factions': 14,
             'forces': 48,
             'ousted': 8,
             'kuwait': 6,
             'gulf': 17,
             'clashes': 12,
             'parties': 25,
             'broke': 23,
             'weekend': 54,
             'most': 78,
             'serious': 16,
             'fighting': 41,
             'u.s.-sponsored': 1,
             'ceasefire': 36,
             'shelling': 6,
             'positions': 19,
             'qasri': 1,
             'region': 39,
             'suleimaniya': 1,
             'near': 55,
             'border': 27,
             'days': 76,
             'killed': 82,
             'wounded': 15,
             'attack': 38,
             'both': 71,
             'turkey': 30,
             'mount': 4,
             'air': 29,
             'land': 26,
             'strikes': 5,
             'own': 38,
             'u.s.-led': 6,
             'southern': 53,
             'protects': 2,
             'possible': 32,
             'attacks': 25,
             'saudi': 26,
             'riyal': 5,
             'rates': 33,
             'steady': 21,
             'quiet': 20,
             'summer': 20,
             'trade': 66,
             'manama': 2,
             'spot': 29,
             'dollar': 55,
             'interbank': 7,
             'deposit': 7,
             'mainly': 9,
             'dealers': 32,
             'kingdom': 12,
             'changes': 8,
             'market': 149,
             'holidays': 3,
             'dealer': 18,
             '3.7504': 1,
             '06': 1,
             'one-month': 2,
             'deposits': 1,
             '5-1/2': 1,
             '3/8': 1,
             'three': 214,
             'months': 60,
             '5-5/8': 1,
             '1/2': 84,
             'six': 127,
             '5-3/4': 1,
             '5/8': 5,
             'one-year': 1,
             'funds': 13,
             '5-7/8': 1,
             'approves': 3,
             'arafat': 47,
             'flight': 26,
             ...})



vocab이 위와 같은 데이터의 형식을 갖고 있기 때문에, vocab의 길이를 측정하면 단어의 개수를 알수 있습니다.


```python
len(vocab)
```




    21009



훈련 데이터에는 약 21,000여개의 단어가 있음을 알 수 있습니다.

앞서 개체명 태깅의 종류를 알기 위해서 ner_set에다가 중복은 허용하지 않고, 개체명 태깅을 저장했습니다.


```python
print(ner_set)
```

    {'B-ORG', 'I-MISC', 'I-LOC', 'I-PER', 'B-LOC', 'O', 'B-PER', 'I-ORG', 'B-MISC'}
    

훈련 데이터에 등장하는 개체명 태깅의 수는 위와 같습니다.

이제 vocab 안의 단어들을 빈도수 순으로 정렬해보도록 하겠습니다.


```python
vocab_sorted=sorted(vocab.items(), key=lambda x:x[1], reverse=True)
# vocab을 빈도수 순으로 정렬한다.
vocab_sorted
# 출력
```




    [('the', 8390),
     ('.', 7374),
     (',', 7290),
     ('of', 3815),
     ('in', 3621),
     ('to', 3424),
     ('a', 3199),
     ('and', 2872),
     ('(', 2861),
     (')', 2861),
     ('"', 2178),
     ('on', 2092),
     ('said', 1849),
     ("'s", 1566),
     ('for', 1465),
     ('1', 1421),
     ('-', 1243),
     ('at', 1146),
     ('was', 1095),
     ('2', 973),
     ('0', 945),
     ('3', 932),
     ('with', 867),
     ('that', 797),
     ('he', 792),
     ('from', 768),
     ('it', 762),
     ('by', 726),
     ('is', 694),
     (':', 691),
     ('as', 630),
     ('4', 581),
     ('had', 561),
     ('his', 559),
     ('has', 559),
     ('but', 545),
     ('an', 542),
     ('not', 541),
     ('were', 531),
     ('be', 525),
     ('after', 509),
     ('have', 502),
     ('first', 469),
     ('new', 466),
     ('who', 440),
     ('will', 419),
     ('they', 392),
     ('5', 392),
     ('two', 388),
     ('u.s.', 377),
     ('been', 368),
     ('$', 362),
     ('--', 356),
     ('their', 356),
     ('beat', 352),
     ('are', 347),
     ('6', 343),
     ('which', 330),
     ('would', 330),
     ('this', 328),
     ('up', 325),
     ('its', 313),
     ('year', 309),
     ('i', 308),
     ('last', 305),
     ('percent', 303),
     ('out', 302),
     ('we', 300),
     ('thursday', 292),
     ('one', 291),
     ('million', 281),
     ('over', 281),
     ('government', 271),
     ('wednesday', 268),
     ('police', 268),
     ('7', 259),
     ('results', 257),
     ('against', 244),
     ('second', 242),
     ('when', 242),
     ('/', 235),
     ('also', 220),
     ('tuesday', 217),
     ('three', 214),
     ('soccer', 214),
     ('president', 205),
     ('no', 204),
     ('division', 204),
     ('told', 199),
     ('10', 197),
     ('monday', 197),
     ('people', 193),
     ('about', 189),
     ('or', 188),
     ('friday', 185),
     ('league', 185),
     ('some', 179),
     ('london', 179),
     ('there', 178),
     ('world', 178),
     ('her', 177),
     ('minister', 176),
     ('under', 171),
     ('more', 171),
     ('york', 168),
     ('9', 167),
     ('1996-08-28', 167),
     ('won', 163),
     ('into', 161),
     ('state', 161),
     ('sunday', 160),
     ('8', 155),
     ('before', 153),
     ('south', 153),
     ('played', 151),
     ('group', 149),
     ('market', 149),
     ('week', 148),
     ('england', 148),
     ('all', 147),
     ("'", 146),
     ('time', 146),
     ('germany', 144),
     ('than', 144),
     ('could', 143),
     ('other', 142),
     ('australia', 140),
     ('she', 139),
     ('between', 138),
     ('since', 138),
     ('points', 138),
     ('match', 138),
     ('company', 136),
     ('bank', 135),
     ('round', 135),
     ('britain', 134),
     ('officials', 134),
     ('years', 134),
     ('games', 134),
     ('if', 131),
     ('newsroom', 131),
     ('only', 130),
     ('saturday', 130),
     ('national', 129),
     ('france', 128),
     ('party', 127),
     ('six', 127),
     ('former', 126),
     ('1996-08-22', 125),
     ('four', 125),
     ('third', 124),
     ('home', 124),
     ('1996-08-29', 124),
     ('city', 122),
     ('1996', 122),
     ('off', 122),
     ('cup', 122),
     ('five', 122),
     ('win', 122),
     ('foreign', 121),
     ('11', 121),
     ('1996-08-27', 121),
     ('open', 120),
     ('day', 120),
     ('down', 119),
     ('august', 119),
     ('13', 119),
     ('united', 118),
     ('because', 117),
     ('6-4', 117),
     ('6-3', 117),
     ('official', 115),
     ('did', 115),
     ('21', 114),
     ('just', 114),
     ('next', 112),
     ('15', 112),
     ('him', 111),
     ('spain', 111),
     ('standings', 111),
     ('1996-08-26', 111),
     ('expected', 110),
     ('shares', 109),
     ('do', 108),
     ('statement', 108),
     ('spokesman', 107),
     ('news', 107),
     ('pakistan', 107),
     ('through', 106),
     ('may', 106),
     ('women', 106),
     ('made', 106),
     ('70', 105),
     ('v', 105),
     ('while', 104),
     ('men', 104),
     ('12', 104),
     ('chicago', 104),
     ('1996-08-23', 103),
     ('international', 102),
     ('them', 102),
     ('14', 102),
     ('lost', 102),
     ('german', 101),
     ('july', 101),
     ('where', 100),
     ('russian', 100),
     ('back', 100),
     ('20', 100),
     ('6-2', 100),
     ('september', 99),
     ("n't", 98),
     ('china', 98),
     ('italy', 98),
     ('british', 96),
     ('2.', 95),
     ('3.', 95),
     ('european', 94),
     ('any', 94),
     ('peace', 94),
     ('team', 94),
     ('1.', 94),
     ('clinton', 94),
     ('pct', 93),
     ('matches', 93),
     ('japan', 92),
     ('seconds', 92),
     ('prime', 91),
     ('reported', 91),
     ('billion', 91),
     ('69', 91),
     ('month', 90),
     ('earlier', 90),
     ('june', 90),
     ('central', 90),
     ('now', 89),
     ('russia', 89),
     ('meeting', 88),
     ('still', 88),
     ('metres', 87),
     ('final', 87),
     ('30', 86),
     ('being', 86),
     ('talks', 85),
     ('west', 85),
     ('added', 85),
     ('71', 85),
     ('1996-08-25', 85),
     ('half', 84),
     ('1/2', 84),
     ('during', 84),
     ('french', 84),
     ('season', 84),
     ('b', 84),
     ('you', 84),
     ('set', 83),
     ('capital', 83),
     ('leading', 83),
     ('san', 83),
     ('take', 82),
     ('tonnes', 82),
     ('sweden', 82),
     ('killed', 82),
     ('st', 82),
     ('india', 82),
     ('lead', 81),
     ('held', 81),
     ('elections', 81),
     ('net', 81),
     ('around', 81),
     ('end', 80),
     ('war', 80),
     ('tennis', 80),
     ('security', 80),
     ('should', 79),
     ('reuters', 79),
     ('our', 79),
     ('most', 78),
     ('part', 78),
     ('mark', 78),
     ('game', 78),
     ('due', 77),
     ('early', 77),
     ('took', 76),
     ('states', 76),
     ('days', 76),
     ('so', 76),
     ('innings', 76),
     ('ministry', 75),
     ('saying', 75),
     ('number', 75),
     ('moscow', 75),
     ('68', 75),
     ('cricket', 75),
     ('6-1', 75),
     ('american', 75),
     ('major', 75),
     ('minutes', 75),
     ('union', 74),
     ('political', 74),
     ('seven', 74),
     ('per', 74),
     ('7-6', 74),
     ('netherlands', 74),
     ('100', 73),
     ('republic', 73),
     ('hong', 73),
     ('can', 72),
     ('north', 72),
     ('victory', 72),
     ('championship', 72),
     ('what', 71),
     ('both', 71),
     ('22', 71),
     ('well', 71),
     ('belgium', 71),
     ('total', 70),
     ('country', 70),
     ('iraq', 70),
     ('play', 70),
     ('court', 70),
     ('kong', 70),
     ('50', 69),
     ('close', 69),
     ('africa', 69),
     ('czech', 69),
     ('visit', 68),
     ('dutch', 68),
     ('25', 68),
     ('result', 68),
     ('champion', 68),
     ('profit', 68),
     ('commission', 67),
     ('general', 67),
     ('israel', 67),
     ('left', 67),
     ('eight', 67),
     ('local', 67),
     ('minute', 67),
     ('says', 66),
     ('price', 66),
     ('trade', 66),
     ('town', 66),
     ('paris', 66),
     ('66', 66),
     ('run', 66),
     ('1996-08-24', 66),
     ('sales', 65),
     ('4.', 65),
     ('very', 64),
     ('press', 64),
     ('67', 64),
     ('vs.', 64),
     ('5.', 64),
     ('6.', 64),
     ('support', 63),
     ('put', 63),
     ('1995', 63),
     ('24', 63),
     ('leader', 63),
     ('prices', 63),
     ('7-5', 63),
     ('then', 63),
     ('found', 62),
     ('record', 62),
     ('go', 62),
     ('same', 62),
     ('man', 62),
     ('western', 62),
     ('de', 61),
     ('english', 61),
     ('washington', 61),
     ('called', 61),
     ('northern', 61),
     ('inc', 61),
     ('say', 61),
     ('issue', 60),
     ('agency', 60),
     ('test', 60),
     ('opposition', 60),
     ('meet', 60),
     ('months', 60),
     ('military', 60),
     ('runs', 60),
     ('rate', 60),
     ('top', 59),
     ('these', 59),
     ('lower', 59),
     ('ago', 59),
     ('singles', 59),
     ('64', 59),
     ('make', 58),
     ('amsterdam', 58),
     ('72', 58),
     ('race', 58),
     ('newspaper', 58),
     ('deal', 58),
     ('goals', 58),
     ('16', 57),
     ('office', 57),
     ('democratic', 57),
     ('ended', 57),
     ('60', 57),
     ('cents', 57),
     ('until', 56),
     ('gave', 56),
     ('following', 56),
     ('leaders', 56),
     ('behind', 56),
     ('series', 56),
     ('another', 56),
     ('tour', 56),
     ('sri', 56),
     ('several', 56),
     ('chief', 55),
     ('late', 55),
     ('australian', 55),
     ('good', 55),
     ('african', 55),
     ('rebels', 55),
     ('near', 55),
     ('dollar', 55),
     ('players', 55),
     ('75', 55),
     ('michael', 55),
     ('my', 55),
     ('report', 54),
     ('iraqi', 54),
     ('weekend', 54),
     ('economic', 54),
     ('start', 54),
     ('halftime', 54),
     ('tournament', 54),
     ('c', 54),
     ('miles', 54),
     ('refugees', 54),
     ('agreed', 53),
     ('southern', 53),
     ('many', 53),
     ('74', 53),
     ('get', 53),
     ('power', 53),
     ('7.', 53),
     ('plan', 53),
     ('going', 52),
     ('service', 52),
     ('east', 52),
     ('those', 52),
     ('election', 52),
     ('73', 52),
     ('attendance', 52),
     (';', 52),
     ('sydney', 52),
     ('drawn', 52),
     ('place', 51),
     ('date', 51),
     ('want', 51),
     ('business', 51),
     ('paul', 51),
     ('white', 51),
     ('stock', 51),
     ('john', 50),
     ('television', 50),
     ('demand', 50),
     ('israeli', 50),
     ('david', 50),
     ('gmt', 50),
     ('taking', 50),
     ('baseball', 50),
     ('atlanta', 50),
     ('grand', 50),
     ('including', 50),
     ('8.', 50),
     ('index', 50),
     ('arrested', 50),
     ('authorities', 49),
     ('quoted', 49),
     ('later', 49),
     ('palestinian', 49),
     ('parliament', 49),
     ('corp', 49),
     ('ahmed', 49),
     ('allowed', 49),
     ('forces', 48),
     ('asked', 48),
     ('cash', 48),
     ('hit', 48),
     ('km', 48),
     ('california', 48),
     ('zealand', 48),
     ('brazil', 48),
     ('house', 48),
     ('already', 48),
     ('army', 48),
     ('arafat', 47),
     ('rights', 47),
     ('martin', 47),
     ('club', 47),
     ('17', 47),
     ('28', 47),
     ('yeltsin', 47),
     ('fell', 46),
     ('toronto', 46),
     ('current', 46),
     ('return', 46),
     ('work', 46),
     ('higher', 46),
     ('ahead', 46),
     ('loss', 46),
     ('31', 46),
     ('weeks', 46),
     ('m.', 46),
     ('1-0', 46),
     ('26', 46),
     ('exchange', 46),
     ('trading', 46),
     ('dole', 46),
     ('whether', 45),
     ('hours', 45),
     ('rose', 45),
     ('quarter', 45),
     ('came', 45),
     ('reporters', 45),
     ('announced', 45),
     ('jerusalem', 45),
     ('troops', 45),
     ('share', 45),
     ('closed', 45),
     ('finland', 45),
     ('way', 45),
     ('squad', 45),
     ('draw', 45),
     ('vs', 45),
     ('hospital', 45),
     ('agreement', 45),
     ('money', 45),
     ('19', 45),
     ('countries', 44),
     ('...', 44),
     ('like', 44),
     ('23', 44),
     ('night', 44),
     ('oil', 44),
     ('head', 44),
     ('austria', 44),
     ('1994', 44),
     ('best', 44),
     ('baltimore', 44),
     ('18', 44),
     ('65', 44),
     ('conference', 44),
     ('financial', 43),
     ('began', 43),
     ('scores', 43),
     ('away', 43),
     ('olympic', 43),
     ('went', 43),
     ('lanka', 43),
     ('morning', 43),
     ('decision', 43),
     ('high', 43),
     ('area', 43),
     ('old', 43),
     ('plans', 43),
     ('bonds', 43),
     ('such', 42),
     ('aug', 42),
     ('previous', 42),
     ('main', 42),
     ('few', 42),
     ('1997', 42),
     ('past', 42),
     ('francisco', 42),
     ('los', 42),
     ('hits', 42),
     ('nations', 42),
     ('budget', 42),
     ('lebed', 42),
     ('taken', 41),
     ('human', 41),
     ('public', 41),
     ('us', 41),
     ('fighting', 41),
     ('fourth', 41),
     ('little', 41),
     ('champions', 41),
     ('eastern', 41),
     ('angeles', 41),
     ('nine', 41),
     ('latest', 41),
     ('gold', 41),
     ('italian', 41),
     ('van', 40),
     ('ban', 40),
     ('long', 40),
     ('march', 40),
     ('growth', 40),
     ('campaign', 40),
     ('side', 40),
     ('u.n.', 40),
     ('strike', 40),
     ('63', 40),
     ('fall', 40),
     ('scored', 40),
     ('signed', 40),
     ('canada', 40),
     ('seattle', 40),
     ('colorado', 40),
     ('co', 40),
     ('much', 40),
     ('bill', 40),
     ('europe', 39),
     ('ukraine', 39),
     ('daily', 39),
     ('manager', 39),
     ('law', 39),
     ('working', 39),
     ('control', 39),
     ('region', 39),
     ('markets', 39),
     ('future', 39),
     ('winning', 39),
     ('boston', 39),
     ('40', 39),
     ('times', 39),
     ('recent', 39),
     ('think', 39),
     ('call', 38),
     ('health', 38),
     ('accused', 38),
     ('right', 38),
     ('hold', 38),
     ('pay', 38),
     ('without', 38),
     ('attack', 38),
     ('own', 38),
     ('plane', 38),
     ('available', 38),
     ('96', 38),
     ('akram', 38),
     ('wickets', 38),
     ('62', 38),
     ('order', 38),
     ('airport', 38),
     ('workers', 38),
     ('again', 38),
     ('disease', 37),
     ('led', 37),
     ('died', 37),
     ('27', 37),
     ('period', 37),
     ('give', 37),
     ('free', 37),
     ('average', 37),
     ('interest', 37),
     ('industry', 37),
     ('members', 37),
     ('ireland', 37),
     ('texas', 37),
     ('ajax', 37),
     ('9.', 37),
     ('production', 37),
     ('forecast', 37),
     ('seen', 37),
     ('vote', 37),
     ('---', 37),
     ('might', 36),
     ('sent', 36),
     ('despite', 36),
     ('embassy', 36),
     ('ceasefire', 36),
     ('failed', 36),
     ('help', 36),
     ('does', 36),
     ('centre', 36),
     ('released', 36),
     ('louis', 36),
     ('overs', 36),
     ('case', 36),
     ("'re", 36),
     ('dutroux', 36),
     ('further', 35),
     ('across', 35),
     ('florida', 35),
     ('island', 35),
     ('short', 35),
     ('me', 35),
     ('october', 35),
     ('started', 35),
     ('county', 35),
     ('wasim', 35),
     ('captain', 35),
     ('costa', 35),
     ('cut', 35),
     ('belgian', 35),
     ('29', 35),
     ('children', 35),
     ('death', 35),
     ('least', 35),
     ('council', 35),
     ('rugby', 35),
     ('black', 34),
     ('mother', 34),
     ('kurdish', 34),
     ('jordan', 34),
     ('planned', 34),
     ('stories', 34),
     ('analysts', 34),
     ('strong', 34),
     ('april', 34),
     ('given', 34),
     ('romania', 34),
     ('premier', 34),
     ('59', 34),
     ('thomas', 34),
     ('coach', 34),
     ('minnesota', 34),
     ('kansas', 34),
     ('diego', 34),
     ('declared', 34),
     ('playing', 34),
     ('chechnya', 34),
     ('convention', 34),
     ('wheat', 34),
     ('brussels', 33),
     ('yr', 33),
     ('rates', 33),
     ('according', 33),
     ('slovakia', 33),
     ('title', 33),
     ('54', 33),
     ('manchester', 33),
     ('houston', 33),
     ('shot', 33),
     ('fifth', 33),
     ('10.', 33),
     ('yet', 33),
     ('civil', 33),
     ('confirmed', 33),
     ('bosnia', 33),
     ('bosnian', 33),
     ('republican', 33),
     ('reports', 32),
     ('contract', 32),
     ('chairman', 32),
     ('nearly', 32),
     ('used', 32),
     ('figures', 32),
     ('secretary', 32),
     ('force', 32),
     ('term', 32),
     ('must', 32),
     ('trying', 32),
     ('today', 32),
     ('poland', 32),
     ('possible', 32),
     ('dealers', 32),
     ('forced', 32),
     ('golf', 32),
     ('a.', 32),
     ('tabulate', 32),
     ('detroit', 32),
     ('oakland', 32),
     ('each', 32),
     ('companies', 32),
     ('seed', 32),
     ('point', 32),
     ('come', 32),
     ('passengers', 32),
     ('moslem', 32),
     ('treaty', 32),
     ('ltd', 32),
     ('peter', 31),
     ('action', 31),
     ('known', 31),
     ('brought', 31),
     ('taiwan', 31),
     ('opening', 31),
     ('federal', 31),
     ('sale', 31),
     ('face', 31),
     ('received', 31),
     ('estimated', 31),
     ('guerrillas', 31),
     ('exports', 31),
     ('aggregate', 31),
     ('named', 31),
     ('2-0', 31),
     ('robert', 31),
     ('injured', 31),
     ('road', 31),
     ('woman', 31),
     ('clear', 31),
     ('small', 31),
     ('charges', 31),
     ('mexico', 31),
     ('traders', 31),
     ('kenya', 31),
     ('nigeria', 31),
     ('committee', 30),
     ('senior', 30),
     ('how', 30),
     ('radio', 30),
     ('details', 30),
     ('association', 30),
     ('straight', 30),
     ('full', 30),
     ('turkey', 30),
     ('red', 30),
     ('likely', 30),
     ('earnings', 30),
     ('immediately', 30),
     ('services', 30),
     ('use', 30),
     ('l', 30),
     ('wimbledon', 30),
     ('argentina', 30),
     ('cleveland', 30),
     ('1,000', 30),
     ('*', 30),
     ('note', 30),
     ('soon', 30),
     ('post', 30),
     ('investors', 30),
     ('showed', 30),
     ('private', 30),
     ('department', 30),
     ('yen', 30),
     ('japanese', 30),
     ('train', 30),
     ('48', 30),
     ('bond', 29),
     ('egypt', 29),
     ('iran', 29),
     ('level', 29),
     ('air', 29),
     ('spot', 29),
     ('armed', 29),
     ('76', 29),
     ('rally', 29),
     ('unless', 29),
     ('prix', 29),
     ('better', 29),
     ('croft', 29),
     ('0-0', 29),
     ('cincinnati', 29),
     ('61', 29),
     ('montreal', 29),
     ('philadelphia', 29),
     ('got', 29),
     ('medical', 29),
     ('groups', 29),
     ('although', 29),
     ('trip', 29),
     ('stage', 29),
     ('process', 29),
     ('policy', 29),
     ('see', 28),
     ('conditions', 28),
     ('believed', 28),
     ('bonn', 28),
     ('sold', 28),
     ('tax', 28),
     ('port', 28),
     ('coming', 28),
     ('board', 28),
     ('seeding', 28),
     ('holiday', 28),
     ('waqar', 28),
     ('younis', 28),
     ('mushtaq', 28),
     ('qualifier', 28),
     ('johnson', 28),
     ('milwaukee', 28),
     ('young', 28),
     ('psv', 28),
     ('defence', 28),
     ('swiss', 28),
     ('violence', 28),
     ('even', 28),
     ('school', 28),
     ('university', 28),
     ('prison', 28),
     ('information', 28),
     ('rise', 28),
     ('independence', 28),
     ('pound', 28),
     ('serb', 28),
     ('6-0', 28),
     ('sell', 27),
     ('here', 27),
     ('message', 27),
     ('baghdad', 27),
     ('border', 27),
     ('scheduled', 27),
     ('data', 27),
     ('keep', 27),
     ('winner', 27),
     ('s.', 27),
     ('rule', 27),
     ('within', 27),
     ('homer', 27),
     ('jose', 27),
     ('making', 27),
     ('system', 27),
     ('levels', 27),
     ('56', 27),
     ('municipal', 27),
     ('deputy', 27),
     ('comment', 27),
     ('nuclear', 27),
     ('change', 27),
     ('buy', 26),
     ('polish', 26),
     ('based', 26),
     ('met', 26),
     ('rebel', 26),
     ('continue', 26),
     ('land', 26),
     ('saudi', 26),
     ('flight', 26),
     ('street', 26),
     ('53', 26),
     ('know', 26),
     ('course', 26),
     ('scorers', 26),
     ('director', 26),
     ('khan', 26),
     ('mullally', 26),
     ('great', 26),
     ('canadian', 26),
     ('switzerland', 26),
     ('pittsburgh', 26),
     ('inning', 26),
     ('tried', 26),
     ('55', 26),
     ('letter', 26),
     ('grozny', 26),
     ('completed', 26),
     ('charged', 26),
     ('wife', 26),
     ('volume', 26),
     ('illegal', 26),
     ('position', 26),
     ('indian', 26),
     ('economy', 26),
     ('reached', 26),
     ('struck', 26),
     ('bid', 26),
     ('king', 26),
     ('chinese', 25),
     ('goal', 25),
     ('talk', 25),
     ('netanyahu', 25),
     ('source', 25),
     ('kept', 25),
     ('movement', 25),
     ('areas', 25),
     ('outside', 25),
     ('puk', 25),
     ('parties', 25),
     ('attacks', 25),
     ('problem', 25),
     ('turkish', 25),
     ('step', 25),
     ('45', 25),
     ('2-1', 25),
     ('1-1', 25),
     ('prefix', 25),
     ('opened', 25),
     ('tie', 25),
     ('percentage', 25),
     ('58', 25),
     ('ground', 25),
     ('stocks', 25),
     ('able', 25),
     ('deficit', 25),
     ('arrived', 25),
     ("'m", 25),
     ('poor', 25),
     ('village', 25),
     ('coast', 25),
     ('osce', 25),
     ('leave', 25),
     ('having', 25),
     ('drug', 25),
     ('lot', 25),
     ('eu', 24),
     ('needed', 24),
     ('similar', 24),
     ('denied', 24),
     ('paper', 24),
     ('beijing', 24),
     ('province', 24),
     ('executive', 24),
     ('compared', 24),
     ('300', 24),
     ('schedule', 24),
     ('firm', 24),
     ('digest', 24),
     ...]



사실 해당 데이터는 등장 빈도가 1인 데이터도 다수입니다. 전체 데이터에서 단어의 등장 빈도수가 5이하인 경우는 데이터에서 배제시키고, 모르는 단어. OOV(Out-of-Vocabulary)로 간주합니다. 여기서 vocab_sorted 부분에서 굳이 빈도수가 5이하인 경우를 빼서 저장하지 않고 vocab_sorted로 부터 또 다시 새로운 단어 집합을 만듭니다.

이번에 만드는 단어 집합은 등장 빈도수 순으로 인덱스를 부여합니다. 그리고 이 때, 빈도수가 5이하인 단어들을 배제시킵니다.


```python
word_to_index = {w: i + 2 for i, (w, n) in enumerate(vocab_sorted) if n > 5}
word_to_index['PAD'] = 0  # 패딩을 위해 인덱스 0 할당
word_to_index['OOV'] = 1  # 모르는 단어을 위해 인덱스 1 할당
word_to_index # 출력
```




    {'the': 2,
     '.': 3,
     ',': 4,
     'of': 5,
     'in': 6,
     'to': 7,
     'a': 8,
     'and': 9,
     '(': 10,
     ')': 11,
     '"': 12,
     'on': 13,
     'said': 14,
     "'s": 15,
     'for': 16,
     '1': 17,
     '-': 18,
     'at': 19,
     'was': 20,
     '2': 21,
     '0': 22,
     '3': 23,
     'with': 24,
     'that': 25,
     'he': 26,
     'from': 27,
     'it': 28,
     'by': 29,
     'is': 30,
     ':': 31,
     'as': 32,
     '4': 33,
     'had': 34,
     'his': 35,
     'has': 36,
     'but': 37,
     'an': 38,
     'not': 39,
     'were': 40,
     'be': 41,
     'after': 42,
     'have': 43,
     'first': 44,
     'new': 45,
     'who': 46,
     'will': 47,
     'they': 48,
     '5': 49,
     'two': 50,
     'u.s.': 51,
     'been': 52,
     '$': 53,
     '--': 54,
     'their': 55,
     'beat': 56,
     'are': 57,
     '6': 58,
     'which': 59,
     'would': 60,
     'this': 61,
     'up': 62,
     'its': 63,
     'year': 64,
     'i': 65,
     'last': 66,
     'percent': 67,
     'out': 68,
     'we': 69,
     'thursday': 70,
     'one': 71,
     'million': 72,
     'over': 73,
     'government': 74,
     'wednesday': 75,
     'police': 76,
     '7': 77,
     'results': 78,
     'against': 79,
     'second': 80,
     'when': 81,
     '/': 82,
     'also': 83,
     'tuesday': 84,
     'three': 85,
     'soccer': 86,
     'president': 87,
     'no': 88,
     'division': 89,
     'told': 90,
     '10': 91,
     'monday': 92,
     'people': 93,
     'about': 94,
     'or': 95,
     'friday': 96,
     'league': 97,
     'some': 98,
     'london': 99,
     'there': 100,
     'world': 101,
     'her': 102,
     'minister': 103,
     'under': 104,
     'more': 105,
     'york': 106,
     '9': 107,
     '1996-08-28': 108,
     'won': 109,
     'into': 110,
     'state': 111,
     'sunday': 112,
     '8': 113,
     'before': 114,
     'south': 115,
     'played': 116,
     'group': 117,
     'market': 118,
     'week': 119,
     'england': 120,
     'all': 121,
     "'": 122,
     'time': 123,
     'germany': 124,
     'than': 125,
     'could': 126,
     'other': 127,
     'australia': 128,
     'she': 129,
     'between': 130,
     'since': 131,
     'points': 132,
     'match': 133,
     'company': 134,
     'bank': 135,
     'round': 136,
     'britain': 137,
     'officials': 138,
     'years': 139,
     'games': 140,
     'if': 141,
     'newsroom': 142,
     'only': 143,
     'saturday': 144,
     'national': 145,
     'france': 146,
     'party': 147,
     'six': 148,
     'former': 149,
     '1996-08-22': 150,
     'four': 151,
     'third': 152,
     'home': 153,
     '1996-08-29': 154,
     'city': 155,
     '1996': 156,
     'off': 157,
     'cup': 158,
     'five': 159,
     'win': 160,
     'foreign': 161,
     '11': 162,
     '1996-08-27': 163,
     'open': 164,
     'day': 165,
     'down': 166,
     'august': 167,
     '13': 168,
     'united': 169,
     'because': 170,
     '6-4': 171,
     '6-3': 172,
     'official': 173,
     'did': 174,
     '21': 175,
     'just': 176,
     'next': 177,
     '15': 178,
     'him': 179,
     'spain': 180,
     'standings': 181,
     '1996-08-26': 182,
     'expected': 183,
     'shares': 184,
     'do': 185,
     'statement': 186,
     'spokesman': 187,
     'news': 188,
     'pakistan': 189,
     'through': 190,
     'may': 191,
     'women': 192,
     'made': 193,
     '70': 194,
     'v': 195,
     'while': 196,
     'men': 197,
     '12': 198,
     'chicago': 199,
     '1996-08-23': 200,
     'international': 201,
     'them': 202,
     '14': 203,
     'lost': 204,
     'german': 205,
     'july': 206,
     'where': 207,
     'russian': 208,
     'back': 209,
     '20': 210,
     '6-2': 211,
     'september': 212,
     "n't": 213,
     'china': 214,
     'italy': 215,
     'british': 216,
     '2.': 217,
     '3.': 218,
     'european': 219,
     'any': 220,
     'peace': 221,
     'team': 222,
     '1.': 223,
     'clinton': 224,
     'pct': 225,
     'matches': 226,
     'japan': 227,
     'seconds': 228,
     'prime': 229,
     'reported': 230,
     'billion': 231,
     '69': 232,
     'month': 233,
     'earlier': 234,
     'june': 235,
     'central': 236,
     'now': 237,
     'russia': 238,
     'meeting': 239,
     'still': 240,
     'metres': 241,
     'final': 242,
     '30': 243,
     'being': 244,
     'talks': 245,
     'west': 246,
     'added': 247,
     '71': 248,
     '1996-08-25': 249,
     'half': 250,
     '1/2': 251,
     'during': 252,
     'french': 253,
     'season': 254,
     'b': 255,
     'you': 256,
     'set': 257,
     'capital': 258,
     'leading': 259,
     'san': 260,
     'take': 261,
     'tonnes': 262,
     'sweden': 263,
     'killed': 264,
     'st': 265,
     'india': 266,
     'lead': 267,
     'held': 268,
     'elections': 269,
     'net': 270,
     'around': 271,
     'end': 272,
     'war': 273,
     'tennis': 274,
     'security': 275,
     'should': 276,
     'reuters': 277,
     'our': 278,
     'most': 279,
     'part': 280,
     'mark': 281,
     'game': 282,
     'due': 283,
     'early': 284,
     'took': 285,
     'states': 286,
     'days': 287,
     'so': 288,
     'innings': 289,
     'ministry': 290,
     'saying': 291,
     'number': 292,
     'moscow': 293,
     '68': 294,
     'cricket': 295,
     '6-1': 296,
     'american': 297,
     'major': 298,
     'minutes': 299,
     'union': 300,
     'political': 301,
     'seven': 302,
     'per': 303,
     '7-6': 304,
     'netherlands': 305,
     '100': 306,
     'republic': 307,
     'hong': 308,
     'can': 309,
     'north': 310,
     'victory': 311,
     'championship': 312,
     'what': 313,
     'both': 314,
     '22': 315,
     'well': 316,
     'belgium': 317,
     'total': 318,
     'country': 319,
     'iraq': 320,
     'play': 321,
     'court': 322,
     'kong': 323,
     '50': 324,
     'close': 325,
     'africa': 326,
     'czech': 327,
     'visit': 328,
     'dutch': 329,
     '25': 330,
     'result': 331,
     'champion': 332,
     'profit': 333,
     'commission': 334,
     'general': 335,
     'israel': 336,
     'left': 337,
     'eight': 338,
     'local': 339,
     'minute': 340,
     'says': 341,
     'price': 342,
     'trade': 343,
     'town': 344,
     'paris': 345,
     '66': 346,
     'run': 347,
     '1996-08-24': 348,
     'sales': 349,
     '4.': 350,
     'very': 351,
     'press': 352,
     '67': 353,
     'vs.': 354,
     '5.': 355,
     '6.': 356,
     'support': 357,
     'put': 358,
     '1995': 359,
     '24': 360,
     'leader': 361,
     'prices': 362,
     '7-5': 363,
     'then': 364,
     'found': 365,
     'record': 366,
     'go': 367,
     'same': 368,
     'man': 369,
     'western': 370,
     'de': 371,
     'english': 372,
     'washington': 373,
     'called': 374,
     'northern': 375,
     'inc': 376,
     'say': 377,
     'issue': 378,
     'agency': 379,
     'test': 380,
     'opposition': 381,
     'meet': 382,
     'months': 383,
     'military': 384,
     'runs': 385,
     'rate': 386,
     'top': 387,
     'these': 388,
     'lower': 389,
     'ago': 390,
     'singles': 391,
     '64': 392,
     'make': 393,
     'amsterdam': 394,
     '72': 395,
     'race': 396,
     'newspaper': 397,
     'deal': 398,
     'goals': 399,
     '16': 400,
     'office': 401,
     'democratic': 402,
     'ended': 403,
     '60': 404,
     'cents': 405,
     'until': 406,
     'gave': 407,
     'following': 408,
     'leaders': 409,
     'behind': 410,
     'series': 411,
     'another': 412,
     'tour': 413,
     'sri': 414,
     'several': 415,
     'chief': 416,
     'late': 417,
     'australian': 418,
     'good': 419,
     'african': 420,
     'rebels': 421,
     'near': 422,
     'dollar': 423,
     'players': 424,
     '75': 425,
     'michael': 426,
     'my': 427,
     'report': 428,
     'iraqi': 429,
     'weekend': 430,
     'economic': 431,
     'start': 432,
     'halftime': 433,
     'tournament': 434,
     'c': 435,
     'miles': 436,
     'refugees': 437,
     'agreed': 438,
     'southern': 439,
     'many': 440,
     '74': 441,
     'get': 442,
     'power': 443,
     '7.': 444,
     'plan': 445,
     'going': 446,
     'service': 447,
     'east': 448,
     'those': 449,
     'election': 450,
     '73': 451,
     'attendance': 452,
     ';': 453,
     'sydney': 454,
     'drawn': 455,
     'place': 456,
     'date': 457,
     'want': 458,
     'business': 459,
     'paul': 460,
     'white': 461,
     'stock': 462,
     'john': 463,
     'television': 464,
     'demand': 465,
     'israeli': 466,
     'david': 467,
     'gmt': 468,
     'taking': 469,
     'baseball': 470,
     'atlanta': 471,
     'grand': 472,
     'including': 473,
     '8.': 474,
     'index': 475,
     'arrested': 476,
     'authorities': 477,
     'quoted': 478,
     'later': 479,
     'palestinian': 480,
     'parliament': 481,
     'corp': 482,
     'ahmed': 483,
     'allowed': 484,
     'forces': 485,
     'asked': 486,
     'cash': 487,
     'hit': 488,
     'km': 489,
     'california': 490,
     'zealand': 491,
     'brazil': 492,
     'house': 493,
     'already': 494,
     'army': 495,
     'arafat': 496,
     'rights': 497,
     'martin': 498,
     'club': 499,
     '17': 500,
     '28': 501,
     'yeltsin': 502,
     'fell': 503,
     'toronto': 504,
     'current': 505,
     'return': 506,
     'work': 507,
     'higher': 508,
     'ahead': 509,
     'loss': 510,
     '31': 511,
     'weeks': 512,
     'm.': 513,
     '1-0': 514,
     '26': 515,
     'exchange': 516,
     'trading': 517,
     'dole': 518,
     'whether': 519,
     'hours': 520,
     'rose': 521,
     'quarter': 522,
     'came': 523,
     'reporters': 524,
     'announced': 525,
     'jerusalem': 526,
     'troops': 527,
     'share': 528,
     'closed': 529,
     'finland': 530,
     'way': 531,
     'squad': 532,
     'draw': 533,
     'vs': 534,
     'hospital': 535,
     'agreement': 536,
     'money': 537,
     '19': 538,
     'countries': 539,
     '...': 540,
     'like': 541,
     '23': 542,
     'night': 543,
     'oil': 544,
     'head': 545,
     'austria': 546,
     '1994': 547,
     'best': 548,
     'baltimore': 549,
     '18': 550,
     '65': 551,
     'conference': 552,
     'financial': 553,
     'began': 554,
     'scores': 555,
     'away': 556,
     'olympic': 557,
     'went': 558,
     'lanka': 559,
     'morning': 560,
     'decision': 561,
     'high': 562,
     'area': 563,
     'old': 564,
     'plans': 565,
     'bonds': 566,
     'such': 567,
     'aug': 568,
     'previous': 569,
     'main': 570,
     'few': 571,
     '1997': 572,
     'past': 573,
     'francisco': 574,
     'los': 575,
     'hits': 576,
     'nations': 577,
     'budget': 578,
     'lebed': 579,
     'taken': 580,
     'human': 581,
     'public': 582,
     'us': 583,
     'fighting': 584,
     'fourth': 585,
     'little': 586,
     'champions': 587,
     'eastern': 588,
     'angeles': 589,
     'nine': 590,
     'latest': 591,
     'gold': 592,
     'italian': 593,
     'van': 594,
     'ban': 595,
     'long': 596,
     'march': 597,
     'growth': 598,
     'campaign': 599,
     'side': 600,
     'u.n.': 601,
     'strike': 602,
     '63': 603,
     'fall': 604,
     'scored': 605,
     'signed': 606,
     'canada': 607,
     'seattle': 608,
     'colorado': 609,
     'co': 610,
     'much': 611,
     'bill': 612,
     'europe': 613,
     'ukraine': 614,
     'daily': 615,
     'manager': 616,
     'law': 617,
     'working': 618,
     'control': 619,
     'region': 620,
     'markets': 621,
     'future': 622,
     'winning': 623,
     'boston': 624,
     '40': 625,
     'times': 626,
     'recent': 627,
     'think': 628,
     'call': 629,
     'health': 630,
     'accused': 631,
     'right': 632,
     'hold': 633,
     'pay': 634,
     'without': 635,
     'attack': 636,
     'own': 637,
     'plane': 638,
     'available': 639,
     '96': 640,
     'akram': 641,
     'wickets': 642,
     '62': 643,
     'order': 644,
     'airport': 645,
     'workers': 646,
     'again': 647,
     'disease': 648,
     'led': 649,
     'died': 650,
     '27': 651,
     'period': 652,
     'give': 653,
     'free': 654,
     'average': 655,
     'interest': 656,
     'industry': 657,
     'members': 658,
     'ireland': 659,
     'texas': 660,
     'ajax': 661,
     '9.': 662,
     'production': 663,
     'forecast': 664,
     'seen': 665,
     'vote': 666,
     '---': 667,
     'might': 668,
     'sent': 669,
     'despite': 670,
     'embassy': 671,
     'ceasefire': 672,
     'failed': 673,
     'help': 674,
     'does': 675,
     'centre': 676,
     'released': 677,
     'louis': 678,
     'overs': 679,
     'case': 680,
     "'re": 681,
     'dutroux': 682,
     'further': 683,
     'across': 684,
     'florida': 685,
     'island': 686,
     'short': 687,
     'me': 688,
     'october': 689,
     'started': 690,
     'county': 691,
     'wasim': 692,
     'captain': 693,
     'costa': 694,
     'cut': 695,
     'belgian': 696,
     '29': 697,
     'children': 698,
     'death': 699,
     'least': 700,
     'council': 701,
     'rugby': 702,
     'black': 703,
     'mother': 704,
     'kurdish': 705,
     'jordan': 706,
     'planned': 707,
     'stories': 708,
     'analysts': 709,
     'strong': 710,
     'april': 711,
     'given': 712,
     'romania': 713,
     'premier': 714,
     '59': 715,
     'thomas': 716,
     'coach': 717,
     'minnesota': 718,
     'kansas': 719,
     'diego': 720,
     'declared': 721,
     'playing': 722,
     'chechnya': 723,
     'convention': 724,
     'wheat': 725,
     'brussels': 726,
     'yr': 727,
     'rates': 728,
     'according': 729,
     'slovakia': 730,
     'title': 731,
     '54': 732,
     'manchester': 733,
     'houston': 734,
     'shot': 735,
     'fifth': 736,
     '10.': 737,
     'yet': 738,
     'civil': 739,
     'confirmed': 740,
     'bosnia': 741,
     'bosnian': 742,
     'republican': 743,
     'reports': 744,
     'contract': 745,
     'chairman': 746,
     'nearly': 747,
     'used': 748,
     'figures': 749,
     'secretary': 750,
     'force': 751,
     'term': 752,
     'must': 753,
     'trying': 754,
     'today': 755,
     'poland': 756,
     'possible': 757,
     'dealers': 758,
     'forced': 759,
     'golf': 760,
     'a.': 761,
     'tabulate': 762,
     'detroit': 763,
     'oakland': 764,
     'each': 765,
     'companies': 766,
     'seed': 767,
     'point': 768,
     'come': 769,
     'passengers': 770,
     'moslem': 771,
     'treaty': 772,
     'ltd': 773,
     'peter': 774,
     'action': 775,
     'known': 776,
     'brought': 777,
     'taiwan': 778,
     'opening': 779,
     'federal': 780,
     'sale': 781,
     'face': 782,
     'received': 783,
     'estimated': 784,
     'guerrillas': 785,
     'exports': 786,
     'aggregate': 787,
     'named': 788,
     '2-0': 789,
     'robert': 790,
     'injured': 791,
     'road': 792,
     'woman': 793,
     'clear': 794,
     'small': 795,
     'charges': 796,
     'mexico': 797,
     'traders': 798,
     'kenya': 799,
     'nigeria': 800,
     'committee': 801,
     'senior': 802,
     'how': 803,
     'radio': 804,
     'details': 805,
     'association': 806,
     'straight': 807,
     'full': 808,
     'turkey': 809,
     'red': 810,
     'likely': 811,
     'earnings': 812,
     'immediately': 813,
     'services': 814,
     'use': 815,
     'l': 816,
     'wimbledon': 817,
     'argentina': 818,
     'cleveland': 819,
     '1,000': 820,
     '*': 821,
     'note': 822,
     'soon': 823,
     'post': 824,
     'investors': 825,
     'showed': 826,
     'private': 827,
     'department': 828,
     'yen': 829,
     'japanese': 830,
     'train': 831,
     '48': 832,
     'bond': 833,
     'egypt': 834,
     'iran': 835,
     'level': 836,
     'air': 837,
     'spot': 838,
     'armed': 839,
     '76': 840,
     'rally': 841,
     'unless': 842,
     'prix': 843,
     'better': 844,
     'croft': 845,
     '0-0': 846,
     'cincinnati': 847,
     '61': 848,
     'montreal': 849,
     'philadelphia': 850,
     'got': 851,
     'medical': 852,
     'groups': 853,
     'although': 854,
     'trip': 855,
     'stage': 856,
     'process': 857,
     'policy': 858,
     'see': 859,
     'conditions': 860,
     'believed': 861,
     'bonn': 862,
     'sold': 863,
     'tax': 864,
     'port': 865,
     'coming': 866,
     'board': 867,
     'seeding': 868,
     'holiday': 869,
     'waqar': 870,
     'younis': 871,
     'mushtaq': 872,
     'qualifier': 873,
     'johnson': 874,
     'milwaukee': 875,
     'young': 876,
     'psv': 877,
     'defence': 878,
     'swiss': 879,
     'violence': 880,
     'even': 881,
     'school': 882,
     'university': 883,
     'prison': 884,
     'information': 885,
     'rise': 886,
     'independence': 887,
     'pound': 888,
     'serb': 889,
     '6-0': 890,
     'sell': 891,
     'here': 892,
     'message': 893,
     'baghdad': 894,
     'border': 895,
     'scheduled': 896,
     'data': 897,
     'keep': 898,
     'winner': 899,
     's.': 900,
     'rule': 901,
     'within': 902,
     'homer': 903,
     'jose': 904,
     'making': 905,
     'system': 906,
     'levels': 907,
     '56': 908,
     'municipal': 909,
     'deputy': 910,
     'comment': 911,
     'nuclear': 912,
     'change': 913,
     'buy': 914,
     'polish': 915,
     'based': 916,
     'met': 917,
     'rebel': 918,
     'continue': 919,
     'land': 920,
     'saudi': 921,
     'flight': 922,
     'street': 923,
     '53': 924,
     'know': 925,
     'course': 926,
     'scorers': 927,
     'director': 928,
     'khan': 929,
     'mullally': 930,
     'great': 931,
     'canadian': 932,
     'switzerland': 933,
     'pittsburgh': 934,
     'inning': 935,
     'tried': 936,
     '55': 937,
     'letter': 938,
     'grozny': 939,
     'completed': 940,
     'charged': 941,
     'wife': 942,
     'volume': 943,
     'illegal': 944,
     'position': 945,
     'indian': 946,
     'economy': 947,
     'reached': 948,
     'struck': 949,
     'bid': 950,
     'king': 951,
     'chinese': 952,
     'goal': 953,
     'talk': 954,
     'netanyahu': 955,
     'source': 956,
     'kept': 957,
     'movement': 958,
     'areas': 959,
     'outside': 960,
     'puk': 961,
     'parties': 962,
     'attacks': 963,
     'problem': 964,
     'turkish': 965,
     'step': 966,
     '45': 967,
     '2-1': 968,
     '1-1': 969,
     'prefix': 970,
     'opened': 971,
     'tie': 972,
     'percentage': 973,
     '58': 974,
     'ground': 975,
     'stocks': 976,
     'able': 977,
     'deficit': 978,
     'arrived': 979,
     "'m": 980,
     'poor': 981,
     'village': 982,
     'coast': 983,
     'osce': 984,
     'leave': 985,
     'having': 986,
     'drug': 987,
     'lot': 988,
     'eu': 989,
     'needed': 990,
     'similar': 991,
     'denied': 992,
     'paper': 993,
     'beijing': 994,
     'province': 995,
     'executive': 996,
     'compared': 997,
     '300': 998,
     'schedule': 999,
     'firm': 1000,
     'digest': 1001,
     ...}



향후 훈련 데이터의 길이를 모두 맞추기위해 인덱스 0에는 'PAD'라는 단어를 넣고, 인덱스 1에는 단어 집합에 없는 단어들을 별도로 표시하기 위한 'OOV'라는 단어를 부여합니다.

해당 단어 집합은 단어의 등장 빈도 순위에 따라서 인덱스를 부여합니다. word_to_index의 인덱스 0에 'PAD'와 인덱스 1에 'OOV'도 추가된 상태입니다. 인덱스 2부터는 등장 빈도수가 가장 높은 단어 순서대로 부여되는데, 등장 빈도수가 가장 높은 the의 경우에는 인덱스 2에 할당됩니다.

해당 단어 집합의 크기를 확인해보겠습니다.


```python
print(len(word_to_index))
```

    3939
    

약 21,000개에 달했던 단어 집합이 빈도수 5이하인 단어들을 배제시키자, 3939개의 단어만을 가진 단어 집합으로 단어의 개수가 대폭 줄어든 것을 확인할 수 있습니다.

이제 word_to_index를 통해 단어를 입력하면, 인덱스를 리턴받을 수 있습니다.


```python
word_to_index['the']
```




    2



 입력받은 개체명 태깅에 대해서 인덱스를 리턴하는 ner_to_index를 만들어보도록 하겠습니다.


```python
ner_to_index={}
ner_to_index['PAD'] = 0
i=1
for ner in ner_set:
    ner_to_index[ner]=i
    i=i+1
print(ner_to_index)
```

    {'PAD': 0, 'B-ORG': 1, 'I-MISC': 2, 'I-LOC': 3, 'I-PER': 4, 'B-LOC': 5, 'O': 6, 'B-PER': 7, 'I-ORG': 8, 'B-MISC': 9}
    

이제 ner_to_index에다가 개체명 태깅을 입력하면 인덱스를 리턴받을 수 있습니다.


```python
ner_to_index['I-PER']
```




    4



이제 훈련 데이터를 정수 인코딩할 준비가 끝났습니다. 

이제 모든 훈련 데이터를 담고 있는 sentences로 부터 word_to_index와 ner_to_index를 통해 모든 훈련 데이터를 숫자로 바꿉니다.

우선 word_to_index를 사용하여 단어에 대해서 훈련 데이터인 data_X를 만듭니다.


```python
data_X = []

for s in sentences:
    temp_X = []
    for w, label in s:
        try:
            temp_X.append(word_to_index.get(w,1))
        except KeyError:
            temp_X.append(word_to_index['OOV'])

    data_X.append(temp_X)
print(data_X[:10])
```

    [[989, 1, 205, 629, 7, 1, 216, 1, 3], [774, 1872], [726, 150], [2, 219, 334, 14, 13, 70, 28, 1, 24, 205, 1, 7, 2404, 7, 1, 216, 1, 406, 3382, 2009, 519, 1745, 1873, 648, 309, 41, 1, 7, 1632, 3], [124, 15, 2991, 7, 2, 219, 300, 15, 2660, 801, 1, 1, 14, 13, 75, 2404, 276, 914, 1, 27, 539, 127, 125, 137, 406, 2, 2405, 1, 20, 1, 3], [12, 69, 185, 213, 357, 220, 567, 1, 170, 69, 185, 213, 859, 220, 3383, 16, 28, 4, 12, 2, 334, 15, 416, 187, 1, 594, 2010, 1, 90, 8, 188, 3384, 3], [26, 14, 683, 2405, 2181, 20, 2661, 9, 141, 28, 20, 365, 25, 775, 20, 990, 28, 276, 41, 580, 29, 2, 219, 300, 3], [26, 14, 8, 2182, 66, 233, 29, 989, 1129, 2406, 1, 3385, 7, 595, 1632, 1, 4, 1, 9, 1, 1, 27, 2, 581, 9, 2662, 1029, 3386, 20, 8, 3387, 2992, 9, 1, 1301, 7, 1, 581, 630, 3], [3385, 1544, 1, 1462, 42, 744, 27, 137, 9, 146, 25, 104, 1, 860, 1632, 126, 745, 1, 1, 1, 10, 2011, 11, 54, 1745, 1873, 648, 3], [37, 3385, 438, 7, 2993, 35, 2182, 42, 2, 989, 15, 2407, 2660, 801, 4, 1, 2662, 630, 138, 4, 1874, 141, 567, 775, 20, 1, 32, 100, 20, 143, 8, 1633, 1229, 7, 581, 630, 3]]
    

정상적으로 변환이 되었는지 보기 위해서 첫번째 샘플에 대해서만 기존의 단어 시퀀스를 출력해보겠습니다.


```python
index_to_word={}
for key, value in word_to_index.items(): # 인덱스를 단어로 바꾸기 위해 index_to_word를 생성
    index_to_word[value] = key


temp = []
for index in data_X[0] : # 첫번째 샘플 안의 인덱스들에 대해서
    temp.append(index_to_word[index]) # 다시 단어로 변환

print(sentences[0])    
print(temp)
```

    [['eu', 'B-ORG'], ['rejects', 'O'], ['german', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['british', 'B-MISC'], ['lamb', 'O'], ['.', 'O']]
    ['eu', 'OOV', 'german', 'call', 'to', 'OOV', 'british', 'OOV', '.']
    

 rejects와 boycott, lamb는 사라지고 'OOV'로 바뀌었는데, 이는 앞에서 빈도수가 5 이하인 단어들에 대해서는 단어 집합에 해당되는 word_to_index에 저장시키지 않았고 이에 따라 모르는 단어 'OOV'로 간주되었기 때문입니다.

 이제 모든 훈련 데이터를 담고 있는 sentences에서 개체명 태깅에 해당되는 부분을 모아 data_y에 저장하는 작업을 수행해보도록 하겠습니다.


```python
data_y = []

for s in sentences:
    temp_y = []
    for w, label in s:
            temp_y.append(ner_to_index.get(label))

    data_y.append(temp_y)
```


```python
print(data_X[:4]) # X 데이터 4개만 출력
print(data_y[:4]) # y 데이터 4개만 출력
```

    [[989, 1, 205, 629, 7, 1, 216, 1, 3], [774, 1872], [726, 150], [2, 219, 334, 14, 13, 70, 28, 1, 24, 205, 1, 7, 2404, 7, 1, 216, 1, 406, 3382, 2009, 519, 1745, 1873, 648, 309, 41, 1, 7, 1632, 3]]
    [[1, 6, 9, 6, 6, 6, 9, 6, 6], [7, 4], [5, 6], [6, 1, 8, 6, 6, 6, 6, 6, 6, 9, 6, 6, 6, 6, 6, 9, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]]
    

이제 단어에 대한 훈련 데이터인 data_X와 개체명 태깅에 대한 훈련 데이터인 data_y가 만들어졌습니다.

그런데 data_X의 단어 시퀀스 데이터 각각과 data_y 단어 시퀀스 데이터 각각은 전부 길이가 천차만별입니다.  

각 샘플의 길이가 대체적으로 어떻게 되는지 시각화해보도록 하겠습니다.


```python
%matplotlib inline
import matplotlib.pyplot as plt

plt.hist([len(s) for s in data_X], bins=50)
plt.xlabel('length of Data')
plt.ylabel('number of Data')
plt.show()
```


![png](NLP_basic_11_Tagging_Task_files/NLP_basic_11_Tagging_Task_98_0.png)


위의 그래프는 샘플들의 길이가 대체적으로 0~40의 길이를 가지며, 특히 0~20의 길이를 가진 샘플이 상당한 비율을 차지하는 것을 보여줍니다.

양방향 LSTM 모델에 손쉽게 데이터를 입력으로 사용하기 위해서, 여기서는 모든 샘플의 길이를 동일하게 맞추도록 하겠습니다.

가장 길이가 긴 샘플의 길이를 우선 구해보도록 하겠습니다.


```python
print(max(len(l) for l in data_X)) # 전체 데이터에서 길이가 가장 긴 샘플의 길이 출력
print(max(len(l) for l in data_y)) # 전체 데이터에서 길이가 가장 긴 샘플의 길이 출력
```

    113
    113
    

길이가 가장 긴 샘플의 길이는 113입니다. X에 해당되는 데이터 data_X와 y에 해당되는 데이터 data_y의 모든 샘플의 길이를 113으로 맞출 수도 있겠지만, 앞서 그래프로 봤듯이 대부분의 샘플들의 길이는 40~60에 편중되어 있습니다. 이번에는 임의의 숫자 70에 맞추어서 모든 데이터를 패딩하겠습니다.


```python
max_len=70
from keras.preprocessing.sequence import pad_sequences
pad_X = pad_sequences(data_X, padding='post', maxlen=max_len)
# data_X의 모든 샘플들의 길이를 맞출 때 뒤의 공간에 숫자 0으로 채움.
pad_y = pad_sequences(data_y, padding='post', maxlen=max_len)
# data_y의 모든 샘플들의 길이를 맞출 때 뒤의 공간에 숫자0으로 채움.
```

    Using TensorFlow backend.
    

data_X와 data_y의 전체 데이터에서 길이가 70보다 작은 샘플들은 길이를 맞추기 위해 남는 공간에 숫자 0이 들어갑니다. 숫자 0은 앞서 단어 집합 word_to_index와 ner_to_index에서 임의로 패딩을 위해 만들어준 단어 'PAD'에 해당됩니다.


```python
print(min(len(l) for l in pad_X)) # 모든 데이터에서 길이가 가장 짧은 샘플의 길이 출력
print(min(len(l) for l in pad_y)) # 모든 데이터에서 길이가 가장 짧은 샘플의 길이 출력
```

    70
    70
    

pad_X와 pad_y를 훈련 데이터와 테스트 데이터로 8:2로 분할합니다.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(pad_X, pad_y, test_size=.2, random_state=777)
```

random_state를 지정했기 때문에 기존의 pad_X와 pad_y에서 순서가 섞이면서 훈련 데이터와 테스트 데이터로 분할됩니다. 즉, 이제 X_train의 첫번째 샘플과 pad_X에서의 첫번째 샘플은 서로 다른 샘플일 수 있습니다.


```python
print(len(X_train), len(X_test), len(y_train), len(y_test))
```

    11232 2809 11232 2809
    


```python
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, TimeDistributed
from keras.optimizers import Adam
```

모델 설계를 하겠습니다.


```python
n_words = len(word_to_index)
n_labels = len(ner_to_index)
```


```python
model = Sequential()
model.add(Embedding(input_dim=n_words, output_dim=16, input_length=max_len, mask_zero=True))
model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(TimeDistributed(Dense(n_labels, activation='softmax')))
```

Many-to-Many 문제이므로 LSTM()에 return_sequences=True를 설정해준 것을 볼 수 있습니다. 

이번 실습과 같이 각 데이터의 길이가 달라서 패딩을 하느라 숫자 0이 많아질 경우에는 Embedding()에 mask_zero=True를 설정하여 데이터에서 숫자 0은 패딩을 의미하므로 연산에서 제외시킨다는 옵션을 줄 수 있습니다.


```python
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
```

훈련 데이터에 대해서 원-핫 인코딩을 진행합니다.


```python
from keras.utils import np_utils
y_train2 = np_utils.to_categorical(y_train)
y_train2[0][0]
```




    array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)




```python
model.fit(X_train, y_train2, epochs=2) #원래는 예제에서 epochs=8 이지만 시간단축을 위해 epochs=2로 설정 
```

    Epoch 1/2
    11232/11232 [==============================] - 119s 11ms/step - loss: 0.7041 - acc: 0.8312
    Epoch 2/2
    11232/11232 [==============================] - 120s 11ms/step - loss: 0.4444 - acc: 0.8601
    




    <keras.callbacks.History at 0x1c2bfa21be0>




```python
y_test2 = np_utils.to_categorical(y_test)
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test2)[1]))
```

    2809/2809 [==============================] - 11s 4ms/step
    
     테스트 정확도: 0.8771
    

실제로 맞추고 있는지를 테스트 데이터를 주고 직접 실제값과 비교해보도록 하겠습니다.


```python
import numpy as np

index_to_word={}
for key, value in word_to_index.items():
    index_to_word[value] = key

index_to_ner={}
for key, value in ner_to_index.items():
    index_to_ner[value] = key


i=10 # 확인하고 싶은 테스트용 샘플의 인덱스.
y_predicted = model.predict(np.array([X_test[i]])) # 입력한 테스트용 샘플에 대해서 예측 y를 리턴
y_predicted = np.argmax(y_predicted, axis=-1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.
true = np.argmax(y_test2[i], -1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.

print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
print(35 * "-")

for w, t, pred in zip(X_test[i], true, y_predicted[0]):
    if w != 0: # PAD값은 제외함.
        print("{:17}: {:7} {}".format(index_to_word[w], index_to_ner[t], index_to_ner[pred]))
```

    단어             |실제값  |예측값
    -----------------------------------
    sarah            : B-PER   B-PER
    brady            : I-PER   B-LOC
    ,                : O       O
    whose            : O       O
    republican       : B-MISC  B-LOC
    husband          : O       O
    was              : O       O
    OOV              : O       O
    OOV              : O       O
    in               : O       O
    an               : O       O
    OOV              : O       O
    attempt          : O       O
    on               : O       O
    president        : O       O
    ronald           : B-PER   B-LOC
    reagan           : I-PER   I-ORG
    ,                : O       O
    took             : O       O
    centre           : O       O
    stage            : O       O
    at               : O       O
    the              : O       O
    democratic       : B-MISC  B-LOC
    national         : I-MISC  I-ORG
    convention       : I-MISC  O
    on               : O       O
    monday           : O       O
    night            : O       O
    to               : O       O
    OOV              : O       O
    president        : O       O
    bill             : B-PER   O
    clinton          : I-PER   I-PER
    's               : O       O
    gun              : O       O
    control          : O       O
    efforts          : O       O
    .                : O       O
    

위의 결과는 정확도가 높지 않지만 epochs = 8로 한다면 정확도 높은 결과를 볼 수 있다.

하지만 출력 결과는 그럴듯해 보이지만 사실 이번에 사용한 정확도 측정 방법이 그다지 적절하지는 않았습니다.

그 이유에 대해서는 뒤에서 나오는 양방향 LSTM과 CRF에서 찾을 수 있다. 

## 4. 양방향 LSTM을 이용한 품사 태깅(Part-of-speech Tagging using Bi-LSTM)

직접 양방향 LSTM을 이용한 품사 태깅을 수행하는 모델을 만들기

### 1) 양방향 LSTM(Bi-directional LSTM)으로 POS Tagger 만들기

 NLTK를 이용하면 영어 코퍼스에 토큰화와 품사 태깅 전처리를 진행한 총 3,914개의 문장 데이터를 받아올 수 있습니다.


```python
import nltk
tagged_sentences = nltk.corpus.treebank.tagged_sents() # 토큰화에 품사 태깅이 된 데이터 받아오기
# 미설치 에러 발생시 nltk.download('treebank')
print(tagged_sentences[0]) # 첫번째 문장 샘플 출력
print("품사 태깅이 된 문장 개수: ", len(tagged_sentences)) # 문장 샘플의 개수 출력
```

    [('Pierre', 'NNP'), ('Vinken', 'NNP'), (',', ','), ('61', 'CD'), ('years', 'NNS'), ('old', 'JJ'), (',', ','), ('will', 'MD'), ('join', 'VB'), ('the', 'DT'), ('board', 'NN'), ('as', 'IN'), ('a', 'DT'), ('nonexecutive', 'JJ'), ('director', 'NN'), ('Nov.', 'NNP'), ('29', 'CD'), ('.', '.')]
    품사 태깅이 된 문장 개수:  3914
    

품사 태깅 전처리가 수행된 첫번째 문장이 출력된 것을 볼 수 있습니다. 이러한 문장 샘플이 총 3,914개가 있습니다.

그런데 훈련을 시키려면 훈련 데이터에서 단어에 해당되는 부분과 품사 태깅 정보에 해당되는 부분을 분리시켜야 합니다. 즉, [('Pierre', 'NNP'), ('Vinken', 'NNP')]와 같은 문장 샘플이 있다면 Pierre과 Vinken을 같이 저장하고, NNP와 NNP를 같이 저장할 필요가 있습니다.

이런 경우 파이썬 함수 중에서 zip()함수가 유용한 역할을 합니다.


```python
import numpy as np

sentences, pos_tags =[], [] 
for tagged_sentence in tagged_sentences: # 3,914개의 문장 샘플을 1개씩 불러온다.
    sentence, tag_info = zip(*tagged_sentence) # 각 샘플에서 단어는 sentence에 품사 태깅 정보는 tags에 저장한다.
    sentences.append(np.array(sentence)) # 각 샘플에서 단어 정보만 저장한다.
    pos_tags.append(np.array(tag_info)) # 각 샘플에서 품사 태깅 정보만 저장한다.
```

각 문장 샘플에 대해서 단어는 sentences에 태깅 정보는 pos_tags에 저장하였습니다.


```python
print(sentences[0])
print(pos_tags[0])
```

    ['Pierre' 'Vinken' ',' '61' 'years' 'old' ',' 'will' 'join' 'the' 'board'
     'as' 'a' 'nonexecutive' 'director' 'Nov.' '29' '.']
    ['NNP' 'NNP' ',' 'CD' 'NNS' 'JJ' ',' 'MD' 'VB' 'DT' 'NN' 'IN' 'DT' 'JJ'
     'NN' 'NNP' 'CD' '.']
    

sentences는 예측을 위한 X에 해당되며 pos_tags는 예측 대상인 y에 해당됩니다.


```python
%matplotlib inline
import matplotlib.pyplot as plt

plt.hist([len(s) for s in sentences], bins=50)
plt.xlabel('length of Data')
plt.ylabel('number of Data')
plt.show()
```


![png](NLP_basic_11_Tagging_Task_files/NLP_basic_11_Tagging_Task_139_0.png)


위의 그래프는 샘플의 길이가 대부분 0~50의 길이를 가지는 것을 보여줍니다.

훈련 데이터에서 sentences를 X로, pos_tags를 y로 하여 X와 y를 분리한 셈입니다. 이제 X에 대한 단어 집합과 y에 대한 단어 집합을 만들어보겠습니다.


```python
from collections import Counter
vocab=Counter()
tag_set=set()

for sentence in sentences: # 훈련 데이터 X에서 문장 샘플을 1개씩 꺼내온다.
  for word in sentence: # 샘플에서 단어를 1개씩 꺼내온다.
    vocab[word.lower()]=vocab[word.lower()]+1 # 각 단어의 빈도수를 카운트한다.

for tags_list in pos_tags: # 훈련 데이터 y에서 품사 태깅 정보 샘플을 1개씩 꺼내온다.
  for tag in tags_list: # 샘플에서 품사 태깅 정보를 1개씩 꺼내온다.
    tag_set.add(tag) # 각 품사 태깅 정보에 대해서 중복을 허용하지 않고 집합을 만든다.
```


```python
print(len(vocab)) # X 데이터의 단어 집합의 길이 출력
print(len(tag_set)) # y 데이터의 단어 집합의 길이 출력
```

    11387
    46
    

각각의 길이를 출력하므로서 단어의 개수는 11,387개. 품사 태깅 정보의 개수는 46개임을 알 수 있습니다.

이제는 각 단어에 대해서 인덱스를 부여하고, 각 품사 태깅 정보에 대해서도 인덱스를 부여해보겠습니다.


```python
vocab_sorted=sorted(vocab.items(), key=lambda x:x[1], reverse=True)
print(vocab_sorted[:5])
```

    [(',', 4885), ('the', 4764), ('.', 3828), ('of', 2325), ('to', 2182)]
    

단어의 인덱스를 부여하기 위한 기준으로는 빈도수를 사용하겠습니다. 이를 위해 기존에 만든 단어 집합을 빈도수가 큰 순서대로 정렬합니다.

','가 4,885번 등장하여 최다 빈도수를 가진 단어이고, 두번째로는 the가 4,764번 등장하여 두번째로 빈도수가 높은 단어임을 확인할 수 있습니다. 

이제 빈도수가 높은 순서대로 정렬된 단어 집합에 대해서 순차적으로 인덱스를 부여합니다. 단, 이 때 뒤에서 모든 문장 샘플의 길이를 맞추기 위한 'PAD'와 모르는 단어를 의미하는 'OOV'도 넣어줍니다.


```python
word_to_index={'PAD' : 0, 'OOV' :1}
i=1
# 인덱스 0은 각각 입력값들의 길이를 맞추기 위한 PAD(padding을 의미)라는 단어에 사용된다.
# 인덱스 1은 모르는 단어를 의미하는 OOV라는 단어에 사용된다.
for (word, frequency) in vocab_sorted :
    # if frequency > 1 :
    # 빈도수가 1인 단어를 제거하는 것도 가능하겠지만 이번에는 별도 수행하지 않고 해보겠음.
    # 참고로 제거를 수행할 경우 단어 집합의 크기가 절반 정도로 줄어듬.
        i=i+1
        word_to_index[word]=i
print(word_to_index)
print(len(word_to_index))
```

    {'PAD': 0, 'OOV': 1, ',': 2, 'the': 3, '.': 4, 'of': 5, 'to': 6, 'a': 7, 'in': 8, 'and': 9, '*-1': 10, '0': 11, '*': 12, "'s": 13, 'for': 14, 'that': 15, '*t*-1': 16, '*u*': 17, '$': 18, '``': 19, "''": 20, 'is': 21, 'said': 22, 'it': 23, 'on': 24, '%': 25, 'by': 26, 'at': 27, 'as': 28, 'with': 29, 'from': 30, 'million': 31, 'mr.': 32, '*-2': 33, 'are': 34, 'was': 35, 'be': 36, '*t*-2': 37, 'its': 38, 'has': 39, 'an': 40, 'new': 41, 'have': 42, "n't": 43, 'but': 44, 'he': 45, 'or': 46, 'will': 47, 'they': 48, 'company': 49, '--': 50, 'which': 51, 'this': 52, 'u.s.': 53, 'says': 54, 'year': 55, 'about': 56, 'would': 57, 'more': 58, 'were': 59, 'market': 60, 'their': 61, 'than': 62, 'stock': 63, ';': 64, 'who': 65, 'trading': 66, 'had': 67, 'also': 68, 'president': 69, 'billion': 70, 'up': 71, 'one': 72, 'been': 73, 'some': 74, ':': 75, 'other': 76, 'not': 77, 'program': 78, '*-3': 79, 'his': 80, 'because': 81, 'if': 82, 'could': 83, 'share': 84, 'corp.': 85, 'all': 86, 'years': 87, 'i': 88, 'first': 89, 'shares': 90, '-rrb-': 91, 'two': 92, 'any': 93, 'york': 94, '-lrb-': 95, 'last': 96, 'there': 97, 'many': 98, 'no': 99, 'such': 100, 'when': 101, 'she': 102, 'inc.': 103, '*t*-3': 104, 'we': 105, 'can': 106, 'you': 107, 'so': 108, 'japanese': 109, 'after': 110, 'do': 111, 'prices': 112, 'into': 113, 'government': 114, '&': 115, 'business': 116, 'over': 117, 'most': 118, 'only': 119, 'may': 120, 'sales': 121, 'out': 122, 'these': 123, 'even': 124, 'federal': 125, 'say': 126, 'japan': 127, 'make': 128, 'co.': 129, 'under': 130, 'while': 131, 'board': 132, "'": 133, 'index': 134, 'recent': 135, 'big': 136, 'exchange': 137, 'price': 138, 'time': 139, 'what': 140, 'department': 141, 'futures': 142, 'now': 143, '*ich*-1': 144, 'them': 145, 'cents': 146, 'bank': 147, 'group': 148, 'investors': 149, 'stocks': 150, 'funds': 151, 'next': 152, 'executive': 153, 'yesterday': 154, 'american': 155, 'trade': 156, 'companies': 157, 'profit': 158, 'investment': 159, 'much': 160, 'people': 161, 'house': 162, 'like': 163, 'did': 164, 'those': 165, 'rose': 166, 'bonds': 167, 'october': 168, 'common': 169, 'money': 170, 'securities': 171, 'issue': 172, 'months': 173, 'financial': 174, 'mrs.': 175, 'markets': 176, 'net': 177, 'chairman': 178, 'down': 179, 'made': 180, 'week': 181, 'since': 182, 'yen': 183, 'buy': 184, 'her': 185, 'research': 186, 'does': 187, 'take': 188, 'banks': 189, 'interest': 190, 'high': 191, 'three': 192, 'expected': 193, 'higher': 194, 'rates': 195, 'earlier': 196, 'days': 197, 'international': 198, 'chief': 199, '10': 200, 'before': 201, 'officials': 202, 'plan': 203, 'traders': 204, 'own': 205, 'yield': 206, 'report': 207, 'another': 208, 'just': 209, '30': 210, 'debt': 211, 'country': 212, 'offer': 213, 'court': 214, 'get': 215, 'tuesday': 216, 'well': 217, 'congress': 218, 'economic': 219, 'state': 220, '50': 221, 'each': 222, 'small': 223, 'major': 224, 'sell': 225, 'number': 226, '*?*': 227, 'rate': 228, 'vice': 229, 'industry': 230, 'off': 231, '1988': 232, 'among': 233, 'still': 234, 'both': 235, 'might': 236, 'month': 237, 'bush': 238, 'chicago': 239, 'early': 240, 'cash': 241, 'during': 242, '1990': 243, 'growth': 244, '1': 245, 'help': 246, 'income': 247, 'test': 248, 'according': 249, 'average': 250, 'treasury': 251, 'capital': 252, 'good': 253, 'through': 254, 'september': 255, '15': 256, 'concern': 257, 'street': 258, 'fiscal': 259, 'south': 260, 'pay': 261, 'investor': 262, 'law': 263, 'due': 264, 'managers': 265, '1989': 266, 'added': 267, 'less': 268, 'earnings': 269, '100': 270, 'firm': 271, 'part': 272, '?': 273, 'issues': 274, 'sold': 275, 'including': 276, 'use': 277, 'spokesman': 278, 'analysts': 279, 'ago': 280, 'should': 281, '500': 282, 'several': 283, 'reported': 284, 'contract': 285, 'same': 286, 'world': 287, 'wall': 288, 'school': 289, 'yeargin': 290, 'columbia': 291, 'unit': 292, 'where': 293, 'until': 294, 'plans': 295, 'between': 296, 'bid': 297, 'already': 298, 'cray': 299, 'computer': 300, '8': 301, 'right': 302, 'bill': 303, 'quarter': 304, 'few': 305, 'general': 306, 'without': 307, 'however': 308, 'against': 309, 'firms': 310, 'administration': 311, 's&p': 312, 'products': 313, '*-4': 314, '*ich*-2': 315, 'current': 316, 'management': 317, 'then': 318, '*rnr*-1': 319, 'service': 320, 'public': 321, 'john': 322, 'being': 323, 'used': 324, 'paper': 325, 'far': 326, 'city': 327, 'below': 328, 'compared': 329, '*exp*-1': 330, 'notes': 331, 'old': 332, 'director': 333, 'fell': 334, 'began': 335, 'officer': 336, 'loss': 337, '1987': 338, 'national': 339, 'very': 340, 'operations': 341, 'based': 342, 'association': 343, 'corporate': 344, 'end': 345, 'increase': 346, 'lower': 347, 'six': 348, 'construction': 349, 'health': 350, 'students': 351, 'points': 352, 'problem': 353, 'five': 354, 'though': 355, 'news': 356, '2': 357, 'demand': 358, 'way': 359, 'around': 360, 'closed': 361, 'record': 362, 'orders': 363, 'charge': 364, 'your': 365, 'services': 366, 'junk': 367, "'re": 368, 'information': 369, 'whether': 370, 'university': 371, 'large': 372, 'point': 373, 'sale': 374, 'ad': 375, 'past': 376, 'set': 377, 'case': 378, 'want': 379, 'foreign': 380, 'see': 381, 'bond': 382, 'today': 383, 'us': 384, 'took': 385, 'maker': 386, 'return': 387, 'long': 388, 'largest': 389, 'close': 390, 'problems': 391, 'little': 392, 'cut': 393, 'london': 394, 'how': 395, 'safety': 396, '3\\/4': 397, 'although': 398, 'work': 399, 'example': 400, 'currently': 401, 'manager': 402, 'economy': 403, 'offered': 404, 'spending': 405, '1\\/2': 406, 'action': 407, 'finance': 408, 'analyst': 409, 'campbell': 410, 'appropriations': 411, 'our': 412, 'recently': 413, 'total': 414, 'force': 415, 'back': 416, 'future': 417, 'costs': 418, 'move': 419, 'operating': 420, 'purchase': 421, 'power': 422, 'additional': 423, 'priced': 424, 'terms': 425, 'certain': 426, 'offering': 427, 'members': 428, '13': 429, 'union': 430, 'contracts': 431, 'materials': 432, 'commercial': 433, 'california': 434, 'suspension': 435, 'industrial': 436, 'fund': 437, 'ended': 438, 'period': 439, 'despite': 440, 'approved': 441, 'division': 442, '20': 443, 'nation': 444, 'fall': 445, 'fine': 446, 'value': 447, 'come': 448, 'noted': 449, 'possible': 450, 'strong': 451, 'china': 452, 'put': 453, 'often': 454, 'oct.': 455, 'committee': 456, 'low': 457, 'think': 458, 'volume': 459, 'nov.': 460, 'day': 461, 'fees': 462, 'savings': 463, 'manufacturing': 464, 'credit': 465, 'too': 466, 'commission': 467, 'robert': 468, 'local': 469, 'customers': 470, 'dealers': 471, 'least': 472, '40': 473, 'result': 474, 'office': 475, 'stake': 476, 'dividend': 477, 'tokyo': 478, 'family': 479, 'insurance': 480, 'programs': 481, 'ringers': 482, 'arbitrage': 483, 'named': 484, 'england': 485, '*t*-4': 486, 'found': 487, 'almost': 488, 'holding': 489, 'legislation': 490, 'william': 491, 'increased': 492, 'magazine': 493, 'washington': 494, 'become': 495, 'yet': 496, 'judge': 497, '12': 498, 'development': 499, 'addition': 500, 'face': 501, 'half': 502, 'growing': 503, 'oil': 504, 'here': 505, 'voice': 506, 'loan': 507, 'takeover': 508, 'georgia': 509, 'card': 510, 'volatility': 511, 'veto': 512, 'sugar': 513, 'usx': 514, 'workers': 515, 'latest': 516, 'results': 517, 'likely': 518, 'james': 519, 'different': 520, 'countries': 521, 'support': 522, 'agency': 523, 'go': 524, 'acquisition': 525, 'trying': 526, 'gains': 527, 'system': 528, 'proposed': 529, 'banking': 530, 'received': 531, 'estimated': 532, 'march': 533, 'weeks': 534, 'cases': 535, 'ii': 536, 'production': 537, 'level': 538, 'payments': 539, 'private': 540, 'declined': 541, 'change': 542, 'great': 543, 'questions': 544, 'stock-index': 545, 'four': 546, 'plant': 547, 'dividends': 548, 'raise': 549, 'act': 550, 'senior': 551, 'held': 552, 'give': 553, 'full': 554, 'decline': 555, 'edison': 556, 'every': 557, 'came': 558, 'product': 559, 'home': 560, 'outstanding': 561, 'buying': 562, 'rise': 563, 'never': 564, 'proposal': 565, 'funding': 566, 'transactions': 567, 'america': 568, 'wine': 569, '15,000': 570, 'limit': 571, 'trades': 572, 'hahn': 573, 'corn': 574, 'making': 575, 'times': 576, 'place': 577, 'bills': 578, 'continue': 579, 'shareholders': 580, 'transaction': 581, 'official': 582, 'political': 583, '3': 584, 'called': 585, 'paid': 586, 'rights': 587, 'white': 588, 'designed': 589, '7': 590, 'late': 591, 'talks': 592, 'lead': 593, 'going': 594, 'employees': 595, 'deal': 596, 'rep.': 597, 'my': 598, 'dow': 599, 'jones': 600, 'gulf': 601, 'bells': 602, 'former': 603, 'further': 604, 'institutions': 605, 'assets': 606, 'ltd.': 607, 'expects': 608, 'senate': 609, 'decision': 610, 'executives': 611, 'food': 612, 'cars': 613, 'korea': 614, '5': 615, 'drop': 616, 'united': 617, 'seeking': 618, 'commerce': 619, 'order': 620, 'amount': 621, 'previous': 622, 'comment': 623, 'enough': 624, 'restructuring': 625, 'above': 626, 'better': 627, 'soviet': 628, 'must': 629, 'life': 630, 'industries': 631, 'gain': 632, 'pressure': 633, 'wo': 634, 'san': 635, 'inc': 636, 'express': 637, 'publishing': 638, 'form': 639, 'once': 640, 'a.': 641, 'team': 642, '18': 643, 'standard': 644, 'particularly': 645, 'investments': 646, 'august': 647, 'reached': 648, '9': 649, 'dollar': 650, 'thrift': 651, 'free': 652, 'second': 653, 'increasing': 654, 'saying': 655, 'equity': 656, 'got': 657, 'owns': 658, 'composite': 659, 'performance': 660, 'control': 661, '1991': 662, 'rather': 663, 'acquired': 664, 'clients': 665, 'purchasing': 666, 'survey': 667, 'run': 668, 'conditions': 669, 'real': 670, 'job': 671, 'effect': 672, 'states': 673, 'need': 674, 'gained': 675, '10,000': 676, 'drug': 677, 'nixon': 678, 'fined': 679, 'wage': 680, 'farmers': 681, 'buick': 682, 'boston': 683, 'institute': 684, 'nations': 685, 'annual': 686, 'marketing': 687, 'steel': 688, 'away': 689, 'announced': 690, '11': 691, 'dec.': 692, '1985': 693, 'tax': 694, 'texas': 695, 'filed': 696, 'agreement': 697, 'became': 698, 'following': 699, 'dollars': 700, 'seek': 701, 'nearly': 702, 'agreed': 703, 'michael': 704, '6': 705, 'sector': 706, 'activity': 707, 'included': 708, 'important': 709, 'young': 710, 'charges': 711, 'campaign': 712, 'light': 713, 'containers': 714, 'package': 715, 'using': 716, 'led': 717, 'factory': 718, 'energy': 719, 'meeting': 720, 'decided': 721, 'wednesday': 722, '4': 723, 'keep': 724, 'cost': 725, '...': 726, 'increases': 727, 'r.': 728, '31': 729, 'poor': 730, 'disclosed': 731, 'him': 732, 'able': 733, 'posted': 734, 'principal': 735, 'computers': 736, 'data': 737, 'telephone': 738, 'shearson': 739, 'changes': 740, 'co': 741, 'traded': 742, 'find': 743, 'clear': 744, 'failed': 745, 'delivery': 746, 'article': 747, 'always': 748, 'tests': 749, 'women': 750, 'taken': 751, 'know': 752, 'lawyers': 753, 'region': 754, 'policy': 755, 'budget': 756, 'coming': 757, 'section': 758, "'ve": 759, 'third': 760, 'violations': 761, 'calif.': 762, 'sea': 763, 'car': 764, 'ford': 765, '#': 766, 'grain': 767, 'ual': 768, 'moody': 769, 'closely': 770, 'probably': 771, 'huge': 772, 'continued': 773, 'again': 774, 'd.': 775, 'completed': 776, 'j.': 777, 'special': 778, 'imports': 779, 'potential': 780, 'fact': 781, 'review': 782, 'consumer': 783, 'trucks': 784, 'hills': 785, '17': 786, 'reports': 787, 'personal': 788, 'built': 789, 'jobs': 790, 'via': 791, 'raised': 792, 'buy-out': 793, 'long-term': 794, 'asia': 795, 'defense': 796, 'needed': 797, 'mitsubishi': 798, 'across': 799, 'series': 800, 'ward': 801, 'things': 802, 'lost': 803, 'justice': 804, 'statement': 805, 'speculation': 806, 'st.': 807, 'wines': 808, 'nl': 809, 'phelan': 810, 'church': 811, '3\\/8': 812, 'researchers': 813, 'show': 814, 'later': 815, 'kind': 816, 'question': 817, 'short-term': 818, 'revenue': 819, 'previously': 820, 'fed': 821, 'paying': 822, 'matter': 823, 'holders': 824, 'commonwealth': 825, 'whose': 826, 'changed': 827, 'rule': 828, 'domestic': 829, 'financing': 830, 'working': 831, 'include': 832, 'believe': 833, 'receive': 834, 'why': 835, 'software': 836, '-lcb-': 837, '-rcb-': 838, 'aid': 839, 'continuing': 840, 'bad': 841, 'david': 842, 'significant': 843, 'wants': 844, 'capacity': 845, 'scheduled': 846, 'thomas': 847, 'conference': 848, 'went': 849, 'june': 850, 'regulators': 851, 'account': 852, 'available': 853, 'individual': 854, 'johnson': 855, 'meanwhile': 856, 'indicated': 857, 'generally': 858, 'building': 859, 'provide': 860, 'helped': 861, 'evidence': 862, 'short': 863, 'thing': 864, 'man': 865, 'something': 866, 'ms.': 867, 'line': 868, 'river': 869, 'planned': 870, 'look': 871, 'goes': 872, 'abortion': 873, 'teacher': 874, 'district': 875, 'told': 876, 'others': 877, 'parent': 878, 'cause': 879, 'pretax': 880, 'irs': 881, 'francisco': 882, 'care': 883, 'december': 884, 'minimum': 885, 'options': 886, 'u.k.': 887, 'apparently': 888, 'herald': 889, 'bell': 890, 'antitrust': 891, 'rating': 892, 'fields': 893, 'plc': 894, 'caused': 895, 'dr.': 896, 'study': 897, 'west': 898, 'western': 899, 'process': 900, 'continues': 901, 'authority': 902, 'dropped': 903, 'auto': 904, 'position': 905, 'nine': 906, 'showed': 907, 'advertising': 908, 'advertisers': 909, 'concerns': 910, '25': 911, 'ordered': 912, 'richard': 913, 'april': 914, 'ruling': 915, '*ich*-3': 916, '60': 917, 'fujitsu': 918, 'instead': 919, 'slow': 920, 'launched': 921, 'convertible': 922, 'preferred': 923, 'gives': 924, '70': 925, 'reduce': 926, 'organization': 927, '90': 928, 'key': 929, 'anything': 930, 'crash': 931, 'traditional': 932, 'soon': 933, "'m": 934, 'levels': 935, 'quickly': 936, 'legal': 937, 'goods': 938, 'leading': 939, 'export': 940, 'mark': 941, 'strike': 942, 'sony': 943, 'partners': 944, 'stores': 945, 'daily': 946, 'newspaper': 947, 'homeless': 948, 'person': 949, 'situation': 950, 'wanted': 951, 'known': 952, 'transportation': 953, 'remains': 954, 'greenville': 955, 'carolina': 956, 'alleged': 957, 'scoring': 958, 'advanced': 959, '7\\/8': 960, '5\\/8': 961, 'third-quarter': 962, 'losses': 963, 'sept.': 964, 'profits': 965, 'buyers': 966, 'los': 967, 'within': 968, 'departure': 969, 'airline': 970, 'hearst': 971, 'pence': 972, 'drexel': 973, 'wedtech': 974, 'having': 975, 'neither': 976, 'nor': 977, 'risk': 978, 'ban': 979, 'sign': 980, 'monday': 981, 'top': 982, 'overseas': 983, 'corp': 984, 'makers': 985, 'valley': 986, 'followed': 987, 'figures': 988, 'exports': 989, 'remain': 990, 'january': 991, 'meet': 992, 'risks': 993, 'hard': 994, 'jr.': 995, 'electronics': 996, 'subject': 997, 'units': 998, '21': 999, 'fully': 1000, 'perhaps': 1001, 'age': 1002, '16': 1003, 'believes': 1004, 'includes': 1005, 'improve': 1006, 'themselves': 1007, 'best': 1008, 'history': 1009, 'shareholder': 1010, 'similar': 1011, 'issued': 1012, '14': 1013, 'head': 1014, 'asked': 1015, 'slowing': 1016, 'lot': 1017, 'ca': 1018, 'really': 1019, 'highly': 1020, 'view': 1021, 'returns': 1022, 'given': 1023, 'serious': 1024, 'involved': 1025, 'plants': 1026, 'taking': 1027, 'partly': 1028, 'ability': 1029, 'term': 1030, '5,000': 1031, 'radio': 1032, 'random': 1033, 'constitution': 1034, '1\\/4': 1035, 'angeles': 1036, 'adds': 1037, 'largely': 1038, 'bottle': 1039, 'fourth': 1040, 'floor': 1041, 'program-trading': 1042, 'des': 1043, 'usia': 1044, 'mcgovern': 1045, 'waertsilae': 1046, 'gold': 1047, 'british': 1048, 'asbestos': 1049, 'percentage': 1050, 'makes': 1051, 'journal': 1052, 'medical': 1053, 'type': 1054, '*-5': 1055, 't.': 1056, 'rejected': 1057, 'areas': 1058, 'relatively': 1059, 'rising': 1060, 'watch': 1061, 'editor': 1062, 'systems': 1063, 'idea': 1064, 'course': 1065, 'ministry': 1066, 'sharply': 1067, 'circulation': 1068, 'utilities': 1069, 'outside': 1070, 'bankruptcy': 1071, 'related': 1072, 'turned': 1073, 'hopes': 1074, 'groups': 1075, 'underlying': 1076, 'beginning': 1077, 'note': 1078, 'technology': 1079, 'enforcement': 1080, 'measure': 1081, 'efforts': 1082, 'television': 1083, 'greater': 1084, 'reserves': 1085, 'effort': 1086, 'role': 1087, 'publicly': 1088, 'single': 1089, 'reason': 1090, 'reading': 1091, 'recession': 1092, 'boost': 1093, 'read': 1094, 'relations': 1095, 'feel': 1096, 'central': 1097, 'win': 1098, 'chinese': 1099, 'lane': 1100, 'takes': 1101, 'selling': 1102, 'tv': 1103, 'dinkins': 1104, 'coleman': 1105, 'mary': 1106, 'answers': 1107, 'teachers': 1108, 'scores': 1109, 'education': 1110, 'learning': 1111, 'prevent': 1112, 'louis': 1113, 'one-time': 1114, 'media': 1115, 'rally': 1116, 'criminal': 1117, 'member': 1118, 'benefits': 1119, 'retail': 1120, 'offset': 1121, 'georgia-pacific': 1122, 'ratings': 1123, 'barred': 1124, '30-year': 1125, '200': 1126, 'simmons': 1127, 'unchanged': 1128, 'earned': 1129, 'bernstein': 1130, 'consented': 1131, 'osha': 1132, 'drugs': 1133, 'marine': 1134, '1\\/8': 1135, 'spiegel': 1136, '55': 1137, 'men': 1138, 'class': 1139, 'material': 1140, 'longer': 1141, 'open': 1142, 'pacific': 1143, 'approval': 1144, 'ceiling': 1145, 'gave': 1146, '1986': 1147, 'target': 1148, 'newsweek': 1149, 'competition': 1150, 'attempt': 1151, 'totaled': 1152, 'acquire': 1153, 'worth': 1154, 'field': 1155, 'required': 1156, 'begin': 1157, 'supreme': 1158, 'moreover': 1159, 'interview': 1160, 'either': 1161, 'require': 1162, 'equipment': 1163, 'direct': 1164, 'list': 1165, 'patent': 1166, 'secretary': 1167, 'joint': 1168, 'businesses': 1169, 'produced': 1170, 'left': 1171, 'bought': 1172, 'mortgage': 1173, 'institutional': 1174, 'marks': 1175, 'announcement': 1176, 'charles': 1177, 'associates': 1178, 'easy': 1179, 'europe': 1180, 'soviets': 1181, 'unless': 1182, '`': 1183, 'consider': 1184, 'difference': 1185, 'usually': 1186, 'carry': 1187, 'practice': 1188, 'size': 1189, 'sometimes': 1190, 'hours': 1191, 'consumers': 1192, 'build': 1193, 'necessary': 1194, 'poland': 1195, 'loans': 1196, 'estate': 1197, 'martin': 1198, 'student': 1199, 'artist': 1200, 'seem': 1201, 'expensive': 1202, 'hit': 1203, 'getting': 1204, 'shows': 1205, 'peter': 1206, 'battle': 1207, 'introduced': 1208, 'stations': 1209, 'courter': 1210, 'southeast': 1211, 'asian': 1212, 'east': 1213, 'won': 1214, 'testing': 1215, 'critics': 1216, 'unfair': 1217, 'widely': 1218, 'iowa': 1219, 'aimed': 1220, 'c.': 1221, 'slightly': 1222, 'heritage': 1223, 'swap': 1224, 'year-earlier': 1225, 'bids': 1226, 'leaders': 1227, 'limited': 1228, 'hurt': 1229, '130': 1230, 'congressional': 1231, 'cited': 1232, 'acquisitions': 1233, 'minority': 1234, 'williams': 1235, 'sought': 1236, 'tons': 1237, 'carrier': 1238, 'retirement': 1239, 'unconstitutional': 1240, 'line-item': 1241, 'crop': 1242, 'promotion': 1243, 'guard': 1244, 'reliance': 1245, 'upjohn': 1246, 'carnival': 1247, 'cancer': 1248, 'medicine': 1249, 'bring': 1250, 'schools': 1251, 'appears': 1252, 'college': 1253, 'remaining': 1254, 'area': 1255, 'resources': 1256, 'mutual': 1257, 'signs': 1258, 'portfolio': 1259, 'considered': 1260, 'seven': 1261, 'regulatory': 1262, 'engineering': 1263, 'ohio': 1264, 'suspended': 1265, 'motor': 1266, 'manufacturers': 1267, 'prove': 1268, 'space': 1269, 'block': 1270, 'treatment': 1271, 'looking': 1272, 'base': 1273, 'per': 1274, 'doing': 1275, 'flat': 1276, 'electric': 1277, 'reorganization': 1278, 'm.': 1279, 'circuit': 1280, 'refund': 1281, 'ever': 1282, '45': 1283, 'near': 1284, 'involving': 1285, 'opened': 1286, 'cray-3': 1287, 'filing': 1288, 'book': 1289, 'brazil': 1290, 'standards': 1291, 'elaborate': 1292, 'stand': 1293, 'together': 1294, 'chemical': 1295, 'purchases': 1296, 'france': 1297, 'boosted': 1298, 'giving': 1299, 'giant': 1300, 'november': 1301, '24': 1302, 'fixed': 1303, 'listed': 1304, 'runs': 1305, 'mature': 1306, 'premium': 1307, 'americans': 1308, 'separate': 1309, 'accounts': 1310, 'provided': 1311, 'sides': 1312, 'especially': 1313, 'stadium': 1314, 'center': 1315, 'prepared': 1316, 'woman': 1317, 'ahead': 1318, 'negative': 1319, 'candidates': 1320, 'side': 1321, 'himself': 1322, 'security': 1323, 'cheating': 1324, 'me': 1325, 'ones': 1326, 'nothing': 1327, 'staff': 1328, 'source': 1329, 'indeed': 1330, 'allowed': 1331, 'britain': 1332, 'fetal-tissue': 1333, 'transplants': 1334, 'roughly': 1335, 'trust': 1336, 'approve': 1337, 'ratners': 1338, 'pop': 1339, 'customer': 1340, 'payment': 1341, 'letters': 1342, 'individuals': 1343, 'guild': 1344, 'activities': 1345, 'community': 1346, 'thursday': 1347, 'nekoosa': 1348, 'commodity': 1349, 'cosby': 1350, 'air': 1351, 'wrote': 1352, 'opportunity': 1353, 'temple': 1354, 'requirements': 1355, 'reagan': 1356, 'reduced': 1357, 'experience': 1358, '300': 1359, 'vote': 1360, 'advantage': 1361, 'benchmark': 1362, 'mitsui': 1363, 'fla.': 1364, 'ways': 1365, 'session': 1366, 'compromise': 1367, 'homelessness': 1368, 'baum': 1369, 'farm': 1370, 'closing': 1371, 'superconductors': 1372, '2009': 1373, 'guilders': 1374, 'rumors': 1375, 'reuters': 1376, 'kent': 1377, 'findings': 1378, 'attention': 1379, 'spokeswoman': 1380, 'mass.': 1381, 'highest': 1382, 'professor': 1383, 'july': 1384, 'uses': 1385, 'mixed': 1386, 'described': 1387, 'yields': 1388, 'expect': 1389, 'thought': 1390, '30-day': 1391, 'seats': 1392, 'along': 1393, 'name': 1394, 'monthly': 1395, 'labor': 1396, 'sluggish': 1397, '19': 1398, 'spent': 1399, 'pages': 1400, 'heavy': 1401, 'none': 1402, 'hampshire': 1403, 'northeast': 1404, 'raising': 1405, 'factors': 1406, 'request': 1407, 'summer': 1408, 'chain': 1409, 'curry': 1410, 'actual': 1411, 'courts': 1412, 'expenses': 1413, 'hope': 1414, 'introduction': 1415, 'produce': 1416, 'spinoff': 1417, 'initial': 1418, 'range': 1419, 'hand': 1420, 'taiwan': 1421, 'spring': 1422, 'placed': 1423, 'scientists': 1424, 'movie': 1425, 'producers': 1426, 'works': 1427, 'lawyer': 1428, 'provision': 1429, 'minister': 1430, 'reduction': 1431, 'assistant': 1432, 'advance': 1433, 'developed': 1434, 'allow': 1435, '23': 1436, 'sells': 1437, 'holdings': 1438, 'barrels': 1439, 'operate': 1440, 'start': 1441, 'main': 1442, 'magna': 1443, 'mcalpine': 1444, 'turn': 1445, 'jointly': 1446, '900': 1447, 'extend': 1448, 'aerospace': 1449, 'temporarily': 1450, 'surged': 1451, 'e.': 1452, 'hold': 1453, '1992': 1454, 'jumped': 1455, 'essentially': 1456, 'affairs': 1457, 'democratic': 1458, '*exp*-2': 1459, 'war': 1460, 'reserve': 1461, 'lack': 1462, 'bankers': 1463, 'eventually': 1464, 'pressures': 1465, 'improvement': 1466, 'talk': 1467, 'offers': 1468, 'seems': 1469, 'call': 1470, 'game': 1471, 'rules': 1472, 'except': 1473, 'prime': 1474, 'ending': 1475, 'par': 1476, 'claim': 1477, 'night': 1478, 'sense': 1479, 'final': 1480, 'sound': 1481, 'client': 1482, 'press': 1483, 'ads': 1484, 'opposed': 1485, 'wilder': 1486, 'created': 1487, 'victims': 1488, 'focus': 1489, 'heating': 1490, 'environment': 1491, 'commitments': 1492, 'military': 1493, 'approach': 1494, 'seen': 1495, 'germany': 1496, 'forms': 1497, 'laws': 1498, '1984': 1499, 'standardized': 1500, 'experts': 1501, 'overall': 1502, 'mostly': 1503, 'pretty': 1504, 'majority': 1505, 'attorneys': 1506, 'miami': 1507, 'judges': 1508, 'restrictions': 1509, 'park': 1510, 'headquarters': 1511, 'access': 1512, 'communications': 1513, 'valued': 1514, 'network': 1515, 'losing': 1516, 'willing': 1517, 'packaging': 1518, 'print': 1519, '75': 1520, 'philadelphia': 1521, 'employment': 1522, 'metals': 1523, 'upon': 1524, 'segments': 1525, 'la': 1526, 'appeal': 1527, 'orleans': 1528, 'merchant': 1529, 'estimates': 1530, 'smaller': 1531, 'provides': 1532, 'corporations': 1533, 'duties': 1534, 'packages': 1535, 'training': 1536, 'confidence': 1537, 'dingell': 1538, 'let': 1539, 'sources': 1540, 'tower': 1541, 'successor': 1542, 'telerate': 1543, 'newhouse': 1544, 'strategy': 1545, 'diaper': 1546, 'current-carrying': 1547, 'fourth-quarter': 1548, 'steinberg': 1549, 'filters': 1550, 'stopped': 1551, 'story': 1552, '33': 1553, 'worked': 1554, 'diseases': 1555, 'environmental': 1556, 'protection': 1557, 'virtually': 1558, 'parts': 1559, 'human': 1560, '35': 1561, 'declines': 1562, '400': 1563, '1.5': 1564, 'heavily': 1565, 'complete': 1566, 'interests': 1567, 'giants': 1568, 'mayor': 1569, 'standing': 1570, 'fifth': 1571, 'forecast': 1572, 'pace': 1573, 'introduce': 1574, 'post': 1575, 'b.': 1576, 'watches': 1577, 'ps': 1578, 'internal': 1579, 'conn.': 1580, 'improved': 1581, 'true': 1582, 'seemed': 1583, 'troubled': 1584, 'directors': 1585, '250': 1586, 'ill.': 1587, 'instruments': 1588, 'devices': 1589, 'machine': 1590, 'prospects': 1591, 'nec': 1592, 'fewer': 1593, 'leave': 1594, 'so-called': 1595, 'canada': 1596, 'combined': 1597, 'believed': 1598, 'venture': 1599, 'subsidiary': 1600, 'itself': 1601, 'english': 1602, 'hong': 1603, 'kong': 1604, 'goldman': 1605, 'sachs': 1606, 'eliminate': 1607, 'forces': 1608, 'l.': 1609, 'file': 1610, 'expire': 1611, 'currency': 1612, 'managing': 1613, 'partner': 1614, 'popular': 1615, 'claims': 1616, 'instance': 1617, 'specialists': 1618, 'european': 1619, 'invested': 1620, 'suggests': 1621, 'represent': 1622, 'moscow': 1623, 'branch': 1624, 'settle': 1625, 'society': 1626, 'measures': 1627, 'economists': 1628, 'inventories': 1629, 'front': 1630, 'strength': 1631, 'reporting': 1632, 'particular': 1633, 'trend': 1634, 'items': 1635, 'supply': 1636, 'contributed': 1637, 'chase': 1638, 'figure': 1639, 'employee': 1640, 'social': 1641, 'indicates': 1642, 'asking': 1643, 'beijing': 1644, 'debentures': 1645, '1,000': 1646, 'film': 1647, 'streets': 1648, 'playing': 1649, 'purpose': 1650, 'children': 1651, 'achievement': 1652, 'resistance': 1653, 'possibly': 1654, 'larger': 1655, 'discussions': 1656, 'elections': 1657, 'simply': 1658, 'refused': 1659, 'powers': 1660, 'attorney': 1661, 'bob': 1662, 'quoted': 1663, 'assistance': 1664, 'decade': 1665, 'comes': 1666, 'specialist': 1667, 'regional': 1668, 'friends': 1669, 'difficult': 1670, 'ease': 1671, 'uncertainty': 1672, 'pennsylvania': 1673, 'comprehensive': 1674, 'passed': 1675, 'unusual': 1676, 'reform': 1677, 'wrongdoing': 1678, 'numerous': 1679, 'classroom': 1680, 'hearing': 1681, 'interested': 1682, 'avoid': 1683, 'mason': 1684, 'letter': 1685, 'threatened': 1686, 'mental': 1687, 'biggest': 1688, 'connecticut': 1689, 'massachusetts': 1690, 'water': 1691, 'substantially': 1692, 'assuming': 1693, '51': 1694, 'trudeau': 1695, 'suit': 1696, 'opinion': 1697, 'violate': 1698, 'dallas': 1699, 'n.j.': 1700, 'sure': 1701, 'design': 1702, 'operates': 1703, 'g.': 1704, 'kept': 1705, 'viacom': 1706, 'midwest': 1707, "'ll": 1708, 'officially': 1709, 'garbage': 1710, 'n.y.': 1711, 'soup': 1712, 'reaction': 1713, '65': 1714, '80': 1715, 'soared': 1716, '62': 1717, 'exercise': 1718, 'eliminated': 1719, 'morgan': 1720, 'stanley': 1721, 'becomes': 1722, 'differences': 1723, 'signal': 1724, 'glass': 1725, 'ships': 1726, 'houses': 1727, 'airlines': 1728, 'create': 1729, 'active': 1730, 'margins': 1731, 'profitable': 1732, 'send': 1733, 'foster': 1734, 'merc': 1735, 'edward': 1736, 'ringing': 1737, 'thus': 1738, 'vicar': 1739, 'council': 1740, 'stock-market': 1741, 'sharp': 1742, 'nasd': 1743, 'impact': 1744, 'index-arbitrage': 1745, 'hour': 1746, 'pound': 1747, 'coupon': 1748, 'leveraged': 1749, 'amendment': 1750, 'shall': 1751, 'clause': 1752, 'drought': 1753, 'basket': 1754, 'widget': 1755, 'lines': 1756, 'partnership': 1757, 'emissions': 1758, 'meridian': 1759, 'deaths': 1760, 'cigarettes': 1761, 'preliminary': 1762, 'anyone': 1763, 'amounts': 1764, 'replaced': 1765, 'finding': 1766, 'owned': 1767, 'argue': 1768, 'easily': 1769, 'blue': 1770, 'six-month': 1771, 'beat': 1772, 'world-wide': 1773, 'elected': 1774, 'resigned': 1775, 'controls': 1776, 'default': 1777, 'chrysler': 1778, 'machines': 1779, 'meetings': 1780, 'registered': 1781, 'deficit': 1782, 'electronic': 1783, 'bidding': 1784, 'independent': 1785, 'suffer': 1786, 'substantial': 1787, 'worried': 1788, 'complicated': 1789, 'succeed': 1790, 'illegal': 1791, 'exact': 1792, 'administrative': 1793, '12.5': 1794, '49': 1795, 'vehicles': 1796, 'directly': 1797, 'documents': 1798, 'project': 1799, '120': 1800, 'details': 1801, 'presence': 1802, 'concept': 1803, 'sec': 1804, 'contain': 1805, 'ltd': 1806, 'occur': 1807, 'nasdaq': 1808, 'primarily': 1809, 'clearly': 1810, 'donald': 1811, '59': 1812, 'patents': 1813, 'mexico': 1814, 'sanctions': 1815, 'progress': 1816, 'pursue': 1817, 'effective': 1818, 'showing': 1819, 'malaysia': 1820, 'pharmaceutical': 1821, 'requires': 1822, 'met': 1823, '1977': 1824, 'apple': 1825, 'crude': 1826, 'sets': 1827, 'stephen': 1828, 'pcs': 1829, 'unlike': 1830, 'types': 1831, 'ibm': 1832, 'atlanta': 1833, 'shipments': 1834, 'brought': 1835, 'started': 1836, 'southern': 1837, 'lehman': 1838, 'hutton': 1839, 'response': 1840, 'covered': 1841, 'philippines': 1842, 'eligible': 1843, 'expansion': 1844, 'steady': 1845, 'quarterly': 1846, 'parliament': 1847, 'running': 1848, 'mae': 1849, 'rest': 1850, 'fixed-rate': 1851, 'creditors': 1852, 'debts': 1853, 'tender': 1854, 'surge': 1855, 'smith': 1856, 'brokers': 1857, 'gyrations': 1858, 'broader': 1859, 'plunged': 1860, 'discount': 1861, 'global': 1862, 'party': 1863, 'millions': 1864, 'fallen': 1865, 'ran': 1866, 'significantly': 1867, 'inflation': 1868, 'economist': 1869, 'cleveland': 1870, 'warning': 1871, 'mcgraw-hill': 1872, 'import': 1873, 'language': 1874, 'ties': 1875, 'lives': 1876, 'whom': 1877, 'parties': 1878, 'fired': 1879, 'crime': 1880, 'deputy': 1881, 'editorial': 1882, 'becoming': 1883, 'elevators': 1884, 'membership': 1885, '1979': 1886, '150': 1887, 'gas': 1888, '100,000': 1889, 'warrants': 1890, 'date': 1891, 'tramp': 1892, 'contained': 1893, 'black': 1894, 'calls': 1895, 'managed': 1896, 'needs': 1897, 'means': 1898, 'politicians': 1899, 'marshall': 1900, 'de': 1901, 'suggested': 1902, 'accused': 1903, 'positive': 1904, 'appeared': 1905, 'step': 1906, 'basic': 1907, 'parents': 1908, 'colleagues': 1909, 'nature': 1910, 'provisions': 1911, 'bonus': 1912, 'wrong': 1913, 'telegraph': 1914, 'reforms': 1915, 'death': 1916, 'blacks': 1917, 'green': 1918, 'helping': 1919, 'hands': 1920, 'discovered': 1921, 'trial': 1922, 'someone': 1923, 'add': 1924, 'cat': 1925, 'referred': 1926, 'bar': 1927, 'oppose': 1928, 'stop': 1929, 'privately': 1930, 'panel': 1931, 'views': 1932, 'moves': 1933, 'applications': 1934, 'debate': 1935, 'interstate': 1936, 'uptick': 1937, 'trader': 1938, 'merger': 1939, 'sent': 1940, 'formed': 1941, 'aba': 1942, 'moderate': 1943, 'litigation': 1944, 'damages': 1945, 'murray': 1946, 'alone': 1947, 'auctions': 1948, 'factor': 1949, 'surprise': 1950, 'santa': 1951, 'resulting': 1952, 'reasons': 1953, 'entertainment': 1954, 'northern': 1955, 'turning': 1956, 'baltimore': 1957, '1.1': 1958, 'asset': 1959, 'belts': 1960, 'model': 1961, 'sen.': 1962, 'interesting': 1963, 'cent': 1964, 'cabernet': 1965, '!': 1966, 'shipped': 1967, 'thin': 1968, 'rapidly': 1969, 'forced': 1970, 'door': 1971, 'easier': 1972, 'chance': 1973, '2019': 1974, 'two-year': 1975, 'chemicals': 1976, 'alternatives': 1977, 'various': 1978, 'junk-bond': 1979, 'candela': 1980, 'deals': 1981, 'starting': 1982, 'segment': 1983, 'travel': 1984, 'institution': 1985, 'mercantile': 1986, 'breaker': 1987, 'equal': 1988, 'markey': 1989, 'agriculture': 1990, 'ton': 1991, 'attract': 1992, 'merrill': 1993, 'lynch': 1994, 'painewebber': 1995, 'voters': 1996, 'settlement': 1997, 'admitting': 1998, 'denying': 1999, 'pa.': 2000, 'disgorge': 2001, 'one-year': 2002, 'bridge': 2003, 'alcohol': 2004, 'motors': 2005, 'attributed': 2006, 'buyer': 2007, 'neuberger': 2008, 'margin': 2009, 'ownership': 2010, 'liquidity': 2011, 'filings': 2012, 'ballot': 2013, 'van': 2014, 'tentatively': 2015, 'underwriter': 2016, 'tomorrow': 2017, 'ship': 2018, 'copper': 2019, 'younkers': 2020, 'reupke': 2021, 'shipyard': 2022, 'ralston': 2023, 'mississippi': 2024, '61': 2025, 'nonexecutive': 2026, '29': 2027, 'crocidolite': 2028, 'brief': 2029, 'causing': 2030, 'lorillard': 2031, 'york-based': 2032, 'studied': 2033, 'substance': 2034, '*t*-5': 2035, 'slide': 2036, 'compound': 2037, 'declining': 2038, 'stronger': 2039, 'auction': 2040, 'typically': 2041, 'yielding': 2042, 'toronto': 2043, '27': 2044, 'trillion': 2045, 'cutting': 2046, 'voted': 2047, '3.1': 2048, 'settled': 2049, 'expand': 2050, 'h.': 2051, 'police': 2052, 'red': 2053, 'honor': 2054, 'announcer': 2055, 'hotel': 2056, 'indiana': 2057, 'stood': 2058, 'rival': 2059, 'page': 2060, 'proceedings': 2061, 'places': 2062, 'rowe': 2063, 'ross': 2064, 'adviser': 2065, 'separately': 2066, '63': 2067, 'appliances': 2068, 'retired': 2069, 'nuclear': 2070, 'utility': 2071, 'moved': 2072, 'appeals': 2073, 'round': 2074, 'plus': 2075, 'records': 2076, 'miles': 2077, 'supercomputer': 2078, 'seymour': 2079, 'balance': 2080, 'pending': 2081, 'providing': 2082, 'heads': 2083, 'anticipated': 2084, 'p.': 2085, 'minneapolis': 2086, 'chip': 2087, 'chips': 2088, 'handling': 2089, 'intense': 2090, 'distribution': 2091, 'marketplace': 2092, 'decide': 2093, 'colo.': 2094, '47': 2095, 'douglas': 2096, 'hardware': 2097, 'positions': 2098, 'eastern': 2099, 'intellectual-property': 2100, 'thailand': 2101, 'representative': 2102, 'property': 2103, 'authors': 2104, 'creating': 2105, 'prosecutors': 2106, 'prompted': 2107, 'investigation': 2108, 'developing': 2109, 'reach': 2110, 'models': 2111, 'paul': 2112, 'leader': 2113, 'australia': 2114, 'launch': 2115, 'broken': 2116, 'duty-free': 2117, 'grant': 2118, 'producer': 2119, 'stronach': 2120, 'maintained': 2121, 'excess': 2122, 'automotive': 2123, 'downturn': 2124, 'canadian': 2125, 'founder': 2126, 'seat': 2127, 'throughout': 2128, 'career': 2129, 'reducing': 2130, 'expanding': 2131, 'hefty': 2132, 'consecutive': 2133, 'follows': 2134, 'sweeping': 2135, 'portfolios': 2136, 'ranged': 2137, 'visit': 2138, 'george': 2139, 'focused': 2140, 'behind': 2141, 'wide': 2142, 'wild': 2143, 'frenzy': 2144, 'climbed': 2145, 'play': 2146, 'reflect': 2147, 'exposure': 2148, 'insist': 2149, 'technical': 2150, 'owed': 2151, 'happen': 2152, 'adjusted': 2153, 'cutbacks': 2154, 'awarded': 2155, 'imminent': 2156, 'indicators': 2157, 'sheep': 2158, 'worry': 2159, 'funny': 2160, 'copies': 2161, 'books': 2162, 'written': 2163, 'version': 2164, 'spirit': 2165, 'players': 2166, 'katzenstein': 2167, 'certainly': 2168, 'fast-food': 2169, 'urged': 2170, 'carried': 2171, 'identified': 2172, '20,000': 2173, 'application': 2174, 'disease': 2175, 'confirmed': 2176, 'dam': 2177, 'czechoslovakia': 2178, '1999': 2179, '1976': 2180, 'sidewalk': 2181, 'thinking': 2182, 'picture': 2183, 'thousands': 2184, 'attack': 2185, 'tight': 2186, 'straight': 2187, 'giuliani': 2188, 'choose': 2189, 'rape': 2190, 'tried': 2191, 'legislative': 2192, 'lawmakers': 2193, '1983': 2194, 'florio': 2195, 'female': 2196, 'mean': 2197, 'pictures': 2198, 'waste': 2199, 'lying': 2200, 'asks': 2201, 'democrat': 2202, 'rockefeller': 2203, 'toward': 2204, 'suggest': 2205, 'flow': 2206, 'influence': 2207, 'falling': 2208, 'discuss': 2209, 'telecommunications': 2210, 'participants': 2211, 'try': 2212, 'matched': 2213, 'examination': 2214, 'geography': 2215, 'alternative': 2216, 'wake': 2217, 'actions': 2218, 'educators': 2219, 'concluded': 2220, 'widespread': 2221, 'practices': 2222, 'booklets': 2223, 'sophisticated': 2224, 'county': 2225, 'entire': 2226, 'abuse': 2227, 'keeping': 2228, '1980s': 2229, 'kids': 2230, 'prior': 2231, 'trouble': 2232, "'d": 2233, 'stands': 2234, 'salary': 2235, 'returned': 2236, 'bottom': 2237, 'newspapers': 2238, 'authorities': 2239, 'policies': 2240, 'familiar': 2241, 'favor': 2242, 'mich.': 2243, 'macmillan\\/mcgraw': 2244, 'offices': 2245, 'funded': 2246, 'sullivan': 2247, 'dispute': 2248, 'withdrawn': 2249, '*ich*-4': 2250, 'over-the-counter': 2251, 'modestly': 2252, 'targets': 2253, 'saw': 2254, 'signed': 2255, 'weisfield': 2256, 'operator': 2257, 'cities': 2258, 'citing': 2259, 'taxpayers': 2260, 'judicial': 2261, 'troubles': 2262, 'lawsuit': 2263, 'mainly': 2264, 'adopted': 2265, 'jack': 2266, 'gaf': 2267, 'opening': 2268, 'specialty': 2269, 'attempting': 2270, 'pricing': 2271, 'municipal': 2272, 'yamamoto': 2273, 'accepted': 2274, 'normal': 2275, 'machinery': 2276, 'cover': 2277, 'extraordinary': 2278, 'older': 2279, 'facility': 2280, 'facing': 2281, 'n.j': 2282, 'sterling': 2283, 'madison': 2284, '10-year': 2285, 'merely': 2286, 'expectations': 2287, 'ounce': 2288, 'conventional': 2289, 'considerable': 2290, 'economics': 2291, 'initially': 2292, 'programming': 2293, 'expanded': 2294, 'at&t': 2295, 'leaving': 2296, 'raises': 2297, 'vans': 2298, 'minivans': 2299, 'vehicle': 2300, 'surprising': 2301, 'fairly': 2302, '1982': 2303, 'heights': 2304, 'petroleum': 2305, 'declared': 2306, 'precious': 2307, '180': 2308, 'owner': 2309, 'spread': 2310, 'push': 2311, 'plunge': 2312, '1993': 2313, 'finished': 2314, 'afternoon': 2315, 'supplies': 2316, 'regarding': 2317, 'consent': 2318, 'retin-a': 2319, 'elsewhere': 2320, 'successful': 2321, 'frequently': 2322, 'commodities': 2323, 'basis': 2324, 'deposit': 2325, 'competitive': 2326, 'population': 2327, 'charlotte': 2328, 'travelers': 2329, 'direction': 2330, 'realize': 2331, 'felt': 2332, 'remained': 2333, 'shortly': 2334, 'expressed': 2335, 'done': 2336, 'live': 2337, 'worker': 2338, 'basically': 2339, 'replace': 2340, 'nbi': 2341, 'embassy': 2342, 'guards': 2343, 'harder': 2344, 'connection': 2345, 'anthony': 2346, '25,000': 2347, 'sizable': 2348, 'bronx': 2349, 'salomon': 2350, '*rnr*-2': 2351, 'brokerage': 2352, 'option': 2353, 'democrats': 2354, 'covers': 2355, 'rockwell': 2356, 'guarantees': 2357, 'initiatives': 2358, 'natural': 2359, 'numbers': 2360, 'complex': 2361, 'fear': 2362, 'buy-outs': 2363, 'benefit': 2364, 'harper': 2365, 'prudential-bache': 2366, 'stay': 2367, 'pension': 2368, 'mortgages': 2369, 'constitutional': 2370, 'nomination': 2371, 'excision': 2372, 'orange': 2373, 'hammersmith': 2374, 'cheaper': 2375, 'swings': 2376, 'ftc': 2377, 'cruise': 2378, 'crystals': 2379, '99': 2380, 'swiss': 2381, 'reed': 2382, 'burnham': 2383, 'lambert': 2384, 'financings': 2385, 'veraldi': 2386, 'defendants': 2387, 'ackerman': 2388, 'estimate': 2389, 'acid': 2390, 'investment-grade': 2391, 'michelin': 2392, 'tyre': 2393, 'rain': 2394, 'intelogic': 2395, 'exporter': 2396, 'tony': 2397, 'lama': 2398, 'freeport-mcmoran': 2399, 'delivered': 2400, 'bone': 2401, 'qualified': 2402, 'lentjes': 2403, 'army': 2404, 'exporters': 2405, 'join': 2406, 'cigarette': 2407, 'enters': 2408, 'appear': 2409, 'talking': 2410, 'heard': 2411, 'talcott': 2412, '28': 2413, 'died': 2414, 'fibers': 2415, '*-6': 2416, '*-7': 2417, 'dumped': 2418, 'imported': 2419, 'dry': 2420, 'amid': 2421, 'assume': 2422, '41': 2423, 'capture': 2424, 'indicator': 2425, '7.90': 2426, 'grew': 2427, 'boosts': 2428, 'simple': 2429, 'grace': 2430, 'holds': 2431, 'italian': 2432, 'mechanical': 2433, 'borrowing': 2434, 'lift': 2435, 'taxes': 2436, 'hot': 2437, 'indianapolis': 2438, 'treat': 2439, 'owners': 2440, '125': 2441, 'belt': 2442, 'receiving': 2443, 'message': 2444, 'joined': 2445, 'evening': 2446, 'champagne': 2447, 'morning': 2448, 'lights': 2449, 'forward': 2450, 'reflecting': 2451, 'released': 2452, 'incentive': 2453, 'maintaining': 2454, 'lowered': 2455, 'effectively': 2456, 'exceed': 2457, '3.2': 2458, 'gaining': 2459, '7.3': 2460, 'chapter': 2461, '2.2': 2462, 'forecasts': 2463, 'electricity': 2464, 'persistent': 2465, 'illinois': 2466, 'attempts': 2467, 'considering': 2468, 'determined': 2469, '2.5': 2470, 'upheld': 2471, 'challenge': 2472, 'ruled': 2473, 'precedent': 2474, '22': 2475, 'reductions': 2476, 'located': 2477, 'seoul': 2478, 'survival': 2479, 'designer': 2480, 'tied': 2481, 'gary': 2482, 'besides': 2483, 'stockholders': 2484, 'applied': 2485, '53': 2486, 'regarded': 2487, 'father': 2488, '450': 2489, 'saudi': 2490, 'failing': 2491, 'priority': 2492, 'interim': 2493, 'varying': 2494, 'degrees': 2495, 'bilateral': 2496, 'apply': 2497, 'spain': 2498, 'protecting': 2499, 'tells': 2500, 'complained': 2501, 'ask': 2502, 'rapanelli': 2503, 'century': 2504, 'pet': 2505, 'club': 2506, 'triggered': 2507, 'faster': 2508, 'memory': 2509, '1981': 2510, 'f.': 2511, 'engineers': 2512, 'entered': 2513, 'annually': 2514, 'affiliate': 2515, 'whiting': 2516, 'hill': 2517, 'producing': 2518, 'denied': 2519, '58': 2520, 'status': 2521, 'aide': 2522, 'frank': 2523, 'manufacturer': 2524, 'us$': 2525, 'consulting': 2526, 'fannie': 2527, 'maxwell': 2528, 'mortgage-backed': 2529, 'extremely': 2530, 'exclusive': 2531, 'code': 2532, 'sixth': 2533, 'reflects': 2534, 'foot': 2535, 'porter': 2536, 'swing': 2537, 'integration': 2538, 'increasingly': 2539, 'adrs': 2540, '*-52': 2541, '46': 2542, 'residential': 2543, 'slowly': 2544, 'predicting': 2545, 'donaldson': 2546, 'lufkin': 2547, 'jenrette': 2548, 'slowdown': 2549, 'intended': 2550, 'dodge': 2551, 'purchased': 2552, 'suspect': 2553, 'dozen': 2554, 'milk': 2555, 'blamed': 2556, 'novel': 2557, 'entirely': 2558, 'drink': 2559, 'reruns': 2560, 'contemporary': 2561, 'author': 2562, 'writers': 2563, 'carries': 2564, 'reality': 2565, 'image': 2566, 'male': 2567, 'conduct': 2568, 'road': 2569, 'responsibility': 2570, 'lesson': 2571, 'appointment': 2572, 'learned': 2573, 'features': 2574, 'singapore': 2575, 'centers': 2576, 'smoking': 2577, 'one-third': 2578, 'renewed': 2579, 'organizations': 2580, 'financially': 2581, 'hungary': 2582, '1972': 2583, 'chaplin': 2584, 'writer': 2585, 'stories': 2586, 'friend': 2587, 'earns': 2588, 'precisely': 2589, 'words': 2590, 'walk': 2591, 'box': 2592, 'child': 2593, 'french': 2594, 'viewed': 2595, 'die': 2596, 'speculated': 2597, 'secondary': 2598, 'campaigns': 2599, 'fears': 2600, 'politics': 2601, 'art': 2602, 'jersey': 2603, 'virginia': 2604, 'screen': 2605, 'republican': 2606, 'master': 2607, 'gets': 2608, 'wait': 2609, 'caught': 2610, 'finally': 2611, 'fail': 2612, 'insists': 2613, 'incomplete': 2614, 'knows': 2615, 'pass': 2616, 'grown': 2617, 'gotten': 2618, 'jim': 2619, 'grows': 2620, 'responded': 2621, 'land': 2622, 'motion': 2623, 'ground': 2624, 'decisions': 2625, 'integrated': 2626, 'closer': 2627, 'speech': 2628, 'flows': 2629, 'everyone': 2630, 'hardly': 2631, 'skills': 2632, 'word': 2633, 'jail': 2634, 'districts': 2635, 'extra': 2636, 'projects': 2637, 'posts': 2638, 'lowest': 2639, 'primary': 2640, 'worst': 2641, 'foundation': 2642, 'preparation': 2643, 'winning': 2644, 'broad': 2645, 'split': 2646, 'encourage': 2647, 'teaching': 2648, 'chosen': 2649, 'beth': 2650, '48': 2651, 'admits': 2652, 'watson': 2653, 'dismissed': 2654, 'host': 2655, 'realized': 2656, 'answer': 2657, 'kaminski': 2658, 'knowledge': 2659, 'sounds': 2660, '44': 2661, 'scientific': 2662, 'prestigious': 2663, 'nih': 2664, 'serve': 2665, 'damaged': 2666, 'real-estate': 2667, 'london-based': 2668, 'associated': 2669, 'jaguar': 2670, '4.4': 2671, 'trotter': 2672, 'slipped': 2673, 'prison': 2674, 'argued': 2675, 'wish': 2676, 'sonnett': 2677, 'dozens': 2678, 'requirement': 2679, 'christopher': 2680, 'detroit': 2681, 'professional': 2682, 'disciplinary': 2683, 'k.': 2684, 'seeks': 2685, 'preventing': 2686, 'limits': 2687, 'comments': 2688, 'sparked': 2689, 'manhattan': 2690, 'd.c.': 2691, 'tire': 2692, 'no.': 2693, 'fire': 2694, 'fueled': 2695, 'complaints': 2696, 'unfairly': 2697, 'municipalities': 2698, 'publications': 2699, 'german': 2700, '-': 2701, 'thrifts': 2702, 'extended': 2703, 'weakness': 2704, 'techniques': 2705, '42': 2706, 'poorly': 2707, 'lake': 2708, 'lure': 2709, 'solid': 2710, 'high-yield': 2711, 'considerably': 2712, 'negotiate': 2713, 'pleased': 2714, 'disappointed': 2715, 'quite': 2716, 'founded': 2717, 'brooklyn': 2718, 'poore': 2719, 'promise': 2720, 'editors': 2721, 'spend': 2722, 'original': 2723, 'shipping': 2724, 'sherwood': 2725, 'possibility': 2726, 'subsidiaries': 2727, 'represents': 2728, 'actually': 2729, 'highway': 2730, 'roof-crush': 2731, 'deadline': 2732, 'ill': 2733, 'railroad': 2734, 'sir': 2735, 'accessories': 2736, 'leap': 2737, 'diamond': 2738, 'creek': 2739, 'quality': 2740, 'stable': 2741, 'petrus': 2742, 'magnitude': 2743, 'salon': 2744, '98': 2745, 'pratt': 2746, 'sort': 2747, 'explains': 2748, 'equally': 2749, '32': 2750, 'beautiful': 2751, 'ideas': 2752, 'guffey': 2753, 'objective': 2754, 'weak': 2755, 'slower': 2756, 'halt': 2757, 'divided': 2758, 'presidents': 2759, '23.5': 2760, 'refunding': 2761, '2.50': 2762, '7.88': 2763, 'mining': 2764, 'harold': 2765, 'aggressive': 2766, 'proposing': 2767, 'jerry': 2768, 'friendly': 2769, 'representing': 2770, 'dallara': 2771, 'fuel': 2772, 'pickens': 2773, 'controversial': 2774, 'financed': 2775, 'klauser': 2776, 'u.s.a.': 2777, 'deposits': 2778, 'fee': 2779, 'banker': 2780, 'bulk': 2781, 'moving': 2782, 'peripheral': 2783, 'vacation': 2784, 'healthy': 2785, 'lewis': 2786, 'subcommittee': 2787, 'collar': 2788, 'execute': 2789, 'existing': 2790, 'minutes': 2791, 'volatile': 2792, 'beyond': 2793, 'writing': 2794, 'discussed': 2795, 'ropes': 2796, 'churches': 2797, 'band': 2798, 'ringer': 2799, 'ring': 2800, 'train': 2801, 'eight': 2802, 'baldwin': 2803, 'payouts': 2804, 'strongest': 2805, 'relative': 2806, '50,000': 2807, 'quarters': 2808, 'hired': 2809, 'normally': 2810, 'citizen': 2811, 'professionals': 2812, '85': 2813, 'henry': 2814, '30,000': 2815, '10-day': 2816, 'whelen': 2817, '2,500': 2818, 'ticket': 2819, 'houston': 2820, 'agents': 2821, 'stupid': 2822, 's.': 2823, 'begins': 2824, 'reflected': 2825, 'acceptance': 2826, 'observers': 2827, '1970': 2828, 'administrator': 2829, 'diversified': 2830, 'portion': 2831, 'bridges': 2832, 'railings': 2833, 'fines': 2834, 'employer': 2835, 'penalties': 2836, 'hazards': 2837, 'causes': 2838, 'disorders': 2839, 'psychiatric': 2840, 'persons': 2841, 'unable': 2842, 'motive': 2843, 'text': 2844, 'democracy': 2845, 'propaganda': 2846, 'facilities': 2847, 'hear': 2848, 'dissemination': 2849, 'reporters': 2850, 'plaintiffs': 2851, 'determine': 2852, 'xerox': 2853, 'dorrance': 2854, 'profitability': 2855, 'switzerland': 2856, 'interpretation': 2857, 'separation': 2858, 'commissions': 2859, 'execution': 2860, 'second-largest': 2861, 'dealings': 2862, 'gasoline': 2863, 'gmac': 2864, '3.8': 2865, 'councils': 2866, 'kidder': 2867, 'stockbrokers': 2868, 'evil': 2869, 'pickers': 2870, 'speculative': 2871, 'expert': 2872, 'agencies': 2873, 'mergers': 2874, 'cuts': 2875, 'marketed': 2876, 'quantity': 2877, 'superconductor': 2878, 'dover': 2879, 'samples': 2880, 'after-tax': 2881, '3.7': 2882, 'disposal': 2883, 'cs': 2884, 'warned': 2885, 'scandal': 2886, 'lens': 2887, 'defensive': 2888, 'metal': 2889, 'hurricane': 2890, 'hugo': 2891, 'clean-air': 2892, 'labor-management': 2893, 'wolf': 2894, 'primerica': 2895, 'edelman': 2896, 'brazilian': 2897, 'ganes': 2898, 'oxnard': 2899, 'equitable': 2900, 'equus': 2901, 'dd': 2902, "dunkin'": 2903, 'gencorp': 2904, 'nissan': 2905, 'finnish': 2906, 'genetics': 2907, 'interleukin-3': 2908, 'itc': 2909, 'feeding': 2910, 'mariotta': 2911, 'barge': 2912, 'corps': 2913, 'bushel': 2914, 'harvest': 2915, 'n.v.': 2916, 'dutch': 2917, 'rudolph': 2918, 'conglomerate': 2919, 'exposures': 2920, 'decades': 2921, 'questionable': 2922, 'aware': 2923, 'harvard': 2924, 'modest': 2925, 'surviving': 2926, 'asbestos-related': 2927, 'buildings': 2928, 'regulation': 2929, 'explained': 2930, 'imposed': 2931, 'fans': 2932, 'recognize': 2933, 'events': 2934, 'bearing': 2935, 'money-market': 2936, 'seven-day': 2937, 'eased': 2938, 'maturity': 2939, 'maturities': 2940, 'indicate': 2941, 'permit': 2942, 'retain': 2943, 'comparable': 2944, 'dollar-denominated': 2945, 'w.r.': 2946, 'formerly': 2947, 'obtain': 2948, 'bailey': 2949, 'employs': 2950, 'acts': 2951, 'obligations': 2952, 'fight': 2953, 'arm': 2954, '43': 2955, 'jet': 2956, 'boca': 2957, 'springs': 2958, 'guests': 2959, 'iii': 2960, 'governor': 2961, 'banned': 2962, 'drivers': 2963, 'race': 2964, 'pointed': 2965, 'buses': 2966, 'town': 2967, 'february': 2968, 'disputes': 2969, '71': 2970, 'warner': 2971, 'alan': 2972, 'spoon': 2973, 'guaranteed': 2974, '7.5': 2975, 'matters': 2976, 'audit': 2977, 'decrease': 2978, 'leaves': 2979, 'illuminating': 2980, 'bidders': 2981, 'hartford': 2982, 'values': 2983, 'withdraw': 2984, 'withdrawal': 2985, 'speed': 2986, '5.5': 2987, 'negotiations': 2988, '52': 2989, 'retailing': 2990, 'collected': 2991, '195': 2992, '140': 2993, 'braidwood': 2994, 'passenger': 2995, 'medium-sized': 2996, 'benefited': 2997, 'doubled': 2998, 'household': 2999, 'sheet': 3000, 'attached': 3001, 'barnum': 3002, 'theory': 3003, 'jump': 3004, 'link': 3005, 'minute': 3006, 'processors': 3007, 'roll': 3008, 'competitor': 3009, 'compete': 3010, 'sheets': 3011, 'favored': 3012, 'incurred': 3013, 'contractor': 3014, 'arrangement': 3015, '*-25': 3016, 'messrs.': 3017, 'joseph': 3018, '37': 3019, 'arthur': 3020, 'success': 3021, 'removed': 3022, 'arabia': 3023, 'allegedly': 3024, 'carla': 3025, 'accelerated': 3026, 'scrutiny': 3027, 'investing': 3028, 'task': 3029, 'teams': 3030, 'procedures': 3031, 'copyright': 3032, 'protect': 3033, 'films': 3034, 'taipei': 3035, 'italy': 3036, 'recognition': 3037, 'concerned': 3038, 'disturbing': 3039, '64': 3040, 'carlos': 3041, 'pc': 3042, 'allen': 3043, '1975': 3044, 'disk': 3045, 'faulding': 3046, 'australian': 3047, 'pharmaceuticals': 3048, 'moleculon': 3049, 'bass': 3050, 'esso': 3051, '10.2': 3052, 'timex': 3053, 'requested': 3054, 'stepping': 3055, 'overhead': 3056, 'curb': 3057, 'load': 3058, 'controlling': 3059, 'snapped': 3060, 'floating-rate': 3061, 'interbank': 3062, 'semiannual': 3063, 'montedison': 3064, 'pursuant': 3065, 'drawing': 3066, 'pick': 3067, 'craze': 3068, 'closed-end': 3069, 'capped': 3070, 'guinea': 3071, 'mass': 3072, '1929': 3073, 'billions': 3074, 'barney': 3075, 'tend': 3076, 'historically': 3077, 'premiums': 3078, 'partially': 3079, 'rich': 3080, 'newly': 3081, 'viewpoint': 3082, 'cast': 3083, 'plenty': 3084, 'depositary': 3085, 'receipts': 3086, 'folks': 3087, 'permitted': 3088, 'reflection': 3089, '47.6': 3090, '2.1': 3091, 'remove': 3092, 'effects': 3093, 'usual': 3094, 'patterns': 3095, 'revive': 3096, 'draw': 3097, 'signals': 3098, '0.1': 3099, 'soft': 3100, 'excessive': 3101, '1.6': 3102, 'predicted': 3103, 'homes': 3104, 'contrast': 3105, 'counts': 3106, 'bretz': 3107, 'stamford': 3108, 'suppliers': 3109, '*t*-51': 3110, 'odd': 3111, 'coupled': 3112, 'reader': 3113, 'belong': 3114, 'else': 3115, 'wood': 3116, 'wa': 3117, 'baseball': 3118, 'describes': 3119, 'ballplayers': 3120, 'commitment': 3121, 'symbol': 3122, 'played': 3123, 'shame': 3124, 'strict': 3125, 'complaint': 3126, 'style': 3127, '*exp*-3': 3128, 'committed': 3129, 'globe': 3130, 'restaurants': 3131, 'sports': 3132, 'pressured': 3133, 'olympic': 3134, 'aids': 3135, 'hospital': 3136, 'tested': 3137, '*-64': 3138, 'polish': 3139, 'damage': 3140, 'periods': 3141, 'statistics': 3142, 'underwriters': 3143, 'shot': 3144, 'revived': 3145, 'feeling': 3146, '*-73': 3147, 'mother': 3148, 'turns': 3149, 'strongly': 3150, 'beaten': 3151, 'harsh': 3152, 'marie-louise': 3153, 'executed': 3154, 'killed': 3155, 'husband': 3156, 'presented': 3157, 'nice': 3158, 'deserve': 3159, 'scenes': 3160, 'mention': 3161, 'spot': 3162, 'seeing': 3163, 'discussing': 3164, 'regular': 3165, '767': 3166, 'stages': 3167, 'event': 3168, 'presidential': 3169, 'era': 3170, 'consultant': 3171, 'convicted': 3172, 'cop-killer': 3173, 'bullets': 3174, 'telling': 3175, 'classic': 3176, 'agree': 3177, 'flag': 3178, 'freedom': 3179, 'denies': 3180, 'dynamics': 3181, 'remainder': 3182, 'season': 3183, 'truce': 3184, 'attacks': 3185, 'pinocchio': 3186, 'photograph': 3187, 'nose': 3188, 'barrel': 3189, 'cooperation': 3190, 'pursued': 3191, 'starts': 3192, 'fresh': 3193, 'economies': 3194, 'hormats': 3195, 'spurred': 3196, 'accommodate': 3197, 'baker': 3198, 'intention': 3199, 'shape': 3200, 'architecture': 3201, 'optimism': 3202, 'understanding': 3203, 'functions': 3204, 'bases': 3205, 'graders': 3206, 'crib': 3207, 'surrendered': 3208, 'low-ability': 3209, 'gone': 3210, 'guilty': 3211, 'teach': 3212, 'statute': 3213, 'greatly': 3214, 'booming': 3215, 'macmillan\\/mcgraw-hill': 3216, 'academic': 3217, 'cannell': 3218, 'seriously': 3219, 'worries': 3220, 'textile': 3221, 'serving': 3222, 'restore': 3223, 'educational': 3224, 'restructured': 3225, 'curriculum': 3226, 'fast': 3227, 'taught': 3228, 'creation': 3229, 'civilization': 3230, 'feared': 3231, 'meant': 3232, 'eager': 3233, 'studies': 3234, 'badly': 3235, 'heart': 3236, 'adding': 3237, 'quiet': 3238, 'resignation': 3239, 'offenders': 3240, 'doubt': 3241, 'mathematics': 3242, 'reinstatement': 3243, 'afraid': 3244, 'sentiment': 3245, 'worksheets': 3246, 'parallels': 3247, 'publishes': 3248, 'improving': 3249, 'michigan': 3250, 'fifth-grade': 3251, 'metric': 3252, 'intent': 3253, 'ignoring': 3254, 'represented': 3255, 'sacramento': 3256, 'north': 3257, 'moratorium': 3258, 'fetal': 3259, 'transplant': 3260, 'involve': 3261, 'hhs': 3262, 'controversy': 3263, 'ethical': 3264, 'controlled': 3265, 'names': 3266, 'consideration': 3267, 'antonio': 3268, 'charged': 3269, 'mechanism': 3270, 'middle': 3271, 'regions': 3272, 'speculators': 3273, 'immediately': 3274, 'otc': 3275, 'bancorp': 3276, 'fared': 3277, 'eliminates': 3278, 'dramatic': 3279, 'tumbled': 3280, 'redeemed': 3281, 'resolve': 3282, 'sci': 3283, 'detailed': 3284, 'organized': 3285, 'stem': 3286, 'hire': 3287, 'circumstances': 3288, 'filling': 3289, 'gerald': 3290, 'resolution': 3291, 'permission': 3292, 'obtained': 3293, 'develop': 3294, 'confidential': 3295, 'victim': 3296, 'lucrative': 3297, 'calif': 3298, 'couple': 3299, 'eye': 3300, 'high-priced': 3301, 'laughing': 3302, 'darkhorse': 3303, 'employed': 3304, 'samnick': 3305, 'prohibits': 3306, 'promote': 3307, 'inquiry': 3308, 'homosexual': 3309, 'hampton': 3310, 'impose': 3311, 'principle': 3312, 'consist': 3313, 'feet': 3314, 'equivalent': 3315, 'fair': 3316, 'investigating': 3317, 'reputation': 3318, 'accepting': 3319, 'touch': 3320, 'insisted': 3321, 'contacted': 3322, 'backe': 3323, 'structural': 3324, 'logic': 3325, 'switch': 3326, 'traditionally': 3327, 'driving': 3328, '1.8500': 3329, '143.80': 3330, 'contends': 3331, 'cites': 3332, '3.5': 3333, 'ounces': 3334, 'station': 3335, 'n.c.': 3336, 'disappointment': 3337, 'crisis': 3338, 'premiere': 3339, 'plastic': 3340, '*-80': 3341, 'printed': 3342, 'surprisingly': 3343, 'advertise': 3344, 'publisher': 3345, 'sleep': 3346, 'interpublic': 3347, 'breaks': 3348, 'pittsburgh': 3349, 'buy-back': 3350, 'proceeds': 3351, 'ag': 3352, 'device': 3353, 'advocates': 3354, 'steps': 3355, 'automobiles': 3356, 'ongoing': 3357, 'address': 3358, 'automatic': 3359, 'pounds': 3360, 'rear-seat': 3361, 'shoulder': 3362, 'railcars': 3363, 'opposition': 3364, 'passage': 3365, 'slump': 3366, 'warren': 3367, 'stag': 3368, 'shops': 3369, 'vineyard': 3370, 'bordeaux': 3371, 'romanee-conti': 3372, 'tache': 3373, 'roederer': 3374, 'cristal': 3375, 'chateau': 3376, 'barrier': 3377, 'larry': 3378, 'shapiro': 3379, 'originally': 3380, 'merchants': 3381, 'check': 3382, 'responses': 3383, 'thinks': 3384, 'movement': 3385, 'moment': 3386, 'looming': 3387, 'agrees': 3388, 'kansas': 3389, 'hopefully': 3390, 'expecting': 3391, '0.25': 3392, 'exercised': 3393, 'copperweld': 3394, 'basham': 3395, '13.8': 3396, 'fails': 3397, 'auctioned': 3398, 'foods': 3399, '2645.90': 3400, 'advancing': 3401, '3.18': 3402, 'pattern': 3403, '9.9': 3404, 'subordinated': 3405, 'hundred': 3406, 'respond': 3407, 'acne': 3408, 'researcher': 3409, 'barriers': 3410, 'half-hour': 3411, 'notably': 3412, '1.2': 3413, 'koito': 3414, 'structures': 3415, 'recommendations': 3416, 'quick': 3417, 'high-tech': 3418, 'secured': 3419, 'skin': 3420, 'fit': 3421, 'wakui': 3422, 'goal': 3423, 'performed': 3424, 'extent': 3425, 'hudson': 3426, 'stearn': 3427, 'maintenance': 3428, 'convenient': 3429, 'checking': 3430, 'loyalty': 3431, 'targeting': 3432, 'moore': 3433, 'stepped': 3434, 'branches': 3435, 'panama': 3436, 'market-share': 3437, 'n.c': 3438, 'costly': 3439, 'sharedata': 3440, 'registration': 3441, '500,000': 3442, '2.3': 3443, 'chevrolet': 3444, 'demise': 3445, 'performing': 3446, 'ranieri': 3447, 'delay': 3448, 'worse': 3449, 'stemming': 3450, 'halts': 3451, 'one-hour': 3452, 'industrials': 3453, '12-point': 3454, 'panic': 3455, 'intermediate': 3456, 'executing': 3457, 'shut': 3458, 'regulated': 3459, 'dorothy': 3460, 'aslacton': 3461, 'ancient': 3462, 'stone': 3463, 'modern': 3464, 'hammond': 3465, 'youth': 3466, 'methods': 3467, 'chamber': 3468, 'absorbed': 3469, 'belfry': 3470, 'attend': 3471, 'promptly': 3472, 'reopen': 3473, 'nearby': 3474, 'fault': 3475, 'crunch': 3476, 'bang': 3477, 'drawn': 3478, 'weekly': 3479, 'kill': 3480, 'boesel': 3481, '500-stock': 3482, 'watchers': 3483, '3.3': 3484, 'rebound': 3485, 'w.': 3486, 'coxon': 3487, 'einhorn': 3488, 'weaken': 3489, 'trailed': 3490, 'deliver': 3491, 'appreciation': 3492, 'plight': 3493, 'berliner': 3494, 'beverly': 3495, 'kingdom': 3496, 'historic': 3497, 'interference': 3498, 'yang': 3499, 'respect': 3500, 'occurred': 3501, 'exchanges': 3502, 'arms': 3503, 'protests': 3504, 'tree': 3505, 'everything': 3506, 'compensation': 3507, 'changing': 3508, 'accept': 3509, 'continually': 3510, 'favorable': 3511, 'assumption': 3512, 'expelled': 3513, 'vargas': 3514, 'listing': 3515, 'otero': 3516, 'triton': 3517, 'crane': 3518, 'glenn': 3519, 'canepa': 3520, 'dell': 3521, 'meaning': 3522, 'markdown': 3523, 'fred': 3524, 'davis': 3525, 'amounted': 3526, 'scott': 3527, '45,000': 3528, 'jamaica': 3529, 'compliance': 3530, '2,000': 3531, 'randolph': 3532, 'las': 3533, 'vegas': 3534, 'nev.': 3535, 'furor': 3536, 'abroad': 3537, 'scrutinizing': 3538, 'suspend': 3539, 'maybe': 3540, 'strategies': 3541, 'ft-se': 3542, 'compares': 3543, 'backed': 3544, 'proponents': 3545, 'opponents': 3546, 'nickel': 3547, 'impossible': 3548, 'replacement': 3549, 'completion': 3550, 'knew': 3551, 'joining': 3552, 'transition': 3553, 'swiftly': 3554, 'turnaround': 3555, 'repair': 3556, 'match': 3557, 'waters': 3558, 'african': 3559, 'variety': 3560, 'quota': 3561, 'gray': 3562, 'caribbean': 3563, 'conservative': 3564, '17.3': 3565, 'beauty': 3566, 'railing': 3567, 'f': 3568, 'upset': 3569, 'similarly': 3570, 'competes': 3571, 'designers': 3572, 'eggers': 3573, 'cell': 3574, 'efficient': 3575, 'penalty': 3576, 'fairless': 3577, '1,500': 3578, 'electrical': 3579, 'citations': 3580, 'severe': 3581, 'discrepancies': 3582, 'edition': 3583, 'illness': 3584, 'm.d.': 3585, 'wright': 3586, 'combination': 3587, 'yourself': 3588, 'namely': 3589, 'conversion': 3590, 'voa': 3591, 'copy': 3592, 'moines': 3593, 'mccormick': 3594, 'helpful': 3595, 'duty': 3596, 'conclude': 3597, 'ought': 3598, 'gordon': 3599, 'responsibilities': 3600, 'mcmillin': 3601, 'operation': 3602, 'dominated': 3603, 'pushed': 3604, 'disappointing': 3605, '2.8': 3606, 'unsecured': 3607, '8.50': 3608, '800': 3609, 'alfred': 3610, 'divisions': 3611, 'rumored': 3612, 'respected': 3613, 'prerogatives': 3614, 'specified': 3615, 'understand': 3616, 'presidency': 3617, 'supported': 3618, 'framers': 3619, 'perform': 3620, 'declaring': 3621, 'appointments': 3622, 'recess': 3623, 'choosing': 3624, 'rider': 3625, 'agricultural': 3626, 'v.': 3627, 'gorbachev': 3628, 'midwestern': 3629, 'plains': 3630, 'crops': 3631, '2.7': 3632, '16.7': 3633, 'struggling': 3634, 'examiner': 3635, 'upscale': 3636, 'abandoned': 3637, 'bradley': 3638, 'encouraging': 3639, 'riese': 3640, 'considers': 3641, 'responsible': 3642, 'interest-rate': 3643, '600': 3644, 'peabody': 3645, 'contel': 3646, 'blue-chip': 3647, 'brothers': 3648, 'psychology': 3649, 'constantly': 3650, 'fundamental': 3651, 'ultimate': 3652, '*ppa*-3': 3653, 'offsetting': 3654, 'prints': 3655, 'stability': 3656, 'break': 3657, 'patients': 3658, 'blood': 3659, 'israel': 3660, 'follow': 3661, 'egnuss': 3662, 'weather': 3663, 'magnetic': 3664, 'multi-crystal': 3665, 'crystal': 3666, 'citadel': 3667, 'taylor': 3668, '750': 3669, 'rated': 3670, 'obligation': 3671, '6.20': 3672, '7.272': 3673, 'serial': 3674, 'ana': 3675, '6.40': 3676, '7.458': 3677, '2029': 3678, '1994': 3679, 'francs': 3680, 'guarantee': 3681, 'candlestick': 3682, 'stadiums': 3683, 'ballpark': 3684, 'backers': 3685, 'longer-term': 3686, 'collapse': 3687, 'squeeze': 3688, 'proving': 3689, 'liable': 3690, 'liability': 3691, 'hymowitz': 3692, 'pill': 3693, 'justices': 3694, 'dsm': 3695, 'cataract': 3696, 'el': 3697, '1.65': 3698, 'oshkosh': 3699, 'truck': 3700, 'chassis': 3701, 'edged': 3702, '8.5': 3703, 'profit-taking': 3704, 'indication': 3705, 'frankfurt': 3706, 'unemployment': 3707, '22\\/32': 3708, '2\\/32': 3709, 'earthquake': 3710, 'property\\/casualty': 3711, 'weaker': 3712, 'cost-sharing': 3713, 'acid-rain': 3714, 'russell': 3715, 'mix': 3716, 'skeptical': 3717, 'financial-services': 3718, '0.82': 3719, 'specify': 3720, 'tendered': 3721, '1989-90': 3722, '1.25': 3723, 'bougainville': 3724, 'savin': 3725, 'rms': 3726, 'finland': 3727, 'coleco': 3728, 'unveiled': 3729, 'crum': 3730, 'forster': 3731, 'treating': 3732, 'rulings': 3733, 'anti-dumping': 3734, 'scammers': 3735, 'oy': 3736, 'markkaa': 3737, 'nichol': 3738, 'battery': 3739, 'barges': 3740, 'ports': 3741, 'missouri': 3742, 'stockpiles': 3743, 'silver': 3744, 'steelmakers': 3745, 'professors': 3746, 'vinken': 3747, 'consolidated': 3748, 'exposed': 3749, 'unusually': 3750, '1956': 3751, 'forum': 3752, 'smokers': 3753, 'useful': 3754, 'users': 3755, '1950s': 3756, 'filter': 3757, '9.8': 3758, 'lung': 3759, 'striking': 3760, 'industrialized': 3761, 'hollingsworth': 3762, 'vose': 3763, '*t*-6': 3764, 'regulate': 3765, 'chrysotile': 3766, '*t*-7': 3767, 'mossman': 3768, 'body': 3769, '*t*-8': 3770, '*-8': 3771, 'exhaust': 3772, 'contracted': 3773, 'phillips': 3774, 'taxable': 3775, '8.45': 3776, '8.47': 3777, '*-9': 3778, 'sooner': 3779, 'nevertheless': 3780, 'blip': 3781, '8.04': 3782, 'vary': 3783, '*t*-9': 3784, 'slid': 3785, '*t*-10': 3786, '*-10': 3787, 'succeeds': 3788, '*t*-11': 3789, 'year-end': 3790, 'finmeccanica': 3791, 'state-owned': 3792, 'lifted': 3793, 'midnight': 3794, '2.87': 3795, '*-11': 3796, 'capital-gains': 3797, 'vitulli': 3798, '*-12': 3799, 'mazda': 3800, 'resort': 3801, 'towns': 3802, 'raton': 3803, 'rock': 3804, 'stars': 3805, 'rusty': 3806, 'du': 3807, 'pont': 3808, 'victor': 3809, 'speedway': 3810, 'welcomed': 3811, 'buffet': 3812, 'museum': 3813, 'visitors': 3814, 'exhibition': 3815, 'sponsor': 3816, 'downtown': 3817, 'squeezed': 3818, 'dinner': 3819, 'roof': 3820, 'chefs': 3821, 'meal': 3822, 'ceos': 3823, '101': 3824, 'setback': 3825, 'casting': 3826, 'mere': 3827, 'boom': 3828, '*t*-12': 3829, 'prolonged': 3830, 'conflicts': 3831, '68': 3832, 'recorded': 3833, 'surplus': 3834, '*t*-13': 3835, 'permanent': 3836, 'weeklies': 3837, 'fierce': 3838, 'announce': 3839, 'credits': 3840, 'reward': 3841, 'bonuses': 3842, 'bureau': 3843, '*t*-14': 3844, '*t*-15': 3845, '2.6': 3846, 'bowed': 3847, '*rnr*-4': 3848, 'justify': 3849, '*-13': 3850, 'haven': 3851, '*-14': 3852, 'conn': 3853, 'rewards': 3854, '*t*-16': 3855, '*t*-17': 3856, '4.8': 3857, 'asserted': 3858, '3.75': 3859, 'norman': 3860, 'frederick': 3861, '*-15': 3862, 'daniel': 3863, 'undersecretary': 3864, 'refunds': 3865, '*t*-18': 3866, '*t*-19': 3867, 'feb.': 3868, 'appealing': 3869, '*-16': 3870, 'collections': 3871, '1.55': 3872, '*t*-20': 3873, 'byron': 3874, 'rockford': 3875, '*-17': 3876, 'unreasonable': 3877, 'collecting': 3878, '190': 3879, 'faces': 3880, 'automobile': 3881, 'inched': 3882, '*t*-21': 3883, 'manufacture': 3884, 'creativity': 3885, '*-18': 3886, '64-year-old': 3887, '*-19': 3888, 'strings': 3889, 'choice': 3890, 'gregory': 3891, '*-20': 3892, '*t*-22': 3893, 'anticipates': 3894, 'scenario': 3895, '38': 3896, 'valuation': 3897, '*-21': 3898, 'smaby': 3899, 'describe': 3900, '*-22': 3901, 'fragile': 3902, '*t*-23': 3903, '*t*-24': 3904, 'supercomputers': 3905, 'presumably': 3906, '*-23': 3907, 'calculate': 3908, 'transferring': 3909, '*t*-25': 3910, 'drain': 3911, '*-24': 3912, 'colorado': 3913, '600,000': 3914, '*-26': 3915, 'neil': 3916, 'malcolm': 3917, '*t*-26': 3918, 'stevens': 3919, '*-27': 3920, 'hatch': 3921, '*-28': 3922, '*-29': 3923, 'claiming': 3924, 'stiff': 3925, 'intellectual': 3926, 'realization': 3927, 'citizens': 3928, 'negotiators': 3929, 'inadequate': 3930, 'hurting': 3931, 'officers': 3932, 'trained': 3933, 'instituted': 3934, 'introducing': 3935, 'vowed': 3936, 'completely': 3937, 'hook': 3938, '*t*-27': 3939, 'deemed': 3940, 'pose': 3941, 'threat': 3942, '*t*-28': 3943, 'turkey': 3944, 'videocassette': 3945, 'merit': 3946, 'argentina': 3947, 'creditor': 3948, 'argentine': 3949, 'stature': 3950, '*t*-29': 3951, 'feels': 3952, 'solved': 3953, '*-31': 3954, 'forgiven': 3955, '*t*-30': 3956, '*t*-31': 3957, 'computing': 3958, '*-32': 3959, 'screens': 3960, 'stored': 3961, '*-33': 3962, 'steven': 3963, 'store': 3964, 'memories': 3965, 'counterparts': 3966, 'pioneer': 3967, 'gates': 3968, 'versions': 3969, '*t*-32': 3970, 'drives': 3971, 'dennis': 3972, '*t*-33': 3973, 'kalipharma': 3974, '*t*-34': 3975, 'voting': 3976, '*-34': 3977, '11,000': 3978, '*-35': 3979, 'exxon': 3980, '*-36': 3981, 'excise': 3982, 'r.p.': 3983, 'scherer': 3984, 'optical': 3985, '*t*-35': 3986, 'quantities': 3987, 'virgin': 3988, 'islands': 3989, 'petition': 3990, 'preferences': 3991, '*-37': 3992, 'tariff': 3993, 'seller': 3994, 'assembled': 3995, '*t*-36': 3996, 'satisfactory': 3997, 'ambitious': 3998, '37.5': 3999, 'c$': 4000, 'unsuccessfully': 4001, 'influential': 4002, '*-38': 4003, 'lord': 4004, '*-39': 4005, 'single-handedly': 4006, 'totaling': 4007, '*-40': 4008, '*-41': 4009, '*-42': 4010, 'daiwa': 4011, 'linked': 4012, 'advantages': 4013, 'prepayment': 4014, '*-43': 4015, 'burden': 4016, 'attractive': 4017, 'rapid': 4018, 'purchasers': 4019, '2.4': 4020, '*-44': 4021, 'lawsuits': 4022, 'erbamont': 4023, 'netherlands': 4024, 'advertised': 4025, '72': 4026, '*-45': 4027, 'currencies': 4028, 'protracted': 4029, 'intensity': 4030, '*t*-37': 4031, 'invest': 4032, 'simon': 4033, 'chile': 4034, 'philippine': 4035, '*-46': 4036, 'kicked': 4037, 'newgate': 4038, 'explosion': 4039, '1920s': 4040, '*t*-38': 4041, 'brings': 4042, '*t*-39': 4043, 'harris': 4044, 'upham': 4045, 'scrambled': 4046, '*t*-40': 4047, 'planners': 4048, 'diversify': 4049, '*-47': 4050, '*t*-41': 4051, 'fat': 4052, '*t*-42': 4053, '*t*-43': 4054, 'skyrocketed': 4055, 'startling': 4056, 'targeted': 4057, '*-48': 4058, '*t*-44': 4059, 'nonetheless': 4060, '*t*-45': 4061, 'advice': 4062, 'ready': 4063, '188': 4064, 'repaid': 4065, '*-49': 4066, '*-50': 4067, 'crippled': 4068, '*-51': 4069, 'lent': 4070, 'monetary': 4071, 'belongs': 4072, 'stressed': 4073, 'obstacles': 4074, 'hundreds': 4075, 'leveling': 4076, 'manufactured': 4077, 'kenneth': 4078, '*t*-46': 4079, 'slack': 4080, 'payrolls': 4081, 'cite': 4082, 'imbalances': 4083, '*t*-47': 4084, 'clues': 4085, 'provoke': 4086, '*t*-48': 4087, 'landing': 4088, 'platt': 4089, '*t*-49': 4090, '0.3': 4091, 'climbing': 4092, '0.9': 4093, '3.9': 4094, '5.4': 4095, 'backlogs': 4096, 'excluding': 4097, '*t*-50': 4098, 'single-family': 4099, '4.3': 4100, '88': 4101, 'adjusting': 4102, '*-53': 4103, '*-54': 4104, 'renovation': 4105, '*-55': 4106, 'steeper': 4107, 'handle': 4108, '*-56': 4109, 'polled': 4110, '73': 4111, 'row': 4112, 'shortage': 4113, 'exceptionally': 4114, 'quotas': 4115, 'pamela': 4116, 'seasonally': 4117, 'americana': 4118, 'murakami': 4119, 'kodansha': 4120, 'characters': 4121, 'careers': 4122, 'marriages': 4123, 'engaging': 4124, '*t*-52': 4125, '*t*-53': 4126, 'notion': 4127, '*-57': 4128, 'hero': 4129, 'star': 4130, 'stanford': 4131, 'degree': 4132, '*t*-54': 4133, 'christian': 4134, '*t*-55': 4135, 'phone': 4136, 'sweet': 4137, '*t*-56': 4138, 'beatles': 4139, '*t*-57': 4140, 'gotta': 4141, 'macmillan': 4142, '17.95': 4143, 'nipponese': 4144, 'mirror': 4145, 'harmony': 4146, 'polls': 4147, 'unrecognizable': 4148, 'foul': 4149, 'zone': 4150, 'depending': 4151, 'sidestep': 4152, 'defeat': 4153, 'amusing': 4154, 'fare': 4155, 'enormous': 4156, 'sums': 4157, 'plate': 4158, '228': 4159, '*t*-58': 4160, 'regret': 4161, '*-59': 4162, 'letting': 4163, '*t*-59': 4164, 'science': 4165, 'ultimately': 4166, 'venerable': 4167, 'akio': 4168, 'morita': 4169, 'entering': 4170, 'discos': 4171, 'clubs': 4172, '*-60': 4173, 'visiting': 4174, 'designated': 4175, 'materialistic': 4176, 'colony': 4177, '*t*-60': 4178, '*t*-61': 4179, 'endorsed': 4180, '*t*-62': 4181, '*-61': 4182, 'balked': 4183, '*-62': 4184, 'palestinian': 4185, 'sex': 4186, '*t*-63': 4187, '*-63': 4188, 'compensate': 4189, 'coal': 4190, 'victory': 4191, 'environmentalists': 4192, 'terminated': 4193, 'nagymaros': 4194, 'nemeth': 4195, 'modify': 4196, '*t*-64': 4197, '*-67': 4198, '*-68': 4199, 'operated': 4200, '*-69': 4201, 'peak': 4202, 'stockholm': 4203, 'weddings': 4204, '6,000': 4205, 'rings': 4206, 'bramalea': 4207, 'toronto-based': 4208, 'warrant': 4209, 'entitles': 4210, 'holder': 4211, '*-70': 4212, '*t*-65': 4213, 'sketch': 4214, 'piece': 4215, 'dialogue': 4216, '*-71': 4217, 'silent': 4218, '*t*-66': 4219, 'living': 4220, 'double': 4221, 'music': 4222, 'score': 4223, '*t*-67': 4224, 'romanticized': 4225, 'strip': 4226, 'avenue': 4227, 'crack': 4228, 'cardboard': 4229, 'routine': 4230, 'spends': 4231, 'condemned': 4232, 'competing': 4233, '*-72': 4234, 'blind': 4235, 'girl': 4236, 'returning': 4237, 'murdered': 4238, '*-74': 4239, 'blessing': 4240, 'stakes': 4241, 'romance': 4242, '*t*-68': 4243, '*t*-69': 4244, 'apartment': 4245, 'ends': 4246, 'rough': 4247, 'voices': 4248, 'chabrol': 4249, 'brilliant': 4250, '*t*-70': 4251, 'character': 4252, 'sympathetic': 4253, 'angle': 4254, 'historical': 4255, 'vichy': 4256, 'germans': 4257, 'abortionist': 4258, 'openly': 4259, 'latour': 4260, 'recommend': 4261, 'confused': 4262, 'fighting': 4263, 'bright': 4264, 'energetic': 4265, '*t*-71': 4266, 'tip': 4267, 'boeing': 4268, 'specific': 4269, 'sections': 4270, 'aircraft': 4271, 'election': 4272, 'contests': 4273, 'content': 4274, '*t*-72': 4275, 'stage': 4276, 'bold': 4277, '*t*-73': 4278, 'tone': 4279, 'fills': 4280, 'candidate': 4281, 'roger': 4282, 'ailes': 4283, 'links': 4284, 'kicker': 4285, 'corruption': 4286, '*t*-74': 4287, 'exist': 4288, 'stung': 4289, 'consultants': 4290, 'compare': 4291, 'banning': 4292, 'pro-choice': 4293, 'truth': 4294, 'everybody': 4295, 'nobody': 4296, '*t*-75': 4297, 'accurate': 4298, 'voluntarily': 4299, 'admitted': 4300, 'reservations': 4301, 'gov.': 4302, 'tradition': 4303, '*t*-76': 4304, 'referendum': 4305, 'advertisements': 4306, '*t*-77': 4307, 'attracted': 4308, 'featured': 4309, 'courtroom': 4310, '*t*-78': 4311, 'younger': 4312, 'tired': 4313, 'scientist': 4314, 'campaigning': 4315, 'devote': 4316, 'featuring': 4317, 'hazardous': 4318, 'suing': 4319, 'fraud': 4320, 'partisans': 4321, 'photographs': 4322, 'shrinks': 4323, 'salmore': 4324, 'devastating': 4325, 'credibility': 4326, 're-election': 4327, "o'connor": 4328, 'gop': 4329, 'nelson': 4330, '350,000': 4331, '16,000': 4332, 'friday': 4333, 'aug.': 4334, 'domination': 4335, 'tripled': 4336, 'steep': 4337, 'pumping': 4338, 'wages': 4339, 'forcing': 4340, 'contributing': 4341, 'recipient': 4342, 'spur': 4343, 'constraints': 4344, 'evolution': 4345, 'concentrated': 4346, '1990s': 4347, 'donor': 4348, 'graduate': 4349, 'parallel': 4350, 'princeton': 4351, 'indonesia': 4352, 'pull': 4353, 'crucial': 4354, 'caution': 4355, 'attitude': 4356, 'leases': 4357, 'regard': 4358, 'desirable': 4359, 'lee': 4360, 'cathryn': 4361, 'rice': 4362, 'eyes': 4363, 'social-studies': 4364, 'protest': 4365, 'nancy': 4366, 'classes': 4367, '*t*-79': 4368, 'breach': 4369, 'pleaded': 4370, '*t*-80': 4371, 'inspired': 4372, '*t*-81': 4373, 'defended': 4374, 'treated': 4375, 'harshly': 4376, 'casts': 4377, 'temptation': 4378, 'violated': 4379, 'enforce': 4380, 'albuquerque': 4381, 'n.m.': 4382, 'outright': 4383, 'surfaced': 4384, 'suspects': 4385, 'adult': 4386, 'statewide': 4387, 'test-coaching': 4388, 'instruction': 4389, 'concentrate': 4390, 'promotions': 4391, 'stressing': 4392, 'allegations': 4393, 'run-down': 4394, 'governors': 4395, 'physicist': 4396, '*ppa*-1': 4397, 'violence': 4398, 'linda': 4399, 'immediate': 4400, 'predecessor': 4401, 'suffered': 4402, 'nervous': 4403, 'evenly': 4404, 'neighborhoods': 4405, 'faculty': 4406, 'behalf': 4407, 'ambitions': 4408, 'reformers': 4409, 'loved': 4410, 'favorite': 4411, 'cadet': 4412, 'advised': 4413, 'lady': 4414, 'studying': 4415, 'marchand': 4416, 'furniture': 4417, 'football': 4418, 'fellow': 4419, 'pushing': 4420, 'earn': 4421, 'meaningful': 4422, 'elizabeth': 4423, '*t*-92': 4424, 'seminar': 4425, 'mistake': 4426, 'correct': 4427, 'subjects': 4428, 'whole': 4429, 'alive': 4430, 'somebody': 4431, 'blow': 4432, 'joe': 4433, 'disclosure': 4434, 'save': 4435, 'certificate': 4436, 'supportive': 4437, 'callers': 4438, 'murder': 4439, 'interviewed': 4440, 'explain': 4441, 'angry': 4442, 'harm': 4443, 'damn': 4444, 'wisdom': 4445, 'jury': 4446, 'instances': 4447, 'communication': 4448, 'test-preparation': 4449, 'coaching': 4450, 'gauge': 4451, 'grade': 4452, 'metropolitan': 4453, 'florida': 4454, 'maryland': 4455, 'kean': 4456, 'mehrens': 4457, 'scale': 4458, 'measured': 4459, 'subskills': 4460, '69': 4461, 'rick': 4462, 'brownell': 4463, 'format': 4464, 'deny': 4465, 'alleghany': 4466, 's&l': 4467, 'tissue': 4468, 'alzheimer': 4469, 'parkinson': 4470, 'anti-abortionists': 4471, 'indefinitely': 4472, 'institutes': 4473, 'implications': 4474, 'suffering': 4475, 'hampered': 4476, 'prominent': 4477, 'fill': 4478, 'novello': 4479, 'nominated': 4480, 'assured': 4481, 'ideological': 4482, 'uncharted': 4483, 'genel': 4484, 'associate': 4485, 'dean': 4486, 'yale': 4487, 'conducting': 4488, 'climate': 4489, 'visible': 4490, 'flap': 4491, 'tissues': 4492, 'genes': 4493, 'summary': 4494, 'paltry': 4495, 'turnover': 4496, '145': 4497, 'tracks': 4498, 'merge': 4499, 'merged': 4500, 'jennison': 4501, 'ed': 4502, '70.7': 4503, '89.9': 4504, 'mobile': 4505, 'dan': 4506, 'core': 4507, 'fluctuation': 4508, 'achieve': 4509, '57.50': 4510, 'acquiring': 4511, 'restaurant': 4512, 'burt': 4513, 'sugarman': 4514, '42.5': 4515, 'nine-member': 4516, 'warnings': 4517, 'attorney-client': 4518, 'privilege': 4519, '8300': 4520, 'punishable': 4521, 'misdemeanor': 4522, 'felony': 4523, 'neal': 4524, 'necessarily': 4525, 'retained': 4526, 'spark': 4527, 'chaired': 4528, 'grand': 4529, 'stance': 4530, 'submit': 4531, 'sending': 4532, 'salaries': 4533, 'ramirez': 4534, 'refusal': 4535, 'unjust': 4536, 'sudden': 4537, 'relegated': 4538, 'cartoonist': 4539, 'alleging': 4540, 'mounted': 4541, 'harass': 4542, 'crossing': 4543, 'picket': 4544, 'involves': 4545, 'co-owner': 4546, 'illegally': 4547, 'threats': 4548, 'unjustified': 4549, 'family-planning': 4550, 'title': 4551, 'x': 4552, 'assist': 4553, 'counseling': 4554, 'bias': 4555, 'defendant': 4556, 'killing': 4557, 'boys': 4558, 'fairness': 4559, 'judiciary': 4560, 'arguments': 4561, 'wayne': 4562, 'prosecution': 4563, 'brown': 4564, 'bromwich': 4565, 'oliver': 4566, 'served': 4567, 'cooper': 4568, 'rubber': 4569, 'ga.': 4570, 'bridgestone\\/firestone': 4571, '1.8': 4572, 'square': 4573, 'acres': 4574, 'apology': 4575, 'cutthroat': 4576, 'competitors': 4577, 'lottery': 4578, 'understood': 4579, 'takuma': 4580, 'contrary': 4581, 'situations': 4582, 'tense': 4583, 'foreigners': 4584, 'minus': 4585, 'hiroshima': 4586, 'library': 4587, 'nagano': 4588, 'prefecture': 4589, 'papers': 4590, 'atlantic': 4591, 'cbs': 4592, 'ntg': 4593, 'retaining': 4594, 'furukawa': 4595, 'belonging': 4596, 'dresser': 4597, 'shovels': 4598, 'compiled': 4599, 'monitor': 4600, 'b': 4601, '35.7': 4602, 'billings': 4603, 'oldest': 4604, 'six-inch': 4605, 'silicon': 4606, 'additions': 4607, 'aggressively': 4608, 'suddenly': 4609, 'positioned': 4610, 'stark': 4611, 'dead': 4612, 'plunging': 4613, 'locked': 4614, 'stalemate': 4615, 'jay': 4616, 'tom': 4617, '1.8415': 4618, '142.85': 4619, 'waiting': 4620, 'wings': 4621, 'perception': 4622, 'rolled': 4623, 'prospect': 4624, 'lock': 4625, 'drifted': 4626, 'release': 4627, 'nbc': 4628, 'debut': 4629, 'keeps': 4630, 'affiliates': 4631, 'episodes': 4632, 'distributor': 4633, 'persuade': 4634, 'gillespie': 4635, 'syndication': 4636, '*-76': 4637, 'providence': 4638, 'louisville': 4639, 'renew': 4640, '*-78': 4641, 'frankly': 4642, 'articles': 4643, 'survive': 4644, 'practical': 4645, 'scoop': 4646, 'combines': 4647, 'pieces': 4648, 'topics': 4649, 'identify': 4650, 'dumpster': 4651, 'offender': 4652, 'souper': 4653, 'combo': 4654, 'chastised': 4655, '*-79': 4656, 'pointing': 4657, '*-81': 4658, 'spenders': 4659, 'needham': 4660, 'haul': 4661, 'subscriptions': 4662, 'scared': 4663, '*-83': 4664, 'replies': 4665, '4,000': 4666, '*-84': 4667, '*-85': 4668, 'handled': 4669, 'fax': 4670, 'rubicam': 4671, 'enterprise': 4672, 'specializes': 4673, 'landor': 4674, 'pressed': 4675, 'apiece': 4676, 'hostile': 4677, 'stena': 4678, 'tiphook': 4679, 'sweetened': 4680, 'approximately': 4681, 'allocated': 4682, '*-87': 4683, 'criticized': 4684, 'conditional': 4685, 'superior': 4686, 'mired': 4687, '62.5': 4688, 'responding': 4689, 'requiring': 4690, 'roofs': 4691, 'extension': 4692, 'cargo': 4693, 'systematic': 4694, 'danforth': 4695, 'praised': 4696, 'noting': 4697, 'light-truck': 4698, 'fatalities': 4699, 'withstand': 4700, 'phasing': 4701, 'installed': 4702, 'leinonen': 4703, 'joins': 4704, 'circle': 4705, 'formal': 4706, 'driscoll': 4707, 'conversations': 4708, 'prevailing': 4709, 'altogether': 4710, '18,000': 4711, 'cosmetic': 4712, 'year-ago': 4713, 'succeeding': 4714, 'butler': 4715, 'winiarski': 4716, 'cask': 4717, 'weighed': 4718, 'category': 4719, 'superpremiums': 4720, 'perceived': 4721, 'classics': 4722, 'growths': 4723, 'burgundies': 4724, 'yquem': 4725, 'biondi-santi': 4726, 'brunello': 4727, 'tuscany': 4728, 'releases': 4729, 'vintages': 4730, 'vintage': 4731, '179': 4732, 'mesnil': 4733, 'blanc': 4734, 'blancs': 4735, '115': 4736, 'burgundy': 4737, 'richebourg': 4738, '225': 4739, '155': 4740, 'hermitage': 4741, 'command': 4742, 'marty': 4743, 'grapes': 4744, 'perfectly': 4745, 'wholesale': 4746, '*-98': 4747, 'six-packs': 4748, 'remarked': 4749, 'dramatically': 4750, 'cabernets': 4751, 'lowering': 4752, 'moderated': 4753, 'richmond': 4754, 'posting': 4755, 'stated': 4756, 'downward': 4757, 'discretionary': 4758, '*-102': 4759, 'mailing': 4760, 'integra': 4761, 'hallwood': 4762, 'steelmaker': 4763, 'reject': 4764, 'redeem': 4765, 'three-year': 4766, 'five-year': 4767, 'aim': 4768, 'lancaster': 4769, 'reames': 4770, 'marketer': 4771, 'frozen': 4772, 'fractionally': 4773, '847': 4774, '644': 4775, 'enthusiasm': 4776, 'rushed': 4777, 'eaton': 4778, 'sierra': 4779, 'restructure': 4780, 'proposals': 4781, '*ppa*-2': 4782, 'pit': 4783, 'bull': 4784, 'leming': 4785, 'valhi': 4786, 'two-thirds': 4787, 'surprised': 4788, 'collapsed': 4789, '1.85': 4790, 'license': 4791, 'albert': 4792, 'kligman': 4793, '1960s': 4794, 'licensed': 4795, 'criticism': 4796, 'color': 4797, 'disagree': 4798, 'heated': 4799, 'focusing': 4800, 'initiative': 4801, 'rhetoric': 4802, 'impending': 4803, 'publicized': 4804, 'boone': 4805, 'hay': 4806, 'anxious': 4807, 'clarify': 4808, 'miti': 4809, 'laser': 4810, 'tiny': 4811, 'putting': 4812, 'heightened': 4813, 'acceleration': 4814, 'conspicuous': 4815, 'feed': 4816, 'anxieties': 4817, 'catch': 4818, 'strategic': 4819, 'va.': 4820, 'ronald': 4821, 'bodner': 4822, 'window': 4823, 'merchandise': 4824, 'itoh': 4825, 'ventures': 4826, 'objectives': 4827, 'drive': 4828, 'businessman': 4829, 'omitted': 4830, 'assumed': 4831, 'genie': 4832, 'driskill': 4833, 'neighborhood': 4834, 'crown': 4835, 'installment': 4836, 'qualify': 4837, 'anne': 4838, 'synergistics': 4839, 'throws': 4840, 'cash-flow': 4841, 'demographic': 4842, 'macdonald': 4843, 'barnett': 4844, 'seniors': 4845, 'games': 4846, 'checks': 4847, 'promoting': 4848, 'emphasis': 4849, 'switched': 4850, 'enabling': 4851, 'deregulation': 4852, '1970s': 4853, 'certificates': 4854, 'bigger': 4855, 'unions': 4856, 'scrambling': 4857, 'define': 4858, 'jacob': 4859, 'demanding': 4860, 'alvin': 4861, 'chandler': 4862, 'ariz.': 4863, 'exercisable': 4864, '4.1': 4865, 'savings-and-loan': 4866, 'ailing': 4867, 'gift': 4868, 'slides': 4869, 'arguing': 4870, 'p.m': 4871, 'hitting': 4872, 'leo': 4873, '30-point': 4874, '*-128': 4875, 'one-day': 4876, 'five-point': 4877, 'aides': 4878, 'congressmen': 4879, 'manually': 4880, 'capitol': 4881, 'preset': 4882, 'trigger': 4883, 'congressman': 4884, '26': 4885, 'regulating': 4886, 'change-ringing': 4887, 'sayers': 4888, 'rural': 4889, 'sounded': 4890, 'peal': 4891, 'autumn': 4892, 'sunday': 4893, 'youngsters': 4894, 'ranks': 4895, 'nationwide': 4896, 'rung': 4897, 'continental': 4898, 'invented': 4899, 'physical': 4900, 'weigh': 4901, 'rhythm': 4902, 'abbot': 4903, 'speaks': 4904, 'skilled': 4905, 'well-known': 4906, 'passion': 4907, 'bit': 4908, 'stephanie': 4909, 'worship': 4910, 'sit': 4911, 'strong-willed': 4912, 'vicars': 4913, 'hummerstone': 4914, 'torrington': 4915, 'devon': 4916, 'congregation': 4917, 'wound': 4918, 'colleges': 4919, 'publish': 4920, 'everywhere': 4921, 'signing': 4922, 'peals': 4923, 'observed': 4924, 'desired': 4925, 'reference': 4926, 'invariably': 4927, 'recessionary': 4928, 'environments': 4929, 'risen': 4930, 'peaks': 4931, 'double-digit': 4932, 'trends': 4933, 'advances': 4934, 'sustained': 4935, 'exception': 4936, 'perritt': 4937, '1965': 4938, '1971': 4939, 'manner': 4940, 'cigna': 4941, 'insurer': 4942, '3.6': 4943, 'argument': 4944, 'tally': 4945, 'vs.': 4946, 'outlook': 4947, 'page-one': 4948, 'supposedly': 4949, 'microphone': 4950, 'sued': 4951, 'grandfather': 4952, 'boulder': 4953, 'jerritts': 4954, 'commit': 4955, 'servicing': 4956, 'nowhere': 4957, 'easing': 4958, 'strains': 4959, 'sino-u.s.': 4960, 'relationship': 4961, 'pro-democracy': 4962, 'demonstrators': 4963, 'phrase': 4964, 'sphere': 4965, 'toast': 4966, 'assault': 4967, 'crackdown': 4968, 'turmoil': 4969, 'greatest': 4970, 'fang': 4971, 'wife': 4972, 'li': 4973, 'traveling': 4974, 'welcome': 4975, 'contacts': 4976, 'concession': 4977, 'premier': 4978, 'blocks': 4979, 'contingent': 4980, 'rifles': 4981, 'loaded': 4982, 'passing': 4983, 'vacations': 4984, 'puts': 4985, 'perspective': 4986, 'fernando': 4987, 'del': 4988, 'j.l.': 4989, 'unpublished': 4990, 'biscayne': 4991, 'rosenblum': 4992, 'suspensions': 4993, 'danville': 4994, 'limited-partnership': 4995, 'mount': 4996, 'escrow': 4997, 'record-keeping': 4998, 'two-week': 4999, 'inaccurate': 5000, 'principals': 5001, 'eugene': 5002, 'island': 5003, 'mcfall': 5004, 'inappropriate': 5005, 'georgetown': 5006, 'del.': 5007, 'transacting': 5008, '62%-owned': 5009, 'andrew': 5010, 'francis': 5011, 'arlington': 5012, '7,500': 5013, 'miss.': 5014, '200,000': 5015, 'jeffrey': 5016, 'gerard': 5017, 'felten': 5018, 'parking': 5019, '60,000': 5020, 'ann': 5021, 'ore.': 5022, 'sun': 5023, 'rolling': 5024, 'dahl': 5025, 'cutrer': 5026, 'baton': 5027, 'rouge': 5028, 'one-month': 5029, 'kane': 5030, 'aurora': 5031, 'miller': 5032, '3,500': 5033, 'actively': 5034, 'engaged': 5035, 'computer-driven': 5036, 'plays': 5037, 'resist': 5038, 'marching': 5039, 'cope': 5040, 'wary': 5041, 'drove': 5042, 'operators': 5043, 'chunk': 5044, 'maughan': 5045, 'attractions': 5046, 'pegged': 5047, 'privacy': 5048, 'assurance': 5049, 'times-stock': 5050, '100-share': 5051, '4.25': 5052, 'enacted': 5053, '*-163': 5054, '3.35': 5055, 'pat': 5056, 'roukema': 5057, 'subminimum': 5058, 'employers': 5059, 'restriction': 5060, 'zenith': 5061, 'navy': 5062, 'gunship': 5063, 'intelligence': 5064, 'cold': 5065, 'faced': 5066, 'negotiated': 5067, 'tough': 5068, 'implies': 5069, 'fold': 5070, 'correll': 5071, 'pulp': 5072, 'consensus': 5073, 'insiders': 5074, 'onto': 5075, 'careful': 5076, 'protesters': 5077, 'pamplin': 5078, 'son': 5079, 'physics': 5080, 'new-home': 5081, '9.5': 5082, 'forest-products': 5083, 'house-senate': 5084, 'portions': 5085, 'relies': 5086, '240': 5087, 'stimulating': 5088, 'appropriators': 5089, 'confined': 5090, 'planning': 5091, 'amendments': 5092, 'exemption': 5093, 'struggle': 5094, 'fate': 5095, 'basin': 5096, 'powerful': 5097, 'hawaii': 5098, 'inouye': 5099, 'abandon': 5100, 'flights': 5101, 'leadership': 5102, 'disagreement': 5103, 'lose': 5104, 'install': 5105, 'concrete': 5106, 'openings': 5107, 'structure': 5108, 'trays': 5109, 'outlets': 5110, 'loops': 5111, 'hang': 5112, 'marvin': 5113, 'promises': 5114, 'citicorp': 5115, 'carnegie-mellon': 5116, 'droz': 5117, 'sees': 5118, 'root': 5119, 'preserving': 5120, 'tubular': 5121, 'architect': 5122, 'pyramids': 5123, 'egypt': 5124, 'vacant': 5125, 'occupational': 5126, '6.1': 5127, 'covering': 5128, 'clairton': 5129, 'dole': 5130, 'resulted': 5131, 'scannell': 5132, 'deficiencies': 5133, 'failures': 5134, 'injuries': 5135, 'proved': 5136, 'cincinnati': 5137, 'mentally': 5138, 'examined': 5139, 'shelter': 5140, 'consequence': 5141, 'lacks': 5142, 'prof.': 5143, 'poverty': 5144, 'housing': 5145, 'connected': 5146, 'obviously': 5147, '77': 5148, 'consequences': 5149, 'assertion': 5150, 'cambridge': 5151, 'allied': 5152, 'greed': 5153, 'subscribe': 5154, 'charities': 5155, 'nonprofit': 5156, '203': 5157, 'underwent': 5158, 'predict': 5159, 'thunderbird': 5160, 'rogers': 5161, '175': 5162, 'placement': 5163, 'perpetual': 5164, 'supporters': 5165, 'hammerschmidt': 5166, 'diminished': 5167, 'occasionally': 5168, 'broadcasting': 5169, 'tune': 5170, 'extensive': 5171, 'reporter': 5172, 'broadcasts': 5173, 'censorship': 5174, 'tours': 5175, 'enables': 5176, 'mind': 5177, 'facts': 5178, '1948': 5179, 'scholars': 5180, 'texts': 5181, 'intend': 5182, "o'brien": 5183, 'scripts': 5184, 'answered': 5185, 'reshaping': 5186, '3.375': 5187, '47.125': 5188, 'oriented': 5189, 'pepperidge': 5190, 'veteran': 5191, 'eliminating': 5192, 'agenda': 5193, 'naming': 5194, '8.55': 5195, '8.25': 5196, '89': 5197, '270': 5198, '8.07': 5199, '7.95': 5200, 'negotiable': 5201, 'acceptances': 5202, '7\\/16': 5203, 'lending': 5204, 'location': 5205, 'adjustable': 5206, 'annualized': 5207, 'empire': 5208, 'unhappy': 5209, 'tenure': 5210, 'succeeded': 5211, 'susan': 5212, 'petersen': 5213, 'knopf': 5214, 'evans': 5215, 'mehta': 5216, 'full-time': 5217, 'pretext': 5218, 'purse': 5219, 'withhold': 5220, 'independence': 5221, 'confederation': 5222, 'contradict': 5223, 'placing': 5224, 'rewrite': 5225, 'violation': 5226, 'severable': 5227, 'nominate': 5228, '605': 5229, 'muzzling': 5230, 'regulations': 5231, 'prevents': 5232, 'implement': 5233, 'duly': 5234, 'kinds': 5235, 'discharge': 5236, 'establish': 5237, 'vindication': 5238, 'entrusted': 5239, 'extending': 5240, 'analysis': 5241, 'duke': 5242, 'droughts': 5243, '59.9': 5244, 'pockets': 5245, 'gross': 5246, 'nebraska': 5247, 'wheat': 5248, 'collins': 5249, '6.5': 5250, 'minnesota': 5251, 'relief': 5252, 'soybeans': 5253, 'unsuccessful': 5254, 'exceeding': 5255, 'register': 5256, 'beach': 5257, 'dailies': 5258, 'attempted': 5259, 'prospective': 5260, 'editorially': 5261, 'tabloid': 5262, 'marginally': 5263, 'condition': 5264, 'noble': 5265, 'deterioration': 5266, 'bitter': 5267, 'recovered': 5268, 'consistently': 5269, 'newsroom': 5270, 'preference': 5271, '1.64': 5272, 'begun': 5273, 'gm': 5274, 'direct-mail': 5275, 'maximize': 5276, 'nameplate': 5277, '6.9': 5278, 'establishment': 5279, 'tickets': 5280, 'chooses': 5281, 'fly': 5282, 'prizes': 5283, 'riviera': 5284, 'luxury': 5285, 'borough': 5286, 'cancellation': 5287, 'fulham': 5288, 'engage': 5289, 'betting': 5290, 'obligated': 5291, 'swaps': 5292, 'capital-markets': 5293, 'recover': 5294, 'aftermath': 5295, 'baskets': 5296, 'labeled': 5297, 'fast-growing': 5298, 'refuse': 5299, 'practitioners': 5300, 'berman': 5301, 'racket': 5302, 'casino': 5303, 'civil': 5304, 'denounce': 5305, 'brean': 5306, 'computer-assisted': 5307, 'occurs': 5308, 'seize': 5309, 'movements': 5310, 'vanderbilt': 5311, 'stoll': 5312, '6.79': 5313, 'halted': 5314, 'indexers': 5315, 'virtue': 5316, 'dislike': 5317, 'absolutely': 5318, 'broker': 5319, 'legg': 5320, 'transformed': 5321, 'complains': 5322, 'sigler': 5323, 'champion': 5324, 'anti-programmers': 5325, 'anytime': 5326, 'sellers': 5327, 'jihad': 5328, 'widgets': 5329, 'index-fund': 5330, 'fundamentalist': 5331, 'contribute': 5332, 'surrounding': 5333, 'opportunities': 5334, 'inefficiencies': 5335, 'transfers': 5336, 'cleaner': 5337, 'cautious': 5338, 'tolerate': 5339, 'spreads': 5340, 'surely': 5341, 'stellar': 5342, 'profitably': 5343, 'affect': 5344, '34': 5345, 'edwards': 5346, 'odd-year': 5347, 'patrick': 5348, 'mcguigan': 5349, 'oklahoma': 5350, 'backing': 5351, 'stieglitz': 5352, 'ray': 5353, 'trimming': 5354, 'cloth': 5355, 'fasteners': 5356, 'pins': 5357, 'arighi': 5358, 'arnold': 5359, 'creates': 5360, 'inherent': 5361, 'limiting': 5362, 'denouncing': 5363, 'immune': 5364, 'taccetta': 5365, 'gorman': 5366, 'corporation': 5367, 'underwood': 5368, 'laboratories': 5369, 'crystal-lattice': 5370, 'moderately': 5371, 'cooled': 5372, 'overcome': 5373, 'cautioned': 5374, 'feasible': 5375, 'processes': 5376, 'syndicate': 5377, 'non-callable': 5378, '57': 5379, 'triple-a': 5380, 'distributable': 5381, '2000': 5382, 'redevelopment': 5383, '107': 5384, 'allocation': 5385, '6.25': 5386, '7.74': 5387, '2017': 5388, '7.40': 5389, '2005': 5390, 'eurobonds': 5391, 'indicating': 5392, 'svenska': 5393, 'franc': 5394, 'dai-ichi': 5395, 'kangyo': 5396, 'proposition': 5397, 'wonder': 5398, 'sink': 5399, 'boosters': 5400, 'wealth': 5401, 'claimed': 5402, 'spaces': 5403, 'robbie': 5404, 'superdome': 5405, '118': 5406, 'discontinued': 5407, '4.6': 5408, 'downgrade': 5409, 'maintain': 5410, 'mahoney': 5411, 'delays': 5412, 'bowman': 5413, 'campeau': 5414, '1.7': 5415, 'goodwill': 5416, 'uncertainties': 5417, 'taurus': 5418, 'mercury': 5419, 'midsized': 5420, 'instrumental': 5421, 'bureaucracy': 5422, 'synthetic': 5423, 'daughters': 5424, 'reasoning': 5425, 'beneficial': 5426, 'caldor': 5427, 'allergan': 5428, 'surgery': 5429, 'habit': 5430, 'unfocused': 5431, 'elco': 5432, '1.125': 5433, 'start-up': 5434, 'anticipation': 5435, 'larger-than-normal': 5436, 'nikkei': 5437, 'selected': 5438, 'ample': 5439, 'sidelines': 5440, 'participate': 5441, 'nippon': 5442, 'bolstered': 5443, 'b.a.t': 5444, 'hoylake': 5445, 'highs': 5446, 'market-makers': 5447, 'holiday': 5448, 'irvine': 5449, 'hearings': 5450, 'linden': 5451, 'present': 5452, '19.6': 5453, '26.8': 5454, 'households': 5455, 'toledo': 5456, 'respondents': 5457, '8.1': 5458, '2.9': 5459, 'lieberman': 5460, '102': 5461, 'three-month': 5462, 'debenture': 5463, '7.2': 5464, '4.2': 5465, 'ginnie': 5466, 'drew': 5467, '111': 5468, 'btr': 5469, 'unveil': 5470, 'lobbyists': 5471, 'sulfur-dioxide': 5472, 'dirtiest': 5473, 'devise': 5474, 'shaw': 5475, 'falls': 5476, 'processing': 5477, 'superdot': 5478, 'financier': 5479, 'doubted': 5480, 'clearance': 5481, '300-a-share': 5482, '4.7': 5483, 'formally': 5484, 'coniston': 5485, '177': 5486, 'accord': 5487, 'a.l.': 5488, 'beall': 5489, 'first-half': 5490, 'b-1b': 5491, 'graphics': 5492, 'fixed-price': 5493, 'intel': 5494, '386': 5495, 'rumor': 5496, 'stevenson': 5497, 'shift': 5498, '645,000': 5499, 'am': 5500, 'curtail': 5501, 'confirm': 5502, 'gallon': 5503, 'firmness': 5504, 'grains': 5505, '310': 5506, 'bushels': 5507, 'chilean': 5508, 'mine': 5509, 'mines': 5510, 'papua-new': 5511, '13.65': 5512, 'unicorp': 5513, 'cara': 5514, 'diluted': 5515, 'poison': 5516, 'softening': 5517, 'hadson': 5518, 'write': 5519, 'exploration': 5520, '154,240,000': 5521, 'subdued': 5522, 'somewhat': 5523, 'philip': 5524, 'puccio': 5525, 'eakle': 5526, 'revival': 5527, 'avon': 5528, 'trump': 5529, '39': 5530, 'texaco': 5531, 'fe': 5532, 'otherwise': 5533, 'entity': 5534, 'liquidated': 5535, 'unitholders': 5536, '1.76': 5537, 'in-store': 5538, 'accident': 5539, 'ohio-based': 5540, 'assembly': 5541, 'transamerica': 5542, '8.9': 5543, '750,000': 5544, 'jeweler': 5545, '1.26': 5546, 'miami-based': 5547, 'shipbuilding': 5548, 'capitalized': 5549, '6.6': 5550, 'ortega': 5551, 'contras': 5552, 'rebels': 5553, 'balloting': 5554, 'troops': 5555, 'communist': 5556, 'berlin': 5557, 'massive': 5558, 'exodus': 5559, 'discussion': 5560, 'pretoria': 5561, 'salinas': 5562, 'votes': 5563, 'studio': 5564, 'guber\\/peters': 5565, 'cost-cutting': 5566, 'comparison': 5567, 'komatsu': 5568, 'per-share': 5569, 'shrank': 5570, 'markdowns': 5571, 'reaching': 5572, 'carl': 5573, 'jonathan': 5574, 'morphogenetic': 5575, 'cartilage': 5576, 'bmp-1': 5577, 'proteins': 5578, 'bmp': 5579, 'liberal': 5580, 'narrowed': 5581, 'metallgesellschaft': 5582, 'specialized': 5583, 'backlog': 5584, 'sweaters': 5585, '1,100': 5586, 'co-author': 5587, 'sternberg': 5588, 'bribe': 5589, 'carter': 5590, 'bribed': 5591, 'karns': 5592, 'cake': 5593, 'storage': 5594, 'platinum': 5595, 'icahn': 5596, 'kurland': 5597, 'tribe': 5598, 'procedure': 5599, 'kennedy': 5600, 'trinity': 5601, 'pierre': 5602, 'elsevier': 5603, 'agnew': 5604, 'fiber': 5605, 'resilient': 5606, 'lungs': 5607, 'symptoms': 5608, 'loews': 5609, 'micronite': 5610, 'spokewoman': 5611, 'properties': 5612, 'dana-farber': 5613, '1953': 5614, '1955': 5615, 'diagnosed': 5616, 'malignant': 5617, 'mesothelioma': 5618, 'asbestosis': 5619, 'morbidity': 5620, 'groton': 5621, 'stringently': 5622, 'smooth': 5623, 'needle-like': 5624, 'classified': 5625, 'amphobiles': 5626, 'brooke': 5627, 'pathlogy': 5628, 'vermont': 5629, 'curly': 5630, 'gradual': 5631, '1997': 5632, 'cancer-causing': 5633, 'outlawed': 5634, '160': 5635, 'dusty': 5636, 'burlap': 5637, 'sacks': 5638, 'bin': 5639, 'poured': 5640, 'cotton': 5641, 'acetate': 5642, 'mechanically': 5643, 'clouds': 5644, 'dust': 5645, 'hung': 5646, 'ventilated': 5647, 'darrell': 5648, 'tracked': 5649, 'ibc': 5650, 'fraction': 5651, 'reinvestment': 5652, 'lengthened': 5653, 'longest': 5654, 'donoghue': 5655, 'shorter': 5656, 'brenda': 5657, 'malizia': 5658, 'negus': 5659, 'rises': 5660, 'pour': 5661, '352.7': 5662, 'money-fund': 5663, 'dreyfus': 5664, 'top-yielding': 5665, '9.37': 5666, '9.45': 5667, 'invests': 5668, 'waiving': 5669, '8.12': 5670, '8.14': 5671, '8.19': 5672, '8.22': 5673, '8.53': 5674, '8.56': 5675, 'j.p.': 5676, 'bolduc': 5677, '83.4': 5678, 'energy-services': 5679, 'terrence': 5680, 'daniels': 5681, 'royal': 5682, 'trustco': 5683, '212': 5684, 'mcdermott': 5685, 'babcock': 5686, 'wilcox': 5687, 's.p': 5688, '295': 5689, 'wickliffe': 5690, 'computerized': 5691, '2,700': 5692, '370': 5693, '2.80': 5694, 'ensnarled': 5695, 'earliest': 5696, 'clark': 5697, 'oversee': 5698, 'biannual': 5699, 'powwow': 5700, 'titans': 5701, 'sunny': 5702, 'confines': 5703, 'hoosier': 5704, 'royalty': 5705, 'buckle': 5706, 'rust': 5707, 'maytag': 5708, 'lesser': 5709, 'knowns': 5710, 'trojan': 5711, 'queen': 5712, 'cheese': 5713, 'starters': 5714, 'hudnut': 5715, 'symphony': 5716, 'orchestra': 5717, 'guest': 5718, 'pianist-comedian': 5719, 'borge': 5720, 'dessert': 5721, 'escort': 5722, 'busloads': 5723, 'wives': 5724, 'raced': 5725, 'unimpeded': 5726, 'traffic': 5727, 'lieutenant': 5728, 'breakfast': 5729, 'drinks': 5730, 'everyday': 5731, 'hauled': 5732, 'crews': 5733, '10-lap': 5734, 'fortune': 5735, 'drooled': 5736, 'schoolboys': 5737, 'dummies': 5738, 'execs': 5739, 'boarding': 5740, 'dancing': 5741, 'moons': 5742, 'renovated': 5743, 'ballroom': 5744, 'hottest': 5745, 'duckling': 5746, 'mousseline': 5747, 'lobster': 5748, 'consomme': 5749, 'veal': 5750, 'mignon': 5751, 'chocolate': 5752, 'terrine': 5753, 'raspberry': 5754, 'sauce': 5755, 'knowing': 5756, 'tasty': 5757, 'eat': 5758, 'ovation': 5759, 'red-carpet': 5760, 'tempts': 5761, 'heartland': 5762, 'winter': 5763, 'sluggishness': 5764, 'tallies': 5765, 'cloud': 5766, 'export-oriented': 5767, '5.29': 5768, '0.7': 5769, '5.39': 5770, 'gloomy': 5771, 'accumulated': 5772, '50.45': 5773, '50.38': 5774, 'discounts': 5775, 'fixtures': 5776, 'underscore': 5777, 'mortimer': 5778, 'zuckerman': 5779, 'four-color': 5780, '100,980': 5781, 'mid-october': 5782, 'subscriber': 5783, '120,000': 5784, 'awards': 5785, 'renewal': 5786, '325,000': 5787, '340,000': 5788, 'shore': 5789, '1,620': 5790, 'publishers': 5791, 'giveaways': 5792, 'subscribers': 5793, 'telephones': 5794, 'circulations': 5795, 'newsweekly': 5796, '4,393,237': 5797, '3,288,453': 5798, '2,303,328': 5799, 'payoff': 5800, 'westborough': 5801, '2.29': 5802, '2.25': 5803, 'manchester': 5804, 'n.h.': 5805, 'efficiencies': 5806, 'evaluated': 5807, 'emerges': 5808, 'attracts': 5809, 'wilbur': 5810, 'rothschild': 5811, 'cluttered': 5812, 'refile': 5813, 'expedited': 5814, 'ferc': 5815, 'ricken': 5816, 'toys': 5817, 'r': 5818, 'deane': 5819, 'signet': 5820, 'rexinger': 5821, 'glauber': 5822, '12-member': 5823, 'ratepayers': 5824, 'overruns': 5825, 'entertain': 5826, 'pool': 5827, 'hostage': 5828, 'slash': 5829, '737.5': 5830, '3.01': 5831, 'tracking': 5832, 'addresses': 5833, 'nightmare': 5834, '38.375': 5835, 'disputed': 5836, 'refunded': 5837, 'calculations': 5838, 'court-ordered': 5839, 'summer\\/winter': 5840, 'differential': 5841, 'appellate': 5842, '245': 5843, '72.7': 5844, 'lasalle': 5845, '500,004': 5846, 'year-to-year': 5847, 'setting': 5848, '0.4': 5849, '361,376': 5850, 'arising': 5851, 'consumption': 5852, '30,841': 5853, '13,056': 5854, 'chinchon': 5855, 'diversifying': 5856, 'fledgling': 5857, 'depend': 5858, 'longevity': 5859, 'product-design': 5860, 'scrapped': 5861, 'operational': 5862, 'prototype': 5863, 'minneapolis-based': 5864, 'needing': 5865, 'worst-case': 5866, '2.875': 5867, '98.3': 5868, 'promissory': 5869, 'complicate': 5870, 'tricky': 5871, 'unproven': 5872, 'gallium': 5873, 'arsenide': 5874, 'robotic': 5875, 'twice': 5876, 'c-90': 5877, 'hitachi': 5878, '4.75': 5879, 'pro-forma': 5880, '19.3': 5881, '5.9': 5882, 'existed': 5883, '20.5': 5884, '240,000': 5885, 'davenport': 5886, 'blanchard': 5887, 'hammerton': 5888, 'wheeland': 5889, '241': 5890, 'pardus': 5891, 'electric-utility': 5892, 'carney': 5893, 'tassinari': 5894, 'diplomacy': 5895, 'watching': 5896, 'copyrights': 5897, 'india': 5898, 'unfair-trade': 5899, 'investigations': 5900, 'genuine': 5901, 'touchy': 5902, 'denial': 5903, 'harms': 5904, 'inventiveness': 5905, 'offending': 5906, 'protections': 5907, 'discouraging': 5908, 'deterring': 5909, 'high-technology': 5910, 'lauded': 5911, 'pirates': 5912, 'search-and-seizure': 5913, 'initialing': 5914, 'amending': 5915, 'trademark': 5916, 'unauthorized': 5917, 'showings': 5918, 'compel': 5919, 'video-viewing': 5920, 'parlors': 5921, 'enact': 5922, 'compatible': 5923, 'literary': 5924, 'lower-priority': 5925, 'greece': 5926, 'less-serious': 5927, 'hoffman': 5928, 'specializing': 5929, 'retaliation': 5930, 'improvements': 5931, 'craft': 5932, 'developments': 5933, 'piracy': 5934, 'disregard': 5935, '301': 5936, 'halve': 5937, 'third-highest': 5938, 'declaration': 5939, 'nestor': 5940, 'latin': 5941, 'aspires': 5942, 'external': 5943, 'miguel': 5944, 'alurralde': 5945, 'mulford': 5946, 'negotiator': 5947, 'carballo': 5948, 'menem': 5949, 'centennial': 5950, 'milestones': 5951, 'commodore': 5952, 'tandy': 5953, 'trs-80': 5954, 'audiocassettes': 5955, 'garage': 5956, 'wozniak': 5957, 'hobbyists': 5958, 'homebrew': 5959, 'affordable': 5960, '1,298': 5961, 'explosive': 5962, 'desktop': 5963, 'mainframe': 5964, 'built-from-kit': 5965, 'altair': 5966, 'sol': 5967, 'imsai': 5968, 'keyboards': 5969, 'contributors': 5970, 'language-housekeeper': 5971, 'billionaire': 5972, 'adapted': 5973, 'shugart': 5974, 'seagate': 5975, 'hayes': 5976, 'dale': 5977, 'heatherington': 5978, 'co-developers': 5979, 'modems': 5980, '38.3': 5981, 'f.h.': 5982, 'jersey-based': 5983, 'purepac': 5984, 'label': 5985, 'strait': 5986, '321,000': 5987, 'pty.': 5988, 'output': 5989, 'gradually': 5990, 'reaches': 5991, 'perch': 5992, 'dolphin': 5993, 'seahorse': 5994, 'tarwhine': 5995, 'sloan': 5996, 'gelatin': 5997, 'capsules': 5998, 'divest': 5999, 'non-encapsulating': 6000, 'possessions': 6001, 'generalized': 6002, 'classifications': 6003, 'categories': 6004, 'injury': 6005, 'low-priced': 6006, 'battery-operated': 6007, 'beneficiaries': 6008, '37.3': 6009, 'automotive-parts': 6010, 'achieved': 6011, 'akerfeldt': 6012, 'wallowing': 6013, '52-week': 6014, '16.125': 6015, '13.73': 6016, '9.625': 6017, 'resume': 6018, 'personally': 6019, 'assisted': 6020, 'manfred': 6021, 'gingl': 6022, 'chilver': 6023, '63-year-old': 6024, 'clays': 6025, 'securities-based': 6026, '701': 6027, 'mortgage-based': 6028, 'o.': 6029, '570': 6030, 'blackstone': 6031, 'seven-year': 6032, 'redeploy': 6033, 'channel': 6034, 'addressing': 6035, 'tenth': 6036, 'pate': 6037, '54-year-old': 6038, 'ltv': 6039, 's.p.a.': 6040, 'indirect': 6041, '37-a-share': 6042, 'incorporated': 6043, 'editions': 6044, '1.82': 6045, '84.29': 6046, 'yen-support': 6047, 'intervention': 6048, '150.00': 6049, 'sharper': 6050, '86.12': 6051, 'rash': 6052, 'triple': 6053, 'washington-based': 6054, 'turf': 6055, 'austria': 6056, 'portugal': 6057, 'corazon': 6058, 'aquino': 6059, 'province': 6060, 'quips': 6061, 'northampton': 6062, 'mirrors': 6063, 'mania': 6064, 'narrowly': 6065, 'wildly': 6066, 'oblivion': 6067, 'open-end': 6068, 'one-country': 6069, 'issuing': 6070, 'hoopla': 6071, 'heavy-duty': 6072, 'stretching': 6073, 'nets': 6074, 'urge': 6075, 'smattering': 6076, 'emerging': 6077, 'outpaced': 6078, 'taste': 6079, 'burned': 6080, 'whipsaw': 6081, 'clobbered': 6082, 'alarmed': 6083, 'valuations': 6084, 'fattened': 6085, 'foreign-stock': 6086, 'resistant': 6087, 'aghast': 6088, 'lofty': 6089, 'jumping': 6090, 'repayment': 6091, 'pre-communist': 6092, 'russian': 6093, 'coincident': 6094, 'short-lived': 6095, 'kerensky': 6096, 'communists': 6097, 'seized': 6098, '1917': 6099, '1934': 6100, 'amended': 6101, 'u.s.s.r.': 6102, 'pre-1917': 6103, 'pre-1933': 6104, 'satisfying': 6105, 'lend-lease': 6106, 'factories': 6107, 'booked': 6108, '236.74': 6109, '236.79': 6110, '59.6': 6111, 'contractors': 6112, '415.6': 6113, '415.8': 6114, 'seasonal': 6115, 'mayland': 6116, 'taper': 6117, 'blank': 6118, 'industrial-production': 6119, 'slip': 6120, 'watched': 6121, 'buildup': 6122, 'conforms': 6123, 'elliott': 6124, 'eases': 6125, 'nondurable': 6126, '109.73': 6127, 'durable': 6128, '0.2': 6129, '127.03': 6130, 'durable-goods': 6131, '234.4': 6132, 'unfilled': 6133, '0.5': 6134, '497.34': 6135, '191.9': 6136, 'berson': 6137, 'nonresidential': 6138, '99.1': 6139, 'f.w.': 6140, 'goverment': 6141, 'signaling': 6142, 'pitney': 6143, 'bowes': 6144, 'purhasing': 6145, 'vendors': 6146, 'delivering': 6147, 'inflationary': 6148, 'abating': 6149, 'lengthen': 6150, 'gauges': 6151, 'worsening': 6152, 'acknowledging': 6153, 'numbered': 6154, 'newcomer': 6155, 'powder': 6156, 'nonfat': 6157, 'dairy': 6158, 'sebastian': 6159, 'judging': 6160, 'haruki': 6161, '320': 6162, '18.95': 6163, 'baby': 6164, 'boomers': 6165, 'texture': 6166, 'salty': 6167, 'dogs': 6168, 'whistle': 6169, 'johnny': 6170, 'goode': 6171, 'bugs': 6172, 'bunny': 6173, 'mickey': 6174, 'spillane': 6175, 'groucho': 6176, 'harpo': 6177, 'desultory': 6178, 'charm': 6179, 'recognizing': 6180, 'buttoned-down': 6181, 'lore': 6182, 'refreshing': 6183, 'self-aggrandizing': 6184, 'we-japanese': 6185, 'perpetuate': 6186, 'unique': 6187, 'unfathomable': 6188, 'outsiders': 6189, 'implicit': 6190, 'nutty': 6191, 'plot': 6192, 'rooted': 6193, 'imaginative': 6194, 'disaffected': 6195, 'hard-drinking': 6196, 'nearly-30': 6197, 'snow': 6198, 'search': 6199, 'elusive': 6200, 'behest': 6201, 'sinister': 6202, 'erudite': 6203, 'mobster': 6204, 'tow': 6205, 'prescient': 6206, 'girlfriend': 6207, 'sassy': 6208, 'retorts': 6209, 'docile': 6210, 'butterfly': 6211, 'meets': 6212, 'solicitous': 6213, 'chauffeur': 6214, 'god': 6215, 'roughhewn': 6216, 'wears': 6217, 'sheepskin': 6218, '40-year-old': 6219, 'sensation': 6220, 'norwegian': 6221, 'fluent': 6222, 'lyrics': 6223, 'published': 6224, 'youthful': 6225, 'brat': 6226, 'pack': 6227, 'dominating': 6228, 'best-seller': 6229, 'charts': 6230, 'idiomatic': 6231, 'dashes': 6232, '339': 6233, 'fabled': 6234, 'virtues': 6235, 'player': 6236, 'batting': 6237, 'tatsunori': 6238, 'hara': 6239, 'humble': 6240, 'uncomplaining': 6241, 'obedient': 6242, 'soul': 6243, 'besuboru': 6244, 'ball': 6245, 'bat': 6246, 'politely': 6247, 'balls': 6248, 'ushers': 6249, 'expands': 6250, 'hitter': 6251, 'honorably': 6252, 'abide': 6253, 'wear': 6254, 'chronicle': 6255, 'rationed': 6256, 'soho': 6257, 'petulant': 6258, 'impudent': 6259, 'hosted': 6260, 'luce': 6261, 'fellowship': 6262, 'supercilious': 6263, 'vicious': 6264, 'passages': 6265, 'invades': 6266, 'mundane': 6267, 'aspects': 6268, 'regimented': 6269, 'assigned': 6270, 'lunch': 6271, 'austere': 6272, 'dormitory': 6273, 'prying': 6274, 'caretaker': 6275, 'observations': 6276, 'salarymen': 6277, 'unproductive': 6278, 'overtime': 6279, 'sake': 6280, 'solidarity': 6281, 'hierarchical': 6282, 'chary': 6283, 'enormously': 6284, 'frustrating': 6285, 'raring': 6286, 'invent': 6287, 'walkman': 6288, 'kirkpatrick': 6289, 'corners': 6290, 'tobacco': 6291, 'smoke': 6292, 'exempt': 6293, 'bars': 6294, 'bans': 6295, 'theaters': 6296, 'hospitals': 6297, 'siti': 6298, 'zaharah': 6299, 'sulaiman': 6300, 'no-smoking': 6301, 'mara': 6302, 'kuala': 6303, 'lumpur': 6304, 'on-campus': 6305, '26,000': 6306, 'stalls': 6307, 'posters': 6308, 'signboards': 6309, 'restricts': 6310, 'backer': 6311, 'spielvogel': 6312, 'bates': 6313, 'surveyed': 6314, 'espouse': 6315, 'stress': 6316, 'thai': 6317, 'cabinet': 6318, 'pramual': 6319, 'sabhavasu': 6320, 'bangkok': 6321, 'plaza': 6322, 'undertaking': 6323, 'yasser': 6324, 'arafat': 6325, 'palestine': 6326, 'liberation': 6327, 'wafa': 6328, 'plo': 6329, 'tourism': 6330, 'food-shop': 6331, 'mainland': 6332, 'chaotic': 6333, 'pap': 6334, 'unrealistically': 6335, 'happier': 6336, 'establishing': 6337, 'diplomatic': 6338, 'strapped': 6339, 'warsaw': 6340, 'multibillion-dollar': 6341, 'danube': 6342, 'austrian': 6343, 'twinned': 6344, 'upstream': 6345, 'authorized': 6346, 'miklos': 6347, 'twin': 6348, 'dams': 6349, 'twindam': 6350, 'czech': 6351, 'solely': 6352, 'painting': 6353, 'strindberg': 6354, 'scandinavian': 6355, '2.44': 6356, 'lighthouse': 6357, 'painted': 6358, 'oils': 6359, 'playwright': 6360, '1901': 6361, 'upturn': 6362, 'couples': 6363, 'exchanging': 6364, '271,124': 6365, '400,000': 6366, '85.1': 6367, '10.5': 6368, 'accrued': 6369, 'swapped': 6370, 'scotia': 6371, 'mcleod': 6372, 'rbc': 6373, 'dominion': 6374, 'actor': 6375, 'inheritor': 6376, 'charlie': 6377, 'steve': 6378, 'laid': 6379, 'obsessed': 6380, 'refitting': 6381, 'campus': 6382, '36-minute': 6383, 'black-and-white': 6384, 'full-length': 6385, 'poignant': 6386, 'modern-day': 6387, 'composer': 6388, 'marc': 6389, 'marder': 6390, 'classical': 6391, 'ensembles': 6392, 'exciting': 6393, 'eclectic': 6394, 'intertitles': 6395, 'good-hearted': 6396, 'filmed': 6397, 'lovely': 6398, 'dill': 6399, 'benign': 6400, 'noticing': 6401, 'jostle': 6402, 'cabs': 6403, 'hangs': 6404, 'greenwich': 6405, 'village': 6406, 'populated': 6407, 'jugglers': 6408, 'magicians': 6409, 'good-natured': 6410, 'hustlers': 6411, 'dead-eyed': 6412, 'four-year-old': 6413, 'cosmopolitan': 6414, 'curled': 6415, 'sketching': 6416, 'passers-by': 6417, 'skirmishes': 6418, 'carefree': 6419, 'cure': 6420, 'two-year-old': 6421, 'waif': 6422, 'nicole': 6423, 'alysia': 6424, 'thugs': 6425, 'cute': 6426, 'curse': 6427, 'alerts': 6428, 'inadequacy': 6429, 'vagrant': 6430, 'beds': 6431, 'bowery': 6432, 'mission': 6433, 'drearier': 6434, 'tuck': 6435, 'dreamed': 6436, 'improbable': 6437, 'shop': 6438, 'high-rise': 6439, 'resonate': 6440, 'camera': 6441, 'glamorize': 6442, 'vagabond': 6443, 'existence': 6444, 'whimsical': 6445, 'enviable': 6446, 'claude': 6447, 'weird': 6448, 'captivating': 6449, 'disagreeable': 6450, 'giraud': 6451, 'significance': 6452, 'hypocrisy': 6453, 'collaborated': 6454, 'fighters': 6455, 'jews': 6456, 'diversionary': 6457, 'symbolic': 6458, 'traitor': 6459, 'small-time': 6460, 'accidentally': 6461, 'enabled': 6462, 'jam': 6463, 'cocoa': 6464, 'war-rationed': 6465, 'goodies': 6466, 'untrained': 6467, 'botched': 6468, 'remorse': 6469, 'shallow': 6470, 'playful': 6471, 'dreadful': 6472, 'war-damaged': 6473, 'lover': 6474, 'thin-lipped': 6475, 'isabelle': 6476, 'huppert': 6477, 'marie': 6478, 'chopped': 6479, 'gringo': 6480, 'rendering': 6481, 'fuentes': 6482, 'mexican': 6483, 'revolution': 6484, 'endless': 6485, 'eating': 6486, 'drinking': 6487, 'celebrate': 6488, 'movies': 6489, 'peck': 6490, 'marvelously': 6491, 'loose': 6492, 'portrayal': 6493, 'video': 6494, 'finest': 6495, 'twin-jet': 6496, 'seattle': 6497, 'kawasaki': 6498, 'fuji': 6499, 'accounting': 6500, 'contribution': 6501, 'plane': 6502, 'mid-1990s': 6503, 'irony': 6504, 'off-off': 6505, 'scattered': 6506, 'hostility': 6507, 'mudslinging': 6508, 'empty': 6509, 'ushering': 6510, 'napolitan': 6511, 'stirrings': 6512, 'dawn': 6513, 'sometimes-tawdry': 6514, 'entertaining': 6515, 'confrontational': 6516, 'truthful': 6517, 'fights': 6518, 'commercials': 6519, 'facial': 6520, 'disembodied': 6521, 'contributions': 6522, 'accurately': 6523, 'hid': 6524, 'kidnapper': 6525, 'phony': 6526, 'nasty': 6527, 'innuendoes': 6528, 'siegal': 6529, 'prosecute': 6530, 'shrum': 6531, 'doak': 6532, 'unleashed': 6533, 'distorted': 6534, 'photos': 6535, 'oversight': 6536, 'secret': 6537, '95,142': 6538, 'matching': 6539, 'errors': 6540, 'get-out-the-vote': 6541, 'kidnapping': 6542, 'deceptive': 6543, 'argues': 6544, 'lt.': 6545, 'gubernatorial': 6546, 'greer': 6547, 'persuasion': 6548, 'tour': 6549, 'monticello': 6550, 'superimposed': 6551, 'liberty': 6552, 'virginians': 6553, 'nurtured': 6554, 'generations': 6555, 'statue': 6556, 'jefferson': 6557, 'dissolves': 6558, 'incest': 6559, 'transforming': 6560, 'goodman': 6561, 'shake': 6562, 'counterattack': 6563, 'close-up': 6564, 'shadows': 6565, 'recalling': 6566, 'unpleasant': 6567, 'ordeal': 6568, "c'mon": 6569, 'boyfriends': 6570, 'interjects': 6571, 'interrogated': 6572, 'rapists': 6573, 'constituent': 6574, 'technique': 6575, 'unfounded': 6576, 'interrogation': 6577, 'stigma': 6578, 'campaigner': 6579, 'rozell': 6580, 'onus': 6581, 'lasted': 6582, 'carrying': 6583, 'sensitivity': 6584, 'aired': 6585, 'remember': 6586, 'squier': 6587, 'dirty': 6588, 'rusted': 6589, 'drums': 6590, 'swim': 6591, 'purrs': 6592, 'neighbors': 6593, 'cry': 6594, 'cleaned': 6595, 'pollution': 6596, 'eagleton': 6597, '1966': 6598, 'route': 6599, 'rout': 6600, '331,000': 6601, '550,000': 6602, 'propelling': 6603, 'interviews': 6604, 'fits': 6605, 'sparking': 6606, 'posing': 6607, 'whereby': 6608, 'knitted': 6609, '5.57': 6610, '705.6': 6611, 'cash-rich': 6612, 'sites': 6613, 'labor-intensive': 6614, 'tigers': 6615, 'subordinate': 6616, 'fearful': 6617, 'hegemony': 6618, 'encourages': 6619, 'burdens': 6620, 'resists': 6621, 'u.s.-japanese': 6622, 'behemoth': 6623, 'swelling': 6624, 'multinationals': 6625, 'offend': 6626, 'lender': 6627, 'drobnick': 6628, 'cohesive': 6629, 'sectors': 6630, 'calder': 6631, 'woodrow': 6632, 'wilson': 6633, 'internatonal': 6634, 'tubes': 6635, 'assemble': 6636, 'framework': 6637, 'ministers': 6638, 'zealand': 6639, 'brunei': 6640, 'rim': 6641, 'hawke': 6642, 'reasserts': 6643, 'designing': 6644, 'dominance': 6645, 'outstrips': 6646, 'outranks': 6647, 'enlarged': 6648, 'convey': 6649, 'undertone': 6650, 'farren': 6651, 'benevolent': 6652, 'altruistic': 6653, 'apprehensive': 6654, 'troop': 6655, 'asians': 6656, 'counterweight': 6657, 'marbles': 6658, 'juggernaut': 6659, 'monopolize': 6660, 'sew': 6661, 'chong-sik': 6662, 'ninth': 6663, 'spotted': 6664, 'uncanny': 6665, 'stockbroker': 6666, 'profession': 6667, 'senate-house': 6668, 'ok': 6669, 'confronted': 6670, 'display': 6671, 'projector': 6672, 'underline': 6673, 'prosecuted': 6674, 'downfall': 6675, 'unstinting': 6676, 'laurels': 6677, 'bitterness': 6678, 'anger': 6679, 'betrayer': 6680, '*t*-82': 6681, '*t*-83': 6682, 'school-district': 6683, 'stunned': 6684, 'bald-faced': 6685, 'martyr': 6686, 'dark': 6687, 'high-stakes': 6688, 'enhanced': 6689, 'cheat': 6690, 'school-improvement': 6691, 'bolster': 6692, 'depended': 6693, 'student-test': 6694, 'incredible': 6695, 'walt': 6696, 'haney': 6697, '*t*-84': 6698, '50-state': 6699, 'school-research': 6700, 'inflated': 6701, 'erasures': 6702, '*t*-85': 6703, 'occurrences': 6704, 'revising': 6705, 'beforehand': 6706, 'precise': 6707, 'wrenching': 6708, 'state-supervised': 6709, 'interventions': 6710, '*t*-86': 6711, 'firings': 6712, 'lab': 6713, 'grants': 6714, 'superintendent': 6715, 'scholastic': 6716, 'aptitude': 6717, '*t*-87': 6718, 'sat': 6719, 'entrance': 6720, 'prosecuting': 6721, 'administrators': 6722, 'sandifer': 6723, 'purely': 6724, 'inferences': 6725, 'track': 6726, 'achievement-test': 6727, 'shaded': 6728, 'educated': 6729, 'brightest': 6730, 'nobel': 6731, 'prize': 6732, 'townes': 6733, 'actress': 6734, 'joanne': 6735, 'woodward': 6736, 'glory': 6737, 'faded': 6738, 'yellow': 6739, 'bricks': 6740, 'facade': 6741, 'gangs': 6742, 'awful': 6743, 'enrollment': 6744, 'honors': 6745, 'seventh': 6746, 'breakdown': 6747, 'bled': 6748, 'halls': 6749, 'stabbed': 6750, 'academically': 6751, 'disparate': 6752, 'privileged': 6753, 'elite': 6754, 'monied': 6755, 'inner': 6756, 'resolved': 6757, 'clean': 6758, 'deadwood': 6759, '*t*-88': 6760, 'ushered': 6761, 'betterment': 6762, '*t*-89': 6763, '37-year-old': 6764, 'dismissal': 6765, 'dreamt': 6766, 'struggled': 6767, '14-hour': 6768, '1986-87': 6769, '1987-88': 6770, 'encouraged': 6771, 'cheerleaders': 6772, 'pep': 6773, 'cultural': 6774, 'literacy': 6775, 'pta': 6776, 'inspirational': 6777, 'laura': 6778, 'dobson': 6779, 'freshman': 6780, '*t*-90': 6781, 'teacher-cadet': 6782, '11th': 6783, 'grader': 6784, 'kelli': 6785, 'distinguished': 6786, 'herself': 6787, 'approaches': 6788, 'pair': 6789, 'college-bowl': 6790, 'competitions': 6791, 'weekends': 6792, 'prepare': 6793, 'correcting': 6794, 'homework': 6795, 'cocky': 6796, '*t*-91': 6797, 'grandstander': 6798, 'deteriorating': 6799, 'incentive-bonus': 6800, '23,000': 6801, 'pride': 6802, 'ariail': 6803, 'attending': 6804, 'adequately': 6805, 'copied': 6806, 'motives': 6807, 'sociology': 6808, 'rankings': 6809, 'self-esteem': 6810, 'broke': 6811, 'desperately': 6812, '*t*-93': 6813, 'cared': 6814, '*t*-94': 6815, 'drag-down': 6816, '*t*-95': 6817, 'defeats': 6818, 'inkling': 6819, 'underprivileged': 6820, 'prosecutor': 6821, '*t*-96': 6822, 'alumni': 6823, 'concedes': 6824, 'sympathy': 6825, '*t*-97': 6826, 'morale-damaging': 6827, 'dumbfounded': 6828, 'recalls': 6829, 'knife': 6830, 'astonishment': 6831, 'dismay': 6832, 'superiors': 6833, 'unpopularity': 6834, 'mrs': 6835, 'school-board': 6836, 'crowded': 6837, '*t*-98': 6838, 'testify': 6839, 'decried': 6840, 'particulars': 6841, 'offense': 6842, 'talk-show': 6843, 'editorials': 6844, 'overused': 6845, 'enraged': 6846, 'first-time': 6847, '*t*-99': 6848, 'expunged': 6849, 'conviction': 6850, 'cranked': 6851, 'worthy': 6852, 'witnesses': 6853, 'cheerleading': 6854, 'squad': 6855, 'crushed': 6856, '17-year-old': 6857, 't-shirts': 6858, 'corridors': 6859, 'red-and-white': 6860, 'ghs': 6861, 'logo': 6862, 'shirts': 6863, '*t*-100': 6864, '*t*-101': 6865, 'aspersions': 6866, 'incident': 6867, 'evaluating': 6868, 'gayle': 6869, 'worms': 6870, 'relieved': 6871, 'chalk': 6872, 'touched': 6873, 'slate': 6874, 'schoolchildren': 6875, 'workbooks': 6876, 'roman': 6877, 'numeral': 6878, 'ix': 6879, 'two-sevenths': 6880, 'three-sevenths': 6881, 'test-practice': 6882, 'kit': 6883, '*t*-102': 6884, 'subindustry': 6885, 'school-sponsored': 6886, 'justifying': 6887, 'traverse': 6888, '*t*-103': 6889, '*t*-104': 6890, 'aces': 6891, 'psychiatrist': 6892, '*t*-105': 6893, 'kindergarten': 6894, 'eighth': 6895, 'houghton': 6896, 'mifflin': 6897, 'harcourt': 6898, 'brace': 6899, 'jovanovich': 6900, 'test-prep': 6901, 'arizona': 6902, 'louisiana': 6903, 'tools': 6904, 'binders': 6905, 'best-selling': 6906, 'ctb': 6907, '*t*-106': 6908, 'replicated': 6909, '*t*-107': 6910, 'coincidental': 6911, 'schoolteacher': 6912, 'similarity': 6913, 'devised': 6914, '69-point': 6915, 'awarding': 6916, 'subskill': 6917, 'closeness': 6918, 'preparatives': 6919, 'symmetry': 6920, 'geometrical': 6921, 'measurement': 6922, 'pie': 6923, 'graphs': 6924, 'kits': 6925, 'replicate': 6926, 'familiarization': 6927, '66.5': 6928, '64.5': 6929, 'two-letter': 6930, 'consonant': 6931, 'exclusion': 6932, 'contains': 6933, 'examples': 6934, 'matches': 6935, 'scrupulously': 6936, 'replicating': 6937, 'publication': 6938, 'outraged': 6939, 'advisory': 6940, 'ctbs': 6941, '*t*-108': 6942, '*t*-109': 6943, 'unaware': 6944, 'discontinue': 6945, 'h.n.': 6946, 'frances': 6947, 'berger': 6948, 'sacramento-based': 6949, '*t*-110': 6950, 'ancillary': 6951, 'transplantation': 6952, 'humans': 6953, 'juvenile': 6954, 'diabetes': 6955, 'degenerative': 6956, 'huntington': 6957, 'therapies': 6958, 'abortions': 6959, 'tissue-transplant': 6960, 'federally': 6961, '*t*-111': 6962, '*t*-112': 6963, 'acting': 6964, 'implant': 6965, 'brain': 6966, 'patient': 6967, 'nih-appointed': 6968, 'recommended': 6969, 'carefully': 6970, 'embroiled': 6971, 'anti-abortion': 6972, 'recruit': 6973, 'doctors': 6974, 'helm': 6975, '*t*-113': 6976, 'surgeon': 6977, 'reportedly': 6978, 'opposes': 6979, 'imposing': 6980, 'defuse': 6981, 'cdc': 6982, 'judged': 6983, 'excellence': 6984, 'disturbs': 6985, 'judgments': 6986, 'myron': 6987, 'polarized': 6988, 'exists': 6989, 'warns': 6990, 'discourage': 6991, 'unavailability': 6992, 'foundations': 6993, 'fronts': 6994, 'regenerate': 6995, '*t*-114': 6996, '*t*-115': 6997, 'syndrome': 6998, 'retardation': 6999, 'rekindled': 7000, 'lackluster': 7001, '1.01': 7002, '456.64': 7003, '118.6': 7004, 'inauspicious': 7005, '133.8': 7006, 'busiest': 7007, 'averaged': 7008, 'nonfinancial': 7009, '1.39': 7010, '446.62': 7011, '1.28': 7012, '449.04': 7013, '*t*-116': 7014, '3.23': 7015, '436.01': 7016, 'unattractive': 7017, 'anticipating': 7018, 'permitting': 7019, '*t*-117': 7020, 'beneficiary': 7021, '*t*-118': 7022, '*t*-119': 7023, 'expires': 7024, 'macheski': 7025, 'wilton': 7026, '*t*-120': 7027, 'wfrr': 7028, 'l.p.': 7029, 'ghkm': 7030, '273.5': 7031, '*t*-121': 7032, 'centerbank': 7033, 'nesb': 7034, 'pennview': 7035, 'leapt': 7036, 'univest': 7037, '25.50': 7038, 'nelms': 7039, 'near-record': 7040, 'definitive': 7041, '78': 7042, '*t*-122': 7043, '858,000': 7044, 'huntsville': 7045, 'ala.': 7046, '225.6': 7047, '*t*-123': 7048, '*t*-124': 7049, 'outcry': 7050, '*t*-125': 7051, 'protected': 7052, '*t*-126': 7053, 'receives': 7054, 'document': 7055, 'passport': 7056, 'failure': 7057, '*t*-127': 7058, 'identities': 7059, 'rarely': 7060, 'acted': 7061, 'witness': 7062, 'red-flag': 7063, '*t*-128': 7064, '*t*-129': 7065, 'computer-generated': 7066, 'certified': 7067, 'mail': 7068, '*t*-130': 7069, '*t*-131': 7070, 'lefcourt': 7071, 'delegates': 7072, 'condemning': 7073, '*t*-132': 7074, 'prohibited': 7075, 'ethics': 7076, 'disclosing': 7077, 'committing': 7078, '*t*-133': 7079, 'notice': 7080, 'lezovich': 7081, 'summons': 7082, 'initiated': 7083, 'dating': 7084, 'correspondence': 7085, 'mailed': 7086, '8300s': 7087, 'assertions': 7088, 'relating': 7089, 'fanfare': 7090, 'raul': 7091, 'quitting': 7092, 'clerks': 7093, 'quipped': 7094, '89,500': 7095, 'accountants': 7096, 'blinks': 7097, 'sum': 7098, 'orrick': 7099, 'herrington': 7100, 'sutcliffe': 7101, 'detail': 7102, 'doonesbury': 7103, "creator's": 7104, 'garry': 7105, 'punish': 7106, 'screenwriters': 7107, 'productions': 7108, '*t*-134': 7109, 'collective-bargaining': 7110, 'reviewing': 7111, 'harassment': 7112, 'consists': 7113, 'year-long': 7114, '*t*-135': 7115, 'punishment': 7116, 'punishing': 7117, 'retaliating': 7118, 'abortion-related': 7119, '*t*-136': 7120, 'obtaining': 7121, 'referrals': 7122, '*t*-137': 7123, 'advocate': 7124, 'providers': 7125, 'pregnant': 7126, 'clears': 7127, 'remarks': 7128, 'sentencing': 7129, '18-year-old': 7130, 'referring': 7131, 'queers': 7132, 'cruising': 7133, 'picking': 7134, 'teenage': 7135, 'appointed': 7136, 'commenting': 7137, 'observing': 7138, 'exhibited': 7139, 'prejudice': 7140, 'impartial': 7141, 'prostitute': 7142, 'discredit': 7143, '*t*-138': 7144, 'empowered': 7145, 'stock-manipulation': 7146, 'lowe': 7147, 'eight-count': 7148, 'indictment': 7149, 'sherwin': 7150, 'manipulate': 7151, 'carbide': 7152, 'trials': 7153, 'mistrials': 7154, 'switching': 7155, 'iran\\/contra': 7156, 'affair': 7157, 'mayer': 7158, 'three-lawyer': 7159, '520-lawyer': 7160, 'specialize': 7161, 'white-collar': 7162, 'narcotics': 7163, 'albany': 7164, 'warehousing': 7165, '353': 7166, 'apologizing': 7167, 'indulging': 7168, 'rebuked': 7169, 'fretted': 7170, 'penny': 7171, '*t*-139': 7172, 'tie-breaking': 7173, 'computer-system-design': 7174, 'summoned': 7175, 'antitrust-law': 7176, 'apologize': 7177, 'packed': 7178, 'sorry': 7179, 'embarrassing': 7180, 'sacrificing': 7181, '*t*-140': 7182, 'gifts': 7183, 'businessmen': 7184, 'extramarital': 7185, 'behavior': 7186, 'municipality': 7187, 'low-ball': 7188, 'complain': 7189, 'procurement': 7190, 'undercut': 7191, 'excessively': 7192, 'slashing': 7193, 'semiconductors': 7194, 'u.s.-japan': 7195, 'one-yen': 7196, 'map': 7197, 'waterworks': 7198, '77,000': 7199, 'saitama': 7200, 'prefectural': 7201, 'wakayama': 7202, 'emerge': 7203, 'michio': 7204, 'sasaki': 7205, 'keidanren': 7206, 'federation': 7207, '*t*-141': 7208, 'osborn': 7209, 'desai': 7210, 'michaels': 7211, '*t*-142': 7212, '14.6': 7213, '32.8': 7214, '28.6': 7215, '29.3': 7216, '28.4': 7217, 'locally': 7218, 'hydraulically': 7219, 'wheel-loader': 7220, 'heidelberg': 7221, '280': 7222, '*t*-143': 7223, 'computer-aided': 7224, 'automation': 7225, '*t*-144': 7226, 'productivity': 7227, '@': 7228, 'deposits-a': 7229, '6.21': 7230, 'lsi': 7231, '*t*-145': 7232, 'industry-wide': 7233, 'semiconductor': 7234, 'custom-chip': 7235, 'lagging': 7236, 'economical': 7237, 'wilfred': 7238, 'corrigan': 7239, 'midyear': 7240, 'phase': 7241, 'appropriate': 7242, 'equals': 7243, '86': 7244, 'counting': 7245, '133.7': 7246, '94': 7247, 'five-inch': 7248, 'more-efficient': 7249, 'wafers': 7250, 'fabricate': 7251, 'converting': 7252, 'clara': 7253, 'speculate': 7254, '*t*-146': 7255, 'robertson': 7256, 'stephens': 7257, 'jitters': 7258, 'ingersoll-rand': 7259, 'woodcliff': 7260, 'kuhns': 7261, 'buoyed': 7262, 'cautiously': 7263, 'bearish': 7264, 'underpin': 7265, 'narrow': 7266, 'goldinger': 7267, 'insight': 7268, '77.70': 7269, '77.56': 7270, 'dollar-yen': 7271, 'trettien': 7272, 'banque': 7273, 'paribas': 7274, 'convinced': 7275, 'erode': 7276, '1.5755': 7277, '1.5805': 7278, '143.93': 7279, '143.08': 7280, 'traced': 7281, 'wave': 7282, 'vitriolic': 7283, 'mollified': 7284, 'knight': 7285, 'undisclosed': 7286, 'forthcoming': 7287, 'yen-denominated': 7288, 'redeeming': 7289, 'unclear': 7290, 'unabated': 7291, 'recede': 7292, 'cues': 7293, '*t*-147': 7294, '45.3': 7295, '*t*-148': 7296, 'minimal': 7297, '374.20': 7298, '374.19': 7299, 'huxtable': 7300, 'viewers': 7301, '*t*-149': 7302, '187': 7303, '*t*-150': 7304, 'fuming': 7305, 'ultimatum': 7306, 'spin-off': 7307, '*t*-151': 7308, 'tactics': 7309, 'tell': 7310, 'flooded': 7311, 'comedies': 7312, '*t*-152': 7313, 'networks': 7314, 'pre-emptive': 7315, '2-8': 7316, 'a.c.': 7317, 'nielsen': 7318, 'r.i.': 7319, 'raleigh': 7320, 'ky.': 7321, 'dick': 7322, 'lobo': 7323, 'wtvj': 7324, 'nbc-owned': 7325, 'kuvin': 7326, 'whas': 7327, 'uncomfortable': 7328, 'wu': 7329, 'atlanta-based': 7330, 'life-insurance': 7331, 'nationale': 7332, 'nederlanden': 7333, 'frantic': 7334, 'revenue-desperate': 7335, 'magazines': 7336, 'cozy': 7337, 'fawning': 7338, 'advertorial': 7339, 'downright': 7340, 'thumbing': 7341, 'billed': 7342, 'entrepreneur': 7343, 'patricia': 7344, 'how-to': 7345, 'backyard': 7346, 'composting': 7347, 'explanatory': 7348, 'essays': 7349, 'happens': 7350, 'flush': 7351, 'toilet': 7352, 'hard-hitting': 7353, 'whirling': 7354, 'rampage': 7355, 'supermarket': 7356, 'aisles': 7357, 'guys': 7358, 'feature': 7359, 'deem': 7360, 'standpoint': 7361, '*t*-153': 7362, 'alienated': 7363, 'would-be': 7364, 'ire': 7365, 'furious': 7366, 'microwave': 7367, 'column': 7368, 'diagram': 7369, 'arrows': 7370, 'polystyrene': 7371, 'foam': 7372, 'polyproplene': 7373, 'polyester': 7374, 'non-biodegradable': 7375, 'landfill': 7376, 'monster': 7377, 'practicing': 7378, 'journalistic': 7379, 'fumes': 7380, 'modifications': 7381, 'portrayed': 7382, 'recyclability': 7383, 'soups': 7384, 'mike': 7385, 'ddb': 7386, 'big-time': 7387, 'handful': 7388, 'adolph': 7389, 'coors': 7390, 'bumkins': 7391, '*t*-154': 7392, 'relied': 7393, 'subscription': 7394, 'revenues': 7395, '2.95': 7396, 'yearly': 7397, 'recycled': 7398, 'old-house': 7399, '126,000': 7400, 'newsstands': 7401, '93,000': 7402, 'e.c.': 7403, 'fremantle': 7404, 'supplier': 7405, 'corestates': 7406, 'earle': 7407, 'palmer': 7408, 'spiro': 7409, '*-86': 7410, 'vansant': 7411, 'dugdale': 7412, 'ogilvy': 7413, 'mather': 7414, 'wpp': 7415, '*t*-155': 7416, 'serviced': 7417, 'rent-a-car': 7418, 'replacement-car': 7419, 'rentals': 7420, 'accidents': 7421, 'avrett': 7422, 'ginsberg': 7423, 'pitches': 7424, 'consumer-driven': 7425, 'pick-up': 7426, 'drop-off': 7427, 'identity-management': 7428, 'ketchum': 7429, 'braun': 7430, 'investor-relations': 7431, 'marketing-communications': 7432, '70-a-share': 7433, '*t*-156': 7434, 'hamilton': 7435, 'bermuda-based': 7436, '777': 7437, '963': 7438, 'asset-sale': 7439, '620': 7440, '490': 7441, 'flexibility': 7442, 'leeway': 7443, 'characterizing': 7444, 'entrench': 7445, 'confuse': 7446, 'materialize': 7447, '36': 7448, 'converted': 7449, '*-88': 7450, 'bermuda': 7451, '62.625': 7452, 'equip': 7453, 'lap-shoulder': 7454, 'rear': 7455, '*-89': 7456, 'samuel': 7457, 'skinner': 7458, 'milestone': 7459, 'occupant': 7460, 'equipped': 7461, '*-90': 7462, 'front-seat': 7463, 'headrests': 7464, '*-91': 7465, 'urging': 7466, 'car-safety': 7467, '*t*-157': 7468, 'classed': 7469, 'therefore': 7470, 'luck': 7471, 'chuck': 7472, 'hurley': 7473, 'mo': 7474, 'rollover': 7475, 'crashes': 7476, 'bags': 7477, 'side-crash': 7478, 'weighing': 7479, 'unloaded': 7480, 'weight': 7481, 'depressed': 7482, '*-92': 7483, 'inches': 7484, 'lap': 7485, 'engineer': 7486, 'auto-safety': 7487, 'installing': 7488, 'f-series': 7489, 'crew': 7490, 'cab': 7491, 'pickups': 7492, 'explorer': 7493, 'sport-utility': 7494, 'rail': 7495, 'enclosed': 7496, 'transporting': 7497, 'autos': 7498, 'multilevel': 7499, 'thrall': 7500, 'duchossois': 7501, 'elmhurst': 7502, '850': 7503, 'walters': 7504, '58-year-old': 7505, 'cement': 7506, 'milne': 7507, '*t*-158': 7508, 'retires': 7509, '*-93': 7510, 'longstanding': 7511, '*t*-159': 7512, 'prospectively': 7513, '*t*-160': 7514, '*-94': 7515, 'merger-related': 7516, '*-95': 7517, '121.6': 7518, 'lay': 7519, 'exclusively': 7520, 'leasing': 7521, 'goody': 7522, '11.5': 7523, 'payable': 7524, 'jan.': 7525, 'kearny': 7526, 'n.j.-based': 7527, 'hair': 7528, '992,000': 7529, '1.9': 7530, 'anti-takeover': 7531, 'henderson': 7532, '51-year-old': 7533, 'ian': 7534, '*t*-161': 7535, 'retiring': 7536, '1\\/10th': 7537, 'redemption': 7538, '*-96': 7539, 'proprietor': 7540, 'cellars': 7541, 'napa': 7542, 'tag': 7543, 'wine-making': 7544, 'estimation': 7545, '700': 7546, 'sauvignon': 7547, 'sticker': 7548, 'fastest': 7549, 'exceptional': 7550, 'exceedingly': 7551, 'lafite-rothschild': 7552, 'haut-brion': 7553, 'cru': 7554, 'deluxe': 7555, 'champagnes': 7556, 'dom': 7557, 'perignon': 7558, 'rarefied': 7559, 'trockenbeerenauslesen': 7560, 'rieslings': 7561, 'riserva': 7562, 'flashy': 7563, 'zoomed': 7564, 'priciest': 7565, 'boast': 7566, 'lion': 7567, 'bottles': 7568, 'smallest': 7569, '*t*-162': 7570, 'sauternes': 7571, 'lighter': 7572, 'spectacularly': 7573, 'prestige': 7574, 'cuvees': 7575, 'inching': 7576, "'82": 7577, 'taittinger': 7578, 'comtes': 7579, 'encroaching': 7580, 'reds': 7581, 'rhone': 7582, 'guigal': 7583, 'cote': 7584, 'rotie': 7585, 'landonne': 7586, 'steal': 7587, 'domaine': 7588, 'anywhere': 7589, 'commanded': 7590, 'three-digit': 7591, 'tags': 7592, 'coche-dury': 7593, 'corton-charlemagne': 7594, 'angelo': 7595, 'gaja': 7596, 'barbaresco': 7597, 'piero': 7598, 'antinori': 7599, 'solaia': 7600, 'vega': 7601, 'secilia': 7602, 'unico': 7603, '10th': 7604, 'grange': 7605, 'cult': 7606, '*t*-163': 7607, '*t*-164': 7608, 'happening': 7609, 'scarce': 7610, 'exhausted': 7611, "'40s": 7612, "'50s": 7613, '*t*-165': 7614, 'newer': 7615, 'bargain': 7616, 'ripen': 7617, 'acre': 7618, '*t*-166': 7619, 'yielded': 7620, 'al': 7621, 'brownstein': 7622, 'retailer': 7623, 're-thought': 7624, 'yes': 7625, '*-97': 7626, 'six-bottle': 7627, 'retailers': 7628, 'awfully': 7629, 'schaefer': 7630, 'skokie': 7631, 'suburban': 7632, 'opinions': 7633, '*t*-167': 7634, 'wins': 7635, 'sticker-shock': 7636, 'talked': 7637, 'excited': 7638, 'astronomical': 7639, 'collection': 7640, 'one-upsmanship': 7641, 'terrace': 7642, 'dunn': 7643, '*t*-168': 7644, '*t*-169': 7645, 'knowledgeable': 7646, 'cedric': 7647, 'cellar': 7648, 'overpriced': 7649, '*-99': 7650, 'grgich': 7651, 'chardonnay': 7652, 'chardonnays': 7653, '*t*-170': 7654, '*t*-171': 7655, '*-100': 7656, 'le': 7657, 'walking': 7658, "'86": 7659, 'opus': 7660, 'dominus': 7661, 'wine-buying': 7662, 'holidays': 7663, '*t*-172': 7664, 'ensrud': 7665, 'free-lance': 7666, 'upward': 7667, 'achieving': 7668, '8.75': 7669, 'shown': 7670, 'big-ticket': 7671, '*-101': 7672, 'resisting': 7673, 'excesses': 7674, 'tilt': 7675, 'integra-a': 7676, '*-103': 7677, '105': 7678, '13.5': 7679, '*t*-173': 7680, '*-104': 7681, '*t*-174': 7682, 'hotels': 7683, '*t*-175': 7684, '445': 7685, 'shelby': 7686, 'steelworkers': 7687, '3057': 7688, 'tube': 7689, 'expired': 7690, 'pact': 7691, '230-215': 7692, 'stoppage': 7693, 'postponed': 7694, '*-105': 7695, 'autions': 7696, 'rescheduled': 7697, '*-106': 7698, '*t*-176': 7699, 'partisan': 7700, 'bickering': 7701, '*t*-177': 7702, 'entangled': 7703, 'disruption': 7704, 'schedule': 7705, 'taxpayer': 7706, 'nicholas': 7707, 'brady': 7708, 'imperative': 7709, '*-107': 7710, 'maturing': 7711, '*-108': 7712, '*-109': 7713, '*-110': 7714, '36-day': 7715, 'when-issued': 7716, 'approves': 7717, 'clearing': 7718, '47.5': 7719, '25.6': 7720, '21.9': 7721, '*-111': 7722, 'decides': 7723, 'noodles': 7724, 'pre-cooked': 7725, 'pasta': 7726, 'clive': 7727, 'tidily': 7728, '*-112': 7729, 'evaporated': 7730, 'sight': 7731, '*t*-178': 7732, 'beige': 7733, '154.2': 7734, 'smelting': 7735, '5.276': 7736, '36.9': 7737, '3.253': 7738, '4.898': 7739, '1.457': 7740, '*t*-179': 7741, 'rebuffed': 7742, 'reviewed': 7743, 'combinations': 7744, 'proof': 7745, '*-113': 7746, 'clamped': 7747, 'ankle': 7748, '1.75': 7749, '51.25': 7750, '22.75': 7751, 'landis': 7752, '*t*-180': 7753, 'lessening': 7754, 'likelihood': 7755, 'agreeing': 7756, 'satrum': 7757, 'spurns': 7758, '*t*-181': 7759, 'solicitation': 7760, 'replacing': 7761, '*-114': 7762, 'nominal': 7763, '*-115': 7764, '46.1': 7765, '251.2': 7766, '278.7': 7767, 'licensing': 7768, 'challenging': 7769, 'westport': 7770, 'punitive': 7771, '*t*-182': 7772, 'combat': 7773, 'patented': 7774, 'brunswick': 7775, 'skittishness': 7776, 'unfettered': 7777, 'removal': 7778, 'perceives': 7779, '*t*-183': 7780, 'stirred': 7781, '*-116': 7782, 'impediments': 7783, '*t*-184': 7784, 'informally': 7785, 'direct-investment': 7786, 'fret': 7787, 'rancor': 7788, 'nervousness': 7789, 'devoted': 7790, 'briefing': 7791, 'journalists': 7792, 'vitally': 7793, 'emotions': 7794, 'taizo': 7795, 'watanabe': 7796, 'escalated': 7797, 'coca-cola': 7798, 'midtown': 7799, 'fires': 7800, 'discontent': 7801, 'stoked': 7802, 'jr': 7803, 'oilman': 7804, '26.2': 7805, 'automotive-lighting': 7806, 'asserting': 7807, 'greenmailer': 7808, 'texan': 7809, 'lloyd': 7810, 'bentsen': 7811, 'highlight': 7812, '*t*-185': 7813, 'disproportionate': 7814, 'table': 7815, 'litany': 7816, '*t*-186': 7817, 'clarified': 7818, '*t*-187': 7819, 'retort': 7820, 'concessions': 7821, 'exactly': 7822, '*t*-188': 7823, 'sorting': 7824, 'specifics': 7825, 'crossed': 7826, 'gauging': 7827, 'sheaf': 7828, 'elisabeth': 7829, 'rubinfien': 7830, 'improves': 7831, 'wayland': 7832, 'sights': 7833, 'myriad': 7834, 'penetrate': 7835, 'joint-venture': 7836, 'guided': 7837, 'bureaucratic': 7838, 'maze': 7839, '*t*-189': 7840, 'kidney': 7841, 'stones': 7842, '*t*-190': 7843, 'treats': 7844, 'lesions': 7845, 'count': 7846, 'olsen': 7847, 'milked': 7848, '*-117': 7849, 'bankroll': 7850, 'promising': 7851, '*t*-191': 7852, '214': 7853, '*-118': 7854, '*t*-192': 7855, 'penetration': 7856, 'low-tech': 7857, 'fancy': 7858, 'warrenton': 7859, 'fabricator': 7860, 'architectural': 7861, 'foundering': 7862, 'chiefly': 7863, 'ichiro': 7864, 'inside': 7865, 'nissho-iwai': 7866, '*t*-193': 7867, 'counterpart': 7868, 'vertically': 7869, 'feudal': 7870, 'globally': 7871, 'sogo-shosha': 7872, 'takeshi': 7873, 'kondo': 7874, '*-119': 7875, 'trading-company': 7876, 'logjam': 7877, 'small-company': 7878, 'davies': 7879, 'alliance': 7880, 'queuing': 7881, 'unsympathetic': 7882, '*t*-194': 7883, 'relation': 7884, 'generate': 7885, 'ai': 7886, '*-120': 7887, '*t*-195': 7888, 'fueling': 7889, 'airports': 7890, '*t*-196': 7891, '*-121': 7892, 'langner': 7893, 'high-balance': 7894, 'pine': 7895, 'safe': 7896, 'competed': 7897, 'bundling': 7898, 'segmenting': 7899, 'ncnb': 7900, 'connections': 7901, 'adults': 7902, 'pre-approved': 7903, 'saving': 7904, 'planters': 7905, 'memphis': 7906, 'tenn.': 7907, 'edge': 7908, 'thirtysomething': 7909, 'crowd': 7910, '*t*-197': 7911, 'borrowed': 7912, 'aiming': 7913, 'elderly': 7914, 'judie': 7915, 'jacksonville': 7916, 'sub-segments': 7917, 'tailoring': 7918, 'styles': 7919, 'life-style': 7920, 'sub-markets': 7921, 'athletic': 7922, '55-year-old': 7923, '75-year-old': 7924, '1973': 7925, 'wells': 7926, 'fargo': 7927, '*t*-198': 7928, 'safe-deposit': 7929, 'begot': 7930, 'slew': 7931, 'copycats': 7932, 'computerize': 7933, 'niches': 7934, '*t*-199': 7935, '*-122': 7936, 'mid-1970s': 7937, 'analyze': 7938, '*t*-200': 7939, '*t*-201': 7940, 'high-rate': 7941, 'cds': 7942, 'passbook': 7943, 'interest-bearing': 7944, 'staggering': 7945, 'norwest': 7946, 'battles': 7947, 'worrying': 7948, 'money-center': 7949, '*t*-202': 7950, 'cultivated': 7951, 'savvier': 7952, '*t*-203': 7953, 'fragmentation': 7954, 'attracting': 7955, 'rate-sensitive': 7956, 'rewarding': 7957, 'captive': 7958, 'audience': 7959, '*t*-204': 7960, '*-123': 7961, 'loyal': 7962, '*t*-205': 7963, 'borrowers': 7964, 'savers\\/investors': 7965, 'drawbacks': 7966, 'personnel': 7967, 'promotional': 7968, 'chemplus': 7969, 'flourish': 7970, 'tailored': 7971, '*-124': 7972, 'boutique': 7973, 'iras': 7974, 'amend': 7975, 'delete': 7976, 'resubmit': 7977, 'develops': 7978, 'low-cost': 7979, 'hawaiian': 7980, 'necklace': 7981, 'lasting': 7982, 'pocket': 7983, 'teetering': 7984, 'insolvency': 7985, 'needy': 7986, 'solvent': 7987, '*t*-206': 7988, 'doors': 7989, 'joy': 7990, '*t*-207': 7991, 'builds': 7992, 'self': 7993, 'sufficiency': 7994, 'critical': 7995, 'mid-size': 7996, 'semesters': 7997, '*t*-208': 7998, 'rap': 7999, 'sagging': 8000, 'morale': 8001, 'baris': 8002, 'emergencies': 8003, 'reinstating': 8004, '150-point': 8005, '20-point': 8006, '*-125': 8007, '1:30': 8008, '3:15': 8009, '*-126': 8010, 'tumultuous': 8011, 'skidded': 8012, '*t*-210': 8013, '30-minute': 8014, '*-127': 8015, 'melamed': 8016, 'lessen': 8017, 'reopened': 8018, 'subsequent': 8019, 'flood': 8020, '*t*-211': 8021, 'knocked': 8022, 'synchronized': 8023, 'circuit-breaker': 8024, 'aggravated': 8025, 'directing': 8026, '*-129': 8027, '*t*-212': 8028, 'respite': 8029, 'sell-offs': 8030, 'maximum': 8031, 'modification': 8032, '*-130': 8033, 'lapses': 8034, '*t*-213': 8035, 'post-hearing': 8036, '*t*-214': 8037, 'comfortable': 8038, 'legislators': 8039, 'breeden': 8040, 'breakers': 8041, '*t*-215': 8042, 'vague': 8043, 'mushy': 8044, 'viewpoints': 8045, 'angered': 8046, '*-131': 8047, 'sensitive': 8048, '*-132': 8049, 'happy': 8050, '*-133': 8051, 'annoyed': 8052, 'fifteen': 8053, 'attended': 8054, '*t*-216': 8055, 'jurisdictional': 8056, '*t*-217': 8057, '*-134': 8058, '*t*-218': 8059, 'committees': 8060, 'peculiar': 8061, 'peculiarities': 8062, 'unintelligible': 8063, 'tailors': 8064, '*t*-219': 8065, 'evoke': 8066, 'loveliest': 8067, 'cascading': 8068, 'calling': 8069, 'faithful': 8070, 'evensong': 8071, 'parishioners': 8072, 'angels': 8073, 'chat': 8074, 'rhythmically': 8075, '*t*-220': 8076, '1614': 8077, 'discordant': 8078, '*-135': 8079, 'church-goers': 8080, 'enjoying': 8081, 'cool': 8082, '*t*-221': 8083, 'derek': 8084, 'octogenarians': 8085, '*t*-222': 8086, 'sometimes-exhausting': 8087, 'sounding': 8088, 'belfries': 8089, 'anglia': 8090, 'scrape': 8091, 'water-authority': 8092, 'dances': 8093, 'drift': 8094, 'flightiness': 8095, 'diminish': 8096, 'anglian': 8097, '*t*-223': 8098, 'pealing': 8099, 'sundays': 8100, 'tunes': 8101, 'carillons': 8102, '*-136': 8103, 'childish': 8104, 'mind-boggling': 8105, '380': 8106, 'dexterity': 8107, 'concentration': 8108, 'proper': 8109, 'rounds': 8110, 'highest-pitched': 8111, 'descending': 8112, 'altering': 8113, 'variation': 8114, 'memorize': 8115, '*t*-224': 8116, 'odd-sounding': 8117, 'treble': 8118, 'grandsire': 8119, 'caters': 8120, 'kensington': 8121, 'ten': 8122, 'shirt-sleeved': 8123, 'prize-fighter': 8124, 'pulling': 8125, 'rope': 8126, '*t*-225': 8127, 'disappears': 8128, 'hole': 8129, 'snaking': 8130, 'muffled': 8131, 'totally': 8132, 'stare': 8133, 'vision': 8134, 'rope-sight': 8135, 'pulls': 8136, 'bronze': 8137, 'wheels': 8138, 'madly': 8139, '360': 8140, 'inverted': 8141, 'mouth-up': 8142, 'wrists': 8143, 'retard': 8144, 'detective-story': 8145, 'novelist': 8146, 'finds': 8147, 'satisfaction': 8148, 'mathematical': 8149, 'completeness': 8150, 'perfection': 8151, 'filled': 8152, 'solemn': 8153, 'intoxication': 8154, '*t*-226': 8155, 'intricate': 8156, 'ritual': 8157, 'faultlessly': 8158, 'obsession': 8159, 'pattenden': 8160, '*t*-227': 8161, 'stays': 8162, 'stuck': 8163, '*-137': 8164, 'sweat': 8165, 'skip': 8166, 'pub': 8167, 'clerics': 8168, 'steadily': 8169, 'dwindling': 8170, 'pressing': 8171, 'non-religious': 8172, 'rev.': 8173, 'jeremy': 8174, '*t*-228': 8175, 'sacked': 8176, 'self-perpetuating': 8177, '*t*-229': 8178, 'premises': 8179, 'ilminster': 8180, 'somerset': 8181, 'dust-up': 8182, 'attendance': 8183, 'w.d.': 8184, 'refuses': 8185, 'c.j.b.': 8186, 'stairs': 8187, '*-138': 8188, 'altar': 8189, 'obvious': 8190, 'exit': 8191, 'prayer': 8192, 'feelings': 8193, 'bell-ringer': 8194, '*-139': 8195, 'fuller': 8196, 'aims': 8197, 'speak': 8198, 'theological': 8199, 'joys': 8200, 'booklet': 8201, 'entitled': 8202, 'attacking': 8203, 'bellringers': 8204, '40,000': 8205, '*-140': 8206, 'parishes': 8207, 'inner-city': 8208, 'lucky': 8209, 'male-dominated': 8210, 'bell-ringing': 8211, 'youths': 8212, '1637': 8213, 'male-only': 8214, '*t*-230': 8215, 'galling': 8216, 'sole': 8217, 'cathedral': 8218, 'westminster': 8219, 'abbey': 8220, 'equal-opportunity': 8221, 'red-blooded': 8222, 'balanced': 8223, 'frequency': 8224, 'fainting': 8225, 'tea': 8226, 'torrent': 8227, 'solihull': 8228, '*-141': 8229, 'dressed': 8230, 'decorated': 8231, 'beer-belly': 8232, 'unwashed': 8233, 'unbearably': 8234, 'flatulent': 8235, 'sheffield': 8236, 'faint': 8237, 'bless': 8238, 'unsettled': 8239, 'comfort': 8240, 'predictable': 8241, 'arrival': 8242, 'breathe': 8243, 'warn': 8244, 'trap': 8245, 'unwary': 8246, 'quantitative': 8247, 'widow': 8248, 'spiders': 8249, '*t*-231': 8250, 'males': 8251, 'mating': 8252, 'robustly': 8253, 'hospitable': 8254, 'cyclical': 8255, 'sliding': 8256, 'smartly': 8257, '*t*-232': 8258, 'unenticing': 8259, 'exits': 8260, 'pushes': 8261, 'tanked': 8262, '*t*-233': 8263, 'hint': 8264, 'escaped': 8265, 'debacle': 8266, '1933': 8267, '1961': 8268, '1968': 8269, 'troublesome': 8270, 'behaving': 8271, 'philadelphia-based': 8272, 'co-chairman': 8273, 'tad': 8274, '*t*-234': 8275, 'single-digit': 8276, 'weakening': 8277, 'forecasting': 8278, 'slowdowns': 8279, 'doerflinger': 8280, 'wherewithal': 8281, 'declare': 8282, '*-142': 8283, 'bulls': 8284, 'expenditures': 8285, '139': 8286, '138': 8287, 'slippage': 8288, 'harbinger': 8289, '*t*-235': 8290, 'element': 8291, 'upside': 8292, 'alexander': 8293, 'graham': 8294, 'invention': 8295, 'father-in-law': 8296, 'gardner': 8297, 'hubbard': 8298, 'wealthy': 8299, 'well-connected': 8300, 'emile': 8301, '*-144': 8302, 'princely': 8303, 'infringed': 8304, 'established': 8305, 'caveat': 8306, '*-145': 8307, 'enter': 8308, 'discontinuing': 8309, '*t*-236': 8310, '266': 8311, '176': 8312, 'structurally': 8313, 'radically': 8314, 'word-processing': 8315, 'legend': 8316, '*t*-237': 8317, 'lacked': 8318, '93': 8319, '*t*-238': 8320, 'rapprochement': 8321, 'spoke': 8322, 'length': 8323, '*t*-239': 8324, 'afflicted': 8325, 'bloody': 8326, 'suppression': 8327, 'harped': 8328, 'outrage': 8329, 'massacre': 8330, '*t*-240': 8331, 'proponent': 8332, 'peaceful': 8333, 'seduce': 8334, 'socialist': 8335, 'capitalist': 8336, 'tension': 8337, 'evident': 8338, 'banquet': 8339, 'reciting': 8340, 'platitudes': 8341, 'eternal': 8342, 'friendship': 8343, 'reminded': 8344, 'shangkun': 8345, '3-4': 8346, '*t*-241': 8347, '*t*-242': 8348, 'ordering': 8349, 'undiplomatic': 8350, 'fashion': 8351, 'killings': 8352, 'prominently': 8353, 'demonstrations': 8354, 'deng': 8355, 'xiaoping': 8356, 'speaking': 8357, '*-146': 8358, 'deeply': 8359, 'counterrevolutionary': 8360, 'rebellion': 8361, '*t*-243': 8362, 'reprove': 8363, 'mend': 8364, '*t*-244': 8365, 'deteriorated': 8366, 'chinese-american': 8367, 'dissident': 8368, 'lizhi': 8369, 'shuxian': 8370, 'refuge': 8371, 'afterwards': 8372, 'anti-china': 8373, 'high-level': 8374, 'codified': 8375, '*-147': 8376, 'unofficial': 8377, 'envoy': 8378, 'brent': 8379, 'scowcroft': 8380, 'saturday': 8381, 'top-level': 8382, 'participation': 8383, 'fulbright': 8384, 'government-funded': 8385, 'pulled': 8386, 'acknowledge': 8387, 'infusion': 8388, 'borders': 8389, 'sdi': 8390, 'weapon': 8391, '*t*-245': 8392, 'shoot': 8393, 'minor': 8394, 'peng': 8395, 'hoped': 8396, 'encounter': 8397, 'guns': 8398, 'arrived': 8399, 'machine-gun-toting': 8400, 'ambassador': 8401, 'residence': 8402, 'encircling': 8403, 'discarded': 8404, 'uzi-model': 8405, 'pistols': 8406, 'plainclothes': 8407, 'unmarked': 8408, 'soldiers': 8409, '*t*-246': 8410, 'diplomats': 8411, 'clicked': 8412, 'graduates': 8413, 'accusing': 8414, 'jeopardizing': 8415, 'barking': 8416, 'buck': 8417, '*t*-247': 8418, 'volunteer': 8419, 'unethical': 8420, 'higher-salaried': 8421, 'unfortunately': 8422, 'impression': 8423, 'visited': 8424, '*-149': 8425, 'images': 8426, 'perceptions': 8427, 'hiroshi': 8428, 'asada': 8429, 'barbara': 8430, 'regrettable': 8431, 'negatives': 8432, 'liberals': 8433, 'progressive': 8434, 'prof': 8435, 'ethel': 8436, 'klein': 8437, '76': 8438, '*t*-248': 8439, 'disapprove': 8440, 'spouse': 8441, 'imply': 8442, 'three-quarters': 8443, 'distasteful': 8444, 'newsworthy': 8445, 'perpetuates': 8446, 'insidious': 8447, 'stereotyped': 8448, 'defined': 8449, 'denominator': 8450, 'preston': 8451, 'birmingham': 8452, 'ala': 8453, 'self-regulatory': 8454, 'disciplined': 8455, '*-150': 8456, '*-151': 8457, 'marina': 8458, 'rey': 8459, '*-152': 8460, '*-153': 8461, 'telephone-information': 8462, 'i.': 8463, 'improper': 8464, '*-154': 8465, '*t*-249': 8466, '*-155': 8467, 'lauderhill': 8468, 'plantation': 8469, '*t*-250': 8470, '*-156': 8471, 'delwin': 8472, '*-157': 8473, 'clemens': 8474, 'inaccurately': 8475, 'weatherly': 8476, 'keehn': 8477, 'northy': 8478, 'prater': 8479, 'mercer': 8480, 'wash.': 8481, 'implication': 8482, 'timing': 8483, 'rectified': 8484, 'w.n.': 8485, 'n.': 8486, 'differ': 8487, 'markup': 8488, '*t*-251': 8489, 'timely': 8490, 'requests': 8491, 'derel': 8492, 'adams': 8493, 'killeen': 8494, 'angier': 8495, 'reddington': 8496, 'shores': 8497, 'stirlen': 8498, 'bonnell': 8499, 'boorse': 8500, 'horsham': 8501, 'chiodo': 8502, 'camille': 8503, 'chafic': 8504, 'cotran': 8505, 'colonsville': 8506, 'dompierre': 8507, 'valrico': 8508, '16,072': 8509, 'marion': 8510, 'stewart': 8511, 'spitler': 8512, '18,444': 8513, '*t*-252': 8514, 'complaining': 8515, 'anybody': 8516, 'fishman': 8517, 'longwood': 8518, 'floyd': 8519, 'amin': 8520, 'jalaalwalikraam': 8521, 'glenham': 8522, 'knapp': 8523, 'deborah': 8524, 'renee': 8525, 'muscolina': 8526, 'palisades': 8527, 'najarian': 8528, 'minn.': 8529, 'norwick': 8530, 'nesconset': 8531, 'phipps': 8532, 'sr.': 8533, 'rankin': 8534, 'mo.': 8535, 'leigh': 8536, 'sanderoff': 8537, 'gaithersburg': 8538, 'md.': 8539, '12,252': 8540, 'sandra': 8541, 'ridgefield': 8542, 'spence': 8543, 'aloha': 8544, 'mona': 8545, 'estates': 8546, 'swearingen': 8547, 'bew': 8548, 'wong': 8549, 'rabia': 8550, 'zayed': 8551, 'veselich': 8552, 'enright': 8553, '11,762': 8554, 'stuart': 8555, 'russel': 8556, 'glendale': 8557, '14,821': 8558, 'nilson': 8559, 'fountain': 8560, '82,389': 8561, 'screwed': 8562, 'breaking': 8563, '*-158': 8564, 'reps': 8565, 'security-type': 8566, 'mistakes': 8567, 'cole': 8568, 'jackson': 8569, 'rita': 8570, 'rae': 8571, 'cross': 8572, 'denver': 8573, 'meinders': 8574, 'five-day': 8575, 'eight-month': 8576, 'la.': 8577, 'karl': 8578, 'hale': 8579, 'midvale': 8580, 'utah': 8581, 'clinton': 8582, 'hayne': 8583, 'one-week': 8584, 'coconut': 8585, '250,000': 8586, 'merrick': 8587, '90-day': 8588, 'brian': 8589, 'pitcher': 8590, 'russo': 8591, 'bridgeville': 8592, '15-day': 8593, 'orville': 8594, 'leroy': 8595, 'sandberg': 8596, 'marchese': 8597, 'eric': 8598, 'monchecourt': 8599, 'gerhard': 8600, 'carson': 8601, 'fond': 8602, 'blocked': 8603, '*-159': 8604, 'emigrate': 8605, 'hurdles': 8606, 'loom': 8607, 'onslaught': 8608, 'shrug': 8609, 'overlap': 8610, 'american-style': 8611, '*t*-253': 8612, 'exploit': 8613, '11.6': 8614, 'imagine': 8615, 'racing': 8616, 'chicago-style': 8617, '*-160': 8618, 'makato': 8619, 'utsumi': 8620, 'home-market': 8621, 'osaka': 8622, 'forgotten': 8623, 'bout': 8624, 'foreign-led': 8625, 'skyward': 8626, '*t*-254': 8627, 'mechanisms': 8628, 'tightened': 8629, 'index-related': 8630, 'catch-up': 8631, 'reaped': 8632, '*t*-255': 8633, 'deryck': 8634, '*t*-256': 8635, 'wadsworth': 8636, '*t*-257': 8637, 'ascribe': 8638, 'futures-related': 8639, 'disruptive': 8640, 'liquid': 8641, 'serves': 8642, 'conduit': 8643, 'counter': 8644, 'tapes': 8645, 'unwind': 8646, '*-161': 8647, 'barfield': 8648, '*t*-258': 8649, 'manages': 8650, '23.72': 8651, '*-162': 8652, 'tenfold': 8653, '9,118': 8654, '4,645': 8655, '917': 8656, 'index-options': 8657, 'derivatives': 8658, '382-37': 8659, 'replete': 8660, '90-cent-an-hour': 8661, 'republicans': 8662, 'bend': 8663, '*t*-259': 8664, 'lifting': 8665, 'four-year': 8666, '2.65': 8667, '3.80': 8668, 'smiles': 8669, 'mont': 8670, '*-164': 8671, 'marge': 8672, 'administrations': 8673, 'adopting': 8674, 'training-wage': 8675, 'diming': 8676, 'lately': 8677, 'touted': 8678, 'impart': 8679, 'entrants': 8680, 'fought': 8681, 'acceded': 8682, 'insistence': 8683, '3.61': 8684, '*not*': 8685, 'government-certified': 8686, '*t*-260': 8687, 'unrestricted': 8688, '*-165': 8689, 'mininum-wage': 8690, 'elimination': 8691, '534': 8692, 'microcomputers': 8693, '84-month': 8694, '130.7': 8695, 'ac-130u': 8696, 'marietta': 8697, '*-166': 8698, '29.9': 8699, 'low-altitude': 8700, 'navigation': 8701, '29.4': 8702, 'mode': 8703, 'gentle': 8704, 'hard-charging': 8705, 'teddy': 8706, 'roosevelt': 8707, '62-year-old': 8708, 'forest-product': 8709, 'unsolicited': 8710, '3.19': 8711, 'entice': 8712, 'negotiating': 8713, 'surrender': 8714, 'opens': 8715, 'dilemma': 8716, 'overpaying': 8717, 'courage': 8718, 'a.d.': 8719, 'long-time': 8720, 'griffin': 8721, 'wtd': 8722, 'picked': 8723, 'polytechnic': 8724, 'universities': 8725, 'strother': 8726, 'researching': 8727, 'willingness': 8728, 'arrest': 8729, 'occupying': 8730, 'impressed': 8731, 'fundraising': 8732, 'enticed': 8733, 'befuddled': 8734, 'demonstrating': 8735, 'raw': 8736, 'possessed': 8737, 'skipped': 8738, 'classmates': 8739, 'graduated': 8740, 'phi': 8741, 'beta': 8742, 'kappa': 8743, 'kentucky': 8744, 'doctorate': 8745, 'retentive': 8746, 'understatement': 8747, 'photographic': 8748, 'engineered': 8749, 'inherited': 8750, 'recession-inspired': 8751, 'building-products': 8752, 'non-core': 8753, 'vinyl': 8754, 'checkbook': 8755, 'refocusing': 8756, 'remodeling': 8757, 'cycles': 8758, 'formula': 8759, 'reins': 8760, '467': 8761, 'attributes': 8762, 'philosophy': 8763, 'concentrating': 8764, 'impressive': 8765, 'diversification': 8766, 'high-quality': 8767, 'kathryn': 8768, 'mcauley': 8769, 'contrasts': 8770, 'provoked': 8771, 'shaping': 8772, 'authorizing': 8773, 'decisive': 8774, 'reallocate': 8775, 'pentagon': 8776, '220': 8777, 'warming': 8778, 'elephant': 8779, 'draws': 8780, 'airplane': 8781, 'intriguing': 8782, 'stripped': 8783, 'noriega': 8784, 'regime': 8785, '30,537': 8786, '21-month': 8787, 'reallocated': 8788, '23,403': 8789, 'growers': 8790, '9.3': 8791, 'whip': 8792, 'pa': 8793, 'english-speaking': 8794, 'barbados': 8795, 'californian': 8796, 'bolivia': 8797, 'broadened': 8798, 'initiate': 8799, 'upsetting': 8800, 'allies': 8801, 'instructed': 8802, 'lobbyist': 8803, 'drafted': 8804, 'insert': 8805, 'waived': 8806, 'supplemental': 8807, 'anti-drug': 8808, '27.1': 8809, 'bounce': 8810, 'departments': 8811, 'backseat': 8812, 'repaired': 8813, 'ornamental': 8814, 'crashing': 8815, 'scenic': 8816, 'planner': 8817, 'prefer': 8818, 'four-foot-high': 8819, 'slab': 8820, 'ind.': 8821, 'arched': 8822, 'g': 8823, 'garret': 8824, 'teaches': 8825, 'earlham': 8826, 'ugly': 8827, 'charter': 8828, 'oak': 8829, 'cast-iron': 8830, 'medallions': 8831, 'compromises': 8832, 'peninsula': 8833, 'floral': 8834, 'tray': 8835, 'bon': 8836, 'cartons': 8837, 'porting': 8838, 'potables': 8839, 'scypher': 8840, 'cup-tote': 8841, 'beverage': 8842, 'resembles': 8843, 'beer': 8844, 'web': 8845, 'tote': 8846, 'cups': 8847, 'inventor': 8848, 'claire': 8849, 'spilling': 8850, 'lids': 8851, 'carriers': 8852, 'acknowledges': 8853, 'driver': 8854, 'sunlight': 8855, 'recyclable': 8856, 'perestroika': 8857, 'touches': 8858, 'gamut': 8859, 'blender': 8860, 'chairs': 8861, 'leningrad': 8862, 'mutchin': 8863, 'learn': 8864, 'corkscrews': 8865, 'seed': 8866, 'solution': 8867, 'birds': 8868, 'architects': 8869, 'propose': 8870, 'prisoners': 8871, 'overcrowding': 8872, 'solutions': 8873, "'30s": 8874, 'walls': 8875, 'semicircular': 8876, 'cells': 8877, 'muster': 8878, 'visits': 8879, 'aesthetic': 8880, 'famed': 8881, 'altered': 8882, 'inmates': 8883, 'upstate': 8884, 'workplace': 8885, 'mill': 8886, 'crane-safety': 8887, 'citation': 8888, 'coke': 8889, 'electrical-safety': 8890, 'indifference': 8891, 'counteract': 8892, 'flagrant': 8893, 'properly': 8894, 'spite': 8895, 'corporate-wide': 8896, 'evaluation': 8897, 'cooperating': 8898, 'corrected': 8899, 'promised': 8900, 'stiffer': 8901, 'unwilling': 8902, 'manpower': 8903, 'removing': 8904, 'safeguarding': 8905, 'anku': 8906, 'contest': 8907, 'morrell': 8908, 'meatpacking': 8909, 'brands': 8910, 'contesting': 8911, 'editing': 8912, 'error': 8913, 'hallett': 8914, 'mistakenly': 8915, 'nrdc': 8916, 'implied': 8917, 'substance-abusing': 8918, 'alcoholics': 8919, 'quoting': 8920, 'emphasized': 8921, 'prevalance': 8922, 'alcoholism': 8923, 'multitude': 8924, 'malnutrition': 8925, 'chest': 8926, 'cardiovascular': 8927, 'infectious': 8928, 'aftereffects': 8929, 'assaults': 8930, 'elementary': 8931, 'necessities': 8932, 'nutrition': 8933, 'cleanliness': 8934, 'predispose': 8935, 'composed': 8936, 'adequate': 8937, 'interactions': 8938, 'defying': 8939, 'generalizations': 8940, 'possess': 8941, 'breakey': 8942, 'fischer': 8943, 'psychiatry': 8944, 'johns': 8945, 'hopkins': 8946, 'tulane': 8947, 'array': 8948, 'thread': 8949, 'exhibits': 8950, 'simultaneously': 8951, 'disaffiliation': 8952, 'welfare': 8953, 'decay': 8954, 'intimately': 8955, 'leighton': 8956, 'cluff': 8957, 'quote': 8958, 'drop-in': 8959, 'robbed': 8960, 'deprivation': 8961, 'scarcely': 8962, 'fend': 8963, 'pre-existing': 8964, 'addiction': 8965, 'cracks': 8966, 'grim': 8967, 'brutal': 8968, 'escape': 8969, 'insanity': 8970, 'r.d.': 8971, 'vos': 8972, 'n.y': 8973, 'dismiss': 8974, 'sentimental': 8975, 'housing-assistance': 8976, 'sleeping': 8977, 'reagan-bush': 8978, 'bothered': 8979, 'inverse': 8980, 'jenkins': 8981, 'sponsors': 8982, 'chose': 8983, 'builders': 8984, 'bricklayers': 8985, 'craftsmen': 8986, 'insinuating': 8987, 'self-serving': 8988, 'crusade': 8989, 'desire': 8990, 'ymca': 8991, 'ywca': 8992, 'catholic': 8993, 'usa': 8994, 'participated': 8995, 'examinations': 8996, 'deprived': 8997, 'families': 8998, 'substitute': 8999, 'chivas': 9000, 'regal': 9001, 'phobias': 9002, 'depressions': 9003, 'ruth': 9004, 'cullowhee': 9005, '148.9': 9006, '153.3': 9007, 'retractable': 9008, 'cable': 9009, 'undesirable': 9010, 'intrusion': 9011, '300-113': 9012, 'override': 9013, 'brakes': 9014, 'impede': 9015, 'ark': 9016, 'impaired': 9017, 'airline-related': 9018, 'overriding': 9019, 'defazio': 9020, 'criteria': 9021, 'traficant': 9022, '271-147': 9023, 'dubbed': 9024, 'labor-backed': 9025, 'two-time-losers': 9026, 'lorenzo': 9027, 'follow-up': 9028, 'scholar': 9029, 'ordinary': 9030, 'cousins': 9031, 'hither': 9032, 'yon': 9033, 'outrageous': 9034, 'propagandize': 9035, 'neat': 9036, 'propagandizes': 9037, 'speeches': 9038, 'briefings': 9039, 'sorts': 9040, 'viewing': 9041, 'absurd': 9042, 'inform': 9043, 'columns': 9044, 'clipped': 9045, 'refrigerator': 9046, 'languages': 9047, 'listeners': 9048, 'first-rate': 9049, '184': 9050, '*-30': 9051, 'copying': 9052, 'photocopying': 9053, 'short-wave': 9054, 'transcribe': 9055, 'reprint': 9056, 'happened': 9057, 'absolute': 9058, 'disseminate': 9059, 'scholarly': 9060, 'memo': 9061, 'preclude': 9062, 'disseminating': 9063, 'domestically': 9064, 'mentioned': 9065, 'notwithstanding': 9066, 'statutory': 9067, 'designations': 9068, 'credentials': 9069, 'appearing': 9070, 'requesting': 9071, 'examine': 9072, 'verbatim': 9073, 'disagreed': 9074, 'proscribes': 9075, 'abridging': 9076, 'prescribe': 9077, 'assure': 9078, 'laboriously': 9079, 'surreptitiously': 9080, 'public-relations': 9081, 'sends': 9082, 'stuff': 9083, '501': 9084, 'thanks': 9085, 'photocopy': 9086, 'trivial': 9087, 'z.': 9088, 'wick': 9089, 'gartner': 9090, 'tribune': 9091, 'ames': 9092, 'wield': 9093, 'herbert': 9094, '53-year-old': 9095, 'edwin': 9096, 'dividing': 9097, 'searching': 9098, 'food-industry': 9099, 'reacted': 9100, 'favorably': 9101, 'distant': 9102, 'freshbake': 9103, 'biscuit': 9104, 'lazzaroni': 9105, 'overproduction': 9106, 'skill': 9107, 'seasoned': 9108, 'rapport': 9109, 'mediocre': 9110, 'heirs': 9111, '343': 9112, '3,600': 9113, 'impatient': 9114, 'ceo': 9115, 'succession': 9116, 'repeatedly': 9117, '877,663': 9118, '244,000': 9119, 'fringe': 9120, 'nine-year': 9121, '5.7': 9122, '274': 9123, 'advocated': 9124, 'convince': 9125, 'worthiness': 9126, 'tremendous': 9127, 'duo': 9128, 'bottom-line': 9129, 'applaud': 9130, 'exuded': 9131, 'champions': 9132, 'sitting': 9133, 'guide': 9134, 'overnight': 9135, 'fulton': 9136, 'prebon': 9137, 'u.s.a': 9138, 'depository': 9139, 'collateral': 9140, '119': 9141, '149': 9142, '7.80': 9143, '7.55': 9144, 'high-grade': 9145, 'multiples': 9146, '8.65': 9147, '8.575': 9148, '8.06': 9149, 'c.d.s': 9150, 'typical': 9151, '8.60': 9152, '8.35': 9153, '8.48': 9154, '8.30': 9155, '8.15': 9156, 'bank-backed': 9157, 'eurodollars': 9158, '13\\/16': 9159, '11\\/16': 9160, 'libor': 9161, 'quotations': 9162, '13.50': 9163, '4.875': 9164, 'indications': 9165, '7.78': 9166, '7.62': 9167, 'freddie': 9168, 'mac': 9169, '9.82': 9170, '9.75': 9171, '8.70': 9172, '6\\/2': 9173, '8.64': 9174, 'clashed': 9175, 's.i.': 9176, 'abrupt': 9177, 'departures': 9178, 'unheard': 9179, 'evolved': 9180, 'si': 9181, 'gut': 9182, 'brilliantly': 9183, 'enjoyed': 9184, 'spectacular': 9185, 'smoothly': 9186, 'bennett': 9187, 'cerf': 9188, '1925': 9189, 'ballantine\\/del': 9190, 'rey\\/fawcett': 9191, 'paperback': 9192, 'cheetham': 9193, 'hutchinson': 9194, 'powerhouse': 9195, 'gottlieb': 9196, 'yorker': 9197, 'most-likely-successor': 9198, 'joni': 9199, 'recruited': 9200, 'sonny': 9201, 'less-than-brilliant': 9202, 'tall': 9203, 'intimidate': 9204, 'uttering': 9205, 'guarding': 9206, 'species': 9207, 'predicated': 9208, 'erroneous': 9209, 'omnipresent': 9210, 'conceivable': 9211, 'iran-contra': 9212, 'broadly': 9213, 'construed': 9214, 'emasculate': 9215, 'swallow': 9216, 'convention': 9217, '1787': 9218, 'ensure': 9219, 'accountability': 9220, 'unitary': 9221, 'technically': 9222, 'limitation': 9223, 'leash': 9224, 'deliberating': 9225, 'breathtaking': 9226, 'containing': 9227, 'alternatively': 9228, 'intrusions': 9229, 'void': 9230, 'appoint': 9231, 'ambassadors': 9232, 'empowers': 9233, 'vacancies': 9234, 'granting': 9235, 'appropriation': 9236, 'repeals': 9237, 'imposes': 9238, 'nominee': 9239, 'anti-deficiency': 9240, 'voluntary': 9241, 'recommendation': 9242, 'blindfold': 9243, 'recommending': 9244, 'discretion': 9245, 'select': 9246, 'inquiring': 9247, 'market-oriented': 9248, 'egregious': 9249, 'proviso': 9250, 'subjecting': 9251, 'cost-benefit': 9252, 'inherently': 9253, 'prohibiting': 9254, 'wasted': 9255, 'illustrates': 9256, 'usurp': 9257, '609': 9258, 'executive-office': 9259, 'administer': 9260, 'disapproved': 9261, '*-58': 9262, 'disapproval': 9263, 'accordance': 9264, 'applicable': 9265, 'one-house': 9266, 'bicameral': 9267, 'presentation': 9268, 'signature': 9269, 'ins': 9270, 'chadha': 9271, 'vetoes': 9272, 'invite': 9273, 'purposes': 9274, 'custom': 9275, 'undo': 9276, 'then-speaker': 9277, 'mikhail': 9278, 'ratified': 9279, 'salt': 9280, 'unworkable': 9281, 'unfunded': 9282, 'assert': 9283, 'restricting': 9284, 'expressly': 9285, 'riders': 9286, 'trespass': 9287, 'prerogative': 9288, 'context': 9289, 'characterized': 9290, 'objectionable': 9291, 'conflict': 9292, 'applicability': 9293, 'item': 9294, 'exerting': 9295, 'downside': 9296, 'asserts': 9297, 'sue': 9298, 'loses': 9299, 'morrison': 9300, 'olson': 9301, 'electorate': 9302, 'valuable': 9303, 'civics': 9304, 'presumes': 9305, 'federalist': 9306, 'legislature': 9307, 'impetuous': 9308, 'vortex': 9309, 'sidak': 9310, '57.7': 9311, 'deducting': 9312, 'saved': 9313, 'reclaim': 9314, 'mortgaged': 9315, 'price-support': 9316, 'soaring': 9317, 'reclaimed': 9318, '240-page': 9319, 'parched': 9320, 'profited': 9321, 'hardest-hit': 9322, 'dakotas': 9323, 'disaster-assistance': 9324, 'confirms': 9325, 'depression': 9326, 'helps': 9327, 'reluctance': 9328, 'lobbies': 9329, 'curtailed': 9330, 'land-idling': 9331, 'price-depressing': 9332, 'surpluses': 9333, 'strengthened': 9334, 'keith': 9335, 'livestock': 9336, 'cattle': 9337, 'inventory': 9338, '3.4': 9339, 'log': 9340, 'disaster': 9341, 'farms': 9342, 'cultivation': 9343, 'cushioned': 9344, '14.5': 9345, '238,000-circulation': 9346, '700,000': 9347, 'one-newspaper': 9348, 'senses': 9349, 'dominates': 9350, '300,000': 9351, '170,000': 9352, 'pasadena': 9353, 'mccabe': 9354, 'materialized': 9355, 'stream': 9356, 'tire-kickers': 9357, 'lookee-loos': 9358, 'newsstand': 9359, 'freeway': 9360, 'inevitable': 9361, 'sprawling': 9362, 'balkanized': 9363, 'mammoth': 9364, 'limbo': 9365, 'torn': 9366, 'old-time': 9367, 'readership': 9368, 'blue-collar': 9369, 'sports-oriented': 9370, 'sprightly': 9371, 'staid': 9372, 'flirted': 9373, 'news-american': 9374, 'folded': 9375, 'herald-american': 9376, 'cornerstones': 9377, 'fanciful': 9378, 'julia': 9379, 'castle': 9380, 'simeon': 9381, 'spanish': 9382, 'renaissance-style': 9383, 'survivor': 9384, 'bygone': 9385, 'kendrick': 9386, 'looks': 9387, '1903': 9388, 'decade-long': 9389, '1967': 9390, 'moments': 9391, 'bellows': 9392, 'brightened': 9393, 'dolan': 9394, 'restored': 9395, 'limping': 9396, 'accomplishments': 9397, 'notable': 9398, 'much-larger': 9399, 'disclosures': 9400, 'coverage': 9401, 'arts': 9402, 'danzig': 9403, '730': 9404, 'long-tenured': 9405, 'representatives': 9406, 'recruiting': 9407, 'emotional': 9408, 'crying': 9409, 'l.a.': 9410, 'headline': 9411, 'beers': 9412, 'drunk': 9413, 'andy': 9414, 'furillo': 9415, 'pressman': 9416, 'headlined': 9417, 'closes': 9418, 'forget': 9419, 'handed': 9420, 'olympia': 9421, '23.4': 9422, 'radio-station': 9423, 'programmer': 9424, 'lenders': 9425, 'slogan': 9426, 'beleaguered': 9427, 'boosting': 9428, 'four-day': 9429, 'explaining': 9430, 'disclose': 9431, 'approached': 9432, 'doctor': 9433, 'broaden': 9434, '*-66': 9435, 'phillip': 9436, 'car-care': 9437, 'truth-in-lending': 9438, 'billing': 9439, 'marketers': 9440, 'capability': 9441, 'screened': 9442, 'card-member': 9443, 'incomes': 9444, 'missed': 9445, 'preapproved': 9446, 'visa': 9447, 'cards': 9448, 'tie-in': 9449, 'express-buick': 9450, 'accommodations': 9451, 'meals': 9452, 'destinations': 9453, 'honolulu': 9454, 'orlando': 9455, 'destination': 9456, 'companion': 9457, 'lieu': 9458, 'clock': 9459, 'stereo': 9460, 'recorder': 9461, 'sweepstakes': 9462, 'test-drive': 9463, 'calculator': 9464, 'travel-related': 9465, 'round-trip': 9466, 'trans': 9467, 'gilt': 9468, 'recovering': 9469, 'gilts': 9470, 'retraced': 9471, 'soured': 9472, 'accounted': 9473, 'hazell': 9474, 'auditor': 9475, 'vested': 9476, 'persuasive': 9477, 'solicitor': 9478, 'barclays': 9479, 'midland': 9480, 'citibank': 9481, 'avenues': 9482, 'illegality': 9483, 'arrangements': 9484, 'gut-wrenching': 9485, '190-point': 9486, '1,400': 9487, 'letter-writing': 9488, 'quashing': 9489, 'wrath': 9490, 'minicrash': 9491, 'shaken': 9492, 'resentment': 9493, 'lightning-fast': 9494, 'reeling': 9495, 'reap': 9496, 'academics': 9497, 'exacerbated': 9498, 'ascendency': 9499, 'consisting': 9500, 'wizards': 9501, 'immense': 9502, 'pools': 9503, 'defending': 9504, 'ramparts': 9505, 'stock-picking': 9506, 'tens': 9507, 'clannish': 9508, 'successfully': 9509, 'mobilizing': 9510, 'bludgeon': 9511, 'tormentors': 9512, 'layer': 9513, 'gigantic': 9514, 'crapshoot': 9515, 'broad-based': 9516, 'livelihood': 9517, 'palace': 9518, 'revolt': 9519, 'wohlstetter': 9520, 'rallying': 9521, 'anti-program': 9522, 'countless': 9523, 'universally': 9524, 'consistent': 9525, 'wedded': 9526, 'waited': 9527, 'sneaked': 9528, 'conceding': 9529, 'headed': 9530, 'curbed': 9531, 'contradictions': 9532, 'pitting': 9533, 'floors': 9534, 'entrenched': 9535, 'tooth': 9536, 'nail': 9537, 'facilitate': 9538, 'theme': 9539, 'greedy': 9540, 'manipulators': 9541, 'shambles': 9542, 'free-enterprise': 9543, 'gambling': 9544, 'den': 9545, 'odds': 9546, 'stacked': 9547, 'off-track': 9548, 'portray': 9549, 'old-fashioned': 9550, 'neanderthals': 9551, 'witches': 9552, 'boogieman': 9553, 'unknown': 9554, 'beg': 9555, 'momentary': 9556, 'divergence': 9557, 'constitute': 9558, 'whichever': 9559, 'seconds': 9560, 'razor-thin': 9561, 'profess': 9562, 'despise': 9563, 'frightened': 9564, 'lotter': 9565, 'zicklin': 9566, 'iota': 9567, 'hans': 9568, 'unraveling': 9569, 'unload': 9570, 'takeover-stock': 9571, 'arbitragers': 9572, 'signore': 9573, 'speculator': 9574, 'arbs': 9575, 'overleveraged': 9576, 'apart': 9577, 'traditionalists': 9578, 'bundles': 9579, 'derisively': 9580, 'jockeys': 9581, 'manage': 9582, 'benchmarks': 9583, 'old-style': 9584, 'juggle': 9585, 'pennies': 9586, 'pension-fund': 9587, 'automated': 9588, 'threatens': 9589, 'dinosaurs': 9590, 'stock-specialist': 9591, 'monopoly': 9592, 'knell': 9593, 'striving': 9594, 'printers': 9595, 'spooked': 9596, 'dismayed': 9597, 'stacking': 9598, 'deck': 9599, 'scaring': 9600, 'raymond': 9601, '71,309': 9602, 'resent': 9603, '*-75': 9604, 'nameless': 9605, 'sweatshirts': 9606, 'sparkplugs': 9607, 'oh': 9608, 'bloods': 9609, 'publicity': 9610, 'orchestrated': 9611, 'hunker': 9612, 'lynch-mob': 9613, 'cow': 9614, 'gored': 9615, 'proven': 9616, 'fastest-growing': 9617, 'minted': 9618, 'millionaires': 9619, '20s': 9620, '30s': 9621, 'thunder': 9622, 'jeopardy': 9623, 'unlikely': 9624, '*-77': 9625, 'middle-ground': 9626, 'enjoy': 9627, 'good-faith': 9628, 'potentially': 9629, 'protects': 9630, 'relentlessly': 9631, 'destroy': 9632, 'efficiency': 9633, 'fundamentalists': 9634, 'stock-price': 9635, 'band-wagon': 9636, 'picks': 9637, 'impetus': 9638, 'practiced': 9639, 'locations': 9640, 'leveraging': 9641, 'owning': 9642, 'cheapest': 9643, 'vast': 9644, 'hysteria': 9645, 'arise': 9646, 'cumbersome': 9647, 'desires': 9648, 'evolve': 9649, 'creature': 9650, 'evoking': 9651, 'curses': 9652, 'implemented': 9653, 'unneeded': 9654, 'harmful': 9655, 'sufficient': 9656, 'responds': 9657, 'initiating': 9658, 'functioning': 9659, 'fundamentally': 9660, 'hypothetical': 9661, 'sacrifice': 9662, 'subtraction': 9663, 'finite': 9664, 'loathsome': 9665, 'tablets': 9666, 'labeling': 9667, 'akin': 9668, 'please': 9669, 'sufficiently': 9670, 'deviation': 9671, 'advent': 9672, 'undergoing': 9673, '1973-75': 9674, '1937-40': 9675, '1928-33': 9676, 'scream': 9677, 'hailing': 9678, 'abounding': 9679, 'deeds': 9680, 'goblins': 9681, 'fixes': 9682, 'whipping': 9683, 'boy': 9684, 'wooing': 9685, 'stock-selection': 9686, 'bringing': 9687, 'damaging': 9688, 'abolishing': 9689, 'merits': 9690, 'championing': 9691, 'loudest': 9692, 'cater': 9693, 'advise': 9694, 'amass': 9695, 'somehow': 9696, 'studiously': 9697, 'devouring': 9698, 'clippings': 9699, 'guy': 9700, 'mutual-fund': 9701, 'sad': 9702, 'performers': 9703, 'grapple': 9704, 'cost-effective': 9705, 'sexy': 9706, 'roadblock': 9707, '*-82': 9708, 'high-volume': 9709, 'legislating': 9710, 'temporary': 9711, 'watchdogs': 9712, 'friction': 9713, 'two-tiered': 9714, 'taxation': 9715, 'etc.': 9716, 'loser': 9717, 'inviting': 9718, 'transfer': 9719, 'executes': 9720, 'brooks': 9721, 'augment': 9722, 'rill': 9723, 'appropriated': 9724, 'hart-scott-rodino': 9725, 'notify': 9726, 'completing': 9727, 'don': 9728, 'stifle': 9729, 'staffs': 9730, 'drastically': 9731, 'dismal': 9732, 'inhibit': 9733, 'acquirers': 9734, 'noticed': 9735, 'off-year': 9736, 'ratcheting': 9737, 'referenda': 9738, 'citizen-sparked': 9739, 'ballots': 9740, 'maine': 9741, 'missiles': 9742, 'dakota': 9743, 'schmidt': 9744, 'cue': 9745, 'magleby': 9746, 'brigham': 9747, 'christie': 9748, 'folio': 9749, 'equivalents': 9750, '396,000': 9751, 'single-lot': 9752, 'variables': 9753, 'documented': 9754, 'anecdotal': 9755, 'gates-warren': 9756, 'sotheby': 9757, 'museums': 9758, 'persky': 9759, 'collector': 9760, 'masters': 9761, 'fetching': 9762, 'barth': 9763, 'photography': 9764, 'dialing': 9765, 'million-a-year': 9766, 'joel': 9767, 'caller': 9768, 'celebrity': 9769, 'chatter': 9770, 'horoscopes': 9771, 'andrea': 9772, 'tutorials': 9773, 'eyeing': 9774, 'merchandising': 9775, 'migrate': 9776, 'predicts': 9777, 'lawless': 9778, 'sprint': 9779, 'pets': 9780, 'recovery': 9781, 'milwaukee': 9782, 'canine': 9783, 'feline': 9784, 'appetite': 9785, 'receptive': 9786, 'therapy': 9787, "o'loughlin": 9788, 'coordinator': 9789, 'hammacher': 9790, 'schlemmer': 9791, 'fiber-optic': 9792, 'christmas': 9793, 'string': 9794, '6,500': 9795, 'continuously': 9796, 'colored': 9797, 'fiber-end': 9798, 'bunches': 9799, 'prompts': 9800, 'bilingual': 9801, 'tokio': 9802, 'protocols': 9803, 'preventative': 9804, 'comeback': 9805, 'shrinking': 9806, 'landfills': 9807, 'super-absorbent': 9808, 'disposables': 9809, 'tots': 9810, '1,200': 9811, 'mogavero': 9812, 'piscataway': 9813, 'syracuse': 9814, 'dydee': 9815, 'stresses': 9816, 'awareness': 9817, 'day-care': 9818, 'spurned': 9819, '672': 9820, 'inquiries': 9821, 'elisa': 9822, 'hollis': 9823, 'shortages': 9824, 'stork': 9825, 'springfield': 9826, 'spurring': 9827, 'velcro': 9828, 'briefs': 9829, '57.6': 9830, 'yorkers': 9831, 'viewership': 9832, 'columbus': 9833, 'freudtoy': 9834, 'pillow': 9835, 'likeness': 9836, 'sigmund': 9837, 'freud': 9838, '24.95': 9839, 'tool': 9840, 'do-it-yourself': 9841, 'doubts': 9842, 'echoed': 9843, 'heading': 9844, 'blames': 9845, 'overvalued': 9846, 'interior': 9847, 'decorator': 9848, 'spook': 9849, 'deviant': 9850, 'curbing': 9851, 'darned': 9852, 'schwab': 9853, 'buckhead': 9854, 'skepticism': 9855, 'anderson': 9856, '59-year-old': 9857, 'fluctuations': 9858, 'heebie-jeebies': 9859, 'outlawing': 9860, 'wamre': 9861, '31-year-old': 9862, 'disappear': 9863, 'dealing': 9864, 'decries': 9865, 'strictly': 9866, 'capitalism': 9867, 'adapting': 9868, 'britta': 9869, '25-year-old': 9870, 'factoring': 9871, 'rigors': 9872, 'silverman': 9873, 'insurance-company': 9874, 'culprit': 9875, 'arbitraging': 9876, 'leery': 9877, 'enzor': 9878, 'defends': 9879, 'accountant': 9880, 'recouped': 9881, 'flim-flammery': 9882, 'storm': 9883, 'lucille': 9884, '84-year-old': 9885, 'housewife': 9886, 'amazingly': 9887, 'jolts': 9888, 'hunted': 9889, 'bargains': 9890, 'sky': 9891, 'wholesaler': 9892, 'spirits': 9893, 'underwoods': 9894, 'amps': 9895, 'centimeter': 9896, 'yttrium-containing': 9897, 'liquid-nitrogen': 9898, 'temperature': 9899, '321': 9900, 'fahrenheit': 9901, 'wires': 9902, 'magnets': 9903, 'generators': 9904, 'ceramic': 9905, 'technologies': 9906, 'generation': 9907, 'aspect': 9908, 'bombarding': 9909, 'neutrons': 9910, 'radioactivity': 9911, 'large-scale': 9912, 'breathed': 9913, 'collective': 9914, 'sigh': 9915, 'demonstrates': 9916, 'flux': 9917, 'pinning': 9918, 'undercutting': 9919, 'determining': 9920, 'enable': 9921, 'combine': 9922, 'melt-textured': 9923, 'walbrecher': 9924, 'francisco-based': 9925, '1st': 9926, 'fidelity': 9927, 'versus': 9928, '5.3': 9929, '1.61': 9930, 'write-downs': 9931, '4.9': 9932, '45.75': 9933, 'offerings': 9934, 'pricings': 9935, 'non-u.s.': 9936, '8.467': 9937, 'bellwether': 9938, '1991-2000': 9939, '81.8': 9940, '7.20': 9941, 'insured': 9942, 'triple-a-rated': 9943, 'a-d': 9944, '1991-1999': 9945, '7.422': 9946, '7.15': 9947, 'single-a': 9948, '80.8': 9949, '30.9': 9950, '1992-1999': 9951, '7.45': 9952, '7.50': 9953, '49.9': 9954, '7.65': 9955, 'double-a': 9956, 'heiwado': 9957, 'equity-purchase': 9958, 'intecknings': 9959, 'garanti': 9960, 'aktiebolaget': 9961, 'sweden': 9962, '6.03': 9963, 'handelsbanken': 9964, 'takashima': 9965, 'yamaichi': 9966, '3.43': 9967, 'pencil': 9968, '106': 9969, '3.42': 9970, 'koizumi': 9971, 'sangyo': 9972, '1996': 9973, 'schweiz': 9974, 'lurie': 9975, 'avid': 9976, 'fan': 9977, 'digs': 9978, 'afford': 9979, 'major-league': 9980, 'ranging': 9981, 'petersburg': 9982, 'complexes': 9983, 'moneymakers': 9984, 'pepperdine': 9985, 'baim': 9986, 'scoffs': 9987, 'looked': 9988, 'dodger': 9989, 'redistribute': 9990, 'mega-stadium': 9991, 'phoenix': 9992, 'thumbs': 9993, 'fielded': 9994, 'concede': 9995, 'endorse': 9996, 'dolphins': 9997, 'disagrees': 9998, 'city-owned': 9999, 'bowl': 10000, 'coliseum': 10001, 'moon': 10002, 'landrieu': 10003, 'cavernous': 10004, 'money-losing': 10005, 'relevance': 10006, 'faith': 10007, 'egyptian': 10008, 'pharaoh': 10009, 'justified': 10010, 'schemes': 10011, 'pharaohs': 10012, 'erect': 10013, 'playgrounds': 10014, 'passions': 10015, '89.7': 10016, '141.9': 10017, '94.8': 10018, '149.9': 10019, '130.6': 10020, '128': 10021, '133': 10022, '135': 10023, '388': 10024, '722': 10025, 'erodes': 10026, 'downgrading': 10027, 'shivers': 10028, 'shudders': 10029, 'issuers': 10030, 'ious': 10031, 'borrowings': 10032, 'difficulty': 10033, 'shoring': 10034, 'credit-rating': 10035, 'whereas': 10036, 'structured': 10037, 'penchant': 10038, 'stretch': 10039, 'disarray': 10040, 'downgraded': 10041, 'catch-22': 10042, 'riskier': 10043, 'acquires': 10044, 'expectation': 10045, 'generated': 10046, 'overcapacity': 10047, 'managements': 10048, 'arranged': 10049, 'high-risk': 10050, 'mattress': 10051, 'bedding': 10052, 'suisse': 10053, 'participant': 10054, 'rebounding': 10055, 'overstated': 10056, 'disadvantage': 10057, 'rivals': 10058, 'toll': 10059, 'settlements': 10060, 'plea': 10061, 'felonies': 10062, 'insider-trading': 10063, 'workforce': 10064, 'circulated': 10065, 'confident': 10066, 'creditworthiness': 10067, 'vicissitudes': 10068, 'sable': 10069, 'experiencing': 10070, 'creator': 10071, 'car-development': 10072, 'eclipse': 10073, 'top-selling': 10074, 'involvement': 10075, 'assembly-line': 10076, 'cycle': 10077, 'responsive': 10078, 'demands': 10079, 'new-car': 10080, 'cougar': 10081, 'american-made': 10082, 'parts-engineering': 10083, 'absurdity': 10084, 'stretched': 10085, 'notch': 10086, 'compassion': 10087, 'solomonic': 10088, 'policy-making': 10089, 'innovation': 10090, 'compelling': 10091, '1940s': 10092, 'hormone': 10093, 'diethylstilbestrol': 10094, 'miscarriages': 10095, 'sickness': 10096, 'generic': 10097, 'labels': 10098, 'mothers': 10099, 'thousand': 10100, 'recall': 10101, 'brand': 10102, '1980': 10103, 'common-law': 10104, 'pills': 10105, 'assessed': 10106, 'proportion': 10107, 'duck': 10108, 'identical': 10109, 'slippery': 10110, 'slope': 10111, 'alike': 10112, 'differently': 10113, 'warranties': 10114, 'lilly': 10115, 'mindy': 10116, 'alstyne': 10117, 'apples': 10118, 'uncompensated': 10119, 'takings': 10120, 'twisting': 10121, 'doctrine': 10122, 'reversed': 10123, 'chilled': 10124, 'prescription': 10125, 'hidden': 10126, 'favors': 10127, 'accompany': 10128, 'pain': 10129, 'unanimous': 10130, 'anti-morning-sickness': 10131, 'bendectin': 10132, 'huber': 10133, 'introduces': 10134, 'anti-miscarriage': 10135, 'tort': 10136, 'lousy': 10137, 'anyway': 10138, 'contingency-fee': 10139, 'vaccines': 10140, 'predictably': 10141, 'understands': 10142, 'utterly': 10143, 'incapable': 10144, 'deserving': 10145, 'billion-dollar': 10146, 'morass': 10147, 'trash': 10148, 'odyssey': 10149, 'norwalk': 10150, 'n.v': 10151, '235': 10152, '113.2': 10153, '6.70': 10154, '144': 10155, '4.10': 10156, 'outlay': 10157, '2.46': 10158, '2.42': 10159, 'phacoflex': 10160, 'intraocular': 10161, 'foldable': 10162, 'silicone': 10163, 'len': 10164, 'foldability': 10165, 'inserted': 10166, 'incisions': 10167, 'cataracts': 10168, 'refer': 10169, 'clouding': 10170, 'ec': 10171, 'harmed': 10172, 'salvador': 10173, 'intentioned': 10174, 'wrecking': 10175, 'incentives': 10176, 'smothering': 10177, 'kindness': 10178, '7.8': 10179, '13.625': 10180, 'wis.': 10181, '4.5': 10182, '1.92': 10183, '352.9': 10184, 'softer': 10185, 'motor-home': 10186, 'deere': 10187, '132': 10188, '14.99': 10189, '35564.43': 10190, '63.79': 10191, '35500.64': 10192, '909': 10193, 'outnumbered': 10194, '454': 10195, '451': 10196, 'incentive-backed': 10197, 'untrue': 10198, 'directionless': 10199, 'topix': 10200, '16.05': 10201, '1.46': 10202, '0.05': 10203, '2691.19': 10204, '6.84': 10205, '5.92': 10206, '0.16': 10207, '3648.82': 10208, 'nomura': 10209, 'clearer': 10210, 'comfortably': 10211, 'tokyu': 10212, 'dominant': 10213, 'yasuda': 10214, '56': 10215, '1,880': 10216, '13.15': 10217, 'continuingly': 10218, 'directed': 10219, 'teikoku': 10220, 'stimulated': 10221, '1,460': 10222, 'showa': 10223, 'shell': 10224, '1,570': 10225, 'sumitomo': 10226, '692': 10227, '960': 10228, 'winners': 10229, 'shokubai': 10230, '2,410': 10231, 'marubeni': 10232, '890': 10233, 'affecting': 10234, '17.5': 10235, '2160.1': 10236, 'intraday': 10237, '2141.7': 10238, '2163.2': 10239, 'tempted': 10240, 'overdone': 10241, 'reversal': 10242, 'ft': 10243, '30-share': 10244, '1738.1': 10245, '372.9': 10246, '334.5': 10247, 'hugging': 10248, '879': 10249, '13.90': 10250, 'shed': 10251, 'waive': 10252, 'protective': 10253, 'golden': 10254, 'waiver': 10255, 'goldsmith': 10256, '753': 10257, 'sweeten': 10258, '397': 10259, 'shopping': 10260, 'carlton': 10261, '778': 10262, 'notched': 10263, 'searched': 10264, 'qualities': 10265, 'wellcome': 10266, '666': 10267, 'glaxo': 10268, '14.13': 10269, 'amsterdam': 10270, 'zurich': 10271, 'paris': 10272, 'brussels': 10273, 'milan': 10274, 'wellington': 10275, 'sydney': 10276, 'manila': 10277, 'calculated': 10278, 'geneva': 10279, '1969': 10280, 'equaling': 10281, 'intermec': 10282, '1,050,000': 10283, 'piper': 10284, 'jaffray': 10285, 'hopwood': 10286, 'middlesex': 10287, '150,000': 10288, 'walker': 10289, 'howard': 10290, 'weil': 10291, 'labouisse': 10292, 'friedrichs': 10293, 'midwesco': 10294, '830,000': 10295, 'nylev': 10296, 'occidental': 10297, 'shelf': 10298, 'inns': 10299, 'zero': 10300, 'montgomery': 10301, 'fracturing': 10302, 'lovett': 10303, 'mitchell': 10304, 'webb': 10305, 'garrison': 10306, 'blunt': 10307, 'ellis': 10308, 'loewi': 10309, '3,250,000': 10310, '3,040,000': 10311, '210,000': 10312, 'hanifen': 10313, 'imhoff': 10314, 'putty': 10315, 'lipsticks': 10316, 'liners': 10317, 'lotions': 10318, 'creams': 10319, 'tackle': 10320, 'paint': 10321, 'spackle': 10322, "d'amico": 10323, 'diceon': 10324, 'chatsworth': 10325, 'charging': 10326, 'calif.-based': 10327, 'circuit-board': 10328, 'disposed': 10329, 'caustic': 10330, 'sewer': 10331, 'leaky': 10332, 'unlabeled': 10333, 'open-top': 10334, 'roland': 10335, 'matthews': 10336, 'jonas': 10337, 'inspection': 10338, 'arraignments': 10339, 'stayed': 10340, 'unsettling': 10341, 'fabian': 10342, 'industry-supported': 10343, '116.4': 10344, 'barely': 10345, 'revised': 10346, '116.3': 10347, '116.9': 10348, '112.9': 10349, '120.7': 10350, '18.3': 10351, 'worsen': 10352, '21.1': 10353, '16.9': 10354, '17.4': 10355, '18.6': 10356, '28.5': 10357, 'day-to-day': 10358, 'paycheck': 10359, 'reasonably': 10360, 'conducted': 10361, '6.7': 10362, '30.6': 10363, '27.4': 10364, '26.5': 10365, 'deluge': 10366, 'abuzz': 10367, 'kathleen': 10368, 'camilli': 10369, 'fixed-income': 10370, 'rare': 10371, 'ednie': 10372, 'ranked': 10373, 'infrequent': 10374, 'scant': 10375, 'hanover': 10376, 'blurred': 10377, '14.': 10378, 'varied': 10379, 'diverse': 10380, 'enhances': 10381, 'importance': 10382, 'non-farm': 10383, '152,000': 10384, '29year': 10385, 'nine-month': 10386, '12\\/32': 10387, '16\\/32': 10388, '7.79': 10389, '7.52': 10390, '7.60': 10391, 'compilation': 10392, 'topped': 10393, '16.5': 10394, 'risk-free': 10395, 'outdistanced': 10396, 'lonski': 10397, 'wanting': 10398, '9.32': 10399, '12-year': 10400, 'derivative': 10401, '114': 10402, 'relaunched': 10403, '107.03': 10404, 'repriced': 10405, 'bucking': 10406, 'unenthusiastic': 10407, 'bund': 10408, '1998': 10409, '95.09': 10410, '5.435': 10411, '2003\\/2007': 10412, '14\\/32': 10413, '10.19': 10414, '1995': 10415, '9\\/32': 10416, '103': 10417, '11.10': 10418, 'double-c': 10419, 'triple-c': 10420, 'clothing': 10421, 'expense': 10422, 'exceeds': 10423, '94.2': 10424, '83': 10425, 'pre-tax': 10426, '306': 10427, '415': 10428, 'underperforming': 10429, 'monopolies': 10430, 'tires': 10431, '221.4': 10432, 's.a': 10433, 'referral': 10434, '420': 10435, 'centerpiece': 10436, 'chaos': 10437, 'single-handed': 10438, 'seven-million-ton': 10439, 'cap': 10440, 'hailed': 10441, 'despised': 10442, 'innovative': 10443, 'market-based': 10444, 'polluters': 10445, 'subsidize': 10446, 'clean-up': 10447, 'coal-fired': 10448, 'sparing': 10449, 'exorbitant': 10450, 'jumps': 10451, 'sticking': 10452, 'vow': 10453, 'avoiding': 10454, 'staunchly': 10455, 'appease': 10456, 'high-polluting': 10457, 'cleanup': 10458, 'burn': 10459, 'cleaner-burning': 10460, 'fuels': 10461, 'quietly': 10462, 'tinker': 10463, 'journals': 10464, 'resign': 10465, 'relocate': 10466, 'co-founded': 10467, 'mo.-based': 10468, 'e.w.': 10469, 'scripps': 10470, 'nickname': 10471, 'coordinate': 10472, 'deliberately': 10473, 'disconnect': 10474, 'routes': 10475, 'scans': 10476, 'resumes': 10477, 'dot': 10478, 'high-speed': 10479, 'zip': 10480, 'handles': 10481, 'instrument': 10482, 'agreed-upon': 10483, 'nullified': 10484, 'opposite': 10485, 'indexing': 10486, 'barometer': 10487, 'swapping': 10488, 'simultaneous': 10489, 'quant': 10490, 'quantitive': 10491, 'newest': 10492, 'breed': 10493, 'rocket': 10494, 'backgrounds': 10495, 'hedging': 10496, 'popularly': 10497, 'fleeting': 10498, 'arbitrager': 10499, 'multiplying': 10500, '20-stock': 10501, 'mimics': 10502, 'certin': 10503, 'indexes': 10504, 'expression': 10505, 'signifying': 10506, 'saul': 10507, 'puzzled': 10508, 'definitely': 10509, 'airways': 10510, '110': 10511, '282': 10512, 'twist': 10513, '1.50': 10514, 'notified': 10515, 'tapping': 10516, 'tiger': 10517, 'acquirer': 10518, 'pilots': 10519, 'machinists': 10520, 'cleared': 10521, '472': 10522, 'delisted': 10523, '23.25': 10524, '28.25': 10525, 'duluth': 10526, 'liabilities': 10527, 'trace': 10528, 'unaffiliated': 10529, '3.625': 10530, 'asher': 10531, '16.2': 10532, 'computer-services': 10533, 'oust': 10534, 'datapoint': 10535, 'explore': 10536, '2.75': 10537, 'above-market': 10538, '18-a-share': 10539, 'p.m.': 10540, 'est': 10541, '576': 10542, '95': 10543, '24,000': 10544, '19.50': 10545, 'barron': 10546, 'printing-press': 10547, 'trail': 10548, 'heavy-truck': 10549, 'passenger-car': 10550, '630.9': 10551, '126.1': 10552, '132.9': 10553, 'prior-year': 10554, 'adjustment': 10555, 'bomber': 10556, 'sewing-machine': 10557, '185.9': 10558, '3.28': 10559, '3.16': 10560, '107.9': 10561, '96.4': 10562, 'emerged': 10563, 'muscling': 10564, 'singled': 10565, '79': 10566, '42.1': 10567, 'colorliner': 10568, 'newspaper-printing': 10569, 'sagged': 10570, 'bombers': 10571, 'resumption': 10572, 'shuttle': 10573, 'expendable': 10574, 'launch-vehicle': 10575, 'engines': 10576, 'hits': 10577, 'weapons-modernization': 10578, 'c-130': 10579, '734.9': 10580, '811.9': 10581, '3.04': 10582, '2.47': 10583, '7.4': 10584, '2.30': 10585, '12.52': 10586, '11.95': 10587, 'austin': 10588, 'texas-based': 10589, '210': 10590, '512': 10591, 'kilobytes': 10592, '40-megabyte': 10593, '2,099': 10594, 'more-advanced': 10595, 'microprocessor': 10596, 'megabytes': 10597, '100-megabyte': 10598, '5,699': 10599, '6,799': 10600, '286': 10601, 'microprocessors': 10602, 'grower': 10603, 'life-of-contract': 10604, '14.54': 10605, '14.28': 10606, '14.53': 10607, '0.56': 10608, 'restraints': 10609, '0.54': 10610, '14.26': 10611, 'permissible': 10612, '0.50': 10613, '14.00': 10614, 'near-limit': 10615, '1990-91': 10616, 'regardless': 10617, 'third-largest': 10618, 'fifth-largest': 10619, 'drastic': 10620, '1988-89': 10621, 'judith': 10622, 'granted': 10623, 'licenses': 10624, 'planting': 10625, 'trees': 10626, 'cane': 10627, 'hackensack': 10628, 'atmosphere': 10629, 'ethanol': 10630, 'fleet': 10631, 'importer': 10632, '60.36': 10633, '20.07': 10634, 'colder': 10635, 'muted': 10636, 'observance': 10637, 'saints': 10638, 'buys': 10639, 'verge': 10640, 'designation': 10641, 'generous': 10642, '1.1650': 10643, 'ignored': 10644, 'disputado': 10645, 'reuter': 10646, 'emergency': 10647, 'copper-rich': 10648, 'inoperative': 10649, 'native': 10650, 'landowners': 10651, 'secede': 10652, 'cos.': 10653, '36-store': 10654, 'rang': 10655, '313': 10656, 'hubbell': 10657, 'all-cash': 10658, 'liquidation': 10659, 'paso': 10660, 'boots': 10661, 'leather': 10662, 'accrue': 10663, 'unspecified': 10664, 'termed': 10665, 'amicable': 10666, '27-year': 10667, 'information-services': 10668, 'several-year': 10669, 'stemmed': 10670, 'directorship': 10671, 'eight-person': 10672, 'shepperd': 10673, 'ubs': 10674, 'irrelevant': 10675, '913': 10676, '14.43': 10677, '43.875': 10678, 'nigel': 10679, 'judah': 10680, 'holland': 10681, 'mannix': 10682, 'kingsbridge': 10683, '45-a-share': 10684, 'donuts': 10685, '50.1': 10686, 'delaware': 10687, 'dunkin': 10688, 'deter': 10689, '38.5': 10690, 'receipt': 10691, '35.2': 10692, 'conn.based': 10693, 'magnified': 10694, 'nonrecurring': 10695, '8.2': 10696, 'asset-valuation': 10697, 'adjustments': 10698, '85.7': 10699, '93.3': 10700, 'natural-gas': 10701, 'reimbursed': 10702, 'amortization': 10703, '169.9': 10704, 'blue-chips': 10705, 'brunt': 10706, '41.60': 10707, '0.84': 10708, '341.20': 10709, '0.99': 10710, '319.75': 10711, '0.60': 10712, '188.84': 10713, 'decliners': 10714, '176.1': 10715, 'arrive': 10716, 'firmed': 10717, 'faltered': 10718, 'pains': 10719, 'arbitrage-related': 10720, 'awaits': 10721, 'reluctant': 10722, 'stick': 10723, 'necks': 10724, 'woolworth': 10725, 'paramount': 10726, 'ferro': 10727, 'early-retirement': 10728, 'amr': 10729, 'developer': 10730, 'withdrew': 10731, '120-a-share': 10732, 'derchin': 10733, '5.1': 10734, 'mead': 10735, 'louisiana-pacific': 10736, '5.6': 10737, 'ex-dividend': 10738, '1.56': 10739, '372.14': 10740, '11,390,000': 10741, 'spaghetti': 10742, 'warehouse': 10743, 'convert': 10744, 'adverse': 10745, 'orleans-based': 10746, 'yet-to-be-formed': 10747, 'distributed': 10748, 'cents-a-unit': 10749, '108': 10750, '88.32': 10751, '618.1': 10752, '77.6': 10753, 'backdrop': 10754, 'continuous': 10755, '40.21': 10756, '16.09': 10757, '28.36': 10758, '11.72': 10759, '1.916': 10760, '1.637': 10761, 'seven-yen': 10762, 'atsushi': 10763, 'muramatsu': 10764, 'experienced': 10765, 'remarkable': 10766, 'difficulties': 10767, 'firmly': 10768, '2.35': 10769, '14.75': 10770, 'customized': 10771, 'simulates': 10772, 'unexpected': 10773, 'fairlawn': 10774, 'full-year': 10775, '148': 10776, '2.19': 10777, 'harry': 10778, 'millis': 10779, 'mcdonald': 10780, 'unanticipated': 10781, 'government-owned': 10782, 'subcontractor': 10783, 'cluster': 10784, 'bombs': 10785, 'aerojet': 10786, 'ordnance': 10787, '93.9': 10788, '1.19': 10789, '92.9': 10790, '1.18': 10791, '6.4': 10792, 'hasbrouk': 10793, 'waivers': 10794, 'distributes': 10795, 'produces': 10796, 'literature': 10797, 'displays': 10798, '158,666': 10799, '26,956': 10800, '608,413': 10801, '967,809': 10802, '1.35': 10803, 'multinational': 10804, 'haden': 10805, 'maclellan': 10806, 'surrey': 10807, 'feniger': 10808, 'acquisition-minded': 10809, 'seattle-based': 10810, '62.1': 10811, 'outbid': 10812, 'ratner': 10813, 'mid-afternoon': 10814, '260': 10815, '87-store': 10816, 'derived': 10817, 'averted': 10818, 'fantasy': 10819, '2,050-passenger': 10820, 'slated': 10821, '12.09': 10822, 'nuys': 10823, '132,000': 10824, 'write-off': 10825, 'realestate': 10826, '361.8': 10827, 'peoria': 10828, 'adopt': 10829, 'kalamazoo': 10830, 'mich.-based': 10831, 'severance': 10832, 'staying': 10833, 'high-flying': 10834, 'toy': 10835, 'peaked': 10836, '430': 10837, '92': 10838, 'reorganized': 10839, 'ranger': 10840, '225,000': 10841, 'adam': 10842, 'plagued': 10843, 'glitches': 10844, 'fortunes': 10845, 'bounced': 10846, 'cabbage': 10847, 'patch': 10848, 'dolls': 10849, 'winner': 10850, 'bankruptcy-law': 10851, 'nicaraguan': 10852, 'u.s.-backed': 10853, '19-month-old': 10854, 'cease-fire': 10855, 'reaffirmed': 10856, 'thwart': 10857, 'demobilize': 10858, 'deplorable': 10859, 'brushed': 10860, 'renewing': 10861, 'insurgents': 10862, 'contra': 10863, 'honduras': 10864, 'sandinista': 10865, 'offensive': 10866, 'rebel': 10867, 'krenz': 10868, 'freedoms': 10869, 'socialism': 10870, 'fled': 10871, 'cross-border': 10872, 'emigres': 10873, 'conferees': 10874, 'africa': 10875, 'armed': 10876, 'namibian': 10877, 'nationalist': 10878, 'guerrillas': 10879, 'neighboring': 10880, 'angola': 10881, 'violating': 10882, 'u.n.-supervised': 10883, 'peace': 10884, 'territory': 10885, 'alert': 10886, 'guerrilla': 10887, 'sabotage': 10888, 'namibia': 10889, 'gunmen': 10890, 'lebanon': 10891, 'assassinated': 10892, 'arabian': 10893, 'pro-iranian': 10894, 'islamic': 10895, 'slaying': 10896, 'avenge': 10897, 'beheading': 10898, 'terrorists': 10899, 'riyadh': 10900, 'beirut': 10901, 'moslem': 10902, 'implements': 10903, 'rulers': 10904, 'pledged': 10905, 'modernization': 10906, 'impeding': 10907, 'pakistan': 10908, 'bhutto': 10909, 'defeated': 10910, 'no-confidence': 10911, '42-year': 10912, '11-month-old': 10913, 'islamabad': 10914, '237-seat': 10915, 'rigged': 10916, 'shipboard': 10917, 'malta': 10918, '2-3': 10919, 'tete-a-tete': 10920, 'trafficking': 10921, 'andean': 10922, 'coffee': 10923, 'pan': 10924, 'subpoenaed': 10925, 'cia': 10926, 'fbi': 10927, 'bomb': 10928, 'planted': 10929, 'aboard': 10930, 'exploded': 10931, 'scotland': 10932, 'attwood': 10933, 'acute': 10934, 'anemic': 10935, 'tendering': 10936, '99.3': 10937, '3.55': 10938, '1.4': 10939, 'unresolved': 10940, 'jon': 10941, 'peters': 10942, 'guber': 10943, 'laying': 10944, 'centralized': 10945, '492': 10946, '4.55': 10947, '12.97': 10948, 'disasters': 10949, 'unconsolidated': 10950, 'sept.30': 10951, '16.68': 10952, '116.7': 10953, '12.68': 10954, '292.32': 10955, '263.07': 10956, '7.63': 10957, '5.82': 10958, '7.84': 10959, '6.53': 10960, 'brisk': 10961, 'bulldozers': 10962, '142.84': 10963, '126.15': 10964, 'climb': 10965, '566.54': 10966, '28.53': 10967, '12.82': 10968, 'outlays': 10969, 'apparent': 10970, 'drops': 10971, 'crises': 10972, '55-a-share': 10973, 'minimum-wage': 10974, 'resigning': 10975, 'depends': 10976, '1206.26': 10977, '220.45': 10978, '3436.58': 10979, '129.91': 10980, '0.28': 10981, '131.01': 10982, '1.17': 10983, '0.95': 10984, '0.0085': 10985, 'fizzled': 10986, 'pall': 10987, 'high-rolling': 10988, '43-year-old': 10989, '12.7': 10990, 'bears': 10991, 'shaky': 10992, 'flies': 10993, 'planes': 10994, 'salable': 10995, '458': 10996, 'cushion': 10997, 'lindner': 10998, 'irwin': 10999, 'jacobs': 11000, 'latter': 11001, 'mulling': 11002, 'mitigate': 11003, 'distinct': 11004, 'reserved': 11005, '227': 11006, 'till': 11007, 'issuer': 11008, 'bust': 11009, 'restructures': 11010, 'longtime': 11011, 'battered': 11012, 'gillett': 11013, 'unrealized': 11014, 'restructurings': 11015, 'balloon': 11016, 'floated': 11017, 'mellon': 11018, 'shopped': 11019, 'spun': 11020, 'workable': 11021, 'capitalize': 11022, 'bread-and-butter': 11023, 'sanford': 11024, 'pauline': 11025, 'yoshihashi': 11026, 'nyse': 11027, 'csv': 11028, '1.49': 11029, '11.57': 11030, '83,206': 11031, 'protein': 11032, 'recombinant': 11033, 'dna': 11034, 'sandoz': 11035, 'preclinical': 11036, 'marrow': 11037, 'blood-cell': 11038, 'protein-1': 11039, 'induce': 11040, 'formation': 11041, 'compositions': 11042, 'defects': 11043, 'fracture': 11044, 'healing': 11045, 'periodontal': 11046, 'cancers': 11047, 'clarence': 11048, 'advocating': 11049, 'discrimination': 11050, 'fourteen': 11051, 'jurisdiction': 11052, 'eeoc': 11053, 'judgment': 11054, 'runkel': 11055, 'nominees': 11056, 'satisfactorily': 11057, 'bench': 11058, 'vacancy': 11059, 'ferdinand': 11060, 'germany-based': 11061, 'boilers': 11062, 'pipes': 11063, 'lurgi': 11064, 'g.m.b': 11065, 'shedding': 11066, '434.4': 11067, 'injuring': 11068, 'investigate': 11069, 'sweater': 11070, 'defines': 11071, 'assess': 11072, 'apparel': 11073, 'agreements': 11074, 'manmade-fiber': 11075, '405': 11076, 'wilmington': 11077, 'neoprene': 11078, 'implementation': 11079, 'theodore': 11080, 'headcount-control': 11081, 'questioned': 11082, 'gelles': 11083, 'wertheim': 11084, 'schroder': 11085, 'trim': 11086, '275': 11087, '350': 11088, '21,000': 11089, '87.5': 11090, '38.875': 11091, 'bidder': 11092, 'shareholder-rights': 11093, 'unwanted': 11094, 'suitors': 11095, 'cost-control': 11096, 'staff-reduction': 11097, 'trimmed': 11098, 'inter-tel': 11099, 'chapman': 11100, 'waymar': 11101, 'holt': 11102, '326': 11103, '19.95': 11104, 'reassuring': 11105, 'preface': 11106, 'integrity': 11107, 'trip': 11108, 'mindful': 11109, 'plaintive': 11110, 'high-minded': 11111, 'sticky': 11112, 'fingers': 11113, 'sweaty': 11114, 'matthew': 11115, 'harrison': 11116, 'path': 11117, 'traveled': 11118, 'inception': 11119, 'full-fledged': 11120, 'vital': 11121, 'revolves': 11122, 'johnson-era': 11123, 'mandates': 11124, 'noncompetitively': 11125, 'ancestry': 11126, 'born': 11127, 'puerto': 11128, 'rico': 11129, 'falsify': 11130, '50\\/50': 11131, 'races': 11132, 'minority-owned': 11133, 'blighted': 11134, 'famous': 11135, 'jimmy': 11136, 'plugged': 11137, 'rebuilding': 11138, "'80s": 11139, 'mario': 11140, 'biaggi': 11141, 'sentence': 11142, 'bribing': 11143, 'wallach': 11144, 'meese': 11145, 'fashioned': 11146, 'bribery': 11147, 'peddling': 11148, 'politically': 11149, 'respectable': 11150, 'confidant': 11151, 'lyn': 11152, 'nofzinger': 11153, 'corrupt': 11154, 'scheme': 11155, 'bag': 11156, 'crook': 11157, 'befell': 11158, 'semiliterate': 11159, 'sensational': 11160, 'revelations': 11161, 'breezy': 11162, 'easy-to-read': 11163, 'gripping': 11164, 'scams': 11165, 'ingenuity': 11166, 'auditors': 11167, 'crookery': 11168, 'garden-variety': 11169, 'lifes': 11170, 'mercedes': 11171, 'clothes': 11172, 'wrestling': 11173, 'intelligent': 11174, 'insane': 11175, 'irving': 11176, 'lobsenz': 11177, 'pediatrician': 11178, 'gambler': 11179, 'blackjack': 11180, 'arrested': 11181, 'doling': 11182, 'tidbits': 11183, 'gloss': 11184, 'auspices': 11185, 'rigid': 11186, 'affirmative': 11187, 'expressing': 11188, 'thieves': 11189, 'scandals': 11190, 'hud': 11191, 'characteristics': 11192, 'tailor-made': 11193, 'insider': 11194, 'whenever': 11195, 'redistributing': 11196, 'influencing': 11197, 'brokering': 11198, 'bloc': 11199, 'nomenklatura': 11200, 'stern': 11201, 'urban': 11202, '1983-85': 11203, 'bankrupt': 11204, 'andersson': 11205, 'finalized': 11206, 'state-appointed': 11207, 'receivers': 11208, 'lease': 11209, 'subcontractors': 11210, 'swift': 11211, 'avert': 11212, 'shipyards': 11213, 'injecting': 11214, 'undelivered': 11215, '170': 11216, 'helsinki': 11217, 'dashed': 11218, 'repay': 11219, 'concurrent': 11220, 'norfolk': 11221, 'sputtered': 11222, 'purina': 11223, '45.2': 11224, '84.9': 11225, '1.24': 11226, '422.5': 11227, '6.44': 11228, '387.8': 11229, '5.63': 11230, '70.2': 11231, 'seafood': 11232, '5.8': 11233, 'phase-out': 11234, 'hostess': 11235, 'bakery': 11236, 'rechargeable': 11237, 'cadmium': 11238, 'carbon': 11239, 'zinc': 11240, 'ingredients': 11241, 'cereal': 11242, 'baking': 11243, 'bread': 11244, 'eveready': 11245, '80.50': 11246, 'five-cent': 11247, 'percent': 11248, '300-day': 11249, '55.1': 11250, 'cash-and-stock': 11251, 'ravenswood': 11252, 'corn-buying': 11253, 'binge': 11254, 'bottlenecks': 11255, 'pipeline': 11256, 'trains': 11257, 'harvested': 11258, 'loading': 11259, 'reaping': 11260, 'windfall': 11261, 'gyrate': 11262, 'scrounge': 11263, 'strain': 11264, 'dunton': 11265, 'upper': 11266, 'tows': 11267, 'feeds': 11268, 'reservoirs': 11269, 'sank': 11270, 'alleviate': 11271, 'slowed': 11272, 'hamstrung': 11273, 'budding': 11274, 'logistical': 11275, '2.375': 11276, 'rebuild': 11277, 'depleted': 11278, 'winding': 11279, 'speculating': 11280, 'gather': 11281, 'permits': 11282, '2.15': 11283, 'lyle': 11284, 'waterloo': 11285, 'one-fifth': 11286, 'biedermann': 11287, 'allendale': 11288, 'port': 11289, 'lakes': 11290, 'coast': 11291, 'relieve': 11292, 'hauling': 11293, 'compressed': 11294, 'delayed': 11295, 'refinery': 11296, 'tightening': 11297, '58.64': 11298, '19.94': 11299, 'sell-off': 11300, '3.20': 11301, '377.60': 11302, '6.50': 11303, '5.2180': 11304, '5.70': 11305, '494.50': 11306, 'influenced': 11307, "o'neill": 11308, 'elders': 11309, 'equities': 11310, 'warehouses': 11311, '170,262': 11312, '226,570,380': 11313, 'miners': 11314, '1.20': 11315, '1.14': 11316, 'bronces': 11317, 'soldado': 11318, 'exxon-owned': 11319, 'minera': 11320, 'disputada': 11321, 'walkout': 11322, 'procedural': 11323, 'upbeat': 11324, 'mood': 11325, 'precedes': 11326, '51.6': 11327, 'readings': 11328, '47.1': 11329, 'ncr': 11330, 'midrange': 11331, 'networking': 11332, 'hub': 11333, 'novell': 11334, 'netware': 11335, 'riding': 11336, 'industrywide': 11337, 'chunks': 11338, 'marcus': 11339, 'appliance': 11340, 'lorain': 11341, '50-50': 11342, 'kobe': 11343, 'earning': 11344, 'bethlehem': 11345, '54': 11346, 'inland': 11347, 'plummeted': 11348, '34.625': 11349, 'exceeded': 11350, 'projections': 11351, 'bradford': 11352, 'richer': 11353, 'pipe': 11354, 'galvanized': 11355, 'coated': 11356, 'lower-priced': 11357, 'marathon': 11358, '198': 11359, 'soliciting': 11360, 'txo': 11361, '15.5': 11362, '257': 11363, '13.1': 11364, '721': 11365, '2.62': 11366, '598': 11367, '2.07': 11368, 'barrett': 11369, 'leon': 11370, 'mcfarlan': 11371, 'diloreto': 11372, 'container': 11373, 'delmont': 11374, 'constitutional-law': 11375, 'laurence': 11376, 'scuttle': 11377, 'implicitly': 11378, 'spectrum': 11379, 'authorizes': 11380, 'partial': 11381, 'shared': 11382, 'lawmaking': 11383, 'supports': 11384, 'reckless': 11385, 'railcar': 11386, 'platforms': 11387, 'trailer': 11388}
    11389
    

기존 단어 집합의 길이는 11,387였으나 'PAD'와 'OOV'를 추가하므로서 길이가 11,389가 된 것을 확인할 수 있습니다.

이제 word_to_index를 통해 단어를 입력하면, 인덱스를 리턴받을 수 있습니다.


```python
word_to_index['the']
```




    3



이제 태깅 정보에 인덱스를 부여하고, 입력받은 품사 태깅 정보에 대해서 인덱스를 리턴하는 tag_to_index를 만들어보겠습니다.


```python
tag_to_index={'PAD' : 0}
i=0
for tag in tag_set:
    i=i+1
    tag_to_index[tag]=i
print(tag_to_index)
```

    {'PAD': 0, 'VBZ': 1, 'VBN': 2, 'NNPS': 3, 'FW': 4, "''": 5, 'IN': 6, 'POS': 7, ',': 8, 'TO': 9, 'VBG': 10, 'VB': 11, 'PRP': 12, 'NNS': 13, 'LS': 14, 'DT': 15, 'NNP': 16, 'PRP$': 17, 'WP$': 18, 'NN': 19, 'JJ': 20, 'CC': 21, '``': 22, 'CD': 23, 'VBP': 24, '-LRB-': 25, '-RRB-': 26, 'UH': 27, 'MD': 28, 'PDT': 29, 'SYM': 30, '.': 31, 'WRB': 32, 'JJS': 33, 'VBD': 34, 'RB': 35, 'EX': 36, 'JJR': 37, 'WP': 38, '-NONE-': 39, 'RP': 40, '$': 41, '#': 42, ':': 43, 'RBR': 44, 'RBS': 45, 'WDT': 46}
    

총 46개의 단어를 가진 단어 집합에 대해서 인덱스가 부여었습니다. 패딩을 위해서 'PAD'라는 단어에는 인덱스 0을 부여하였습니다.


```python
len(tag_to_index)
```




    47



단어 'PAD'가 추가되면서 단어 집합의 크기는 46에서 47이 되었습니다.

이제 tag_to_index에다가 품사 태깅 정보를 입력하면 인덱스를 리턴받을 수 있습니다.


```python
tag_to_index['UH']
```




    27



이제 훈련 데이터에 대해서 정수 인코딩할 준비가 끝났습니다. 이제 모든 훈련 데이터를 담고 있는 sentences와 pos_tags로부터 word_to_index와 tag_to_index를 통해 모든 훈련 데이터를 숫자로 바꿀 것입니다.

우선 word_to_index를 사용하여 단어에 대한 훈련 데이터인 data_X를 만듭니다.


```python
data_X = []

for s in sentences:
    temp_X = []
    for w in s:
        try:
            temp_X.append(word_to_index.get(w.lower(),1))
        except KeyError: # 단어 집합을 만들 때 별도로 단어를 제거하지 않았기 때문에 이 과정에서는 OOV가 존재하지는 않음.
            temp_X.append(word_to_index['OOV'])

    data_X.append(temp_X)
print(data_X[:5])
```

    [[5602, 3747, 2, 2025, 87, 332, 2, 47, 2406, 3, 132, 28, 7, 2026, 333, 460, 2027, 4], [32, 3747, 21, 178, 5, 5603, 2916, 2, 3, 2917, 638, 148, 4], [2918, 5604, 2, 1137, 87, 332, 9, 603, 178, 5, 3748, 1047, 893, 894, 2, 35, 484, 10, 7, 2026, 333, 5, 52, 1048, 436, 2919, 4], [7, 639, 5, 1049, 640, 324, 12, 12, 6, 128, 1377, 2407, 1550, 39, 895, 7, 191, 1050, 5, 1248, 1760, 233, 7, 148, 5, 515, 3749, 12, 6, 23, 58, 62, 210, 87, 280, 2, 813, 284, 11, 16, 4], [3, 1049, 5605, 2, 2028, 2, 21, 3750, 5606, 640, 23, 2408, 3, 5607, 2, 29, 124, 2029, 2920, 6, 23, 2030, 5608, 15, 16, 814, 71, 2921, 815, 2, 813, 22, 11, 37, 4]]
    

변환이 되었는지 보기 위해서 첫번째 샘플에 대해서만 기존의 단어 시퀀스를 출력해보겠습니다.


```python
index_to_word={}
for key, value in word_to_index.items(): # 인덱스를 단어로 바꾸기 위해 index_to_word를 생성
    index_to_word[value] = key


temp = []
for index in data_X[0] : # 첫번째 문장 샘플 안의 인덱스들에 대해서
    temp.append(index_to_word[index]) # 다시 단어로 변환

print(sentences[0]) # 기존 문장 샘플 출력 
print(temp) # 기존 문장 샘플 → 정수 인코딩 → 복원
```

    ['Pierre' 'Vinken' ',' '61' 'years' 'old' ',' 'will' 'join' 'the' 'board'
     'as' 'a' 'nonexecutive' 'director' 'Nov.' '29' '.']
    ['pierre', 'vinken', ',', '61', 'years', 'old', ',', 'will', 'join', 'the', 'board', 'as', 'a', 'nonexecutive', 'director', 'nov.', '29', '.']
    

이제 y에 해당하는 전체 데이터를 담고 있는 pos_tags에서 품사 태깅 정보에 해당되는 부분을 모아 data_y에 저장하는 작업을 수행해보겠습니다.


```python
data_y = []

for s in pos_tags:
    temp_y = []
    for w in s:
            temp_y.append(tag_to_index.get(w))

    data_y.append(temp_y)

print(data_y[0])
```

    [16, 16, 8, 23, 13, 20, 8, 28, 11, 15, 19, 6, 15, 20, 19, 16, 23, 31]
    

이제 단어에 대한 훈련 데이터인 data_X와 품사 태깅 정보에 대한 훈련 데이터인 data_y가 만들어졌습니다. 

양방향 LSTM 모델에 손쉽게 데이터를 입력으로 사용하기 위해서, 여기서는 모든 샘플의 길이를 동일하게 맞추도록 하겠습니다. 이에 따라 가장 길이가 긴 샘플의 길이를 우선 구해보겠습니다.


```python
print(max(len(l) for l in data_X)) # 모든 데이터에서 길이가 가장 긴 샘플의 길이 출력
print(max(len(l) for l in data_y)) # 모든 데이터에서 길이가 가장 긴 샘플의 길이 출력
```

    271
    271
    

길이가 가장 긴 샘플의 길이는 271입니다. 그런데 앞서 본 그래프에 따르면, 대부분의 샘플은 길이가 50이하입니다. X에 해당되는 데이터 data_X의 샘플들과 y에 해당되는 데이터 data_y 샘플들의 모든 길이를 임의로 150정도로 맞추어 보겠습니다. 이를 위해서 케라스의 pad_sequences()를 사용합니다.


```python
max_len=150
from keras.preprocessing.sequence import pad_sequences
pad_X = pad_sequences(data_X, padding='post', maxlen=max_len)
# data_X의 모든 샘플의 길이를 맞출 때 뒤의 공간에 숫자 0으로 채움.
pad_y = pad_sequences(data_y, padding='post', value=tag_to_index['PAD'], maxlen=max_len)
# data_y의 모든 샘플의 길이를 맞출 때 뒤의 공간에 'PAD'에 해당되는 인덱스로 채움.
# 참고로 숫자 0으로 채우는 것과 'PAD'에 해당하는 인덱스로 채우는 것은 결국 0으로 채워지므로 같음
```

전체 훈련 데이터에서 길이가 가장 짧은 샘플의 길이를 출력했을 때 150이 나오는지 확인합니다.


```python
print(min(len(l) for l in pad_X)) # 모든 데이터에서 길이가 가장 짧은 샘플의 길이 출력
print(min(len(l) for l in pad_y)) # 모든 데이터에서 길이가 가장 짧은 샘플의 길이 출력
150
150
```

    150
    150
    




    150



pad_X와 pad_y를 훈련 데이터와 테스트 데이터로 8:2로 분할합니다.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(pad_X, pad_y, test_size=.2, random_state=777)
```

random_state를 지정했기 때문에 기존의 pad_X와 pad_y에서 순서가 섞이면서 훈련 데이터와 테스트 데이터로 분할됩니다.


```python
from keras.utils import np_utils
y_train2 = np_utils.to_categorical(y_train)
```

훈련 데이터 y_train에 대해서 원-핫 인코딩을 수행하고 y_train2에 저장합니다.


```python
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding
from keras.optimizers import Adam

n_words = len(word_to_index)
n_labels = len(tag_to_index)

model = Sequential()
model.add(Embedding(n_words, 128, input_length=max_len, mask_zero=True))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(n_labels, activation=('softmax'))))
model.compile(loss='categorical_crossentropy',optimizer=Adam(0.001),metrics=['accuracy'])
```


```python
model.fit(X_train, y_train2, batch_size=128, epochs=6)
```

    Epoch 1/6
    3131/3131 [==============================] - 13s 4ms/step - loss: 1.3432 - acc: 0.6628
    Epoch 2/6
    3131/3131 [==============================] - 13s 4ms/step - loss: 0.6576 - acc: 0.8595
    Epoch 3/6
    3131/3131 [==============================] - 14s 4ms/step - loss: 0.3299 - acc: 0.9274
    Epoch 4/6
    3131/3131 [==============================] - 14s 4ms/step - loss: 0.1983 - acc: 0.9571
    Epoch 5/6
    3131/3131 [==============================] - 14s 4ms/step - loss: 0.1356 - acc: 0.9710
    Epoch 6/6
    3131/3131 [==============================] - 14s 4ms/step - loss: 0.1010 - acc: 0.9774
    




    <keras.callbacks.History at 0x1c2aea41ba8>




```python
y_test2 = np_utils.to_categorical(y_test)
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test2)[1]))
```

    783/783 [==============================] - 7s 9ms/step
    
     테스트 정확도: 0.9299
    

우선 인덱스로부터 단어와 품사 태깅 정보를 리턴하는 index_to_word와 index_to_tag를 만들고 이를 이용하여 실제값과 예측값을 출력합니다.


```python
import numpy as np

index_to_word={}
for key, value in word_to_index.items():
    index_to_word[value] = key

index_to_tag={}
for key, value in tag_to_index.items():
    index_to_tag[value] = key


i=10 # 확인하고 싶은 테스트용 샘플의 인덱스.
y_predicted = model.predict(np.array([X_test[i]])) # 입력한 테스트용 샘플에 대해서 예측 y를 리턴
y_predicted = np.argmax(y_predicted, axis=-1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.
true = np.argmax(y_test2[i], -1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.

print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
print(35 * "-")

for w, t, pred in zip(X_test[i], true, y_predicted[0]):
    if w != 0: # PAD값은 제외함.
        print("{:17}: {:7} {}".format(index_to_word[w], index_to_tag[t], index_to_tag[pred]))
```

    단어             |실제값  |예측값
    -----------------------------------
    in               : IN      IN
    addition         : NN      NN
    ,                : ,       ,
    buick            : NNP     NNP
    is               : VBZ     VBZ
    a                : DT      DT
    relatively       : RB      RB
    respected        : VBN     VBN
    nameplate        : NN      NN
    among            : IN      IN
    american         : NNP     NNP
    express          : NNP     NNP
    card             : NN      NN
    holders          : NNS     NNS
    ,                : ,       ,
    says             : VBZ     VBZ
    0                : -NONE-  -NONE-
    *t*-1            : -NONE-  -NONE-
    an               : DT      DT
    american         : NNP     NNP
    express          : NNP     NNP
    spokeswoman      : NN      NN
    .                : .       .
    

## 5. 양방향 LSTM과 CRF(Bidirectional LSTM + CRF)

기존의 양방향 LSTM 모델에 CRF(Conditional Random Field)라는 새로운 층을 추가하여 보다 모델을 개선시킨 양방향 LSTM + CRF 모델을 사용하여 개체명 인식(Named Entity Recognition)을 수행합니다.

- 논문 링크 : https://arxiv.org/pdf/1508.01991v1.pdf
- 논문 링크 : https://arxiv.org/pdf/1603.01360.pdf

### 1) CRF(Conditional Random Field)

CRF는 Conditional Random Field의 약자로 양방향 LSTM을 위해 탄생한 모델이 아니라 이전에 독자적으로 존재해왔던 모델입니다.

양방향 LSTM 모델의 위에 CRF를 하나의 층으로 추가하여, 양방향 LSTM + CRF 모델이 탄생하였습니다. 

CRF 층의 역할을 이해하기 위해서 간단한 개체명 인식 작업의 예를 들어보겠습니다. 사람(Person), 조직(Organization) 두 가지만을 태깅하는 간단한 태깅 작업에 BIO 표현을 사용한다면 여기서 사용하는 태깅의 종류는 아래의 5가지입니다.

B-Per, I-Per, B-Org, I-Org, O

아래의 그림은 위의 태깅을 수행하는 기존의 양방향 LSTM 개체명 인식 모델의 예를 보여줍니다.

![](https://wikidocs.net/images/page/34156/bilstmcrf1.PNG)

위 모델은 각 단어를 벡터로 입력받고, 모델의 출력층에서 활성화 함수를 통해 개체명을 예측합니다. 

사실 입력 단어들과 실제 개체명이 무엇인지 모르는 상황이므로 이 모델이 정확하게 개체명을 예측했는지는 위 그림만으로는 알 수 없습니다.

![](https://wikidocs.net/images/page/34156/bilstmcrf2_%EC%88%98%EC%A0%95.PNG)

위 모델은 명확히 틀린 예측을 포함하고 있습니다. 

BIO 표현에 따르면 우선, 첫번째 단어에서 I가 갑자기 등장할 수 없습니다. 또한 I-Per은 반드시 B-Per 뒤에서만 등장할 수 있습니다. 뿐만 아니라, I-Org도 마찬가지로 B-Org 뒤에서만 등장할 수 있는데 위 모델은 이런 BIO 표현 방법의 제약사항들을 모두 위반하고 있습니다.

CRF 층을 추가하면 모델은 예측 개체명, 다시 말해 레이블 사이의 의존성을 고려할 수 있습니다. 아래의 그림은 양방향 LSTM + CRF 모델을 보여줍니다.

![](https://wikidocs.net/images/page/34156/bilstmcrf3.PNG)

기존에 CRF 층이 존재하지 않았던 양방향 LSTM 모델은 활성화 함수를 지난 시점에서 개체명을 결정했지만, CRF 층을 추가한 모델에서는 활성화 함수의 결과들이 CRF 층의 입력으로 전달됩니다.

예를 들어 word1에 대한 Bi-LSTM 셀과 활성화 함수를 지난 출력값 [0.7, 0.12, 0.08, 0.04, 0.06]은 CRF 층의 입력이 됩니다. CRF 층은 레이블 시퀀스에 대해서 가장 높은 점수를 가지는 시퀀스를 예측합니다.

CRF 층은 점차적으로 훈련 데이터로부터 아래와 같은 제약사항 등을 학습하게 됩니다.
1. 문장의 첫번째 단어에서는 I가 나오지 않습니다.
2. O-I 패턴은 나오지 않습니다.
3. B-I-I 패턴에서 개체명은 일관성을 유지합니다. 예를 들어 B-Per 다음에 I-Org는 나오지 않습니다.

CRF를 사용하기 위해 keras_contrib를 설치해야 합니다. 아래의 명령을 수행하여 설치합니다.


```python
# pip install git+https://www.github.com/keras-team/keras-contrib.git
```

### 2) 양방향 LSTM + CRF을 이용한 개체명 인식

 양방향 LSTM과 CRF를 함께 사용하여 앞서 사용한 데이터 외에 다른 데이터를 사용하여 개체명 인식을 수행해보도록 하겠습니다.

데이터 링크 : https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus


```python
import pandas as pd
import numpy as np

data = pd.read_csv("ner_dataset.csv", encoding="latin1")
```


```python
data[:5]
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
      <th>Sentence #</th>
      <th>Word</th>
      <th>POS</th>
      <th>Tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sentence: 1</td>
      <td>Thousands</td>
      <td>NNS</td>
      <td>O</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>of</td>
      <td>IN</td>
      <td>O</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>demonstrators</td>
      <td>NNS</td>
      <td>O</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>have</td>
      <td>VBP</td>
      <td>O</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>marched</td>
      <td>VBN</td>
      <td>O</td>
    </tr>
  </tbody>
</table>
</div>



첫번째 열은 다음과 같은 패턴을 가지고 있습니다. Sentence: 1 있고, Null 값이 이어지다가 다시 Sentence: 2가 나오고 다시 Null 값이 이어지다가 Sentence: 3이 나오고 다시 Null 값이 이어지다가를 반복합니다. 그런데 사실 이는 하나의 문장을 여러 행으로 나눠놓은 것입니다. 숫자값을 t라고 합시다. 

 첫번째 Sentence: t부터 Null 값이 나오다가 Sentence: t+1이 나오기 전까지의 모든 데이터는 원래 하나의 행. 즉, 하나의 샘플이어야 합니다. t번째 문장을 각 단어마다 각 하나의 행으로 나눠놓은 데이터이기 때문입니다. 이는 뒤에서 Pandas의 fillna를 통해 하나로 묶는 작업을 해줍니다.


```python
len(data)
```




    1048575



현재 data의 행의 개수는 1,048,575개입니다. 하지만 뒤에서 기존에 문장 1개였던 행들을 1개의 행으로 병합하는 작업을 해야하기 때문에 최종 샘플의 개수는 이보다 줄어들게 됩니다. 


```python
data.isnull().values.any()
```




    True



Sentence #열에 Null 값들이 존재


```python
data.isnull().sum()
```




    Sentence #    1000616
    Word                0
    POS                 0
    Tag                 0
    dtype: int64



isnull().sum()을 수행하면 각 열마다의 Null 값의 개수를 보여줍니다. 다른 열은 0개인데 오직 Sentences #열에서만 1,000,616개가 나온 것을 볼 수 있습니다.

전체 데이터에서 중복을 허용하지 않고, 유일한 값의 개수를 셀 수 있게 해주는 nunique()를 사용해봅시다.


```python
data['Sentence #'].nunique(), data.Word.nunique(), data.Tag.nunique()
```




    (47959, 35178, 17)



이 데이터에는 47,959개의 문장이 있으며 문장들은 35,178개의 단어를 가지고 17개 종류의 개체명 태깅을 가집니다.

17개의 개체명 태깅이 전체 데이터에서 몇 개가 있는지, 개체명 태깅 개수의 분포를 확인해보도록 하겠습니다.


```python
print(data.groupby('Tag').size().reset_index(name='count'))
```

          Tag   count
    0   B-art     402
    1   B-eve     308
    2   B-geo   37644
    3   B-gpe   15870
    4   B-nat     201
    5   B-org   20143
    6   B-per   16990
    7   B-tim   20333
    8   I-art     297
    9   I-eve     253
    10  I-geo    7414
    11  I-gpe     198
    12  I-nat      51
    13  I-org   16784
    14  I-per   17251
    15  I-tim    6528
    16      O  887908
    

BIO 표현 방법에서 아무런 태깅도 의미하지 않는 O가 가장 887,908개로 가장 많은 개수를 차지함을 볼 수 있습니다. 

이제 데이터를 원하는 형태로 가공해보겠습니다. 우선 Null 값을 제거합니다.


```python
data = data.fillna(method="ffill")
```

Pandas의 (method='ffill')는 Null 값을 가진 행의 바로 앞의 행의 값으로 Null 값을 채우는 작업을 수행합니다. 이렇게 하면 t번째 문장에 속하면서 Null 값을 가진 샘플들은 전부 첫번째 열에 Sentence: t의 값이 들어갑니다. 

이번에는 뒤의 5개의 샘플을 출력해서 정상적으로 수행되었는지 확인해봅시다.


```python
print(data.tail())
```

                  Sentence #       Word  POS Tag
    1048570  Sentence: 47959       they  PRP   O
    1048571  Sentence: 47959  responded  VBD   O
    1048572  Sentence: 47959         to   TO   O
    1048573  Sentence: 47959        the   DT   O
    1048574  Sentence: 47959     attack   NN   O
    

전체 데이터에 Null 값이 여전히 존재하는지 확인해봅시다.


```python
data.isnull().values.any()
```




    False



모든 단어를 소문자화하여 단어의 개수를 줄여보겠습니다.


```python
data['Word'] = data['Word'].str.lower()
# 단어의 소문자화. 이 데이터의 경우 이를 통해 약 3,000개의 중복 단어 제거 가능.
```


```python
print(data[:5])
```

        Sentence #           Word  POS Tag
    0  Sentence: 1      thousands  NNS   O
    1  Sentence: 1             of   IN   O
    2  Sentence: 1  demonstrators  NNS   O
    3  Sentence: 1           have  VBP   O
    4  Sentence: 1        marched  VBN   O
    

이제 중복을 허용하지 않고, 단어들을 모아 단어 집합을 만들도록 하겠습니다.


```python
vocab=(list(set(data["Word"].values))) # 중복을 허용하지 않고 집합으로 만듬.
print(len(vocab)) # 단어 집합의 크기
print(vocab[:5]) # 단어 집합 출력
```

    31817
    ['songwriter', 'zach', 'sprinters', 'december', "o'connor"]
    

단어의 수가 35,178에서 31,817로 줄어든 것을 볼 수 있습니다. 단어들을 소문자화하여 대문자, 소문자로 표기만 다른 중복 단어들이 하나의 단어들로 정규화되었기 때문입니다.

태깅 정보에 대해서도 중복을 허용하지 않고 태깅 정보 집합을 만듭니다.


```python
tags = list(set(data["Tag"].values)) # 중복을 허용하지 않고 집합으로 만듬.
print(len(tags)) # 태깅 정보 집합의 크기
print(tags) # 태깅 정보 출력
```

    17
    ['B-gpe', 'B-org', 'I-geo', 'I-per', 'B-nat', 'I-eve', 'B-per', 'B-geo', 'I-art', 'B-eve', 'O', 'I-org', 'B-tim', 'I-nat', 'I-gpe', 'B-art', 'I-tim']
    

geo는 Geographical Entity<br/>
org는 Organization<br/>
per는 Person<br/>
gpe는 Geopolitical Entity<br/>
tim은 Time indicator<br/>
art는 Artifact<br/>
eve는 Event<br/>
nat는 Natural Phenomenon

 품사 태깅 정보는 고려하지 않고 개체명 인식을 진행하므로, 품사 태깅 정보에 해당되는 열을 삭제해보도록 하겠습니다.


```python
del data['POS']
print(data[:5])
```

        Sentence #           Word Tag
    0  Sentence: 1      thousands   O
    1  Sentence: 1             of   O
    2  Sentence: 1  demonstrators   O
    3  Sentence: 1           have   O
    4  Sentence: 1        marched   O
    

이제 하나의 문장에 등장한 단어와 개체명 태깅 정보끼리 쌍(pair)으로 묶는 작업을 수행합니다.


```python
func = lambda temp: [(w, t) for w, t in zip(temp["Word"].values.tolist(), temp["Tag"].values.tolist())]
All_data=[t for t in data.groupby("Sentence #").apply(func)]
print(len(All_data))
```

    47959
    

1,000,616개의 행의 개수가 각 문장당 하나의 데이터로 묶이면서 47,959개의 샘플로 변환된 것을 확인할 수 있습니다. 


```python
print(All_data[0])
print(len(All_data[0]))
```

    [('thousands', 'O'), ('of', 'O'), ('demonstrators', 'O'), ('have', 'O'), ('marched', 'O'), ('through', 'O'), ('london', 'B-geo'), ('to', 'O'), ('protest', 'O'), ('the', 'O'), ('war', 'O'), ('in', 'O'), ('iraq', 'B-geo'), ('and', 'O'), ('demand', 'O'), ('the', 'O'), ('withdrawal', 'O'), ('of', 'O'), ('british', 'B-gpe'), ('troops', 'O'), ('from', 'O'), ('that', 'O'), ('country', 'O'), ('.', 'O')]
    24
    

하나의 문장에 등장한 단어와 개체명 태깅 정보끼리 쌍으로 묶인 것을 볼 수 있습니다. 

전체 데이터 샘플의 길이 분포를 확인해봅시다.


```python
%matplotlib inline
import matplotlib.pyplot as plt

print('샘플의 최대 길이 :',max(len(l) for l in All_data))
print('샘플의 평균 길이 :',sum(map(len, All_data))/len(All_data))

plt.hist([len(s) for s in All_data], bins=50)
plt.xlabel('length of Data')
plt.ylabel('number of Data')
plt.show()
```

    샘플의 최대 길이 : 104
    샘플의 평균 길이 : 21.863987989741236
    


![png](NLP_basic_11_Tagging_Task_files/NLP_basic_11_Tagging_Task_250_1.png)


이제 단어와 개체명 태깅 정보를 분리하는 작업을 진행합니다. 즉, 레이블을 분리하는 작업입니다.


```python
import numpy as np

sentences, ner_tags =[], []
for tagged_sentence in All_data:
    sentence, ner_info = zip(*tagged_sentence) # 각 샘플에서 단어는 sentence에 개체명 태깅정보는 ner_info에
    sentences.append(np.array(sentence)) # 각 샘플에서 단어 정보만 저장한다.
    ner_tags.append(np.array(ner_info)) # 각 샘플에서 개체명 태깅 정보만 저장한다.
```


```python
print(sentences[0])
```

    ['thousands' 'of' 'demonstrators' 'have' 'marched' 'through' 'london' 'to'
     'protest' 'the' 'war' 'in' 'iraq' 'and' 'demand' 'the' 'withdrawal' 'of'
     'british' 'troops' 'from' 'that' 'country' '.']
    


```python
print(ner_tags[0])
```

    ['O' 'O' 'O' 'O' 'O' 'O' 'B-geo' 'O' 'O' 'O' 'O' 'O' 'B-geo' 'O' 'O' 'O'
     'O' 'O' 'B-gpe' 'O' 'O' 'O' 'O' 'O']
    

단어 집합의 경우에는 Counter()를 이용하여 단어의 등장 빈도수를 계산하는 작업도 함께 진행합니다.


```python
from collections import Counter
vocab=Counter()
tag_set=set()

for sentence in sentences: # 훈련 데이터 X에서 샘플을 1개씩 꺼내온다.
    for word in sentence: # 샘플에서 단어를 1개씩 꺼내온다.
        vocab[word.lower()]=vocab[word.lower()]+1 # 각 단어의 빈도수를 카운트한다.

for tags_list in ner_tags: # 훈련 데이터 y에서 샘플을 1개씩 꺼내온다.
    for tag in tags_list: # 샘플에서 개체명 정보를 1개씩 꺼내온다.
        tag_set.add(tag) # 각 개체명 정보에 대해서 중복을 허용하지 않고 집합을 만든다.
```


```python
vocab
```




    Counter({'thousands': 495,
             'of': 26378,
             'demonstrators': 132,
             'have': 5486,
             'marched': 65,
             'through': 518,
             'london': 261,
             'to': 23249,
             'protest': 238,
             'the': 63905,
             'war': 903,
             'in': 28050,
             'iraq': 1738,
             'and': 20116,
             'demand': 221,
             'withdrawal': 154,
             'british': 637,
             'troops': 1202,
             'from': 4557,
             'that': 6437,
             'country': 1934,
             '.': 47761,
             'iranian': 380,
             'officials': 3390,
             'say': 4178,
             'they': 2397,
             'expect': 66,
             'get': 170,
             'access': 127,
             'sealed': 18,
             'sensitive': 30,
             'parts': 168,
             'plant': 109,
             'wednesday': 1258,
             ',': 32754,
             'after': 2737,
             'an': 4236,
             'iaea': 73,
             'surveillance': 51,
             'system': 267,
             'begins': 51,
             'functioning': 11,
             'helicopter': 110,
             'gunships': 18,
             'saturday': 1152,
             'pounded': 15,
             'militant': 457,
             'hideouts': 16,
             'orakzai': 19,
             'tribal': 170,
             'region': 858,
             'where': 658,
             'many': 600,
             'taliban': 263,
             'militants': 1065,
             'are': 3721,
             'believed': 175,
             'fled': 154,
             'avoid': 82,
             'earlier': 823,
             'military': 2004,
             'offensive': 154,
             'nearby': 107,
             'south': 1012,
             'waziristan': 82,
             'left': 327,
             'a': 22696,
             'tense': 17,
             'hour-long': 2,
             'standoff': 31,
             'with': 5448,
             'riot': 46,
             'police': 1866,
             'u.n.': 849,
             'relief': 201,
             'coordinator': 13,
             'jan': 22,
             'egeland': 14,
             'said': 5329,
             'sunday': 1215,
             'u.s.': 4129,
             'indonesian': 114,
             'australian': 119,
             'helicopters': 69,
             'ferrying': 2,
             'out': 1070,
             'food': 358,
             'supplies': 163,
             'remote': 88,
             'areas': 309,
             'western': 520,
             'aceh': 55,
             'province': 947,
             'ground': 81,
             'crews': 18,
             'can': 509,
             'not': 2709,
             'reach': 124,
             'mr.': 3086,
             'latest': 232,
             'figures': 72,
             'show': 221,
             '1.8': 7,
             'million': 880,
             'people': 2756,
             'need': 147,
             'assistance': 141,
             '-': 1172,
             'greatest': 19,
             'indonesia': 219,
             'sri': 144,
             'lanka': 93,
             'maldives': 6,
             'india': 517,
             'he': 4370,
             'last': 1977,
             'week': 1321,
             "'s": 10923,
             'tsunami': 140,
             'massive': 144,
             'underwater': 7,
             'earthquake': 152,
             'triggered': 73,
             'it': 3823,
             'has': 7216,
             'affected': 88,
             'millions': 144,
             'asia': 219,
             'africa': 352,
             'some': 1109,
             '1,27,000': 3,
             'known': 314,
             'dead': 360,
             'aid': 485,
             'is': 6749,
             'being': 573,
             'rushed': 23,
             'but': 2520,
             'official': 749,
             'stressed': 50,
             'bottlenecks': 2,
             'lack': 107,
             'infrastructure': 101,
             'remain': 236,
             'challenge': 54,
             'lebanese': 214,
             'politicians': 79,
             'condemning': 18,
             'friday': 1279,
             'bomb': 695,
             'blast': 338,
             'christian': 94,
             'neighborhood': 60,
             'beirut': 96,
             'as': 4224,
             'attempt': 150,
             'sow': 2,
             'sectarian': 58,
             'strife': 11,
             'formerly': 26,
             'war-torn': 44,
             'string': 31,
             'voiced': 22,
             'their': 1798,
             'anger': 24,
             'while': 627,
             'at': 4692,
             'united': 2050,
             'nations': 1024,
             'summit': 245,
             'new': 2151,
             'york': 269,
             'prime': 1130,
             'minister': 1867,
             'fouad': 3,
             'siniora': 9,
             'resolute': 1,
             'preventing': 30,
             'such': 423,
             'attempts': 63,
             'destroying': 30,
             'spirit': 12,
             'one': 1692,
             'person': 136,
             'was': 4881,
             'killed': 2861,
             'more': 2331,
             'than': 1890,
             '20': 327,
             'others': 627,
             'injured': 239,
             'late': 563,
             'which': 1633,
             'took': 438,
             'place': 376,
             'on': 7113,
             'residential': 15,
             'street': 83,
             'lebanon': 301,
             'suffered': 102,
             'series': 264,
             'bombings': 243,
             'since': 1358,
             'explosion': 227,
             'february': 261,
             'former': 1020,
             'rafik': 52,
             'hariri': 96,
             'other': 1527,
             'syria': 289,
             'widely': 60,
             'accused': 490,
             'involvement': 116,
             'his': 3469,
             'killing': 585,
             'comes': 234,
             'days': 529,
             'before': 718,
             'investigator': 16,
             'detlev': 11,
             'mehlis': 19,
             'return': 262,
             'damascus': 68,
             'interview': 154,
             'several': 915,
             'syrian': 176,
             'about': 1567,
             'assassination': 130,
             'global': 321,
             'financial': 301,
             'crisis': 256,
             'iceland': 10,
             'economy': 586,
             'shambles': 1,
             'israeli': 990,
             'ariel': 75,
             'sharon': 159,
             'will': 3404,
             'undergo': 12,
             'medical': 172,
             'procedure': 23,
             'thursday': 1333,
             'close': 238,
             'tiny': 18,
             'hole': 10,
             'heart': 70,
             'discovered': 116,
             'during': 1232,
             'treatment': 105,
             'for': 8556,
             'minor': 32,
             'stroke': 35,
             'month': 1048,
             'doctors': 102,
             'describe': 14,
             'birth': 32,
             'defect': 4,
             'partition': 4,
             'between': 1050,
             'upper': 39,
             'chambers': 13,
             'cardiac': 1,
             'catheterization': 1,
             'involves': 24,
             'inserting': 1,
             'catheter': 1,
             'blood': 57,
             'vessel': 53,
             'into': 1159,
             'umbrella-like': 1,
             'device': 40,
             'plug': 1,
             'make': 331,
             'full': 134,
             'recovery': 81,
             'returned': 138,
             'work': 361,
             'december': 310,
             '25': 204,
             'emergency': 213,
             'hospitalization': 4,
             'caused': 217,
             'any': 511,
             'permanent': 80,
             'damage': 120,
             'designers': 3,
             'first': 1112,
             'private': 149,
             'manned': 7,
             'rocket': 136,
             'burst': 13,
             'space': 179,
             'received': 164,
             '$': 1138,
             '10': 432,
             'prize': 55,
             'created': 75,
             'promote': 59,
             'tourism': 119,
             'spaceshipone': 3,
             'designer': 3,
             'burt': 1,
             'rutan': 1,
             'accepted': 43,
             'ansari': 1,
             'x': 1,
             'money': 216,
             'trophy': 2,
             'behalf': 21,
             'team': 247,
             'awards': 21,
             'ceremony': 113,
             'state': 1285,
             'missouri': 9,
             'win': 157,
             'had': 1518,
             'off': 497,
             'twice': 43,
             'two-week': 16,
             'period': 102,
             'fly': 24,
             'least': 1482,
             '100': 197,
             'kilometers': 282,
             'above': 52,
             'earth': 62,
             'spacecraft': 17,
             'made': 658,
             'its': 2672,
             'flights': 80,
             'september': 315,
             'early': 483,
             'october': 246,
             'lifting': 20,
             'california': 92,
             'mojave': 1,
             'desert': 15,
             'three': 1240,
             'major': 426,
             'banks': 73,
             'collapsed': 51,
             'unemployment': 109,
             'soared': 24,
             'value': 55,
             'krona': 1,
             'plunged': 36,
             'vehicle': 191,
             'carry': 83,
             'pilot': 27,
             'weight': 18,
             'equivalent': 5,
             'two': 2804,
             'passengers': 72,
             'financed': 7,
             'paul': 129,
             'allen': 10,
             'co-founder': 4,
             'microsoft': 26,
             'corporation': 45,
             'north': 895,
             'korea': 514,
             'says': 4640,
             'flooding': 65,
             'by': 4506,
             'typhoon': 46,
             'wipha': 1,
             'destroyed': 131,
             '14,000': 17,
             'homes': 187,
             '1,09,000': 1,
             'hectares': 15,
             'crops': 34,
             'news': 761,
             'agency': 808,
             'kcna': 8,
             'reported': 486,
             'monday': 1266,
             'floods': 65,
             'also': 2314,
             'or': 933,
             'damaged': 75,
             '8,000': 29,
             'public': 372,
             'buildings': 92,
             'washed': 16,
             'roads': 55,
             'bridges': 11,
             'railways': 4,
             'report': 806,
             'did': 412,
             'mention': 16,
             'deaths': 221,
             'injuries': 103,
             'most': 635,
             'heavy': 189,
             'rains': 86,
             'occurred': 198,
             'southwestern': 76,
             'part': 638,
             'including': 812,
             'capital': 728,
             'pyongyang': 129,
             'severe': 69,
             '600': 42,
             'missing': 173,
             'displaced': 95,
             '1,00,000': 36,
             'strong': 187,
             'under': 751,
             'ocean': 111,
             'sumatra': 14,
             'nias': 5,
             'islands': 197,
             'panic': 12,
             'no': 984,
             'geological': 18,
             'survey': 73,
             'gave': 118,
             'preliminary': 41,
             'estimate': 29,
             'strength': 39,
             'tuesday': 1394,
             'morning': 92,
             'quake': 92,
             '6.7': 5,
             'richter': 3,
             'scale': 29,
             'epicenter': 9,
             'island': 301,
             'geir': 2,
             'haarde': 1,
             'refused': 148,
             'resign': 43,
             'call': 191,
             'elections': 768,
             'cause': 113,
             'march': 335,
             '900': 15,
             'both': 443,
             'experienced': 38,
             'countless': 2,
             'earthquakes': 18,
             'tsunami-producing': 1,
             '26': 113,
             'death': 503,
             'toll': 151,
             'tragedy': 23,
             'stands': 36,
             '1,76,000': 1,
             '1,28,000': 1,
             'them': 618,
             'nearly': 502,
             '50,000': 21,
             'still': 382,
             'listed': 19,
             'feared': 31,
             'rap': 5,
             'star': 50,
             'snoop': 6,
             'dogg': 6,
             'five': 608,
             'associates': 13,
             'been': 2892,
             'arrested': 465,
             'britain': 335,
             'disturbance': 4,
             'heathrow': 6,
             'airport': 113,
             'told': 685,
             'media': 397,
             'musician': 15,
             'who': 1981,
             'born': 37,
             'name': 121,
             'calvin': 2,
             'broadus': 2,
             'members': 653,
             'entourage': 3,
             'were': 3520,
             'held': 563,
             'charges': 445,
             '"': 3686,
             'violent': 130,
             'disorder': 9,
             'affray': 1,
             'group': 1315,
             'waiting': 29,
             'flight': 53,
             'perform': 10,
             'concert': 23,
             'when': 1230,
             'denied': 221,
             'first-class': 1,
             'lounge': 3,
             'later': 514,
             'threw': 49,
             'bottles': 10,
             'whisky': 1,
             'duty-free': 14,
             'store': 26,
             'scuffled': 6,
             'member': 255,
             'gang': 38,
             'crips': 2,
             'southern': 861,
             'songs': 11,
             'reflect': 25,
             'gritty': 1,
             'life': 160,
             'streets': 88,
             'blames': 32,
             'economic': 736,
             'calamity': 2,
             'commercial': 60,
             'bankers': 6,
             'afghan': 814,
             'president': 3396,
             'hamid': 114,
             'karzai': 150,
             'fired': 263,
             'high-ranking': 20,
             'spying': 43,
             'countries': 814,
             'warns': 27,
             'would': 1152,
             'spare': 14,
             'anyone': 55,
             'engages': 2,
             'activity': 130,
             'disclosure': 3,
             'lunch': 12,
             'meeting': 594,
             'newly': 40,
             'sworn-in': 5,
             'parliament': 505,
             'dismissed': 85,
             'nor': 37,
             'indicate': 61,
             'action': 206,
             'involved': 188,
             'evidence': 136,
             'against': 1257,
             'even': 128,
             'if': 718,
             'punished': 14,
             'found': 474,
             'foreign': 1119,
             'be': 2530,
             'shown': 47,
             'television': 286,
             'put': 183,
             'trial': 241,
             'led': 294,
             'afghanistan': 1041,
             'taleban': 549,
             'ousted': 145,
             '2001': 233,
             'won': 248,
             'presidential': 393,
             'election': 654,
             '2004': 241,
             'now': 387,
             'waging': 14,
             'insurgency': 122,
             'administration': 293,
             'four': 779,
             'weeks': 337,
             'un': 60,
             'secretary-general': 138,
             'kofi': 101,
             'annan': 164,
             'trying': 282,
             'broker': 10,
             'deal': 365,
             'kenyan': 64,
             'government': 3112,
             'mwai': 17,
             'kibaki': 36,
             'opposition': 577,
             'raila': 4,
             'odinga': 6,
             'negotiations': 172,
             'concentrated': 10,
             'power': 518,
             'sharing': 19,
             'agreement': 429,
             'transitional': 37,
             'arrangement': 12,
             'leading': 188,
             'forced': 175,
             'ask': 44,
             'international': 1086,
             'monetary': 55,
             'fund': 120,
             'multi-billion-dollar': 1,
             'loan': 28,
             'recent': 563,
             'complimentary': 2,
             'sets': 46,
             'issues': 208,
             'must': 285,
             'addressed': 29,
             'finalize': 8,
             'detailed': 17,
             'francois': 10,
             'grignon': 1,
             'director': 123,
             'program': 605,
             'icg': 1,
             'telephone': 40,
             'discuss': 268,
             'stake': 19,
             'voa': 237,
             'reporter': 44,
             'akwei': 1,
             'thompson': 1,
             'demonstrate': 13,
             'stronger': 45,
             'political': 734,
             'tackle': 11,
             'task': 22,
             'legal': 136,
             'constitutional': 126,
             'reform': 140,
             'needed': 134,
             'transition': 35,
             'because': 620,
             '\x85': 4,
             '\x94': 4,
             'electoral': 104,
             'dispute': 153,
             'losers': 1,
             'bush': 977,
             'signed': 296,
             'legislation': 100,
             'require': 36,
             'screening': 9,
             'all': 759,
             'air': 269,
             'sea': 144,
             'cargo': 43,
             'provide': 173,
             'cities': 149,
             'deemed': 18,
             'high': 412,
             'risk': 75,
             'terrorist': 356,
             'attack': 994,
             'signing': 46,
             'bill': 207,
             'advisers': 5,
             'counter-terrorism': 26,
             'homeland': 57,
             'security': 1620,
             'teams': 58,
             'doing': 73,
             'everything': 22,
             'protect': 103,
             'what': 569,
             'called': 951,
             'dangerous': 68,
             'enemy': 39,
             'measures': 163,
             'recommendations': 20,
             'independent': 184,
             'commission': 256,
             'investigated': 31,
             '11': 266,
             'attacks': 1076,
             'states': 1518,
             'those': 572,
             'include': 269,
             'grant': 36,
             '4': 68,
             'billion': 371,
             'given': 129,
             'high-risk': 2,
             'upgrade': 11,
             'transit': 33,
             'mandate': 31,
             'u.s.-bound': 6,
             'planes': 61,
             'ships': 69,
             'within': 207,
             'next': 625,
             'years': 1014,
             'burmese': 101,
             'democracy': 181,
             'advocate': 9,
             'aung': 75,
             'san': 121,
             'suu': 64,
             'kyi': 68,
             'calling': 216,
             'citizens': 155,
             'her': 631,
             'toward': 157,
             'national': 751,
             'reconciliation': 47,
             'year': 1548,
             'statement': 717,
             'she': 558,
             'asked': 194,
             'struggle': 58,
             'together': 108,
             'strengths': 1,
             'force': 430,
             'words': 61,
             '2011': 76,
             'health': 590,
             'experts': 178,
             'cancer': 58,
             'become': 171,
             'world': 1340,
             '2010': 218,
             'overtaking': 2,
             'disease': 200,
             '65-year-old': 3,
             'democratic': 363,
             'reforms': 153,
             'burma': 317,
             'released': 529,
             'seven': 333,
             'house': 585,
             'arrest': 204,
             'november': 266,
             '13': 164,
             'just': 311,
             'rulers': 17,
             'claimed': 282,
             'overwhelming': 22,
             'victory': 110,
             'criticized': 191,
             'decades': 143,
             'establish': 45,
             'social': 121,
             'networks': 23,
             'achieve': 21,
             'well': 307,
             'truly': 4,
             'again': 156,
             'leaders': 710,
             'free': 240,
             '2,200': 9,
             'prisoners': 238,
             'engage': 20,
             'talks': 1006,
             'clinton': 119,
             'assembled': 7,
             'activists': 155,
             'academics': 4,
             'address': 163,
             'poverty': 105,
             'warning': 142,
             'conflict': 245,
             'opening': 97,
             'initiative': 51,
             'conference': 267,
             'coincides': 7,
             'millennium': 22,
             'general': 558,
             'assembly': 141,
             'focus': 104,
             'three-day': 50,
             'secure': 59,
             'concrete': 15,
             'pledges': 22,
             'significant': 96,
             'problems': 178,
             'simply': 18,
             'talk': 32,
             'organizers': 40,
             'different': 51,
             'forums': 1,
             'participants': 20,
             'required': 42,
             'pledge': 45,
             'ways': 79,
             'back': 278,
             'progress': 134,
             'expected': 598,
             'attendees': 4,
             'speakers': 11,
             'tony': 92,
             'blair': 129,
             'secretary': 406,
             'israel': 955,
             'deputy': 179,
             'shimon': 14,
             'peres': 29,
             'peruvian': 26,
             'narrow': 22,
             'gap': 30,
             'candidates': 149,
             'vying': 3,
             'second': 447,
             'spot': 21,
             'ballot': 20,
             'run-off': 35,
             'tightened': 11,
             'further': 201,
             'organization': 461,
             'issued': 296,
             'factor': 17,
             'behind': 121,
             'growing': 144,
             'deadliness': 1,
             'rising': 114,
             'cigarette': 5,
             'smoking': 19,
             'developing': 112,
             'center-left': 8,
             'alan': 19,
             'garcia': 10,
             'leads': 26,
             'pro-business': 5,
             'congresswoman': 3,
             'lourdes': 1,
             'flores': 12,
             'less': 129,
             '96,000': 3,
             'votes': 87,
             '90': 87,
             'percent': 668,
             'counted': 27,
             '1,10,000': 3,
             'surge': 35,
             'attributed': 22,
             'support': 470,
             'among': 436,
             'peruvians': 2,
             'living': 126,
             'abroad': 75,
             'whose': 99,
             'apparently': 76,
             'starting': 45,
             'impact': 59,
             'tally': 14,
             'candidate': 175,
             'half': 180,
             'vote': 492,
             'april': 191,
             '9': 82,
             '30': 288,
             'final': 240,
             'results': 215,
             'announced': 495,
             'either': 61,
             'take': 424,
             'nationalist': 20,
             'ollanta': 1,
             'humala': 1,
             '31': 77,
             'chilean': 36,
             'authorities': 1188,
             'freed': 127,
             'bail': 40,
             'wife': 78,
             'adult': 12,
             'children': 300,
             'dictator': 69,
             'augusto': 12,
             'pinochet': 32,
             'day': 686,
             'detained': 328,
             'tax': 141,
             'evasion': 24,
             'lucia': 7,
             'hiriart': 4,
             'investigation': 255,
             'dollars': 157,
             'kept': 43,
             'bank': 475,
             'accounts': 79,
             'fifth': 62,
             'child': 77,
             'daughter': 46,
             'charged': 184,
             'whereabouts': 10,
             'unknown': 55,
             'located': 81,
             'prohibited': 7,
             'leaving': 114,
             'indicted': 64,
             'fraud': 132,
             'allegedly': 118,
             'hiding': 45,
             '27': 81,
             '40': 182,
             'smokers': 6,
             'thought': 57,
             'live': 87,
             'china': 1019,
             'alone': 29,
             'faces': 114,
             'human': 573,
             'rights': 603,
             'related': 74,
             'rule': 163,
             'mid-1970s': 3,
             'lawyers': 81,
             'healthy': 20,
             'enough': 130,
             'stand': 57,
             'court-ordered': 10,
             'fit': 13,
             'do': 278,
             'so': 308,
             'rebel': 299,
             'sources': 116,
             'mexico': 204,
             'female': 54,
             'leader': 897,
             'zapatista': 9,
             'movement': 190,
             'died': 595,
             'subcomandante': 6,
             'marcos': 8,
             'comandante': 2,
             'ramona': 2,
             'saying': 785,
             'lost': 143,
             'fighter': 51,
             'zapatistas': 2,
             'piece': 22,
             'announcement': 146,
             'came': 392,
             'stop': 294,
             'chiapas': 4,
             'six': 477,
             'nationwide': 66,
             'tour': 105,
             'nature': 27,
             'immediately': 154,
             'clear': 211,
             'rumored': 2,
             'kidney': 15,
             'once': 108,
             'transplant': 5,
             'mysterious': 4,
             'tzotzil': 1,
             'indian': 334,
             'promoter': 3,
             'women': 285,
             'longtime': 25,
             'appeared': 110,
             'wearing': 31,
             'black': 82,
             'ski': 14,
             'mask': 1,
             'january': 389,
             '1': 138,
             'emerged': 24,
             'jungle': 10,
             'hideout': 30,
             'begin': 183,
             'six-month': 12,
             'bid': 60,
             'influence': 50,
             'this': 1614,
             'estimated': 127,
             '12': 230,
             'diagnosed': 20,
             'form': 160,
             'representatives': 125,
             'washington-based': 9,
             'council': 473,
             'american-islamic': 2,
             'relations': 240,
             'appealed': 46,
             'release': 305,
             'kidnapped': 211,
             'american': 667,
             'journalist': 118,
             'jill': 17,
             'carroll': 28,
             'baghdad': 763,
             'influential': 32,
             'islamic': 405,
             'abductors': 16,
             'unharmed': 37,
             '28-year-old': 8,
             'well-documented': 2,
             'record': 207,
             'objective': 9,
             'reporting': 56,
             'respect': 40,
             'iraqi': 1144,
             'arab-islamic': 1,
             'culture': 27,
             'there': 823,
             'word': 50,
             'fate': 44,
             'following': 476,
             'threat': 168,
             'kidnappers': 84,
             'execute': 4,
             'unless': 84,
             'eight': 290,
             'detainees': 172,
             ...})




```python
print(tag_set)
```

    {'B-gpe', 'B-org', 'I-geo', 'I-per', 'B-nat', 'I-eve', 'B-per', 'B-geo', 'I-art', 'B-eve', 'O', 'I-org', 'B-tim', 'I-nat', 'I-gpe', 'B-art', 'I-tim'}
    


```python
print(len(vocab)) # X 데이터의 단어 집합의 크기 출력
print(len(tag_set)) # y 데이터의 집합의 크기 출력 (개체명 태깅 정보의 종류 출력)
```

    31817
    17
    

단어 집합의 크기는 31,817입니다. 개체명 태깅 정보는 총 17개의 종류가 존재합니다.

이제 단어 집합을 등장 빈도수를 기준으로 정렬합니다.


```python
vocab_sorted=sorted(vocab.items(), key=lambda x:x[1], reverse=True)
print(vocab_sorted[:5])
```

    [('the', 63905), ('.', 47761), (',', 32754), ('in', 28050), ('of', 26378)]
    

'the'가 63,905번의 빈도수로 가장 많이 등장하였고, 불용어에 속하는 각종 전치사들 또한 등장 빈도수가 굉장히 높게 나옵니다.

이제 단어 집합의 각 단어에 인덱스를 부여해보도록 하겠습니다.


```python
word_to_index={'PAD' : 0, 'OOV' :1}
i=1
# 인덱스 0은 각각 입력값들의 길이를 맞추기 위한 PAD(padding을 의미)라는 단어에 사용된다.
# 인덱스 1은 모르는 단어를 의미하는 OOV라는 단어에 사용된다.
for (word, frequency) in vocab_sorted :
    # if frequency > 1 :
    # 빈도수가 1인 단어를 제거하는 것도 가능하겠지만 이번에는 별도 수행하지 않고 진행함.
        i=i+1
        word_to_index[word]=i
print(word_to_index)
print(len(word_to_index))
```

    {'PAD': 0, 'OOV': 1, 'the': 2, '.': 3, ',': 4, 'in': 5, 'of': 6, 'to': 7, 'a': 8, 'and': 9, "'s": 10, 'for': 11, 'has': 12, 'on': 13, 'is': 14, 'that': 15, 'have': 16, 'with': 17, 'said': 18, 'was': 19, 'at': 20, 'says': 21, 'from': 22, 'by': 23, 'he': 24, 'an': 25, 'as': 26, 'say': 27, 'u.s.': 28, 'it': 29, 'are': 30, '"': 31, 'were': 32, 'his': 33, 'will': 34, 'president': 35, 'officials': 36, 'government': 37, 'mr.': 38, 'been': 39, 'killed': 40, 'two': 41, 'people': 42, 'after': 43, 'not': 44, 'its': 45, 'be': 46, 'but': 47, 'they': 48, 'more': 49, 'also': 50, 'new': 51, 'united': 52, 'military': 53, 'who': 54, 'last': 55, 'country': 56, 'than': 57, 'minister': 58, 'police': 59, 'their': 60, 'iraq': 61, 'one': 62, 'which': 63, 'security': 64, 'this': 65, 'about': 66, 'year': 67, 'other': 68, 'had': 69, 'states': 70, 'least': 71, 'tuesday': 72, 'since': 73, 'forces': 74, 'world': 75, 'thursday': 76, 'week': 77, 'group': 78, 'iran': 79, 'over': 80, 'state': 81, 'friday': 82, 'monday': 83, 'wednesday': 84, 'against': 85, 'three': 86, 'during': 87, 'when': 88, 'sunday': 89, 'troops': 90, 'authorities': 91, '-': 92, 'into': 93, 'saturday': 94, 'would': 95, 'iraqi': 96, '$': 97, 'prime': 98, 'city': 99, 'foreign': 100, 'first': 101, 'some': 102, 'oil': 103, 'international': 104, 'nuclear': 105, 'attacks': 106, 'out': 107, 'militants': 108, 'up': 109, 'between': 110, 'month': 111, 'afghanistan': 112, 'nations': 113, 'former': 114, 'china': 115, 'years': 116, 'south': 117, 'palestinian': 118, 'talks': 119, 'attack': 120, 'israeli': 121, 'no': 122, 'bush': 123, 'israel': 124, 'called': 125, 'province': 126, 'or': 127, 'several': 128, 'war': 129, 'leader': 130, 'north': 131, 'near': 132, 'pakistan': 133, 'million': 134, 'southern': 135, 'region': 136, 'u.n.': 137, 'earlier': 138, 'there': 139, 'afghan': 140, 'countries': 141, 'including': 142, 'agency': 143, 'report': 144, 'party': 145, 'could': 146, 'saying': 147, 'four': 148, 'elections': 149, 'soldiers': 150, 'baghdad': 151, 'news': 152, 'all': 153, 'under': 154, 'national': 155, 'violence': 156, 'official': 157, 'reports': 158, 'economic': 159, 'political': 160, 'spokesman': 161, 'european': 162, 'capital': 163, 'before': 164, 'if': 165, 'statement': 166, 'leaders': 167, 'wounded': 168, 'court': 169, 'rebels': 170, 'bomb': 171, 'day': 172, 'told': 173, 'him': 174, 'percent': 175, 'american': 176, 'union': 177, ')': 178, '(': 179, 'where': 180, 'made': 181, 'election': 182, 'members': 183, 'peace': 184, 'part': 185, 'british': 186, 'most': 187, 'her': 188, 'while': 189, 'others': 190, 'may': 191, 'next': 192, 'because': 193, 'them': 194, 'border': 195, 'five': 196, 'program': 197, 'rights': 198, 'many': 199, 'expected': 200, 'russia': 201, 'died': 202, 'meeting': 203, 'fighting': 204, 'health': 205, 'economy': 206, 'killing': 207, 'house': 208, 'opposition': 209, 'another': 210, 'being': 211, 'human': 212, 'those': 213, 'what': 214, 'flu': 215, 'late': 216, 'held': 217, 'recent': 218, 'ministry': 219, 'northern': 220, 'she': 221, 'general': 222, 'insurgents': 223, 'taleban': 224, 'washington': 225, 'help': 226, 'gaza': 227, 'suspected': 228, 'down': 229, 'days': 230, 'released': 231, 'time': 232, 'town': 233, 'top': 234, 'bird': 235, 'western': 236, 'russian': 237, 'through': 238, 'power': 239, 'india': 240, 'weapons': 241, 'army': 242, 'korea': 243, 'later': 244, 'groups': 245, 'any': 246, 'can': 247, 'parliament': 248, 'men': 249, 'death': 250, 'nearly': 251, 'area': 252, 'off': 253, 'thousands': 254, 'announced': 255, 'coalition': 256, 'vote': 257, 'accused': 258, 'defense': 259, 'reported': 260, 'aid': 261, 'early': 262, 'only': 263, 'six': 264, 'following': 265, 'nato': 266, 'bank': 267, 'central': 268, 'found': 269, 'pakistani': 270, 'council': 271, 'west': 272, 'support': 273, 'set': 274, 'arrested': 275, 'chief': 276, 'number': 277, 'end': 278, 'organization': 279, 'militant': 280, 'meanwhile': 281, 'chinese': 282, 'second': 283, 'charges': 284, 'both': 285, 'plan': 286, 'african': 287, 'took': 288, 'began': 289, 'visit': 290, 'among': 291, 'months': 292, '10': 293, '%': 294, 'al-qaida': 295, 'force': 296, 'agreement': 297, 'major': 298, 'trade': 299, 'take': 300, 'such': 301, 'efforts': 302, 'company': 303, 'suicide': 304, 'prices': 305, 'local': 306, 'home': 307, "'": 308, 'french': 309, 'did': 310, 'high': 311, 'workers': 312, 'energy': 313, 'secretary': 314, 'islamic': 315, 'eastern': 316, 'hamas': 317, 'media': 318, 'car': 319, 'man': 320, 'presidential': 321, 'came': 322, 'january': 323, 'now': 324, 'should': 325, 'still': 326, 'iranian': 327, 'department': 328, 'place': 329, 'plans': 330, 'east': 331, 'public': 332, 'billion': 333, 'sudan': 334, 'japan': 335, 'eu': 336, 'chavez': 337, 'venezuela': 338, 'deal': 339, 'democratic': 340, 'work': 341, 'dead': 342, 'used': 343, 'turkey': 344, 'food': 345, 'until': 346, 'terrorist': 347, 'africa': 348, 'control': 349, 'virus': 350, 'meet': 351, 'growth': 352, 'past': 353, '2003': 354, 'ago': 355, 'nation': 356, 'tehran': 357, 'blast': 358, 'muslim': 359, 'weeks': 360, 'fire': 361, 'newspaper': 362, 'met': 363, 'march': 364, 'britain': 365, 'indian': 366, 'seven': 367, 'hit': 368, 'make': 369, 'darfur': 370, 'reporters': 371, 'office': 372, 'detained': 373, 'confirmed': 374, 'left': 375, '20': 376, "shi'ite": 377, 'global': 378, 'outside': 379, 'france': 380, 'head': 381, 'burma': 382, 'across': 383, 'beijing': 384, 'civilians': 385, 'white': 386, 'agreed': 387, 'september': 388, 'known': 389, 'continue': 390, 'gunmen': 391, 'just': 392, 'december': 393, 'areas': 394, 'so': 395, 'prison': 396, 'information': 397, 'well': 398, 'around': 399, 'along': 400, 'release': 401, 'gas': 402, 'scheduled': 403, 'development': 404, 'lebanon': 405, 'financial': 406, 'island': 407, 'children': 408, 'coast': 409, 'rebel': 410, 'signed': 411, 'issued': 412, 'hundreds': 413, 'lawmakers': 414, 'led': 415, 'stop': 416, 'administration': 417, 'suspects': 418, 'eight': 419, 'syria': 420, '30': 421, 'protests': 422, 'television': 423, 'must': 424, 'women': 425, '15': 426, 'separate': 427, 'use': 428, 'kilometers': 429, 'trying': 430, 'claimed': 431, 'july': 432, 'without': 433, 'law': 434, 'germany': 435, 'back': 436, 'do': 437, 'intelligence': 438, 'constitution': 439, 'operations': 440, 'however': 441, 'victims': 442, 'independence': 443, 'europe': 444, 'incident': 445, 'officers': 446, 'york': 447, 'air': 448, 'include': 449, 'somalia': 450, 'discuss': 451, 'parliamentary': 452, 'korean': 453, 'system': 454, 'conference': 455, '11': 456, 'november': 457, 'largest': 458, 'main': 459, 'series': 460, 'aimed': 461, 'taliban': 462, 'fired': 463, 'third': 464, 'decision': 465, 'return': 466, 'republic': 467, 'london': 468, 'february': 469, 'egypt': 470, 'key': 471, 'operation': 472, 'responsibility': 473, 'shot': 474, 'hurricane': 475, 'crisis': 476, 'commission': 477, 'large': 478, 'member': 479, 'investigation': 480, 'center': 481, 'despite': 482, 'possible': 483, 'ruling': 484, 'strip': 485, 'ahead': 486, 'civil': 487, 'august': 488, 'calls': 489, 'urged': 490, 'soldier': 491, 'service': 492, 'hold': 493, 'move': 494, 'june': 495, 'mission': 496, 'alleged': 497, 'taken': 498, 'won': 499, '2008': 500, 'deadly': 501, 'team': 502, 'today': 503, 'october': 504, 'abbas': 505, 'summit': 506, 'conflict': 507, 'process': 508, 'obama': 509, 'hospital': 510, 'crimes': 511, 'bombings': 512, 'trial': 513, '2004': 514, 'how': 515, 'venezuelan': 516, 'free': 517, 'final': 518, 'relations': 519, 'then': 520, 'small': 521, 'injured': 522, 'protest': 523, 'close': 524, 'prisoners': 525, 'base': 526, 'case': 527, 'sanctions': 528, 'congress': 529, 'voa': 530, 'roadside': 531, 'remain': 532, 'each': 533, 'increase': 534, 'cuba': 535, 'terrorism': 536, 'way': 537, 'mahmoud': 538, 'fuel': 539, 'comes': 540, 'district': 541, '2001': 542, 'latest': 543, 'senior': 544, 'warned': 545, 'much': 546, 'cases': 547, '12': 548, 'approved': 549, 'market': 550, 'rice': 551, 'press': 552, 'palestinians': 553, 'explosion': 554, 'armed': 555, 'effort': 556, 'building': 557, 'same': 558, 'storm': 559, 'water': 560, 'arab': 561, 'john': 562, 'u.s.-led': 563, 'companies': 564, 'turkish': 565, 'campaign': 566, 'carried': 567, 'kurdish': 568, 'production': 569, 'colombia': 570, 'demand': 571, 'show': 572, 'deaths': 573, 'denied': 574, 'federal': 575, 'whether': 576, 'strike': 577, 'working': 578, 'indonesia': 579, 'asia': 580, 'already': 581, 'allow': 582, '2010': 583, 'face': 584, 'hours': 585, 'caused': 586, 'remains': 587, 'protesters': 588, 'bombing': 589, 'money': 590, 'calling': 591, 'attacked': 592, 'results': 593, 'h5n1': 594, 'lebanese': 595, 'failed': 596, 'bomber': 597, 'does': 598, 'recently': 599, 'emergency': 600, 'site': 601, 'cut': 602, 'ukraine': 603, '50': 604, 'clear': 605, 'kidnapped': 606, 'rejected': 607, 'committee': 608, 'pope': 609, 'ordered': 610, 'moscow': 611, 'due': 612, 'issues': 613, 'sent': 614, 'saudi': 615, '2006': 616, '2009': 617, ';': 618, 'majority': 619, 'station': 620, 'drug': 621, 'bill': 622, 'within': 623, 'record': 624, 'sector': 625, 'expressed': 626, 'action': 627, 'residents': 628, 'witnesses': 629, 'give': 630, 'embassy': 631, '25': 632, 'arrest': 633, 'mexico': 634, 'you': 635, 'japanese': 636, 'post': 637, 'ban': 638, 'sunni': 639, 'middle': 640, 'strain': 641, 'opened': 642, 'relief': 643, 'further': 644, 'disease': 645, 'kashmir': 646, 'term': 647, 'policy': 648, 'occurred': 649, 'carrying': 650, '100': 651, 'islands': 652, 'future': 653, 'seized': 654, 'planned': 655, 'ministers': 656, 'concerns': 657, 'taking': 658, 'us': 659, 'family': 660, 'increased': 661, 'neighboring': 662, 'come': 663, 'fighters': 664, 'exports': 665, 'german': 666, 'americans': 667, 'asked': 668, 'launched': 669, 'territory': 670, 'supporters': 671, 'justice': 672, 'vehicle': 673, 'call': 674, 'criticized': 675, 'april': 676, 'captured': 677, 'saddam': 678, 'radio': 679, 'ethnic': 680, 'birds': 681, 'movement': 682, 'corruption': 683, 'taiwan': 684, 'ended': 685, 'heavy': 686, 'nine': 687, 'cabinet': 688, 'involved': 689, 'leading': 690, 'role': 691, 'woman': 692, 'homes': 693, 'strong': 694, 'long': 695, 'civilian': 696, 'king': 697, 'camp': 698, '2002': 699, 'cooperation': 700, 'situation': 701, 'i': 702, 'independent': 703, 'charged': 704, '2007': 705, 'vice': 706, 'parties': 707, 'put': 708, 'begin': 709, 'nigeria': 710, '40': 711, 'haiti': 712, 'democracy': 713, 'ties': 714, 'trip': 715, 'half': 716, 'became': 717, 'own': 718, 'regional': 719, 'space': 720, 'deputy': 721, 'spain': 722, 'go': 723, 'experts': 724, 'problems': 725, '2005': 726, 'outbreak': 727, 'judge': 728, 'commander': 729, 'blamed': 730, 'syrian': 731, 'ms.': 732, 'gdp': 733, 'special': 734, 'poor': 735, 'went': 736, 'labor': 737, 'open': 738, 'comments': 739, 'land': 740, 'believed': 741, 'forced': 742, 'candidate': 743, 'business': 744, 'issue': 745, 'continued': 746, 'missing': 747, 'provide': 748, 'details': 749, 'joint': 750, 'america': 751, 'current': 752, 'ambassador': 753, 'activities': 754, 'exploded': 755, 'lead': 756, 'medical': 757, 'negotiations': 758, 'detainees': 759, 'times': 760, 'uranium': 761, 'become': 762, 'making': 763, 'senate': 764, 'speaking': 765, 'get': 766, 'tribal': 767, 'services': 768, 'fight': 769, 'musharraf': 770, 'parts': 771, 'threat': 772, 'community': 773, 'far': 774, 'missile': 775, 'bodies': 776, 'order': 777, 'price': 778, 'want': 779, 'industry': 780, 'egyptian': 781, '14': 782, 'change': 783, 'illegal': 784, 'per': 785, 'quotes': 786, 'terrorists': 787, 'reached': 788, 'seeking': 789, 'aircraft': 790, 'received': 791, 'annan': 792, '13': 793, 'natural': 794, 'allegations': 795, 'arrived': 796, 'supplies': 797, 'measures': 798, 'address': 799, 'rule': 800, 'muslims': 801, 'run': 802, 'referendum': 803, 'supreme': 804, '18': 805, 'wants': 806, 'gulf': 807, 'travel': 808, 'life': 809, 'form': 810, 'alliance': 811, 'investment': 812, 'village': 813, 'sharon': 814, 'declared': 815, 'officer': 816, 'asian': 817, 'journalists': 818, 'threatened': 819, 'win': 820, 'toward': 821, 'dollars': 822, 'olympic': 823, 'again': 824, 'sides': 825, 'citizens': 826, 'activists': 827, 'published': 828, 'mostly': 829, 'ali': 830, 'exchange': 831, 'kandahar': 832, 'withdrawal': 833, 'fled': 834, 'offensive': 835, 'interview': 836, 'immediately': 837, 'religious': 838, 'based': 839, 'hugo': 840, 'wanted': 841, 'dispute': 842, 'reforms': 843, 'tried': 844, 'fell': 845, 'atomic': 846, 'century': 847, 'seen': 848, 'earthquake': 849, 'kabul': 850, 'similar': 851, 'abu': 852, 'concern': 853, 'body': 854, 'toll': 855, 'construction': 856, 'attempt': 857, 'karzai': 858, 'network': 859, 'few': 860, 'authority': 861, 'private': 862, 'cities': 863, 'candidates': 864, 'interim': 865, 'cuban': 866, 'interior': 867, 'seats': 868, 'refused': 869, 'continues': 870, 'bring': 871, 'sudanese': 872, 'holding': 873, 'using': 874, 'brazil': 875, 'somali': 876, 'need': 877, 'build': 878, 'suspect': 879, 'broke': 880, 'crude': 881, 'announcement': 882, 'almost': 883, 'round': 884, 'fatah': 885, 'putin': 886, 'ousted': 887, '17': 888, 'leave': 889, 'yet': 890, 'convoy': 891, 'response': 892, 'sri': 893, 'massive': 894, 'millions': 895, 'sea': 896, 'growing': 897, 'mass': 898, 'wounding': 899, 'congo': 900, 'passed': 901, 'pressure': 902, 'decades': 903, 'lost': 904, 'governor': 905, 'warning': 906, 'himself': 907, 'prevent': 908, 'assistance': 909, 'assembly': 910, 'tax': 911, 'population': 912, 'rate': 913, 'currently': 914, 'clashes': 915, 'spread': 916, 'likely': 917, 'debt': 918, 'tsunami': 919, 'reform': 920, 'arms': 921, 'ahmadinejad': 922, 'explosives': 923, 'ruled': 924, 'proposed': 925, 'australia': 926, 'raid': 927, 'returned': 928, 'secretary-general': 929, '1': 930, 'soon': 931, 'added': 932, 'total': 933, 'voted': 934, 'katrina': 935, 'allowed': 936, 'annual': 937, 'named': 938, 'charge': 939, 'policemen': 940, '200': 941, 'person': 942, 'rocket': 943, 'evidence': 944, 'legal': 945, 'although': 946, 'communist': 947, 'good': 948, 'peaceful': 949, 'conditions': 950, 'associated': 951, 'hussein': 952, 'jewish': 953, 'full': 954, 'needed': 955, 'progress': 956, 'terror': 957, 'nepal': 958, 'casualties': 959, '16': 960, 'italy': 961, 'probe': 962, 'right': 963, 'school': 964, 'yushchenko': 965, 'away': 966, 'demonstrators': 967, 'fraud': 968, 'spoke': 969, 'insurgent': 970, 'detention': 971, 'hezbollah': 972, 'share': 973, 'analysts': 974, 'destroyed': 975, 'prosecutors': 976, '60': 977, 'patrol': 978, 'elsewhere': 979, 'castro': 980, 'assassination': 981, 'violent': 982, 'activity': 983, 'enough': 984, 'spending': 985, 'diplomatic': 986, 'paul': 987, 'pyongyang': 988, 'given': 989, 'blair': 990, 'less': 991, 'sales': 992, 'keep': 993, 'struck': 994, 'tests': 995, 'believe': 996, 'even': 997, 'agriculture': 998, 'georgia': 999, 'road': 1000, 'elected': 1001, 'facilities': 1002, 'test': 1003, 'proposal': 1004, 'access': 1005, 'freed': 1006, 'estimated': 1007, 'governments': 1008, 'tens': 1009, 'bombs': 1010, 'iraqis': 1011, 'constitutional': 1012, 'living': 1013, 'important': 1014, 'linked': 1015, 'xinhua': 1016, 'representatives': 1017, 'poll': 1018, 'tribunal': 1019, 'like': 1020, 'resolution': 1021, 'voters': 1022, 'reach': 1023, 'resume': 1024, 'dozens': 1025, 'offer': 1026, 'senator': 1027, 'peacekeeping': 1028, 'director': 1029, 'custody': 1030, 'hopes': 1031, 'spanish': 1032, 'policies': 1033, 'closed': 1034, 'pay': 1035, 'develop': 1036, 'insurgency': 1037, 'kill': 1038, 'promised': 1039, 'front': 1040, 'diplomats': 1041, 'facility': 1042, 'borders': 1043, 'video': 1044, 'name': 1045, 'san': 1046, 'social': 1047, 'behind': 1048, 'includes': 1049, 'planning': 1050, 'board': 1051, 'brought': 1052, 'speech': 1053, 'measure': 1054, 'peacekeepers': 1055, 'thailand': 1056, 'poultry': 1057, 'kenya': 1058, 'damage': 1059, 'fund': 1060, 'rose': 1061, 'turned': 1062, 'southeast': 1063, 'cup': 1064, 'australian': 1065, 'tourism': 1066, 'clinton': 1067, 'enrichment': 1068, 'bus': 1069, 'weather': 1070, 'gave': 1071, 'allegedly': 1072, 'journalist': 1073, 'mosque': 1074, 'humanitarian': 1075, 'interest': 1076, 'responsible': 1077, 'games': 1078, 'visited': 1079, 'night': 1080, 'identified': 1081, 'investigating': 1082, 'italian': 1083, '2000': 1084, 'agencies': 1085, 'start': 1086, 'popular': 1087, 'higher': 1088, 'involvement': 1089, 'discovered': 1090, 'sources': 1091, 'pledged': 1092, 'joined': 1093, 'claim': 1094, 'suspended': 1095, 'denies': 1096, 'coming': 1097, 'rival': 1098, 'join': 1099, 'abducted': 1100, 'tensions': 1101, 'mohammed': 1102, 'indonesian': 1103, 'hamid': 1104, 'rising': 1105, 'leaving': 1106, 'faces': 1107, 'condemned': 1108, 'lives': 1109, 'trading': 1110, 'port': 1111, 'provincial': 1112, 'shows': 1113, 'refugees': 1114, 'connection': 1115, 'list': 1116, 'ceremony': 1117, 'cause': 1118, '26': 1119, 'airport': 1120, 'gathered': 1121, 'strikes': 1122, 'lower': 1123, 'jobs': 1124, 'niger': 1125, 'helped': 1126, 'islamabad': 1127, 'ethiopia': 1128, 'developing': 1129, 'begun': 1130, 'request': 1131, 'comment': 1132, 'polls': 1133, 'helmand': 1134, 'budget': 1135, 'ii': 1136, 'secret': 1137, 'immediate': 1138, 'ocean': 1139, 'de': 1140, 'education': 1141, 'able': 1142, '80': 1143, 'ready': 1144, 'showed': 1145, 'internet': 1146, 'plane': 1147, 'helicopter': 1148, 'victory': 1149, 'appeared': 1150, 'convicted': 1151, 'arabia': 1152, 'freedom': 1153, 'worldwide': 1154, 'try': 1155, 'demanding': 1156, 'families': 1157, 'season': 1158, 'step': 1159, 'northwest': 1160, ':': 1161, 'church': 1162, 'targeted': 1163, 'launch': 1164, 'plant': 1165, 'unemployment': 1166, 'staff': 1167, 'every': 1168, 'anniversary': 1169, 'often': 1170, 'chairman': 1171, 'previous': 1172, 'rally': 1173, 'training': 1174, 'together': 1175, 'once': 1176, 'paris': 1177, 'claims': 1178, 'mosul': 1179, 'controversial': 1180, 'criticism': 1181, 'serious': 1182, 'colombian': 1183, 'domestic': 1184, 'nearby': 1185, 'lack': 1186, 'low': 1187, 'included': 1188, 'never': 1189, 'better': 1190, 'condition': 1191, 'infected': 1192, 'offered': 1193, 'democrats': 1194, 'having': 1195, 'aids': 1196, 'resources': 1197, 'missiles': 1198, 'approval': 1199, 'bosnian': 1200, 'river': 1201, 'demands': 1202, 'previously': 1203, 'barrel': 1204, 'treatment': 1205, 'poverty': 1206, 'tour': 1207, 'combat': 1208, 'ship': 1209, 'amid': 1210, 'state-run': 1211, 'electoral': 1212, 'focus': 1213, 'personnel': 1214, 'affairs': 1215, 'clash': 1216, 'separately': 1217, 'demonstrations': 1218, 'provided': 1219, 'injuries': 1220, 'protect': 1221, 'entered': 1222, 'broadcast': 1223, 'northwestern': 1224, 'islamist': 1225, 'event': 1226, 'poland': 1227, 'guilty': 1228, 'hong': 1229, 'chad': 1230, 'kosovo': 1231, 'rescue': 1232, 'care': 1233, 'suffered': 1234, 'doctors': 1235, 'period': 1236, 'described': 1237, 'act': 1238, 'largely': 1239, 'dropped': 1240, 'highest': 1241, 'level': 1242, 'followed': 1243, 'canada': 1244, 'infrastructure': 1245, 'kofi': 1246, 'burmese': 1247, 'daily': 1248, '21': 1249, 'battle': 1250, 'red': 1251, 'murder': 1252, 'legislation': 1253, 'hope': 1254, 'delegation': 1255, 'abuse': 1256, 'crackdown': 1257, 'guards': 1258, 'guantanamo': 1259, 'create': 1260, 'technology': 1261, 'boost': 1262, 'mexican': 1263, 'too': 1264, 'project': 1265, 'whose': 1266, '24': 1267, 'sign': 1268, 'jerusalem': 1269, 'bin': 1270, 'find': 1271, 'condoleezza': 1272, 'safety': 1273, 'imposed': 1274, 'cross': 1275, 'traveling': 1276, 'soviet': 1277, '19': 1278, 'treaty': 1279, 'see': 1280, '1,000': 1281, 'coup': 1282, 'barack': 1283, 'attend': 1284, 'association': 1285, 'holiday': 1286, '2': 1287, 'kong': 1288, 'cia': 1289, 'contact': 1290, 'dollar': 1291, 'opening': 1292, 'voting': 1293, '70': 1294, 'declined': 1295, 'hostages': 1296, 'vehicles': 1297, 'scientists': 1298, 'electricity': 1299, 'reduce': 1300, 'southeastern': 1301, 'very': 1302, 'farm': 1303, 'hurt': 1304, 'beirut': 1305, 'hariri': 1306, 'significant': 1307, 'sentenced': 1308, 'fox': 1309, 'ukrainian': 1310, 'quoted': 1311, 'criminal': 1312, 'targets': 1313, 'mogadishu': 1314, 'mine': 1315, 'programs': 1316, 'young': 1317, 'date': 1318, 'zimbabwe': 1319, 'displaced': 1320, 'islam': 1321, 'helping': 1322, 'agricultural': 1323, 'fourth': 1324, 'critics': 1325, 'cairo': 1326, 'spokeswoman': 1327, 'provinces': 1328, 'common': 1329, 'demanded': 1330, 'raids': 1331, 'funds': 1332, 'goods': 1333, 'inside': 1334, 'supply': 1335, 'running': 1336, 'orleans': 1337, 'christian': 1338, 'concerned': 1339, 'export': 1340, 'robert': 1341, 'appeal': 1342, 'allies': 1343, 'ending': 1344, 'raised': 1345, 'projects': 1346, 'unit': 1347, 'disputed': 1348, 'seek': 1349, 'dutch': 1350, 'rates': 1351, 'lanka': 1352, 'cost': 1353, 'send': 1354, 'jordan': 1355, 'hand': 1356, 'vietnam': 1357, 'refugee': 1358, 'markets': 1359, '22': 1360, 'crash': 1361, 'hour': 1362, 'viktor': 1363, 'california': 1364, 'buildings': 1365, 'morning': 1366, 'quake': 1367, 'tony': 1368, 'needs': 1369, 'improve': 1370, 'rockets': 1371, 'facing': 1372, 'short': 1373, 'republican': 1374, 'texas': 1375, 'separatist': 1376, 'shooting': 1377, 'stability': 1378, 'envoy': 1379, 'additional': 1380, 'withdraw': 1381, 'train': 1382, 'search': 1383, 'message': 1384, 'match': 1385, 'letter': 1386, 'little': 1387, 'rise': 1388, 'these': 1389, 'serb': 1390, 'banned': 1391, 'membership': 1392, 'crew': 1393, 'presence': 1394, 'urging': 1395, 'review': 1396, 'upon': 1397, 'great': 1398, 'watch': 1399, 'disaster': 1400, 'vladimir': 1401, 'bombers': 1402, 'changes': 1403, 'militia': 1404, 'served': 1405, 'mainly': 1406, '1995': 1407, 'fall': 1408, 'regime': 1409, 'heavily': 1410, 'tokyo': 1411, 'holy': 1412, 'raise': 1413, 'decided': 1414, 'remote': 1415, 'affected': 1416, 'streets': 1417, 'regions': 1418, 'study': 1419, 'consumer': 1420, 'amnesty': 1421, 'organizations': 1422, '7': 1423, 'ethiopian': 1424, 'settlement': 1425, 'genocide': 1426, 'offices': 1427, 'students': 1428, 'crossing': 1429, 'votes': 1430, '90': 1431, 'live': 1432, 'remarks': 1433, 'links': 1434, '--': 1435, 'replace': 1436, 'history': 1437, 'firm': 1438, 'formally': 1439, 'sparked': 1440, 'stations': 1441, 'cease-fire': 1442, 'vatican': 1443, 'resolve': 1444, 'might': 1445, 'rains': 1446, 'income': 1447, 'killings': 1448, 'shortly': 1449, 'paid': 1450, 'session': 1451, 'repeatedly': 1452, 'fishing': 1453, 'result': 1454, 'play': 1455, 'products': 1456, 'presidency': 1457, '1991': 1458, 'dismissed': 1459, 'delay': 1460, 'old': 1461, 'started': 1462, 'stepped': 1463, 'continuing': 1464, 'tournament': 1465, 'considered': 1466, 'olmert': 1467, 'stock': 1468, 'kidnappers': 1469, 'unless': 1470, 'son': 1471, 'greater': 1472, 'accident': 1473, 'direct': 1474, 'threats': 1475, 'cleric': 1476, 'invasion': 1477, 'delhi': 1478, 'christmas': 1479, 'limited': 1480, 'truck': 1481, 'laden': 1482, 'prosecutor': 1483, 'filed': 1484, 'resigned': 1485, 'pirates': 1486, 'street': 1487, 'carry': 1488, 'schools': 1489, 'erupted': 1490, 'employees': 1491, 'beginning': 1492, 'laws': 1493, 'consider': 1494, 'target': 1495, 'meetings': 1496, 'abdullah': 1497, 'football': 1498, 'throughout': 1499, 'avoid': 1500, 'waziristan': 1501, '9': 1502, 'black': 1503, 'widespread': 1504, 'incidents': 1505, 'executive': 1506, 'mark': 1507, 'militias': 1508, 'accord': 1509, 'deficit': 1510, 'northeastern': 1511, 'pervez': 1512, 'housing': 1513, 'nigerian': 1514, 'ground': 1515, 'recovery': 1516, 'located': 1517, '27': 1518, 'lawyers': 1519, 'tamil': 1520, 'spent': 1521, 'florida': 1522, 'shut': 1523, 'headquarters': 1524, 'bilateral': 1525, 'ivory': 1526, 'resignation': 1527, 'giving': 1528, 'line': 1529, 'barrels': 1530, 'points': 1531, 'george': 1532, 'permanent': 1533, 'flights': 1534, '8': 1535, 'towns': 1536, 'upcoming': 1537, 'increasing': 1538, 'driver': 1539, 'gold': 1540, 'powers': 1541, 'mubarak': 1542, 'hu': 1543, 'worst': 1544, 'politicians': 1545, 'ways': 1546, 'accounts': 1547, 'singh': 1548, 'research': 1549, 'sending': 1550, 'checkpoint': 1551, 'job': 1552, '500': 1553, 'blew': 1554, 'mohammad': 1555, '3': 1556, '1999': 1557, 'huge': 1558, 'brother': 1559, 'fear': 1560, 'canadian': 1561, 'leadership': 1562, 'wife': 1563, 'diplomat': 1564, 'backed': 1565, 'stopped': 1566, 'deployed': 1567, 'torture': 1568, 'chemical': 1569, 'looking': 1570, 'point': 1571, 'potential': 1572, 'industrial': 1573, 'crime': 1574, 'draft': 1575, 'kim': 1576, '31': 1577, 'child': 1578, 'targeting': 1579, 'fair': 1580, 'visiting': 1581, 'panel': 1582, 'positive': 1583, 'equipment': 1584, 'league': 1585, 'marines': 1586, 'effect': 1587, 'buy': 1588, 'khan': 1589, 'reuters': 1590, 'netherlands': 1591, 'reconstruction': 1592, '23': 1593, '1998': 1594, 'powerful': 1595, 'rumsfeld': 1596, 'southwestern': 1597, '2011': 1598, 'apparently': 1599, 'providing': 1600, 'recession': 1601, 'conducted': 1602, 'winds': 1603, 'critical': 1604, 'best': 1605, 'jose': 1606, 'source': 1607, 'costs': 1608, 'instead': 1609, 'finance': 1610, 'levels': 1611, 'father': 1612, 'ecuador': 1613, 'loss': 1614, 'ariel': 1615, 'created': 1616, 'damaged': 1617, 'risk': 1618, 'aung': 1619, 'abroad': 1620, 'established': 1621, '1990s': 1622, 'arrests': 1623, 'olympics': 1624, 'push': 1625, 'enter': 1626, 'stay': 1627, 'worth': 1628, 'host': 1629, 'ahmed': 1630, 'committed': 1631, 'race': 1632, 'businesses': 1633, 'produce': 1634, 'investigators': 1635, 'girl': 1636, 'related': 1637, 'necessary': 1638, 'protection': 1639, 'worked': 1640, '5': 1641, 'considering': 1642, 'moved': 1643, 'detonated': 1644, 'inflation': 1645, 'divided': 1646, 'agree': 1647, 'latin': 1648, 'guard': 1649, 'iaea': 1650, 'triggered': 1651, 'banks': 1652, 'survey': 1653, 'doing': 1654, 'norway': 1655, 'petroleum': 1656, 'ongoing': 1657, 'waters': 1658, 'kirkuk': 1659, 'imports': 1660, 'liberation': 1661, 'side': 1662, 'pipeline': 1663, 'vowed': 1664, 'plot': 1665, 'attackers': 1666, 'operating': 1667, 'winner': 1668, 'unidentified': 1669, 'zone': 1670, 'agreements': 1671, 'catholic': 1672, 'scene': 1673, 'sharply': 1674, 'figures': 1675, 'passengers': 1676, 'maoist': 1677, 'argentina': 1678, 'decade': 1679, 'document': 1680, 'hague': 1681, 'minority': 1682, 'survivors': 1683, 'bangladesh': 1684, 'allowing': 1685, 'praised': 1686, 'remaining': 1687, 'real': 1688, 'bay': 1689, 'position': 1690, 'present': 1691, 'peninsula': 1692, 'weekly': 1693, 'explosions': 1694, 'tropical': 1695, 'unclear': 1696, 'navy': 1697, 'israelis': 1698, 'benedict': 1699, 'communities': 1700, 'heart': 1701, 'documents': 1702, 'saw': 1703, 'opponents': 1704, 'discussed': 1705, 'haitian': 1706, 'caracas': 1707, 'moving': 1708, '300': 1709, 'restrictions': 1710, 'currency': 1711, 'strengthen': 1712, 'accept': 1713, 'cheney': 1714, 'peru': 1715, 'villages': 1716, 'investors': 1717, 'helicopters': 1718, 'severe': 1719, 'ships': 1720, 'dictator': 1721, 'naval': 1722, 'funeral': 1723, 'so-called': 1724, 'dialogue': 1725, 'failure': 1726, 'escaped': 1727, 'czech': 1728, 'sentence': 1729, 'age': 1730, 'worker': 1731, 'formed': 1732, 'status': 1733, 'fiscal': 1734, 'funding': 1735, 'according': 1736, 'happened': 1737, 'mayor': 1738, 'damascus': 1739, 'dangerous': 1740, '4': 1741, 'kyi': 1742, 'designed': 1743, 'your': 1744, 'blasts': 1745, 'cuts': 1746, 'university': 1747, 'crowd': 1748, 'resort': 1749, 'halt': 1750, 'supporting': 1751, 'placed': 1752, 'imf': 1753, 'safe': 1754, 'counterpart': 1755, 'overall': 1756, 'hostage': 1757, 'actions': 1758, 'completed': 1759, 'truce': 1760, 'pacific': 1761, 'troubled': 1762, 'music': 1763, 'mining': 1764, 'biggest': 1765, 'winter': 1766, '150': 1767, 'jihad': 1768, 'conservative': 1769, 'giant': 1770, 'au': 1771, 'cars': 1772, 'expect': 1773, 'nationwide': 1774, 'destruction': 1775, 'terms': 1776, 'sites': 1777, 'involving': 1778, 'prominent': 1779, 'positions': 1780, 'figure': 1781, 'handed': 1782, '28': 1783, 'occupied': 1784, 'resistance': 1785, 'valley': 1786, 'quickly': 1787, 'jail': 1788, 'chancellor': 1789, "n't": 1790, 'champion': 1791, 'posted': 1792, 'michael': 1793, 'gasoline': 1794, 'marched': 1795, 'flooding': 1796, 'floods': 1797, 'ensure': 1798, 'sectors': 1799, 'al': 1800, '1996': 1801, 'average': 1802, 'david': 1803, 'deadline': 1804, 'web': 1805, 'sought': 1806, 'slow': 1807, 'tennis': 1808, 'extended': 1809, 'omar': 1810, 'though': 1811, 'osama': 1812, 'decline': 1813, 'kingdom': 1814, 'autonomy': 1815, 'receive': 1816, 'events': 1817, 'opinion': 1818, 'hard': 1819, 'leftist': 1820, 'lawyer': 1821, 'nothing': 1822, 'tested': 1823, 'pkk': 1824, 'takes': 1825, 'humans': 1826, 'kenyan': 1827, 'suu': 1828, 'indicted': 1829, 'problem': 1830, 'highly': 1831, 'my': 1832, 'kidnapping': 1833, 'immigrants': 1834, 'croatia': 1835, 'two-day': 1836, 'maintain': 1837, 'played': 1838, 'ceasefire': 1839, 'boat': 1840, 'compound': 1841, 'fought': 1842, 'drugs': 1843, 'officially': 1844, 'goal': 1845, 'environmental': 1846, 'pushed': 1847, 'losses': 1848, 'farc': 1849, 'quarter': 1850, 'attempts': 1851, 'preparing': 1852, 'foreigners': 1853, 'revenue': 1854, 'challenges': 1855, 'boy': 1856, 'internal': 1857, 'believes': 1858, 'approve': 1859, 'conduct': 1860, 'accuses': 1861, 'rain': 1862, 'rules': 1863, 'formal': 1864, 'ugandan': 1865, 'illegally': 1866, '2,00,000': 1867, 'complete': 1868, 'fully': 1869, 'suffering': 1870, 'we': 1871, 'revolutionary': 1872, 'factions': 1873, 'questioned': 1874, 'unity': 1875, 'meters': 1876, 'restore': 1877, 'follows': 1878, 'account': 1879, 'greek': 1880, 'attorney': 1881, 'film': 1882, 'earth': 1883, 'fifth': 1884, 'estimates': 1885, 'itself': 1886, 'wall': 1887, 'reportedly': 1888, 'strongly': 1889, 'talabani': 1890, 'defeated': 1891, 'sovereignty': 1892, 'immigration': 1893, 'delivery': 1894, 'uganda': 1895, 'accuse': 1896, 'showing': 1897, 'bloc': 1898, 'field': 1899, 'producer': 1900, 'hassan': 1901, 'tons': 1902, 'developed': 1903, 'pass': 1904, '1994': 1905, 'bolivia': 1906, 'delta': 1907, 'indicate': 1908, 'planes': 1909, 'words': 1910, 'either': 1911, 'difficult': 1912, 'longer': 1913, 'centers': 1914, 'negotiators': 1915, 'hiv': 1916, 'especially': 1917, 'addition': 1918, 'square': 1919, 'hearing': 1920, 'determine': 1921, 'particularly': 1922, 'uk': 1923, 'light': 1924, 'ever': 1925, 'hotel': 1926, 'tourists': 1927, 'extremists': 1928, 'forecasters': 1929, '6': 1930, 'sweden': 1931, 'polish': 1932, 'threatening': 1933, 'manufacturing': 1934, 'chickens': 1935, 'rwanda': 1936, 'neighborhood': 1937, 'widely': 1938, 'commercial': 1939, 'un': 1940, 'bid': 1941, 'pentagon': 1942, 'heads': 1943, 'boycott': 1944, 'territories': 1945, 'jailed': 1946, 'raided': 1947, 'makes': 1948, '1989': 1949, 'defeat': 1950, 'stronghold': 1951, 'crashed': 1952, 'steps': 1953, 'nationals': 1954, 'fidel': 1955, 'climate': 1956, 'ransom': 1957, 'resumed': 1958, 'break': 1959, 'range': 1960, 'caught': 1961, 'rest': 1962, 'themselves': 1963, 'serbian': 1964, '1990': 1965, 'drop': 1966, 'confidence': 1967, 'unrest': 1968, 'chile': 1969, 'promote': 1970, 'secure': 1971, 'impact': 1972, 'donors': 1973, 'headed': 1974, 'revenues': 1975, 'minutes': 1976, 'gain': 1977, 'tibet': 1978, 'abdul': 1979, 'forward': 1980, 'expand': 1981, 'postponed': 1982, 'initial': 1983, 'assets': 1984, 'intended': 1985, 'temporarily': 1986, 'morocco': 1987, 'failing': 1988, 'opec': 1989, 'opposed': 1990, 'merkel': 1991, 'sectarian': 1992, 'teams': 1993, 'struggle': 1994, 'cancer': 1995, 'ballots': 1996, 'towards': 1997, 'runs': 1998, 'reduced': 1999, 'overnight': 2000, 'pleaded': 2001, 'granted': 2002, 'prophet': 2003, 'sold': 2004, 'material': 2005, 'taxes': 2006, 'contract': 2007, 'below': 2008, 'certain': 2009, 'lines': 2010, 'appears': 2011, 'canceled': 2012, 'avian': 2013, 'returning': 2014, 'blood': 2015, 'homeland': 2016, 'thought': 2017, 'stand': 2018, 'package': 2019, 'collapse': 2020, 'scandal': 2021, 'going': 2022, 'favor': 2023, 'assault': 2024, 'yanukovych': 2025, 'legislative': 2026, 'houses': 2027, 'bringing': 2028, 'aristide': 2029, 'cents': 2030, 'shares': 2031, 'reporting': 2032, 'thousand': 2033, 'let': 2034, 'voice': 2035, 'investigate': 2036, 'turn': 2037, '10,000': 2038, 'violations': 2039, 'caribbean': 2040, 'danish': 2041, 'vienna': 2042, 'recognize': 2043, 'debate': 2044, 'data': 2045, 'mohamed': 2046, 'expects': 2047, 'wars': 2048, 'causing': 2049, 'england': 2050, 'means': 2051, 'adopted': 2052, 'withdrew': 2053, 'appear': 2054, 'representative': 2055, 'serve': 2056, 'presidents': 2057, 'serbia': 2058, 'matter': 2059, 'built': 2060, 'output': 2061, 'scored': 2062, 'congressional': 2063, 'aceh': 2064, 'prize': 2065, 'value': 2066, 'roads': 2067, 'anyone': 2068, 'monetary': 2069, 'unknown': 2070, 'attending': 2071, 'lion': 2072, 'attention': 2073, 'ambushed': 2074, 'increasingly': 2075, 'available': 2076, 'renewed': 2077, 'snow': 2078, 'follow': 2079, 'radical': 2080, 'differences': 2081, 'removed': 2082, 'institutions': 2083, 'agents': 2084, 'attended': 2085, 'remained': 2086, 'speaker': 2087, 'repeated': 2088, 'recovered': 2089, 'larger': 2090, 'strategy': 2091, 'game': 2092, 'reserves': 2093, 'stable': 2094, 'continent': 2095, 'delayed': 2096, 'cooperate': 2097, 'wave': 2098, 'marine': 2099, 'cash': 2100, 'surgery': 2101, 'morales': 2102, 'oppose': 2103, 'nasa': 2104, 'villagers': 2105, 'resulted': 2106, 'challenge': 2107, 'female': 2108, 'bad': 2109, 'block': 2110, 'offering': 2111, 'camps': 2112, 'uprising': 2113, 'los': 2114, 'abuses': 2115, 'presented': 2116, 'receiving': 2117, 'drove': 2118, 'roman': 2119, 'briefly': 2120, 'environment': 2121, 'pro-democracy': 2122, 'donald': 2123, 'yukos': 2124, 'miners': 2125, 'compared': 2126, 'zealand': 2127, 'sale': 2128, 'victim': 2129, 'settlers': 2130, 'banking': 2131, 'vessel': 2132, 'flight': 2133, 'cell': 2134, 'monitor': 2135, 'monitors': 2136, 'evening': 2137, 'nomination': 2138, 'separatists': 2139, 'slightly': 2140, 'kurds': 2141, 'contain': 2142, 'finished': 2143, 'improved': 2144, 'seriously': 2145, 'capacity': 2146, 'entire': 2147, 'responded': 2148, 'prompted': 2149, 'deals': 2150, 'brazilian': 2151, 'reason': 2152, 'eve': 2153, 'survived': 2154, 'admitted': 2155, 'moderate': 2156, 'christians': 2157, 'credit': 2158, 'accusations': 2159, 'accusing': 2160, 'rafik': 2161, 'above': 2162, 'observers': 2163, 'federation': 2164, 'extend': 2165, 'tiger': 2166, 'ehud': 2167, 'grew': 2168, 'blocked': 2169, 'willing': 2170, 'visits': 2171, 'transport': 2172, 'ally': 2173, 'khartoum': 2174, 'traveled': 2175, 'sell': 2176, 'smuggling': 2177, 'nobel': 2178, 'firing': 2179, 'long-term': 2180, 'battling': 2181, 'outbreaks': 2182, 'exile': 2183, 'various': 2184, 'deadliest': 2185, '2,000': 2186, '1997': 2187, 'management': 2188, 'surveillance': 2189, 'begins': 2190, 'collapsed': 2191, 'initiative': 2192, 'different': 2193, 'fighter': 2194, 'organized': 2195, 'earnings': 2196, 'know': 2197, '29': 2198, 'deployment': 2199, 'treated': 2200, 'al-zarqawi': 2201, 'tibetan': 2202, 'gathering': 2203, 'tourist': 2204, 'angeles': 2205, 'forcing': 2206, 'jakarta': 2207, 'devastated': 2208, 'fallujah': 2209, 'relatives': 2210, 'summer': 2211, 'pontiff': 2212, 'award': 2213, 'falling': 2214, 'signs': 2215, 'aziz': 2216, 'struggling': 2217, 'kuwait': 2218, 'karachi': 2219, 'stressed': 2220, 'star': 2221, 'three-day': 2222, 'influence': 2223, 'word': 2224, 'patients': 2225, 'georgian': 2226, 'revolution': 2227, 'burned': 2228, 'grand': 2229, 'suspend': 2230, 'decide': 2231, 'policeman': 2232, 'ibrahim': 2233, 'player': 2234, 'dozen': 2235, 'chance': 2236, 'protesting': 2237, 'northeast': 2238, 'developments': 2239, 'standards': 2240, 'schroeder': 2241, 'conducting': 2242, 'proposals': 2243, 'havana': 2244, 'cast': 2245, 'bases': 2246, 'single': 2247, 'euro': 2248, 'threw': 2249, 'riots': 2250, 'bureau': 2251, 'beat': 2252, 'materials': 2253, 'pilgrims': 2254, 'clashed': 2255, 'prisoner': 2256, 'staged': 2257, 'ease': 2258, '3,000': 2259, 'soil': 2260, 'traditional': 2261, 'pulled': 2262, 'marked': 2263, 'meant': 2264, 'escape': 2265, 'appointed': 2266, 'raising': 2267, 'index': 2268, 'chirac': 2269, 'effective': 2270, 'risen': 2271, 'switzerland': 2272, 'double': 2273, 'farmers': 2274, 'trafficking': 2275, 'magazine': 2276, 'kidnappings': 2277, 'commitment': 2278, 'citing': 2279, 'transferred': 2280, 'singapore': 2281, '1980s': 2282, 'lawmaker': 2283, 'musab': 2284, "shi'ites": 2285, 'driving': 2286, 'austria': 2287, 'defend': 2288, 'mccain': 2289, 'peter': 2290, 'grenade': 2291, 'marking': 2292, 'customs': 2293, 'controls': 2294, 'flying': 2295, 'pact': 2296, 'benefits': 2297, 'command': 2298, 'secretly': 2299, 'orders': 2300, 'shuttle': 2301, 'neighbors': 2302, 'rwandan': 2303, 'ice': 2304, 'thai': 2305, 'shown': 2306, 'reconciliation': 2307, 'gates': 2308, 'asking': 2309, 'capture': 2310, 'lankan': 2311, 'question': 2312, 'stormed': 2313, 'jets': 2314, 'respond': 2315, 'finding': 2316, 'muhammad': 2317, 'interests': 2318, 'malaysia': 2319, 'centuries': 2320, 'airlines': 2321, '1.5': 2322, 'strategic': 2323, 'allawi': 2324, 'phone': 2325, 'done': 2326, 'gained': 2327, 'satellite': 2328, 'provides': 2329, 'searching': 2330, 'kyrgyzstan': 2331, 'tape': 2332, 'partner': 2333, 'requested': 2334, 'polling': 2335, 'rather': 2336, '35': 2337, 'whom': 2338, 'seat': 2339, 'systems': 2340, 'reserve': 2341, 'getting': 2342, 'prepared': 2343, 'retired': 2344, 'why': 2345, 'erdogan': 2346, 'insurance': 2347, 'riot': 2348, 'typhoon': 2349, 'sets': 2350, 'signing': 2351, 'daughter': 2352, 'appealed': 2353, 'track': 2354, 'economists': 2355, 'jalal': 2356, 'memorial': 2357, 'devastating': 2358, 'frequent': 2359, '1993': 2360, 'kazakhstan': 2361, 'afghans': 2362, 'disappeared': 2363, 'ghraib': 2364, 'citizen': 2365, 'insists': 2366, 'pro-government': 2367, 'blame': 2368, 'luis': 2369, 'brown': 2370, 'cyprus': 2371, 'serving': 2372, 'crossed': 2373, 'smaller': 2374, 'expansion': 2375, 'passing': 2376, 'philippine': 2377, 'split': 2378, 'homeless': 2379, 'cartoons': 2380, 'opportunity': 2381, 'amount': 2382, 'plotting': 2383, 'magnitude': 2384, 'treasury': 2385, 'uribe': 2386, 'prisons': 2387, 'corporation': 2388, 'stronger': 2389, 'establish': 2390, 'pledge': 2391, 'starting': 2392, 'hiding': 2393, 'our': 2394, 'yemen': 2395, 'noted': 2396, 'deep': 2397, 'numerous': 2398, 'belarus': 2399, '30,000': 2400, 'seoul': 2401, 'monarchy': 2402, 'philippines': 2403, 'stage': 2404, '400': 2405, 'dominated': 2406, 'buried': 2407, 'players': 2408, 'james': 2409, 'atlantic': 2410, 'allows': 2411, 'van': 2412, 'paper': 2413, 'confirm': 2414, '19th': 2415, 'empire': 2416, 'industries': 2417, 'guerrillas': 2418, 'offshore': 2419, 'produced': 2420, 'prince': 2421, 'improving': 2422, 'rangoon': 2423, 'swiss': 2424, '/': 2425, 'negotiator': 2426, 'holds': 2427, 'restive': 2428, 'coastal': 2429, 'payments': 2430, 'ambush': 2431, 'war-torn': 2432, 'ask': 2433, 'reporter': 2434, 'fate': 2435, 'videotape': 2436, 'basic': 2437, 'got': 2438, 'commerce': 2439, 'numbers': 2440, 'swept': 2441, 'stalled': 2442, 'marks': 2443, 'president-elect': 2444, 'hospitals': 2445, 'heading': 2446, 'acting': 2447, 'humanity': 2448, 'brief': 2449, 'active': 2450, 'arafat': 2451, 'supported': 2452, 'hunger': 2453, 'shortages': 2454, 'website': 2455, 'al-maliki': 2456, 'transfer': 2457, 'suggested': 2458, 'winning': 2459, 'neither': 2460, 'basra': 2461, 'passenger': 2462, 'samples': 2463, 'driven': 2464, 'mother': 2465, 'defended': 2466, 'sarkozy': 2467, 'baluchistan': 2468, 'accepted': 2469, 'twice': 2470, 'resign': 2471, 'spying': 2472, 'cargo': 2473, 'kept': 2474, 'runoff': 2475, 'straight': 2476, 'alexander': 2477, 'powell': 2478, 'koizumi': 2479, 'colonel': 2480, 'primary': 2481, 'kathmandu': 2482, 'rome': 2483, 'lifted': 2484, 'scores': 2485, 'closely': 2486, 'temporary': 2487, 'singer': 2488, '32': 2489, 'delivered': 2490, 'chen': 2491, 'moves': 2492, 'impose': 2493, 'rebuild': 2494, 'acknowledged': 2495, 'royal': 2496, 'commissioner': 2497, 'communications': 2498, 'congolese': 2499, 'nouri': 2500, 'faction': 2501, 'embargo': 2502, 'injury': 2503, 'fish': 2504, 'producing': 2505, 'successful': 2506, 'jintao': 2507, 'coal': 2508, 'backing': 2509, 'heard': 2510, 'participate': 2511, 'incentives': 2512, 'denmark': 2513, 'discussions': 2514, 'mugabe': 2515, 'works': 2516, 'mainland': 2517, 'quit': 2518, 'gazprom': 2519, 'becoming': 2520, '600': 2521, 'required': 2522, 'determined': 2523, 'declining': 2524, 'distribution': 2525, 'advance': 2526, 'trapped': 2527, 'traffic': 2528, 'seconds': 2529, 'al-jazeera': 2530, 'closing': 2531, 'arabs': 2532, 'recover': 2533, 'preval': 2534, 'jordanian': 2535, 'grave': 2536, 'inspectors': 2537, 'temperatures': 2538, 'closer': 2539, 'festival': 2540, 'probably': 2541, 'economies': 2542, 'monitoring': 2543, 'brussels': 2544, 'nato-led': 2545, 'agenda': 2546, 'overthrow': 2547, 'hosni': 2548, 'mortar': 2549, 'leaves': 2550, 'disarm': 2551, 'shelter': 2552, 'milosevic': 2553, 'judges': 2554, 'laboratory': 2555, 'identify': 2556, 'weapon': 2557, 'external': 2558, 'shell': 2559, 'delegates': 2560, 'ramadi': 2561, 'brotherhood': 2562, 'farms': 2563, 'preliminary': 2564, 'poorest': 2565, 'easily': 2566, 'cited': 2567, 'supports': 2568, 'me': 2569, 'aide': 2570, 'society': 2571, '250': 2572, 'possibility': 2573, 'grenades': 2574, 'suspension': 2575, 'trucks': 2576, '06-mar': 2577, 'finland': 2578, 'southwest': 2579, 'setting': 2580, 'faced': 2581, 'broken': 2582, 'prayers': 2583, 'limit': 2584, 'rodriguez': 2585, 'costa': 2586, 'attempted': 2587, 'sustained': 2588, 'rescued': 2589, 'statements': 2590, 'ayatollah': 2591, 'commanders': 2592, 'settlements': 2593, 'reactor': 2594, 'welcomed': 2595, 'bordering': 2596, 'rich': 2597, 'greece': 2598, 'discovery': 2599, 'highway': 2600, 'cutting': 2601, 'directly': 2602, 'u.s.-based': 2603, 'hands': 2604, 'encourage': 2605, 'subway': 2606, '45': 2607, 'lama': 2608, 'sports': 2609, 'senegal': 2610, 'girls': 2611, 'newspapers': 2612, 'mobile': 2613, 'device': 2614, 'newly': 2615, 'telephone': 2616, 'organizers': 2617, 'bail': 2618, 'respect': 2619, 'die': 2620, 'referred': 2621, 'friends': 2622, 'transportation': 2623, 'artillery': 2624, 're-election': 2625, 'demonstration': 2626, 'story': 2627, 'look': 2628, 'prepare': 2629, 'advanced': 2630, 'speak': 2631, 'accidents': 2632, 'astronauts': 2633, 'competition': 2634, 'personal': 2635, 'jews': 2636, 'bosnia': 2637, 'museum': 2638, 'kurdistan': 2639, 'frequently': 2640, 'career': 2641, 'playing': 2642, 'researchers': 2643, 'surrounding': 2644, 'extremist': 2645, 'purposes': 2646, 'husband': 2647, 'khodorkovsky': 2648, 'reasons': 2649, 'english': 2650, 'ankara': 2651, 'nairobi': 2652, 'images': 2653, 'panama': 2654, 'fresh': 2655, 'politics': 2656, 'rescuers': 2657, 'nicaragua': 2658, 'upper': 2659, 'strength': 2660, 'enemy': 2661, 'denounced': 2662, 'prevented': 2663, 'liberia': 2664, 'seeing': 2665, 'socialist': 2666, 'lukashenko': 2667, 'confirmation': 2668, 'manmohan': 2669, 'gives': 2670, 'pullout': 2671, 'route': 2672, 'speed': 2673, 'computer': 2674, 'factory': 2675, 'relatively': 2676, 'swat': 2677, 'mullah': 2678, '5,000': 2679, 'ahmad': 2680, 'teachers': 2681, 'rape': 2682, 'mines': 2683, 'underground': 2684, 'inmates': 2685, 'politician': 2686, 'negotiate': 2687, 'indicated': 2688, 'geneva': 2689, 'uzbekistan': 2690, 'launching': 2691, 'practice': 2692, 'reducing': 2693, 'changed': 2694, 'apparent': 2695, 'fears': 2696, 'replaced': 2697, 'sharp': 2698, 'selling': 2699, 'creation': 2700, 'kiev': 2701, 'plants': 2702, 'khatami': 2703, 'draw': 2704, 'subsidies': 2705, 'lived': 2706, 'rock': 2707, 'student': 2708, 'experienced': 2709, 'gang': 2710, 'declaration': 2711, 'dependent': 2712, 'counter': 2713, 'fallen': 2714, 'eventually': 2715, 'richard': 2716, 'focused': 2717, 'contracted': 2718, 'big': 2719, '36': 2720, '06-apr': 2721, 'suspicion': 2722, 'records': 2723, 'initially': 2724, 'croatian': 2725, 'imported': 2726, '?': 2727, 'overseas': 2728, 'grow': 2729, 'honor': 2730, 'jacques': 2731, 'regulations': 2732, 'cold': 2733, 'vaccine': 2734, 'gunfire': 2735, 'landed': 2736, 'wild': 2737, 'republicans': 2738, 'mali': 2739, 'remove': 2740, 'warming': 2741, 'asylum': 2742, 'causes': 2743, '120': 2744, 'libya': 2745, 'restored': 2746, 'questioning': 2747, 'two-thirds': 2748, 'net': 2749, 'blow': 2750, 'celebrations': 2751, 'born': 2752, 'nor': 2753, 'transitional': 2754, 'unharmed': 2755, 'foundation': 2756, 'roughly': 2757, 'disputes': 2758, 'ismail': 2759, 'withdrawing': 2760, 'wake': 2761, 'nominee': 2762, 'handling': 2763, 'questions': 2764, 'spreading': 2765, 'investments': 2766, 'zabul': 2767, 'violating': 2768, 'sergei': 2769, 'sugar': 2770, 'bosnia-herzegovina': 2771, 'complex': 2772, 'hostile': 2773, 'usually': 2774, 'mountainous': 2775, 'joseph': 2776, 'anti-government': 2777, 'angry': 2778, 'photographs': 2779, 'relationship': 2780, 'deliver': 2781, 'uruzgan': 2782, 'ramadan': 2783, 'implement': 2784, 'native': 2785, 'ripped': 2786, 'rare': 2787, 'state-owned': 2788, 'indicates': 2789, 'courts': 2790, 'alert': 2791, 'testing': 2792, 'illness': 2793, 'drought': 2794, 'shops': 2795, 'save': 2796, 'exercise': 2797, '1992': 2798, 'appeals': 2799, 'hall': 2800, 'retirement': 2801, 'tv': 2802, 'basis': 2803, 'loans': 2804, 'eritrea': 2805, 'sentences': 2806, 'originally': 2807, 'salvador': 2808, 'plunged': 2809, '1,00,000': 2810, 'stands': 2811, 'kibaki': 2812, 'require': 2813, 'grant': 2814, 'chilean': 2815, 'anti-terrorism': 2816, 'clerics': 2817, 'complaints': 2818, 'resolved': 2819, 'partnership': 2820, '800': 2821, 'districts': 2822, 'raul': 2823, 'assistant': 2824, 'lieutenant': 2825, 'pursue': 2826, 'slogans': 2827, 'attacking': 2828, 'extradition': 2829, 'ability': 2830, 'charter': 2831, 'brigades': 2832, 'expanded': 2833, '700': 2834, 'title': 2835, 'executed': 2836, 'centered': 2837, 'dealing': 2838, 'celebrate': 2839, 'televised': 2840, 'barred': 2841, 'hijacked': 2842, 'dick': 2843, 'prosecution': 2844, 'swedish': 2845, 'location': 2846, 'reduction': 2847, 'contributed': 2848, 'gone': 2849, 'paramilitary': 2850, 'arriving': 2851, 'migrants': 2852, 'wide': 2853, 'dalai': 2854, 'pandemic': 2855, 'movie': 2856, 'abandon': 2857, 'aired': 2858, 'channel': 2859, 'angola': 2860, 'william': 2861, 'el': 2862, 'jet': 2863, 'destroy': 2864, 'donor': 2865, 'stroke': 2866, 'transition': 2867, 'run-off': 2868, 'surge': 2869, 'qatar': 2870, 'airstrikes': 2871, 'territorial': 2872, 'surrounded': 2873, 'cnn': 2874, 'breakaway': 2875, 'stem': 2876, 'forming': 2877, 'success': 2878, 'couple': 2879, 'invited': 2880, 'tight': 2881, 'charles': 2882, 'lay': 2883, 'mikhail': 2884, 'agent': 2885, 'promoting': 2886, 'attacker': 2887, 'settled': 2888, 'introduced': 2889, 'cover': 2890, 'size': 2891, 'volatile': 2892, 'gun': 2893, 'al-assad': 2894, 'discussing': 2895, 'pro-syrian': 2896, 'ramallah': 2897, 'gyanendra': 2898, 'rounds': 2899, 'drive': 2900, 'places': 2901, 'aboard': 2902, 'massacre': 2903, 'hospitalized': 2904, 'detected': 2905, 'st.': 2906, 'injuring': 2907, 'gangs': 2908, 'kunar': 2909, 'togo': 2910, 'replied': 2911, 'seventh': 2912, 'six-party': 2913, 'substantial': 2914, 'martyrs': 2915, 'considers': 2916, 'pictures': 2917, 'requires': 2918, 'carlos': 2919, 'kennedy': 2920, 'citizenship': 2921, 'crops': 2922, 'combined': 2923, 'extensive': 2924, 'flew': 2925, 'ass': 2926, 'lift': 2927, 'putting': 2928, 'advisor': 2929, 'fast': 2930, 'bloody': 2931, 'crowded': 2932, 'successfully': 2933, '06-feb': 2934, 'martin': 2935, 'evacuated': 2936, 'green': 2937, 'loyal': 2938, 'khost': 2939, 'affect': 2940, 'boeing': 2941, 'prior': 2942, 'eta': 2943, 'participation': 2944, '20th': 2945, 'norwegian': 2946, 'creating': 2947, 'bashar': 2948, 'polio': 2949, 'ceremonies': 2950, 'macedonia': 2951, 'rural': 2952, 'airstrike': 2953, 'animal': 2954, 'assumed': 2955, 'wealth': 2956, 'owned': 2957, 'entry': 2958, 'emissions': 2959, 'bali': 2960, 'drone': 2961, 'violation': 2962, 'achieved': 2963, 'afternoon': 2964, 'hundred': 2965, 'protested': 2966, 'historic': 2967, 'buddhist': 2968, 'waves': 2969, 'entering': 2970, 'permission': 2971, 'unions': 2972, 'pull': 2973, 'beating': 2974, 'speculation': 2975, 'appearance': 2976, 'operated': 2977, 'firms': 2978, 'industrialized': 2979, 'occupation': 2980, 'warplanes': 2981, 'false': 2982, 'disrupt': 2983, 'sheikh': 2984, 'torch': 2985, 'transit': 2986, 'institute': 2987, 'colony': 2988, 'finally': 2989, 'opposes': 2990, 'bodyguards': 2991, 'pressing': 2992, 'junta': 2993, 'engineers': 2994, 'israeli-palestinian': 2995, 'baquba': 2996, 'photos': 2997, 'five-year': 2998, 'oil-rich': 2999, 'belgium': 3000, 'names': 3001, 'alive': 3002, 'legislature': 3003, 'items': 3004, 'coordinated': 3005, 'troop': 3006, 'boats': 3007, 'latvia': 3008, 'proceedings': 3009, 'uzbek': 3010, 'outgoing': 3011, 'elbaradei': 3012, 'closure': 3013, 'primarily': 3014, 'blockade': 3015, 'mudslides': 3016, 'afp': 3017, 'engaged': 3018, 'kremlin': 3019, 'hutu': 3020, 'publicly': 3021, 'ghana': 3022, 'announce': 3023, 'consulate': 3024, 'alvaro': 3025, 'reaching': 3026, 'intense': 3027, 'hill': 3028, 'scott': 3029, 'shrine': 3030, 'duties': 3031, 'consumers': 3032, 'malawi': 3033, 'book': 3034, 'fujimori': 3035, 'chechen': 3036, 'guatemala': 3037, 'editor': 3038, 'minor': 3039, 'birth': 3040, 'blames': 3041, 'talk': 3042, 'pinochet': 3043, 'influential': 3044, 'madrid': 3045, 'obtained': 3046, 'cleared': 3047, 'capita': 3048, 'contractors': 3049, 'strained': 3050, 'goals': 3051, 'netanyahu': 3052, 'straw': 3053, 'surrender': 3054, '17th': 3055, 'informed': 3056, 'slowly': 3057, '20,000': 3058, 'shootout': 3059, 'cambodia': 3060, 'gains': 3061, '16th': 3062, 'written': 3063, 'apartment': 3064, 'generally': 3065, 'merger': 3066, 'claiming': 3067, 'ambitions': 3068, 'withdrawn': 3069, 'wing': 3070, 'subject': 3071, 'freeze': 3072, 'unable': 3073, 'sexual': 3074, 'busy': 3075, '34': 3076, 'koreans': 3077, 'partners': 3078, 'embassies': 3079, 'fields': 3080, 'silva': 3081, 'performance': 3082, 'conflicts': 3083, 'specific': 3084, 'recorded': 3085, '33': 3086, 'limits': 3087, 'inquiry': 3088, 'gnassingbe': 3089, '75': 3090, 'saint': 3091, 'complained': 3092, 'uncovered': 3093, 'deny': 3094, 'extending': 3095, 'portugal': 3096, 'burning': 3097, 'visitors': 3098, 'penalty': 3099, 'pakistanis': 3100, 'vicente': 3101, 'telecommunications': 3102, 'cricket': 3103, 'registered': 3104, 'sixth': 3105, 'boycotted': 3106, 'anbar': 3107, 'google': 3108, 'imprisoned': 3109, 'instability': 3110, 'accords': 3111, 'standoff': 3112, 'string': 3113, 'feared': 3114, 'investigated': 3115, 'mandate': 3116, 'wearing': 3117, 'regular': 3118, 'mountain': 3119, 'eagle': 3120, '!': 3121, 'ran': 3122, 'billions': 3123, 'succeed': 3124, 'controlled': 3125, 'fierce': 3126, 'charity': 3127, 'depends': 3128, 'solidarity': 3129, 'demonstrated': 3130, 'ghazni': 3131, 'azerbaijan': 3132, 'solution': 3133, 'protecting': 3134, 'rebuilding': 3135, 'boosting': 3136, 'moon': 3137, 'senators': 3138, 'politically': 3139, '3,00,000': 3140, 'property': 3141, 'devices': 3142, 'deposed': 3143, 'normal': 3144, 'eligible': 3145, 'chechnya': 3146, 'soccer': 3147, 'da': 3148, 'remittances': 3149, 'arrive': 3150, 'calm': 3151, 'flow': 3152, 'reject': 3153, 'thomas': 3154, 'yugoslav': 3155, 'zardari': 3156, 'hoping': 3157, 'parents': 3158, 'nominated': 3159, 'endorsed': 3160, 'angela': 3161, '52': 3162, 'rivals': 3163, 'forum': 3164, 'criminals': 3165, 'large-scale': 3166, 'mumbai': 3167, 'encouraged': 3168, 'denying': 3169, 'explosive': 3170, 'successor': 3171, 'medvedev': 3172, 'section': 3173, 'captain': 3174, 'drc': 3175, 'al-aqsa': 3176, 'profits': 3177, 'tymoshenko': 3178, 'managed': 3179, 'recep': 3180, 'tayyip': 3181, 'sworn': 3182, 'messages': 3183, 'sensitive': 3184, 'preventing': 3185, 'destroying': 3186, 'gap': 3187, 'hideout': 3188, 'defendants': 3189, 'predicted': 3190, 'trials': 3191, 'adding': 3192, 'annually': 3193, 'possibly': 3194, 'congressman': 3195, 'conspiracy': 3196, 'jury': 3197, 'allied': 3198, 'surrendered': 3199, 'employment': 3200, 'purchase': 3201, 'boys': 3202, 'louisiana': 3203, 'turin': 3204, 'existing': 3205, 'college': 3206, 'rapidly': 3207, 'centimeters': 3208, 'activist': 3209, 'burundi': 3210, 'map': 3211, '4,000': 3212, 'increases': 3213, 'howard': 3214, 'stadium': 3215, '2012': 3216, 'ireland': 3217, 'rica': 3218, 'assailants': 3219, 'persian': 3220, '55': 3221, 'concluded': 3222, 'joining': 3223, 'compensation': 3224, 'lee': 3225, 'packed': 3226, 'completely': 3227, 'extremely': 3228, 'cypriot': 3229, 'garang': 3230, '7,000': 3231, 'brings': 3232, 'linking': 3233, 'insisted': 3234, 'cross-border': 3235, 'exporting': 3236, 'trained': 3237, 'view': 3238, 'club': 3239, 'clean': 3240, 'downturn': 3241, 'units': 3242, 'sure': 3243, 'experience': 3244, 'schedule': 3245, '5,00,000': 3246, 'individuals': 3247, 'mars': 3248, 'mauritania': 3249, 'technical': 3250, 'airline': 3251, 'compete': 3252, 'sound': 3253, 'angered': 3254, 'uruguay': 3255, 'saakashvili': 3256, '8,000': 3257, 'estimate': 3258, 'scale': 3259, 'waiting': 3260, 'addressed': 3261, 'peres': 3262, 'alone': 3263, 'hunt': 3264, 'link': 3265, 'acts': 3266, 'diseases': 3267, 'voter': 3268, 'subsistence': 3269, 'tell': 3270, 'stopping': 3271, 'maintained': 3272, 'kashmiri': 3273, 'park': 3274, 'division': 3275, 'christopher': 3276, 'operate': 3277, 'recognized': 3278, 'blocking': 3279, 'count': 3280, 'williams': 3281, '140': 3282, 'comprehensive': 3283, 'science': 3284, 'berlin': 3285, 'hamas-led': 3286, 'textile': 3287, 'dr.': 3288, 'manuel': 3289, 'rejecting': 3290, 'impoverished': 3291, 'exercises': 3292, 'choose': 3293, 'wrong': 3294, 'non-proliferation': 3295, '1988': 3296, 'opium': 3297, 'directed': 3298, 'commented': 3299, 'testimony': 3300, 'witness': 3301, 'celebrated': 3302, 'vast': 3303, 'violate': 3304, 'installed': 3305, 'ivanov': 3306, 'abandoned': 3307, 'gunbattle': 3308, 'flee': 3309, 'fifa': 3310, 'minute': 3311, 'governing': 3312, 'enforcement': 3313, 'fellow': 3314, 'seed': 3315, 'unveiled': 3316, 'submitted': 3317, '&': 3318, 'dissidents': 3319, 'weak': 3320, 'port-au-prince': 3321, 'chaos': 3322, 'course': 3323, 'jackson': 3324, 'turning': 3325, 'smith': 3326, 'underwent': 3327, 'belonging': 3328, 'amendment': 3329, 'broad': 3330, 'loan': 3331, 'carroll': 3332, 'algerian': 3333, 'founded': 3334, 'danger': 3335, 'hearings': 3336, 'tough': 3337, 'timetable': 3338, 'disrupted': 3339, 'drew': 3340, 'always': 3341, 'worried': 3342, 'enriched': 3343, 'none': 3344, 'walked': 3345, 'refineries': 3346, 'wrote': 3347, 'irish': 3348, 'vulnerable': 3349, 'earned': 3350, 'picked': 3351, 'motivated': 3352, 'harsh': 3353, 'ammunition': 3354, 'goes': 3355, 'motion': 3356, 'image': 3357, 'pursuing': 3358, 'infection': 3359, 'ayman': 3360, 'disarmament': 3361, 'libyan': 3362, 'irregularities': 3363, 'curb': 3364, 'contribute': 3365, 'quote': 3366, 'gbagbo': 3367, 'alternative': 3368, 'findings': 3369, 'execution': 3370, '1979': 3371, 'planted': 3372, 'crucial': 3373, 'exploration': 3374, 'tikrit': 3375, 'refinery': 3376, 'subsequent': 3377, 'ranch': 3378, 'meets': 3379, 'extension': 3380, 'drinking': 3381, 'article': 3382, 'flooded': 3383, 'fires': 3384, 'aides': 3385, 'soaring': 3386, 'electronic': 3387, 'tanzania': 3388, 'spy': 3389, 'unnamed': 3390, 'disasters': 3391, 'likud': 3392, 'relay': 3393, 'pilot': 3394, 'warns': 3395, 'counted': 3396, 'nature': 3397, 'culture': 3398, 'moroccan': 3399, 'effects': 3400, 'trust': 3401, 'enclave': 3402, 'u.s': 3403, 'pace': 3404, 'ends': 3405, 'tom': 3406, 'indictment': 3407, 'beyond': 3408, 'hard-line': 3409, 'journal': 3410, 'srebrenica': 3411, 'slowed': 3412, 'overwhelmingly': 3413, 'bulgaria': 3414, 'bought': 3415, 'del': 3416, 'surface': 3417, 'modern': 3418, 'approach': 3419, 'standard': 3420, 'prospects': 3421, '1967': 3422, 'standing': 3423, 'foot': 3424, 'band': 3425, 'churches': 3426, 'romania': 3427, 'gerhard': 3428, 'missions': 3429, 'idea': 3430, '1984': 3431, 'mrs.': 3432, 'tradition': 3433, 'hotels': 3434, 'moment': 3435, 'returns': 3436, 'mortars': 3437, 'al-qaida-linked': 3438, 'advantage': 3439, 'exporter': 3440, 'expanding': 3441, 'inadequate': 3442, 'tunnel': 3443, 'aviation': 3444, 'eating': 3445, 'strengthening': 3446, 'class': 3447, 'counts': 3448, 'carolina': 3449, 'rubble': 3450, 'retaliation': 3451, 'shortage': 3452, 'bombed': 3453, 'attempting': 3454, 'generals': 3455, 'resolving': 3456, 'tanks': 3457, 'judicial': 3458, 'weekend': 3459, 'planet': 3460, 'thaksin': 3461, 'balance': 3462, 'leg': 3463, 'integration': 3464, 'ring': 3465, 'rallied': 3466, 'views': 3467, 'obasanjo': 3468, 'identity': 3469, 'processing': 3470, 'corps': 3471, 'nicolas': 3472, 'paying': 3473, 'sichuan': 3474, 'indigenous': 3475, 'formerly': 3476, 'microsoft': 3477, 'store': 3478, 'counter-terrorism': 3479, 'peruvian': 3480, 'leads': 3481, 'innocent': 3482, 'multiple': 3483, 'refusal': 3484, 'refer': 3485, 'interrogation': 3486, 'administrative': 3487, 'represent': 3488, 'abkhazia': 3489, 'mistake': 3490, 'junichiro': 3491, 'residence': 3492, 'highs': 3493, 'municipal': 3494, 'trouble': 3495, 'options': 3496, 'jack': 3497, 'europeans': 3498, 'pop': 3499, 'slovakia': 3500, 'surprise': 3501, 'nepalese': 3502, 'iyad': 3503, 'read': 3504, '1982': 3505, 'language': 3506, 'warn': 3507, 'climbed': 3508, 'art': 3509, 'portion': 3510, 'clearing': 3511, 'holocaust': 3512, 'himalayan': 3513, 'mount': 3514, 'colombo': 3515, 'hitting': 3516, 'path': 3517, 'animals': 3518, 'mixed': 3519, 'attracted': 3520, 'istanbul': 3521, 'unacceptable': 3522, 'grown': 3523, 'rallies': 3524, 'importance': 3525, 'vessels': 3526, 'extra': 3527, '1970s': 3528, 'fugitive': 3529, 'java': 3530, 'broadcasting': 3531, 'underway': 3532, 'hoped': 3533, '1960': 3534, 'hunting': 3535, 'expires': 3536, 'posts': 3537, 'keeping': 3538, 'guerrilla': 3539, 'sun': 3540, 'shots': 3541, 'youth': 3542, '38': 3543, 'wto': 3544, 'contained': 3545, 'sex': 3546, 'bridge': 3547, 'one-day': 3548, 'dubai': 3549, 'rapid': 3550, 'significantly': 3551, 'doctor': 3552, 'lagos': 3553, 'fbi': 3554, 'six-nation': 3555, 'lopez': 3556, 'reflect': 3557, 'longtime': 3558, 'malaria': 3559, 'subsequently': 3560, 'recognition': 3561, 'someone': 3562, 'high-level': 3563, 'belarusian': 3564, 'consecutive': 3565, 'famous': 3566, 'guns': 3567, 'sometimes': 3568, 'donations': 3569, 'tied': 3570, 'spend': 3571, 'payment': 3572, 'deeply': 3573, 'al-jaafari': 3574, 'parade': 3575, 'alberto': 3576, 'upset': 3577, 'austrian': 3578, 'compromise': 3579, 'karbala': 3580, 'revealed': 3581, 'eid': 3582, 'generate': 3583, 'decisions': 3584, 'shared': 3585, 'slowdown': 3586, 'yugoslavia': 3587, 'squad': 3588, 'notes': 3589, '37': 3590, 'revive': 3591, 'emerging': 3592, 'flawed': 3593, 'u.s.-backed': 3594, 'awarded': 3595, 'hurricanes': 3596, 'medicine': 3597, 'hillary': 3598, 'undermine': 3599, 'original': 3600, 'guinea': 3601, 'intention': 3602, 'thanked': 3603, 'authorized': 3604, 'balloting': 3605, '1,300': 3606, 'brain': 3607, 'sidelines': 3608, 'responding': 3609, 'aggressive': 3610, 'lahoud': 3611, 'cartel': 3612, 'jean-bertrand': 3613, 'missed': 3614, 'ki-moon': 3615, 'worse': 3616, 'secular': 3617, 'solana': 3618, 'cubans': 3619, 'bolivian': 3620, 'veto': 3621, 'alito': 3622, 'structure': 3623, 'dam': 3624, 'spiritual': 3625, 'looks': 3626, 'teenager': 3627, 'jesus': 3628, 'fed': 3629, 'warlords': 3630, 'mcclellan': 3631, 'grounds': 3632, '2.5': 3633, 'livestock': 3634, 'stimulus': 3635, 'swine': 3636, 'arizona': 3637, 'taylor': 3638, 'debris': 3639, 'farmer': 3640, 'mounting': 3641, 'rigged': 3642, 'boosted': 3643, '41': 3644, 'chosen': 3645, 'argentine': 3646, 'arrival': 3647, 'anger': 3648, 'involves': 3649, 'fly': 3650, 'soared': 3651, 'evasion': 3652, 'emerged': 3653, 'carter': 3654, 'choice': 3655, 'bahrain': 3656, 'oil-for-food': 3657, 'priority': 3658, 'colin': 3659, 'picture': 3660, 'focusing': 3661, 'maoists': 3662, 'atrocities': 3663, 'srinagar': 3664, 'gunned': 3665, 'armored': 3666, 'motorcycle': 3667, 'easier': 3668, 'palace': 3669, 'shah': 3670, 'visa': 3671, 'sunnis': 3672, 'unprecedented': 3673, 'mourning': 3674, 'ports': 3675, 'gunman': 3676, 'strict': 3677, 'aside': 3678, 'opportunities': 3679, 'regarding': 3680, 'refuge': 3681, 'qureia': 3682, 'tells': 3683, 'iranians': 3684, 'damaging': 3685, 'mottaki': 3686, 'option': 3687, 'antonio': 3688, 'broadcasts': 3689, 'computers': 3690, 'maritime': 3691, 'desire': 3692, 'critic': 3693, 'discrimination': 3694, 'miami': 3695, 'wfp': 3696, 'covered': 3697, 'farming': 3698, 'plagued': 3699, 'contributions': 3700, 'mountains': 3701, 'neighbor': 3702, 'representing': 3703, 'criticizing': 3704, 'abc': 3705, 'outskirts': 3706, 'recommended': 3707, 'tanker': 3708, 'designated': 3709, 'maximum': 3710, 'profit': 3711, '18th': 3712, 'serbs': 3713, 'succeeded': 3714, 'friend': 3715, 'inauguration': 3716, 'welcome': 3717, 'cocaine': 3718, 'unification': 3719, 'peshawar': 3720, 'am': 3721, 'triggering': 3722, 'motors': 3723, 'benefit': 3724, '95': 3725, 'landing': 3726, 'warrant': 3727, 'algeria': 3728, 'here': 3729, 'promise': 3730, 'intensified': 3731, 'fine': 3732, 'orthodox': 3733, 'selected': 3734, 'baseball': 3735, 'hungary': 3736, 'landslide': 3737, 'outlawed': 3738, 'stocks': 3739, 'coverage': 3740, 'elaborate': 3741, 'physical': 3742, 'species': 3743, 'quetta': 3744, 'canal': 3745, 'sean': 3746, 'correspondent': 3747, 'formation': 3748, 'assist': 3749, 'county': 3750, 'married': 3751, 'belgrade': 3752, 'power-sharing': 3753, 'shopping': 3754, 'djibouti': 3755, 'think': 3756, 'rushed': 3757, 'procedure': 3758, 'tragedy': 3759, 'concert': 3760, 'networks': 3761, 'religion': 3762, 'liberties': 3763, 'detainee': 3764, 'comply': 3765, 'harm': 3766, 'stranded': 3767, 'bbc': 3768, 'abduction': 3769, 'arabic': 3770, 'unmanned': 3771, 'dropping': 3772, 'phase': 3773, 'promises': 3774, 'fleeing': 3775, 'nablus': 3776, 'warnings': 3777, 'percentage': 3778, 'storms': 3779, 'plea': 3780, 'flown': 3781, 'vital': 3782, 'marred': 3783, 'dissolved': 3784, 'whole': 3785, 'things': 3786, 'nazi': 3787, 'government-backed': 3788, 'male': 3789, 'hopeful': 3790, '160': 3791, 'clothing': 3792, 'militiamen': 3793, 'al-sadr': 3794, 'stabilize': 3795, 'traffickers': 3796, 'urge': 3797, 'lithuania': 3798, 'cape': 3799, 'bound': 3800, '1.3': 3801, 'margin': 3802, 'lowest': 3803, 'montenegro': 3804, 'cocoa': 3805, 'shipping': 3806, 'andres': 3807, 'represents': 3808, 'rita': 3809, 'fatalities': 3810, 'harry': 3811, 'stores': 3812, 'observed': 3813, 'ten': 3814, 'kind': 3815, 'bhutto': 3816, 'prompting': 3817, 'airports': 3818, 'reverse': 3819, 'zoellick': 3820, 'ones': 3821, 'fence': 3822, 'defending': 3823, 'one-third': 3824, 'ouster': 3825, 'spill': 3826, 'isolation': 3827, 'rebellion': 3828, 'privatization': 3829, 'room': 3830, 'toxic': 3831, 'wounds': 3832, 'weah': 3833, 'reactors': 3834, 'tank': 3835, 'atmosphere': 3836, 'frontier': 3837, 'flood': 3838, 'lawsuit': 3839, 'daniel': 3840, 'two-year': 3841, 'fever': 3842, 'surged': 3843, 'dog': 3844, 'taipei': 3845, 'fail': 3846, 'approximately': 3847, 'fleet': 3848, 'argued': 3849, 'sick': 3850, 'koreas': 3851, 'khamenei': 3852, 'antarctic': 3853, 'recruiting': 3854, 'crowds': 3855, 'meat': 3856, 'wolf': 3857, 'jan': 3858, 'voiced': 3859, 'task': 3860, 'everything': 3861, 'overwhelming': 3862, 'millennium': 3863, 'pledges': 3864, 'narrow': 3865, 'attributed': 3866, 'piece': 3867, 'campaigning': 3868, 'aims': 3869, 'three-year': 3870, 'lose': 3871, 'haniyeh': 3872, 'chicken': 3873, 'alongside': 3874, 'benjamin': 3875, 'spring': 3876, 'renew': 3877, 'maintaining': 3878, 'lavrov': 3879, 'rafah': 3880, 'motive': 3881, 'scientific': 3882, 'juan': 3883, 'mississippi': 3884, 'isolated': 3885, 'wait': 3886, 'code': 3887, 'sparking': 3888, 'badly': 3889, 'freezing': 3890, 'delays': 3891, 'hear': 3892, 'values': 3893, 'punish': 3894, 'ill': 3895, '1949': 3896, 'song': 3897, 'suspicious': 3898, 'mediterranean': 3899, 'shipments': 3900, 'quotas': 3901, 'counterparts': 3902, 'import': 3903, 'capable': 3904, 'medal': 3905, 'aim': 3906, 'silence': 3907, 'abdel': 3908, 'al-zawahiri': 3909, 'suffer': 3910, 'potentially': 3911, 'variety': 3912, 'purported': 3913, 'roof': 3914, 'urban': 3915, 'coordination': 3916, 'deby': 3917, 'oust': 3918, 'traders': 3919, '64': 3920, 'threaten': 3921, 'deploy': 3922, 'aftermath': 3923, 'turmoil': 3924, 'volunteers': 3925, 'symptoms': 3926, 'followers': 3927, 'strengthened': 3928, 'buses': 3929, 'reza': 3930, 'noting': 3931, 'businessman': 3932, 'scientist': 3933, 'seizure': 3934, 'employee': 3935, 'restructuring': 3936, 'exiled': 3937, 'relative': 3938, 'violated': 3939, 'vietnamese': 3940, 'museveni': 3941, 'landmark': 3942, 'evo': 3943, 'apart': 3944, 'colleagues': 3945, 'legislators': 3946, 'prayer': 3947, 'reward': 3948, 'reid': 3949, 'fatal': 3950, 'elders': 3951, 'elect': 3952, 'trees': 3953, 'sovereign': 3954, 'threatens': 3955, 'democrat': 3956, 'resulting': 3957, 'shan': 3958, 'passage': 3959, 'wage': 3960, 'severely': 3961, 'honduras': 3962, 'herald': 3963, 'pollution': 3964, 'license': 3965, 'cyclone': 3966, 'bangladeshi': 3967, 'halted': 3968, 'burden': 3969, 'seeks': 3970, 'beef': 3971, 'marriage': 3972, 'tension': 3973, 'bolton': 3974, 'lahore': 3975, 'ancient': 3976, 'drawn': 3977, 'jones': 3978, 'gonzales': 3979, 'behalf': 3980, 'awards': 3981, '50,000': 3982, 'achieve': 3983, 'spot': 3984, 'founder': 3985, 'benin': 3986, 'equal': 3987, 'wrongdoing': 3988, 'youths': 3989, 'restaurant': 3990, 'rene': 3991, 'contractor': 3992, 'travelers': 3993, 'sahara': 3994, 'islamists': 3995, 'rashid': 3996, 'dinner': 3997, 'zarqawi': 3998, 'diplomacy': 3999, 'carrier': 4000, 'version': 4001, 'disperse': 4002, 'competing': 4003, 'zapatero': 4004, 'corp.': 4005, 'commonwealth': 4006, 'brutal': 4007, 'implemented': 4008, '1974': 4009, 'hemisphere': 4010, 'yasser': 4011, 'root': 4012, 'committing': 4013, 'maker': 4014, 'intends': 4015, 'rivers': 4016, 'advocates': 4017, 'upsurge': 4018, 'consequences': 4019, 'blaming': 4020, 'liberian': 4021, 'diyala': 4022, '6,000': 4023, 'paktika': 4024, 'onto': 4025, 'absolute': 4026, 'starts': 4027, 'opponent': 4028, 'weakened': 4029, 'gadhafi': 4030, 'feed': 4031, 'abusing': 4032, 'routes': 4033, 'addressing': 4034, 'practices': 4035, 'co-defendants': 4036, 'murders': 4037, 'tel': 4038, 'heroin': 4039, 'offenses': 4040, 'cable': 4041, 'carbon': 4042, 'shanghai': 4043, 'chadian': 4044, 'writing': 4045, 'condolences': 4046, 'fails': 4047, 'airspace': 4048, 'films': 4049, 'wreckage': 4050, 'felipe': 4051, 'doubt': 4052, 'conviction': 4053, 'amendments': 4054, 'balad': 4055, 'propaganda': 4056, 'regularly': 4057, 'declare': 4058, 'hunter': 4059, 'counting': 4060, 'factories': 4061, 'damages': 4062, 'contingent': 4063, 'shwe': 4064, 'covering': 4065, 'battles': 4066, 'sydney': 4067, 'extradited': 4068, 'laid': 4069, 'toppled': 4070, 'larijani': 4071, 'confiscated': 4072, 'outstanding': 4073, 'possession': 4074, 'overcome': 4075, 'melbourne': 4076, 'mentioned': 4077, 'long-running': 4078, 'mean': 4079, 'mediators': 4080, 'particular': 4081, 'karen': 4082, 'referring': 4083, 'cultural': 4084, 'greenspan': 4085, 'regulators': 4086, '42': 4087, 'tribesmen': 4088, 'everyone': 4089, 'yemeni': 4090, 'acted': 4091, 'contracts': 4092, 'paraguay': 4093, 'maliki': 4094, 'conversion': 4095, 'tree': 4096, 'pick': 4097, 'executives': 4098, 'interrogators': 4099, '130': 4100, 'thanksgiving': 4101, 'jumped': 4102, 'evacuation': 4103, 'professional': 4104, 'romanian': 4105, 'owner': 4106, 'monsoon': 4107, 'individual': 4108, 'retail': 4109, 'ford': 4110, 'aden': 4111, 'assured': 4112, 'a.u.': 4113, 'cattle': 4114, 'finish': 4115, 'lifting': 4116, 'high-ranking': 4117, 'recommendations': 4118, 'engage': 4119, 'participants': 4120, 'ballot': 4121, 'nationalist': 4122, 'healthy': 4123, 'diagnosed': 4124, 'understanding': 4125, 'offers': 4126, 'prosperous': 4127, '53': 4128, 'awareness': 4129, 'ballistic': 4130, 'machine': 4131, '11th': 4132, 'ossetia': 4133, 'brigadier': 4134, 'pose': 4135, 'faith': 4136, 'studies': 4137, 'type': 4138, 'specifically': 4139, 'championships': 4140, 'matches': 4141, 'coordinate': 4142, 'restoring': 4143, 'seize': 4144, 'forecast': 4145, 'gather': 4146, 'notorious': 4147, 'founding': 4148, 'peacefully': 4149, 'checkpoints': 4150, 'herat': 4151, 'worshippers': 4152, 'investigations': 4153, 'provisions': 4154, 'crack': 4155, 'curfew': 4156, 'basque': 4157, 'classified': 4158, 'granting': 4159, '1976': 4160, '85': 4161, 'shootings': 4162, 'davis': 4163, 'lands': 4164, 'letters': 4165, 'producers': 4166, 'hampered': 4167, 'solar': 4168, 'beneath': 4169, 'kabila': 4170, 'ignored': 4171, 'dumped': 4172, 'one-year': 4173, '65': 4174, 'luxembourg': 4175, 'fans': 4176, 'laureate': 4177, 'inacio': 4178, 'lula': 4179, 'renounce': 4180, 'petition': 4181, 'barrier': 4182, 'except': 4183, 'africans': 4184, 'organizing': 4185, 'deteriorating': 4186, 'aviv': 4187, 'preparations': 4188, 'scotland': 4189, 'abused': 4190, 'branch': 4191, 'non-governmental': 4192, 'quick': 4193, 'dramatically': 4194, 'partial': 4195, 'slain': 4196, 'express': 4197, 'mourners': 4198, 'easing': 4199, 'assassinated': 4200, 'appointment': 4201, 'calderon': 4202, 'virginia': 4203, 'transporting': 4204, 'asif': 4205, 'birthday': 4206, 'attached': 4207, 'participating': 4208, '1,500': 4209, 'uses': 4210, 'influenza': 4211, 'note': 4212, 'escalating': 4213, 'hailed': 4214, 'kuchma': 4215, 'suburb': 4216, 'trains': 4217, 'internationally': 4218, 'repair': 4219, 'rating': 4220, 'archbishop': 4221, 'audience': 4222, 'register': 4223, 'appoint': 4224, 'anderson': 4225, 'patrols': 4226, 'provisional': 4227, 'users': 4228, 'challenged': 4229, 'extremism': 4230, 'communication': 4231, 'something': 4232, 'aware': 4233, 'dry': 4234, 'owners': 4235, 'fueled': 4236, 'd.c.': 4237, 'recalled': 4238, 'luxury': 4239, 'actually': 4240, 'estate': 4241, 'behavior': 4242, 'ducks': 4243, 'exposed': 4244, 'emirates': 4245, 'kidnap': 4246, 'culled': 4247, 'kevin': 4248, 'high-tech': 4249, 'ramirez': 4250, 'tunnels': 4251, 'admiral': 4252, 'dependence': 4253, '15,000': 4254, 'buying': 4255, 'manila': 4256, 'asean': 4257, 'dismantle': 4258, 'abortion': 4259, 'orakzai': 4260, 'greatest': 4261, 'mehlis': 4262, 'listed': 4263, 'sharing': 4264, 'stake': 4265, 'smoking': 4266, 'alan': 4267, 'permit': 4268, 'malaysian': 4269, 'handled': 4270, 'prove': 4271, 'evacuate': 4272, 'recognizes': 4273, 'post-election': 4274, '43': 4275, 'locations': 4276, 'stones': 4277, 'ratified': 4278, 'smugglers': 4279, '07-may': 4280, 'losing': 4281, 'searched': 4282, 'lake': 4283, 'fort': 4284, 'expire': 4285, 'requests': 4286, 'shui-bian': 4287, 'imposing': 4288, 'samarra': 4289, 'duty': 4290, '56': 4291, 'enrich': 4292, 'nationalities': 4293, 'friendly': 4294, 'implementation': 4295, 'fewer': 4296, 'purchased': 4297, 'dujail': 4298, 'crop': 4299, 'album': 4300, 'murdered': 4301, 'wilma': 4302, 'god': 4303, 'rocket-propelled': 4304, 'filled': 4305, 'challenger': 4306, 'month-long': 4307, 'delaying': 4308, 'moqtada': 4309, 'anti-corruption': 4310, 'treating': 4311, 'herself': 4312, 'procedures': 4313, 'kyrgyz': 4314, 'objections': 4315, 'negotiating': 4316, 'chanted': 4317, 'luiz': 4318, 'reaction': 4319, 'convention': 4320, 'baidoa': 4321, 'somalis': 4322, 'establishing': 4323, 'steady': 4324, 'deposits': 4325, 'coffee': 4326, 'laurent': 4327, '48': 4328, 'decree': 4329, 'conflicting': 4330, 'arroyo': 4331, 'expressing': 4332, 'exist': 4333, 'shadow': 4334, 'slaughtered': 4335, 'properly': 4336, 'inventories': 4337, 'weaken': 4338, 'publication': 4339, 'category': 4340, 'platform': 4341, 'businessmen': 4342, 'expelled': 4343, '59': 4344, 'five-day': 4345, 'academy': 4346, '76': 4347, 'inspector': 4348, 'dominican': 4349, 'stepping': 4350, '1962': 4351, 'slovenia': 4352, '15th': 4353, 'firefight': 4354, 'implicated': 4355, 'rio': 4356, 'deliveries': 4357, 'assess': 4358, 'hindu': 4359, 'infections': 4360, '60th': 4361, 'prepares': 4362, 'logistical': 4363, 'drivers': 4364, 'london-based': 4365, 'mil': 4366, 'arcega': 4367, 'annexed': 4368, 'ranks': 4369, 'solid': 4370, 'waste': 4371, 'risks': 4372, 'scheffer': 4373, 'burns': 4374, 'houston': 4375, 'requirements': 4376, 'bangkok': 4377, 'correa': 4378, 'footage': 4379, 'beaten': 4380, 'interfax': 4381, 'warsaw': 4382, 'interviews': 4383, 'villepin': 4384, 'atlanta': 4385, 'wealthy': 4386, 'bloomberg': 4387, 'hike': 4388, 'surveyed': 4389, 'dressed': 4390, 'yousuf': 4391, 'equality': 4392, 'kilograms': 4393, 'resumption': 4394, 'predominantly': 4395, 'promising': 4396, 'beach': 4397, 'marburg': 4398, 'jim': 4399, 'incursion': 4400, 'recording': 4401, 'cache': 4402, 'gedi': 4403, 'chiefs': 4404, 'protocol': 4405, 'whales': 4406, 'reopen': 4407, 'carriles': 4408, 'amounts': 4409, 'hare': 4410, 'guangdong': 4411, 'stationed': 4412, 'appealing': 4413, 'disrupting': 4414, 'gradually': 4415, 'sharif': 4416, 'asia-pacific': 4417, 'zuma': 4418, 'verdict': 4419, 'produces': 4420, 'negroponte': 4421, 'al-shabab': 4422, 'boom': 4423, 'sealed': 4424, 'gunships': 4425, 'crews': 4426, 'condemning': 4427, 'tiny': 4428, 'weight': 4429, 'geological': 4430, 'earthquakes': 4431, 'deemed': 4432, 'simply': 4433, 'eighth': 4434, 'bolster': 4435, 'overthrew': 4436, 'approached': 4437, 'fraudulent': 4438, 'bans': 4439, 'chose': 4440, 'unanimously': 4441, 'rammed': 4442, 'ioc': 4443, 'implementing': 4444, 'permits': 4445, '1973': 4446, 'wind': 4447, 'lord': 4448, 'software': 4449, 'diversify': 4450, 'flag': 4451, 'restricted': 4452, 'rioting': 4453, 'la': 4454, '12,000': 4455, 'heat': 4456, 'ownership': 4457, 'depend': 4458, 'structural': 4459, 'shore': 4460, 'establishment': 4461, 'quiet': 4462, 'recovering': 4463, 'dedicated': 4464, 'interference': 4465, 'manouchehr': 4466, 'thanks': 4467, 'mediate': 4468, 'lacks': 4469, 'windows': 4470, 'no-confidence': 4471, 'akayev': 4472, 'sierra': 4473, 'leone': 4474, 'contacts': 4475, 'biden': 4476, 'accompanied': 4477, 'sabotage': 4478, 'posed': 4479, 'happy': 4480, 'midnight': 4481, 'circumstances': 4482, '1,50,000': 4483, 'kaczynski': 4484, '10th': 4485, 'adequate': 4486, '1986': 4487, 'trips': 4488, 'francisco': 4489, 'sons': 4490, 'refining': 4491, 'amman': 4492, 'rocks': 4493, 'irna': 4494, 'greenhouse': 4495, 'judiciary': 4496, 'lab': 4497, 'celebration': 4498, 'controversy': 4499, 'khaled': 4500, 'shaukat': 4501, 'accidentally': 4502, 'defendant': 4503, 'suspending': 4504, 'difficulties': 4505, 'breaking': 4506, 'stephen': 4507, 'yesterday': 4508, 'pension': 4509, 'bit': 4510, 'predict': 4511, 'li': 4512, 'apologized': 4513, 'drafting': 4514, 'electrical': 4515, 'brothers': 4516, 'smoke': 4517, 'nominees': 4518, 'routine': 4519, 'else': 4520, '57': 4521, 'sexually': 4522, 'resuming': 4523, 'suit': 4524, '1980': 4525, 'firefighters': 4526, 'surplus': 4527, 'rafael': 4528, 'gul': 4529, 'predecessor': 4530, 'sadr': 4531, 'adviser': 4532, 'paramilitaries': 4533, 'pushing': 4534, 'eat': 4535, 'cancel': 4536, '70,000': 4537, '170': 4538, 'stolen': 4539, 'entrance': 4540, 'steel': 4541, 'al-anbar': 4542, 'all-time': 4543, 'gilani': 4544, 'eliminate': 4545, '1,200': 4546, 'substantially': 4547, 'mutharika': 4548, 'sister': 4549, '07-jun': 4550, '39': 4551, 'priest': 4552, 'finals': 4553, 'high-profile': 4554, 'reunification': 4555, 'hanoi': 4556, 'awaiting': 4557, 'enacted': 4558, 'karadzic': 4559, 'traditionally': 4560, 'expert': 4561, 'monthly': 4562, 'lasted': 4563, 'pending': 4564, 'shield': 4565, 'posada': 4566, 'frozen': 4567, 'intentions': 4568, 'zambia': 4569, 'reaffirmed': 4570, 'spin': 4571, 'enjoy': 4572, 'younger': 4573, 'g8': 4574, 'gaining': 4575, 'bengal': 4576, 'scandals': 4577, 'electric': 4578, 'fishermen': 4579, 'mccormack': 4580, 'liberal': 4581, 'true': 4582, 'veterinary': 4583, 'dismiss': 4584, 'intercepted': 4585, 'ranked': 4586, 'apec': 4587, 'apply': 4588, 'tense': 4589, 'spacecraft': 4590, '14,000': 4591, 'mwai': 4592, 'detailed': 4593, 'rulers': 4594, 'factor': 4595, 'jill': 4596, 'cooperating': 4597, 'grants': 4598, 'pieces': 4599, 'long-range': 4600, 'haven': 4601, 'direction': 4602, 'battled': 4603, 'indication': 4604, 'manufactured': 4605, 'councils': 4606, 'duma': 4607, 'spotted': 4608, 'gesture': 4609, 'carol': 4610, 'landfall': 4611, 'learned': 4612, 'connected': 4613, 'effectively': 4614, 'quota': 4615, 'transmission': 4616, 'imam': 4617, 'reopened': 4618, 'sisco': 4619, 'dissident': 4620, 'shells': 4621, 'destabilize': 4622, 'narrowly': 4623, 'interpreter': 4624, 'tear': 4625, 'maintains': 4626, 'gotovina': 4627, '1960s': 4628, 'slaves': 4629, 'rely': 4630, 'describes': 4631, 'online': 4632, 'cambodian': 4633, 'freedoms': 4634, 'bernard': 4635, 'depression': 4636, 'intervention': 4637, 'exhibition': 4638, 'interfering': 4639, 'display': 4640, 'anti-syrian': 4641, 'confrontation': 4642, 'stamp': 4643, 'rifles': 4644, 'bogota': 4645, 'bethlehem': 4646, 'treason': 4647, 'catholics': 4648, 'love': 4649, 'check': 4650, 'kuwaiti': 4651, 'dominant': 4652, 'younis': 4653, 'inc.': 4654, 'deficits': 4655, 'attracting': 4656, 'transparency': 4657, 'candidacy': 4658, 'consideration': 4659, 'worries': 4660, 'represented': 4661, 'cameroon': 4662, 'verified': 4663, 'famine': 4664, 'decreased': 4665, 'publishing': 4666, 'ben': 4667, '110': 4668, 'jersey': 4669, 'degrees': 4670, 'nationality': 4671, 'chemicals': 4672, 'arresting': 4673, 'optimistic': 4674, 'tore': 4675, 'proper': 4676, 'khyber': 4677, 'andrew': 4678, 'mitchell': 4679, 'wen': 4680, 'downtown': 4681, 'spark': 4682, 'welfare': 4683, 'progressive': 4684, 'tobacco': 4685, 'blown': 4686, 'capabilities': 4687, 'abuja': 4688, 'eye': 4689, 'myanmar': 4690, 'alleging': 4691, 'stops': 4692, '1987': 4693, 'product': 4694, 'johannesburg': 4695, 'gabon': 4696, 'unlikely': 4697, 'copper': 4698, 'jeddah': 4699, 'hoop': 4700, 'celebrating': 4701, 'basketball': 4702, 'steadily': 4703, 'airbus': 4704, 'cope': 4705, 'outrage': 4706, 'margaret': 4707, 'pro-western': 4708, 'unilateral': 4709, 'dominique': 4710, 'bodyguard': 4711, 'khin': 4712, 'lady': 4713, 'kenyans': 4714, 'albanian': 4715, 'revised': 4716, 'unfair': 4717, 'flags': 4718, 'eliminated': 4719, 'watched': 4720, 'adam': 4721, 'championship': 4722, 'germans': 4723, 'wildlife': 4724, 'file': 4725, 'releasing': 4726, '8,00,000': 4727, 'santiago': 4728, 'sport': 4729, 'massachusetts': 4730, 'observe': 4731, 'mutate': 4732, 'actress': 4733, 'akbar': 4734, 'four-day': 4735, 'riyadh': 4736, 'oman': 4737, 'answer': 4738, 'guidelines': 4739, 'xvi': 4740, 'puts': 4741, 'right-wing': 4742, 'venture': 4743, 'priests': 4744, 'forms': 4745, 'strongholds': 4746, 'undersecretary': 4747, 'lawsuits': 4748, 'semi-autonomous': 4749, 'miller': 4750, '1975': 4751, 'lethal': 4752, 'succession': 4753, 'derail': 4754, 'notice': 4755, 'six-year': 4756, 'fees': 4757, 'flash': 4758, 'macau': 4759, 'bakiyev': 4760, 'eggs': 4761, 'tung': 4762, 'iowa': 4763, 'hideouts': 4764, 'investigator': 4765, 'two-week': 4766, 'washed': 4767, 'mention': 4768, 'abductors': 4769, 'pradesh': 4770, 'ambitious': 4771, 'e-mails': 4772, 'diamond': 4773, 'factional': 4774, '40,000': 4775, 'integrity': 4776, 'chain': 4777, 'commodities': 4778, 'stages': 4779, 'aging': 4780, 'kamal': 4781, 'veteran': 4782, 'monks': 4783, 'tibetans': 4784, 'diabetes': 4785, 'pearson': 4786, 'balkan': 4787, 'sweep': 4788, 'treat': 4789, 'coach': 4790, 'existence': 4791, 'jericho': 4792, 'autonomous': 4793, 'descent': 4794, 'tajikistan': 4795, 'cites': 4796, 'auction': 4797, 'commitments': 4798, 'obtain': 4799, 'columbus': 4800, 'persons': 4801, 'extreme': 4802, 'employs': 4803, 'yusuf': 4804, 'minorities': 4805, 'consists': 4806, 'ashore': 4807, 'four-year': 4808, 'engineer': 4809, 'burkina': 4810, 'faso': 4811, 'contaminated': 4812, 'phones': 4813, 'ordering': 4814, 'exact': 4815, 'poisoned': 4816, 'ninth': 4817, 'settle': 4818, 'crush': 4819, 'reading': 4820, 'estonia': 4821, 'characterized': 4822, 'addis': 4823, 'explain': 4824, 'commit': 4825, 'retain': 4826, 'postpone': 4827, 'adults': 4828, 'simon': 4829, 'federalism': 4830, 'overseeing': 4831, 'albanians': 4832, 'pray': 4833, 'lot': 4834, '...': 4835, 'mbeki': 4836, 'essential': 4837, 'eln': 4838, 'drilling': 4839, 'struggled': 4840, 'arm': 4841, 'afar': 4842, 'pulling': 4843, 'professionals': 4844, 'emile': 4845, 'bernanke': 4846, 'asefi': 4847, 'complaint': 4848, 'submit': 4849, 'pro-taleban': 4850, 'features': 4851, 'paris-based': 4852, 'turks': 4853, 'consumption': 4854, 'ceded': 4855, 'ottoman': 4856, 'contested': 4857, 'quell': 4858, 'javier': 4859, 'barre': 4860, 'populated': 4861, 'donated': 4862, 'incoming': 4863, 'elements': 4864, 'exit': 4865, '350': 4866, 'encountered': 4867, 'purpose': 4868, 'remnants': 4869, 'smuggled': 4870, 'laundering': 4871, 'employed': 4872, 'bp': 4873, 'abductions': 4874, 'tents': 4875, 'najaf': 4876, 'republics': 4877, 'westerners': 4878, 'destination': 4879, 'volcano': 4880, 'rebounded': 4881, 'operational': 4882, '17,000': 4883, 'attract': 4884, 'belgian': 4885, 'tutsis': 4886, 'rangel': 4887, 'locked': 4888, 'rightist': 4889, 'epidemic': 4890, 'minibus': 4891, 'libby': 4892, 'economist': 4893, 'topics': 4894, 'dmitri': 4895, 'argue': 4896, 'legitimate': 4897, 'consensus': 4898, 'egyptians': 4899, 'homemade': 4900, 'encouraging': 4901, 'minsk': 4902, 'rioters': 4903, 'raza': 4904, 'cleveland': 4905, 'doubles': 4906, 'singles': 4907, 'al-bashir': 4908, 'cholera': 4909, 'tariffs': 4910, 'plastic': 4911, 'justices': 4912, 'stance': 4913, 'authorizing': 4914, 'brokered': 4915, 'liechtenstein': 4916, '1985': 4917, 'staffers': 4918, 'audio': 4919, 'operative': 4920, 'dayton': 4921, 'takeover': 4922, 'koran': 4923, 'metal': 4924, 'card': 4925, 'isaf': 4926, 'learn': 4927, 'e-mail': 4928, 'mozambique': 4929, 'investor': 4930, 'artist': 4931, 'telling': 4932, 'opposing': 4933, 'administrator': 4934, '2.3': 4935, 'upheld': 4936, 'via': 4937, 'cubic': 4938, 'roddick': 4939, 'discussion': 4940, 'maintenance': 4941, 'initiatives': 4942, 'repression': 4943, 'informal': 4944, 'kyoto': 4945, 'gordon': 4946, 'sinai': 4947, 'berlusconi': 4948, 'captivity': 4949, 'landslides': 4950, 'articles': 4951, 'laos': 4952, 'bachelet': 4953, 'trillion': 4954, 'obrador': 4955, 'actor': 4956, 'throne': 4957, 'client': 4958, 'ruler': 4959, 'illinois': 4960, 'bills': 4961, 'namibia': 4962, 'catch': 4963, 'mike': 4964, 'peretz': 4965, 'cypriots': 4966, 'kadima': 4967, 'besigye': 4968, 'pounded': 4969, 'residential': 4970, 'desert': 4971, 'hectares': 4972, '900': 4973, 'musician': 4974, 'concrete': 4975, 'kidney': 4976, 'invest': 4977, 'monitored': 4978, 'timber': 4979, 'macroeconomic': 4980, 'secured': 4981, 'pound': 4982, 'portions': 4983, 'corrupt': 4984, 'conclusion': 4985, 'pro-russian': 4986, 'brian': 4987, 'narrates': 4988, 'slowing': 4989, 'books': 4990, 'barriers': 4991, 'owns': 4992, 'sporadic': 4993, 'assume': 4994, 'cells': 4995, 'suddenly': 4996, 'bulk': 4997, 'veterans': 4998, 'musical': 4999, 'performing': 5000, 'emperor': 5001, 'governors': 5002, 'bar': 5003, 'chris': 5004, 'peak': 5005, 'maryland': 5006, 'tip': 5007, 'patrolling': 5008, 'osman': 5009, 'watchdog': 5010, 'hair': 5011, 'adopt': 5012, '1983': 5013, 'drowned': 5014, 'tulkarem': 5015, 'restart': 5016, 'planting': 5017, 'manager': 5018, 'joe': 5019, 'removing': 5020, 'rebel-held': 5021, 'jamaica': 5022, 'willingness': 5023, 'remark': 5024, 'add': 5025, 'ellen': 5026, 'salaries': 5027, 'overturned': 5028, 'silver': 5029, 'medals': 5030, 'downhill': 5031, 'sharm': 5032, 'albania': 5033, 'example': 5034, 'specify': 5035, 'masked': 5036, 'congratulated': 5037, 'thinks': 5038, 'ababa': 5039, 'framework': 5040, 'associate': 5041, 'oath': 5042, '88': 5043, 'sayyaf': 5044, 'plus': 5045, '1945': 5046, 'plots': 5047, 'components': 5048, 'dominate': 5049, 'sponsored': 5050, 'diamonds': 5051, 'ouattara': 5052, 'challenging': 5053, 'armenia': 5054, 'ricardo': 5055, 'adjourned': 5056, 'pipelines': 5057, 'bitter': 5058, 'g-8': 5059, 'pain': 5060, 'shelters': 5061, 'chaudhry': 5062, 'defuse': 5063, 'mofaz': 5064, 'nangarhar': 5065, 'advised': 5066, 'airliner': 5067, 'installations': 5068, 'secrets': 5069, '1,80,000': 5070, 'multinational': 5071, 'horse': 5072, 'mouth': 5073, 'fill': 5074, 'kinshasa': 5075, 'janeiro': 5076, 'inspection': 5077, '51': 5078, 'roh': 5079, 'restaurants': 5080, 'prevention': 5081, 'window': 5082, 'blaze': 5083, 'hosted': 5084, 'suffers': 5085, 'elderly': 5086, 'origin': 5087, 'crossings': 5088, 'approaching': 5089, 'studying': 5090, 'clothes': 5091, 'polar': 5092, 'negative': 5093, 'tandja': 5094, 'leonid': 5095, 'landmine': 5096, '46': 5097, 'spurred': 5098, 'moussa': 5099, 'fines': 5100, 'aso': 5101, 'restrict': 5102, 'sense': 5103, 'ambassadors': 5104, 'financing': 5105, 'siege': 5106, 'proof': 5107, 'invitation': 5108, 'revolt': 5109, 'lying': 5110, 'archipelago': 5111, 'cotton': 5112, 'sustain': 5113, 'statistics': 5114, 'unified': 5115, 'administered': 5116, 'simple': 5117, 'athens': 5118, 'confessed': 5119, 'condemns': 5120, 'sufficient': 5121, 'bears': 5122, 'arctic': 5123, 'bear': 5124, '1.2': 5125, 'laborers': 5126, 'slaughter': 5127, 'piracy': 5128, 'distance': 5129, 'participated': 5130, 'faster': 5131, 'drives': 5132, 'tal': 5133, 'assessment': 5134, 'apology': 5135, 'intestinal': 5136, 'tortured': 5137, 'dictatorship': 5138, 'brigade': 5139, 'transmitted': 5140, 'confident': 5141, 'extradite': 5142, 'janjaweed': 5143, 'popularity': 5144, '3,500': 5145, 'garden': 5146, 'immigrant': 5147, 'supposed': 5148, 'seychelles': 5149, 'fisherman': 5150, 'santa': 5151, 'breaks': 5152, 'meantime': 5153, 'customers': 5154, '2,500': 5155, 'canadians': 5156, 'sao': 5157, 'relies': 5158, 'semifinal': 5159, 'oversee': 5160, 'hingis': 5161, 'favored': 5162, 'incumbent': 5163, 'riding': 5164, 'junior': 5165, '13th': 5166, 'hunters': 5167, 'envoys': 5168, 'edition': 5169, 'knowledge': 5170, '1981': 5171, 'recruits': 5172, 'basayev': 5173, 'recommendation': 5174, 'collected': 5175, 'parked': 5176, 'reference': 5177, 'raw': 5178, 'expression': 5179, 'mongolia': 5180, 'registration': 5181, 'mousavi': 5182, 'inspections': 5183, 'representation': 5184, 'chanting': 5185, 'wins': 5186, 'professor': 5187, 'wilson': 5188, 'mandela': 5189, 'hungarian': 5190, 'malta': 5191, 'egeland': 5192, 'describe': 5193, 'sumatra': 5194, 'duty-free': 5195, 'spare': 5196, 'punished': 5197, 'waging': 5198, 'shimon': 5199, 'tally': 5200, 'ski': 5201, 'dying': 5202, 'separated': 5203, 'protectorate': 5204, 'landlocked': 5205, 'sitting': 5206, 'shop': 5207, 'criticize': 5208, 'reveal': 5209, 'wartime': 5210, 'doubled': 5211, 'wheat': 5212, 'detentions': 5213, 'managing': 5214, '06-jan': 5215, 'advancing': 5216, 'appropriate': 5217, 'athletes': 5218, 'preparation': 5219, 'skills': 5220, 'slammed': 5221, 'battered': 5222, 'eventual': 5223, 'warrants': 5224, 'oldest': 5225, 'stemming': 5226, 'pointed': 5227, 'bulgarian': 5228, 'palestine': 5229, 'cartoon': 5230, 'invaded': 5231, 'acquisition': 5232, 'roger': 5233, 'accounting': 5234, 'ronald': 5235, 'indians': 5236, 'marxist': 5237, 'aggression': 5238, 'hurled': 5239, 'lunar': 5240, 'excessive': 5241, 'examine': 5242, 'lieberman': 5243, 'broadcaster': 5244, 'temple': 5245, 'departure': 5246, 'mideast': 5247, 'capsized': 5248, 'dhaka': 5249, 'sailors': 5250, 'remember': 5251, 'ratify': 5252, 'reviewing': 5253, 'contains': 5254, 'pledging': 5255, 'papers': 5256, 'watches': 5257, 'pair': 5258, 'johnson': 5259, 'askar': 5260, 'harmful': 5261, 'keys': 5262, 'partly': 5263, '25-year-old': 5264, 'mastermind': 5265, 'barroso': 5266, 'shelling': 5267, 'stealing': 5268, 'promotion': 5269, 'travels': 5270, 'throwing': 5271, 'samuel': 5272, 'floor': 5273, 'ailing': 5274, 'bolivar': 5275, 'york-based': 5276, '58': 5277, 'considerable': 5278, 'rail': 5279, 'long-standing': 5280, 'sergeant': 5281, 'replacement': 5282, 'refusing': 5283, 'belong': 5284, 'eased': 5285, 'gloria': 5286, 'ukrainians': 5287, 'supporter': 5288, 'latortue': 5289, 'baltic': 5290, 'table': 5291, 'enforce': 5292, 'cartels': 5293, 'horn': 5294, '4,00,000': 5295, 'matters': 5296, 'tycoon': 5297, '73': 5298, 'repeat': 5299, 'guardian': 5300, 'croats': 5301, '44': 5302, 'harbor': 5303, 'tankers': 5304, 'iron': 5305, 'kills': 5306, 'coca': 5307, 'summoned': 5308, 'shift': 5309, 'theater': 5310, 'farah': 5311, 'gm': 5312, 'infighting': 5313, 'khalid': 5314, 'phoenix': 5315, 'ranging': 5316, 'kilometer': 5317, 'anti-drug': 5318, 'fixed': 5319, '72': 5320, 'nevertheless': 5321, 'independently': 5322, '1-0': 5323, 'hosts': 5324, 'improvement': 5325, 'jaap': 5326, 'agassi': 5327, 'hanging': 5328, 'bagram': 5329, 'unique': 5330, 'zebari': 5331, 'rotating': 5332, 'poorly': 5333, 'generated': 5334, 'lists': 5335, 'charging': 5336, 'identities': 5337, 'minimum': 5338, '180': 5339, 'flowing': 5340, 'door': 5341, 'placing': 5342, 'truth': 5343, 'sanitation': 5344, 'outcome': 5345, 'olusegun': 5346, 'distributing': 5347, 'eritrean': 5348, 'accusation': 5349, '10-year': 5350, 'delivering': 5351, 'jurisdiction': 5352, 'crow': 5353, 'shook': 5354, 'sees': 5355, 'human-to-human': 5356, 'unicef': 5357, 'consultations': 5358, 'outpost': 5359, 'colonial': 5360, 'mladic': 5361, 'triple': 5362, 'dna': 5363, 'innings': 5364, 'rafsanjani': 5365, 'observing': 5366, 'hidden': 5367, 'auto': 5368, 'install': 5369, 'nationalists': 5370, 'acquitted': 5371, 'improvements': 5372, 'marino': 5373, 'thus': 5374, 'principle': 5375, 'monopoly': 5376, 'fame': 5377, 'topple': 5378, 'populations': 5379, 'facebook': 5380, 'boldak': 5381, 'uniforms': 5382, 'expensive': 5383, 'dump': 5384, 'flock': 5385, 'al-islam': 5386, 'heightened': 5387, 'slight': 5388, 'stood': 5389, 'clinic': 5390, 'mwanawasa': 5391, 'mosques': 5392, 'governance': 5393, 're-opened': 5394, 'referral': 5395, 'chamber': 5396, 'yekhanurov': 5397, 'kagame': 5398, 'logistics': 5399, 'yonhap': 5400, 'enriching': 5401, '1947': 5402, 'johnson-sirleaf': 5403, 'contraction': 5404, 'metric': 5405, 'multilateral': 5406, 'wood': 5407, 'fertilizer': 5408, 'supplying': 5409, 'dow': 5410, 'baby': 5411, 'movements': 5412, 'mir': 5413, 'hossein': 5414, 'rejects': 5415, 'goose': 5416, 'roche': 5417, 'expired': 5418, 'allege': 5419, 'wickets': 5420, 'tighten': 5421, 'monument': 5422, 'walls': 5423, 'peoples': 5424, 'feet': 5425, 'lasting': 5426, 'coordinator': 5427, 'chambers': 5428, 'burst': 5429, 'associates': 5430, 'demonstrate': 5431, 'fit': 5432, 'bars': 5433, 'courtroom': 5434, 'post-war': 5435, 'casting': 5436, 'amir': 5437, 'jaffna': 5438, 'breast': 5439, 'twitter': 5440, 'empty': 5441, 'coffin': 5442, 'reinstated': 5443, 'obstacles': 5444, '[': 5445, ']': 5446, 'finds': 5447, 'stayed': 5448, 'loaded': 5449, 'lhasa': 5450, 'helps': 5451, 'disarming': 5452, 'speeches': 5453, 'normally': 5454, 'taiwanese': 5455, 'persuade': 5456, 'bloodshed': 5457, 'highways': 5458, 'universal': 5459, 'billionaire': 5460, 'commodity': 5461, 'airliners': 5462, 'foiled': 5463, 'shareholders': 5464, 'dismantling': 5465, 'proved': 5466, 'unsuccessful': 5467, 'jointly': 5468, '9,000': 5469, 'hampering': 5470, 'talking': 5471, 'intel': 5472, 'instruments': 5473, 'orbiting': 5474, 'catastrophe': 5475, 'rouge': 5476, 'situations': 5477, 'divide': 5478, '220': 5479, 'cbs': 5480, 'occasion': 5481, 'sacrifice': 5482, '49': 5483, 'muzaffarabad': 5484, 'hosting': 5485, 'retailers': 5486, 'doubts': 5487, 'screen': 5488, '1,600': 5489, 'handing': 5490, 'orange': 5491, 'hebron': 5492, 'bronze': 5493, 'surfaced': 5494, 'older': 5495, 'supplied': 5496, 'error': 5497, 'elder': 5498, 'conversation': 5499, 'zawahiri': 5500, 'karimov': 5501, 'indirect': 5502, 'guarantees': 5503, 'drones': 5504, 'devastation': 5505, 'storage': 5506, 'winners': 5507, 'signature': 5508, 'sole': 5509, 'advocacy': 5510, 'guaranteed': 5511, 'portuguese': 5512, 'offset': 5513, 'generation': 5514, 'compact': 5515, 'fragile': 5516, 'kerry': 5517, 'engine': 5518, 'embattled': 5519, 'yield': 5520, 'dispatched': 5521, 'engineering': 5522, 'dan': 5523, 'intervene': 5524, 'deadlock': 5525, 'object': 5526, 'idriss': 5527, 'flames': 5528, 'hybrid': 5529, 'tougher': 5530, 'railway': 5531, 'exposure': 5532, 'dismissal': 5533, 'persistent': 5534, 'itar-tass': 5535, 'slobodan': 5536, 'trend': 5537, 'bahamas': 5538, 'alleges': 5539, 'content': 5540, 'rehabilitation': 5541, '105': 5542, 'celsius': 5543, 'afford': 5544, '15-year-old': 5545, 'wrapped': 5546, 'bicycle': 5547, 'centrifuges': 5548, 'alpha': 5549, 'mistakenly': 5550, 'controlling': 5551, 'personally': 5552, 'curfews': 5553, 'yoweri': 5554, '12th': 5555, 'divisions': 5556, 'era': 5557, 'gunfight': 5558, 'low-level': 5559, 'siad': 5560, 'searches': 5561, 'hub': 5562, 'inspect': 5563, 'saad': 5564, 'kicked': 5565, 'motorcade': 5566, 'anti-war': 5567, 'absence': 5568, 'negotiated': 5569, 'gate': 5570, 'suggests': 5571, 'convert': 5572, 'critically': 5573, 'blocks': 5574, 'multiparty': 5575, 'faithful': 5576, 'hometown': 5577, 'machinery': 5578, 'anything': 5579, 'convoys': 5580, 'queen': 5581, 'pool': 5582, 'federer': 5583, 'nicholas': 5584, 'hubble': 5585, 'uncertain': 5586, 'dance': 5587, 'row': 5588, 'hikes': 5589, 'beslan': 5590, 'well-known': 5591, 'spies': 5592, 'puerto': 5593, 'jenin': 5594, 'sam': 5595, 'respects': 5596, 'teenagers': 5597, 'casualty': 5598, 'coincide': 5599, 'nld': 5600, 'odds': 5601, 'entirely': 5602, 'contest': 5603, 'advisors': 5604, 'greeted': 5605, 'boris': 5606, 'tadic': 5607, 'strongest': 5608, 'jump': 5609, 'fined': 5610, 'goodwill': 5611, 'endangered': 5612, 'wiped': 5613, 'bribes': 5614, 'productivity': 5615, 'steep': 5616, 'tripoli': 5617, 'commandos': 5618, 'suburbs': 5619, 'tigers': 5620, 'operates': 5621, 'swan': 5622, 'jonathan': 5623, '31-year-old': 5624, 'steve': 5625, 'imprisonment': 5626, 'schwarzenegger': 5627, 'chicago': 5628, 'grain': 5629, 'three-quarters': 5630, 'uige': 5631, 'affiliated': 5632, 'enemies': 5633, 'suggesting': 5634, 'bankruptcy': 5635, 'preferred': 5636, 'model': 5637, 'modified': 5638, 'sub-saharan': 5639, 'mediator': 5640, 'credible': 5641, 'dubbed': 5642, 'subsidiary': 5643, 'drill': 5644, 'exclusive': 5645, 'abdullahi': 5646, 'coma': 5647, 'bob': 5648, 'musa': 5649, 'outlets': 5650, 'drink': 5651, 'reflected': 5652, 'printed': 5653, 'waged': 5654, 'jong-il': 5655, 'staging': 5656, 'punishment': 5657, 'hardline': 5658, 'heating': 5659, 'credibility': 5660, 'prosecute': 5661, 'rocked': 5662, 'directors': 5663, '13,000': 5664, 'victor': 5665, 'rounded': 5666, 'semifinals': 5667, 'undergoing': 5668, 'aiding': 5669, 'expectations': 5670, 'honest': 5671, 'belt': 5672, 'uniform': 5673, 'tactics': 5674, 'wali': 5675, 'stars': 5676, '47': 5677, 'presents': 5678, 'stalemate': 5679, 'attorneys': 5680, 'selection': 5681, 'fastest': 5682, 'gutierrez': 5683, 'fashion': 5684, 'nagin': 5685, 'routinely': 5686, 'fake': 5687, '2.7': 5688, 'moldova': 5689, 'quite': 5690, 'shepherd': 5691, 'eliminating': 5692, 'spacewalk': 5693, 'gore': 5694, 'tbilisi': 5695, 'confront': 5696, 'deported': 5697, 'solve': 5698, 'hot': 5699, 'deportation': 5700, 'plays': 5701, 'contacted': 5702, 'bushehr': 5703, 'bishkek': 5704, 'radar': 5705, 'nour': 5706, 'slick': 5707, 'homeowners': 5708, 'methods': 5709, 'floodwaters': 5710, 'teacher': 5711, 'worry': 5712, 'wal-mart': 5713, 'yen': 5714, 'cow': 5715, 'mercosur': 5716, 'aquino': 5717, 'spirit': 5718, 'undergo': 5719, 'panic': 5720, 'lunch': 5721, 'arrangement': 5722, 'flores': 5723, 'adult': 5724, 'augusto': 5725, 'six-month': 5726, 'melinda': 5727, 'tuberculosis': 5728, 'finalized': 5729, 'types': 5730, 'initiated': 5731, '1972': 5732, 'accounted': 5733, 'pursuit': 5734, 'mutual': 5735, 'forest': 5736, 'escalated': 5737, 'harder': 5738, 'treaties': 5739, 'rifle': 5740, 'donate': 5741, 'diesel': 5742, 'graves': 5743, 'honoring': 5744, 'seeded': 5745, 'andy': 5746, 'mothers': 5747, 'doping': 5748, 'symbol': 5749, 'skilled': 5750, 'strait': 5751, 'anonymity': 5752, 'week-long': 5753, 'enjoyed': 5754, 'hardest-hit': 5755, '83': 5756, 'de~facto': 5757, 'extraordinary': 5758, 'lra': 5759, 'condemnation': 5760, 'guest': 5761, 'passes': 5762, 'classes': 5763, 'depicting': 5764, 'vision': 5765, 'performed': 5766, 'compliance': 5767, '1978': 5768, 'regulatory': 5769, 'combination': 5770, 'mineral': 5771, 'mind': 5772, 'tie': 5773, 'crown': 5774, 'indian-controlled': 5775, 'jammu': 5776, 'design': 5777, 'unarmed': 5778, 'repay': 5779, 'gandhi': 5780, 'barghouti': 5781, 'anti-american': 5782, 'touch': 5783, 'regained': 5784, 'kasuri': 5785, 'proliferation': 5786, 'vacation': 5787, 'authenticity': 5788, 'defectors': 5789, 'beans': 5790, 'overcrowded': 5791, 'provision': 5792, 'tomb': 5793, 'lech': 5794, 'phosphate': 5795, 'please': 5796, 'thabo': 5797, 'racism': 5798, 'strasbourg': 5799, 'impossible': 5800, 'cultivation': 5801, 'shoes': 5802, 'smashed': 5803, 'objects': 5804, 'removal': 5805, 'anti-terror': 5806, 'shaul': 5807, 'calendar': 5808, 'failures': 5809, 'ranking': 5810, 'gerard': 5811, 'viewed': 5812, 'petersburg': 5813, 'photographer': 5814, 'principles': 5815, 'clients': 5816, '63': 5817, 'benazir': 5818, 'togolese': 5819, 'premier': 5820, 'tin': 5821, 'stiff': 5822, 'chest': 5823, 'frank': 5824, 'declines': 5825, 'botswana': 5826, 'applied': 5827, 'launches': 5828, 'jiabao': 5829, 'exchanged': 5830, 'bullets': 5831, 'institution': 5832, 'felt': 5833, 'refuses': 5834, 'vaccination': 5835, 'tribes': 5836, 'detaining': 5837, 'organize': 5838, 'warheads': 5839, 'cameraman': 5840, 'inform': 5841, 'three-week': 5842, 'coups': 5843, 'stabilization': 5844, 'stole': 5845, 'thing': 5846, 'robust': 5847, 'twin': 5848, 'lengthy': 5849, 'concessions': 5850, 'expense': 5851, '30th': 5852, 'beit': 5853, 'accordance': 5854, 'surprised': 5855, 'hatred': 5856, 'privately': 5857, 'warfare': 5858, 'slalom': 5859, 'universities': 5860, 'probing': 5861, 'turkmenistan': 5862, 'workforce': 5863, 'unspecified': 5864, 'ratification': 5865, 'bonds': 5866, 'investigative': 5867, 'photograph': 5868, '2020': 5869, 'icy': 5870, 'espionage': 5871, 'oil-producing': 5872, 'yulia': 5873, 'biased': 5874, 'conservation': 5875, 'greatly': 5876, 'hid': 5877, 'americas': 5878, 'democratically': 5879, 'kirchner': 5880, 'wimbledon': 5881, 'preserve': 5882, 'ethiopians': 5883, 'assassinate': 5884, 'stretch': 5885, 'shinawatra': 5886, 'experiments': 5887, 'disagreements': 5888, 'broader': 5889, 'rosneft': 5890, 'plutonium': 5891, 'oecd': 5892, 'journey': 5893, 'tail': 5894, 'looked': 5895, 'fact': 5896, 'chung': 5897, 'score': 5898, 'australians': 5899, 'rig': 5900, 'fueling': 5901, 'dawn': 5902, '1,400': 5903, 'legally': 5904, 'containing': 5905, 'ton': 5906, 'housed': 5907, '1959': 5908, 'disclosed': 5909, 'radovan': 5910, 'relocate': 5911, 'record-high': 5912, 'moratorium': 5913, 'whaling': 5914, 'teenage': 5915, 'patient': 5916, 'tolerate': 5917, 'rctv': 5918, 'mustafa': 5919, 'stone': 5920, 'serpent': 5921, 'reconnaissance': 5922, 'arabian': 5923, 'burn': 5924, 'mortgage': 5925, 'cardinal': 5926, 'severed': 5927, 'knocked': 5928, 'murdering': 5929, 'consular': 5930, 'urgent': 5931, '02-jan': 5932, 'menezes': 5933, 'servants': 5934, 'merge': 5935, 'occasions': 5936, 'launchers': 5937, 'medication': 5938, 'caucasus': 5939, 'electronics': 5940, 'poppy': 5941, 'exiles': 5942, 'persecution': 5943, 'afterwards': 5944, 'carries': 5945, 'ira': 5946, 'satellites': 5947, 'louis': 5948, 'drive-by': 5949, 'drawing': 5950, 'alcohol': 5951, 'rugged': 5952, 'ould': 5953, 're-elected': 5954, 'yudhoyono': 5955, '81': 5956, 'cautioned': 5957, 'assurances': 5958, 'core': 5959, 'thrown': 5960, 'xinjiang': 5961, 'fema': 5962, 'ussr': 5963, 'migratory': 5964, 'andean': 5965, 'cheaper': 5966, 'banning': 5967, 'jean': 5968, 'ray': 5969, 'monarch': 5970, 'nasser': 5971, 'chased': 5972, 'promoted': 5973, 'actual': 5974, 'acquired': 5975, 'lies': 5976, 'swans': 5977, 'caretaker': 5978, 'ortega': 5979, 'cancun': 5980, 'arguing': 5981, '3.5': 5982, 'walk': 5983, 'nose': 5984, '1948': 5985, 'hollywood': 5986, 'suggest': 5987, 'checks': 5988, 'resource': 5989, 'briefing': 5990, 'assaults': 5991, 'ushered': 5992, 'antarctica': 5993, 'interpol': 5994, 'zelaya': 5995, 'impeachment': 5996, 'dera': 5997, 'contribution': 5998, 'nationalize': 5999, 'saleh': 6000, 'easter': 6001, 'audiences': 6002, 'goss': 6003, 'commemorate': 6004, 'chertoff': 6005, 'procession': 6006, 'khalilzad': 6007, 'talat': 6008, 'anwar': 6009, 'currencies': 6010, 'managers': 6011, 'ferry': 6012, 'happen': 6013, 'gay': 6014, 'defected': 6015, 'functioning': 6016, 'strife': 6017, 'detlev': 6018, 'bridges': 6019, 'songs': 6020, 'tackle': 6021, 'upgrade': 6022, 'speakers': 6023, 'tightened': 6024, '750': 6025, 'vaccines': 6026, 'glass': 6027, 'observer': 6028, 'audit': 6029, 'permanently': 6030, 'regards': 6031, 'proceed': 6032, 'globe': 6033, 'gunbattles': 6034, 'serves': 6035, 'capability': 6036, 'inflows': 6037, 'sentiment': 6038, 'corporate': 6039, 'kharrazi': 6040, 'short-term': 6041, 'employers': 6042, 'hired': 6043, 'inaugurated': 6044, 'batons': 6045, 'choosing': 6046, 'factors': 6047, 'affects': 6048, 'clark': 6049, 'roed-larsen': 6050, 'warm': 6051, 'master': 6052, 'guarantee': 6053, 'obligations': 6054, '61': 6055, 'colorado': 6056, 'interrupted': 6057, 'kony': 6058, 'reversed': 6059, 'fm': 6060, 'arsenal': 6061, 'yuri': 6062, 'inter-american': 6063, 'emerge': 6064, 'gross': 6065, 'scottish': 6066, 'latter': 6067, 'attained': 6068, 'ivan': 6069, 'sailing': 6070, 'manner': 6071, 'sends': 6072, 'armenian': 6073, 'invested': 6074, 'utc': 6075, 'programming': 6076, 'dignitaries': 6077, 'discipline': 6078, 'denouncing': 6079, '69': 6080, 'gifts': 6081, 'imperial': 6082, 'writer': 6083, 'eyes': 6084, 'heights': 6085, 'nervous': 6086, 'midst': 6087, 'interviewed': 6088, 'stamps': 6089, 'dirty': 6090, 'improvised': 6091, 'approving': 6092, 'complain': 6093, 'shoot': 6094, 'dismantled': 6095, 'last-minute': 6096, 'executions': 6097, 'andijan': 6098, 'restraint': 6099, 'evil': 6100, 'moammar': 6101, 'safely': 6102, 'protected': 6103, 'mob': 6104, 'detect': 6105, 'defenses': 6106, 'restricting': 6107, 'pursued': 6108, 'replacing': 6109, 'explanation': 6110, 'reliance': 6111, 'completion': 6112, 'exploiting': 6113, 'livelihood': 6114, 'mismanagement': 6115, 'character': 6116, 'martial': 6117, 'scrutiny': 6118, 'slavery': 6119, 'curtail': 6120, 'patrick': 6121, 'identification': 6122, 'cards': 6123, 'shipment': 6124, 'runway': 6125, 'supplier': 6126, 'acknowledge': 6127, 'notified': 6128, 'dates': 6129, 'efficiency': 6130, 'knee': 6131, '20-year-old': 6132, 'occupying': 6133, 'separation': 6134, 'pardoned': 6135, 'pardons': 6136, 'fao': 6137, 'reiterated': 6138, 'deliberately': 6139, 'taxi': 6140, 'milan': 6141, 'basilica': 6142, 'insulting': 6143, 'suggestions': 6144, 'oversight': 6145, 'festivities': 6146, 'medalist': 6147, 'irresponsible': 6148, 'eyadema': 6149, 'chronic': 6150, 'cleaning': 6151, 'load': 6152, 'enjoys': 6153, 'revenge': 6154, 'spur': 6155, 'musicians': 6156, 'ecuadorean': 6157, 'saeb': 6158, 'erekat': 6159, 'amend': 6160, 'turnout': 6161, 'theft': 6162, 'learning': 6163, 'slum': 6164, 'drug-related': 6165, 'radios': 6166, 'u.s.-funded': 6167, 'non-profit': 6168, 'regret': 6169, 'disappointed': 6170, 'sandra': 6171, "o'connor": 6172, 'moo-hyun': 6173, '25,000': 6174, 'automaker': 6175, 'shipped': 6176, 'confederation': 6177, 'entities': 6178, 'ultimately': 6179, 'warring': 6180, 'nikkei': 6181, 'hang': 6182, 'indexes': 6183, 'breakthrough': 6184, 'rush': 6185, 'hurting': 6186, 'hardest': 6187, 'cote': 6188, 'accession': 6189, 'polisario': 6190, 'prayed': 6191, 'holidays': 6192, 'drops': 6193, 'endured': 6194, '7,00,000': 6195, 'havens': 6196, 'bag': 6197, 'tolerance': 6198, 'text': 6199, 'expulsion': 6200, 'fix': 6201, 'photo': 6202, 'seizing': 6203, 'changing': 6204, 'regain': 6205, 'somewhat': 6206, 'filmmaker': 6207, 'viewers': 6208, 'credentials': 6209, 'arrives': 6210, '62': 6211, 'emancipation': 6212, 'banny': 6213, 'legislator': 6214, 'feeling': 6215, 'hayden': 6216, 'experiencing': 6217, 'ratings': 6218, 'favorable': 6219, 'jazeera': 6220, 'pressured': 6221, 'nyunt': 6222, 'rainfall': 6223, 'canary': 6224, 'resorts': 6225, '84': 6226, 'kostelic': 6227, 'fairly': 6228, 'confirms': 6229, 'retaliated': 6230, 'kids': 6231, 'shouting': 6232, 'beheaded': 6233, 'arbitration': 6234, 'teen': 6235, 'correct': 6236, 'ansar': 6237, 'restoration': 6238, 'fernandez': 6239, 'constructed': 6240, 'pharmaceutical': 6241, 'betancourt': 6242, 'nearing': 6243, '1.7': 6244, 'armenians': 6245, 'bystanders': 6246, 'handful': 6247, 'midwestern': 6248, 'meaning': 6249, 'jails': 6250, 'argument': 6251, 'ratko': 6252, 'timing': 6253, 'knew': 6254, 'clarke': 6255, 'conservatives': 6256, 'countryside': 6257, 'belonged': 6258, 'inciting': 6259, 'becomes': 6260, 'haditha': 6261, 'il': 6262, 'cat': 6263, 'madonna': 6264, 'committees': 6265, 'studio': 6266, 'grade': 6267, 'ap': 6268, 'sanctuary': 6269, 'medina': 6270, 'oklahoma': 6271, 'translator': 6272, 'liter': 6273, '#': 6274, 'transfers': 6275, 'adoption': 6276, 'christianity': 6277, 'sheep': 6278, 'lashed': 6279, 'populous': 6280, 'blues': 6281, 'highlight': 6282, 'completing': 6283, 'diverse': 6284, 'bounty': 6285, 'appearing': 6286, 'rubber': 6287, 'priorities': 6288, 'swaziland': 6289, 'lindsay': 6290, 'nawaz': 6291, 'prolonged': 6292, 'neighborhoods': 6293, 'voluntarily': 6294, 'respondents': 6295, 'undercover': 6296, 'patriot': 6297, 'justified': 6298, 'widow': 6299, 'inhabitants': 6300, 'marches': 6301, 'repeal': 6302, 'interested': 6303, 'banners': 6304, 'kivu': 6305, 'loved': 6306, 'perez': 6307, 'socialism': 6308, 'separating': 6309, 'prosperity': 6310, 'tickets': 6311, 'airplanes': 6312, 'minerals': 6313, 'quarterfinals': 6314, '54': 6315, 'select': 6316, 'qala': 6317, 'nuristan': 6318, 'elephants': 6319, 'teach': 6320, 'torrential': 6321, 'abramoff': 6322, 'resurgent': 6323, 'mqm': 6324, 'ashura': 6325, 'poison': 6326, 'apartheid': 6327, 'madoff': 6328, 'benzene': 6329, 'deter': 6330, 'praise': 6331, 'enterprises': 6332, 'ebola': 6333, 'downward': 6334, 'wishing': 6335, 'virtually': 6336, 'undisclosed': 6337, 'ohio': 6338, 'enhance': 6339, 'spector': 6340, 'ereli': 6341, 'capitol': 6342, 'zidane': 6343, 'haradinaj': 6344, 'saud': 6345, 'barzani': 6346, 'iceland': 6347, 'hole': 6348, 'allen': 6349, 'perform': 6350, 'bottles': 6351, 'broker': 6352, 'concentrated': 6353, 'francois': 6354, 'garcia': 6355, 'whereabouts': 6356, 'court-ordered': 6357, 'jungle': 6358, 'predicts': 6359, 'precautions': 6360, 'dramatic': 6361, 'witnessed': 6362, 'exported': 6363, 'transformed': 6364, 'bloodless': 6365, 'forestry': 6366, 'anti-japanese': 6367, 'inspired': 6368, 'acute': 6369, 'respiratory': 6370, 'misuse': 6371, 'hard-hit': 6372, 'doha': 6373, 'rampage': 6374, 'haitians': 6375, 'tribute': 6376, 'highlighting': 6377, 'poses': 6378, 'accepting': 6379, 'agrees': 6380, 'pre-dawn': 6381, '1.9': 6382, 'postal': 6383, 'jalalabad': 6384, '30-year-old': 6385, 'intervened': 6386, 'forged': 6387, 'manufacturer': 6388, 'resisted': 6389, 'ponte': 6390, 'suriname': 6391, 'striking': 6392, 'simultaneous': 6393, 'artists': 6394, 'technologies': 6395, 'authorize': 6396, 'censorship': 6397, 'khmer': 6398, 'outright': 6399, 'cayman': 6400, 'shape': 6401, '21st': 6402, 'greetings': 6403, 'engaging': 6404, 'toured': 6405, 'qaida': 6406, 'pilots': 6407, 'suppliers': 6408, 'michel': 6409, 'batch': 6410, 'deputies': 6411, 'substance': 6412, 'michelle': 6413, 'outdoor': 6414, 'pennsylvania': 6415, '93': 6416, 'remote-controlled': 6417, 'zenawi': 6418, '77-year-old': 6419, 'weapons-grade': 6420, 'dated': 6421, 'repressive': 6422, 'extent': 6423, 'gustav': 6424, 'liquid': 6425, 'announcing': 6426, 'oslo': 6427, '1968': 6428, 'fortified': 6429, 'transported': 6430, 'quantities': 6431, 'incomes': 6432, 'w.': 6433, 'rough': 6434, 'cancellation': 6435, 'observances': 6436, '1977': 6437, 'indiana': 6438, 'locals': 6439, 'unfairly': 6440, '1.4': 6441, 'chocolate': 6442, 'originated': 6443, '115': 6444, 'tent': 6445, 'shaken': 6446, 'interfere': 6447, 'dark': 6448, 'convince': 6449, 'jemaah': 6450, '202': 6451, 'weakening': 6452, 'mcdonald': 6453, 'parchin': 6454, 'torsello': 6455, 'downer': 6456, '13-year-old': 6457, 'fatality': 6458, 'repairs': 6459, '1971': 6460, 'faure': 6461, 'contentious': 6462, 'excellent': 6463, 'airplane': 6464, 'providencia': 6465, 'inappropriate': 6466, 'densely': 6467, 'hijacking': 6468, 'retiring': 6469, 'clan': 6470, 'really': 6471, 'ticket': 6472, 'crawford': 6473, 'stampede': 6474, 'competitive': 6475, 'mastrogiacomo': 6476, 'mediation': 6477, 'diarrhea': 6478, 'ministerial': 6479, 'salary': 6480, 'nuevo': 6481, 'indefinitely': 6482, 'deploying': 6483, 'stories': 6484, 'mashaal': 6485, 'richards': 6486, 'ball': 6487, '66': 6488, 'passport': 6489, 'walking': 6490, 'makers': 6491, 'pressed': 6492, 'mdc': 6493, 'blowing': 6494, 'continental': 6495, 'collision': 6496, 'punjab': 6497, 'containers': 6498, 'unusual': 6499, '1,700': 6500, 'russians': 6501, 'maria': 6502, 'madagascar': 6503, 'transferring': 6504, 'responsibilities': 6505, 'yao': 6506, 'nba': 6507, 'telescope': 6508, 'aqsa': 6509, 'celebrates': 6510, 'parades': 6511, '330': 6512, '2014': 6513, 'major-general': 6514, 'documentary': 6515, 'classic': 6516, 'transactions': 6517, 'swap': 6518, '78': 6519, 'granma': 6520, 'bajaur': 6521, 'harcourt': 6522, 'alonso': 6523, 'steinmeier': 6524, 'robbery': 6525, 'dual': 6526, 'physician': 6527, 'describing': 6528, 'slashed': 6529, 'tunisia': 6530, 'conversations': 6531, 'coastline': 6532, 'declaring': 6533, 'perceived': 6534, 'manufacture': 6535, 'freeing': 6536, 'gets': 6537, 'servicing': 6538, 'harassment': 6539, 'disclose': 6540, 'decision-making': 6541, 'deteriorated': 6542, 'akhtar': 6543, 'zhang': 6544, 'skating': 6545, 'usa': 6546, 'zimbabwean': 6547, 'poisoning': 6548, 'harare': 6549, 'syed': 6550, 'co.': 6551, 'modest': 6552, 'soft': 6553, 'maneuvers': 6554, 'u.n.-backed': 6555, 'securing': 6556, 'lending': 6557, 'multi-party': 6558, 'clues': 6559, 'captors': 6560, 'ministries': 6561, 'cousin': 6562, 'reconsider': 6563, 'random': 6564, 'merck': 6565, 'luanda': 6566, 'anywhere': 6567, 'cantoni': 6568, 'assad': 6569, 'nephew': 6570, 'three-month': 6571, 'secession': 6572, 'uncertainty': 6573, 'cigarettes': 6574, 'scrap': 6575, 'impasse': 6576, 'unresolved': 6577, 'cruel': 6578, 'al-fitr': 6579, 'chair': 6580, 'filing': 6581, 'sarajevo': 6582, 'signal': 6583, 'warships': 6584, '78-year-old': 6585, 'insufficient': 6586, 'bermuda': 6587, 'savings': 6588, 'asleep': 6589, 'raises': 6590, 'manage': 6591, 'rahman': 6592, 'compensate': 6593, 'harvest': 6594, 'slated': 6595, 'renegade': 6596, 'understand': 6597, '4.5': 6598, 'exhausted': 6599, 'democracies': 6600, 'style': 6601, 'retreat': 6602, 'peacekeeper': 6603, 'verify': 6604, 'migrant': 6605, 'captives': 6606, 'maskhadov': 6607, '135': 6608, 'household': 6609, 'exceed': 6610, '3.6': 6611, 'checked': 6612, 'mud': 6613, 'wells': 6614, 'pearl': 6615, 'renewable': 6616, 'detonate': 6617, 'absentia': 6618, 'euphrates': 6619, 'desperate': 6620, 'jong': 6621, 'spots': 6622, 'assisting': 6623, 'fee': 6624, 'mainstream': 6625, 'publish': 6626, 'confidential': 6627, 'moral': 6628, 'balkans': 6629, 'memo': 6630, 'deadlocked': 6631, 'difficulty': 6632, 'defied': 6633, 'grounded': 6634, 'examined': 6635, 'toyota': 6636, 'zoo': 6637, 'panda': 6638, 'lie': 6639, '148': 6640, 'distributed': 6641, 'cook': 6642, 'columbia': 6643, 'watching': 6644, 'mohmand': 6645, 'perpetrators': 6646, 'youngest': 6647, 'algiers': 6648, 'courage': 6649, 'lights': 6650, 'distant': 6651, 'taught': 6652, 'techniques': 6653, 'quito': 6654, 'alternate': 6655, 'barcelona': 6656, 'pattani': 6657, 'yala': 6658, 'nicaraguan': 6659, 'disbanded': 6660, 'undermining': 6661, 'flame': 6662, 'roles': 6663, 'bhutan': 6664, 'tiananmen': 6665, 'keeps': 6666, 'stan': 6667, 'entertainment': 6668, 'decade-long': 6669, 'accidental': 6670, 'households': 6671, 'guangzhou': 6672, 'openly': 6673, 'seekers': 6674, 'heritage': 6675, 'author': 6676, 'stream': 6677, 'reacted': 6678, 'settler': 6679, 'disgraced': 6680, 'harper': 6681, 'rican': 6682, 'transparent': 6683, 'yuan': 6684, 'ruins': 6685, 'harbin': 6686, 'advisory': 6687, 'involve': 6688, 'qualified': 6689, 'oxfam': 6690, 'beta': 6691, 'paulo': 6692, 'ma': 6693, 'test-fired': 6694, 'hawk': 6695, 'livni': 6696, 'projected': 6697, 'tortoise': 6698, 'chevron': 6699, 'nationalization': 6700, 'hate': 6701, 'liu': 6702, 'booming': 6703, 'carefully': 6704, 'hakimi': 6705, 'sony': 6706, 'wolfowitz': 6707, 'mistreatment': 6708, 'whittington': 6709, 'shoulder': 6710, 'juarez': 6711, 'el-sheikh': 6712, 'venus': 6713, 'handover': 6714, 'hwang': 6715, 'levees': 6716, 'seal': 6717, 'siniora': 6718, 'missouri': 6719, 'epicenter': 6720, 'disorder': 6721, 'screening': 6722, 'advocate': 6723, '2,200': 6724, 'zapatista': 6725, 'washington-based': 6726, 'objective': 6727, 'allegiance': 6728, 'combatants': 6729, '1963': 6730, 'imminent': 6731, 'copies': 6732, 'deciding': 6733, 'authoritarian': 6734, 'capturing': 6735, 'locally': 6736, 'slipped': 6737, 'sniper': 6738, 'visas': 6739, 'friendship': 6740, 'determination': 6741, '20-year': 6742, 'unchanged': 6743, 'unhurt': 6744, 'forensic': 6745, 'achievements': 6746, 'honors': 6747, 'top-seeded': 6748, '04-jun': 6749, 'contributing': 6750, 'quarters': 6751, 'consistent': 6752, 'inability': 6753, 'unofficial': 6754, '104': 6755, 'six-day': 6756, 'sky': 6757, 'guardsmen': 6758, 'dostum': 6759, 'resolutions': 6760, 'disarmed': 6761, 'miguel': 6762, 'deport': 6763, 'emphasized': 6764, 'airways': 6765, 'flat': 6766, 'testify': 6767, 'sustainable': 6768, 'covers': 6769, 'arrangements': 6770, 'zagreb': 6771, 'literature': 6772, 'proclaimed': 6773, 'occur': 6774, 'austerity': 6775, 'recruitment': 6776, 'worsened': 6777, 'investing': 6778, 'respected': 6779, 'first-round': 6780, 'pakistan-based': 6781, 'justify': 6782, 'nazis': 6783, 'arguments': 6784, 'recognizing': 6785, 'batticaloa': 6786, 'rahim': 6787, 'minister-designate': 6788, 'ian': 6789, 'golan': 6790, 'stabilizing': 6791, 'withdrawals': 6792, 'boucher': 6793, 're-run': 6794, 'convinced': 6795, 'clandestine': 6796, 'titles': 6797, 'cafe': 6798, 'al-sistani': 6799, 'hijackers': 6800, 'weaponry': 6801, 'repatriated': 6802, 'anti-aircraft': 6803, 'satisfied': 6804, 'ernesto': 6805, 'meles': 6806, 'nationally': 6807, 'dolphins': 6808, 'solomon': 6809, 'instituted': 6810, '1,100': 6811, 'collection': 6812, 'creole': 6813, 'moroccans': 6814, 'hydropower': 6815, 'amended': 6816, 'exploitation': 6817, 'successive': 6818, 'siberia': 6819, 'generations': 6820, 'flowers': 6821, 'adel': 6822, 'longest': 6823, 'buildup': 6824, 'ivorian': 6825, 'doors': 6826, 'ingredient': 6827, 'unlike': 6828, 'hungry': 6829, 'en': 6830, 'al-hariri': 6831, 'director-general': 6832, 'muslim-majority': 6833, 'dioxide': 6834, 'inner': 6835, 'nightclub': 6836, 'medics': 6837, 'detain': 6838, 'counsel': 6839, 'pardon': 6840, 'dangers': 6841, 'blasphemous': 6842, 'ignore': 6843, 'affecting': 6844, 'nordic': 6845, 'length': 6846, 'beauty': 6847, 'brand': 6848, 'box': 6849, 'hawaii': 6850, 'richest': 6851, 'advice': 6852, 'fact-finding': 6853, 'year-long': 6854, 'scattered': 6855, 'stockholm': 6856, 'virtual': 6857, 'beckham': 6858, 'assisted': 6859, 'indies': 6860, 'austro-hungarian': 6861, 'dissolution': 6862, 'dynasty': 6863, 'bishop': 6864, 'exceeded': 6865, 'lesotho': 6866, 'stag': 6867, 'height': 6868, 'sank': 6869, 'strategies': 6870, 'eradication': 6871, 'regrets': 6872, 'measured': 6873, 'pro-independence': 6874, 'boarded': 6875, 'lacked': 6876, 'wedding': 6877, 'sections': 6878, 'objected': 6879, 'generating': 6880, 'mired': 6881, 'tainted': 6882, 'excuse': 6883, 'replaces': 6884, 'postponement': 6885, 'explosives-laden': 6886, 'dig': 6887, 'composite': 6888, 'seng': 6889, 'frankfurt': 6890, 'tim': 6891, 'spaniard': 6892, "d'ivoire": 6893, 'mild': 6894, 'ate': 6895, 'wish': 6896, 'chairmanship': 6897, 'reflects': 6898, 'mid': 6899, 'eruption': 6900, 'collided': 6901, 'maharashtra': 6902, 'bombay': 6903, 'sued': 6904, 'connecticut': 6905, 'warehouse': 6906, 'safeguard': 6907, 'islamiyah': 6908, 'collect': 6909, 'pride': 6910, 'ordinary': 6911, 'borrowing': 6912, 'slam': 6913, '19-year-old': 6914, 'andre': 6915, 'dissolve': 6916, 'carnival': 6917, 'fuel-efficient': 6918, 'youssef': 6919, 'jupiter': 6920, 'unpopular': 6921, 'battery': 6922, 'looting': 6923, 'hoshyar': 6924, 'covert': 6925, 'dakar': 6926, 'honduran': 6927, 'fatally': 6928, 'bids': 6929, 'intensive': 6930, '1000': 6931, 'beasts': 6932, 'swift': 6933, 'lenders': 6934, 'presiding': 6935, 'small-scale': 6936, 'releases': 6937, 'closest': 6938, 'reputation': 6939, 'mary': 6940, 'disappearance': 6941, 'contends': 6942, 'jobless': 6943, '119': 6944, 'application': 6945, 'farther': 6946, 'melting': 6947, 'consortium': 6948, 'wa': 6949, 'minnesota': 6950, 'nahr': 6951, '97': 6952, 'rejection': 6953, 'opens': 6954, 'throw': 6955, 'hurriyat': 6956, 'chairs': 6957, 'convenes': 6958, 'touched': 6959, 'objectives': 6960, 'squads': 6961, 'drafted': 6962, '77': 6963, 'lull': 6964, 'derailed': 6965, '5.7': 6966, 'baath': 6967, 'auschwitz': 6968, 'kerekou': 6969, 'brownfield': 6970, 'fluids': 6971, "'ll": 6972, 'self-imposed': 6973, 'embezzlement': 6974, 'bertel': 6975, 'standings': 6976, '650': 6977, 'liberty': 6978, 'telegraph': 6979, 'quarantine': 6980, 'contributes': 6981, 'neutrality': 6982, 'honored': 6983, 'companion': 6984, "'m": 6985, 'carolyn': 6986, 'journalism': 6987, 'relating': 6988, 'smuggle': 6989, 'soviet-era': 6990, 'readiness': 6991, 'bracing': 6992, 'vaccinations': 6993, 'conclude': 6994, 'bashir': 6995, 'indictments': 6996, 'hijackings': 6997, 'exchanges': 6998, 'yes': 6999, 'timor': 7000, 'tracking': 7001, 'displayed': 7002, 'oversaw': 7003, 'disappointing': 7004, 'depart': 7005, 'stomach': 7006, 'constructive': 7007, 'contracting': 7008, 'passports': 7009, 'centrist': 7010, 'briton': 7011, 'raped': 7012, 'singled': 7013, '14th': 7014, 'airborne': 7015, 'anti-secession': 7016, 'le': 7017, 'breach': 7018, 'indebted': 7019, '5.3': 7020, 'wages': 7021, 'reelected': 7022, 'picking': 7023, 'isfahan': 7024, 'appointments': 7025, 'competitors': 7026, '6.4': 7027, 'brad': 7028, 'boston': 7029, 'afterward': 7030, 'contents': 7031, 'basescu': 7032, 'collecting': 7033, 'nazarbayev': 7034, 'violates': 7035, 'pence': 7036, 'neutral': 7037, 'preserving': 7038, 'dogs': 7039, 'lightning': 7040, 'engines': 7041, 'pre-election': 7042, 'ituri': 7043, 'appearances': 7044, 'comedy': 7045, 'rodrigo': 7046, 'rid': 7047, 'servicemen': 7048, 'roll': 7049, 'mehmet': 7050, 'lohan': 7051, 'chrysler': 7052, 'starvation': 7053, 'zambian': 7054, 'banda': 7055, 'guilders': 7056, 'flows': 7057, 'bambang': 7058, 'hussain': 7059, 'perhaps': 7060, 'extends': 7061, 'kumaratunga': 7062, 'regard': 7063, 'karami': 7064, 'posing': 7065, 'outlined': 7066, 'accountable': 7067, 'sweeping': 7068, 'crushed': 7069, 'strapped': 7070, 'narathiwat': 7071, 'archaeologists': 7072, 'protestant': 7073, 'pelosi': 7074, 'surveys': 7075, 'packages': 7076, 'drills': 7077, 'dating': 7078, 'brent': 7079, '87': 7080, 'sits': 7081, '23-year-old': 7082, 'saved': 7083, 'operatives': 7084, 'proceeds': 7085, 'motor': 7086, '15-member': 7087, 'anthony': 7088, 'gibraltar': 7089, 'quality': 7090, 'everest': 7091, 'second-largest': 7092, 'dependency': 7093, 'pigs': 7094, 'admit': 7095, 'intensify': 7096, 'analysis': 7097, 'astronaut': 7098, 'addresses': 7099, 'bloodiest': 7100, 'mounted': 7101, 'jennifer': 7102, 'slump': 7103, 'commuter': 7104, 'bomb-making': 7105, 'valued': 7106, 'fiji': 7107, 'arrivals': 7108, 'hipc': 7109, 'abe': 7110, 'torched': 7111, 'britney': 7112, 'propose': 7113, 'proven': 7114, 'textiles': 7115, 'instances': 7116, 'licenses': 7117, 'cruise': 7118, 'scope': 7119, 'armor': 7120, 'chinook': 7121, 'nasrallah': 7122, 'ethics': 7123, 'tashkent': 7124, 'termed': 7125, 'mehdi': 7126, 'burial': 7127, 'flotilla': 7128, 'prey': 7129, 'carriers': 7130, 'lashkar': 7131, 'kurmanbek': 7132, 'bidding': 7133, 'kerik': 7134, 'slashing': 7135, 'poured': 7136, 'songhua': 7137, 'meteorological': 7138, 'stadiums': 7139, 'anc': 7140, 'nelson': 7141, 'hay': 7142, 'setback': 7143, 'mv': 7144, 'baluch': 7145, 'detroit': 7146, 'evacuees': 7147, 'flush': 7148, 'same-sex': 7149, 'toure': 7150, 'mahathir': 7151, 'najib': 7152, 'atop': 7153, 'oas': 7154, 'stored': 7155, 'ideas': 7156, 'qinghai': 7157, 'finances': 7158, 'somewhere': 7159, 'reality': 7160, 'macedonian': 7161, 'bowler': 7162, '94': 7163, 'locate': 7164, 'adds': 7165, 'geese': 7166, 'christ': 7167, 'hero': 7168, 'channels': 7169, 'brunei': 7170, 'sponsor': 7171, 'zero': 7172, 'campaigns': 7173, 'maduro': 7174, 'sizable': 7175, 'blackout': 7176, '\x92s': 7177, 'exactly': 7178, 'hiroshima': 7179, 'petraeus': 7180, 'laying': 7181, 'recall': 7182, 'upjohn': 7183, 'hart': 7184, 'disagreement': 7185, 'roberts': 7186, 'bagapsh': 7187, 'laredo': 7188, 'laptop': 7189, 'hicks': 7190, 'tata': 7191, 'kcna': 7192, 'finalize': 7193, 'center-left': 7194, 'marcos': 7195, '28-year-old': 7196, 'al-arabiya': 7197, 'scholars': 7198, 'harassed': 7199, 'whenever': 7200, 'vaccinate': 7201, '60,000': 7202, 'assaulted': 7203, 'crippled': 7204, 'enabled': 7205, 'partially': 7206, 'trap': 7207, 'arranged': 7208, 'sevan': 7209, 'diverted': 7210, 'proclamation': 7211, 'johnston': 7212, 'short-range': 7213, 'invade': 7214, 'collaboration': 7215, 'sher': 7216, 'ethical': 7217, 'u.s.-made': 7218, 'quarterly': 7219, 'indicating': 7220, 'dispersed': 7221, 'affair': 7222, 'funded': 7223, 'corpses': 7224, 'abducting': 7225, 'blake': 7226, '03-jun': 7227, 'inequality': 7228, 'authorizes': 7229, 'unannounced': 7230, 'offense': 7231, 'campbell': 7232, '185': 7233, 'degree': 7234, 'enjoying': 7235, 'victoria': 7236, '1917': 7237, 'revered': 7238, 'communism': 7239, 'geographic': 7240, 'commanded': 7241, 'juba': 7242, 'shouted': 7243, 'docked': 7244, 'purportedly': 7245, 'midday': 7246, 'hindered': 7247, 'benefited': 7248, 'refuse': 7249, 'momentum': 7250, 'transaction': 7251, 'historically': 7252, 'wales': 7253, 'pirate': 7254, 'rescheduled': 7255, 'medium': 7256, 'grenada': 7257, 'fischer': 7258, 'vincent': 7259, 'criticizes': 7260, 'volcanic': 7261, 'memory': 7262, 'manipulating': 7263, 'parking': 7264, 'croat': 7265, 'bond': 7266, 'harming': 7267, 'saving': 7268, 'mouse': 7269, 'videotaped': 7270, 'lacking': 7271, 'papal': 7272, 'valid': 7273, 'detonating': 7274, 'twelve': 7275, 'anticipated': 7276, 'unauthorized': 7277, 'audiotape': 7278, 'raged': 7279, 'optimism': 7280, 'modernization': 7281, 'verde': 7282, 'disciplinary': 7283, 'qualifying': 7284, 'cameras': 7285, 'ventures': 7286, 'handle': 7287, 'confirming': 7288, 'virgin': 7289, 'apparel': 7290, 'proximity': 7291, 'boss': 7292, 'neck': 7293, 'condemn': 7294, 'venue': 7295, 'government-controlled': 7296, 'colleague': 7297, 'assessing': 7298, 'edward': 7299, 'stress': 7300, 'makeshift': 7301, 'rapes': 7302, '1969': 7303, '11,000': 7304, 'masters': 7305, 'pull-out': 7306, '103': 7307, 'nutrition': 7308, 'examining': 7309, 'applications': 7310, 'logar': 7311, 'remembered': 7312, 'munitions': 7313, 'installation': 7314, 'munich': 7315, 'second-in-command': 7316, 'regulate': 7317, 'branches': 7318, 'complicity': 7319, 'shocked': 7320, 'qadeer': 7321, 'u': 7322, 'stuart': 7323, 'indications': 7324, 'realized': 7325, 'magna': 7326, 'mcalpine': 7327, 'dividend': 7328, 'consulting': 7329, '10-day': 7330, 'historical': 7331, 'principality': 7332, '1966': 7333, 'commanding': 7334, '6,00,000': 7335, 'welcomes': 7336, 'campaigned': 7337, 'laboratories': 7338, 'allocated': 7339, 'p.m.': 7340, 'miers': 7341, 'assigned': 7342, 'infant': 7343, 'charities': 7344, 'newly-elected': 7345, 'capitals': 7346, 'trash': 7347, 'dana': 7348, 'understands': 7349, 'explained': 7350, 'memorandum': 7351, 'hydroelectric': 7352, 'mamadou': 7353, 'vacuum': 7354, 'golden': 7355, 'holiest': 7356, 'whatever': 7357, 'pharmaceuticals': 7358, 'incorporated': 7359, 'seasonal': 7360, 'rebound': 7361, 'retains': 7362, 'yellow': 7363, 'kitts': 7364, 'champions': 7365, 'rarely': 7366, 'possessing': 7367, 'tracks': 7368, 'pages': 7369, 'hazardous': 7370, 'taro': 7371, 'wrapping': 7372, 'digging': 7373, 'feels': 7374, 'highlighted': 7375, 'patriarch': 7376, 'joins': 7377, 'sharapova': 7378, 'taleban-led': 7379, 'mainstay': 7380, 'vocal': 7381, '114': 7382, 'crippling': 7383, 'talked': 7384, 'galaxies': 7385, 'aggressively': 7386, 'dealt': 7387, 'featuring': 7388, 'fasting': 7389, 'boycotting': 7390, 'malik': 7391, 'decades-old': 7392, 'convened': 7393, 'feel': 7394, 'grab': 7395, 'bigger': 7396, 'tamiflu': 7397, 'offensives': 7398, 'militancy': 7399, 'lewis': 7400, 'al-masri': 7401, 'idol': 7402, 'axe': 7403, 'sang': 7404, 'employ': 7405, 'confinement': 7406, 'swallow': 7407, 'injunction': 7408, 'pitcher': 7409, 'milestone': 7410, 'amr': 7411, 'outlining': 7412, 'ignoring': 7413, 'gradual': 7414, 'debts': 7415, 'shifting': 7416, 'government-run': 7417, 'intimidation': 7418, 'afraid': 7419, 'laura': 7420, 'reunite': 7421, 'hughes': 7422, 'throat': 7423, 'discharged': 7424, 'backs': 7425, 'supervision': 7426, 'secessionist': 7427, 'chan': 7428, 'retire': 7429, 'academic': 7430, '3.8': 7431, 'hiv-positive': 7432, 'applying': 7433, 'graft': 7434, 'focuses': 7435, 'circuit': 7436, 'medicines': 7437, 'guidance': 7438, 'heaviest': 7439, 'morris': 7440, 'hailing': 7441, 'thieves': 7442, 'racial': 7443, 'mar': 7444, 'nestor': 7445, 'multi-million': 7446, 'azahari': 7447, '2,50,000': 7448, 'publicist': 7449, 'lowered': 7450, 'diversified': 7451, 'feathers': 7452, 'alaska': 7453, 'denounce': 7454, 'andhra': 7455, 'midwest': 7456, 'chilumpha': 7457, 'would-be': 7458, 'three-year-old': 7459, 'wound': 7460, 'decrease': 7461, 'dining': 7462, 'revived': 7463, 'arnold': 7464, 'seeds': 7465, 'interrogated': 7466, 'viruses': 7467, 'stricken': 7468, '123': 7469, '1930s': 7470, 'toilet': 7471, 'halutz': 7472, 'turban': 7473, 'falls': 7474, 'bargain': 7475, 'lima': 7476, 'thein': 7477, 'owes': 7478, 'sewage': 7479, 'ophelia': 7480, 'classify': 7481, 'sharia': 7482, 'compounds': 7483, 'stand-off': 7484, '1953': 7485, 'subjected': 7486, 'unite': 7487, 'explore': 7488, 'authorization': 7489, 'blindfolded': 7490, 'kid': 7491, 'michigan': 7492, 'existed': 7493, 'labeled': 7494, 'famed': 7495, 'nets': 7496, 'benchmark': 7497, 'sophisticated': 7498, 'pressuring': 7499, 'molestation': 7500, 'approaches': 7501, 'stoppage': 7502, 'aref': 7503, '91': 7504, 'indoor': 7505, 'hansen': 7506, 'contender': 7507, 'ryan': 7508, 'insist': 7509, 'mercantile': 7510, 'declares': 7511, 'elite': 7512, '8.5': 7513, 'remainder': 7514, 'one-time': 7515, 'shannon': 7516, 'principal': 7517, 'receipts': 7518, 'bailout': 7519, 'influenced': 7520, 'secondary': 7521, 'colonies': 7522, 'insisting': 7523, 'notably': 7524, 'milinkevich': 7525, 'harboring': 7526, 'belize': 7527, 'natwar': 7528, 'quarantined': 7529, 'obstacle': 7530, 'slate': 7531, 'sacred': 7532, 'garrigues': 7533, 'charitable': 7534, 'left-wing': 7535, 'salman': 7536, 'resistant': 7537, 're-vote': 7538, 'perry': 7539, 'avalanche': 7540, 'launcher': 7541, 'biotechnology': 7542, 'shalom': 7543, 'spokesmen': 7544, 'adjacent': 7545, 'exacerbated': 7546, 'flared': 7547, 'clijsters': 7548, 'migrating': 7549, 'ward': 7550, 'grip': 7551, 'liaoning': 7552, 'lyon': 7553, 'turns': 7554, 'guests': 7555, 'belongs': 7556, 'assertion': 7557, 'submerged': 7558, 'f-16': 7559, 'escalation': 7560, 'advertising': 7561, 'barry': 7562, 'steal': 7563, 'accurate': 7564, '73,000': 7565, 'lodge': 7566, 'diego': 7567, 'levy': 7568, 'baseless': 7569, 'ceremonial': 7570, 'susilo': 7571, 'grouping': 7572, 'recruit': 7573, 'moya': 7574, 'third-seeded': 7575, 'downed': 7576, 'sells': 7577, 'upward': 7578, 'cub': 7579, 'pandas': 7580, 'forcibly': 7581, 'vowing': 7582, 'shields': 7583, 'catastrophic': 7584, 'hadley': 7585, 'cemetery': 7586, 'muslim-dominated': 7587, 'lawless': 7588, 'recess': 7589, 'missionaries': 7590, 'courses': 7591, 'massoud': 7592, 'clarkson': 7593, '1.6': 7594, 'tribe': 7595, 'disney': 7596, 'eight-year': 7597, 'dominance': 7598, 'eduardo': 7599, 'intend': 7600, '1970': 7601, 'disguised': 7602, 'wear': 7603, 'gaming': 7604, 'yucatan': 7605, 'apologize': 7606, 'sacu': 7607, 'royalties': 7608, 'vanuatu': 7609, 'kuala': 7610, 'lumpur': 7611, 'arts': 7612, 'surviving': 7613, 'hide': 7614, 'movies': 7615, 'urges': 7616, 'unconstitutional': 7617, 'familiar': 7618, 'background': 7619, 'manhattan': 7620, 'errors': 7621, 'jorge': 7622, 'database': 7623, 'automobile': 7624, 'conventional': 7625, 'a.m.': 7626, 'seizures': 7627, 'landless': 7628, 'waited': 7629, 'sat': 7630, 'soul': 7631, 'determining': 7632, 'veracruz': 7633, 'namibian': 7634, 'spears': 7635, 'federline': 7636, 'holdings': 7637, 'ambushes': 7638, 'pdvsa': 7639, 'military-ruled': 7640, 'buenos': 7641, 'aires': 7642, 'worsening': 7643, 'wrongly': 7644, 'awami': 7645, 'waved': 7646, 'cluster': 7647, 'hunan': 7648, 'luther': 7649, 'katsav': 7650, 'lobbying': 7651, 'irving': 7652, 'guarding': 7653, 'orbit': 7654, 'santos': 7655, 'exporters': 7656, 'oceans': 7657, 'tommy': 7658, 'immunity': 7659, 'requiring': 7660, 'reshuffle': 7661, 'negligence': 7662, 'legend': 7663, 'expenditures': 7664, '7.5': 7665, 'imbalance': 7666, 'wardak': 7667, 'senegalese': 7668, 'automatically': 7669, 'pill': 7670, 'operators': 7671, 'slide': 7672, 'purchases': 7673, 'exxonmobil': 7674, 'knows': 7675, 'minster': 7676, 'confession': 7677, 'overs': 7678, 'opera': 7679, 'suggestion': 7680, 'comprises': 7681, 'hilla': 7682, 'mohsen': 7683, 'outspoken': 7684, 'oversees': 7685, 'kozulin': 7686, 'impressive': 7687, 'arrange': 7688, 'dividing': 7689, 'scoring': 7690, 'observance': 7691, 'hikers': 7692, 'decades-long': 7693, 'retained': 7694, 'zalmay': 7695, 'phil': 7696, 'favors': 7697, 'b.c.': 7698, 'wishes': 7699, 'retailer': 7700, 'videos': 7701, 'tea': 7702, 'mall': 7703, 'posters': 7704, 'jamal': 7705, 'embezzling': 7706, 'shetty': 7707, 'recommend': 7708, 'mad': 7709, 'enclaves': 7710, '\x96': 7711, 'melilla': 7712, 'cheek': 7713, 'relationships': 7714, 'judgment': 7715, 'machines': 7716, 'prescription': 7717, 'mesa': 7718, '1.8': 7719, 'underwater': 7720, 'manned': 7721, 'financed': 7722, 'assembled': 7723, 'coincides': 7724, 'lucia': 7725, 'prohibited': 7726, 'mercy': 7727, 'jimmy': 7728, 'thaw': 7729, 'diversification': 7730, 'functions': 7731, 'perfect': 7732, 'provoked': 7733, 'lighting': 7734, 'consolidate': 7735, 'alliances': 7736, 'allegation': 7737, 'automatic': 7738, 'unemployed': 7739, 'inclusion': 7740, 'migration': 7741, 'bandits': 7742, 'nancy': 7743, 'ivo': 7744, 'mayer': 7745, 'murray': 7746, 'gender': 7747, 'pregnant': 7748, 'cross-country': 7749, 'anti-doping': 7750, 'salt': 7751, 'low-lying': 7752, 'hire': 7753, 'displays': 7754, 'exists': 7755, 'skies': 7756, 'rushing': 7757, 'brutality': 7758, 'kurram': 7759, 'hamdi': 7760, 'issac': 7761, 'warlord': 7762, 'praying': 7763, 'al-adha': 7764, '1961': 7765, 'achieving': 7766, '737': 7767, 'quartet': 7768, 'turkmen': 7769, 'ante': 7770, 'boundary': 7771, 'explored': 7772, 'periods': 7773, 'cheered': 7774, 'inhabited': 7775, 'smallest': 7776, 'touring': 7777, 'sake': 7778, 'caution': 7779, 'gogh': 7780, 'bangalore': 7781, 'stimulate': 7782, '12-year-old': 7783, 'architect': 7784, 'resist': 7785, '240': 7786, 'abide': 7787, 'muslim-croat': 7788, '16,000': 7789, 'compassion': 7790, 'enact': 7791, 'captive': 7792, 'marwan': 7793, '9th': 7794, 'walid': 7795, 'signals': 7796, 'paralysis': 7797, 'metals': 7798, 'heated': 7799, 'vacant': 7800, 'biological': 7801, 'substances': 7802, 'nuclear-capable': 7803, 'paktia': 7804, 'prestigious': 7805, 'vanunu': 7806, 'moments': 7807, 'bells': 7808, 'coincided': 7809, 'discourage': 7810, 'conspiring': 7811, 'undermined': 7812, 'radicals': 7813, 'multi-candidate': 7814, 'weaknesses': 7815, 'regimes': 7816, 'anti-u.s.': 7817, 'jolo': 7818, 'economics': 7819, 'barrage': 7820, 'petrobras': 7821, 'equipped': 7822, 'analyst': 7823, 'certified': 7824, '128': 7825, 'mauritius': 7826, 'slums': 7827, 'federated': 7828, 'outlook': 7829, 'equatorial': 7830, 'phased': 7831, 'intimidated': 7832, 'alassane': 7833, 'narrowed': 7834, 'aerial': 7835, 'galan': 7836, 'tone': 7837, 'cease': 7838, 'feelings': 7839, 'reinstate': 7840, 'curbing': 7841, 'distribute': 7842, 'spike': 7843, 'excess': 7844, 'possess': 7845, 'bekaa': 7846, 'commando': 7847, 'lieutenants': 7848, 'mandatory': 7849, 'gases': 7850, 'maher': 7851, 'instructed': 7852, 'suspicions': 7853, 'taped': 7854, 'sudden': 7855, 'attractive': 7856, 'pump': 7857, 'tool': 7858, 'recruited': 7859, 'meshaal': 7860, 'unusually': 7861, 'shattered': 7862, 'cafes': 7863, 'marijuana': 7864, 'highest-ranking': 7865, 'strictly': 7866, 'adnan': 7867, 'al-dulaimi': 7868, 'by-election': 7869, 'blankets': 7870, 'trio': 7871, 'write': 7872, 'vow': 7873, 'withheld': 7874, 'overhead': 7875, 'shareholder': 7876, 'unsuccessfully': 7877, '1929': 7878, 'transformation': 7879, 'constituent': 7880, 'iii': 7881, 'budgetary': 7882, 'inquired': 7883, 'indicators': 7884, 'grieving': 7885, 'govern': 7886, 'globalization': 7887, 'venezuelans': 7888, 'reinforcements': 7889, 'barracks': 7890, 'bulldozers': 7891, 'sincere': 7892, 'confined': 7893, 'cycle': 7894, 'feud': 7895, 'suppression': 7896, 'third-largest': 7897, 'lung': 7898, 'maung': 7899, 'perino': 7900, 'fernando': 7901, 'martian': 7902, 'assassinations': 7903, 'self-defense': 7904, 'beheadings': 7905, 'wetlands': 7906, 'cheering': 7907, 'wealthiest': 7908, 'monaco': 7909, 'gambling': 7910, 'deserve': 7911, 'cried': 7912, 'answered': 7913, 'thwart': 7914, 'horns': 7915, 'converted': 7916, 'privacy': 7917, 'skin': 7918, 'citrus': 7919, 'fruits': 7920, 'volunteer': 7921, 'elizabeth': 7922, 'joy': 7923, 'cathedral': 7924, 'method': 7925, 'mid-1990s': 7926, 'miss': 7927, 'apartments': 7928, 'camera': 7929, 'ravalomanana': 7930, 'al-hakim': 7931, 'blows': 7932, 'oregon': 7933, 'slash': 7934, 'chaotic': 7935, 'spoken': 7936, 'hutus': 7937, 'tariq': 7938, 'sympathizers': 7939, 'gatherings': 7940, 'nepali': 7941, 'persuading': 7942, 'insult': 7943, 'assaulting': 7944, 'finnish': 7945, 'mend': 7946, 'karl': 7947, 'frank-walter': 7948, 'timothy': 7949, 'publisher': 7950, 'third-quarter': 7951, 'educational': 7952, 'a.d.': 7953, '1956': 7954, 'prague': 7955, 'hanged': 7956, 'downgraded': 7957, 'pockets': 7958, 'unfounded': 7959, 'crushing': 7960, 'nominate': 7961, 'blazes': 7962, 'havel': 7963, 'desmond': 7964, 'tutu': 7965, 'tube': 7966, 'manger': 7967, 'siblings': 7968, 'seem': 7969, 'shutting': 7970, 'mechanism': 7971, 'mill': 7972, 'reviewed': 7973, 'h1n1': 7974, 'reagan': 7975, 'otherwise': 7976, 'disapprove': 7977, 'disruption': 7978, 'sarah': 7979, 'combines': 7980, 'arson': 7981, 'plata': 7982, 'uss': 7983, 'mohamad': 7984, 'husin': 7985, 'formula': 7986, '24-year-old': 7987, '450': 7988, 'eleven': 7989, 'nov.': 7990, 'fuad': 7991, 'prudent': 7992, 'color': 7993, 'khabarovsk': 7994, "shi'ite-dominated": 7995, 'non-binding': 7996, 'creditors': 7997, 'compatriot': 7998, 'valuable': 7999, '2.4': 8000, 'commemorations': 8001, '1.1': 8002, 'maps': 8003, 'update': 8004, 'page': 8005, 'crimea': 8006, 'lucrative': 8007, '6-0': 8008, 'panamanian': 8009, 'sultan': 8010, 'dehydration': 8011, 'roadblocks': 8012, 'bodily': 8013, 'aiming': 8014, 'sight': 8015, 'kibo': 8016, 'petitioned': 8017, '1965': 8018, 'beaches': 8019, 'overrun': 8020, 'tasked': 8021, 'permitted': 8022, 'easy': 8023, 'equals': 8024, 'utilities': 8025, 'frogs': 8026, 'dmitry': 8027, 'robbed': 8028, 'nicotine': 8029, 'concerning': 8030, 'accomplished': 8031, 'trends': 8032, 'marshall': 8033, 'uri': 8034, 'shaky': 8035, 'yahoo': 8036, 'switch': 8037, 'opinions': 8038, '190': 8039, 'totally': 8040, 'depending': 8041, 'ceased': 8042, 'newsweek': 8043, 'prachanda': 8044, 'bananas': 8045, 'lu': 8046, 'then-president': 8047, 'hanoun': 8048, 'productive': 8049, 'collects': 8050, 'lawlessness': 8051, 'kiribati': 8052, 'exclaimed': 8053, 'o': 8054, 'obtaining': 8055, 'jurors': 8056, 'leaked': 8057, 'hashemi': 8058, 'strains': 8059, 'printing': 8060, 'demonstrating': 8061, 'infiltrated': 8062, 'russian-made': 8063, 'physicians': 8064, 'endorsement': 8065, 'prosecuted': 8066, 'bipartisan': 8067, 'crises': 8068, 'vilks': 8069, 'snap': 8070, '31st': 8071, 'consulates': 8072, 'lage': 8073, 'jr.': 8074, '68': 8075, 'slower': 8076, 'fundamental': 8077, 'deeper': 8078, 'merged': 8079, 'gambia': 8080, 'sit': 8081, 'counterterrorism': 8082, 'unnecessary': 8083, 'silent': 8084, 'bathroom': 8085, 'canberra': 8086, 'anatolia': 8087, 'encourages': 8088, 'digital': 8089, 'grass': 8090, 'ideology': 8091, 'fireworks': 8092, 'martina': 8093, 'unrelated': 8094, 'occurs': 8095, 'desecrated': 8096, 'dragan': 8097, 'tutsi': 8098, 'andorra': 8099, 'actors': 8100, 'jazz': 8101, 'fraction': 8102, 'unlawful': 8103, 'discoveries': 8104, 'physically': 8105, 'izmir': 8106, 'pumping': 8107, 'fault': 8108, 'statehood': 8109, 'detail': 8110, 'bullet': 8111, 'mauritanian': 8112, 'structures': 8113, 'hamza': 8114, 'scare': 8115, 'gamal': 8116, 'suleiman': 8117, 'waving': 8118, 'silvio': 8119, 'shocks': 8120, 'abdelaziz': 8121, 'fugitives': 8122, '26,000': 8123, 'rauf': 8124, 'moustapha': 8125, 'mistaken': 8126, 'occasionally': 8127, 'postage': 8128, 'handicrafts': 8129, 'extraction': 8130, 'actively': 8131, 'editorial': 8132, '80,000': 8133, 'cool': 8134, 'cameron': 8135, 'meter': 8136, 'explode': 8137, 'roberto': 8138, 'roy': 8139, 'upgraded': 8140, 'sometime': 8141, 'shamil': 8142, 'restrictive': 8143, 'sect': 8144, 'ox': 8145, 'astronomers': 8146, 'guarded': 8147, 'turkeys': 8148, "yar'adua": 8149, '98': 8150, 'airbase': 8151, 'gen.': 8152, 'ruz': 8153, 'seattle': 8154, '850': 8155, 'shock': 8156, 'seas': 8157, 'survive': 8158, 'ratio': 8159, '89': 8160, 'vegetables': 8161, 'yard': 8162, 'kite': 8163, 'accelerating': 8164, 'pet': 8165, 'panels': 8166, 'long-time': 8167, 'renounced': 8168, 'expenses': 8169, 'alabama': 8170, '32-year-old': 8171, 'dagestan': 8172, 'nickel': 8173, 'execution-style': 8174, 'retaliate': 8175, 'resting': 8176, 'sessions': 8177, 'hebei': 8178, 'automakers': 8179, 'clubs': 8180, 'fiercely': 8181, 'mistreated': 8182, '2.2': 8183, 'pemex': 8184, 'scared': 8185, 'theaters': 8186, 'countryman': 8187, 'loeb': 8188, 'alerted': 8189, 'admission': 8190, 'reformist': 8191, 'swing': 8192, 'subcommittee': 8193, 'myers': 8194, 'oaxaca': 8195, 'elephant': 8196, 'culling': 8197, 'reyes': 8198, 'garment': 8199, 'lawrence': 8200, '27-year-old': 8201, 'haider': 8202, 'spilled': 8203, 'marchers': 8204, '147': 8205, '16-year-old': 8206, 'undecided': 8207, 'berenson': 8208, 'ritual': 8209, 'reverend': 8210, 'zhejiang': 8211, 'goldman': 8212, 'depot': 8213, 'gah': 8214, 'otunbayeva': 8215, 'cracking': 8216, 'connect': 8217, 'abruptly': 8218, 'czechoslovakia': 8219, 'citigroup': 8220, 'gheit': 8221, 'chances': 8222, 'trigger': 8223, 'levee': 8224, 'buyers': 8225, 'aig': 8226, 'publications': 8227, 'wiesenthal': 8228, 'unexpected': 8229, 'moyo': 8230, 'debut': 8231, 'coordinating': 8232, 'las': 8233, 'hilary': 8234, 'girlfriend': 8235, 'rooms': 8236, 'receipt': 8237, 'troubles': 8238, 'seemed': 8239, 'forbes': 8240, 'bird-flu': 8241, 'bleeding': 8242, 'heels': 8243, '82': 8244, 'mudslide': 8245, 'armitage': 8246, 'boundaries': 8247, 'lifetime': 8248, 'achievement': 8249, 'catherine': 8250, 'mobilize': 8251, 'faiths': 8252, 'descendants': 8253, '4.8': 8254, 'f.': 8255, 'cafta': 8256, 'offenders': 8257, 'seems': 8258, 'spaniards': 8259, 'rewards': 8260, 'elaine': 8261, 'low-cost': 8262, 'sivaram': 8263, 'acknowledges': 8264, 'vacationing': 8265, 'abidjan': 8266, 'pattern': 8267, 'preaching': 8268, 'henin-hardenne': 8269, 'prematurely': 8270, 'revoked': 8271, 'foreclosures': 8272, 'sint': 8273, 'one-party': 8274, 'titan': 8275, 'facilitate': 8276, 'additionally': 8277, 'courageous': 8278, 'constrained': 8279, 'engagement': 8280, 'shourd': 8281, 'tombs': 8282, 'rankings': 8283, 'saturn': 8284, 'evolved': 8285, 'reaches': 8286, '67': 8287, 'chalabi': 8288, 'nuclear-armed': 8289, 'meridian': 8290, 'sheriff': 8291, 'salim': 8292, 'anti-piracy': 8293, 'g-20': 8294, 'chase': 8295, 'ratners': 8296, 'repelled': 8297, 'refrain': 8298, 'donating': 8299, 'chihuahua': 8300, 'accomplice': 8301, 'restarted': 8302, 'pork': 8303, 'guerillas': 8304, '4,500': 8305, 'childhood': 8306, 'contamination': 8307, 'rawalpindi': 8308, 'urgently': 8309, 'glasgow': 8310, 'coral': 8311, 'krajicek': 8312, 'disastrous': 8313, 'gere': 8314, 'pierre': 8315, 'backers': 8316, 'forbids': 8317, 'ash': 8318, 'mtv': 8319, 'ceuta': 8320, 'wings': 8321, 'rehnquist': 8322, 'antiquities': 8323, 'saberi': 8324, 'kadyrov': 8325, 'al-faisal': 8326, 'fujian': 8327, 'glacier': 8328, 'rowhani': 8329, 'baiji': 8330, 'skull': 8331, 'tuvalu': 8332, 'lubanga': 8333, 'masri': 8334, 'obelisk': 8335, 'luck': 8336, 'passaro': 8337, 'tb': 8338, 'asghari': 8339, 'abac': 8340, 'maldives': 8341, 'snoop': 8342, 'dogg': 8343, 'heathrow': 8344, 'scuffled': 8345, 'bankers': 8346, 'odinga': 8347, 'u.s.-bound': 8348, 'smokers': 8349, 'subcomandante': 8350, 'uttar': 8351, 'studied': 8352, 'suppress': 8353, 'salafist': 8354, 'escorted': 8355, 'non-oil': 8356, 'busiest': 8357, 'longstanding': 8358, 'drag': 8359, 'talons': 8360, 'snatched': 8361, 'proceeded': 8362, 'bribery': 8363, 'volcker': 8364, 'networking': 8365, 'dies': 8366, 'barak': 8367, 'concludes': 8368, 'repairing': 8369, 'corn': 8370, 'deborah': 8371, 'tires': 8372, 'pelted': 8373, 'sangin': 8374, 'reclaim': 8375, 'forbidden': 8376, 'rode': 8377, 'difference': 8378, 'riders': 8379, 'hewitt': 8380, 'frenchman': 8381, 'sing': 8382, 'biathlon': 8383, 'walter': 8384, 'vancouver': 8385, 'domination': 8386, 'terje': 8387, 'rein': 8388, 'kurt': 8389, 'suppressing': 8390, '86': 8391, 'grandson': 8392, 'alexandria': 8393, 'plains': 8394, 'paved': 8395, 'fazlullah': 8396, 'recaptured': 8397, 'dereliction': 8398, 'specialist': 8399, 'plead': 8400, 'naked': 8401, 'eaten': 8402, 'moreno': 8403, 'mail': 8404, 'sponsors': 8405, '3.7': 8406, 'worsen': 8407, 'yassin': 8408, 'philadelphia': 8409, 'resumes': 8410, 'azeri': 8411, 'three-decade': 8412, 'renounces': 8413, 'carla': 8414, 'one-fourth': 8415, '1821': 8416, 'guitarist': 8417, 'sizeable': 8418, 'bauxite': 8419, 'introduction': 8420, 'surpassed': 8421, 'ants': 8422, 'stung': 8423, 'indeed': 8424, 'alike': 8425, 'innovation': 8426, 'indiscriminately': 8427, 'farda': 8428, 'wolfgang': 8429, 'tomas': 8430, 'continuously': 8431, 'internally': 8432, 'swim': 8433, 'reprimanded': 8434, 'commemorates': 8435, 'exchanging': 8436, 'bureaucracy': 8437, 'withstand': 8438, 'exhibit': 8439, 'mandelson': 8440, 'jerry': 8441, 'gunpoint': 8442, 'sikhs': 8443, 'disruptions': 8444, 'sirleaf': 8445, 'gems': 8446, 'overloaded': 8447, 'stated': 8448, 'kwan': 8449, 'muscle': 8450, 'gardens': 8451, 'deserted': 8452, 'amur': 8453, 'kunduz': 8454, 'thinking': 8455, 'armies': 8456, 'victorious': 8457, 'polluted': 8458, 'minus': 8459, 'pretoria': 8460, 'shutdown': 8461, 'spla': 8462, 'disabled': 8463, 'hardware': 8464, '275-member': 8465, 'discontent': 8466, '1,40,000': 8467, 'possessions': 8468, 'plantation': 8469, 'averaged': 8470, 'one-fifth': 8471, 'inefficient': 8472, 'embarked': 8473, 'extractive': 8474, 'zinc': 8475, 'hood': 8476, 'front-runner': 8477, 'intolerance': 8478, 'alarcon': 8479, 'valencia': 8480, 'cheap': 8481, 'barricades': 8482, 'nine-member': 8483, 'precious': 8484, 'alternatives': 8485, 'fearing': 8486, 'packing': 8487, 'respective': 8488, 'kouchner': 8489, 'spate': 8490, 'torturing': 8491, 'hosseini': 8492, 'lessen': 8493, 'participates': 8494, 'non-aligned': 8495, '17-year': 8496, 'profile': 8497, 'safin': 8498, 'pistol': 8499, 'noordin': 8500, 'ethnically': 8501, 'treatments': 8502, 'medications': 8503, 'traces': 8504, 'impacted': 8505, 'makeup': 8506, 'broadcasters': 8507, 'performances': 8508, 'survival': 8509, 'lined': 8510, 'voices': 8511, 'restructure': 8512, 'closures': 8513, 'mistook': 8514, 'toy': 8515, 'khalis': 8516, 'thwarted': 8517, 'italians': 8518, 'reflection': 8519, 'ingredients': 8520, 'halabja': 8521, 'technological': 8522, 'shortfall': 8523, 'reconcile': 8524, 'races': 8525, 'efficient': 8526, 'naming': 8527, '35-year-old': 8528, 'galaxy': 8529, 'ailments': 8530, 'automotive': 8531, '1918': 8532, 'dissatisfied': 8533, 'heilongjiang': 8534, 'hotly': 8535, 'well-developed': 8536, 'refined': 8537, 'hamper': 8538, 'guide': 8539, 'overcame': 8540, 'beautiful': 8541, 'vanished': 8542, 'iskandariyah': 8543, 'ravaged': 8544, '6.6': 8545, 'holed': 8546, 'karni': 8547, 'incursions': 8548, 'ridiculous': 8549, 'house-to-house': 8550, 'reviving': 8551, 'receives': 8552, 'sheehan': 8553, 'rumors': 8554, 'ore': 8555, 'inflated': 8556, 'habits': 8557, 'feature': 8558, 'moshe': 8559, 'teachings': 8560, 'skeleton': 8561, 'technician': 8562, 'lieutenant-general': 8563, 'deliberate': 8564, 'clearly': 8565, 'habitats': 8566, 'demobilized': 8567, 'accelerate': 8568, 'casino': 8569, 'feeding': 8570, 'element': 8571, 'unfortunately': 8572, 'dragged': 8573, 'commemoration': 8574, 'zanu-pf': 8575, 'complied': 8576, 'reigning': 8577, 'bottom': 8578, '01-jan': 8579, 'importing': 8580, 'occurring': 8581, 'tractor': 8582, 'illnesses': 8583, 'spokesperson': 8584, 'beer': 8585, 'teargas': 8586, 'domestically': 8587, 'employing': 8588, 'averaging': 8589, 'brazilians': 8590, 'excerpts': 8591, 'signaled': 8592, 'faltering': 8593, 'positioned': 8594, 'frist': 8595, 'arranging': 8596, 'ills': 8597, 'khursheed': 8598, 'incorrect': 8599, 'rico': 8600, 'paving': 8601, '24-hour': 8602, 'walkout': 8603, 'stockpiles': 8604, 'reserved': 8605, '64-year-old': 8606, 'beckett': 8607, 'leak': 8608, 'andaman': 8609, 'coconuts': 8610, 'baghlan': 8611, 'pedro': 8612, 'nick': 8613, 'cowell': 8614, 'prospered': 8615, 'leslie': 8616, 'cautious': 8617, 'endemic': 8618, 'caspian': 8619, 'disappointment': 8620, 'bystander': 8621, 'quantity': 8622, 'livelihoods': 8623, 'ganguly': 8624, 'bowlers': 8625, 'kumble': 8626, 'anti-terrorist': 8627, 'tips': 8628, 'sayed': 8629, 'agha': 8630, 'ahmadi': 8631, 'fencing': 8632, 'tunceli': 8633, 'roadblock': 8634, 'complaining': 8635, 'accra': 8636, 'syrian-backed': 8637, 'dennis': 8638, 'wildfires': 8639, 'gran': 8640, 'triangle': 8641, 'outcry': 8642, 'hrw': 8643, 'danilo': 8644, 'rises': 8645, 'habitat': 8646, 'bugti': 8647, 'outdated': 8648, 'predicting': 8649, 'outages': 8650, 'utility': 8651, 'chaired': 8652, 'mukherjee': 8653, 'el-bared': 8654, 'independents': 8655, 'highlights': 8656, 'two-time': 8657, 'tatiana': 8658, 'delegations': 8659, 'kitchen': 8660, 'asset': 8661, 'lavish': 8662, 'concentration': 8663, 'penitentiary': 8664, 'airing': 8665, 'merchant': 8666, 'resettled': 8667, 'lakes': 8668, 'dust': 8669, 'first-ever': 8670, 'circulated': 8671, 'benchmarks': 8672, 'paulson': 8673, 'remembrance': 8674, 'wade': 8675, 'ram': 8676, 'bingu': 8677, 'foreigner': 8678, 'icrc': 8679, 'conveyed': 8680, 'gunshot': 8681, 'guinea-bissau': 8682, 'ramon': 8683, 'freely': 8684, '34-day': 8685, 'cliff': 8686, 'observatory': 8687, 'cervical': 8688, 'gaza-egypt': 8689, 'updated': 8690, 'fundamentalism': 8691, 'buys': 8692, 'contagious': 8693, 'expelling': 8694, 'shrink': 8695, 'clementina': 8696, 'penalties': 8697, 'occasional': 8698, 'geneva-based': 8699, 'perpetual': 8700, 'defensive': 8701, 'succeeding': 8702, 'customer': 8703, 'dong-young': 8704, 'machimura': 8705, 'presutti': 8706, 'resettlement': 8707, 'enhanced': 8708, 'polled': 8709, 'iranian-american': 8710, 'attendance': 8711, 'yunnan': 8712, 'bennett': 8713, 'mehsud': 8714, 'surayud': 8715, 'sighting': 8716, 'crescent': 8717, 'meals': 8718, '26-year-old': 8719, 'extracted': 8720, 'demarcation': 8721, 'desecration': 8722, 'interceptor': 8723, 'scrapped': 8724, '125': 8725, 'parliamentarians': 8726, 'royalist': 8727, 'await': 8728, 'sue': 8729, 'four-month': 8730, 'sticks': 8731, 'whale': 8732, 'testified': 8733, 'entertainer': 8734, 'liquor': 8735, 'blue': 8736, '1941': 8737, 'consented': 8738, 'accepts': 8739, 'briefed': 8740, 'credited': 8741, 'rover': 8742, 'fourth-largest': 8743, 'costly': 8744, 'fundraising': 8745, '1,30,000': 8746, 'breathing': 8747, 'crocker': 8748, 'pronk': 8749, 'knocking': 8750, 'fulfill': 8751, 'converting': 8752, 'scheme': 8753, 'impressed': 8754, 'barbaric': 8755, '8th': 8756, 'pleased': 8757, 'yediot': 8758, 'sentencing': 8759, 'orphans': 8760, 'h5': 8761, 'reasonable': 8762, 'costner': 8763, 'songwriter': 8764, 'concerts': 8765, 'speaks': 8766, '74-year-old': 8767, 'stockpile': 8768, 'painful': 8769, 'partnerships': 8770, 'petrochemical': 8771, 'struggles': 8772, 'anticipation': 8773, 'departments': 8774, 'purchasing': 8775, 'catches': 8776, 'ridge': 8777, 'ultimate': 8778, 'raping': 8779, 'preconditions': 8780, 'traditions': 8781, 'farid': 8782, 'angel': 8783, 'zulima': 8784, 'palacio': 8785, 'specialists': 8786, 'adumim': 8787, '23rd': 8788, '02-jun': 8789, 'youtube': 8790, 'high-security': 8791, 'trincomalee': 8792, 'bucharest': 8793, 'traian': 8794, 'marc': 8795, 'convictions': 8796, 'high-technology': 8797, 'jannati': 8798, 'theory': 8799, 'choices': 8800, 'ultra-orthodox': 8801, '4.6': 8802, 'disposal': 8803, 'ppp': 8804, 'upswing': 8805, 'parent': 8806, '1940': 8807, 'dancing': 8808, 'merely': 8809, 'mid-september': 8810, 'nowak': 8811, 'tennessee': 8812, 'vault': 8813, 'thick': 8814, 'one-year-old': 8815, 'readings': 8816, 'sein': 8817, 'anti-taliban': 8818, 'societies': 8819, 'erkinbayev': 8820, 'rift': 8821, 'granda': 8822, 'froze': 8823, '12-year': 8824, 'worst-hit': 8825, 'unearthed': 8826, 'bags': 8827, 'dealers': 8828, 'manufacturers': 8829, 'philip': 8830, 'heather': 8831, 'performers': 8832, 'examination': 8833, 'utah': 8834, '21-year-old': 8835, 'maghreb': 8836, 'nonpermanent': 8837, '3.2': 8838, 'protester': 8839, 'openness': 8840, '79': 8841, 'ljubicic': 8842, '100-meter': 8843, 'taha': 8844, 'chandrika': 8845, 'accountability': 8846, 'pulls': 8847, 'pave': 8848, '96': 8849, 'pits': 8850, 'jacob': 8851, 'uae': 8852, 'vioxx': 8853, 'rolled': 8854, 'beatings': 8855, 'khalaf': 8856, 'evacuating': 8857, 'pressures': 8858, 'abolished': 8859, 'recurrent': 8860, 'fruit': 8861, 'fowl': 8862, 'wreath': 8863, 'presided': 8864, 'promptly': 8865, 'ganji': 8866, 'unanimous': 8867, 'blanco': 8868, 'tidal': 8869, 'marathon': 8870, 'fukuda': 8871, 'paula': 8872, '27th': 8873, 'rampant': 8874, 'ravi': 8875, 'khanna': 8876, 'defiance': 8877, 'santo': 8878, 'compensated': 8879, 'nbc': 8880, 'violators': 8881, 'disneyland': 8882, 'downing': 8883, 'narcotics': 8884, 'albums': 8885, 'discount': 8886, 'plachkov': 8887, 'duck': 8888, 'wu': 8889, 'revision': 8890, 'lt.': 8891, 'predominately': 8892, 'contrary': 8893, 'starbucks': 8894, 'entitled': 8895, 'curtain': 8896, 'frustration': 8897, 'mayors': 8898, 'qazi': 8899, 'masses': 8900, 'breaches': 8901, 'enterprise': 8902, 'licensing': 8903, 'equally': 8904, 'cleanup': 8905, 'liters': 8906, 'halting': 8907, 'olive': 8908, 'kiir': 8909, 'infectious': 8910, 'supplement': 8911, '191': 8912, 'nicobar': 8913, 'sindh': 8914, 'canaveral': 8915, 'safarova': 8916, 'safina': 8917, '12.5': 8918, 'charm': 8919, 'state-controlled': 8920, 'wta': 8921, 'exceeding': 8922, 'natanz': 8923, 'eviction': 8924, 'transcripts': 8925, 'tighter': 8926, 'borrowers': 8927, 'el-maan': 8928, 'frustrated': 8929, '06-jul': 8930, 'car-bomb': 8931, 'tall': 8932, 'companions': 8933, 'rang': 8934, 'walchhofer': 8935, 'super': 8936, 'provocative': 8937, 'mujahedin': 8938, 'outraged': 8939, 'hijack': 8940, 'warship': 8941, 'thirteen': 8942, 'haaretz': 8943, 'advising': 8944, 'prudential': 8945, 'strayed': 8946, 'resilient': 8947, 'totaling': 8948, 'signatures': 8949, 'muttahida': 8950, 'yi': 8951, '270': 8952, 'dispatch': 8953, 'mei': 8954, 'boxes': 8955, 'kot': 8956, 'besides': 8957, 'telerate': 8958, 'evolution': 8959, 'entity': 8960, 'noticed': 8961, 'proud': 8962, 'taxation': 8963, 'praising': 8964, 'infantry': 8965, 'roza': 8966, 'uzbeks': 8967, 'intercontinental': 8968, 'mahdi': 8969, 'legendary': 8970, 'safer': 8971, 'hitler': 8972, 'asks': 8973, 'helpful': 8974, 'nfl': 8975, '2.1': 8976, '1910': 8977, '1937': 8978, 'col.': 8979, 'shall': 8980, 'armistice': 8981, 'paint': 8982, '5.8': 8983, 'acid': 8984, 'boarding': 8985, 'prompt': 8986, 'photographed': 8987, 'pro-taliban': 8988, 'menem': 8989, 'ngos': 8990, 'dar': 8991, 'wei': 8992, 'tampering': 8993, 'mario': 8994, 'grammy': 8995, 'baker': 8996, 'funerals': 8997, 'hoax': 8998, 'sort': 8999, 'nargis': 9000, 'jewelry': 9001, 'poised': 9002, 'legality': 9003, 'thorough': 9004, 'males': 9005, '1,800': 9006, 'archaeologist': 9007, 'osthoff': 9008, 'shaanxi': 9009, 'hemorrhagic': 9010, 'renault': 9011, 'ferrari': 9012, 'intrusion': 9013, 'delaware': 9014, 'liverpool': 9015, 'shanxi': 9016, 'manhunt': 9017, 'sooner': 9018, 'dwyer': 9019, 'pro-kurdish': 9020, 'blind': 9021, 'aspirations': 9022, 'celebrity': 9023, 'janet': 9024, 'sporting': 9025, 'spills': 9026, '5.5': 9027, 'cambodians': 9028, '1863': 9029, 'colom': 9030, 'divorce': 9031, 'alliot-marie': 9032, 'fours': 9033, 'colorful': 9034, 'slim': 9035, 'rooting': 9036, 'tower': 9037, 'hubei': 9038, 'porter': 9039, 'banner': 9040, 'mirror': 9041, 'nebraska': 9042, 'boko': 9043, '2.8': 9044, 'nevis': 9045, 'patrolled': 9046, 'newest': 9047, 'weaver': 9048, 'bandar': 9049, 'websites': 9050, 'properties': 9051, 'widening': 9052, 'fifteen': 9053, 'clone': 9054, 'natsios': 9055, 'cannes': 9056, 'yuganskneftegaz': 9057, 'vomiting': 9058, 'museums': 9059, 'vukovar': 9060, 'fuels': 9061, 'precaution': 9062, 'commemorating': 9063, 'draws': 9064, 'comparing': 9065, 'stirred': 9066, 'debt-to-gdp': 9067, 'doses': 9068, 'baku': 9069, 'expresses': 9070, 'flexibility': 9071, 'latif': 9072, 'proposes': 9073, 'pak': 9074, 'usual': 9075, 'wooden': 9076, 'provoke': 9077, 'kano': 9078, 'shortfalls': 9079, 'aftershock': 9080, 'assuming': 9081, 'daoud': 9082, 'moodie': 9083, 'dravid': 9084, 'atlantis': 9085, 'circulating': 9086, 'fat': 9087, 'surgical': 9088, 'zhao': 9089, 'kosumi': 9090, 'dissent': 9091, 'telephoned': 9092, 'papua': 9093, 'pohamba': 9094, 'quarterfinal': 9095, 'kurd': 9096, 'ciudad': 9097, 'disqualified': 9098, 'buffer': 9099, 'trump': 9100, 'igor': 9101, 'amin': 9102, 'chapter': 9103, 'owed': 9104, 'processes': 9105, 'zones': 9106, 'rigging': 9107, 'uninhabited': 9108, 'ashraf': 9109, '99': 9110, 'competed': 9111, 'kickbacks': 9112, 'favoring': 9113, 'slot': 9114, 'gibbs': 9115, 'wives': 9116, 'skier': 9117, 'mecca': 9118, '1898': 9119, 'cadmium': 9120, 'cloud': 9121, 'falkland': 9122, 'resignations': 9123, 'benigno': 9124, 'ralston': 9125, 'curtailed': 9126, 'nationalized': 9127, 'manslaughter': 9128, 'favorite': 9129, 'pickens': 9130, 'loose': 9131, 'papacy': 9132, 'pristina': 9133, 'teaching': 9134, 'lesser': 9135, 'lashkar-e-toiba': 9136, 'combs': 9137, 'prodi': 9138, 'chariot': 9139, 'mussa': 9140, 'self': 9141, 'hostilities': 9142, 'konare': 9143, 'pepsico': 9144, 'ears': 9145, 'surging': 9146, '11-year-old': 9147, 'obesity': 9148, '26-year': 9149, 'decorations': 9150, 'motorcycles': 9151, 'blogger': 9152, 'stimulant': 9153, 'paz': 9154, 'breakup': 9155, 'grievances': 9156, 'medicare': 9157, 'depressed': 9158, 'bricks': 9159, 'tusk': 9160, 'touchdown': 9161, 'reopening': 9162, 'nissan': 9163, 'miliband': 9164, 'alkhanov': 9165, 'slovaks': 9166, 'hamm': 9167, 'nsa': 9168, 'fourth-seeded': 9169, 'moore': 9170, 'mueller': 9171, 'greenland': 9172, 'eide': 9173, 'nevirapine': 9174, 'pierce': 9175, 'razuri': 9176, 'equivalent': 9177, 'nias': 9178, '6.7': 9179, 'rap': 9180, 'sworn-in': 9181, 'advisers': 9182, 'cigarette': 9183, 'pro-business': 9184, 'transplant': 9185, 'seminary': 9186, '7.6': 9187, 'madrassas': 9188, 'imad': 9189, '370': 9190, 'hamad': 9191, 'legacy': 9192, 'assuring': 9193, 'zhvania': 9194, 'systematic': 9195, 'hung': 9196, 'probes': 9197, 'crossfire': 9198, 'kfc': 9199, 'syndrome': 9200, 'briefings': 9201, 'therapy': 9202, 'amer': 9203, '9.5': 9204, 'hiring': 9205, 'generous': 9206, 'arbitrary': 9207, 'hardcourt': 9208, 'karlovic': 9209, 'unseeded': 9210, 'nieminen': 9211, 'exert': 9212, 'evacuations': 9213, 'hills': 9214, 'loud': 9215, 'rigs': 9216, 'exclude': 9217, '65,000': 9218, 'decides': 9219, 'snowfall': 9220, 'sunny': 9221, 'denver': 9222, 'skubiszewski': 9223, 'p.j.': 9224, 'crowley': 9225, 'insight': 9226, 'recipients': 9227, 'helsinki': 9228, 'dismayed': 9229, 'maulana': 9230, 'withholding': 9231, 'u.s.-russian': 9232, 'nikolai': 9233, 'felony': 9234, 'integrated': 9235, 'anonymous': 9236, 'buyer': 9237, 'b': 9238, 'fun': 9239, 'toronto': 9240, 'stretched': 9241, 'cancelled': 9242, '63-year-old': 9243, 'liberalize': 9244, 'wire': 9245, 'mercury': 9246, 'yourself': 9247, 'dealings': 9248, 'joschka': 9249, 'combating': 9250, 'contemporary': 9251, 'smart': 9252, 'craft': 9253, 'graham': 9254, 'tools': 9255, 'government-sponsored': 9256, 'ransacked': 9257, 'prevents': 9258, '325': 9259, 'forecasts': 9260, 'yahya': 9261, 'worship': 9262, 'concentrate': 9263, 'guterres': 9264, 'titled': 9265, 'incite': 9266, 'sikh': 9267, 'chinese-made': 9268, 'expel': 9269, 'withdraws': 9270, 'eradicate': 9271, 'apple': 9272, 'screens': 9273, 'patriotic': 9274, 'barnier': 9275, 'parallel': 9276, 'technically': 9277, 'unstable': 9278, 'osh': 9279, 'symbols': 9280, 'cardinals': 9281, 'radioactive': 9282, 's.': 9283, 'hip': 9284, 'alpine': 9285, 'skiing': 9286, 'tamils': 9287, 'registering': 9288, 'nominations': 9289, 'submarines': 9290, 'informants': 9291, 'congratulate': 9292, '129': 9293, 'religions': 9294, 'forgiveness': 9295, 'spectators': 9296, 'scenes': 9297, 'mindanao': 9298, 'cooking': 9299, 'five-member': 9300, 'operator': 9301, 'post-abc': 9302, 'presumed': 9303, 'primaries': 9304, 'cite': 9305, 'healthcare': 9306, 'credits': 9307, 'alarmed': 9308, 'denial': 9309, 'ogaden': 9310, 'preferences': 9311, 'totaled': 9312, 'fta': 9313, 'subsidized': 9314, 'sluggish': 9315, 'confronting': 9316, 'prevalence': 9317, 'relieve': 9318, 'hinder': 9319, 'establishes': 9320, 'montreal': 9321, 'solely': 9322, 'touching': 9323, 'profession': 9324, 'human-rights': 9325, 'centuries-old': 9326, 'marching': 9327, 'magnate': 9328, 'adequately': 9329, 'gorge': 9330, 'privileged': 9331, "n'djamena": 9332, 'prasad': 9333, 'knives': 9334, 'veered': 9335, 'jumblatt': 9336, 'circulation': 9337, 'foods': 9338, 'hooded': 9339, 'edge': 9340, 'al-majid': 9341, 'gambari': 9342, 'nam': 9343, 'iftikhar': 9344, 'sochi': 9345, 'bet': 9346, 'requesting': 9347, '18,000': 9348, 'boards': 9349, 'julian': 9350, 'laghman': 9351, 'blacks': 9352, 'tissue': 9353, 'interpreted': 9354, 'reductions': 9355, 'alarm': 9356, 'meteorologists': 9357, 'day-long': 9358, 'gabriele': 9359, 'rini': 9360, 'retrieved': 9361, 'ebadi': 9362, 'handles': 9363, 'maine': 9364, 'dir': 9365, '171': 9366, 'innocence': 9367, 'zeng': 9368, 'fifth-largest': 9369, 'assert': 9370, 'sciences': 9371, 'ecowas': 9372, 'reluctant': 9373, 'earmarked': 9374, 'documentation': 9375, 'henan': 9376, 'stronach': 9377, 'grenadines': 9378, 'renamed': 9379, 'curacao': 9380, 'accommodate': 9381, 'conquer': 9382, 'dakota': 9383, 'noon': 9384, 'scrambling': 9385, 'gauge': 9386, 'sail': 9387, 'forth': 9388, 'loyalists': 9389, 'mistakes': 9390, 'u.n.-afghan': 9391, 'intergovernmental': 9392, 'daylight': 9393, 'mobilized': 9394, 'torn': 9395, 'terminal': 9396, 'saran': 9397, 'obstructing': 9398, 'proportion': 9399, 'sailed': 9400, 'immunizations': 9401, 'surgeon': 9402, 'lineup': 9403, 'proves': 9404, 'fourth-quarter': 9405, 'miles': 9406, 'marketplace': 9407, 'kingpin': 9408, 'counterproductive': 9409, 'compelled': 9410, 'indefinite': 9411, 'scenario': 9412, 'ideal': 9413, 'lingering': 9414, 'plunge': 9415, 'revitalize': 9416, 'backer': 9417, 'contacting': 9418, 'mekong': 9419, 'methamphetamine': 9420, 'evict': 9421, 'durable': 9422, 'fortunes': 9423, 'uneven': 9424, 'contesting': 9425, 'seashore': 9426, 'fulfilled': 9427, 'constituencies': 9428, 'singing': 9429, 'praises': 9430, 'striker': 9431, '22-year-old': 9432, 'qualify': 9433, 'towers': 9434, 'jeopardize': 9435, 'abdominal': 9436, 'lend': 9437, 'spree': 9438, 'distributors': 9439, 'relying': 9440, 'cracked': 9441, 'anxious': 9442, 'eroded': 9443, 'wise': 9444, 'nadal': 9445, 'number-one': 9446, 'ankle': 9447, 'liberalization': 9448, 'all-star': 9449, 'termination': 9450, 'wracked': 9451, 'monrovia': 9452, 'richardson': 9453, 'endorse': 9454, 'hispanic': 9455, 'wary': 9456, 'lent': 9457, 'masks': 9458, 'filmed': 9459, 'ingushetia': 9460, 'caledonia': 9461, 'tactic': 9462, 'drastically': 9463, 'fleets': 9464, 'entreated': 9465, 'copy': 9466, 'inaugural': 9467, 'unsafe': 9468, 'shelled': 9469, 'anti-viral': 9470, 'rehman': 9471, 'discouraging': 9472, 'uniformed': 9473, 'hopefuls': 9474, 'tueni': 9475, 'nice': 9476, '1830': 9477, 'bureaucratic': 9478, 'governed': 9479, 'ravine': 9480, 'arsenals': 9481, 'stumps': 9482, 'matthew': 9483, 'anil': 9484, 'nabih': 9485, 'berri': 9486, 'mutilated': 9487, 'pains': 9488, 'consolidation': 9489, 'communists': 9490, 'arabic-language': 9491, 'lynndie': 9492, 'bias': 9493, 'entrenched': 9494, '6.1': 9495, 'meal': 9496, 'rainy': 9497, 'depletion': 9498, 'stabilized': 9499, 'inspected': 9500, 'forests': 9501, 'environmentalists': 9502, 're-establish': 9503, '84-year-old': 9504, 'nativity': 9505, 'registry': 9506, 'unexpectedly': 9507, 'speculate': 9508, 'charred': 9509, 'unjustified': 9510, 'privatizations': 9511, 'aligned': 9512, 'javed': 9513, 'treats': 9514, 'electrocuted': 9515, 'destroyer': 9516, 'featured': 9517, 'lashkar-e-jhangvi': 9518, 'multan': 9519, 'abyan': 9520, '111': 9521, 'poles': 9522, 'spanta': 9523, 'morgan': 9524, 'sheltering': 9525, 'westward': 9526, 'underlying': 9527, 'invasions': 9528, 'tuna': 9529, 'arose': 9530, 'swam': 9531, 'alter': 9532, 'balloon': 9533, 'batteries': 9534, 'malnutrition': 9535, 'divert': 9536, 'parliamentarian': 9537, 'debating': 9538, 'henry': 9539, 'distress': 9540, 'pessimistic': 9541, 'avoided': 9542, 'motorists': 9543, 'extermination': 9544, 'case-by-case': 9545, 'hostility': 9546, 'flaws': 9547, 'syrians': 9548, 'justin': 9549, 'al-sunna': 9550, 'pakistani-controlled': 9551, 'profiling': 9552, 'revise': 9553, 'redistribute': 9554, 'stems': 9555, 'accompanying': 9556, 'minas': 9557, '92': 9558, 'eric': 9559, 'ebola-like': 9560, 'minimize': 9561, 'conte': 9562, 'installing': 9563, 'landmines': 9564, 'whoever': 9565, 'fog': 9566, '102': 9567, '166': 9568, 'douste-blazy': 9569, 'acquire': 9570, '58-year-old': 9571, 'logging': 9572, 'rogers': 9573, 'restrain': 9574, 'coaches': 9575, 'pleasure': 9576, 'complicated': 9577, 'clerk': 9578, 'pathogenic': 9579, 'kaduna': 9580, 'anders': 9581, 'rasmussen': 9582, 'slow-moving': 9583, 'yielded': 9584, 'brands': 9585, 'patience': 9586, 'hmong': 9587, 'filling': 9588, 'whitman': 9589, 'plateau': 9590, 'loading': 9591, 'psychological': 9592, 'northern-based': 9593, 'displacing': 9594, 'lander': 9595, 'clean-up': 9596, 'resident': 9597, 'panjwayi': 9598, 'reputed': 9599, 'rioted': 9600, 'bayelsa': 9601, 'berger': 9602, '1950s': 9603, 'argues': 9604, 'alexei': 9605, 'camilla': 9606, 'glimpse': 9607, 'aslam': 9608, '70.85': 9609, 'initiate': 9610, 'turki': 9611, 'gotten': 9612, 'graphic': 9613, 'half-brother': 9614, 'clampdown': 9615, '36-year-old': 9616, 'span': 9617, 'wide-ranging': 9618, 'brawl': 9619, 'presenting': 9620, 'sped': 9621, 'confrontations': 9622, 'lara': 9623, 'climb': 9624, 'affirmed': 9625, 'gilbert': 9626, 'victories': 9627, 'relinquished': 9628, '0.2': 9629, 'reversal': 9630, 'competitiveness': 9631, 'sacrifices': 9632, 'leaks': 9633, 'iran-iraq': 9634, 'cooling': 9635, 'plunging': 9636, 'mahinda': 9637, 'rajapakse': 9638, 'enhancing': 9639, 'intimidate': 9640, '7,50,000': 9641, 'north-south': 9642, 'azimi': 9643, 'crater': 9644, 'pregnancy': 9645, 'leta': 9646, 'fincher': 9647, 'organizer': 9648, 'one-quarter': 9649, '1958': 9650, 'isolate': 9651, 'looters': 9652, 'enduring': 9653, 'bongo': 9654, 'bangladeshis': 9655, 'letting': 9656, 'chun': 9657, '8,500': 9658, 'unreported': 9659, 'bodman': 9660, 'refiners': 9661, 'zahir': 9662, 'finishing': 9663, 'shocking': 9664, 'devoted': 9665, 'conform': 9666, '101st': 9667, 'slept': 9668, 'constant': 9669, 'plight': 9670, 'blessing': 9671, 'breached': 9672, 'fledgling': 9673, 'mahee': 9674, 'marketing': 9675, 'reneged': 9676, 'demolition': 9677, 'popularly': 9678, 'ann': 9679, 'tome': 9680, 'rogue': 9681, 'alvarez': 9682, 'succeeds': 9683, 'expatriate': 9684, 'comparable': 9685, 'nauru': 9686, 'autocratic': 9687, 'discounted': 9688, 'profitable': 9689, 'illegitimate': 9690, 'instance': 9691, 'typical': 9692, 'commissioned': 9693, 'financially': 9694, 'yuganskneftegas': 9695, 'maale': 9696, 'delp': 9697, '55-year-old': 9698, 'protective': 9699, '03-feb': 9700, 'anabel': 9701, 'cho': 9702, 'mid-1980s': 9703, 'embrace': 9704, 'distances': 9705, 'clergy': 9706, 'bowl': 9707, 'correctly': 9708, 'waheed': 9709, 'arshad': 9710, 'politkovskaya': 9711, 'fred': 9712, 'mechanical': 9713, 'grozny': 9714, 'discredit': 9715, 'rudd': 9716, 'guam': 9717, 'salehi': 9718, 'nursultan': 9719, 'miranshah': 9720, 'hainan': 9721, 'advances': 9722, '4.7': 9723, 'nonetheless': 9724, 'pocket': 9725, 'tearing': 9726, 'wished': 9727, 'thief': 9728, 'raffaele': 9729, 'impunity': 9730, 'cleaner': 9731, 'h5n2': 9732, 'carmona': 9733, 'bunker': 9734, 'comic': 9735, 'ringleader': 9736, 'fourteen': 9737, 'brisbane': 9738, '15-year': 9739, 'overwhelmed': 9740, 'blasted': 9741, 'droughts': 9742, 'comoros': 9743, 'verbal': 9744, 'courthouse': 9745, 'mills': 9746, 'mccartney': 9747, 'candles': 9748, 'indonesians': 9749, 'miran': 9750, 'periodic': 9751, 'cirque': 9752, 'baja': 9753, 'undergone': 9754, 'rupiah': 9755, 'vojislav': 9756, 'emigration': 9757, 'ecuadorian': 9758, 'tire': 9759, 'staple': 9760, 'improper': 9761, 'poorer': 9762, 'asadabad': 9763, 'four-nation': 9764, 'balanced': 9765, 'lendu': 9766, 'barinov': 9767, 'hersh': 9768, 'athletics': 9769, 'moi': 9770, 'knesset': 9771, 're-appointed': 9772, 'fisheries': 9773, 'light-water': 9774, 'lucio': 9775, 'cement': 9776, 'floating': 9777, 'forging': 9778, 'anti-apartheid': 9779, 'outlines': 9780, 'nuclear-free': 9781, 'resettle': 9782, 'bouteflika': 9783, 'camara': 9784, '7.8': 9785, 'appreciation': 9786, 'prospect': 9787, 'copra': 9788, 'coins': 9789, 'medieval': 9790, '1865': 9791, 'emigrants': 9792, 'reader': 9793, 'grounding': 9794, 'pigeons': 9795, 'flattened': 9796, 'stab': 9797, 'ahmet': 9798, 'justification': 9799, 'farmland': 9800, 'photojournalist': 9801, 'reviews': 9802, 'self-governing': 9803, 'beliefs': 9804, 'commonly': 9805, 'acquiring': 9806, 'latvian': 9807, 'raymond': 9808, 'thigh': 9809, 'holder': 9810, 'identifying': 9811, 'hallums': 9812, 'shipyard': 9813, 'sustainability': 9814, 'u.s-led': 9815, 'tarasyuk': 9816, 'doubling': 9817, 'facts': 9818, 'nguyen': 9819, 'taliban-led': 9820, 'illicit': 9821, 'bremer': 9822, 'smithsonian': 9823, 'mix': 9824, 'wholesale': 9825, 'sultanate': 9826, 'uprisings': 9827, 'free-market': 9828, 'warden': 9829, 'rage': 9830, 'elementary': 9831, 'shandong': 9832, 'non-permanent': 9833, 'transmit': 9834, 'exception': 9835, 'madrazo': 9836, 'ethiopian-backed': 9837, 'c-130': 9838, 'wayne': 9839, 'needy': 9840, 'unused': 9841, 'soar': 9842, 'yuriy': 9843, 'interviewer': 9844, '18-month': 9845, 'jason': 9846, 'respectively': 9847, 'barno': 9848, 'hillah': 9849, 'beheading': 9850, 'transform': 9851, 'eldest': 9852, 'pneumonia': 9853, 'solutions': 9854, 'innovative': 9855, 'ron': 9856, 'survivor': 9857, 'drunk': 9858, 'forever': 9859, 'criteria': 9860, 'introduce': 9861, 'inch': 9862, 'teeth': 9863, '275': 9864, 'cleaned': 9865, 'executing': 9866, 'crying': 9867, 'robotic': 9868, 'salva': 9869, 'wider': 9870, 'mao': 9871, 'discover': 9872, 'aliens': 9873, 'kerala': 9874, 'stakes': 9875, 'lucie': 9876, 'dinara': 9877, '2-0': 9878, '2.6': 9879, 'economically': 9880, 'vazquez': 9881, 'montevideo': 9882, 'nevada': 9883, 'uighurs': 9884, 'uighur': 9885, 'binding': 9886, 'fruitful': 9887, 'carmaker': 9888, 'robot': 9889, 'mandated': 9890, 'harmed': 9891, 'outsiders': 9892, 'temperature': 9893, 'diving': 9894, '165': 9895, 'divides': 9896, '80-year-old': 9897, 'shinzo': 9898, 'wielgus': 9899, 'telesur': 9900, 'attach': 9901, 'bode': 9902, 'rahlves': 9903, 'anton': 9904, 'appointing': 9905, 'ultimatum': 9906, 'container': 9907, 'mikheil': 9908, 'serb-held': 9909, 'heinous': 9910, 'leftists': 9911, 'bravo': 9912, 'blasting': 9913, 'intelogic': 9914, 'chartered': 9915, 'denis': 9916, 'tong': 9917, 'alston': 9918, 'alejandro': 9919, 'toledo': 9920, '67-year-old': 9921, 'grateful': 9922, 'diaz': 9923, 'andrei': 9924, 'speeding': 9925, 'pro-palestinian': 9926, 'gaza-bound': 9927, 'secularists': 9928, 'tendered': 9929, 'aloft': 9930, 'reminded': 9931, 'narrated': 9932, 'sachs': 9933, 'tenure': 9934, '1954': 9935, '112': 9936, 'soup': 9937, 'aboul': 9938, 'hadassah': 9939, 'hemorrhage': 9940, 'function': 9941, 'historian': 9942, 'damrey': 9943, 'al-kidwa': 9944, 'updates': 9945, '260': 9946, 'euros': 9947, 'frequency': 9948, 'khz': 9949, 'crimean': 9950, 'el~paso': 9951, 'imbalances': 9952, 'decent': 9953, 'underemployment': 9954, 'full-scale': 9955, 'tuareg': 9956, 'hail': 9957, 'durban': 9958, 'privately-owned': 9959, 'cooked': 9960, 'improperly': 9961, 'warmer': 9962, 'spectrum': 9963, 'present-day': 9964, 'bombardment': 9965, 'al-azhar': 9966, 'haifa': 9967, 'shrinking': 9968, 'monde': 9969, 'pounding': 9970, '60-year-old': 9971, '113': 9972, 'abdulkadir': 9973, 'es': 9974, 'salaam': 9975, 'jokonya': 9976, 'regiment': 9977, 'tremendous': 9978, 'osce': 9979, 'regardless': 9980, 'casablanca': 9981, 'issuing': 9982, 'clans': 9983, 'angolan': 9984, 'indirectly': 9985, 'tsang': 9986, 'intent': 9987, 'ruegen': 9988, 'environmentally': 9989, 'abductees': 9990, 'ottawa': 9991, 'susanne': 9992, 'prone': 9993, '6.9': 9994, 'thoughts': 9995, 'middle-income': 9996, 'polynesian': 9997, 'dynamic': 9998, 'mid-2008': 9999, 'writers': 10000, 'spam': 10001, 'thank': 10002, 'hyderabad': 10003, 'akihito': 10004, '1944': 10005, 'reservoir': 10006, 'glory': 10007, 'conocophillips': 10008, 'clinics': 10009, 'eavesdropping': 10010, 'rulings': 10011, 'orchestrated': 10012, 'ancic': 10013, 'three-set': 10014, 'hekmatyar': 10015, 'emmanuel': 10016, 'akitani-bob': 10017, 'husaybah': 10018, 'wang': 10019, 'cancer-causing': 10020, 'mohamud': 10021, 'posting': 10022, 'measuring': 10023, 'joyful': 10024, 'ghazi': 10025, 'parole': 10026, 'phnom': 10027, 'penh': 10028, 'myth': 10029, 'nuns': 10030, 'symonds': 10031, 'balls': 10032, 'michele': 10033, 'batsman': 10034, 'rogge': 10035, 'sebastian': 10036, 'theme': 10037, 'lowering': 10038, 'advertisements': 10039, 'abkhaz': 10040, 'uefa': 10041, 'postponing': 10042, 'kansas': 10043, 'shirts': 10044, 'j.': 10045, 'slave': 10046, 'anguilla': 10047, 'brutally': 10048, 'shirt': 10049, 'parkinson': 10050, 'duelfer': 10051, 'non-arab': 10052, 'karnataka': 10053, 'accreditation': 10054, 'budapest': 10055, 'adhere': 10056, 'gestures': 10057, 'anti-insurgent': 10058, 'overturn': 10059, 'effectiveness': 10060, 'relieved': 10061, 'opener': 10062, 'reunited': 10063, 'flocks': 10064, 'processed': 10065, 'amorim': 10066, 'maarten': 10067, 'antilles': 10068, 'constitute': 10069, '1964': 10070, 'uncle': 10071, 'sonia': 10072, 'beside': 10073, 'flurry': 10074, 'earn': 10075, 'frances': 10076, 'pinera': 10077, 'relics': 10078, 'saints': 10079, 'weeklong': 10080, 'conquered': 10081, 'hawass': 10082, 'enormous': 10083, 'eager': 10084, 'suppressed': 10085, 'husseinov': 10086, 'provocation': 10087, 'huygens': 10088, 'roque': 10089, 'egg': 10090, 'abandoning': 10091, 'rightly': 10092, 'sampling': 10093, '1939': 10094, 'dp': 10095, 'bakiev': 10096, 'brasilia': 10097, 'anti-islamic': 10098, 'umbrella': 10099, 'turbulent': 10100, 'mental': 10101, 'confronted': 10102, 'two-state': 10103, 'foreclosure': 10104, 'cruz': 10105, 'fend': 10106, 'profitability': 10107, 'renewing': 10108, 'prefecture': 10109, '7.9': 10110, 'sorry': 10111, 'demolished': 10112, '40-year-old': 10113, 'daughters': 10114, 'outage': 10115, 'anatoly': 10116, 'kampala': 10117, '81-year-old': 10118, 'undetermined': 10119, 'protein': 10120, 'non-muslims': 10121, 'disturbed': 10122, 'inspire': 10123, 'widen': 10124, 'bacteria': 10125, 'caps': 10126, 'dprk': 10127, '1950': 10128, 'myung-bak': 10129, 'guyana': 10130, 'abolition': 10131, 'downplayed': 10132, 'auspices': 10133, 'far-right': 10134, 'cdc': 10135, 'inflicted': 10136, 'bakery': 10137, 'nujoma': 10138, 'construct': 10139, 'scarce': 10140, 'outer': 10141, 'renewal': 10142, 'bull': 10143, 'unocal': 10144, 'mentioning': 10145, 'anfal': 10146, 'suffocation': 10147, 'barcodes': 10148, 'bolivians': 10149, 'silvan': 10150, 'padilla': 10151, 'lone': 10152, 'metro': 10153, 'shaft': 10154, 'mardi': 10155, 'gras': 10156, 'ngwira': 10157, 'iles': 10158, 'dolphin': 10159, 'stewart': 10160, 'goats': 10161, '11-day': 10162, 'donation': 10163, 'preacher': 10164, 'infamous': 10165, 'seismologists': 10166, 'tianjin': 10167, '550': 10168, 'pricing': 10169, 'cyber': 10170, 'protects': 10171, 'referees': 10172, 'belief': 10173, 'auckland': 10174, 'flour': 10175, 'chaco': 10176, '1932': 10177, 'noise': 10178, 'bei': 10179, 'caucus': 10180, 'privileges': 10181, 'gatlin': 10182, 'aided': 10183, '1955': 10184, 'backlash': 10185, 'agca': 10186, 'elton': 10187, 'poisonous': 10188, 'smooth': 10189, 'securities': 10190, '120-member': 10191, 'francis': 10192, 'qari': 10193, 'abubakar': 10194, 'borrow': 10195, 'cordoned': 10196, 'haas': 10197, 'hydrocarbons': 10198, 'conquest': 10199, 'halliburton': 10200, 'advocating': 10201, 'danielle': 10202, 'manning': 10203, 'fences': 10204, 'cardenas': 10205, 'lip': 10206, 'rohmer': 10207, 'absent': 10208, 'erik': 10209, 'solheim': 10210, 'km': 10211, 'rezko': 10212, 'interrogations': 10213, 'republican-controlled': 10214, 'pre-war': 10215, 'adjust': 10216, 'consciousness': 10217, 'conferred': 10218, 'fillon': 10219, 'abusive': 10220, 'consult': 10221, 'molina': 10222, 'stripped': 10223, 'zia': 10224, 'isle': 10225, 'aswat': 10226, 'raffarin': 10227, 'mukasey': 10228, 'jos': 10229, 'homicide': 10230, 'non-violent': 10231, 'hercules': 10232, 'ramda': 10233, 'rsf': 10234, 'concede': 10235, 'djindjic': 10236, 'sticking': 10237, 'insecurity': 10238, 'expansionary': 10239, 'filipinos': 10240, 'mashaie': 10241, 'constituency': 10242, '101': 10243, 'breakfast': 10244, 'stern': 10245, 'exams': 10246, 'jameson': 10247, 'grill': 10248, 'standby': 10249, 'fiery': 10250, 'bakri': 10251, 'cosatu': 10252, 'monk': 10253, 'mullen': 10254, 'traded': 10255, 'affiliates': 10256, 'fda': 10257, 'moiseyev': 10258, 'reyna': 10259, 'steele': 10260, 'acceptable': 10261, 'witty': 10262, 'flea': 10263, 'pius': 10264, 'ltte': 10265, 'axum': 10266, 'pastrana': 10267, 'dome': 10268, 'papadopoulos': 10269, 'designs': 10270, 'srebotnik': 10271, 'mourn': 10272, 'nurses': 10273, 'nun': 10274, 'badghis': 10275, 'mikati': 10276, 'fitting': 10277, 'jankovic': 10278, 'falun': 10279, 'gong': 10280, 'gamsakhurdia': 10281, 'asteroid': 10282, 'bishara': 10283, 'upscale': 10284, 'python': 10285, 'cerkez': 10286, 'trench': 10287, 'tripura': 10288, 'mowlam': 10289, 'defect': 10290, 'partition': 10291, 'hospitalization': 10292, 'co-founder': 10293, 'railways': 10294, 'disturbance': 10295, 'raila': 10296, '\x85': 10297, '\x94': 10298, 'truly': 10299, 'academics': 10300, 'attendees': 10301, 'hiriart': 10302, 'chiapas': 10303, 'mysterious': 10304, 'execute': 10305, 'brigadier-general': 10306, 'sabah': 10307, 'contradicts': 10308, 'kinds': 10309, 'foundations': 10310, 'immunization': 10311, 'dire': 10312, 'deprived': 10313, 'sleep': 10314, 'diversifying': 10315, 'continuous': 10316, 'siphoning': 10317, 'constraints': 10318, 'year-old': 10319, 'unequal': 10320, 'peasant': 10321, 'lesson': 10322, 'potato': 10323, 'sacks': 10324, 'pink': 10325, 'mansion': 10326, 'geoff': 10327, 'testifying': 10328, 'kupwara': 10329, 'fried': 10330, 'sanders': 10331, 'platforms': 10332, 'pleas': 10333, 'frivolous': 10334, 'comfortable': 10335, 'filipino': 10336, 'conferences': 10337, 'alex': 10338, 'beset': 10339, 'inexpensive': 10340, 'abed': 10341, 'aghazadeh': 10342, 'tightly': 10343, 'complications': 10344, 'exhumed': 10345, 'rolling': 10346, 'lincoln': 10347, 'raiders': 10348, 'lleyton': 10349, 'number-two': 10350, 'second-seeded': 10351, 'florian': 10352, 'andreas': 10353, 'jarkko': 10354, 'kenneth': 10355, 'mandy': 10356, 'specialty': 10357, 'kilometer-per-hour': 10358, 'enforcing': 10359, 'injected': 10360, 'tuz': 10361, 'visibility': 10362, 'sympathy': 10363, 'archaeological': 10364, 'prize-winning': 10365, '82-year-old': 10366, 'exercising': 10367, 'assemble': 10368, 'court-martial': 10369, 'humiliating': 10370, 'robbing': 10371, 'al-haidari': 10372, 'deepen': 10373, 'proving': 10374, 'boosts': 10375, 'piano': 10376, 'plague': 10377, 'cartoonist': 10378, 'editors': 10379, 'azeris': 10380, 'thrift': 10381, 'fertile': 10382, 'highlands': 10383, 'alumina': 10384, 'estates': 10385, 'injustice': 10386, 'nest': 10387, 'trampled': 10388, 'paintings': 10389, 'artwork': 10390, 'clay': 10391, 'plain': 10392, '52-year-old': 10393, 'blank': 10394, 'prijedor': 10395, 'blessings': 10396, 'abraham': 10397, 'nicosia': 10398, 'impartial': 10399, 'telecommunication': 10400, 'provider': 10401, 'months-long': 10402, 'diwaniyah': 10403, 'ruined': 10404, 'dioxin': 10405, 'groin': 10406, 'al-qaeda': 10407, 'kommersant': 10408, 'sinhalese': 10409, 'missionary': 10410, 'relaxed': 10411, 'qalqiliya': 10412, 'nimroz': 10413, 'naqib': 10414, '25-member': 10415, 'lithuanian': 10416, 'adamkus': 10417, 'vastly': 10418, 'egyptian-born': 10419, 'complying': 10420, 'pullback': 10421, 'two-year-old': 10422, 'swaths': 10423, 'firmly': 10424, 'criticisms': 10425, 'challengers': 10426, 'hiroyuki': 10427, 'hosoda': 10428, 'occurrence': 10429, 'saddened': 10430, 'ike': 10431, 'chooses': 10432, 'icon': 10433, 'franklin': 10434, 'roosevelt': 10435, 'paralyzed': 10436, 'shabelle': 10437, 'tariff': 10438, 'low-priced': 10439, 'malay': 10440, 'anti-submarine': 10441, 'premises': 10442, 'widened': 10443, 'phosphates': 10444, 'gray': 10445, 'catalyst': 10446, 'spared': 10447, 'high-grade': 10448, 'connections': 10449, 'neglect': 10450, 'diminished': 10451, 'peaked': 10452, '3,50,000': 10453, 'module': 10454, 'soyuz': 10455, 'gaps': 10456, 'gholam': 10457, 'haddad': 10458, 'endanger': 10459, 'exceptional': 10460, 'censor': 10461, '90,000': 10462, 'jams': 10463, 'british-based': 10464, 'airfield': 10465, 'typhoons': 10466, 'splinter': 10467, 'gleneagles': 10468, 'implicates': 10469, 'roaming': 10470, 'assignments': 10471, 'bombmaker': 10472, 'hopman': 10473, 'perth': 10474, 'group-a': 10475, 'hijacker': 10476, 'surrendering': 10477, 'circle': 10478, 'accomplices': 10479, 'avenge': 10480, 'subpoena': 10481, 'believers': 10482, 'responses': 10483, 'madeleine': 10484, 'farewell': 10485, 'bayji': 10486, 'drawings': 10487, 'objection': 10488, 'foes': 10489, 'prohibiting': 10490, 'instrument': 10491, 'vigilant': 10492, 'uphold': 10493, 'tracked': 10494, 'fundamentalist': 10495, 'recorder': 10496, 'blizzard': 10497, 'tijuana': 10498, 'blatant': 10499, 'al-zawahri': 10500, 'suspensions': 10501, 'punishments': 10502, 'superiors': 10503, 'hindering': 10504, 'reacting': 10505, 'unconditional': 10506, 'haste': 10507, '22nd': 10508, 'withhold': 10509, 'satisfactory': 10510, 'c': 10511, '1783': 10512, 'slovenes': 10513, 'distanced': 10514, 'absorbed': 10515, 'mainstays': 10516, 'complicate': 10517, 'prefer': 10518, 'bramble': 10519, 'hedge': 10520, 'vain': 10521, 'personality': 10522, 'junk': 10523, 'auditing': 10524, 'annulled': 10525, 'corrected': 10526, 'surges': 10527, 'mulford': 10528, 'shyam': 10529, 'petersen': 10530, 'tsunami-hit': 10531, '225-seat': 10532, 'maize': 10533, 'semlow': 10534, 'mastery': 10535, 'harriet': 10536, 'democratization': 10537, 'secondhand': 10538, '430': 10539, 'rugova': 10540, 'contrast': 10541, 'cindy': 10542, 'aye': 10543, 'daniele': 10544, 'repubblica': 10545, 'sentiments': 10546, 'reversing': 10547, 'co-chair': 10548, '49-year-old': 10549, 'covertly': 10550, 'signaling': 10551, 'tnk-bp': 10552, 'enrique': 10553, 'oumarou': 10554, 'amadou': 10555, 'nardiello': 10556, 'rejoin': 10557, 'shoot-out': 10558, 'instill': 10559, 'paddy': 10560, 'bow': 10561, 'assure': 10562, 'tender': 10563, 'anemic': 10564, 'railroad': 10565, 'specifics': 10566, 'chinamasa': 10567, '25th': 10568, 'mid-afternoon': 10569, 'kick': 10570, 'inconclusive': 10571, 'head-on': 10572, 'moussaoui': 10573, 'dumping': 10574, 'accelerated': 10575, 'davos': 10576, 'stormy': 10577, 'availability': 10578, 'non-essential': 10579, 'clamp': 10580, 'petty': 10581, 'diversity': 10582, 'siberian': 10583, 'feast': 10584, 'kings': 10585, 'infiltration': 10586, 'imposes': 10587, 'tries': 10588, 'normalize': 10589, 'erosion': 10590, 'aggravated': 10591, 'affiliate': 10592, 'universe': 10593, 'authenticated': 10594, 'hispanics': 10595, 'caucuses': 10596, 'revelers': 10597, 'pack': 10598, 'samba': 10599, 'costumes': 10600, 'showdown': 10601, 'filibusters': 10602, 'jetliners': 10603, 'sirnak': 10604, 'growers': 10605, 'face-to-face': 10606, 'rewrite': 10607, 'beachfront': 10608, 'dishes': 10609, 'punching': 10610, 'handcuffs': 10611, 'ekmeleddin': 10612, 'ihsanoglu': 10613, 'rove': 10614, '1,70,000': 10615, 'recapture': 10616, 'spiraling': 10617, 'absolutely': 10618, 'implicating': 10619, 'market-oriented': 10620, 'impede': 10621, 'introducing': 10622, 'heal': 10623, 'prescribe': 10624, 'bitterly': 10625, 'lamented': 10626, '18-year-old': 10627, 'skirmish': 10628, 'ejected': 10629, 'lagging': 10630, 'requirement': 10631, 'exhaustion': 10632, 'spends': 10633, 'ghanaian': 10634, 'conceived': 10635, 'occupy': 10636, 'mistreating': 10637, 'escalate': 10638, 'decapitated': 10639, 'likelihood': 10640, 'multi-billion': 10641, 'amanullah': 10642, '2,700': 10643, 'differs': 10644, 'sulawesi': 10645, 'tenerife': 10646, 'wildfire': 10647, 'autocracy': 10648, 'subsided': 10649, 'chart': 10650, 'applause': 10651, 'microphone': 10652, 'yasukuni': 10653, 'election-related': 10654, 'kate': 10655, 'mccann': 10656, 'celebrities': 10657, 'regarded': 10658, 'occupants': 10659, 'ak-47': 10660, 'prosecuting': 10661, 'ibero-american': 10662, 'assessed': 10663, 'smoothly': 10664, 'snapped': 10665, '5.1': 10666, 'ken': 10667, 'faulty': 10668, 'nato-afghan': 10669, 'schemes': 10670, 'cheema': 10671, 'jirga': 10672, 'precinct': 10673, 'pensions': 10674, 'cracks': 10675, 'residency': 10676, 'bypass': 10677, 'recovers': 10678, 'collective': 10679, 'pranab': 10680, 'bombarded': 10681, 'skepticism': 10682, 'intensifying': 10683, 'barring': 10684, 'bargaining': 10685, 'pairs': 10686, 'grandmother': 10687, 'grabs': 10688, '25-year': 10689, 'little-known': 10690, 'unverifiable': 10691, 'sprayed': 10692, 'dadfar': 10693, 'qureshi': 10694, 'mere': 10695, 'pegged': 10696, 'procurement': 10697, 'remarkable': 10698, 'cleansing': 10699, 'perished': 10700, 'hurts': 10701, 'nile': 10702, 'frame': 10703, 'boroujerdi': 10704, 'desk': 10705, 'forgiven': 10706, 'luge': 10707, 'shattering': 10708, 'shrank': 10709, '0.4': 10710, 'salih': 10711, 'josh': 10712, 'ropes': 10713, 'sounded': 10714, 'feuding': 10715, 'lessons': 10716, 'impression': 10717, 'spirits': 10718, 'geneina': 10719, 'mathieu': 10720, 'babies': 10721, 'jaroslaw': 10722, 'sickened': 10723, 'malisse': 10724, 'guillermo': 10725, 'plaguing': 10726, 'sheets': 10727, 'practicing': 10728, 'treasure': 10729, 'bold': 10730, 'print': 10731, 'protections': 10732, 'badr': 10733, 'cristina': 10734, 'rebate': 10735, 'sana': 10736, '71': 10737, 'al-yousifi': 10738, 'qaim': 10739, 'incurable': 10740, 'restraints': 10741, 'conceded': 10742, 'crane': 10743, 'widows': 10744, 'sprint': 10745, '10-kilometer': 10746, 'canyon': 10747, 'ecosystem': 10748, 'downstream': 10749, 'discovering': 10750, 'philippe': 10751, 'ingrid': 10752, 'publicity': 10753, 'konstantin': 10754, 'luzon': 10755, 'weightlifting': 10756, 'holders': 10757, 'convertible': 10758, 'induced': 10759, 'nominal': 10760, 'efta': 10761, 'grey': 10762, 'vigorous': 10763, '2.9': 10764, 'pounds': 10765, '5.2': 10766, 'centralized': 10767, 'tow': 10768, 'insured': 10769, 'softball': 10770, 'cure': 10771, 'improves': 10772, 'disinfected': 10773, 'applies': 10774, 'fogh': 10775, 'battering': 10776, 'nova': 10777, 'fifty': 10778, 'grows': 10779, 'nobutaka': 10780, 'nimeiri': 10781, '79-year-old': 10782, '22-year': 10783, 'gunship': 10784, 'airstrip': 10785, 'build-up': 10786, 'bryan': 10787, 'gift': 10788, 'friction': 10789, 'fitzgerald': 10790, 'dresden': 10791, 'yorker': 10792, 'plame': 10793, 'flipped': 10794, 'drenched': 10795, 'garmser': 10796, 'tangerang': 10797, '29-year-old': 10798, 'transmissible': 10799, 'zhari': 10800, 'expands': 10801, 'warmest': 10802, 'arming': 10803, 'olli': 10804, 'heinonen': 10805, 'mau': 10806, '1952': 10807, 'undersea': 10808, 'rattle': 10809, 'retracted': 10810, 'onlookers': 10811, 'relation': 10812, 'accumulated': 10813, 'wo': 10814, 'inviting': 10815, 'headlines': 10816, 'provoking': 10817, 'recount': 10818, 'government-in-exile': 10819, 'seven-year-old': 10820, 'greenpeace': 10821, '280': 10822, 'loyalty': 10823, 'pro-opposition': 10824, 'katowice': 10825, 'nearest': 10826, 'three-member': 10827, 'sidr': 10828, 'derives': 10829, 'self-rule': 10830, '16-year': 10831, 'privatize': 10832, 'unhappy': 10833, 'pasture': 10834, 'punishing': 10835, 'thereafter': 10836, 'barbara': 10837, 'disgusting': 10838, 'secede': 10839, 'majid': 10840, 'aged': 10841, 'mid-december': 10842, 'yu': 10843, 'adams': 10844, 'stabbing': 10845, 'overthrown': 10846, 'herbert': 10847, 'specified': 10848, 'oppressed': 10849, 'strides': 10850, 'joaquin': 10851, 'navarro-valls': 10852, 'bishops': 10853, '1924': 10854, 'segolene': 10855, 'encounter': 10856, 'loaned': 10857, 'money-laundering': 10858, 'susan': 10859, 'overhaul': 10860, 'al-baghdadi': 10861, 'poppies': 10862, 'fahd': 10863, 'al-sabah': 10864, 'unwelcome': 10865, 'comrades': 10866, 'ridicule': 10867, 'demolish': 10868, 'jalgaon': 10869, 'seals': 10870, 'hearts': 10871, 'therefore': 10872, 'evicted': 10873, 'scenarios': 10874, 'advises': 10875, 'stuttgart': 10876, 'vaz': 10877, 'almeida': 10878, 'torrijos': 10879, 'splitting': 10880, 'versus': 10881, 'aluminum': 10882, 'repatriate': 10883, 'arkansas': 10884, 'horseback': 10885, 'begged': 10886, 'physicist': 10887, 'bucket': 10888, 'solved': 10889, 'modeled': 10890, 'harmony': 10891, 'billingslea': 10892, 'creates': 10893, 'roadmap': 10894, 'ultranationalist': 10895, 'confederations': 10896, 'three-hour': 10897, 'julia': 10898, 'robertson': 10899, 'butt': 10900, 'pascal': 10901, 'lakshman': 10902, 'kadirgamar': 10903, 'aslan': 10904, 'consistently': 10905, 'unconfirmed': 10906, '38-year-old': 10907, 'shaikh': 10908, 'facilitating': 10909, 'sermon': 10910, 'federico': 10911, 'lombardi': 10912, 'samir': 10913, 'rabbi': 10914, 'judaism': 10915, 'lightly': 10916, 'sundown': 10917, 'reed': 10918, 'delphi': 10919, 'rebounding': 10920, '3.3': 10921, '1814': 10922, '1905': 10923, 'referenda': 10924, 'bent': 10925, 'kigali': 10926, 'edwards': 10927, 'frontrunner': 10928, 'inspiration': 10929, 'chittagong': 10930, 'drain': 10931, 'quadruple': 10932, 'tournaments': 10933, 'warm-up': 10934, 'breath': 10935, 'variant': 10936, 'reformers': 10937, 'obligation': 10938, 'bernie': 10939, 'disagree': 10940, '2,400': 10941, 'killers': 10942, 'titanosaurs': 10943, 'jordanian-born': 10944, 'dodd': 10945, '12-month': 10946, 'departed': 10947, 'receding': 10948, 'instructions': 10949, 'airlifted': 10950, 'pleading': 10951, 'contend': 10952, 'swede': 10953, 'take-off': 10954, 'frequencies': 10955, 'tsunamis': 10956, 'artificial': 10957, 'slipping': 10958, 'wiping': 10959, '0.5': 10960, 'pays': 10961, '230': 10962, 'airlift': 10963, 'nabil': 10964, 'jolted': 10965, 'bam': 10966, 'ca': 10967, 'administering': 10968, 'excluded': 10969, 'nouakchott': 10970, '144': 10971, 'defaulted': 10972, 'navigation': 10973, 'sgrena': 10974, 'engulfed': 10975, 'nath': 10976, 'truckers': 10977, 'ride': 10978, 'transforming': 10979, 'lukoil': 10980, 'neverland': 10981, '4-0': 10982, 'accorded': 10983, 'jeep': 10984, 'clogged': 10985, 'elias': 10986, 'mena': 10987, 'olsson': 10988, 'stall': 10989, 'wreck': 10990, 'cease-fires': 10991, 'brazzaville': 10992, '9.8': 10993, 'deployments': 10994, 'folk': 10995, 'casey': 10996, 'sweeps': 10997, 'shoppers': 10998, 'erez': 10999, 'mid-june': 11000, 'subdue': 11001, 'h.i.v.': 11002, 'consensual': 11003, 'defines': 11004, 'invite': 11005, 'debates': 11006, 'fatwa': 11007, 'taba': 11008, '1928': 11009, 'urine': 11010, 'enable': 11011, 'undercut': 11012, 'bedouin': 11013, 'stir': 11014, '10-year-old': 11015, 'frightened': 11016, 'teamed': 11017, 'translation': 11018, '155': 11019, 'tokelau': 11020, 'underdeveloped': 11021, 'encouragement': 11022, 'dwellings': 11023, 'arlington': 11024, '137': 11025, 'escaping': 11026, 'homelands': 11027, 'ruthless': 11028, 'flare': 11029, 'necdet': 11030, 'sezer': 11031, 'swedes': 11032, 'consuming': 11033, 'projections': 11034, '1867': 11035, 'flies': 11036, 'kathleen': 11037, 'noguchi': 11038, 'defused': 11039, 'streamline': 11040, 'affluent': 11041, 'fairness': 11042, 'quitting': 11043, 'premiere': 11044, 'convincing': 11045, 'agreeing': 11046, 'domingo': 11047, 'resurfaced': 11048, 'pornography': 11049, 'appropriately': 11050, 'menatep': 11051, 'overshot': 11052, 'pumped': 11053, 'hizbul': 11054, 'figueredo': 11055, 'sabotaged': 11056, 'presentation': 11057, 'year-earlier': 11058, 'start-up': 11059, 'softer': 11060, 'hockey': 11061, '1946': 11062, 'reestablished': 11063, 'al-asad': 11064, 'shane': 11065, 'group-b': 11066, 'trafficked': 11067, 'fortune': 11068, 'ah': 11069, 'qingdao': 11070, 'kemal': 11071, 'cancellations': 11072, 'fared': 11073, 'amazon': 11074, 'petroecuador': 11075, 'basin': 11076, 'thirty': 11077, 'pan': 11078, 'institutional': 11079, 'pri': 11080, 'ailment': 11081, 'electing': 11082, 'conclusions': 11083, 'masterminds': 11084, 'lovato': 11085, 'citgo': 11086, 'shorter': 11087, 'publishes': 11088, 'noriega': 11089, 'neutralized': 11090, 'joel': 11091, 'u.n.-congolese': 11092, '21-member': 11093, 'personalities': 11094, 'caches': 11095, 'panamanian-flagged': 11096, 'equitable': 11097, 'nights': 11098, 'petroleos': 11099, 'portfolio': 11100, 'impending': 11101, 'hugh': 11102, 'market-based': 11103, 'contributor': 11104, 'alleviate': 11105, 'jay': 11106, 'peacocks': 11107, 'annoyed': 11108, 'claws': 11109, 'auditors': 11110, 'waterborne': 11111, 'self-government': 11112, 'sir': 11113, 'unreasonable': 11114, 'spearheaded': 11115, 'cats': 11116, 'asians': 11117, 'seasons': 11118, 'injure': 11119, 'moro': 11120, 'slumping': 11121, 'u2': 11122, '167': 11123, 'kalam': 11124, 'clears': 11125, 'distinctive': 11126, 'carl': 11127, 'streak': 11128, 'gregory': 11129, 'psychiatric': 11130, 'zulia': 11131, 'warner': 11132, 'dominguez': 11133, 'raced': 11134, 'three-tenths': 11135, 'pretty': 11136, 'khel': 11137, 'airlifting': 11138, 'marco': 11139, 'tours': 11140, 'applicants': 11141, 'stepped-up': 11142, 'bruce': 11143, 'golding': 11144, 'nathan': 11145, 'justine': 11146, 'commuters': 11147, 'miner': 11148, 'wiretapping': 11149, 'wiretaps': 11150, 'das': 11151, 'gateway': 11152, 'peasants': 11153, 'roots': 11154, 'homosexuality': 11155, 'primerica': 11156, 'endowed': 11157, 'relied': 11158, 'secretive': 11159, 'opted': 11160, 'disguise': 11161, 'statue': 11162, '2,30,000': 11163, '21,000': 11164, 'clarification': 11165, 'angrily': 11166, 'spinal': 11167, 'reimposed': 11168, 'vary': 11169, 'communist-era': 11170, 'rhetoric': 11171, 'fiction': 11172, '53-nation': 11173, 'sway': 11174, 'beast': 11175, 'rajoelina': 11176, 'mid-march': 11177, 'advocated': 11178, 'stretching': 11179, 'melted': 11180, 'subaru': 11181, 'demobilization': 11182, '37-year-old': 11183, 'banker': 11184, 'gearing': 11185, 'sleeping': 11186, 'charsadda': 11187, 'kam': 11188, 'al-hindi': 11189, 'unexploded': 11190, 'painted': 11191, 'thin': 11192, 'four-decade': 11193, 'trace': 11194, 'edelman': 11195, 'ackerman': 11196, 'harvesting': 11197, 'manpower': 11198, 'rank': 11199, 'purely': 11200, 'guiana': 11201, 'sugarcane': 11202, 'stating': 11203, 'debated': 11204, 'sheik': 11205, 'climbing': 11206, 'emotional': 11207, 'lobbyist': 11208, 'knowing': 11209, 'reprinting': 11210, 'depiction': 11211, 'reprinted': 11212, 'urdu-speaking': 11213, 'pashtuns': 11214, 'teammate': 11215, 'deplored': 11216, 'galle': 11217, 'lankans': 11218, 'don': 11219, 'laughed': 11220, 'bannu': 11221, 'cursed': 11222, 'karroubi': 11223, 'recordings': 11224, 'ghulam': 11225, '22,000': 11226, 'choking': 11227, 'steven': 11228, 'hamas-controlled': 11229, 'reelection': 11230, 'canisters': 11231, 'catching': 11232, 'islamist-rooted': 11233, 'backyard': 11234, 'rand': 11235, '1815': 11236, 'colonized': 11237, '1951': 11238, 'subdued': 11239, 'ex-president': 11240, 'bertrand': 11241, 'johndroe': 11242, 'xi': 11243, 'battalion': 11244, 'tolerated': 11245, 'topol-m': 11246, 'topol': 11247, 'camped': 11248, 'award-winning': 11249, 'choreographer': 11250, 'nobody': 11251, 'traced': 11252, 'degrading': 11253, 'bratislava': 11254, 'slovak': 11255, 'pyramid': 11256, 'painting': 11257, 'farouk': 11258, 'weighing': 11259, 'hindus': 11260, 'birthplace': 11261, 'u.n.-sponsored': 11262, 'atrophy': 11263, 'capped': 11264, 'commended': 11265, 'zuloaga': 11266, 'peyton': 11267, 'onboard': 11268, 'equus': 11269, 'waterways': 11270, 'bare': 11271, 'minimal': 11272, 'agrarian': 11273, 'oxen': 11274, 'lamb': 11275, 'oh': 11276, 'enabling': 11277, 'thirst': 11278, 'certificate': 11279, 'softening': 11280, 'pretext': 11281, 'vandalism': 11282, 'dale': 11283, 'three-match': 11284, 'reassured': 11285, 'forbid': 11286, 'infecting': 11287, 'usaid': 11288, 'longwang': 11289, 'cartagena': 11290, 'paerson': 11291, 'differ': 11292, 'installment': 11293, 'sharma': 11294, 'escorting': 11295, 'reckless': 11296, 'sofia': 11297, 'comprised': 11298, 'grammys': 11299, 'dignity': 11300, 'orderly': 11301, 'guitar': 11302, 'on-going': 11303, 'endeavour': 11304, 'salvadoran': 11305, 'intellectual': 11306, 'rebuilt': 11307, 'rafiq': 11308, 'exploitable': 11309, 'peg': 11310, '1957': 11311, 'wrath': 11312, 'sung': 11313, 'hammered': 11314, 'micheletti': 11315, 'reclusive': 11316, 'earner': 11317, 'five-year-old': 11318, 'theo': 11319, 'bouyeri': 11320, 'normalization': 11321, 'sighted': 11322, 'medecins': 11323, 'bihar': 11324, 'derbez': 11325, 'eighteen': 11326, 'nyamwasa': 11327, 'benn': 11328, 'dock': 11329, 'scorpions': 11330, 'reconstruct': 11331, 'parks': 11332, 'planner': 11333, 'memoir': 11334, "shi'ite-led": 11335, 'symbolic': 11336, 'hispaniola': 11337, 'bore': 11338, 'summits': 11339, 'larry': 11340, 'dd': 11341, "dunkin'": 11342, 'palm': 11343, 'decreasing': 11344, '1936': 11345, 'franco': 11346, 'sixteen': 11347, 'corner': 11348, 'tzipi': 11349, '41-year-old': 11350, 'gallon': 11351, 'animated': 11352, 'referee': 11353, 'singers': 11354, 'lips': 11355, 'unwanted': 11356, 'expertise': 11357, 'grief': 11358, 'sided': 11359, 'jing': 11360, 'guatemalan': 11361, 'miroslav': 11362, 'nastase': 11363, 'pensioners': 11364, 'maneuver': 11365, 'bosch': 11366, 'boskovski': 11367, 'chee-hwa': 11368, '74': 11369, 'conventions': 11370, 'u.s.-iraqi': 11371, '320': 11372, 'ashkelon': 11373, '71-year-old': 11374, 'beattie': 11375, 'nur': 11376, 'aegean': 11377, 'timberlake': 11378, 'non-combat': 11379, 'non-american': 11380, 'ripping': 11381, 'township': 11382, 'hamadi': 11383, 'segments': 11384, 'principally': 11385, 'adjustment': 11386, 'dhekelia': 11387, 'fold': 11388, 'fatigue': 11389, 'interrupting': 11390, '3.4': 11391, 'pedophile': 11392, 'oscar': 11393, 'arias': 11394, 'backdrop': 11395, 'concealed': 11396, 'typically': 11397, 'wrestler': 11398, 'purge': 11399, 'steroid': 11400, 'cauldron': 11401, 'simultaneously': 11402, 'low-income': 11403, 'pouring': 11404, 'chlorine': 11405, 'suitcase': 11406, 'allowance': 11407, 'sums': 11408, 'napolitano': 11409, 'flatly': 11410, 'kumar': 11411, 'rests': 11412, 'snowstorm': 11413, 'ndc': 11414, 'safeguards': 11415, 'quang': 11416, 'incitement': 11417, 'culminated': 11418, 'unabated': 11419, 'owe': 11420, 'policymakers': 11421, 'concerted': 11422, 'seminar': 11423, 'jamie': 11424, 'sen': 11425, 'spiral': 11426, 'recalling': 11427, 'explains': 11428, 'warmup': 11429, 'kicking': 11430, 'drums': 11431, 'cost-cutting': 11432, "'re": 11433, 'decisive': 11434, 'vine': 11435, 'mood': 11436, 'one-half': 11437, 'mule': 11438, 'handgun': 11439, 'listeners': 11440, 'beck': 11441, 'ortiz': 11442, 'jaffer': 11443, 'laxman': 11444, 'defying': 11445, 'panicked': 11446, 'restated': 11447, 'eprdf': 11448, 'fossil': 11449, 'fuji': 11450, 'daimlerchrysler': 11451, 'zebra': 11452, 'copei': 11453, 'couples': 11454, 'obvious': 11455, 'files': 11456, 'austin': 11457, 'stalin': 11458, 'stagnated': 11459, 'niue': 11460, 'ant': 11461, 'honey': 11462, 'cream': 11463, 'dove': 11464, 'battlefield': 11465, 'aftershocks': 11466, 'bakool': 11467, 'sevastopol': 11468, 'nabi': 11469, 'bauer': 11470, 'fattal': 11471, '50-year-old': 11472, 'ilham': 11473, 'armstrong': 11474, 'genetic': 11475, 'nonproliferation': 11476, 'earning': 11477, 'tumors': 11478, 'horizons': 11479, 'bonuses': 11480, 'stuck': 11481, 'answering': 11482, 'mosquito': 11483, 'nerve': 11484, 'rent': 11485, 'sandstorm': 11486, 'frigate': 11487, 'models': 11488, 'pittsburgh': 11489, 'finalizing': 11490, 'jerzy': 11491, 're-liberation': 11492, 'robredo': 11493, 'ayad': 11494, 'unsealed': 11495, 'tech': 11496, 'iraqiya': 11497, 'claus': 11498, 'bonus': 11499, 'nine-year-old': 11500, 'perisic': 11501, 'erratic': 11502, 'preceding': 11503, 'presides': 11504, 'secrecy': 11505, 'atoll': 11506, 'lash': 11507, 'intercept': 11508, 'technique': 11509, 'siti': 11510, 'supari': 11511, 'subways': 11512, 'runner-up': 11513, 'funneling': 11514, 'stoning': 11515, 'devil': 11516, 'ugandans': 11517, 'capitalized': 11518, 'accuracy': 11519, 'inaccuracies': 11520, 'pat': 11521, '132': 11522, 'deadlines': 11523, 'stabbed': 11524, 'din': 11525, 'sketchy': 11526, 'ursula': 11527, 'prohibit': 11528, 'hospitality': 11529, 'fats': 11530, 'domino': 11531, 'verifying': 11532, 'provocations': 11533, 'ensuing': 11534, '1920s': 11535, 'flag-draped': 11536, 'disciplined': 11537, 'bajram': 11538, 'ceku': 11539, 'foster': 11540, 'janata': 11541, 'corridor': 11542, 'infants': 11543, 'backward': 11544, 'julio': 11545, 'designation': 11546, 'dawei': 11547, 'legalized': 11548, 'corporations': 11549, 'subjects': 11550, 'discusses': 11551, 'mortality': 11552, 'arena': 11553, 'al-khaznawi': 11554, 'gusts': 11555, 'u.s.-run': 11556, 'diyarbakir': 11557, 'perspective': 11558, 'clot': 11559, 'suits': 11560, 'apparatus': 11561, 'tajik': 11562, 'two-hour': 11563, 'underscore': 11564, 'ostrich': 11565, 'reprehensible': 11566, 'diwali': 11567, 'hun': 11568, '31,000': 11569, 'strife-torn': 11570, 'm': 11571, '35-member': 11572, 'discounts': 11573, 'ghad': 11574, 'informing': 11575, 'burundians': 11576, 'nagasaki': 11577, 'atom': 11578, 'consequence': 11579, 'tumor': 11580, 'delimit': 11581, 'kanu': 11582, 'disparities': 11583, 'deceased': 11584, 'kyodo': 11585, 'groenefeld': 11586, 'clemency': 11587, 'kai': 11588, 'inundated': 11589, 'juventus': 11590, 'clunkers': 11591, 'suez': 11592, '141': 11593, 'comprise': 11594, 'sham': 11595, '34-year-old': 11596, 'jumping': 11597, 'pitcairn': 11598, 'akrotiri': 11599, 'embroiled': 11600, 'undertook': 11601, 'degradation': 11602, 'davenport': 11603, 'mutually': 11604, 'practical': 11605, 'bouterse': 11606, 'shilpa': 11607, 'kissed': 11608, 'cords': 11609, 'cord': 11610, 'bayan': 11611, 'jabr': 11612, 'ariana': 11613, 'greek-owned': 11614, 'temples': 11615, 'collins': 11616, 'warhead': 11617, 'orissa': 11618, 'ningbo': 11619, 'racing': 11620, 'newer': 11621, 'pilgrimage': 11622, '1921': 11623, 'sutherland': 11624, 'friedan': 11625, 'infect': 11626, 'cannon': 11627, 'abortions': 11628, 'destabilizing': 11629, 'anti-missile': 11630, 'dairy': 11631, 'ramush': 11632, '14-year-old': 11633, 'careful': 11634, 'coastlines': 11635, 'exceeds': 11636, 'hoon': 11637, 'run-up': 11638, 'anti-israel': 11639, 'wheeler': 11640, 'terror-related': 11641, 'commentary': 11642, 'camel': 11643, 'misery': 11644, 'inaccessible': 11645, 'garrison': 11646, 'wet': 11647, 'ourselves': 11648, 'short-selling': 11649, 'mutua': 11650, 'leasing': 11651, 'chad-sudan': 11652, 're-write': 11653, 'abdallah': 11654, 'outline': 11655, 'reinforced': 11656, 'adkins': 11657, 'homecoming': 11658, '40-year': 11659, 'bolstered': 11660, 'weigh': 11661, 'double-digit': 11662, '1492': 11663, 'paths': 11664, 'karim': 11665, 'choudhury': 11666, 'negatively': 11667, 'futures': 11668, 'solider': 11669, 'carlo': 11670, 'tsunami-devastated': 11671, 'corporal': 11672, 'converging': 11673, 'giants': 11674, 'usher': 11675, 'associations': 11676, '32,000': 11677, 'shrapnel': 11678, 'desai': 11679, 'wisdom': 11680, 'oo': 11681, 'bollea': 11682, 'zetas': 11683, 'lease': 11684, 'kizza': 11685, 'sadness': 11686, 'paisley': 11687, 'sinn': 11688, 'fein': 11689, 'recommending': 11690, '5.4': 11691, 'mate': 11692, 'depictions': 11693, 'kulov': 11694, '1872': 11695, 'fearfully': 11696, 'hemp': 11697, 'foca': 11698, '1922': 11699, 'g7': 11700, 'medic': 11701, 'horror': 11702, 'ardzinba': 11703, 'khadjimba': 11704, 'pedestrians': 11705, 'toys': 11706, 'dallas': 11707, 'bahr': 11708, 'affection': 11709, 'pluto': 11710, 'injection': 11711, 'insistence': 11712, 'genetically': 11713, 'shaped': 11714, 'oumar': 11715, 'deterioration': 11716, 'nevzlin': 11717, 'non-energy': 11718, 'parachinar': 11719, 'sphere': 11720, 'duekoue': 11721, 'ramzan': 11722, 'videotapes': 11723, 'epsilon': 11724, 'doctrine': 11725, 'zaragoza': 11726, 'wheelchair': 11727, 'weaker': 11728, 'avert': 11729, 'sanchez': 11730, 'pa': 11731, 'crept': 11732, 'assumes': 11733, 'shoving': 11734, 'al-harbi': 11735, 'bed': 11736, 'eternal': 11737, 'yearly': 11738, 'seniors': 11739, 'maiduguri': 11740, 'sandinista': 11741, 'lobby': 11742, '109': 11743, 'hogg': 11744, 'sudharmono': 11745, 'photographers': 11746, 'seselj': 11747, 'curve': 11748, '444': 11749, 'slaying': 11750, 'tomatoes': 11751, 'firearms': 11752, 'rivalry': 11753, 'trent': 11754, 'uhm': 11755, 'spacewalks': 11756, 'slots': 11757, 'betting': 11758, 'bluefin': 11759, 'measles': 11760, 'karbouli': 11761, 'drinks': 11762, 'malpractice': 11763, 'josef': 11764, '747': 11765, 'explosive-laden': 11766, 'overflowed': 11767, 'lebedev': 11768, 'defy': 11769, 'creatures': 11770, 'playoffs': 11771, 'roofs': 11772, 'andijon': 11773, 'owl': 11774, 'scam': 11775, 'di': 11776, 'solitary': 11777, 'comeback': 11778, 'stein': 11779, 'foolish': 11780, 'thorn': 11781, 'reply': 11782, 'jeffrey': 11783, 'five-story': 11784, 'echoupal': 11785, 'stirring': 11786, 'buckovski': 11787, 'clinical': 11788, 'bolt': 11789, 'shenzhou': 11790, 'worshipers': 11791, 'vajpayee': 11792, 'traveler': 11793, 'cobain': 11794, 'martyrdom': 11795, 'craybas': 11796, 'melt': 11797, 'pole': 11798, 'parma': 11799, 'anheuser-busch': 11800, 'caldera': 11801, 'qantas': 11802, 'chang': 11803, 'antibiotics': 11804, 'materazzi': 11805, 'newseum': 11806, 'benesova': 11807, 'bild': 11808, "l'aquila": 11809, 'acosta': 11810, 'banderas': 11811, 'viper': 11812, 'foreman': 11813, 'advani': 11814, 'karabilah': 11815, 'satterfield': 11816, 'sliding': 11817, 'sperling': 11818, 'lapdog': 11819, 'nist': 11820, 'adverse': 11821, 'terrain': 11822, 'ellison': 11823, 'groundhog': 11824, 'convergence': 11825, 'lithium': 11826, 'hondo': 11827, 'guernsey': 11828, 'fayaz': 11829, 'karpinski': 11830, 'opel': 11831, 'hakim': 11832, 'mattel': 11833, 'fleihan': 11834, 'aguilar': 11835, 'zinser': 11836, 'boudhiba': 11837, 'arauca': 11838, 'czechs': 11839, 'rabbis': 11840, 'beiring': 11841, 'jolie': 11842, 'mprp': 11843, 'echeverria': 11844, 'certificates': 11845, 'qanuni': 11846, 'shotgun': 11847, 'buechel': 11848, '1,27,000': 11849, 'fouad': 11850, 'designers': 11851, 'spaceshipone': 11852, 'designer': 11853, 'richter': 11854, 'entourage': 11855, 'lounge': 11856, 'disclosure': 11857, '65-year-old': 11858, 'vying': 11859, 'congresswoman': 11860, '96,000': 11861, '1,10,000': 11862, 'mid-1970s': 11863, 'promoter': 11864, 'oppression': 11865, '290': 11866, 'normandy': 11867, 'thani': 11868, 'generates': 11869, 'extraordinarily': 11870, 'contrive': 11871, 'zurab': 11872, 'leisure': 11873, 'grandfather': 11874, 'sack': 11875, 'benon': 11876, 'satisfaction': 11877, 'lit': 11878, 'mock': 11879, 'poonch': 11880, 'kentucky': 11881, 'effigy': 11882, '0.8': 11883, 'propelled': 11884, 'bahadur': 11885, 'deuba': 11886, 'campaign-finance': 11887, 'hotbed': 11888, 'endorsements': 11889, 'leukemia': 11890, 'bone': 11891, 'himalayas': 11892, 'displacement': 11893, 'gholamreza': 11894, 'juvenile': 11895, 'eyewitnesses': 11896, 'zvornik': 11897, 'thunder': 11898, 'topping': 11899, 'eighth-seeded': 11900, 'ignacio': 11901, 'chela': 11902, 'serra': 11903, 'carlsen': 11904, 'tune': 11905, 'unborn': 11906, 'incompatible': 11907, '1559': 11908, 'counter-attack': 11909, 'mughal': 11910, 'west-northwest': 11911, 'doctorate': 11912, 'overtures': 11913, 'inseparable': 11914, 'collapses': 11915, 'snowstorms': 11916, 'post-communist': 11917, 'statesman': 11918, 'lyudmila': 11919, 'connecting': 11920, 'skyrocketing': 11921, 'reliant': 11922, 'cosmonauts': 11923, 'demoted': 11924, 'simonyi': 11925, 'defiant': 11926, 'mazar-i-sharif': 11927, 'stuffed': 11928, 'toughen': 11929, 'curriculum': 11930, 'merit': 11931, 'readying': 11932, 'manuscript': 11933, 'beethoven': 11934, 'sotheby': 11935, 'library': 11936, 'palmer': 11937, 'duet': 11938, '1920': 11939, 'reluctance': 11940, 'saeed': 11941, 'sanader': 11942, 'ltd.': 11943, 'year-end': 11944, 'pursues': 11945, 'wrangling': 11946, 'cooler': 11947, '1838': 11948, 'pete': 11949, 'devalued': 11950, 'carib': 11951, 'vigorously': 11952, 'performs': 11953, 'gallery': 11954, '2,800': 11955, 'oriented': 11956, 'barrett': 11957, '04-jan': 11958, 'orphanage': 11959, 'endangering': 11960, 'rigid': 11961, '210': 11962, 'colliding': 11963, '1933': 11964, 'scandinavian': 11965, '454': 11966, 'sneak': 11967, 'loving': 11968, 'streamlining': 11969, 'materialize': 11970, 'fleming': 11971, 'hardship': 11972, 'embedded': 11973, 'indira': 11974, 'arbiter': 11975, 'agitation': 11976, 'jamming': 11977, 'portable': 11978, 'supportive': 11979, 'completes': 11980, 'meaningful': 11981, 'schoolchildren': 11982, 'uncommon': 11983, 'consultant': 11984, '12-hour': 11985, 'superstar': 11986, 'recurring': 11987, 'bye': 11988, 'mordechai': 11989, 'tolled': 11990, 'state-of-the-art': 11991, 'expressions': 11992, 'kalashnikov': 11993, 'hitch': 11994, 'paperwork': 11995, 'objectivity': 11996, 'troubling': 11997, 'millimeters': 11998, 'rife': 11999, 'precondition': 12000, 'editions': 12001, 'condolence': 12002, 'rations': 12003, 'vegetable': 12004, 'prizes': 12005, 'chemistry': 12006, 'alfred': 12007, 'banquet': 12008, 'literary': 12009, '1,38,000': 12010, 'hypothetical': 12011, 'nears': 12012, 'restricts': 12013, 'yoh': 12014, 'materialized': 12015, 'legs': 12016, 'timex': 12017, 'seller': 12018, 'cane': 12019, 'napoleonic': 12020, 'strategically': 12021, 'vi': 12022, 'fdi': 12023, 'thermal': 12024, 'micronesia': 12025, 'fsm': 12026, '2023': 12027, 'payouts': 12028, 'tears': 12029, 'attractions': 12030, 'forgive': 12031, 'reservists': 12032, 'imaginary': 12033, 'motorbike': 12034, 'abdul-mahdi': 12035, 'contention': 12036, 'enforced': 12037, 'tramway': 12038, 'feat': 12039, 'monastery': 12040, 'attraction': 12041, 'slaughtering': 12042, 'qasab': 12043, 'gripping': 12044, 'koirala': 12045, 'curtailing': 12046, '143': 12047, 'egypt-gaza': 12048, 'throats': 12049, 'diminishing': 12050, 'nighttime': 12051, 'karshi-khanabad': 12052, 'taste': 12053, 'inmate': 12054, 'caraballo': 12055, 'topic': 12056, '45,000': 12057, 'waterfowl': 12058, 'marat': 12059, 'normalizing': 12060, 'al-fasher': 12061, 'uda': 12062, 'authentic': 12063, 'azhari': 12064, 'kapisa': 12065, 'headscarves': 12066, 'secretaries': 12067, 'senses': 12068, 'strokes': 12069, 'headaches': 12070, 'reflecting': 12071, 'two-tenths': 12072, 'tangible': 12073, 'speculators': 12074, 'pointing': 12075, 'disagrees': 12076, 'perceptions': 12077, 'imposition': 12078, 'ceiling': 12079, 'meddling': 12080, 'stricter': 12081, 'specially': 12082, 'indecent': 12083, 'supervise': 12084, 'snyder': 12085, 'honiara': 12086, 'asserts': 12087, 'shirin': 12088, 'instantly': 12089, 'unlicensed': 12090, 'pisanu': 12091, 'submission': 12092, 'presently': 12093, 'anraat': 12094, '62-year-old': 12095, 'coinciding': 12096, 'oil-exporting': 12097, 'naturally': 12098, 'retroactive': 12099, 'r.': 12100, 'rationing': 12101, 'mls': 12102, '76th': 12103, 'incompetence': 12104, 'taxpayer': 12105, 'anti-fraud': 12106, 'kwasniewski': 12107, 'valdas': 12108, 'alphabet': 12109, 'proponents': 12110, 'manfred': 12111, 'colonization': 12112, 'slovene': 12113, 'acceded': 12114, 'severing': 12115, 'cave-ins': 12116, 'quarrel': 12117, 'traveller': 12118, 'splendid': 12119, 'ellsworth': 12120, '0.3': 12121, 'mwenda': 12122, 'broadly': 12123, 'sterilization': 12124, 'centimeter': 12125, 'overseen': 12126, 'promotes': 12127, 'intermittent': 12128, 'sixto': 12129, '27-year': 12130, 'ninh': 12131, 'zhaoxing': 12132, 'chengdu': 12133, 'clashing': 12134, 'lobbing': 12135, 'shantytown': 12136, 'blackouts': 12137, 'generators': 12138, 'stray': 12139, 'shihab': 12140, '4.9': 12141, 'kilinochchi': 12142, 'freighter': 12143, 'chitral': 12144, 'abdel-al': 12145, 'al-ahbash': 12146, 'clarify': 12147, 'state-sponsored': 12148, 'arbil': 12149, 'asthma': 12150, '46,000': 12151, 'lisbon': 12152, 'elevated': 12153, 'beni': 12154, 'cordon': 12155, '73-year-old': 12156, 'gravely': 12157, 'bullet-riddled': 12158, 'jean-claude': 12159, 'staunch': 12160, 'resurgence': 12161, 'directions': 12162, '55,000': 12163, 'orlandez': 12164, 'gamboa': 12165, 'lords': 12166, 'distorting': 12167, 'harms': 12168, 'dudley': 12169, 'monterrey': 12170, 'ramos': 12171, 'lodged': 12172, '227': 12173, 'rebates': 12174, 'yaalon': 12175, 'farce': 12176, 'outlet': 12177, 'hide-out': 12178, 'retirees': 12179, 'slap': 12180, 'forcefully': 12181, '39-year-old': 12182, 'almallah': 12183, 'dabas': 12184, 'magistrate': 12185, 'montedison': 12186, 'n.v.': 12187, 'pursuant': 12188, 'density': 12189, 'crab': 12190, 'mustard': 12191, 'p.': 12192, 'chidambaram': 12193, 'colonists': 12194, 'chiweshe': 12195, 'consultation': 12196, 'pharaohs': 12197, '35,000': 12198, 'kick-off': 12199, 'maazou': 12200, 'defender': 12201, 'g': 12202, 'pushes': 12203, 'hormone': 12204, 'yadav': 12205, 'transmissions': 12206, 'zacarias': 12207, 'grapes': 12208, 'manganese': 12209, 'beverages': 12210, 'disgruntled': 12211, 'welcoming': 12212, 'component': 12213, 'interruptions': 12214, 'anti-india': 12215, 'deeds': 12216, 'regulation': 12217, 'fitness': 12218, 'infiltrating': 12219, 'life-threatening': 12220, 'ming': 12221, 'toe': 12222, 'rebounds': 12223, 'abhisit': 12224, 'dispersing': 12225, 'deforestation': 12226, 'infidels': 12227, 'endorses': 12228, 'portland': 12229, 'sambadrome': 12230, 'squeeze': 12231, 'collaborating': 12232, 'owen': 12233, 'commits': 12234, 'al-hashimi': 12235, 'vengeance': 12236, 'bikindi': 12237, 'composed': 12238, 'lyrics': 12239, 'baathist': 12240, 'soured': 12241, 'defusing': 12242, 'hence': 12243, 'geo': 12244, 'averted': 12245, 'four-hour': 12246, 'anticipates': 12247, 'marti': 12248, 'anti-castro': 12249, 'spanish-language': 12250, 'internationally-recognized': 12251, 'offend': 12252, '57-member': 12253, 'konan': 12254, 'mian': 12255, 'tactical': 12256, 'tegucigalpa': 12257, 'markey': 12258, '44-year-old': 12259, 'jan.': 12260, 'accessories': 12261, 'cosmetic': 12262, 'technologically': 12263, 'french-speaking': 12264, 'consumed': 12265, 'antonella': 12266, 'liberalizing': 12267, 'reared': 12268, 'unwilling': 12269, 'pashtun': 12270, 'disregard': 12271, '124': 12272, 'sourav': 12273, 'surpassing': 12274, 'countrymen': 12275, 'juma': 12276, 'segment': 12277, 'discomfort': 12278, 's': 12279, 'p': 12280, 'inspiring': 12281, 'emirate': 12282, 'defenders': 12283, 'volunteered': 12284, 'skip': 12285, '48-year-old': 12286, 'exploited': 12287, 'canaria': 12288, 'moderated': 12289, 'arsonists': 12290, 'vaclav': 12291, 'authors': 12292, 'soros': 12293, 'overlooking': 12294, 'cheers': 12295, 'infiltrate': 12296, 'supervised': 12297, 'hard-liners': 12298, 'flare-up': 12299, '3,600': 12300, 'four-year-old': 12301, 'gerry': 12302, 'absurd': 12303, 'bench': 12304, 'javad': 12305, 'haqqani': 12306, 'honorary': 12307, 'positively': 12308, 'condom': 12309, 'salazar': 12310, 'janica': 12311, '0.08': 12312, 'son-in-law': 12313, 'spilling': 12314, 'sanctuaries': 12315, 'life-saving': 12316, 'absentee': 12317, 'gubernatorial': 12318, 'christine': 12319, 'ruin': 12320, 'stretches': 12321, 'explicitly': 12322, 'khurmatu': 12323, 'max': 12324, 'bastion': 12325, 'arteries': 12326, 'tripled': 12327, 'firefights': 12328, 'announcements': 12329, 'al-sheikh': 12330, 'shalit': 12331, 'marinin': 12332, 'totmianina': 12333, 'poiree': 12334, 'miracle': 12335, 'rented': 12336, "l'equipe": 12337, '5,300': 12338, '1500': 12339, 'sheltered': 12340, 'rangin': 12341, 'komuri': 12342, 'tsvangirai': 12343, 'mehmood': 12344, 'exercised': 12345, 'hammer': 12346, 'advantages': 12347, 'lagged': 12348, 'timor-leste': 12349, 'markedly': 12350, 'duchy': 12351, 'fluctuations': 12352, 'export-driven': 12353, 'raven': 12354, 'desired': 12355, 'supposing': 12356, 'washing': 12357, 'silk': 12358, 'financier': 12359, 'bible': 12360, 'stationary': 12361, 'trailing': 12362, 'norad': 12363, 'peterson': 12364, 'uranium-enrichment': 12365, 'preston': 12366, 'niccum': 12367, 'samantha': 12368, 'lantos': 12369, 'swollen': 12370, 'barham': 12371, '02-feb': 12372, 'yankees': 12373, 'sirens': 12374, 'three-kilometer': 12375, 'trek': 12376, '21-year': 12377, 'assassin': 12378, 'mates': 12379, 'mislead': 12380, 'listing': 12381, 'lefevre': 12382, 'staffer': 12383, '17-year-old': 12384, 'teens': 12385, 'kojo': 12386, 'upbeat': 12387, 'centanni': 12388, 'olaf': 12389, 'wiig': 12390, 'caption': 12391, 'instigate': 12392, 'delray': 12393, 'xavier': 12394, 'humanely': 12395, 'jemua': 12396, 'soy': 12397, 'behead': 12398, 'immune': 12399, 'glaxosmithkline': 12400, 'atmar': 12401, 'separatism': 12402, 'hygienic': 12403, 'mughani': 12404, 'lansana': 12405, 'plenty': 12406, 'ancestry': 12407, 'sculptures': 12408, 'andresen': 12409, 'muzadi': 12410, 'bhumibol': 12411, 'adulyadej': 12412, 'der': 12413, 'spiegel': 12414, 'adre': 12415, 'man-made': 12416, 'sediment': 12417, 'tallest': 12418, 'shabab': 12419, 'bisengimina': 12420, 'confess': 12421, 'ensuring': 12422, 'anyway': 12423, 'suing': 12424, 'placement': 12425, 'franc': 12426, 'information-sharing': 12427, 'export-oriented': 12428, '4.2': 12429, 'considerably': 12430, 'accustomed': 12431, 'blackboard': 12432, 'chalk': 12433, 'nato-russia': 12434, 'northward': 12435, 'khushab': 12436, 'cautiously': 12437, 'non-muslim': 12438, 'warri': 12439, 'ijaw': 12440, 'screenings': 12441, 'joyce': 12442, 'mujuru': 12443, 'vacancy': 12444, 'ariane': 12445, 'adrian': 12446, 'yanjin': 12447, '03-jan': 12448, 'echoing': 12449, 'valerie': 12450, 'betty': 12451, 'knowingly': 12452, 'marlon': 12453, 'shelves': 12454, 'tightening': 12455, 'submarine': 12456, 'soe': 12457, 'wanting': 12458, 'chechens': 12459, 'chulanont': 12460, 'abstaining': 12461, 'scorching': 12462, 'ineffectiveness': 12463, 'ages': 12464, 'bilfinger': 12465, 'flushed': 12466, 'duchess': 12467, 'cornwall': 12468, 'tasnim': 12469, 'amarah': 12470, 'relocated': 12471, 'ms-13': 12472, 'lynn': 12473, 'ihab': 12474, 'al-sherif': 12475, 'krishna': 12476, 'yelled': 12477, 'ritchie': 12478, 'presumptive': 12479, 'babar': 12480, 'combatant': 12481, 'biologist': 12482, 'reformists': 12483, 'accuser': 12484, '13-year': 12485, 'concession': 12486, 'pigeon': 12487, 'wine': 12488, '9,00,000': 12489, 'jeweler': 12490, 'tareq': 12491, 'plentiful': 12492, 'reinsurance': 12493, 'amphibious': 12494, '9.2': 12495, 'domain': 12496, 'madagonia': 12497, 'ensued': 12498, 'garnered': 12499, 'malicious': 12500, 'overpowered': 12501, 'mahabad': 12502, 'petra': 12503, 'litani': 12504, 'prevail': 12505, 'izzadeen': 12506, 'disruptive': 12507, 'checking': 12508, 'layers': 12509, '90th': 12510, 'leveled': 12511, 'prop': 12512, 'deterrent': 12513, 'stifle': 12514, '10.5': 12515, 'walker': 12516, 'moody': 12517, 'defaults': 12518, 'sensitivities': 12519, 'high-altitude': 12520, 'south-central': 12521, 'mai': 12522, 'terminals': 12523, 'lifts': 12524, 'tract': 12525, 'gravity': 12526, 'serviceman': 12527, 'macy': 12528, 'characters': 12529, 'guided': 12530, 'endorsing': 12531, 'putumayo': 12532, 'extortion': 12533, '184': 12534, 'all-important': 12535, 'counter-narcotics': 12536, 'us-led': 12537, 'modifications': 12538, 'ahronot': 12539, 'dismay': 12540, 'hasan': 12541, 'ambushing': 12542, 'blantyre': 12543, 'translated': 12544, 'lilongwe': 12545, 'lenient': 12546, 'parish': 12547, 'moderates': 12548, '65th': 12549, 'annexation': 12550, 'non-peaceful': 12551, 'kfar': 12552, 'sailor': 12553, 'villaraigosa': 12554, 'german-born': 12555, 'figaro': 12556, 'unsustainable': 12557, 'qiyue': 12558, 'kaka': 12559, "'ve": 12560, 'ntawukulilyayo': 12561, 'bernardo': 12562, '1st': 12563, 'citadel': 12564, 'sept.': 12565, 'hobbled': 12566, 'overdependence': 12567, 'stand-by': 12568, 'impediment': 12569, 'blueprint': 12570, 'sponsorship': 12571, 'subsidy': 12572, 'necessities': 12573, 'mined': 12574, 'casamance': 12575, 'undeclared': 12576, 'pretense': 12577, 'sorely': 12578, 'mathematician': 12579, 'relevant': 12580, 'ashcroft': 12581, 'restarting': 12582, 'tossed': 12583, 'helm': 12584, 'ceo': 12585, 'genuine': 12586, 'planche': 12587, 'wana': 12588, 'embarrassment': 12589, 'admiration': 12590, 'coping': 12591, 'anti-trust': 12592, 'luxembourg-based': 12593, 'feasible': 12594, 'sulaymaniya': 12595, 'hampshire': 12596, 'regrettable': 12597, 'toxicology': 12598, 'sealing': 12599, 'charcoal': 12600, 'twenty-two': 12601, 'bounced': 12602, 'sixth-seeded': 12603, 'catalonia': 12604, 'classical': 12605, 'french-colombian': 12606, 'languages': 12607, 'tabloid': 12608, 'icc': 12609, 'derived': 12610, 'anti-malaria': 12611, 'ineffective': 12612, 'grabbed': 12613, '275-seat': 12614, 'elevator': 12615, 'eckhard': 12616, 'h.w.': 12617, 'stocking': 12618, 'u.n': 12619, 'timeline': 12620, 'badakhshan': 12621, 'exploring': 12622, 'ion': 12623, 'eduard': 12624, 'atrocity': 12625, 'tapped': 12626, '257': 12627, 'mammoth': 12628, 'verdicts': 12629, 'instigating': 12630, 'schelling': 12631, 'aumann': 12632, 'politically-motivated': 12633, 'sumaidaie': 12634, 'sabbath': 12635, 'noble': 12636, 'kyaw': 12637, 'varies': 12638, 'muhammed': 12639, 'everyday': 12640, 'campus': 12641, 'plc': 12642, 'oct.': 12643, 'demographic': 12644, '7.4': 12645, 'attributable': 12646, 'likewise': 12647, '2016': 12648, 'viking': 12649, 'norwegians': 12650, 'nationalism': 12651, 'rpf': 12652, 'well-to-do': 12653, 'consist': 12654, 'monkeys': 12655, 'danced': 12656, 'spectacle': 12657, 'nuts': 12658, 'laughter': 12659, 'new~york': 12660, 'rooted': 12661, 'unleashed': 12662, 'taji': 12663, 'douchevina': 12664, 'mauresmo': 12665, 'flamingo': 12666, 'milder': 12667, 'virulent': 12668, '10-member': 12669, 'strengthens': 12670, 'ambition': 12671, 'pipes': 12672, 'erupt': 12673, 'opposite': 12674, 'hauling': 12675, 'mac': 12676, 'drama': 12677, 'leash': 12678, 'karrada': 12679, 'greed': 12680, 'hazem': 12681, 'colombians': 12682, 'dinosaurs': 12683, 'bones': 12684, 'tails': 12685, 'jordanians': 12686, 'fielding': 12687, 'rajasthan': 12688, 'no.': 12689, 'guerrero': 12690, 'amritsar': 12691, 'problematic': 12692, 'ploy': 12693, 'desperately': 12694, 'podium': 12695, 'ferrer': 12696, 'atp': 12697, 'robin': 12698, 'pale': 12699, '195': 12700, 'vows': 12701, 'postwar': 12702, 'adversely': 12703, 'kadhimiya': 12704, 'aquifer': 12705, 'seismic': 12706, 'coretta': 12707, 'ovarian': 12708, 'bunkers': 12709, 'jessen-petersen': 12710, 'kostunica': 12711, 'hamma': 12712, 'dsm': 12713, '4.1': 12714, 'barthelemy': 12715, 'attracts': 12716, 'abundant': 12717, 'prix': 12718, 'frelimo': 12719, 'delicate': 12720, 'armando': 12721, 'questionable': 12722, '7.2': 12723, 'remittance': 12724, 'domenech': 12725, 'respecting': 12726, 'prevalent': 12727, 'manifesto': 12728, 'energy-efficient': 12729, 'desperation': 12730, 'flexible': 12731, 'reprisals': 12732, 'dc': 12733, 'cavaco': 12734, 'center-right': 12735, 'cap': 12736, 'protectionism': 12737, 'abdel-rahman': 12738, 'landmarks': 12739, 'socialists': 12740, 'hygiene': 12741, 'premature': 12742, '25-nation': 12743, 'african-americans': 12744, '67,000': 12745, 'chennai': 12746, 'rallying': 12747, 'collapsing': 12748, 'drainage': 12749, 'kenteris': 12750, 'thanou': 12751, '200-meter': 12752, 'systemic': 12753, 'arap': 12754, 'gomez': 12755, 'gallbladder': 12756, 'shinui': 12757, 'unimaginable': 12758, 'powered': 12759, 'errant': 12760, 'busan': 12761, 'sandy': 12762, 'swimming': 12763, 'motorist': 12764, 'greenback': 12765, 'priced': 12766, 'al-qawasmi': 12767, 'lure': 12768, 'infiltrations': 12769, 'incidence': 12770, '450-seat': 12771, 'liable': 12772, 'tentatively': 12773, 'explorer': 12774, 'levey': 12775, 'cornerstone': 12776, 'fallout': 12777, 'umm': 12778, 'u.n.-african': 12779, 'kellenberger': 12780, 'perjury': 12781, 'contradictory': 12782, '29,000': 12783, '3,30,000': 12784, 'atta': 12785, '1.65': 12786, 'onset': 12787, 'dried': 12788, 'ec': 12789, 'default': 12790, 'emu': 12791, '1885': 12792, 'bloated': 12793, 'mongolian': 12794, 'dug': 12795, 'sown': 12796, 'barber': 12797, 'simeus': 12798, 'harmless': 12799, 'adjourn': 12800, 'anhui': 12801, 'cowardly': 12802, 'curling': 12803, '07-feb': 12804, '10-nation': 12805, 'arturo': 12806, 'setbacks': 12807, 'grids': 12808, 'holbrooke': 12809, 'relocation': 12810, 'quebec': 12811, 'dominion': 12812, 'referendums': 12813, 'wrap': 12814, 'liaison': 12815, 'tragic': 12816, 'parcel': 12817, 'stranding': 12818, 'tarongoy': 12819, 'jeff': 12820, 'gdansk': 12821, 'mansour': 12822, 'destabilized': 12823, 'adventure': 12824, 'hilltop': 12825, 'expecting': 12826, 'stagnant': 12827, '16.8': 12828, 'kenichiro': 12829, 'sasae': 12830, 'sacrificed': 12831, 'detecting': 12832, '145': 12833, '248': 12834, 'incurred': 12835, 'plummeted': 12836, 'kajaki': 12837, 'harvests': 12838, 'correspondents': 12839, '228': 12840, 'fist': 12841, 'ahlu': 12842, 'sunna': 12843, 'tribesman': 12844, '35-nation': 12845, 'oshkosh': 12846, 'chassis': 12847, 'arab-israeli': 12848, 'july-august': 12849, 'mid-april': 12850, 'remarkably': 12851, '7.1': 12852, 'bite': 12853, 'military-led': 12854, 'sirte': 12855, 'cologne': 12856, 'mausoleum': 12857, 'ghorbanpour': 12858, '1,60,000': 12859, 'masood': 12860, 'visible': 12861, 'reinforce': 12862, 'sucumbios': 12863, 'orinoco': 12864, '16.7': 12865, '3.1': 12866, 'goodluck': 12867, 'notify': 12868, 'notification': 12869, 'merida': 12870, 'unpunished': 12871, 'precursor': 12872, '169': 12873, 'hafun': 12874, 'sand': 12875, 'granada': 12876, 'rider': 12877, 'three-time': 12878, 'danube': 12879, 'populist': 12880, 'roar': 12881, 'evaluating': 12882, '1795': 12883, 'partitioned': 12884, 'unlawfully': 12885, 'tolerant': 12886, 'siddiqui': 12887, 'ubaydi': 12888, 'ho': 12889, 'hayat': 12890, '80th': 12891, 'culminate': 12892, 'mid-october': 12893, 'emptied': 12894, 'jun': 12895, 'chaman': 12896, 'altitude': 12897, 'extort': 12898, 'daschle': 12899, 'majlis-e-amal': 12900, 'npt': 12901, '46-year-old': 12902, 'swung': 12903, 'insurer': 12904, 'defined': 12905, 'inefficiencies': 12906, '1902': 12907, 'administer': 12908, 'bolted': 12909, 'lisa': 12910, 'examples': 12911, 'inland': 12912, 'juveniles': 12913, 'headscarf': 12914, 'nitrogen': 12915, 'ridiculed': 12916, 'fig-tree': 12917, 'yasuo': 12918, 'radicalism': 12919, 'milf': 12920, 'pure': 12921, 'airs': 12922, 'spaceship': 12923, 'geldof': 12924, 'grim': 12925, 'refers': 12926, 'tammy': 12927, 'wizard': 12928, 'restructured': 12929, 'rosales': 12930, 'pennetta': 12931, '01-jun': 12932, 'solis': 12933, 'compassionate': 12934, 'floral': 12935, 'vigilance': 12936, 'ruptured': 12937, '3.9': 12938, 'repayment': 12939, 'astana': 12940, 'serhiy': 12941, 'holovaty': 12942, 'suitable': 12943, 'datta': 12944, 'bertie': 12945, 'ahern': 12946, 'belfast': 12947, 'overturning': 12948, 'formality': 12949, 'solemn': 12950, 'yasir': 12951, 'adil': 12952, 'muslim-inhabited': 12953, 'metropolitan': 12954, 'morgenthau': 12955, '15.5': 12956, 'glorifies': 12957, 'mishandling': 12958, 'shenzhen': 12959, 'textbooks': 12960, 'uruguayan': 12961, 'gallup': 12962, 'acceptance': 12963, 'kashmiris': 12964, 'guise': 12965, 'griffin': 12966, 'netted': 12967, 'u.n.-chartered': 12968, 'now-defunct': 12969, 'nations-run': 12970, 'paracha': 12971, 'slip': 12972, 'plotted': 12973, 'al-mashhadani': 12974, 'irbil': 12975, 'disagreed': 12976, 'aspects': 12977, 'dublin': 12978, 'flushing': 12979, 'ernest': 12980, 'serena': 12981, 'combine': 12982, 'yachts': 12983, 'backpack': 12984, 'imprisoning': 12985, 'aquatic': 12986, 'description': 12987, 'financial-services': 12988, 'marketed': 12989, 'dipped': 12990, 'benefiting': 12991, 'civic': 12992, 'sculptor': 12993, 'juno': 12994, 'sum': 12995, 'certainly': 12996, 'one-seat': 12997, 'hollow': 12998, 'ai': 12999, 'documented': 13000, 'screened': 13001, 'projectiles': 13002, 'alzheimer': 13003, 'sellers': 13004, 'skidded': 13005, 'drug-resistant': 13006, 'line-item': 13007, 'attitude': 13008, 'telegram': 13009, 'veal': 13010, 'uncensored': 13011, 'rank-and-file': 13012, 'admitting': 13013, 'collaborated': 13014, 'mouthpiece': 13015, 'vinci': 13016, 'disclaimer': 13017, 'maier': 13018, 'fritz': 13019, 'pipe': 13020, 'forensics': 13021, 'harmonious': 13022, 'forbidding': 13023, 'overran': 13024, 'filmyard': 13025, 'deposit': 13026, 'pulp': 13027, 'soviet-backed': 13028, 'pills': 13029, 'anti-dumping': 13030, 'petter': 13031, 'solberg': 13032, 'auc': 13033, 'plagues': 13034, 'lahiya': 13035, 'n': 13036, 'gurgenidze': 13037, 'barricade': 13038, '75-year-old': 13039, 'jabaliya': 13040, 'veneman': 13041, 'cimoszewicz': 13042, 'trucking': 13043, 'littered': 13044, 'vigil': 13045, 'ruiz': 13046, 'anti-semitic': 13047, 'internationally-backed': 13048, 'premeditated': 13049, 'nyan': 13050, 'marty': 13051, 'krill': 13052, 'nation-state': 13053, 'administrations': 13054, 'guadeloupe': 13055, 'reunion': 13056, 'three-fourths': 13057, 'beware': 13058, 'mossad': 13059, 'weather-related': 13060, 'quarter-century': 13061, 'tamara': 13062, '2,300': 13063, 'gansu': 13064, 'quami': 13065, 'insults': 13066, 'sassou-nguesso': 13067, 'palma': 13068, 'british-born': 13069, 'co-founded': 13070, 'vines': 13071, 'airmen': 13072, 'emergencies': 13073, 'rotate': 13074, 'kaibyshev': 13075, 'lauderdale': 13076, 'reservations': 13077, 'resembles': 13078, 'avenue': 13079, 'underneath': 13080, 'neumann': 13081, 'phoned': 13082, 'portrays': 13083, 'showcasing': 13084, 'showcase': 13085, 'mort': 13086, 'line-up': 13087, 'earliest': 13088, 'crypt': 13089, 'harshly': 13090, 'four-party': 13091, 'pulwama': 13092, 'cereal': 13093, 'shortcomings': 13094, 'phenomenon': 13095, 'dynastic': 13096, 'cede': 13097, 'orientation': 13098, 'kooyong': 13099, 'tentative': 13100, 'izvestia': 13101, 'prolong': 13102, 'ninevah': 13103, 'wipe': 13104, 'mutation': 13105, 'interceptors': 13106, 'kidd': 13107, 'vientiane': 13108, 'ablaze': 13109, 'complicating': 13110, 'al-ghad': 13111, 'elimination': 13112, 'adb': 13113, 'tanzanian': 13114, 'dheere': 13115, 'poisons': 13116, 'diluted': 13117, 'underscored': 13118, 'resolutely': 13119, 'exploratory': 13120, 'dreamliner': 13121, 'nippon': 13122, 'hamdania': 13123, 'orbiter': 13124, 'harassing': 13125, 'petro': 13126, 'succumbed': 13127, 'abundance': 13128, 'oakland': 13129, 'preference': 13130, 'boers': 13131, 'apartheid-era': 13132, 'locomotives': 13133, 'constitutionally': 13134, 'intensively': 13135, 'single-party': 13136, 'standstill': 13137, 'selfish': 13138, 'reign': 13139, 'swooped': 13140, 'deserves': 13141, 'afflicted': 13142, 'samuels': 13143, 'crease': 13144, 'steyn': 13145, 'english-language': 13146, 'towed': 13147, 'crewmembers': 13148, 'unconditionally': 13149, 'u.s.-mediated': 13150, 'retake': 13151, 'al-qaim': 13152, 'experiences': 13153, 'interrupt': 13154, 'jailing': 13155, 'franks': 13156, 'tireless': 13157, 'adolf': 13158, 'fixing': 13159, 'valenzuela': 13160, 'franz': 13161, 'dorfmeister': 13162, 'versions': 13163, 'bribe': 13164, 'downplaying': 13165, 'tornado': 13166, 'haley': 13167, 'barbour': 13168, 'counties': 13169, 'malindi': 13170, 'vermont': 13171, 'kuo': 13172, 'bergersen': 13173, 'dongzhou': 13174, 'savannah': 13175, 'georgetown': 13176, 'churns': 13177, 'albert': 13178, 'vegas': 13179, 'duff': 13180, 'realizes': 13181, 'mindful': 13182, 'sounds': 13183, 'anarchy': 13184, 'tasks': 13185, 'dextre': 13186, 'closes': 13187, 'havoc': 13188, 'reining': 13189, 'manigat': 13190, 'savimbi': 13191, 'possesses': 13192, 'revival': 13193, '1942': 13194, 'razak': 13195, 'bamako': 13196, 'jem': 13197, 'khalil': 13198, 'notices': 13199, 'warren': 13200, 'buffett': 13201, 'gem': 13202, 'hsieh': 13203, 'andras': 13204, 'batiz': 13205, 'rattled': 13206, 'bingol': 13207, 'eagles': 13208, 'nadia': 13209, 'venues': 13210, 'kokang': 13211, 'levin': 13212, 'sans': 13213, 'frontieres': 13214, 'labado': 13215, 'influx': 13216, 'insulza': 13217, 'recessed': 13218, 'ngabo': 13219, 'hasina': 13220, 'inhuman': 13221, 'professors': 13222, 'octopus': 13223, 'creature': 13224, 'counternarcotics': 13225, 'expeditions': 13226, 'toppling': 13227, 'firecrackers': 13228, '30-year': 13229, 'herzegovina': 13230, 'jovic': 13231, 'vieira': 13232, 'pistols': 13233, 'luggage': 13234, 'trafficker': 13235, 'relax': 13236, 'patagonian': 13237, 'toothfish': 13238, 'overflights': 13239, 'kites': 13240, 'vanquished': 13241, 'cock': 13242, 'telecom': 13243, 'kewell': 13244, 'aussies': 13245, 'inconsistent': 13246, 'merk': 13247, 'stuff': 13248, 'zedong': 13249, 'empress': 13250, 'plowed': 13251, 'prostate': 13252, 'bright': 13253, 'dai': 13254, 'chopan': 13255, 'exaggerated': 13256, 'depicts': 13257, 'jesse': 13258, 'racist': 13259, 'ruth': 13260, 'turk': 13261, 'scooter': 13262, 'poaching': 13263, 'bunia': 13264, 'bralo': 13265, 'top-level': 13266, 'najim': 13267, 'mironov': 13268, 'cppcc': 13269, 'useful': 13270, 'retaliatory': 13271, 'categorically': 13272, 'gulbuddin': 13273, 'commandeered': 13274, 'schultz': 13275, 'demonstrates': 13276, 'endangers': 13277, 'pavarotti': 13278, 'upstream': 13279, 'adaado': 13280, 'jiangxi': 13281, '5.9': 13282, 'costume': 13283, 'convene': 13284, 'duarte': 13285, '223': 13286, 'newmont': 13287, 'focal': 13288, 'candy': 13289, '51-year-old': 13290, 'savin': 13291, 'cent': 13292, 'surpluses': 13293, 'un-sponsored': 13294, 'amounted': 13295, 'falklands': 13296, 'wayside': 13297, 'shame': 13298, 'foxes': 13299, 'tallies': 13300, 'entries': 13301, 'greg': 13302, 'schulte': 13303, 'co-chairman': 13304, 'pinault': 13305, 'ugly': 13306, 'dismissing': 13307, 'statute': 13308, 'breeding': 13309, '337': 13310, 'yousef': 13311, 'phrase': 13312, 'luo': 13313, 'first-time': 13314, 'pitch': 13315, 'mutates': 13316, 'extravagant': 13317, 'civilization': 13318, 'hague-based': 13319, 'cholesterol': 13320, 'mok': 13321, 'ad': 13322, 'lipsky': 13323, 'pumps': 13324, 'voluntary': 13325, '8.6': 13326, 'sinking': 13327, 'undertaken': 13328, 'previously-owned': 13329, 'acquisitions': 13330, 'dharmeratnam': 13331, 'pro-rebel': 13332, 'tamilnet': 13333, 'columnist': 13334, 'staying': 13335, 'newly-built': 13336, 'qin': 13337, '1,250': 13338, 'detection': 13339, 'hiv-aids': 13340, 'nonprofit': 13341, 'speculated': 13342, 'fights': 13343, 'hooliganism': 13344, 'damn': 13345, 'robbers': 13346, 'gagged': 13347, 'classroom': 13348, 'villager': 13349, "b'tselem": 13350, 'rabat': 13351, '375': 13352, 'communist-ruled': 13353, 'algerian-based': 13354, 'connie': 13355, 'eradicated': 13356, 'breakdown': 13357, 'bab': 13358, 'foxx': 13359, 'hudson': 13360, 'unpredictable': 13361, 'ashdown': 13362, 'scaling': 13363, 'cropper': 13364, 'automobiles': 13365, 'homage': 13366, 'evaluation': 13367, 'edict': 13368, 'recipient': 13369, 'baton': 13370, 'thad': 13371, 'inducted': 13372, 'inductees': 13373, 'competent': 13374, 'side-by-side': 13375, 'mitrovica': 13376, 'seceded': 13377, 'disadvantaged': 13378, 'celso': 13379, 'manas': 13380, 'bunch': 13381, 'muentefering': 13382, 'smiled': 13383, '4.3': 13384, 'presley': 13385, 'memphis': 13386, 'graceland': 13387, 'shariat': 13388, 'hymns': 13389, 'colombant': 13390, 'bowled': 13391, 'wong': 13392, 'reforming': 13393, 'kazemi': 13394, 'qomi': 13395, 'naji': 13396, 'eden': 13397, 'wicket': 13398, 'five-match': 13399, 'kutesa': 13400, 'costing': 13401, 'golovin': 13402, 'marta': 13403, 'marion': 13404, '225': 13405, 'brink': 13406, 'mile': 13407, 'delegate': 13408, 'straight-set': 13409, 'gilles': 13410, 'srichaphan': 13411, 'mid-august': 13412, 'extinction': 13413, 'rituals': 13414, 'biblical': 13415, 'crucifixion': 13416, 'liberians': 13417, 'unchallenged': 13418, 'bolstering': 13419, 'enlisted': 13420, 'fabricated': 13421, '05-jul': 13422, 'inter-korean': 13423, 'fictional': 13424, 'clearance': 13425, 'rms': 13426, '8.9': 13427, 'mongol': 13428, 'russo-japanese': 13429, '1904': 13430, 'shifted': 13431, 'legitimacy': 13432, 'uninhabitable': 13433, 'topped': 13434, 'passion': 13435, 'cottage': 13436, 'durbin': 13437, 'machar': 13438, 'rezaei': 13439, 'seated': 13440, 'mostafa': 13441, 'stun': 13442, 'constantinople': 13443, 'daunting': 13444, 'clause': 13445, 'velasco': 13446, 'bot': 13447, 'zahi': 13448, 'saqqara': 13449, 'marriott': 13450, 'milk': 13451, 'russert': 13452, 'undefeated': 13453, 'elmar': 13454, 'oppressive': 13455, 'aliyev': 13456, 'inherit': 13457, 'relaying': 13458, 'cassini': 13459, 'rooftops': 13460, 'feasibility': 13461, 'trick': 13462, 'engagements': 13463, 'overpass': 13464, 'woods': 13465, 'longevity': 13466, 'breakthroughs': 13467, 'long-awaited': 13468, 'ghor': 13469, 'madrassa': 13470, 'foxy': 13471, 'rapper': 13472, 'schoolgirl': 13473, 'fast-growing': 13474, 'indianapolis': 13475, 'writes': 13476, 'chiyangwa': 13477, 'kenny': 13478, 'disapproval': 13479, 'judged': 13480, 'principe': 13481, 'incredible': 13482, 'manipulated': 13483, 'pryor': 13484, 'florence': 13485, 'conducts': 13486, 'extraditing': 13487, 'tolo': 13488, '360': 13489, 'replenish': 13490, 'ferencevych': 13491, 'war-shattered': 13492, 'definitive': 13493, 'imran': 13494, '30-day': 13495, 'kyiv': 13496, 'phyu': 13497, 'szmajdzinski': 13498, 'zionism': 13499, 'ruhollah': 13500, 'khomeini': 13501, 'stanisic': 13502, 'hide-outs': 13503, 'crafting': 13504, 'blackmail': 13505, 'cables': 13506, 'workshop': 13507, 'blasphemy': 13508, 'defaming': 13509, 'punishable': 13510, 'saarc': 13511, '142': 13512, 'confusion': 13513, 'simkins': 13514, 'barretto': 13515, 'incomprehensible': 13516, 'spit': 13517, 'anticipate': 13518, 'ahtisaari': 13519, 'ging': 13520, 'cultivated': 13521, 'rivalries': 13522, 'cave': 13523, 'du': 13524, 'lambert': 13525, 'leonel': 13526, 'helen': 13527, 'weisfield': 13528, 'qalqilya': 13529, 'gerald': 13530, 'fluctuates': 13531, '6.3': 13532, 'payroll': 13533, 'mixture': 13534, 'vendors': 13535, 'interrogating': 13536, 'lana': 13537, 'aces': 13538, 'preside': 13539, 'perkins': 13540, 'hajj': 13541, 'milton': 13542, 'obote': 13543, 'murat': 13544, 'descended': 13545, 'anti-globalization': 13546, 'reunify': 13547, 'schumer': 13548, 'jilin': 13549, '8,40,000': 13550, '44,000': 13551, '117': 13552, 'near-simultaneous': 13553, 'compiled': 13554, '192': 13555, 'rahul': 13556, 'anchored': 13557, '136': 13558, 'cermak': 13559, 'mladen': 13560, 'markac': 13561, 'floated': 13562, 'ericsson': 13563, 'leaflets': 13564, 'adversaries': 13565, 'blades': 13566, 'starring': 13567, 'hostage-takers': 13568, 'mutating': 13569, 'sadio': 13570, 'grappling': 13571, '1,900': 13572, 'equity': 13573, 'defamation': 13574, 'clouds': 13575, 'nickname': 13576, 'hits': 13577, 'uncover': 13578, 'portrait': 13579, 'handsome': 13580, 'apprehended': 13581, 'kujundzic': 13582, 'al-ahram': 13583, 'kazimierz': 13584, 'marcinkiewicz': 13585, 'rok': 13586, 'demilitarized': 13587, '1493': 13588, '1648': 13589, 'assumption': 13590, 'importation': 13591, 'plantations': 13592, 'persisted': 13593, 'anticorruption': 13594, 'abrupt': 13595, 'reefs': 13596, 'beja': 13597, 'commands': 13598, '173': 13599, 'elsa': 13600, 'shortwave': 13601, '159': 13602, 'valleys': 13603, 'jawad': 13604, 'record-breaking': 13605, 'denktash': 13606, 'klebnikov': 13607, 'hanover': 13608, 'bomb-grade': 13609, 'autobiography': 13610, 'norms': 13611, 'overuse': 13612, 'isaias': 13613, 'lublin': 13614, 'synagogue': 13615, 'u.s.-mexican': 13616, 'legislatures': 13617, 'military-backed': 13618, 'severance': 13619, 'defeating': 13620, 'governmental': 13621, 'geographical': 13622, 'helpless': 13623, 'belts': 13624, 'fan': 13625, 'ozawa': 13626, 'brett': 13627, 'prabhakaran': 13628, 'wyoming': 13629, 'renowned': 13630, 'chahine': 13631, 'imperialism': 13632, 'solaiman': 13633, 'cnooc': 13634, 'forceful': 13635, 'pollutants': 13636, 'bearing': 13637, 'sepat': 13638, 'weathered': 13639, 'vote-rigging': 13640, 'voyage': 13641, 'mariam': 13642, 'henri': 13643, 'inventory': 13644, 'mack': 13645, 'indian-born': 13646, 'sabeel': 13647, 'kafeel': 13648, 'haneef': 13649, 'soliciting': 13650, 'destined': 13651, 'informer': 13652, 'smashing': 13653, 'cohen': 13654, 'hazards': 13655, 'andreev': 13656, 'appease': 13657, 'jean-marie': 13658, 'coleco': 13659, 'anjouan': 13660, 'azali': 13661, 'de-facto': 13662, 'moreover': 13663, 'bursting': 13664, '1861': 13665, 'benito': 13666, 'mussolini': 13667, 'fascist': 13668, 'azerbaijani': 13669, 'encamped': 13670, 'impediments': 13671, 'chodo': 13672, 'ship-to-ship': 13673, 'maronite': 13674, 'blog': 13675, 'ilo': 13676, '24th': 13677, 'tomorrow': 13678, 'memorials': 13679, 'bakara': 13680, 'yuschenko': 13681, 'nasiriyah': 13682, 'ansa': 13683, 'medium-sized': 13684, 'bellerive': 13685, 'spreads': 13686, 'analyze': 13687, 'vx': 13688, 'emergence': 13689, 'plenary': 13690, 'kirk': 13691, 'tops': 13692, 'luncheon': 13693, 'peters': 13694, 'guber': 13695, '1840': 13696, 'eligibility': 13697, 'unreliable': 13698, 'kiel': 13699, 'hydrographic': 13700, 'latitude': 13701, 'automated': 13702, 'ought': 13703, 'drunken': 13704, 'pig': 13705, 'quoting': 13706, 'drug-trafficking': 13707, '2,20,000': 13708, 'chorus': 13709, 'filtering': 13710, 'regulating': 13711, 'worded': 13712, 'yields': 13713, 'choe': 13714, 'best-known': 13715, 'peng': 13716, 'anna-lena': 13717, 'stanley': 13718, 'zinedine': 13719, 'm.': 13720, 'saves': 13721, 'tirin': 13722, 'farraj': 13723, 'al-libbi': 13724, 'insein': 13725, 'pad': 13726, 'equator': 13727, 'convey': 13728, 'asserted': 13729, 'stalls': 13730, 'paralympic': 13731, 'shortened': 13732, 'chongqing': 13733, 'wheels': 13734, 'double-decker': 13735, 'prototype': 13736, 'boosters': 13737, 'britons': 13738, '70s': 13739, 'starred': 13740, 'rhymes': 13741, '38-year': 13742, 'kanyarukiga': 13743, 'xerox': 13744, 'crum': 13745, 'forster': 13746, 'comparison': 13747, 'transshipment': 13748, 'hooper': 13749, '254': 13750, 'southernmost': 13751, 'ensured': 13752, 'obstructed': 13753, 'huntsmen': 13754, 'revolutionaries': 13755, 'unveil': 13756, 'stick': 13757, 'happens': 13758, 'removes': 13759, 'tribunals': 13760, '540': 13761, 'dunem': 13762, 'discharges': 13763, 'cheeks': 13764, '16-day': 13765, 'colder': 13766, 'frail': 13767, 'directive': 13768, 'aharonot': 13769, 'chileans': 13770, 'murungi': 13771, 'michoacan': 13772, 'al-shihri': 13773, 'umar': 13774, 'worn': 13775, 'commuted': 13776, 'nottingham': 13777, 'raich': 13778, '106': 13779, 'soonthorn': 13780, 'botched': 13781, 'state-funded': 13782, 'socio-economic': 13783, 'bowed': 13784, 'fonseka': 13785, 'expedition': 13786, '1914': 13787, 'wellington': 13788, 'obscenity': 13789, 'dilute': 13790, 'administers': 13791, 'year-and-a-half': 13792, 'controllers': 13793, 'ordaz': 13794, 'heroes': 13795, 'obstruction': 13796, 'sprinter': 13797, 'azima': 13798, 'shakeri': 13799, 'samawa': 13800, '139': 13801, 'guerilla': 13802, 'subordinates': 13803, 'scarcely': 13804, 'tremors': 13805, 'destructive': 13806, 'regev': 13807, 'landfill': 13808, 'interactive': 13809, 'filter': 13810, 'jockeys': 13811, 'horne': 13812, 'georges': 13813, 'dominates': 13814, 'yams': 13815, 'bind': 13816, 'occupies': 13817, 'tramp': 13818, 'carved': 13819, 'verification': 13820, 'mengistu': 13821, '156': 13822, 'archives': 13823, 'preview': 13824, 'surface-to-surface': 13825, 'trader': 13826, 'deporting': 13827, 'chacon': 13828, 'akerson': 13829, 'turnaround': 13830, 'shoddy': 13831, 'bel-air': 13832, 'intersection': 13833, 'broadband': 13834, 'continuation': 13835, 'sinbad': 13836, 'wikipedia': 13837, '193': 13838, 'dinners': 13839, 'stone-throwing': 13840, 'supervisor': 13841, 'outposts': 13842, 'intact': 13843, 'bread': 13844, 'taino': 13845, 'dahlan': 13846, 'cafta-dr': 13847, 'mahmud': 13848, 'educated': 13849, 'excluding': 13850, 'limitations': 13851, 'crunch': 13852, 'monkey': 13853, 'saluted': 13854, '154': 13855, 'first-hand': 13856, 'houghton': 13857, 'soulja': 13858, 'sidelined': 13859, 'recorders': 13860, 'bagosora': 13861, 'immunized': 13862, 'routed': 13863, 'afghan-pakistani': 13864, 'sparrows': 13865, 'five-point': 13866, 'leeward': 13867, 'yoadimnadji': 13868, 'dell': 13869, 'veerman': 13870, 'emphasizing': 13871, 'government-held': 13872, 'jebel': 13873, 'marra': 13874, 'farooq': 13875, 'dense': 13876, 'scaled': 13877, 'razor-wire': 13878, '93,000': 13879, 'novel': 13880, 'humane': 13881, 'blagojevic': 13882, 'logical': 13883, 'betraying': 13884, 'jafarzadeh': 13885, 'rakhine': 13886, 'median': 13887, 'jassar': 13888, 'speedskater': 13889, 'endure': 13890, 'teaches': 13891, 'reasonably': 13892, 'drug-smuggling': 13893, 'kopra': 13894, 'spanish-american': 13895, 'guaranteeing': 13896, 'unionist': 13897, 'fodder': 13898, 'confer': 13899, 'ruggiero': 13900, 'cannons': 13901, 'jane': 13902, 'fonda': 13903, 'scolded': 13904, 'massey': 13905, 'charleston': 13906, 'ansip': 13907, 'explaining': 13908, 'yugansk': 13909, 'asses': 13910, 'on-line': 13911, 'blanketed': 13912, '6.5': 13913, 'felix': 13914, 'rechnitzer': 13915, 'gamma': 13916, 'elgon': 13917, 'exempt': 13918, 'avigdor': 13919, 'shepherds': 13920, 'yayi': 13921, 'commercially': 13922, 'emulate': 13923, 'extract': 13924, 'heed': 13925, 'non-emergency': 13926, 'recommends': 13927, 'romano': 13928, 'reinstatement': 13929, 'todovic': 13930, 'alexandre': 13931, 'wikileaks': 13932, 'al-arian': 13933, 'reschedule': 13934, 'african-american': 13935, 'depth': 13936, 'vladislav': 13937, 'vicious': 13938, 'pollsters': 13939, '2013': 13940, 'forge': 13941, 'disappear': 13942, 'reprocessing': 13943, 'invading': 13944, 'compounded': 13945, 'siraj': 13946, 'eldawoody': 13947, 'non-stop': 13948, 'observes': 13949, 'mas': 13950, 'wit': 13951, 'alias': 13952, 'initials': 13953, 'lao': 13954, 'filmmakers': 13955, 'suli': 13956, 'small-arms': 13957, 'naqoura': 13958, 'swearing-in': 13959, 'nooyi': 13960, 'closely-watched': 13961, 'cholmondeley': 13962, 'sympathetic': 13963, 'neighbour': 13964, 'farmyard': 13965, 'seibu': 13966, 'sogo': 13967, 'diary': 13968, 'suspecting': 13969, 'abating': 13970, 'blacklisted': 13971, 'military-run': 13972, 'super-combined': 13973, 'enlargement': 13974, 'sars': 13975, 'pre-emptive': 13976, 'trapping': 13977, 'billboards': 13978, 'blonska': 13979, 'shaways': 13980, 'kareem': 13981, 'hamed': 13982, '373': 13983, 'erode': 13984, 'raouf': 13985, 'cud': 13986, 'hebrides': 13987, 'cultures': 13988, 'private-sector': 13989, 'didier': 13990, 'bedside': 13991, 'misconduct': 13992, 'detachment': 13993, 'hobart': 13994, 'introduces': 13995, 'reprisal': 13996, 'skirmishes': 13997, 'sub-commander': 13998, 'gwyneth': 13999, 'veto-wielding': 14000, 'runners': 14001, 'portray': 14002, 'retreated': 14003, 'val': 14004, 'lukic': 14005, 'sling': 14006, 'lament': 14007, 'cries': 14008, 'orator': 14009, 'unblotted': 14010, 'escutcheon': 14011, 'haq': 14012, 'foy': 14013, 'misusing': 14014, 'sibal': 14015, 'campuses': 14016, 'dharamsala': 14017, 'self-proclaimed': 14018, 'martic': 14019, 'duke': 14020, 'detainment': 14021, 'toss': 14022, 'hoggard': 14023, 'panesar': 14024, 'baba': 14025, 'banditry': 14026, 'dictate': 14027, 'abqaiq': 14028, 'detector': 14029, 'al-mahdi': 14030, 'arbor': 14031, 'al-bared': 14032, 'zaidi': 14033, 'murderers': 14034, 'empires': 14035, 'manipulation': 14036, 'contra': 14037, 'rope': 14038, 'pastor': 14039, 'divorced': 14040, 'pitting': 14041, 'anti-retroviral': 14042, 'tac': 14043, 'nutrients': 14044, '162': 14045, 'dimona': 14046, 'nasdaq': 14047, 'cac-40': 14048, 'dax': 14049, 'ounce': 14050, 'tomohito': 14051, 'counterfeit': 14052, 'affordable': 14053, 'baki': 14054, 'dunham': 14055, 'alexis': 14056, 'haro-rodriguez': 14057, 'overheating': 14058, 'predictions': 14059, 'cutbacks': 14060, 'concluding': 14061, 'saadi': 14062, 'yedioth': 14063, 'ahronoth': 14064, 'privatized': 14065, 'fundamentals': 14066, 'lows': 14067, 'mid-may': 14068, 'plummeting': 14069, 'mortally': 14070, 'avalanches': 14071, 'consisting': 14072, '42-year-old': 14073, 'misguided': 14074, 'gao': 14075, 'al-shara': 14076, 'a.k.': 14077, '20s': 14078, 'dream': 14079, 'pretending': 14080, 'variation': 14081, 'flood-prone': 14082, 'hawks': 14083, 'outlaw': 14084, 'explorers': 14085, 'capitalism': 14086, 'rejoined': 14087, 'reaffirm': 14088, 'schuessel': 14089, 'sample': 14090, 'aisha': 14091, 'assailed': 14092, 'taya': 14093, 'leigh': 14094, 'liang': 14095, 'identifies': 14096, 'shadi': 14097, 'unaccounted': 14098, 'israeli-arab': 14099, 'raffle': 14100, 'distinct': 14101, 'estrada': 14102, 'premiums': 14103, 'exposing': 14104, 'tareen': 14105, 'inserted': 14106, 'abbott': 14107, 'xii': 14108, 'platon': 14109, 'alexeyev': 14110, 'kappes': 14111, 'quarterback': 14112, 'repaired': 14113, 'thu': 14114, 'mende': 14115, 'merchandise': 14116, 'cozumel': 14117, 'guttenberg': 14118, 'schneiderhan': 14119, 'cheapest': 14120, 'atolls': 14121, 'altered': 14122, 'origins': 14123, 'gymnast': 14124, 'backgrounds': 14125, 'take-over': 14126, 'tshabalala': 14127, 'curator': 14128, 'shula': 14129, 'laughing': 14130, 'believing': 14131, 'chambliss': 14132, 'rohrabacher': 14133, 'lions': 14134, 'touchdowns': 14135, 'ipad': 14136, 'tablet': 14137, 'seaside': 14138, '3d': 14139, 'fronts': 14140, 'riddled': 14141, 'waive': 14142, 'shrines': 14143, 'storming': 14144, 'saudis': 14145, 'superjumbo': 14146, 'kremlin-backed': 14147, 'alu': 14148, 'sadeq': 14149, 'horrified': 14150, 'kramer': 14151, 'abizaid': 14152, 'booster': 14153, 'tsunami-ravaged': 14154, 'rowing': 14155, 'adolfo': 14156, '1,000-meter': 14157, '54,000': 14158, 'smyr': 14159, 'confidentiality': 14160, 'skulls': 14161, 'oceanic': 14162, 'turtles': 14163, 'equatoria': 14164, 'facilitated': 14165, 'banana': 14166, 'al-uloum': 14167, '23,000': 14168, 'berman': 14169, 'aka': 14170, 'acapulco': 14171, 'rood': 14172, 'highest-level': 14173, 'akayeva': 14174, 'reactions': 14175, 'approves': 14176, 'genoino': 14177, 'nestle': 14178, 'schoch': 14179, 'bruhin': 14180, 'baptist': 14181, 'vuitton': 14182, 'canterbury': 14183, 'archive': 14184, 'flashing': 14185, 'tombstone': 14186, 'free-trade': 14187, 'al-manar': 14188, 'iqbal': 14189, 'gene': 14190, 'war-ravaged': 14191, 'conducive': 14192, 'yakubov': 14193, 'peso': 14194, 'dean': 14195, 'afforded': 14196, 'groaned': 14197, 'rehn': 14198, 'manipulate': 14199, 'demonizing': 14200, 'lugar': 14201, 'satisfy': 14202, 'dad': 14203, 'broadened': 14204, 'arsala': 14205, 'sheet': 14206, 'glaciers': 14207, 'corpus': 14208, 'christi': 14209, 'redraw': 14210, 'combining': 14211, 'agitated': 14212, 'walikale': 14213, 'exotic': 14214, 'maaleh': 14215, 'nullified': 14216, 'khayam': 14217, 'coca-cola': 14218, 'cse': 14219, 'stringent': 14220, 'organs': 14221, 'miljus': 14222, 'bankrupt': 14223, 'balaguer': 14224, 'catholicism': 14225, 'vasp': 14226, 'raviglione': 14227, 'eavesdrop': 14228, 'midway': 14229, 'nwr': 14230, 'shbak': 14231, 'gasquet': 14232, 'mirnyi': 14233, 'byzantine': 14234, 'guild': 14235, 'cate': 14236, 'blanchett': 14237, 'europol': 14238, 'nhs': 14239, 'paktiawal': 14240, 'verveer': 14241, 'al-naimi': 14242, 'ellice': 14243, 'tonga': 14244, 'zanzibar': 14245, 'fowler': 14246, 'myself': 14247, 'disengagement': 14248, 'shas': 14249, 'abolish': 14250, 's-300': 14251, 'kononov': 14252, 'solecki': 14253, 'montagnards': 14254, 'tokage': 14255, 'kindergarten': 14256, 'war-crimes': 14257, 'hound': 14258, 'clauses': 14259, 'attorney-general': 14260, 'negotiation': 14261, 'matesi': 14262, 'quist': 14263, 'petkoff': 14264, 'larose': 14265, 'arising': 14266, 'depended': 14267, 'anp': 14268, 'hares': 14269, '258': 14270, 'realizing': 14271, 'gration': 14272, 'saqib': 14273, 'nazareth': 14274, 'emigrated': 14275, '0.7': 14276, 'clarifies': 14277, 'meoni': 14278, 'benisede': 14279, 'jimenez': 14280, 'bergman': 14281, 'idema': 14282, 'hornets': 14283, 'rezai': 14284, 'preventive': 14285, '29th': 14286, 'olpc': 14287, 'samper': 14288, 'methane': 14289, 'vortex': 14290, 'taliban-linked': 14291, 'pokhrel': 14292, 'ahnlund': 14293, 'clemens': 14294, 'frightening': 14295, 'housedog': 14296, 'pro-moscow': 14297, 'collier': 14298, 'ousting': 14299, 'greeting': 14300, 'zim': 14301, 'merapi': 14302, 'merimee': 14303, 'flood-ravaged': 14304, 'demirel': 14305, 'fiat': 14306, 'kamin': 14307, 'pamphlets': 14308, 'ghost': 14309, 'mswati': 14310, 'soprano': 14311, 'costeira': 14312, 'rowers': 14313, 'azizuddin': 14314, 'diouf': 14315, 'minuteman': 14316, 'mirrors': 14317, 'mongols': 14318, 'hen': 14319, 'bittok': 14320, 'ney': 14321, 'johanns': 14322, 'quail': 14323, 'phrases': 14324, 'nightingale': 14325, 'kindhearts': 14326, 'rhee': 14327, 'sbs': 14328, 'prints': 14329, 'inoperable': 14330, 'murtha': 14331, 'rostropovich': 14332, 'monte': 14333, 'heifer': 14334, 'worm': 14335, 'hour-long': 14336, 'ferrying': 14337, 'bottlenecks': 14338, 'sow': 14339, 'trophy': 14340, 'geir': 14341, 'countless': 14342, 'calvin': 14343, 'broadus': 14344, 'crips': 14345, 'calamity': 14346, 'engages': 14347, 'complimentary': 14348, 'high-risk': 14349, 'overtaking': 14350, 'peruvians': 14351, 'comandante': 14352, 'ramona': 14353, 'zapatistas': 14354, 'rumored': 14355, 'american-islamic': 14356, 'well-documented': 14357, 'abd': 14358, 'un-islamic': 14359, 'concept': 14360, 'immunize': 14361, '2015': 14362, 'barakat': 14363, 'keyser': 14364, 'assessments': 14365, '2030': 14366, 'responsibly': 14367, 'stays': 14368, 'nicephore': 14369, 'soglo': 14370, 'ignorance': 14371, 'sheds': 14372, 'rebuffed': 14373, 'insurgency-related': 14374, 'snows': 14375, 'accessible': 14376, 'dia': 14377, '1819': 14378, 'qatari': 14379, 'khalifa': 14380, 'backbone': 14381, 'outlying': 14382, 'unskilled': 14383, 'bundle': 14384, 'rendered': 14385, 'wasted': 14386, 'tetiana': 14387, 'koprowicz': 14388, 'steadfastly': 14389, 'udhampur': 14390, 'brisk': 14391, 'superpower': 14392, 'qalat': 14393, 'flaming': 14394, 'downplay': 14395, 'baduel': 14396, 'f-5': 14397, 'formalize': 14398, 'condemnations': 14399, 'transplants': 14400, 'jama': 14401, 'dora': 14402, 'formidable': 14403, 'sweilam': 14404, 'payrolls': 14405, 'indicator': 14406, 'tend': 14407, 'census': 14408, 'petrocaribe': 14409, 'jaafar': 14410, 'diabetics': 14411, 'exhuming': 14412, 'garage': 14413, 'skeletons': 14414, 'massacred': 14415, 'revere': 14416, 'hrbaty': 14417, 'adelaide': 14418, 'onslaught': 14419, 'bouncing': 14420, 'hernych': 14421, 'junaid': 14422, 'philippoussis': 14423, 'seppi': 14424, 'florent': 14425, 'residences': 14426, 'skiers': 14427, 'stalling': 14428, 'arif': 14429, 'h1-b': 14430, 'balata': 14431, 'clout': 14432, 'lead-up': 14433, 'omi': 14434, 'measurable': 14435, '40-day': 14436, 'sandstorms': 14437, 'six-story': 14438, 'runways': 14439, 'krzysztof': 14440, 'visionary': 14441, 'taunting': 14442, 'bujumbura': 14443, 'fnl': 14444, 'remanded': 14445, 'ismael': 14446, 'oleg': 14447, 'privilege': 14448, 'gourmet': 14449, 'idb': 14450, 'enrolled': 14451, 'composer': 14452, 'ink': 14453, 'pencil': 14454, 'gara': 14455, 'portrayed': 14456, 'centerpiece': 14457, '212': 14458, 'zenith': 14459, 'depleted': 14460, 'daltrey': 14461, 'natives': 14462, '1563': 14463, 'townshend': 14464, 'barely': 14465, 'volatility': 14466, 'inherited': 14467, 'owing': 14468, 'nutmeg': 14469, 'philosopher': 14470, 'providence': 14471, 'perish': 14472, 'reflections': 14473, 'chaldean': 14474, 'pablo': 14475, 'chips': 14476, 'seville': 14477, 'al-mansoorain': 14478, 'smart-1': 14479, 'excellence': 14480, '7,200': 14481, 'propulsion': 14482, 'kariye': 14483, 'lindsey': 14484, 'howes': 14485, 'huot': 14486, 'wat': 14487, 'two-hectare': 14488, 'eisenman': 14489, 'evokes': 14490, 'non-jewish': 14491, 'archer': 14492, 'extrajudicial': 14493, 'farmhouse': 14494, 'internationally-brokered': 14495, '85,000': 14496, 'proportionately': 14497, 'encounters': 14498, 'lax': 14499, 'tying': 14500, 'daybreak': 14501, 'itv': 14502, 'reeling': 14503, 'kevljani': 14504, 'safavi': 14505, 'allah': 14506, 'dimitris': 14507, 'christofias': 14508, 'island-nation': 14509, 'earthquake-ravaged': 14510, 'quake-hit': 14511, 'centenary': 14512, 'burge': 14513, 'sattler': 14514, "o'hara": 14515, 'limiting': 14516, 'anti-sikh': 14517, '45-year-old': 14518, 'brushing': 14519, 'azamiyah': 14520, 'amiriyah': 14521, 'eutelsat': 14522, '1,170': 14523, 'infertility': 14524, 'patent': 14525, 'competitor': 14526, 'shaab': 14527, 'overdue': 14528, 'redeploy': 14529, 'khurshid': 14530, 'hardliners': 14531, 'tcdd': 14532, 'regrouped': 14533, 'eluded': 14534, 're-arrested': 14535, 'divulging': 14536, 'el-sheik': 14537, 'bazaar': 14538, 'innocents': 14539, 'bagpipes': 14540, 'lawn': 14541, 'shanksville': 14542, 'pokhara': 14543, 'legitimize': 14544, 'specifies': 14545, 'mazar-e-sharif': 14546, 'collaborators': 14547, 'jailhouse': 14548, 'debby': 14549, 'stephanides': 14550, 'siphoned': 14551, 'devised': 14552, 'settling': 14553, 'nascent': 14554, 'cans': 14555, 'jang': 14556, 'appalling': 14557, 'prohibits': 14558, 'canned': 14559, 'warehouses': 14560, 'shallow': 14561, 'pens': 14562, 'philanthropist': 14563, '1896': 14564, '1901': 14565, 'unexplored': 14566, 'rehabilitate': 14567, 'match-up': 14568, 'lecturer': 14569, 'gai': 14570, 'disability': 14571, 'integrate': 14572, 'disabilities': 14573, 'al-rai': 14574, 'cash-strapped': 14575, 'beneficiaries': 14576, 'amiin': 14577, 'illiteracy': 14578, 'disparity': 14579, 'medium-term': 14580, 'fluctuating': 14581, 'swings': 14582, 'pre-independence': 14583, 'undeveloped': 14584, 'taunted': 14585, 'headlights': 14586, 'unjust': 14587, 'remind': 14588, 'apples': 14589, 'lonely': 14590, 'graner': 14591, 'sabrina': 14592, 'near-earth': 14593, 'orbits': 14594, '20-ton': 14595, 'capsule': 14596, '2018': 14597, 'landings': 14598, '43,000': 14599, 'enslaved': 14600, 'saharan': 14601, 'u.s.-owned': 14602, 'preempt': 14603, 'taksim': 14604, 'off-limits': 14605, 'dawa': 14606, 'monasteries': 14607, 'burton': 14608, 'austrians': 14609, 'casts': 14610, 'girija': 14611, 'saadoun': 14612, 'skirted': 14613, 'suffocating': 14614, 'stopover': 14615, 'takeoff': 14616, 'azarov': 14617, 'waits': 14618, 'bury': 14619, 'nanmadol': 14620, 'two-thousand': 14621, 'eight-thousand': 14622, 'issa': 14623, 'tchiroma': 14624, 'examinations': 14625, 'bakassi': 14626, 'advancement': 14627, 'flavinols': 14628, 'motorway': 14629, 'intermediary': 14630, 'abandons': 14631, 'precedents': 14632, 'schalkwyk': 14633, '2025': 14634, 'deif': 14635, 'humiliation': 14636, 'season-ending': 14637, 'teimuraz': 14638, 'gabashvili': 14639, 'serbia-montenegro': 14640, 'subur': 14641, 'sugiarto': 14642, 'shin': 14643, 'intoxicated': 14644, 'infringe': 14645, 'rudolf': 14646, 'apostolic': 14647, 'eucharist': 14648, 'assadullah': 14649, 'parwan': 14650, 'crucifixes': 14651, 'reverence': 14652, 'mini-strokes': 14653, 'oxford': 14654, 'researcher': 14655, 'facial': 14656, 'thinning': 14657, 'disturbing': 14658, 'exceptions': 14659, 'circumvent': 14660, 'wording': 14661, 'inconvenient': 14662, 'rohani': 14663, 'european-brokered': 14664, 'nevsky': 14665, 'prospekt': 14666, 'terrorist-related': 14667, 'cater': 14668, 'interpretation': 14669, 'al-ghoul': 14670, 'ethnic-chinese': 14671, 'bust': 14672, 'holmes': 14673, 'summons': 14674, 'certify': 14675, 'zaman': 14676, 'cds': 14677, 'laptops': 14678, 'sardinia': 14679, 'bologna': 14680, 'ideologies': 14681, 'pretrial': 14682, 'motions': 14683, 'drawdown': 14684, 'marginalized': 14685, 'qinghong': 14686, 'five-nation': 14687, 'trinidad': 14688, 'tobago': 14689, 'widower': 14690, 'zimmermann': 14691, 'grubbs': 14692, 'schrock': 14693, 'yves': 14694, 'chauvin': 14695, 'environmentally-friendly': 14696, 'molecules': 14697, 'reproduce': 14698, 'impassable': 14699, 'branded': 14700, 'tottenham': 14701, 'stint': 14702, 'teammates': 14703, 'achilles': 14704, 'ac': 14705, 'bowen': 14706, 'aleksander': 14707, 'caicos': 14708, 'eilat': 14709, 'armorgroup': 14710, 'outburst': 14711, 'enters': 14712, '37.5': 14713, '108': 14714, '1719': 14715, 'eurozone': 14716, 'zeta': 14717, 'invoked': 14718, 'threshold': 14719, 'mutiny': 14720, 'aggrieved': 14721, 'proportional': 14722, 'leases': 14723, 'soils': 14724, 'reins': 14725, 'saddled': 14726, 'foil': 14727, 'apple-tree': 14728, 'dear': 14729, 'flourishing': 14730, 'accountant': 14731, '33-year-old': 14732, 'assignment': 14733, 'doomed': 14734, 'government-funded': 14735, 'erben': 14736, 'kanyabayonga': 14737, 'fragments': 14738, 'dwellers': 14739, 'urgency': 14740, 'renovation': 14741, 'hottest': 14742, 'appliances': 14743, 'yunis': 14744, 'forgives': 14745, 'humiliations': 14746, 'tsunami-affected': 14747, '386': 14748, '9.00': 14749, 'somalians': 14750, 'miltzow': 14751, 'unloaded': 14752, 'torgelow': 14753, 'qualities': 14754, 'hamidzada': 14755, 'misunderstanding': 14756, 'aleem': 14757, 'siddique': 14758, 'ntv': 14759, 'forget': 14760, 'nonsmokers': 14761, 'implies': 14762, '8.20': 14763, 'suef': 14764, 'rationale': 14765, 'hypertension': 14766, 'squared': 14767, 'counter-protesters': 14768, '722': 14769, 'rajouri': 14770, 'juncker': 14771, 'judgments': 14772, 'boasted': 14773, 'plo': 14774, 'hayward': 14775, 'skeptical': 14776, 'leon': 14777, 'mezni': 14778, 'jendayi': 14779, 'frazer': 14780, 'khawaza': 14781, 'desires': 14782, 'layer': 14783, 'glitch': 14784, 'tumbled': 14785, 'stave': 14786, 'seyni': 14787, '31-member': 14788, 'yanukovich': 14789, 'inadmissible': 14790, 'sliders': 14791, 'lund': 14792, '18-year': 14793, 'hydrogen': 14794, 'unrealistic': 14795, 'tachileik': 14796, 'seven-hour': 14797, '756': 14798, 'carnage': 14799, 'post-saddam': 14800, 'degraded': 14801, 'revitalizing': 14802, 'jumpstart': 14803, 'erbamont': 14804, 'blaise': 14805, 'compaore': 14806, 'stubbornly': 14807, 'recreation': 14808, 'meadow': 14809, 'happiness': 14810, 'groom': 14811, 'alas': 14812, 'scoundrel': 14813, 'pitching': 14814, 'excessively': 14815, 'preparedness': 14816, 'festivals': 14817, 'montserrat': 14818, 'islamist-led': 14819, 'qualifiers': 14820, 'surprising': 14821, 'stunning': 14822, 'niamey': 14823, 'stade': 14824, 'abel': 14825, 'essam': 14826, 'seven-time': 14827, 'stallone': 14828, 'performance-enhancing': 14829, '86,000': 14830, 'nagpur': 14831, 'oral': 14832, 'histories': 14833, 'radiation': 14834, 'ulcers': 14835, 'alcoholic': 14836, 'feeds': 14837, 'hartford': 14838, 'equip': 14839, 'repatriating': 14840, 'renovating': 14841, 'holes': 14842, 'solo': 14843, 'one-meter': 14844, 'seventeen-year-old': 14845, 'capitalize': 14846, 'entirety': 14847, 'gratitude': 14848, 'alexy': 14849, 'yakutsk': 14850, 'epiphany': 14851, 'iacovou': 14852, 'greek-led': 14853, 'sadrist': 14854, 'boomed': 14855, '266': 14856, '19.9': 14857, 'agoa': 14858, 'pro-thaksin': 14859, 'firewood': 14860, 'persuaded': 14861, 'outlandish': 14862, 'anytime': 14863, 'penal': 14864, 'filibuster': 14865, 'priscilla': 14866, '2019': 14867, 'mankind': 14868, 'prevailed': 14869, 'heaven': 14870, 'righteous': 14871, 'consulted': 14872, 'juvenal': 14873, 'habyarimana': 14874, 'el~universal': 14875, 'attaché': 14876, 'decreed': 14877, 'henceforth': 14878, 'illustrate': 14879, 'ben-eliezer': 14880, 'ueberroth': 14881, 'arises': 14882, 'singly': 14883, 'yasin': 14884, 'attendants': 14885, 'varieties': 14886, 'wagner': 14887, 'valle': 14888, 'stripping': 14889, 'redefine': 14890, 'ilam': 14891, 'tang': 14892, 'undertaking': 14893, 'drunkenness': 14894, 'kaliningrad': 14895, 'refuted': 14896, 'tribespeople': 14897, 'deutsche': 14898, 'welle': 14899, 'musayyib': 14900, 'atrocious': 14901, 'pronounced': 14902, 'mourned': 14903, 'rendition': 14904, 'dec.': 14905, 'kearny': 14906, 'year-ago': 14907, 'irrigated': 14908, 'simone': 14909, 'ex-communist': 14910, 'margins': 14911, 'barba': 14912, 'berdimuhamedow': 14913, 'polyglot': 14914, 'frog': 14915, 'lame': 14916, 'wrinkled': 14917, 'snake': 14918, 'filberts': 14919, 'grasped': 14920, 'readily': 14921, 'federally': 14922, 'malakand': 14923, '337-9': 14924, '28th': 14925, 'damper': 14926, 'entice': 14927, 'hezbollah-led': 14928, 'himat': 14929, '60-day': 14930, 'cheerful': 14931, 'kwame': 14932, 'nkrumah': 14933, 'pre-trial': 14934, 'gongadze': 14935, 'maltreatment': 14936, 'mistrial': 14937, 'bowman': 14938, 'bahraini': 14939, 'aiyar': 14940, 'three-way': 14941, 'pangandaran': 14942, 'skipped': 14943, 'herds': 14944, '5000': 14945, 'scorched': 14946, 'robinson': 14947, 'nasal': 14948, 'calorie': 14949, 'bless': 14950, 'hinted': 14951, 'timely': 14952, 'canceling': 14953, 'saboteurs': 14954, 'blessed': 14955, 'devout': 14956, 'emotions': 14957, 'publicize': 14958, 'vaeedi': 14959, '9.4': 14960, '14.5': 14961, '787': 14962, 'facilitator': 14963, 'guiding': 14964, 'glenn': 14965, 'antitrust': 14966, 'domodedovo': 14967, 'novosti': 14968, 'guess': 14969, 'u.s.-brokered': 14970, 'balochistan': 14971, 'gas-rich': 14972, 'berlin-based': 14973, '146': 14974, 'ignores': 14975, 'kryvorizhstal': 14976, 'qaeda': 14977, 'woefully': 14978, 'providers': 14979, '95,000': 14980, '429': 14981, 'locales': 14982, 'indoors': 14983, 'trades': 14984, 'concealing': 14985, 'conscripted': 14986, 'shallah': 14987, 'chip': 14988, 'gilad': 14989, 'randomly': 14990, 'salerno': 14991, 'duel': 14992, 'five-time': 14993, 'maxim': 14994, '15-kilometer': 14995, 'snowboarding': 14996, 'kelly': 14997, '85-year-old': 14998, 'encompass': 14999, '676': 15000, 'rossi': 15001, 'runner': 15002, 'compiling': 15003, 'shoved': 15004, 'pediatricians': 15005, '85th': 15006, 'mullaittivu': 15007, 'upheaval': 15008, 'duncan': 15009, 'denounces': 15010, '23.5': 15011, 'integra': 15012, 'hallwood': 15013, 'masoum': 15014, 'piped': 15015, 'repository': 15016, 'mid-2006': 15017, 'idps': 15018, 'plumage': 15019, 'pools': 15020, 'habit': 15021, 'nettle': 15022, 'boldly': 15023, 'philosophy': 15024, 'previously-unknown': 15025, 'hiking': 15026, 'draining': 15027, 'ninety': 15028, 'aerospace': 15029, 'extorting': 15030, 'ransoms': 15031, 'trainers': 15032, 'anti-european': 15033, 'davies': 15034, 'democratic-controlled': 15035, 'override': 15036, 'derailment': 15037, 'griffal': 15038, 'joye': 15039, 'quinn': 15040, 'two-run': 15041, 'medalists': 15042, 'myles': 15043, 'erin': 15044, 'courtney': 15045, 'yang': 15046, 'bhat': 15047, 'stroll': 15048, 'clad': 15049, 'colored': 15050, 'marlins': 15051, 'shofar': 15052, 'birkenau': 15053, '21-month-old': 15054, 'cassim': 15055, 'flu-related': 15056, '530': 15057, 'sterility': 15058, 'vice-presidential': 15059, 'spotlight': 15060, 'padden': 15061, 'lapses': 15062, '177': 15063, 'delgado': 15064, 'oliver': 15065, 'garcia-lopez': 15066, 'raad': 15067, 'hashim': 15068, 'darren': 15069, '4,20,000': 15070, 'capping': 15071, 'laith': 15072, 'iranian-backed': 15073, 'al-rashid': 15074, 'rockslide': 15075, 'manshiyet': 15076, 'nasr': 15077, 'rockslides': 15078, 'grains': 15079, 'mid-week': 15080, 'sidnaya': 15081, 'scanned': 15082, 'iraqi-born': 15083, 'spear': 15084, 'karabila': 15085, 'experimental': 15086, 'gardasil': 15087, 'cancerous': 15088, 'hpvs': 15089, 'kiraithe': 15090, 'hanif': 15091, 'hai': 15092, 'puppets': 15093, 'milliyet': 15094, 'willingly': 15095, 'fatah-led': 15096, 'everglades': 15097, 'behnam': 15098, 'nateghi': 15099, 'tearful': 15100, 'raphael': 15101, '11.8': 15102, '298': 15103, 'laps': 15104, 'itno': 15105, 'mulino': 15106, 'khartoum-backed': 15107, 'kempthorne': 15108, 'regulates': 15109, 'u.n.-brokered': 15110, 'rightwing': 15111, 'gabriel': 15112, 'unthinkable': 15113, 'stunt': 15114, 'kosachyov': 15115, 'faxed': 15116, 'contempt': 15117, 'shein': 15118, 'competitions': 15119, '175': 15120, 'free-enterprise': 15121, 'incorporation': 15122, 'sanctioned': 15123, 'harmonize': 15124, 'readmitted': 15125, 'float': 15126, 'depreciate': 15127, 'sanction': 15128, 'localities': 15129, '1848': 15130, '1874': 15131, 'jealous': 15132, 'omen': 15133, 'loudly': 15134, 'wondered': 15135, 'caw': 15136, 'tanner': 15137, 'smell': 15138, 'inconvenience': 15139, 'centre': 15140, 'dissemble': 15141, 'pond': 15142, 'shantytowns': 15143, 'leaned': 15144, 'scotia': 15145, 'varied': 15146, '116': 15147, 'sook': 15148, 'omdurman': 15149, 'chapters': 15150, 'alienated': 15151, 'barges': 15152, 'wisconsin': 15153, 'derailing': 15154, 'educating': 15155, 'transatlantic': 15156, 'kar': 15157, 'seismological': 15158, 'hillsides': 15159, 'repealing': 15160, 'regressing': 15161, 'across-the-board': 15162, 'substitute': 15163, 'deflected': 15164, 'two-goal': 15165, 'shaktoi': 15166, 'hakimullah': 15167, 'schapelle': 15168, 'corby': 15169, 'toughest': 15170, 'vienna-based': 15171, 'savic': 15172, 'visegrad': 15173, 'recreated': 15174, 'vargas': 15175, 'importer': 15176, 'kamchatka': 15177, 'remorse': 15178, 'unfit': 15179, 'mutilation': 15180, 'circumcision': 15181, 'no-win': 15182, 'abusers': 15183, 'military-installed': 15184, 'sunset': 15185, 'nyala': 15186, 'naturalized': 15187, 'marseille': 15188, 'royal-dutch': 15189, 'multi-national': 15190, 'deep-sea': 15191, 'analyzing': 15192, 'uproar': 15193, 'kuznetsov': 15194, 'fabricating': 15195, 'massacring': 15196, 'princess': 15197, 'dragomir': 15198, 'airstrips': 15199, 'yusef': 15200, 'manchester': 15201, 'fawaz': 15202, 'nineteen': 15203, 'mara': 15204, 'salvatrucha': 15205, 'chandler': 15206, 'yacht': 15207, 'high-seas': 15208, 'accumulation': 15209, 'ningxia': 15210, 'barzan': 15211, 'divorcing': 15212, 'pamela': 15213, 'exempts': 15214, 'inconsistencies': 15215, 'blindness': 15216, 'saqiz': 15217, 'radioed': 15218, 'mammals': 15219, 'drown': 15220, 'pro-whaling': 15221, 'molested': 15222, 'narsingh': 15223, 'disqualify': 15224, 'overly': 15225, 'exposition': 15226, 'homing': 15227, '1.32': 15228, 'carat': 15229, 'jewelers': 15230, 'moriarty': 15231, 'mid-2000s': 15232, 'arable': 15233, 'bozize': 15234, 'persist': 15235, 'sparsely': 15236, 'capitalist': 15237, 'jalil': 15238, 'highly-indebted': 15239, 'greedily': 15240, 'mortal': 15241, 'agony': 15242, 'windfall': 15243, 'desiring': 15244, 'novakatka': 15245, 'novakatkan': 15246, 'novakatkans': 15247, 'pul-i-charki': 15248, 'heidar': 15249, 'moslehi': 15250, 'expediency': 15251, 'arch-rival': 15252, '404': 15253, '7,500': 15254, 'electrician': 15255, 'bibles': 15256, 'slit': 15257, 'malatya': 15258, 'demonstrator': 15259, 'appeasing': 15260, 'taunt': 15261, 'affront': 15262, 'punched': 15263, 'shoulders': 15264, 'cremation': 15265, 'uncertainties': 15266, 'stakeholder': 15267, 'formations': 15268, 'mystery': 15269, 'al-fagih': 15270, 'overweight': 15271, 'obese': 15272, 'clearer': 15273, 'sub-prime': 15274, 'u-2': 15275, 'bonny': 15276, 'overboard': 15277, 'birmingham': 15278, 'antonovs': 15279, 'naypyidaw': 15280, 'yong': 15281, 'alperon': 15282, 'careers': 15283, 'balloons': 15284, 'resemble': 15285, 'hat': 15286, 'lamppost': 15287, 'bayrou': 15288, 'patron': 15289, 'leyva': 15290, 'caverns': 15291, 'hapoalim': 15292, 'schwab': 15293, 'lender': 15294, 'lars': 15295, 'bountiful': 15296, 'osuji': 15297, 'comprising': 15298, '27.5': 15299, '50.9': 15300, 'bragg': 15301, 'bolivarian': 15302, 'orchestrate': 15303, 'liberated': 15304, 'exterminate': 15305, 'darom': 15306, 'evaded': 15307, 'kato': 15308, 'sustaining': 15309, 'possibilities': 15310, '44th': 15311, 'wonderful': 15312, 'professionalism': 15313, 'norman': 15314, 'swords': 15315, 'righteousness': 15316, 'rebellious': 15317, 'arusha': 15318, 'in-flight': 15319, 'demagogue': 15320, 'fidelity': 15321, 'l.': 15322, 'kane': 15323, 'tunisians': 15324, 'after-tax': 15325, 'terminated': 15326, 'modernizing': 15327, 'emphasis': 15328, 'public-private': 15329, 'competes': 15330, 'furniture': 15331, 'paints': 15332, 'tiles': 15333, 'outflows': 15334, 'cushion': 15335, 'varying': 15336, 'abdoulaye': 15337, 'horseman': 15338, 'faculty': 15339, 'sink': 15340, 'melissa': 15341, 'schoomaker': 15342, 'nuclear-related': 15343, 'triumph': 15344, 'cleland': 15345, 'steer': 15346, 'government-owned': 15347, 'leumi': 15348, 'stockpiling': 15349, 'enticed': 15350, 'slovenian': 15351, 'sutanto': 15352, 'batu': 15353, 'masterminding': 15354, '69th': 15355, 'fiorina': 15356, 'hp': 15357, 'resisting': 15358, 'vintsuk': 15359, 'vyachorka': 15360, 'quashing': 15361, 'keen': 15362, 'preceded': 15363, 'mireya': 15364, 'moscoso': 15365, 'servers': 15366, 'mejia': 15367, 'inflammatory': 15368, 'waterway': 15369, 'baikal': 15370, 'twins': 15371, 'fiancee': 15372, 'monoxide': 15373, 'adriano': 15374, 'ronaldinho': 15375, 'naftogaz': 15376, 'sullivan': 15377, 'ekaterina': 15378, 'peer': 15379, 'aiko': 15380, 'nakamura': 15381, 'cassation': 15382, '05-feb': 15383, 'yoon-jeong': 15384, 'pianist': 15385, 'irina': 15386, 'leverage': 15387, 'calmed': 15388, 'notable': 15389, 'cricketers': 15390, 'match-fixing': 15391, 'intentionally': 15392, 'mosquito-borne': 15393, 'imitation': 15394, 'aleppo': 15395, 'sofyan': 15396, 'edited': 15397, 'ambiguous': 15398, 'anna': 15399, 'rick': 15400, 'misunderstood': 15401, 'pattaya': 15402, 'homelessness': 15403, 'crowns': 15404, 'atagi': 15405, 'adamantly': 15406, 'faizabad': 15407, 'kohistan': 15408, 'marie-jeanne': 15409, 'sorin': 15410, 'miscoci': 15411, 'ovidiu': 15412, 'ohanesian': 15413, 'crnogorac': 15414, 'banja': 15415, 'luka': 15416, 'minustah': 15417, 'mansoor': 15418, 'ayodhya': 15419, 'nano-technology': 15420, 'atoms': 15421, 'anatolian': 15422, 'israeli-american': 15423, 'runoffs': 15424, 'galina': 15425, 'dogged': 15426, 'firebombing': 15427, 'eid-al-fitr': 15428, 'century-old': 15429, 'hsann': 15430, 'subsidizing': 15431, 'sunrise': 15432, 'greedy': 15433, 'water-borne': 15434, '17.3': 15435, 'packaging': 15436, 'discontinued': 15437, 'pretax': 15438, '133': 15439, 'reorganize': 15440, 'fertility': 15441, 'municipalities': 15442, 'chronically': 15443, 'deepest': 15444, 'upheavals': 15445, 'culminating': 15446, 'retribution': 15447, 'zaire': 15448, 'rout': 15449, 'perfumes': 15450, 'apt': 15451, 'till': 15452, 'forgot': 15453, 'robes': 15454, 'huntsman': 15455, 'basket': 15456, 'abstain': 15457, 'dragging': 15458, 'inaction': 15459, 'fluid': 15460, 'machine-gun': 15461, 'claudia': 15462, 'complication': 15463, 'ted': 15464, 'greats': 15465, 'parker': 15466, 'hooker': 15467, 'eyesight': 15468, 'surgeons': 15469, 'lasts': 15470, 'flashpoint': 15471, 'golf': 15472, 'vera': 15473, 'amelie': 15474, 'stewardship': 15475, 'dealership': 15476, 'daring': 15477, 'phony': 15478, 'araujo': 15479, 'maltreating': 15480, 'night-time': 15481, 'rampaging': 15482, 'seclusion': 15483, 'sur': 15484, 'grinning': 15485, 'sunk': 15486, 'compliant': 15487, 'explosives-filled': 15488, 'shaalan': 15489, 'respirator': 15490, 'inhumane': 15491, 'bayaman': 15492, 'queensland': 15493, 'nicknamed': 15494, 'cooper': 15495, 'ranchers': 15496, 'necks': 15497, 'al-rishawi': 15498, 'bluntly': 15499, 'possessed': 15500, 'mozambican': 15501, 'anjar': 15502, 'wagah': 15503, 'miami-based': 15504, 'kurzban': 15505, 'depleting': 15506, 'berdych': 15507, 'novak': 15508, 'djokovic': 15509, 'verdasco': 15510, 'nikolay': 15511, 'davydenko': 15512, 'branislav': 15513, 'warplane': 15514, 'cartosat-1': 15515, 'hamsat': 15516, 'sriharokota': 15517, 'pslav': 15518, 'well-supplied': 15519, 'platero': 15520, 'synagogues': 15521, 'advertise': 15522, '05-apr': 15523, 'violence-plagued': 15524, 'altercation': 15525, 'highest-rated': 15526, 'panicking': 15527, 'fairer': 15528, 'deepening': 15529, 'resorted': 15530, 'anti-israeli': 15531, 'turkmens': 15532, 'per-capita': 15533, 'depress': 15534, 'steepest': 15535, 'ryazanov': 15536, 'sundance': 15537, 'monica': 15538, 'surgeries': 15539, 'x-rays': 15540, 'pyinmana': 15541, 'soren': 15542, 'vuk': 15543, 'draskovic': 15544, 'albanian-led': 15545, 'bomb-proof': 15546, 'gereshk': 15547, 'outlay': 15548, 'longest-serving': 15549, 'guebuza': 15550, 'disqualification': 15551, 'terminate': 15552, 'discouraged': 15553, 'mid-year': 15554, 'awori': 15555, 'giuliana': 15556, 'blockaded': 15557, 'environment-friendly': 15558, 'conserve': 15559, 'necessarily': 15560, 'laments': 15561, 'stove': 15562, 'onitsha': 15563, 'anambra': 15564, 'scuffle': 15565, 'sparks': 15566, 'overshadowed': 15567, 'repercussions': 15568, 'inter': 15569, 'persecuting': 15570, 'al-muhajer': 15571, 'calmer': 15572, 'succumb': 15573, 'al-shahristani': 15574, 'timeframe': 15575, 'molesting': 15576, 'mannar': 15577, 'squandered': 15578, 'tiebreaker': 15579, 'vliegen': 15580, 'seymour': 15581, 'drowning': 15582, 'electrocution': 15583, 'sprinters': 15584, 'lausanne-based': 15585, 'render': 15586, 'provisionally': 15587, 'napoleon': 15588, 'grooming': 15589, 'dn': 15590, 'ethnicity': 15591, 'pedophiles': 15592, 'halls': 15593, 'six-country': 15594, 'infiltrators': 15595, 'gaza-based': 15596, 'straining': 15597, 'truckloads': 15598, 'weakest': 15599, 'baluchis': 15600, 'mendez': 15601, 'soweto': 15602, 'corruption-related': 15603, 'constitutes': 15604, 'al-attiya': 15605, 'native-born': 15606, 'migrated': 15607, 'builds': 15608, 'lutfullah': 15609, 'mashal': 15610, 'newborn': 15611, 'yinghua': 15612, 'cubs': 15613, 'authenticate': 15614, 'algerians': 15615, 'facilitation': 15616, 'daytime': 15617, 'go-ahead': 15618, 'neglecting': 15619, 'war-related': 15620, 'jakob': 15621, 'dadis': 15622, 'erecting': 15623, 'knock': 15624, 'celebratory': 15625, 'turpan': 15626, 'offshoot': 15627, 'elco': 15628, 'ill.': 15629, '1.125': 15630, 'over-the-counter': 15631, 'commodity-based': 15632, '1829': 15633, 'protracted': 15634, 'hydrocarbon': 15635, 'saparmurat': 15636, 'nyyazow': 15637, 'pearls': 15638, 'rekindled': 15639, 'tongue': 15640, 'thistles': 15641, 'kari': 15642, 'reveals': 15643, 'haitian-born': 15644, 'dumarsais': 15645, '134': 15646, 'symbolizing': 15647, 'thais': 15648, 'pacts': 15649, 'disadvantage': 15650, 'looms': 15651, 'dongfeng': 15652, 'uniting': 15653, 'strips': 15654, 'crumpton': 15655, '324': 15656, 'paired': 15657, 'south-southeast': 15658, 'relocating': 15659, 'immigrated': 15660, 'dictatorial': 15661, 'rejections': 15662, 'riga': 15663, 'tirah': 15664, 'azar': 15665, 'fracture': 15666, 'australian-led': 15667, 'sting': 15668, 'switched': 15669, 'vitaly': 15670, 'churkin': 15671, 'tattoo': 15672, 'storied': 15673, 'california-based': 15674, 'bordered': 15675, 'hefty': 15676, 'walesa': 15677, 'speedy': 15678, 'al-nida': 15679, 'most-watched': 15680, '7.7': 15681, 'ashley': 15682, '80s': 15683, 'stalinist': 15684, 'kharazzi': 15685, 'nuclear-fuel': 15686, 'kocharian': 15687, 'characterizing': 15688, 'russia-backed': 15689, 'thanh': 15690, 'hoa': 15691, 'nalchik': 15692, 'reliable': 15693, 'maulvi': 15694, 'decentralized': 15695, 'resilience': 15696, 'shoulder-fired': 15697, 'gonzalo': 15698, 'dimming': 15699, 'nonessential': 15700, 'riches': 15701, 'pyramids': 15702, 'cube': 15703, 'candlelight': 15704, 'maarib': 15705, 'phasing': 15706, 'midsized': 15707, 'muscat': 15708, 'qaboos': 15709, 'reshuffled': 15710, 'ostensible': 15711, 'hizballah': 15712, 'antigovernment': 15713, 'dredging': 15714, 'locks': 15715, 'wonder': 15716, 'awakened': 15717, 'grudge': 15718, 'ross': 15719, 'metin': 15720, 'kaplan': 15721, 'ataturk': 15722, 'cyber-dissidents': 15723, 'basilan': 15724, 'precipitation': 15725, 'zionist': 15726, 'lucky': 15727, 'marshals': 15728, 'roasted': 15729, 'nicole': 15730, 'mitofsky': 15731, 'umaru': 15732, 'edged': 15733, 'medellin': 15734, '780': 15735, 'assassinating': 15736, 'democratically-elected': 15737, 'punk': 15738, 'defunct': 15739, 'mest': 15740, 'matt': 15741, 'label': 15742, 'disbanding': 15743, 'blumenthal': 15744, 'swimmers': 15745, 'co-star': 15746, 'five-tenths': 15747, 'one-tenth': 15748, 'markos': 15749, 'kongsak': 15750, 'bangguo': 15751, 'mebki': 15752, 'bouake': 15753, 'ivorians': 15754, 'contestants': 15755, 'ljoekelsoey': 15756, 'jumps': 15757, '788': 15758, 'widhoelzl': 15759, 'gunboats': 15760, 'madden': 15761, 'charlotte': 15762, 'halfway': 15763, 'hamburgers': 15764, '8.2': 15765, 'hey': 15766, 'download': 15767, 'hamburg': 15768, 'yazidi': 15769, 'sinjar': 15770, 'qasim': 15771, 'saeedi': 15772, 'disorders': 15773, '64,000': 15774, 'prussia': 15775, 'levies': 15776, 'foreign-backed': 15777, 'nioka': 15778, 'spied': 15779, 'amisom': 15780, 'bahonar': 15781, 'vahidi': 15782, 'vahid': 15783, 'dilapidated': 15784, 'sajjad': 15785, 'sharki': 15786, 'shadid': 15787, 'thoroughly': 15788, 'depots': 15789, 'culmination': 15790, 'belatedly': 15791, 'gala': 15792, 'danes': 15793, '3,937': 15794, 'wrongfully': 15795, 'purged': 15796, 'rafts': 15797, 'banjul': 15798, 'privatizing': 15799, 'simplify': 15800, 'autopsy': 15801, 'self-declared': 15802, 'thune': 15803, '0.9': 15804, 'employer': 15805, 'ed': 15806, 'xingning': 15807, 'harmonized': 15808, 'mutahida': 15809, 'religion-based': 15810, 'betterment': 15811, 'retaining': 15812, 'consolidated': 15813, 'tub': 15814, 'baked': 15815, 'weddings': 15816, 'milling': 15817, 'leather': 15818, 'jute': 15819, '1907': 15820, 'bhutanese': 15821, 'formalized': 15822, 'unhcr': 15823, 'jigme': 15824, 'wangchuck': 15825, 'abdicated': 15826, 'renegotiated': 15827, 'preferential': 15828, 'borrowed': 15829, 'richly': 15830, 'sour': 15831, 'nine-year': 15832, 'skimming': 15833, 'vents': 15834, 'containment': 15835, '53,000': 15836, 'minors': 15837, 'gallows': 15838, 'hoshide': 15839, 'plaintiffs': 15840, 'mayardit': 15841, 'purity': 15842, 'subscribers': 15843, 'disappearing': 15844, 'folly': 15845, 'near-record': 15846, '199': 15847, 'selections': 15848, 'bono': 15849, 'vinh': 15850, 'nadu': 15851, 'dissipating': 15852, 'oz': 15853, 'mgm': 15854, 'dvds': 15855, 'patterson': 15856, 'firearm': 15857, 'co-conspirator': 15858, 'hammad': 15859, 'hilda': 15860, 'makeover': 15861, 'dhahran': 15862, 'frequented': 15863, 'gustavo': 15864, 'feeder': 15865, 'pitches': 15866, 'shurpayev': 15867, 'reestablish': 15868, 'arboleda': 15869, 'malfunctioned': 15870, 'strangled': 15871, 'untapped': 15872, 'uranium-enriching': 15873, 'observation': 15874, 'seventeen': 15875, '1,20,000': 15876, 'north-west': 15877, 'piled': 15878, 'protestants': 15879, 'vandalized': 15880, 'sergio': 15881, 'viera': 15882, 'mello': 15883, 'wirayuda': 15884, 'albar': 15885, 'diplomatically': 15886, 'disturbances': 15887, 'singer-actress': 15888, 'unaware': 15889, 'tolls': 15890, 'misunderstandings': 15891, 'supermarket': 15892, 'tabare': 15893, 'meddle': 15894, 'u.s.-venezuelan': 15895, 'heydari': 15896, 'inevitable': 15897, 'shaping': 15898, 'voicing': 15899, 'vested': 15900, 'factual': 15901, 'electronically': 15902, 'email': 15903, 'turkic-speaking': 15904, 'explores': 15905, 'chandrayaan-1': 15906, 'sensors': 15907, 'point-blank': 15908, 'iscuande': 15909, 'moreno-ocampo': 15910, 'duped': 15911, 'reassure': 15912, 'sulaimaniya': 15913, 'foe': 15914, 'quadrupled': 15915, 'aliases': 15916, 'conditional': 15917, 'baotou': 15918, 'canadian-made': 15919, 'crj-200': 15920, 'bakr': 15921, 'intercepting': 15922, 'wrongful': 15923, 'hurtado': 15924, 'u.s.-built': 15925, 'roth': 15926, 'murderous': 15927, 'builders': 15928, 'developers': 15929, 'malls': 15930, 'a.l.': 15931, 'subsidiaries': 15932, 'liabilities': 15933, 'fijians': 15934, 'smarter': 15935, 'realize': 15936, 'mcc': 15937, 'violently': 15938, 'flourishes': 15939, 'gum': 15940, 'flesh': 15941, 'statues': 15942, 'gods': 15943, 'nearer': 15944, 'outrun': 15945, 'realities': 15946, 'sadly': 15947, 'ample': 15948, 'retrieve': 15949, 'spray': 15950, 'lifestyle': 15951, 'lobbed': 15952, 'causalities': 15953, 'inclusive': 15954, 'militarily': 15955, 'mingled': 15956, 'fireball': 15957, 'wasteful': 15958, 'xdr-tb': 15959, 'quo': 15960, 'namibians': 15961, 'grimes': 15962, 'angelo': 15963, 'sodano': 15964, 'mad-cow': 15965, 'vladivostok': 15966, 'navies': 15967, 'al-hassani': 15968, 'stoned': 15969, 'belaroussi': 15970, 'wakil': 15971, 'muttawakil': 15972, 'stanislaw': 15973, 'izarra': 15974, '520': 15975, 'ex-husband': 15976, 'lapd': 15977, 'probed': 15978, 'eimiller': 15979, 'jayden': 15980, 'daron': 15981, 'strobl': 15982, 'stressing': 15983, 'daud': 15984, 'cooperative': 15985, 'strive': 15986, 'forgo': 15987, 'limmt': 15988, 'miramax': 15989, 'shakespeare': 15990, 'bidders': 15991, 'weinstein': 15992, 'stanizai': 15993, 'militarized': 15994, 'ronnie': 15995, 'andry': 15996, '1,11,000': 15997, 'najibullah': 15998, 'mwangura': 15999, 'seafarers': 16000, 'kismayo': 16001, 'inconceivable': 16002, 'fradkov': 16003, 'anti-communist': 16004, '760': 16005, 'proposing': 16006, 'byrd': 16007, 'sebastien': 16008, 'choong': 16009, '1.16': 16010, 'hurriyah': 16011, 'marcus': 16012, 'peugeot': 16013, '1.24': 16014, 'eight-year-old': 16015, 'zahar': 16016, 'malfunction': 16017, 'hamas-dominated': 16018, 'guajira': 16019, 'connects': 16020, 'inquest': 16021, 'routing': 16022, 'bernama': 16023, 'patricia': 16024, 'massing': 16025, 'atenco': 16026, 'tamin': 16027, 'nuristani': 16028, 'underscores': 16029, 'untouched': 16030, 'tarnished': 16031, 'searchers': 16032, 'parliaments': 16033, 'el-dabaa': 16034, 'barot': 16035, 'newark': 16036, 'julie': 16037, 'ulises': 16038, 'resigns': 16039, 'anxiety': 16040, 'anti-semitism': 16041, 'interfaith': 16042, 'daylam': 16043, 'camilo': 16044, 'razali': 16045, '9.9': 16046, 'asher': 16047, 'maximize': 16048, 'specialized': 16049, 'hampers': 16050, 'susceptibility': 16051, 'plank': 16052, 'brook': 16053, 'lest': 16054, 'dynamics': 16055, 'hrytsenko': 16056, 'catering': 16057, 'symbolically': 16058, 'ch-47': 16059, 'jean-charles': 16060, 'foment': 16061, 'revolutions': 16062, '49.7': 16063, 'bekasi': 16064, 'okinawa': 16065, 'abhorrent': 16066, 'caring': 16067, 'gerais': 16068, 'casket': 16069, 'lori': 16070, 'tupac': 16071, 'amaru': 16072, 'keesler': 16073, 'patented': 16074, 'outrageous': 16075, 'framed': 16076, 'dishonest': 16077, 're-starting': 16078, 'unlimited': 16079, 'nesterenko': 16080, 'proceeding': 16081, 'mourino': 16082, 'tellez': 16083, 'student-led': 16084, 'whitcomb': 16085, 'servicemembers': 16086, 'manila-based': 16087, 'galleries': 16088, 'myuran': 16089, 'sukumaran': 16090, 'jalawla': 16091, 'embraced': 16092, 'antalya': 16093, 'emily': 16094, 'henochowicz': 16095, 'tear-gas': 16096, 'bartering': 16097, 'co-conspirators': 16098, 'quezon': 16099, 'emptying': 16100, 'u.s.-india': 16101, 'empowered': 16102, 'iranian-born': 16103, 'farsi': 16104, 'ak': 16105, '24,000': 16106, 'alluvial': 16107, 'hides': 16108, '70.7': 16109, '1806': 16110, 'sar': 16111, 'norm': 16112, 'polish-lithuanian': 16113, 'exodus': 16114, 'socioeconomic': 16115, 'hardened': 16116, 'unifying': 16117, 'strangle': 16118, 'thereupon': 16119, 'bother': 16120, 'sukhoi': 16121, 'mid-level': 16122, 'detailing': 16123, 'geoffrey': 16124, 'allotted': 16125, 'inked': 16126, 'pacifist': 16127, 'hangzhou': 16128, 'reinforcement': 16129, 'nations-backed': 16130, 'drought-stricken': 16131, 'laborer': 16132, 'modify': 16133, 'somber': 16134, '10-percent': 16135, 'ex-soviet': 16136, 'payload': 16137, 'fragmented': 16138, 'undemocratic': 16139, 'asylum-seekers': 16140, 'fatma': 16141, 'incentive': 16142, 'broadway': 16143, 'dolls': 16144, 'choreography': 16145, 'hafez': 16146, 'parisian': 16147, 'nanny': 16148, 'marjah': 16149, 'thefts': 16150, 'shed': 16151, 'swindler': 16152, 'mets': 16153, 'grandchild': 16154, 'white-owned': 16155, 'pneumonia-like': 16156, 'guangxi': 16157, 'naeem': 16158, 'noor': 16159, 'brace': 16160, 'payloads': 16161, 'exomars': 16162, 'flowed': 16163, 'adopting': 16164, 'consultative': 16165, 'scrapping': 16166, 'globovision': 16167, 'pendleton': 16168, 'sponsoring': 16169, 'mhz': 16170, 'abid': 16171, 'chained': 16172, 'embankment': 16173, 'anti-militant': 16174, 'mentions': 16175, '13.65': 16176, 'boots': 16177, 'accrue': 16178, '1886': 16179, '1899': 16180, 'boycotts': 16181, '4.4': 16182, 'thinned': 16183, 'dukedom': 16184, '1603': 16185, '1854': 16186, 'modernize': 16187, 'gupta': 16188, 'longed': 16189, 'devouring': 16190, 'exertions': 16191, '53-year-old': 16192, 'gentleman': 16193, 'excused': 16194, 'enlist': 16195, 'aswan': 16196, 'abul': 16197, 'appreciates': 16198, 'denunciations': 16199, 'rabin': 16200, 'theodore': 16201, 'newlands': 16202, 'chanderpaul': 16203, 'ntini': 16204, 'il-sung': 16205, 'mombasa': 16206, 'livingstone': 16207, 'lusaka': 16208, 'mediated': 16209, 'effigies': 16210, 'humidity': 16211, 'opposition-aligned': 16212, 'transmitting': 16213, 'seif': 16214, 'faulting': 16215, 'energetic': 16216, 'thales': 16217, 'six-year-old': 16218, 'cotonou': 16219, '7th': 16220, 'precautionary': 16221, 'tributes': 16222, 'eichmann': 16223, 'case-shiller': 16224, 'subprime': 16225, 'solving': 16226, 'grenoble': 16227, 'robber': 16228, 'swirling': 16229, 'kleinkirchheim': 16230, 'klammer': 16231, 'michaela': 16232, 'finishes': 16233, 'anja': 16234, 'khristenko': 16235, 'panetta': 16236, 'al-obeidi': 16237, 'lembe': 16238, 'bukavu': 16239, 'kenyon': 16240, 'tornadoes': 16241, 'reburied': 16242, 'rodrigues': 16243, 'scares': 16244, 'malachite': 16245, 'determines': 16246, 'rhode': 16247, 'tai': 16248, 'qiang': 16249, 'katzenellenbogen': 16250, 'erect': 16251, 'surpass': 16252, 'keidel': 16253, 'carnegie': 16254, 'endowment': 16255, 'republican-led': 16256, 'gauging': 16257, 'camila': 16258, 'guerra': 16259, 'singer-songwriter': 16260, 'incessant': 16261, 'ex-boyfriend': 16262, 'gypsy': 16263, 'logs': 16264, 'ry': 16265, 'cooder': 16266, 'beloved': 16267, 'clan-based': 16268, 'destiny': 16269, 'disband': 16270, 'lanes': 16271, 'bbl/day': 16272, 'jonas': 16273, 'arrears': 16274, 'borneo': 16275, 'wanderings': 16276, 'glided': 16277, 'pricked': 16278, 'useless': 16279, 'heavily-guarded': 16280, 'ashr': 16281, 'news-agency': 16282, 'hydara': 16283, '11,700': 16284, 'nine-month': 16285, 'martinez': 16286, 'balakot': 16287, 'frees': 16288, 'jade': 16289, 'third-party': 16290, 'ying-jeou': 16291, 'jeopardy': 16292, 'maharastra': 16293, 'guirassy': 16294, 'karliova': 16295, 'tectonic': 16296, 'mercenaries': 16297, 'teodoro': 16298, 'baltica': 16299, 'pillars': 16300, 'peat': 16301, 'wolves': 16302, 're-establishment': 16303, 'dutch-born': 16304, 'sates': 16305, 'cohesion': 16306, '0.1': 16307, 'two-man': 16308, 'salizhan': 16309, 'sharipov': 16310, 'leroy': 16311, 'chiao': 16312, 'branko': 16313, 'decay': 16314, 'psychic': 16315, 'co-chaired': 16316, 'kimchi': 16317, 'cabbage': 16318, 'dish': 16319, 'achin': 16320, 'sarath': 16321, 'portfolios': 16322, 'essay': 16323, 'rear': 16324, 'clocked': 16325, 'mclaren-mercedes': 16326, 'wurz': 16327, 'raikkonen': 16328, 'trails': 16329, 'third-place': 16330, 'falcon': 16331, 'drifted': 16332, 'frayed': 16333, 'shunned': 16334, 'longest-running': 16335, 'trumpet': 16336, 'daya': 16337, 'nino': 16338, 'sanha': 16339, 'bandung': 16340, 'salinas': 16341, 'edgar': 16342, 'norte': 16343, 'contrasted': 16344, 'campo': 16345, 'raiding': 16346, 'unicorp': 16347, 'cara': 16348, 'defection': 16349, 'multi-fiber': 16350, 'tariff-free': 16351, 'high-income': 16352, 'value-added': 16353, 'profited': 16354, 'petronas': 16355, 'malays': 16356, '1925': 16357, 'guarantor': 16358, 'stamping': 16359, 'neigh': 16360, 'brilliant': 16361, 'indolent': 16362, 'hog': 16363, 'cocks': 16364, 'skulked': 16365, 'crowed': 16366, 'boasting': 16367, 'marjayoun': 16368, '145.85': 16369, 'pre-paid': 16370, '1.08': 16371, 'declarations': 16372, 'markus': 16373, 'insulted': 16374, 'mujahedeen': 16375, 'morsink': 16376, 'michiko': 16377, 'saipan': 16378, 'fading': 16379, 'torrent': 16380, 'bassist': 16381, 'vista': 16382, 'acclaim': 16383, 'ruben': 16384, 'gonzalez': 16385, 'memin': 16386, 'pinguin': 16387, 'sa': 16388, 'immensely': 16389, 'offended': 16390, 'demeanor': 16391, 'conservationists': 16392, '213': 16393, 'guji': 16394, 'borena': 16395, 'shakiso': 16396, 'arero': 16397, 'yabello': 16398, '39,000': 16399, 'virunga': 16400, 'segway': 16401, 'lowers': 16402, 'ndjabu': 16403, 'dwain': 16404, 'freelance': 16405, 'rodina': 16406, 'llodra': 16407, 'koukalova': 16408, 'first-set': 16409, 'countrywoman': 16410, 'grass-court': 16411, 'johan': 16412, 'tarulovski': 16413, 'foodstuffs': 16414, 'malcolm': 16415, 'reception': 16416, 'deploys': 16417, 'conferring': 16418, 'striving': 16419, 'ashdod': 16420, 'stand-in': 16421, 'gilchrist': 16422, 'olympio': 16423, 'andiwal': 16424, 'apologizes': 16425, 'levinson': 16426, 'locating': 16427, '243': 16428, 'mar-46': 16429, 'centurion': 16430, 'anthrax': 16431, 'impacts': 16432, 'el-arish': 16433, 'dynamite': 16434, 'sderot': 16435, 'southerners': 16436, 'cutoff': 16437, 'boiled': 16438, 'reassuring': 16439, 'talent': 16440, 'quakes': 16441, 'single-day': 16442, 'beiji': 16443, 'paulos': 16444, 'faraj': 16445, 'nineveh': 16446, 'blackmailed': 16447, 'baber': 16448, 'mcdaniel': 16449, 'ecosystems': 16450, 'yokota': 16451, 'alfonso': 16452, '340': 16453, 'logos': 16454, 'detonators': 16455, '138': 16456, 'arsenic': 16457, 'diver': 16458, 'conn.': 16459, 'adjustments': 16460, 'stimulated': 16461, 'longer-term': 16462, 'ushering': 16463, 'indochina': 16464, 'hardships': 16465, 'semblance': 16466, 'norodom': 16467, 'aboriginal': 16468, '1770': 16469, 'capt.': 16470, 'al-alam': 16471, 'qom': 16472, 'ascension': 16473, 'perceiving': 16474, 'deprivation': 16475, 'spider-man': 16476, 'box-office': 16477, 'thriller': 16478, 'grossing': 16479, 'ideological': 16480, 'gossip': 16481, 'hamilton': 16482, 'hayek': 16483, 'compatible': 16484, 'volt': 16485, 'f-16s': 16486, 'manual': 16487, 'manipur': 16488, 'defenseless': 16489, 'athlete': 16490, 'gayle': 16491, 'internationals': 16492, 'occured': 16493, 'penetrate': 16494, 'balkenende': 16495, 'dragons': 16496, 'torchbearer': 16497, 'constructively': 16498, 'jowhar': 16499, 'medusa': 16500, 'maksim': 16501, 'awakening': 16502, 'sexuality': 16503, 'examines': 16504, 'zheng': 16505, 'ads': 16506, 'denuclearization': 16507, 'donkey': 16508, 'unanswered': 16509, 'ulf': 16510, 'henricsson': 16511, 'claudio': 16512, 'tenet': 16513, 'unwarranted': 16514, 'deactivated': 16515, 'nitrate': 16516, 'nikiforov': 16517, 'staffing': 16518, 'counter-revolutionary': 16519, 'madhav': 16520, 'pre-recorded': 16521, 'day-to-day': 16522, 'yadana': 16523, 'realtors': 16524, 'helgesen': 16525, 'mackay': 16526, 'contrasts': 16527, 'sukhumi': 16528, 'disbursing': 16529, 'millenium': 16530, 'protectionist': 16531, 'trousers': 16532, 'knit': 16533, 'underwear': 16534, 're-examine': 16535, 'asia-europe': 16536, 'vedomosti': 16537, 'muifa': 16538, 'ngai': 16539, 'thua': 16540, 'thien': 16541, 'hue': 16542, 'ruling-party': 16543, 'bonfoh': 16544, 'muslim-christian': 16545, 'offending': 16546, 'b.': 16547, 'hadson': 16548, 'melnichenko': 16549, 'unwillingness': 16550, 'couch': 16551, 'enthusiasm': 16552, 'echo': 16553, 'tellers': 16554, 'whispers': 16555, 'replies': 16556, 'breeze': 16557, '126': 16558, 'discriminatory': 16559, 'israel-palestinian': 16560, 'enshrines': 16561, 'opposition-led': 16562, '131': 16563, 'al-hayat': 16564, 'pan-arab': 16565, 'guy': 16566, 'gazans': 16567, 'streamed': 16568, 'fostering': 16569, 'yousaf': 16570, 'sizably': 16571, 'alerts': 16572, 'lobbied': 16573, 'velach': 16574, "d'affaires": 16575, 'mtetwa': 16576, 'gulu': 16577, 'cooperated': 16578, 'ghalib': 16579, 'kubba': 16580, 'investigates': 16581, 'haiphong': 16582, 'benedetti': 16583, 'blige': 16584, 'pasadena': 16585, 'naacp': 16586, 'csw': 16587, 'u.s.-controlled': 16588, 'laid-off': 16589, 'chiang': 16590, 'whitney': 16591, 'surprisingly': 16592, 'biennial': 16593, 'jumbo': 16594, 'moscow-backed': 16595, 'architecture': 16596, 'minarets': 16597, 'hatem': 16598, 'budgets': 16599, 'hovered': 16600, 'unjustly': 16601, 'traumatic': 16602, '655': 16603, 'understandable': 16604, 'reside': 16605, 'arghandab': 16606, 'feith': 16607, 'scheuer': 16608, 'inflamed': 16609, 'scouts': 16610, 'blacklist': 16611, 'sulphur': 16612, 'fast-food': 16613, 'baqubah': 16614, 'gillard': 16615, '1.76': 16616, 'in-store': 16617, 'four-fifths': 16618, 'shiloh': 16619, 'harbors': 16620, 'second-most': 16621, 'enactment': 16622, '1891': 16623, 'dpp': 16624, '1888': 16625, '1900': 16626, 'strolling': 16627, 'lofty': 16628, 'quoth': 16629, 'morsel': 16630, 'seldom': 16631, 'drawer': 16632, "'d": 16633, 'recruiter': 16634, 'mules': 16635, 'one-ninth': 16636, 'hitched': 16637, 'trawlers': 16638, 'chancellor-designate': 16639, 'mid-november': 16640, 'elvis': 16641, 'dancer': 16642, 'magerovskiy': 16643, "ba'ath": 16644, 'transmitter': 16645, 'rages': 16646, 'abdul-aziz': 16647, 'nation-wide': 16648, 'valdes': 16649, 'nico': 16650, 'donatella': 16651, 'versace': 16652, 'syndicated': 16653, 'insider': 16654, 'gaunt': 16655, 'ferrero': 16656, 'second-biggest': 16657, 'mishandled': 16658, 'batumi': 16659, 'louise': 16660, 'araque': 16661, 'kiryenko': 16662, 'shurja': 16663, 'kolkata': 16664, 'wasim': 16665, 'v.v.s.': 16666, 'keeper': 16667, 'mahendra': 16668, 'looted': 16669, 'yorkers': 16670, 'nathalie': 16671, 'dechy': 16672, 'marrero': 16673, 'bartoli': 16674, 'pascual': 16675, 'three-and-a-half': 16676, '537': 16677, 'cheating': 16678, 'ginepri': 16679, 'muller': 16680, 'paradorn': 16681, 'h7': 16682, 'congratulatory': 16683, 'ultra-nationalist': 16684, 'quagga': 16685, 'hunted': 16686, 'endeavor': 16687, 'impeach': 16688, 'yearlong': 16689, 'zabol': 16690, 'oils': 16691, 'supper': 16692, 'apostles': 16693, 'resurrection': 16694, 'u.s.-mexico': 16695, 'booth': 16696, 'almaty': 16697, 'likud-led': 16698, 'myna': 16699, 'antoine': 16700, 'flavia': 16701, 'patty': 16702, 'schnyder': 16703, 'arraigned': 16704, 'e.u.': 16705, 'gary': 16706, 'waivers': 16707, 'distributes': 16708, 'principalities': 16709, 'gorbachev': 16710, 'inadvertently': 16711, 'splintered': 16712, 'post-soviet': 16713, 'antigua': 16714, 'bedding': 16715, 'hemispheric': 16716, 'whelp': 16717, 'collectors': 16718, 'proxy': 16719, 'hudur': 16720, 'government-allied': 16721, 'masorin': 16722, 'pro-reform': 16723, 'registers': 16724, 'moin': 16725, 'qalibaf': 16726, 'bartholomew': 16727, 'insurmountable': 16728, 'crusaders': 16729, 'najam': 16730, 'most-wanted': 16731, 'pro-chavez': 16732, 'once-a-day': 16733, 'briston-myers': 16734, 'gilead': 16735, 'collaborate': 16736, 'left-of-center': 16737, 'harvard': 16738, 'yaqoob': 16739, 'ne': 16740, 'artifacts': 16741, 'servant': 16742, 'coffins': 16743, 'jurist': 16744, 'asfandyar': 16745, 'idle': 16746, 'nationalizing': 16747, 'black-market': 16748, 'maternity': 16749, 'eight-time': 16750, '09-feb': 16751, 'garnering': 16752, 'national-level': 16753, 'appointees': 16754, 'caldwell': 16755, 'electorate': 16756, 'muscles': 16757, 'delight': 16758, 'helmets': 16759, 'chelyabinsk': 16760, 'rounding': 16761, 'doe': 16762, 'misfortune': 16763, 'invasive': 16764, 'uterine': 16765, 'fibroid': 16766, 'dave': 16767, 'heineman': 16768, 'saifullah': 16769, 'belongings': 16770, 'uproot': 16771, 'probation': 16772, 'unify': 16773, 'sii-sii': 16774, '2,600': 16775, 'profane': 16776, 'cetin': 16777, 'rotation': 16778, '8,000-strong': 16779, 'pen': 16780, 'upside': 16781, 'inspecting': 16782, 'decorated': 16783, 'ohio-based': 16784, '1912': 16785, 'partisans': 16786, 'deficiencies': 16787, '1856': 16788, '1878': 16789, 'ceausescu': 16790, 'draconian': 16791, 'prgf': 16792, 'nomadic': 16793, '1916': 16794, 'long-lasting': 16795, 'spraying': 16796, 'mosquitoes': 16797, 'ziemer': 16798, 'comedian': 16799, '1938': 16800, 'yonts': 16801, 'ambulance': 16802, 'kusadasi': 16803, 'tak': 16804, 'al-qassam': 16805, 'ravaging': 16806, 'spawned': 16807, 'influencing': 16808, 'rezayee': 16809, 'shaer': 16810, 'booking': 16811, '4,200': 16812, 'merrill': 16813, 'lynch': 16814, 'brokerage': 16815, 'sixty': 16816, 'memories': 16817, 'hartmann': 16818, 'hamdan': 16819, 'turboprop': 16820, 'chant': 16821, '53-member': 16822, 'poverty-stricken': 16823, 'seventh-seeded': 16824, 'philipp': 16825, 'coerced': 16826, 'totals': 16827, 'arid': 16828, 'razed': 16829, 'safehouse': 16830, 'kambakhsh': 16831, 'dissolving': 16832, 'shoreline': 16833, 'spanned': 16834, 'tito': 16835, 'rhythm': 16836, 'beached': 16837, 'gift-giving': 16838, 'martti': 16839, 'trillions': 16840, 'bets': 16841, 'bathrooms': 16842, 'alerting': 16843, 'greens': 16844, '22.5': 16845, 'nigerians': 16846, 'momcilo': 16847, 'cinema': 16848, 'downgrading': 16849, 'peers': 16850, 'tipped': 16851, '0.6': 16852, 'revamp': 16853, 'uneasy': 16854, 'stimulating': 16855, 'consequently': 16856, 'incorporate': 16857, 'awaits': 16858, 'polynesia': 16859, 'distinguished': 16860, 're-export': 16861, 'deficient': 16862, 'shipwrecked': 16863, 'awoke': 16864, 'plow': 16865, 'cd': 16866, 'scrambled': 16867, 'superior': 16868, 'reconvene': 16869, 'hostess': 16870, 'fadillah': 16871, 'probability': 16872, 'akmatbayev': 16873, 'substation': 16874, 'chubais': 16875, 'wesley': 16876, 'legg': 16877, 'mason': 16878, '86th': 16879, 'querrey': 16880, 'first-serve': 16881, 'nightfall': 16882, '6,400': 16883, 'bat': 16884, 'muqdadiyah': 16885, 'converge': 16886, 'kazakh': 16887, 'assassins': 16888, 'ancestral': 16889, 'insolent': 16890, 'dervis': 16891, 'eroglu': 16892, 'hinge': 16893, 'tillman': 16894, 'culprit': 16895, 'sampled': 16896, 'demolitions': 16897, 'chepas': 16898, 'el-mallahi': 16899, '255': 16900, 'tendulkar': 16901, 'dinesh': 16902, 'doda': 16903, 'hans': 16904, 'schlegel': 16905, 'krajina': 16906, 'smear': 16907, 'pro-tibet': 16908, 'pol-e-charkhi': 16909, 'raging': 16910, 'mourns': 16911, 're-taking': 16912, 'pro-aristide': 16913, 'shia': 16914, 'disturbia': 16915, 'penguin': 16916, 'surf': 16917, 'plassnik': 16918, 'pluralistic': 16919, 'nad': 16920, 'heartland': 16921, 'sud': 16922, 'orellana': 16923, 'seven-day': 16924, 'condoms': 16925, 'firestorm': 16926, 'homily': 16927, 'qaida-linked': 16928, 'dangerously': 16929, 'doboj': 16930, 'streptococcus': 16931, 'suis': 16932, '174': 16933, 'mid-january': 16934, 'write-off': 16935, 'timed': 16936, 'millennia': 16937, 'cheonan': 16938, 'coat': 16939, 'marxist-leninist': 16940, 'jagan': 16941, 'unequivocally': 16942, 'chiluba': 16943, 'usd': 16944, 'placard': 16945, 'shopkeeper': 16946, 'salesman': 16947, 'pozarevac': 16948, 'milorad': 16949, 'mash': 16950, 'vandalizing': 16951, 'shebaa': 16952, 'grossed': 16953, 'themes': 16954, 'agim': 16955, 'lutfi': 16956, 'angering': 16957, 'bargal': 16958, 'dams': 16959, 'encryption': 16960, 'bharatiya': 16961, 'thanking': 16962, 'quake-stricken': 16963, 'al-bolani': 16964, 'al-waili': 16965, 'aigle': 16966, 'azur': 16967, 'non-military': 16968, 'anti-crime': 16969, 'stifling': 16970, 'robben': 16971, 'flame-throwers': 16972, 'writings': 16973, 'centrifuge': 16974, 'evangelist': 16975, 'wholly': 16976, 'far-left': 16977, 'declassified': 16978, 'assertions': 16979, 'godfather': 16980, 'sensible': 16981, 'sassou': 16982, 'anti-torture': 16983, 'statutes': 16984, 'definition': 16985, 'interstate': 16986, 'fingerprint': 16987, 'fingerprinting': 16988, 'trunk': 16989, 'sufficiently': 16990, 'susy': 16991, 'tekunan': 16992, 'quest': 16993, 'recycled': 16994, 'baking': 16995, 'unfriendly': 16996, 'deserters': 16997, 'thriving': 16998, 'pre-world': 16999, 're-opening': 17000, 'hifikepunye': 17001, 'windhoek': 17002, 'swapo': 17003, 'canals': 17004, 'irrawaddy': 17005, 'exorbitant': 17006, '3,400': 17007, 'violence-free': 17008, '7.00': 17009, 'kph': 17010, 'peoria': 17011, 'gayoom': 17012, 'nasheed': 17013, 'elevation': 17014, 'two-and-a-half-year': 17015, 'tsz': 17016, 'eritrea-ethiopia': 17017, 'remotely': 17018, 'demarcated': 17019, 'coordinates': 17020, 'tracts': 17021, 'frg': 17022, 'gdr': 17023, 'intervening': 17024, 'boar': 17025, 'vultures': 17026, 'crows': 17027, 'gored': 17028, 'vanallen': 17029, 'wead': 17030, 'apr-28': 17031, 'triangular': 17032, '3rd': 17033, 'velupillai': 17034, 'lakhdaria': 17035, 'fomenting': 17036, 'kerch': 17037, 'left-leaning': 17038, 'nashville': 17039, 'wealth-sharing': 17040, 'yekiti': 17041, 'maashouq': 17042, 'kameshli': 17043, 'encountering': 17044, 'converts': 17045, '18.5': 17046, 'bidder': 17047, 'jaafari': 17048, 'defining': 17049, '9.6': 17050, 'fade': 17051, 'comfort': 17052, 'party-goers': 17053, 'lags': 17054, '646': 17055, 'calin': 17056, 'tariceanu': 17057, 'siirt': 17058, 'scanners': 17059, 'item': 17060, 'massed': 17061, 'trainees': 17062, 'cylinders': 17063, 'macapagal': 17064, 'everybody': 17065, 'clots': 17066, 'secretariat': 17067, 'francesca': 17068, 'chess': 17069, 'ex-prime': 17070, 'liberals': 17071, 'emomali': 17072, 'rakhmon': 17073, 'finisher': 17074, 'ultra-conservative': 17075, 'mexicans': 17076, 'reorganization': 17077, 'unsecured': 17078, 'reorganized': 17079, 'effected': 17080, 'dioceses': 17081, '1825': 17082, 'consisted': 17083, 'deep-seated': 17084, 'amerindian': 17085, 'bubble': 17086, 'sicily': 17087, 'eec': 17088, 'conserving': 17089, 'choked': 17090, 'at&t': 17091, 'resigning': 17092, 'instituting': 17093, 'negate': 17094, 'firings': 17095, 'karasin': 17096, 'toussaint': 17097, 'sfeir': 17098, 'unfiltered': 17099, 'click': 17100, 'ignited': 17101, 'bilis': 17102, 'amsterdam': 17103, 'h': 17104, 'korea-based': 17105, 'hyundai': 17106, 'kifaya': 17107, 'forgery': 17108, '2,11,000': 17109, '1,90,000': 17110, 'misquoting': 17111, 'erected': 17112, 'postsays': 17113, 'merging': 17114, 'nizar': 17115, 'trabelsi': 17116, '361': 17117, 'vasil': 17118, 'instructors': 17119, 'ivashov': 17120, 'duc': 17121, 'luong': 17122, 'jean-max': 17123, 'chipmaker': 17124, 'golborne': 17125, 'shafts': 17126, 'precedes': 17127, 'purple': 17128, 'beads': 17129, 'batlle': 17130, 'drained': 17131, 'sodium': 17132, 'escorts': 17133, 'simmering': 17134, 'blueprints': 17135, 'low-key': 17136, 'risked': 17137, 'kandani': 17138, 'miyazaki': 17139, 'pajhwok': 17140, 'blechschmidt': 17141, 'consolidating': 17142, 'yulija': 17143, 'kosachev': 17144, '76-year-old': 17145, 'awake': 17146, 'malignant': 17147, 'jon': 17148, 'litigation': 17149, 'eparses': 17150, 'integral': 17151, 'kerguelen': 17152, 'ile': 17153, 'satisfying': 17154, 'kenyatta': 17155, 'fractured': 17156, 'dislodge': 17157, 'narc': 17158, 'lighthouse': 17159, 'qar': 17160, 'solvent': 17161, 'vakhitov': 17162, 'korans': 17163, 'drifting': 17164, 'sara': 17165, 'mathew': 17166, 'surveying': 17167, 'chelsea': 17168, 'cows': 17169, 'seven-nation': 17170, 'lawbreakers': 17171, 'hotmail': 17172, '336': 17173, 'su': 17174, 'hon': 17175, 'okah': 17176, 'norway-based': 17177, 'bisphenol': 17178, 'liver': 17179, 'rachid': 17180, 'oxygen': 17181, '36th': 17182, 'michaella': 17183, 'wessels': 17184, 'kiefer': 17185, 'dent': 17186, 'preservation': 17187, 'grouped': 17188, 'unnerved': 17189, 'westinghouse': 17190, '157': 17191, 'préval': 17192, 'mobutu': 17193, 'legitimizing': 17194, 'asadullah': 17195, 'shiekh': 17196, 'alcantara': 17197, 'technicians': 17198, 'pervasive': 17199, 'depict': 17200, 'sa-18': 17201, 'iskander-e': 17202, 'urumqi': 17203, 'sleek': 17204, 'magnetic': 17205, 'rental': 17206, 'borrower': 17207, '8.4': 17208, 'paige': 17209, 'kollock': 17210, 'chester': 17211, 'gunsmoke': 17212, 'qualifications': 17213, 'emmy': 17214, 'taher': 17215, 'busta': 17216, 'uncooperative': 17217, 'recreational': 17218, 'amateur': 17219, 'inaccurate': 17220, 'staunchly': 17221, 'freight': 17222, 'overfishing': 17223, 'mutineers': 17224, 'tahitian': 17225, 'vestige': 17226, 'home-grown': 17227, 'qarase': 17228, 'bainimarama': 17229, 'cpa': 17230, 'hounds': 17231, '1911': 17232, 'hani': 17233, 'calabar': 17234, 'kirilenko': 17235, 'seemingly': 17236, 'beachside': 17237, 'exciting': 17238, 'surfers': 17239, 'elham': 17240, 'fundamentally': 17241, 'sepp': 17242, 'blatter': 17243, 'trim': 17244, 'kissing': 17245, 'looped': 17246, 'troedsson': 17247, 'tends': 17248, 'au-sponsored': 17249, 'egyptian-owned': 17250, 'alloceans': 17251, 'smigun': 17252, 'snowboard': 17253, 'sheibani': 17254, 'saderat': 17255, 'cut-off': 17256, 'jabalya': 17257, 'arthritis': 17258, 'surface-to-air': 17259, 'akash': 17260, 'chandipur': 17261, 'bhubaneshwar': 17262, 'drinan': 17263, 'congestive': 17264, 'unsure': 17265, 'shaking': 17266, 'stratfor': 17267, 'turkestan': 17268, 'adumin': 17269, 'westernmost': 17270, 'multi-million-dollar': 17271, 'rocking': 17272, 'githongo': 17273, 'asghar': 17274, 'everywhere': 17275, 'abdulmutallab': 17276, 'cloning': 17277, 'woo-suk': 17278, 'deceiving': 17279, 'scuffles': 17280, 'wengen': 17281, '1.06': 17282, 'practically': 17283, 'plates': 17284, 'ganei': 17285, 'oic': 17286, 'caved': 17287, 'drastic': 17288, 'tyre': 17289, 'sit-in': 17290, 'low-grade': 17291, 'lanterns': 17292, 'souls': 17293, 'horrific': 17294, '8.15': 17295, 'janakpur': 17296, 'delhi-based': 17297, 'al-qaida-inspired': 17298, 'rajapaksa': 17299, 'visual': 17300, 'apiece': 17301, 'reaped': 17302, 'fiscally': 17303, 'attaining': 17304, '1857': 17305, 'biodiversity': 17306, '1903': 17307, 'alfredo': 17308, 'celtic': 17309, 'rebellions': 17310, 'ulster': 17311, 'maiberger': 17312, 'smelter': 17313, 'politicizing': 17314, 'groundwork': 17315, 'feminine': 17316, 'mystique': 17317, 'groundbreaking': 17318, 'husbands': 17319, 'goushmane': 17320, 'twin-engine': 17321, '720': 17322, 'mindoro': 17323, 'albay': 17324, 'mayon': 17325, 'loaning': 17326, 'paves': 17327, 'blockades': 17328, 'eugene': 17329, 'prescribed': 17330, 'iranian-americans': 17331, 'hyde': 17332, 'irrigation': 17333, 'altitudes': 17334, 'rainstorms': 17335, 'napa': 17336, 'swelling': 17337, 'indict': 17338, 'self-ruled': 17339, 'elusive': 17340, 'persecuted': 17341, 'widely-used': 17342, 'dyes': 17343, 'conceal': 17344, 'kuril': 17345, '561': 17346, 'taxpayers': 17347, 'aia': 17348, 'hakkari': 17349, 'wilmington': 17350, 'graduation': 17351, 'benita': 17352, 'breaststroke': 17353, 'verez-bencomo': 17354, 'meningitis': 17355, 'midler': 17356, 'celine': 17357, 'dion': 17358, 'headliner': 17359, '61-year-old': 17360, 'inaugurating': 17361, 'colosseum': 17362, 'vaile': 17363, 'vuvuzelas': 17364, 'robots': 17365, 'lighter': 17366, 'prison-like': 17367, 'chikunova': 17368, 'u.n.-administered': 17369, 'danforth': 17370, 'chesnot': 17371, 'malbrunot': 17372, 'wazirstan': 17373, 'accessing': 17374, 'rak': 17375, 'uninterrupted': 17376, 'unload': 17377, 'earners': 17378, 'nine-tenths': 17379, 'genius': 17380, 'correctness': 17381, 'co-chairs': 17382, 'saiki': 17383, 'bwakira': 17384, 'sherzai': 17385, 'weed': 17386, 'bribed': 17387, 'beds': 17388, '162-7': 17389, 'fanning': 17390, '284-6': 17391, 'kapilvastu': 17392, 'directing': 17393, 'prostitution': 17394, 'haile': 17395, 'politicize': 17396, 'mutilating': 17397, 'three-judge': 17398, 'arak': 17399, 'ensures': 17400, 're-establishing': 17401, 'torah': 17402, 'overcrowding': 17403, 'almost-daily': 17404, 'anglo': 17405, 'tine': 17406, 'six-thousand': 17407, 'deshu': 17408, 'kakar': 17409, 'mujaddedi': 17410, 'cpp': 17411, 'ilbo': 17412, 'open-ended': 17413, 'erroneous': 17414, 'forwarded': 17415, 'rosh': 17416, 'hashanah': 17417, 'strangling': 17418, 'icelandic': 17419, 'kye': 17420, 'stanford': 17421, 'chesney': 17422, 'sub-freezing': 17423, 'chill': 17424, 'clarence': 17425, 'gatemouth': 17426, 'shmatko': 17427, 'malda': 17428, 're-established': 17429, 'tyranny': 17430, 'levan': 17431, 'gachechiladze': 17432, 'flowstations': 17433, 'chevrontexaco': 17434, 'eindhoven': 17435, 'fargo': 17436, 'squatters': 17437, 'plainclothes': 17438, 'afghan-international': 17439, 'dushanbe': 17440, 'seafood': 17441, 'cake': 17442, 'rechargeable': 17443, 'volume': 17444, 'exterminated': 17445, '1834': 17446, 'geographically': 17447, 'ethanol': 17448, 'expiration': 17449, 'statist': 17450, 'workshops': 17451, '41st': 17452, 'geography': 17453, 'viceroyalty': 17454, '1822': 17455, 'offspring': 17456, 'handsomest': 17457, 'tenderness': 17458, 'allot': 17459, 'earthquake-devastated': 17460, 'scattering': 17461, 'demobilize': 17462, 'aiff': 17463, '58.28': 17464, 'refine': 17465, 'reparation': 17466, 'militarism': 17467, 'g-7': 17468, 'stumbling': 17469, 'divers': 17470, 'bancoro': 17471, 'liquidity': 17472, 'texas-based': 17473, '538': 17474, 'fadilah': 17475, 'payne': 17476, 'receptionist': 17477, 'occupiers': 17478, 'semi-final': 17479, 'confrontational': 17480, 'naisse': 17481, '34-nation': 17482, '69,000': 17483, 'tap': 17484, 'lashing': 17485, 'mr.yushchenko': 17486, 'massacres': 17487, 'zahedan': 17488, 'novelist': 17489, 'booker': 17490, 'orphaned': 17491, 'vidoje': 17492, 'jundollah': 17493, 'adapt': 17494, 'sistan-baluchistan': 17495, 'ogun': 17496, 'torun': 17497, 'revisit': 17498, 'barley': 17499, 'cauvin': 17500, 'ojea': 17501, 'quintana': 17502, 'hogan': 17503, 'clearwater': 17504, 'shelor': 17505, 'graziano': 17506, 'joey': 17507, 'emphasizes': 17508, 'samoa': 17509, 'nkurunziza': 17510, 'overland': 17511, 'disgrace': 17512, '122.73': 17513, 'one-on-one': 17514, 'commandoes': 17515, 'mauled': 17516, 'beings': 17517, 'fini': 17518, 'koichi': 17519, 'wakata': 17520, 'patterns': 17521, 'crossroads': 17522, 'philosophical': 17523, 'claire': 17524, 'fulfills': 17525, 'hafeez': 17526, 'heir': 17527, 'first-born': 17528, 'ascend': 17529, 'chrysanthemum': 17530, 'rosa': 17531, 'hurricane-damaged': 17532, 'floats': 17533, 'climatic': 17534, 'flood-stricken': 17535, 'slutskaya': 17536, 'skaters': 17537, 'entrants': 17538, 'elena': 17539, 'reform-minded': 17540, 'technocrats': 17541, 'creativity': 17542, 'distinguish': 17543, 'ronaldo': 17544, 'self-reliance': 17545, 'roseneft': 17546, 'fiercest': 17547, 'calcutta': 17548, 'ghettos': 17549, 'front-runners': 17550, 'diddy': 17551, 'mogul': 17552, 'salvage': 17553, 'bosco': 17554, 'katutsi': 17555, 'kulyab': 17556, 'ordinance': 17557, 'subsidizes': 17558, 'peanuts': 17559, 'boni': 17560, 'centrally': 17561, 'dampened': 17562, 'wool': 17563, 'dane': 17564, 'eco-tourism': 17565, 'jackdaw': 17566, 'envy': 17567, 'whir': 17568, 'daw': 17569, 'assented': 17570, 'sowing': 17571, 'hopping': 17572, 'despised': 17573, 'pausing': 17574, 'nixon': 17575, 'panhandle': 17576, 'hallandale': 17577, 'boating': 17578, 'inscribed': 17579, 'timoshenko': 17580, 'tut': 17581, 'taif': 17582, 'stephanie': 17583, 'breaching': 17584, 'argentines': 17585, 'harrison': 17586, '3-0': 17587, 'takers': 17588, 'al-bahlul': 17589, 'diminish': 17590, 'backpacks': 17591, 'luton': 17592, 'ramazan': 17593, 'sami': 17594, 'understood': 17595, 'novin': 17596, 'mesbah': 17597, 'high-quality': 17598, 'douglas': 17599, 'badakshan': 17600, '186': 17601, 'sinmun': 17602, 'somaliland': 17603, 'renovations': 17604, 'reauthorize': 17605, 'redeployment': 17606, 'hezb-e-islami': 17607, 'hamas-ruled': 17608, 'olympia': 17609, 'patan': 17610, 'stanishev': 17611, 'odom': 17612, '181': 17613, 'diverting': 17614, 'embolden': 17615, 'fushun': 17616, 'rob': 17617, 'flagship': 17618, 'frazier': 17619, 'heist': 17620, 'discretion': 17621, 'malegaon': 17622, 'minivan': 17623, 'undocumented': 17624, 'rizeigat': 17625, 'aliu': 17626, 'hinges': 17627, 'sariyev': 17628, 'soro': 17629, 'tax-free': 17630, 'annapolis': 17631, 'languishing': 17632, 'bulky': 17633, 'smiling': 17634, 'creator': 17635, 'wag': 17636, '34.5': 17637, 'shahidi': 17638, 'annulment': 17639, '7-billion': 17640, 'dormant': 17641, '82,000': 17642, 'uninsured': 17643, 'bales': 17644, 'pastures': 17645, 'roxana': 17646, 'engineered': 17647, 'detached': 17648, 'chairwoman': 17649, 'casings': 17650, 'apache': 17651, 'yankee': 17652, 'frontline': 17653, 'pbs': 17654, 'katyusha': 17655, 'government-ordered': 17656, 'one-month': 17657, 'arc': 17658, 'nuri': 17659, 'potent': 17660, 'inquiries': 17661, 'interpretations': 17662, 'surroundings': 17663, 'lunchtime': 17664, 'excrement': 17665, 'preachers': 17666, 'motivation': 17667, 'revelry': 17668, 'rancher': 17669, 'apondi': 17670, '46-member': 17671, 'maastricht': 17672, '1697': 17673, '1804': 17674, 'postponements': 17675, 'grievously': 17676, 'shake': 17677, 'devoured': 17678, 'surfeited': 17679, 'nursing': 17680, '7-eleven': 17681, 'chains': 17682, 'kezerashvili': 17683, 'sahibzada': 17684, 'anis': 17685, 'kanow': 17686, 'back-up': 17687, 'clinched': 17688, 'well-being': 17689, '203': 17690, 'skied': 17691, 'ivica': 17692, 'darkazanli': 17693, 'entrepreneur': 17694, 'unfolding': 17695, 'revoking': 17696, 'undermines': 17697, 'ibaraki': 17698, 'sabawi': 17699, 'al-hasan': 17700, 'jvp': 17701, 'bleak': 17702, 'unhelpful': 17703, 'swastika': 17704, 'al-tamimi': 17705, 'rowsch': 17706, 'ustarkhanova': 17707, 'inflame': 17708, 'unprofessional': 17709, 'envisions': 17710, 'tauran': 17711, 'pluralism': 17712, 'nita': 17713, 'explosives-packed': 17714, 'extinct': 17715, 'loosened': 17716, 'ratsiraka': 17717, 'lessened': 17718, '03-apr': 17719, 'preserved': 17720, 'melody': 17721, 'stool': 17722, 'newell': 17723, 'letten': 17724, 'kenjic': 17725, 'petitioning': 17726, 'pitted': 17727, 'amezcua': 17728, 'minds': 17729, 'sardar': 17730, 'barometer': 17731, 'georgians': 17732, 'gregorio': 17733, 'alec': 17734, 'chita': 17735, '\x93': 17736, 'handcuffed': 17737, 'bands': 17738, 'discharge': 17739, 'bartlett': 17740, 'spiked': 17741, 'haroon': 17742, 'paltrow': 17743, 'folha': 17744, 'fraenzi': 17745, 'infromation': 17746, 'nutritional': 17747, 'blanket': 17748, 'sprawling': 17749, 'expatriates': 17750, 'fisher': 17751, 'tributary': 17752, 'yangtze': 17753, 'vest': 17754, 'ncri': 17755, '55th': 17756, '2,10,000': 17757, 'baloyi': 17758, 'hip-hop': 17759, 'jean-pierre': 17760, '56-year-old': 17761, 'yearend': 17762, 'intifada': 17763, 'oak': 17764, 'hearty': 17765, 'organ': 17766, 'common-sense': 17767, 'scorn': 17768, 'tired': 17769, 'partitioning': 17770, 'pre-storm': 17771, 'machetes': 17772, 'haram': 17773, 'ghani': 17774, 'phases': 17775, 'undetected': 17776, 'debriefed': 17777, 'zenani': 17778, 'computing': 17779, 'browsing': 17780, 'video-conferencing': 17781, 'gadget': 17782, 'conson': 17783, 'floodwalls': 17784, 'berkeley': 17785, 'minke': 17786, 'bin-laden': 17787, 'anesthetic': 17788, 'bharti': 17789, 'shopkeepers': 17790, 'unaffected': 17791, 'ethem': 17792, 'erdagi': 17793, 'horses': 17794, 'amazed': 17795, 'deception': 17796, 'iranian-trained': 17797, 'military-style': 17798, 'military-police': 17799, 'sangakkara': 17800, 'divisive': 17801, 'pregnancies': 17802, 'roe': 17803, 'legalizing': 17804, 'sha': 17805, 'non-violence': 17806, 'hailu': 17807, 'ammar': 17808, 'royalty': 17809, 'irreparable': 17810, 'v.': 17811, 'ninth-century': 17812, 'al-muntadar': 17813, 'nhk': 17814, 'tsa': 17815, 'grassroots': 17816, 'overshadows': 17817, 'enroll': 17818, 'subjecting': 17819, 'commons': 17820, 'births': 17821, 'muntazer': 17822, 'al-zaidi': 17823, 'invalid': 17824, '1919': 17825, 'short-lived': 17826, 'polarized': 17827, 'soaked': 17828, 'faint': 17829, 'scampered': 17830, 'woe': 17831, 'overtime': 17832, 'quran': 17833, 'alam': 17834, 'terry': 17835, 'rath': 17836, 'defamatory': 17837, 'nenbutsushu': 17838, 'mobs': 17839, 'waxman': 17840, 'suharto': 17841, 'viewing': 17842, 'two-million': 17843, 'unopposed': 17844, 'aes': 17845, 'breakdowns': 17846, 'emperors': 17847, 'heirs': 17848, '450-member': 17849, 'trusted': 17850, 'sinopec': 17851, 'vitoria': 17852, 'bribes-for-votes': 17853, 'datafolha': 17854, 'zoran': 17855, 'milenkovic': 17856, 'covic': 17857, 'baluyevsky': 17858, 'clyburn': 17859, 'dusan': 17860, 'tesic': 17861, 'hargreaves': 17862, 'ranges': 17863, 'edouard': 17864, 'cautions': 17865, 'mountainside': 17866, 'stupid': 17867, 'marriages': 17868, 'gujarat': 17869, 'diseased': 17870, 'tucson': 17871, 'nabbed': 17872, 'garner': 17873, 'obey': 17874, 'denials': 17875, 'adherence': 17876, 'cfa': 17877, 'jeopardized': 17878, 'breadfruit': 17879, 'spoil': 17880, 'modestly': 17881, 'heap': 17882, 'misfortunes': 17883, 'yours': 17884, 'bell': 17885, 'protocols': 17886, 'giddens': 17887, 'barman': 17888, 'naqba': 17889, 'counterinsurgency': 17890, 'ravix': 17891, 'makhachkala': 17892, 'returnees': 17893, 'incapacitated': 17894, 'moviemaking': 17895, 'reinforces': 17896, 'tarnishing': 17897, 'brain-wasting': 17898, 'abu-hafs': 17899, 'three-vehicle': 17900, 'heater': 17901, 'arg': 17902, 'kakdwip': 17903, '225-member': 17904, 'methodology': 17905, 'attache': 17906, 'exploding': 17907, 'escapees': 17908, 'resorting': 17909, 'kohlu': 17910, 'passover': 17911, 'motivating': 17912, 'deir': 17913, 'caterpillar': 17914, 'vacationers': 17915, '6.2': 17916, 'pago': 17917, 'gardener': 17918, 'watered': 17919, 'swamp': 17920, 'alarming': 17921, 'chemist': 17922, 'coupled': 17923, 'bertone': 17924, 'mudasir': 17925, 'notoriously': 17926, 'sarin': 17927, 'informant': 17928, '26th': 17929, 'invaders': 17930, 'frisked': 17931, '17-month': 17932, 'communicate': 17933, 'conscience': 17934, 'curbs': 17935, 'literacy': 17936, 'refueling': 17937, 'stakic': 17938, 'warranted': 17939, 'kharazi': 17940, 'civility': 17941, 'reconstructive': 17942, 'in-laws': 17943, 'midterm': 17944, 'answers': 17945, 'activated': 17946, '55-member': 17947, 'wwf': 17948, 'eighties': 17949, 'sever': 17950, 'deserter': 17951, 'townspeople': 17952, 'sibneft': 17953, 'nomads': 17954, 'comatose': 17955, 'unconscious': 17956, 'initiating': 17957, 'maybe': 17958, 'archery': 17959, 'year-round': 17960, 'dreams': 17961, 'enclosed': 17962, 'bray': 17963, 'linguistic': 17964, '1839': 17965, 'bolshevik': 17966, 'ferdinand': 17967, 'edsa': 17968, 'corazon': 17969, 'macapagal-arroyo': 17970, 'kowal': 17971, 'franken': 17972, 'jung': 17973, 'layoffs': 17974, 'et': 17975, "terre'blanche": 17976, 'stemmed': 17977, 'communal': 17978, 'poroshenko': 17979, 'relaxing': 17980, 'kilimo': 17981, 'habib': 17982, '45-day': 17983, 'zctu': 17984, 'czar': 17985, 'okala': 17986, 'brownback': 17987, 'massouda': 17988, 'enthusiastic': 17989, 'partisan': 17990, 'rod': 17991, 'yards': 17992, 'kaman': 17993, 'hasten': 17994, 'relaunch': 17995, 'yury': 17996, 'gambira': 17997, 'mandalay': 17998, 'orchestrating': 17999, 'bandundu': 18000, 'navigate': 18001, 'nickelodeon': 18002, 'signifies': 18003, 'trivial': 18004, 'meltdown': 18005, 'theodor': 18006, 'zu': 18007, 'ou': 18008, 'sabouri': 18009, 'disposition': 18010, 'nano': 18011, 'seamen': 18012, 'ttf': 18013, 'nz': 18014, 'vulnerability': 18015, 'remoteness': 18016, 'lobster': 18017, 'reuniting': 18018, 'flute': 18019, 'haul': 18020, 'signboard': 18021, 'unwittingly': 18022, 'dashed': 18023, 'terribly': 18024, 'reindeer': 18025, 'deserved': 18026, 'rudolph': 18027, 'cylinder': 18028, 'jumpers': 18029, 'beltrame': 18030, 'chiapolino': 18031, 'fractures': 18032, 'concussion': 18033, 'montas': 18034, 'skewed': 18035, 'arawak': 18036, 'slows': 18037, 'siphiwe': 18038, 'safa': 18039, 'sedibe': 18040, 'memorable': 18041, 'spanning': 18042, 'zaken': 18043, 'cascade': 18044, 'faisalabad': 18045, 'sidique': 18046, 'sidon': 18047, 'shehade': 18048, 'housekeeper': 18049, 'patriots': 18050, '75,000': 18051, 'brammertz': 18052, 'utmost': 18053, 'rightful': 18054, 'geagea': 18055, 'liquids': 18056, 'twenty-three': 18057, 'nazif': 18058, 'saxby': 18059, 'waking': 18060, 'comfortably': 18061, 'abercrombie': 18062, 'silenced': 18063, 'sprout': 18064, 'ivanic': 18065, 'bird-lime': 18066, 'pluck': 18067, '321': 18068, 'mother-to-child': 18069, 'nagano': 18070, 'follow-up': 18071, 'abdulaziz': 18072, 'intending': 18073, 'colts': 18074, 'phillip': 18075, 'hotline': 18076, 'zinjibar': 18077, 'noaman': 18078, 'gomaa': 18079, 'aqaba': 18080, 'aspiazu': 18081, 'rubina': 18082, 'german-made': 18083, 'pro-secular': 18084, 'three-dimensional': 18085, 'modell': 18086, 'nausea': 18087, 'd': 18088, 'feudal': 18089, 'ohrid': 18090, '1609': 18091, '1820s': 18092, 'raft': 18093, 'unknowingly': 18094, 'splm': 18095, 'django': 18096, 'reinhardt': 18097, 'bossa': 18098, 'aneurysm': 18099, 'cerberus': 18100, 'straightened': 18101, 'buhriz': 18102, 'gasses': 18103, 'unscheduled': 18104, 'all-night': 18105, 'babil': 18106, 'qurans': 18107, 'mcadams': 18108, 'razor': 18109, 'brokering': 18110, 'tamir': 18111, 'soleil': 18112, 'persson': 18113, 'laila': 18114, 'freivalds': 18115, 'sharpest': 18116, 'cerveny': 18117, 'dtp': 18118, 'salmonella': 18119, 'rash': 18120, 'summoning': 18121, 'des': 18122, 'danilovich': 18123, 'idled': 18124, 'mpla': 18125, 'dos': 18126, 'unita': 18127, 'helena': 18128, 'wig': 18129, 'bald': 18130, 'abstained': 18131, 'yogi': 18132, 'bathtub': 18133, 'invented': 18134, '1850': 18135, '1875': 18136, 'spammers': 18137, 'upgrading': 18138, 'jeffery': 18139, 'p-3': 18140, 'orion': 18141, 'dances': 18142, 'surfaces': 18143, 'turf': 18144, 'successors': 18145, 'ligament': 18146, 'magazzeni': 18147, 'sermons': 18148, 'christa': 18149, 'mcauliffe': 18150, 'liftoff': 18151, 'cycling': 18152, 'scilingo': 18153, 'recanted': 18154, 'cult': 18155, 'embracing': 18156, 'publicized': 18157, 'patzi': 18158, 'interahamwe': 18159, 'zhou': 18160, 'rings': 18161, '500-meter': 18162, 'hilton': 18163, '580': 18164, 'petrova': 18165, '149': 18166, 'postcard': 18167, 'qaumi': 18168, 'pro-reformist': 18169, 'crackdowns': 18170, 'jean-baptiste': 18171, 'tan': 18172, 'rumbo': 18173, 'propio': 18174, 'courier': 18175, 'samburu': 18176, 'boasts': 18177, '1870s': 18178, 'mahdist': 18179, '1236': 18180, 'dare': 18181, 'unregistered': 18182, 'shaath': 18183, 'staggered': 18184, 'abstentions': 18185, 'spite': 18186, 'sk': 18187, 'twenty': 18188, 'adulthood': 18189, 'camouflage': 18190, 'lowey': 18191, '28,000': 18192, 'yunesi': 18193, 'jamaicans': 18194, 'bermet': 18195, 'steam': 18196, 'closed-door': 18197, 'trickle': 18198, 'coexist': 18199, 'kamel': 18200, 'gollnisch': 18201, 'disputing': 18202, 'nullify': 18203, 'discotheques': 18204, 'trademark': 18205, 'bernd': 18206, 'nogaideli': 18207, 'north-central': 18208, 'anglican': 18209, 'rowan': 18210, 'mostar': 18211, 'krzyzewski': 18212, 'grocery': 18213, 'coalitions': 18214, '265': 18215, 'tremor': 18216, 'halter': 18217, 'middle-aged': 18218, 'courted': 18219, '439': 18220, 'cimpl': 18221, 're-started': 18222, 'getty': 18223, 'bc': 18224, 'trans-atlantic': 18225, 'eight-day': 18226, 'much-needed': 18227, 'al-fayfi': 18228, 'veiled': 18229, 'destroys': 18230, 'bjp': 18231, '24-meter': 18232, 'granite': 18233, 'f-4': 18234, 'ex-dictator': 18235, 'prediction': 18236, 'co-defendant': 18237, 'ovation': 18238, 'stumbled': 18239, 'likened': 18240, '218': 18241, '19-nation': 18242, 'warriors': 18243, 'decisively': 18244, 'indiscriminate': 18245, 'ecotourism': 18246, 'resource-poor': 18247, '48,000': 18248, 'loads': 18249, 'galloped': 18250, 'kohdamani': 18251, 'ajmal': 18252, 'alamzai': 18253, 'toilets': 18254, 'rated': 18255, 'nirvana': 18256, 'wed': 18257, 'aol': 18258, 'realistic': 18259, 'puebla': 18260, 'herceptin': 18261, 'drugmaker': 18262, 'body-searching': 18263, 'intelligent': 18264, 'pm': 18265, 'radius': 18266, 'alluding': 18267, 'badawi': 18268, 'obliged': 18269, 'tahhar': 18270, 'pietersen': 18271, 'islamophobia': 18272, 'xenophobia': 18273, 'haze': 18274, 'peipah': 18275, 'tassos': 18276, 'ashes': 18277, 'nme': 18278, 'ex-girlfriend': 18279, 'confiscation': 18280, 'khagrachhari': 18281, 'bengali': 18282, 'karradah': 18283, 'zelenovic': 18284, 'globally': 18285, 'aggravating': 18286, 'sidi': 18287, 'cheikh': 18288, 'abdallahi': 18289, 'anti-bush': 18290, 'halifax': 18291, 'simulating': 18292, 'goldschmidt': 18293, 'rizgar': 18294, 'unpaid': 18295, 'interpret': 18296, 'millionaire': 18297, 'hopkins': 18298, 'karate': 18299, 'eddie': 18300, 'enzo': 18301, 'redline': 18302, 'budweiser': 18303, 'immense': 18304, 'shifts': 18305, 'hib': 18306, 'placards': 18307, 'mukakibibi': 18308, 'umurabyo': 18309, 'hatf': 18310, '224': 18311, 'confidence-building': 18312, 'asb': 18313, 'asagoe': 18314, 'a380s': 18315, 'silently': 18316, 'kalma': 18317, 'acropolis': 18318, 'straying': 18319, 'expedite': 18320, 'atwood': 18321, 'artificially': 18322, 'hajdib': 18323, 'late-stage': 18324, 'hycamtin': 18325, '3,700': 18326, 'ethiopian-born': 18327, 'loj': 18328, 'cowboy': 18329, 'destinations': 18330, 'ecc': 18331, 'rmi': 18332, 'lackluster': 18333, 'unsettled': 18334, '1930': 18335, 'popes': 18336, 'interreligious': 18337, '380': 18338, 'slater': 18339, 'mcqueen': 18340, 'saturated': 18341, 'dormitory': 18342, 'unforgivable': 18343, 'panyu': 18344, 'xdr': 18345, 'toy-shaped': 18346, 'kuti': 18347, 'vivanco': 18348, 'lustiger': 18349, 'admits': 18350, 'faults': 18351, 'habsadeh': 18352, 'shargudud': 18353, 'feliciano': 18354, '1,15,000': 18355, 'kimunya': 18356, '256': 18357, 'behave': 18358, 'lockerbie': 18359, 'brew': 18360, 'akhmad': 18361, 'sulaimaniyah': 18362, 'hyperinflation': 18363, 'refuges': 18364, 'vegetation': 18365, 'aruba': 18366, 'footing': 18367, 'jaques': 18368, 'harbhajan': 18369, '332': 18370, 'mid-february': 18371, 'aviator': 18372, 'sideways': 18373, 'duress': 18374, 'working-level': 18375, 'zurich': 18376, 'verbally': 18377, 'gunning': 18378, 'arafa': 18379, 'schatten': 18380, 'embryonic': 18381, 'lance': 18382, 'beleaguered': 18383, 'explosively': 18384, 'efps': 18385, 'healing': 18386, 'adhamiya': 18387, 'significance': 18388, 'iveta': 18389, 'rtl': 18390, 'nyathi': 18391, 'downriver': 18392, 'home-made': 18393, 'inflexible': 18394, 'drel': 18395, '46th': 18396, 'contradict': 18397, 'caricatures': 18398, 'disgust': 18399, '163': 18400, '33,000': 18401, 'invests': 18402, 'yomiuri': 18403, 'counselor': 18404, 'gunshots': 18405, '710': 18406, 'convicts': 18407, 'al-muqrin': 18408, 'squash': 18409, 'twigs': 18410, 'trod': 18411, 'undersecretary-general': 18412, 'talha': 18413, 'ulemas': 18414, 'betrayed': 18415, 'dovish': 18416, 'blocs': 18417, 'unilaterally': 18418, 'akhund': 18419, 'jinnah': 18420, 'reese': 18421, 'rdx': 18422, 'gomes': 18423, 'renunciation': 18424, 'lashkar-e-taiba': 18425, 'verizon': 18426, 's&p': 18427, 'kgb': 18428, 'sukhorenko': 18429, 'proclaiming': 18430, 'sandwich': 18431, 'smearing': 18432, 'luca': 18433, 'cantv': 18434, 'yasar': 18435, '37,000': 18436, 'tolling': 18437, 'wept': 18438, 'echoed': 18439, 'wajama': 18440, 'gojko': 18441, 'flourished': 18442, 'riza': 18443, 'bourguiba': 18444, 'viable': 18445, 'alderman': 18446, 'raccoon': 18447, 'outpouring': 18448, 'dicaprio': 18449, 'sparing': 18450, 'kakooza': 18451, 'dye': 18452, 'insulation': 18453, '617': 18454, 'gulag': 18455, 'ri': 18456, 'jo': 18457, 'bextra': 18458, 'passersby': 18459, 'donetsk': 18460, 'belligerence': 18461, 'circa': 18462, 'tenaris': 18463, 'natasha': 18464, 'couso': 18465, 'maori': 18466, 'coleman': 18467, 'brandished': 18468, 'rigorous': 18469, 'usage': 18470, 'wraps': 18471, 'botanical': 18472, 'hindi-language': 18473, 'faroe': 18474, 'taboo': 18475, 'favourite': 18476, 'lap': 18477, 'dhimbil': 18478, 'hawiye': 18479, 'fawzi': 18480, 'geithner': 18481, 'tytler': 18482, 'enticing': 18483, 'equate': 18484, 'sugiura': 18485, '453': 18486, 'lewthwaite': 18487, 'lid': 18488, 'greeks': 18489, 'peace-broker': 18490, 'eidsvig': 18491, 'choir': 18492, 'jac': 18493, 'ricci': 18494, 'hulya': 18495, 'kocyigit': 18496, 'profound': 18497, 'shameful': 18498, 'late-night': 18499, 'doghmush': 18500, 'gebran': 18501, 'nikola': 18502, 'paracel': 18503, 'food-processing': 18504, 'bailouts': 18505, 'dwindling': 18506, 'fasten': 18507, 'overtaken': 18508, 'amusement': 18509, 'compare': 18510, 'perth-based': 18511, 'ludlam': 18512, 'glacial': 18513, 'danuri': 18514, 'spontaneous': 18515, 'top-ranked': 18516, 'near-daily': 18517, '1908': 18518, 'auwaerter': 18519, 'dirt': 18520, 'incited': 18521, 'mummy': 18522, 'sarcophagus': 18523, 'theyab': 18524, 'beatified': 18525, 'sainthood': 18526, 'prizren': 18527, 'lakers': 18528, 'leavitt': 18529, 'interwar': 18530, 'voronin': 18531, 'three-fifths': 18532, 'budgeted': 18533, 'tedder': 18534, 'onerepublic': 18535, 'beatboxing': 18536, 'mistaking': 18537, 'karel': 18538, 'osvaldo': 18539, 'skill': 18540, 'hortefeux': 18541, 'unfolded': 18542, 'nicanor': 18543, 'faeldon': 18544, 'altar': 18545, 'sardenberg': 18546, 'validity': 18547, 'infrared': 18548, 'cia-military': 18549, 'tf-121': 18550, 'jungles': 18551, '38,000': 18552, 'shams': 18553, 'tomar': 18554, 'northernmost': 18555, 'mora': 18556, 'announces': 18557, 'descriptions': 18558, 'comedians': 18559, 'waldron': 18560, 'encompasses': 18561, 'butcher': 18562, 'mcneill': 18563, 'seyranlioglu': 18564, 'astros': 18565, 'comerio': 18566, 'compares': 18567, 'punxsutawney': 18568, 'copa': 18569, 'nassif': 18570, 'klein': 18571, 'maariv': 18572, 'rohingya': 18573, 'bmg': 18574, 'kabylie': 18575, 'ieds': 18576, 'beijing-based': 18577, 'blackmailer': 18578, 'floquet': 18579, 'lek': 18580, 'alimony': 18581, '10-month-old': 18582, 'mcgrady': 18583, 'cavaliers': 18584, 'assists': 18585, 'empowers': 18586, 'mayan': 18587, 'persians': 18588, 'shevardnadze': 18589, 'acc': 18590, 'imply': 18591, 'fool': 18592, 'lausanne': 18593, 'ticona': 18594, 'chute': 18595, 'cave-in': 18596, 'libel': 18597, 'sodomy': 18598, 'ahvaz': 18599, 'chamrouen': 18600, 'colleges': 18601, 'sonntag': 18602, 'sics': 18603, 'sapporo': 18604, 'jeremic': 18605, 'cyclist': 18606, 'duggal': 18607, 'atmospheric': 18608, 'dodik': 18609, 'balasingham': 18610, 'pacheco': 18611, 'merchants': 18612, 'ashtiani': 18613, 'lashes': 18614, 'adultery': 18615, 'surano': 18616, 'eruptions': 18617, 'espersen': 18618, 'baghran': 18619, 'flaring': 18620, 'commissions': 18621, 'kahar': 18622, 'screaming': 18623, 'documenting': 18624, 'cummings': 18625, 'sacirbey': 18626, 'detonator': 18627, 'resembling': 18628, 'flag-waving': 18629, 'thura': 18630, 'mahn': 18631, 'ghormley': 18632, 'janice': 18633, 'kabungulu': 18634, 'flagrant': 18635, 'interesting': 18636, 'herrington': 18637, 'unbeaten': 18638, 'junoon': 18639, 'falu': 18640, 'india-pakistan': 18641, 'rodham': 18642, 'iraqi-led': 18643, 'poleo': 18644, 'mezerhane': 18645, 'jamaican': 18646, 'pul-e-charkhi': 18647, 'lott': 18648, 'witt': 18649, 'digits': 18650, 'redmond': 18651, 'atambayev': 18652, 'absalon': 18653, 'aalam': 18654, 'kotkai': 18655, 'al-zahar': 18656, 'datanalisis': 18657, 'piazza': 18658, 'schwarzenburg': 18659, 'quds': 18660, 'darmawan': 18661, 'magazines': 18662, 'bumper': 18663, 'falcons': 18664, 'rams': 18665, 'invalidate': 18666, 'kubis': 18667, 'tain': 18668, 'marinellis': 18669, 'soundly': 18670, 'bengalis': 18671, 'fundraiser': 18672, 'goatherd': 18673, 'pulikovsky': 18674, 'lively': 18675, 'herd': 18676, 'sonata': 18677, 'follower': 18678, 'puteh': 18679, 'kind-hearted': 18680, 'nursery': 18681, 'abia': 18682, 'ge': 18683, 'hints': 18684, 'tunisian': 18685, 'mentally': 18686, 'volodymyr': 18687, 'bernal': 18688, 'sopranos': 18689, 'reprocess': 18690, 'velvet': 18691, 'udd': 18692, 'falash': 18693, 'mura': 18694, 'non-jews': 18695, 'songwriters': 18696, 'nominating': 18697, 'diaz-balart': 18698, 'antwerp': 18699, 'pitt': 18700, 'henman': 18701, 'mashhad': 18702, 'silos': 18703, 'implicate': 18704, 'congressmen': 18705, 'al-sammarei': 18706, 'westminster': 18707, 'torkham': 18708, 'forklift': 18709, 'avril': 18710, 'lavigne': 18711, 'krasniqi': 18712, 'nicaraguans': 18713, 'mpp': 18714, 'disbursement': 18715, 'partridge': 18716, 'earnestly': 18717, 'mohaqiq': 18718, 'lazarevic': 18719, 'adi': 18720, 'abeto': 18721, 'purposely': 18722, 'culprits': 18723, 'netting': 18724, 'jaime': 18725, 'lloyd': 18726, 'olmo': 18727, 'lazaro': 18728, 'chanet': 18729, 'ushakov': 18730, '03-mar': 18731, 'al-siddiq': 18732, 'surkhanpha': 18733, 'kiriyenko': 18734, 'megawatts': 18735, 'zolfaghari': 18736, 'certainty': 18737, 'kostiw': 18738, 'distraction': 18739, 'tetanus': 18740, 'dabous': 18741, 'yanbu': 18742, 'howlwadaag': 18743, 'karua': 18744, 'murungaru': 18745, 'meek': 18746, 'ssr': 18747, 'truss': 18748, 'katharine': 18749, 'pellets': 18750, 'avid': 18751, 'montiglio': 18752, 'slaughterhouse': 18753, 'dushevina': 18754, 'razzano': 18755, 'tarmac': 18756, 'disembark': 18757, 'post-tsunami': 18758, 'peace-keeping': 18759, 'grytviken': 18760, 'nm': 18761, 'caste': 18762, 'stepanek': 18763, 'lilienfeld': 18764, 'udf': 18765, 'costello': 18766, 'aleksandar': 18767, 'yevpatoria': 18768, 'rumbling': 18769, 'rappers': 18770, 'achakzai': 18771, 'halloween': 18772, 'afars': 18773, 'flavors': 18774, 'lipstick': 18775, 'ear': 18776, 'abetting': 18777, 'cellist': 18778, 'constrain': 18779, 'shawal': 18780, 'syvkovych': 18781, 'bengali-speaking': 18782, 'eminem': 18783, 'mathers': 18784, 'military-to-military': 18785, 'fuller': 18786, 'grasshoppers': 18787, 'nambiar': 18788, 'equations': 18789, 'approximation': 18790, 'gospel': 18791, 'pilgrim': 18792, 'druze': 18793, 'tokar': 18794, 'fay': 18795, "l'aquilla": 18796, 'middlesbrough': 18797, 'essebar': 18798, 'ekici': 18799, 'kohl': 18800, 'annul': 18801, 'resolute': 18802, 'shambles': 18803, 'cardiac': 18804, 'catheterization': 18805, 'inserting': 18806, 'catheter': 18807, 'umbrella-like': 18808, 'plug': 18809, 'burt': 18810, 'rutan': 18811, 'ansari': 18812, 'x': 18813, 'mojave': 18814, 'krona': 18815, 'wipha': 18816, '1,09,000': 18817, 'haarde': 18818, 'tsunami-producing': 18819, '1,76,000': 18820, '1,28,000': 18821, 'affray': 18822, 'first-class': 18823, 'whisky': 18824, 'gritty': 18825, 'multi-billion-dollar': 18826, 'grignon': 18827, 'icg': 18828, 'akwei': 18829, 'thompson': 18830, 'losers': 18831, 'strengths': 18832, 'forums': 18833, 'deadliness': 18834, 'lourdes': 18835, 'ollanta': 18836, 'humala': 18837, 'tzotzil': 18838, 'mask': 18839, 'arab-islamic': 18840, 'al-karim': 18841, '150-year-old': 18842, 'darul-uloom': 18843, 'deoband': 18844, 'falsely': 18845, 'spurs': 18846, 'bulletproof': 18847, 'remodeled': 18848, 'specially-built': 18849, 'syrian-born': 18850, 'eddin': 18851, 'yarkas': 18852, 'veronique': 18853, 'law-and-order': 18854, 'gspc': 18855, 'x-ray': 18856, 'humiliated': 18857, 'liquefied': 18858, 'endeavors': 18859, 'tonnage': 18860, 'mid-1800s': 18861, 'pearling': 18862, 'attain': 18863, 'second-highest': 18864, 'misdirected': 18865, 'revitalization': 18866, 'aclu': 18867, 'admiring': 18868, 'ungrateful': 18869, 'deliverer': 18870, 'marveled': 18871, 'pit': 18872, 'clutched': 18873, 'mathematicians': 18874, 'nostalgia': 18875, 'grammar': 18876, 'blacksmith': 18877, 'toughened': 18878, 'rigors': 18879, 'blacksmithing': 18880, 'potatoes': 18881, 'overlooked': 18882, 'overcharges': 18883, 'audits': 18884, 'clear-cut': 18885, 'bathed': 18886, 'ribbon': 18887, '5,19,000': 18888, 'consul': 18889, 'makepeace': 18890, 'morrell': 18891, 'brakes': 18892, 'quicken': 18893, 'evaluate': 18894, 'pro-socialist': 18895, 'vesna': 18896, 'harlan': 18897, 'patrons': 18898, 'kabobs': 18899, 'pizza': 18900, 'rahimgul': 18901, 'sarawan': 18902, 'multi-year': 18903, 'mini-satellites': 18904, 'carve': 18905, 'toshiyuki': 18906, 'takano': 18907, '22.4': 18908, '14.88': 18909, '64.97': 18910, 'arrows': 18911, 'shays': 18912, 'continual': 18913, 'israeli-based': 18914, 'anti-election': 18915, 'burnt': 18916, 'pilotless': 18917, 'widely-expected': 18918, '7,600': 18919, 'cancers': 18920, 'marrow': 18921, 'autoimmune': 18922, 'villarreal': 18923, 'nayef': 18924, 'forthcoming': 18925, 'busload': 18926, '71,000': 18927, '1,44,000': 18928, 'once-a-decade': 18929, 'vans': 18930, 'suhaila': 18931, 'snagovo': 18932, 'sistan-baluchestan': 18933, 'bikers': 18934, 'sinatra': 18935, 'dominik': 18936, 'braces': 18937, 'rameez': 18938, 'number-five': 18939, 'annie': 18940, 'lennox': 18941, 'parishes': 18942, 'out-of-competition': 18943, 'biathletes': 18944, 'blood-doping': 18945, 'transfusions': 18946, 'sneaked': 18947, 'jehan': 18948, 'pronouncements': 18949, 'churned': 18950, 'programmers': 18951, 'shigeru': 18952, 'appreciate': 18953, 'folks': 18954, 'balmy': 18955, 'snowman': 18956, 'khormato': 18957, 'arbaeen': 18958, 'backlog': 18959, 'graveyards': 18960, 'lush': 18961, 'magezi': 18962, 'triumfalnaya': 18963, 'alexeyeva': 18964, 'sakharov': 18965, 'allahabad': 18966, 'eject': 18967, 'adraskan': 18968, 'megan': 18969, 'ambuhl': 18970, 'summary': 18971, 'soyuzcapsule': 18972, 'abdurahman': 18973, 'fyodor': 18974, 'yurchikhin': 18975, 'kotov': 18976, 'likes': 18977, 'spaceflight': 18978, 'gagarin': 18979, 'tyurin': 18980, 'lopez-alegria': 18981, 'sunita': 18982, 'cropped': 18983, 'beard': 18984, 'svinarov': 18985, 'illegals': 18986, 'reconciled': 18987, 'propagandists': 18988, 'hateful': 18989, 'weathering': 18990, 'down-turn': 18991, 'ups': 18992, 'downs': 18993, 'cr': 18994, '737-800': 18995, '787s': 18996, 'cathay': 18997, '3-billion': 18998, 'mustapha': 18999, 'ludwig': 19000, 'fuge': 19001, 'auctioneers': 19002, 'theological': 19003, '80-page': 19004, '1826': 19005, 'annotations': 19006, 'crayon': 19007, 'rediscovery': 19008, 'reassessment': 19009, 'deaf': 19010, '1827': 19011, 'mortazavi': 19012, 'mana': 19013, 'neyestani': 19014, 'poked': 19015, 'tabriz': 19016, 'cockroach': 19017, 'peineta': 19018, 'foothold': 19019, 'trustco': 19020, 'mi': 19021, 'llion': 19022, 'tampa': 19023, 'devolution': 19024, 'colonizing': 19025, 'mosquito-infested': 19026, 'swamps': 19027, 'cartago': 19028, 'disintegrated': 19029, 'offstage': 19030, 'bronchitis': 19031, 'trawling': 19032, 'venetiaan': 19033, 'tamed': 19034, 'waned': 19035, '1498': 19036, 'uncolonized': 19037, '1762': 19038, 'cacao': 19039, 'ringleaders': 19040, 'reinstituted': 19041, 'endless': 19042, 'shipwreck': 19043, 'inveighed': 19044, 'perchance': 19045, 'indulging': 19046, 'wand': 19047, 'hast': 19048, 'thyself': 19049, 'rami': 19050, 'al-tayyah': 19051, 'counter-terror': 19052, 'picasso': 19053, "o'keeffe": 19054, 'primimoda': 19055, 'ndimyake': 19056, 'mwakalyelye': 19057, 'hardcourts': 19058, 'carson': 19059, 'sopore': 19060, '542': 19061, 'orbited': 19062, 'schwehm': 19063, 'solar-electric': 19064, 'interplanetary': 19065, 'miniaturized': 19066, 'abdikadir': 19067, 'lafoole': 19068, 'evade': 19069, 'short-wave': 19070, 'bristol': 19071, 'huon': 19072, 'angor': 19073, 'puth': 19074, 'lim': 19075, 'thierse': 19076, '2,711': 19077, 'unadorned': 19078, 'slabs': 19079, 'mid-2004': 19080, 'similarly': 19081, 'n.r.c.': 19082, 'surma': 19083, 'sunamganj': 19084, 'caymans': 19085, 'esad': 19086, 'bajramovic': 19087, 'serb-run': 19088, 'omarska': 19089, 'keraterm': 19090, 'wrested': 19091, 'obedience': 19092, 'scripture': 19093, 'talents': 19094, 'generosity': 19095, 'salvaged': 19096, 'generously': 19097, 'miillion': 19098, 'mural': 19099, 'american-made': 19100, 'wired': 19101, 'deh': 19102, 'rawood': 19103, 'militant-related': 19104, 'revising': 19105, '7,800': 19106, 'faruq': 19107, 'qaddumi': 19108, 'pro-hezbollah': 19109, 'moualem': 19110, 'israel-hezbollah': 19111, 'itu': 19112, 'macrumors.com': 19113, 'keyboard': 19114, '30-year-long': 19115, '75-member': 19116, 'freetown': 19117, 'syphon': 19118, 'contaminant': 19119, 'interregnum': 19120, 'conclave': 19121, 'obtainable': 19122, 'toxins': 19123, 'radiological': 19124, 'egyptian-sponsored': 19125, 'fatah-controlled': 19126, 'record-setting': 19127, '268': 19128, 'needles': 19129, 'demining': 19130, 'deminers': 19131, 'iranativu': 19132, 'skater': 19133, 'dashing': 19134, 'nuclear-whistle': 19135, 'blower': 19136, 'ghazala': 19137, 'naama': 19138, 'exaggerate': 19139, '75-minute-long': 19140, 'ascendant': 19141, 'wreathlaying': 19142, 'presumably': 19143, '636-type': 19144, '677': 19145, 'shoot-down': 19146, 'reaffirming': 19147, 'eight-step': 19148, 'designates': 19149, 'seoul-based': 19150, 'precision': 19151, 'europeanunion-russian': 19152, 'razek': 19153, 'majaidie': 19154, 'shielded': 19155, 'anti-tank': 19156, 'deputy-leader': 19157, 'teshiktosh': 19158, 'karasu': 19159, 'chapel': 19160, 'member-states': 19161, 'embarks': 19162, 'jae': 19163, 'carmen': 19164, 'stoves': 19165, 'physics': 19166, 'cages': 19167, 'gavutu': 19168, 'scratches': 19169, 'sunburn': 19170, 'undernourished': 19171, '51-to-44': 19172, '12-point': 19173, 'non-member': 19174, 'unisa': 19175, 'democratize': 19176, 'five-man': 19177, 'delano': 19178, 'zemedkun': 19179, 'tekle': 19180, 'wardheer': 19181, 'generalized': 19182, 'classifications': 19183, 'categories': 19184, 'battery-operated': 19185, '37.3': 19186, 'maurits': 19187, 'nassau': 19188, '1715': 19189, '1810': 19190, 'adawe': 19191, 'saed': 19192, 'alleviating': 19193, 'underdevelopment': 19194, 'adapting': 19195, 'low-value-added': 19196, 'centrally-planned': 19197, 'open-market': 19198, 'dec-15': 19199, 'residing': 19200, 'towering': 19201, 'antiquated': 19202, 'vlore': 19203, 'perpetuity': 19204, 'agriculture-led': 19205, 'reinvest': 19206, 'columbite-tantalite': 19207, 'parting': 19208, 'rite': 19209, 'topologist': 19210, 'doughnut': 19211, 'javal': 19212, 'harman': 19213, 'roskosmos': 19214, 'timidria': 19215, 'ates': 19216, 'bondage': 19217, 'arab-african': 19218, 'rebel-': 19219, 'thirty-seven': 19220, 'kahraman': 19221, 'sadikoglu': 19222, 'adversity': 19223, '46-nation': 19224, 'serzh': 19225, 'sarkisian': 19226, '5.7-kilometer': 19227, 'spans': 19228, 'vorotan': 19229, 'tatev': 19230, 'soften': 19231, 'sino-american': 19232, 'manipulates': 19233, 'midnight-to-dawn': 19234, 'al-janabi': 19235, 'fequiere': 19236, 'jean-betrand': 19237, 'cocoa-growing': 19238, 'siegouekou': 19239, 'mykola': 19240, 'suck': 19241, 'speeded': 19242, 'heave': 19243, 'syrian-lebanese': 19244, 'anti-oxidant': 19245, 'larger-scale': 19246, 'high-calorie': 19247, 'a9': 19248, 'stirling': 19249, 'pivotal': 19250, 'hypothermia': 19251, 'free-lance': 19252, 'input': 19253, 'postings': 19254, 'n.a.m.': 19255, 'marthinus': 19256, 'coal-fired': 19257, 'captures': 19258, 'swore': 19259, 'rana': 19260, 'bhagwandas': 19261, 'lyons': 19262, 'mixed-team': 19263, 'tendonitis': 19264, 'svetlana': 19265, 'kuznetsova': 19266, 'perceive': 19267, 'impulse': 19268, 'islamiah': 19269, 'netanya': 19270, '18.7': 19271, 'proudly': 19272, 'albright': 19273, 'scharping': 19274, '17,765': 19275, 'gregorian': 19276, 'conspicuous': 19277, 'skullcaps': 19278, 'mismanaging': 19279, 'devotion': 19280, 'finest': 19281, 'hasty': 19282, 'disproportionately': 19283, 'journals': 19284, 'lancetand': 19285, 'lancet': 19286, 'neurology': 19287, 'counteracting': 19288, 'rothwell': 19289, 'mini-stroke': 19290, 'numbness': 19291, 'slurred': 19292, 'anti-cholesterol': 19293, '38.68': 19294, '5,49,000': 19295, 'precision-guided': 19296, '103.05': 19297, 'record-highs': 19298, 'prohibitions': 19299, 'damascus-based': 19300, 'shrapnel-packed': 19301, 'disprove': 19302, 'wired.com': 19303, 'salons': 19304, 'parlors': 19305, 'unsuitable': 19306, 'hairdresser': 19307, 'nulcear': 19308, '18,00': 19309, 'foreign-made': 19310, 'addicts': 19311, 'somali-ethiopian': 19312, 'squalid': 19313, 'brutalized': 19314, 'realignment': 19315, 'commissioners': 19316, 'deliberations': 19317, 'mcpherson': 19318, 'football-shaped': 19319, 'hover': 19320, 'hurriyet': 19321, 'pech': 19322, 'younus': 19323, 'giuseppe': 19324, 'cagliari': 19325, 'petronio': 19326, 'fresco': 19327, 'misanthropic': 19328, 'frans': 19329, 'rotterdam': 19330, 'bureaucrats': 19331, 'renaming': 19332, 'sectarianism': 19333, 'discrepancy': 19334, 'roadway': 19335, 'evi': 19336, 'sachenbacher': 19337, 'endurance-boosting': 19338, 'epo': 19339, 'kikkan': 19340, 'randall': 19341, 'leif': 19342, 'h.': 19343, 'plastics': 19344, 'metathesis': 19345, 'rearranged': 19346, 'catalysts': 19347, 'hotspur': 19348, 'two-month': 19349, 'offseason': 19350, 'tendon': 19351, 'hotspurs': 19352, 'redknapp': 19353, '96-million': 19354, 'war-time': 19355, 'two-week-long': 19356, 'barahona': 19357, 'switching': 19358, 'cocked': 19359, 'israeli-egyptian': 19360, 'automotive-parts': 19361, 'dengfeng': 19362, 'akerfeldt': 19363, 'wallowing': 19364, '52-week': 19365, '16.125': 19366, '13.73': 19367, '9.625': 19368, 'gingl': 19369, 'caribs': 19370, 'crnojevic': 19371, 'theocracy': 19372, 'princes': 19373, '1852': 19374, 'looser': 19375, 'basutoland': 19376, 'basuto': 19377, 'hegang': 19378, 'moshoeshoe': 19379, 'letsie': 19380, 'aegis': 19381, 'arisen': 19382, 'jaws': 19383, 'saddle': 19384, 'bridled': 19385, 'theirs': 19386, 'pomegranate': 19387, 'boastful': 19388, 'disputings': 19389, 'boxing-gloves': 19390, 'tongues': 19391, 'pugilists': 19392, 'statistician': 19393, 'frost': 19394, 'c.o': 19395, 'whispered': 19396, 'kfm': 19397, 'sedition': 19398, 'rory': 19399, 'irishman': 19400, 'manta': 19401, 'evolves': 19402, 'one-child': 19403, 'opt': 19404, 'stockhlom': 19405, 'harold': 19406, 'pinter': 19407, 'mocked': 19408, 'shahr-e-kord': 19409, 'raz': 19410, 'binh': 19411, 'nana': 19412, 'effah-apenteng': 19413, 'precede': 19414, 'israeli-lebanese': 19415, 'conditioning': 19416, 'rot': 19417, 'resold': 19418, 'israel-gaza': 19419, 'kissufin': 19420, 'spotting': 19421, 'farmlands': 19422, 'context': 19423, 'conductive': 19424, '1,74,000': 19425, 'alwi': 19426, 'vied': 19427, '176': 19428, 'kush': 19429, 'merka': 19430, 'turnabout': 19431, 'co-sponsoring': 19432, 'tigray': 19433, 'amhara': 19434, 'oromia': 19435, 'humayun': 19436, 'persona': 19437, 'non': 19438, 'grata': 19439, 'bilge': 19440, 'mardin': 19441, 'allotment': 19442, 'besir': 19443, 'atalay': 19444, 'farmaner': 19445, '3,200-strong': 19446, 'troop-reduction': 19447, 'smoke-free': 19448, 'disease-causing': 19449, 'earthquake-triggered': 19450, 'hatched': 19451, 'candle': 19452, 'one-storey': 19453, 'calmly': 19454, 'baden-baden': 19455, 'bins': 19456, '38.7': 19457, 'buyouts': 19458, '74,000': 19459, 'hourly': 19460, 'ettore': 19461, 'francesco': 19462, 'sequi': 19463, 'responsiblity': 19464, '13-point': 19465, 'worrisome': 19466, 'gears': 19467, 'juncture': 19468, 'dubious': 19469, 'heavily-militarized': 19470, 'khalikov': 19471, 'needing': 19472, 'barranquilla': 19473, 'obstruct': 19474, 'reciprocated': 19475, 'barrios': 19476, 'reynaldo': 19477, 'larrazabal': 19478, 'collusion': 19479, 'african-union': 19480, 'nureddin': 19481, 'kidwai': 19482, 'alertness': 19483, 'conceivable': 19484, 'timessays': 19485, 'kehla': 19486, 'staving': 19487, 'bunji': 19488, '123-kilometer': 19489, 'dogharoun': 19490, 'road-building': 19491, 'salahaddin': 19492, 'zafaraniyah': 19493, 'british-run': 19494, 'reutersnews': 19495, 'swapping': 19496, 'swapped': 19497, 'subterranean': 19498, 'headline': 19499, 'vh1': 19500, 'sensex': 19501, 'hama': 19502, 'aichatou': 19503, 'mindaoudou': 19504, 'albade': 19505, 'abouba': 19506, 'mahamane': 19507, 'lamine': 19508, 'zeine': 19509, 'admired': 19510, 'unforseeable': 19511, 'arbitrator': 19512, 'bobsled': 19513, 'altenberg': 19514, 'zach': 19515, 'whistleblower': 19516, 'revealing': 19517, 'neutron': 19518, 'pointless': 19519, 'implicitly': 19520, 'beersheba': 19521, 'underestimated': 19522, 'nawzad': 19523, 'caffeine': 19524, 'tablets': 19525, 'grapple': 19526, 'ignite': 19527, 'ponds': 19528, 'fixes': 19529, 'dahoun': 19530, 'gignor': 19531, 'tabarre': 19532, 'ex-soldiers': 19533, 'moutaz': 19534, 'slough': 19535, 'mohannad': 19536, 'farfetched': 19537, 's.p.a.': 19538, '37-a-share': 19539, 'advertised': 19540, 'volta': 19541, 'securely': 19542, 'burkinabe': 19543, 'credit-driven': 19544, 'tame': 19545, 'kuna': 19546, 'lag': 19547, 'over-reliance': 19548, 'un-brokered': 19549, 'un-organized': 19550, 'genoese': 19551, 'fortress': 19552, '1215': 19553, 'grimaldi': 19554, '1297': 19555, '1331': 19556, '1419': 19557, 'linkup': 19558, 'scenery': 19559, 'forsaking': 19560, 'adapted': 19561, 'contentment': 19562, 'currycombing': 19563, 'rubbing': 19564, 'oats': 19565, '150-thousand': 19566, 'rochester': 19567, 'ny': 19568, 'fervently': 19569, 'styling': 19570, 'gel': 19571, 'hair-raising': 19572, 'alarmist': 19573, 'advisories': 19574, '1632': 19575, 'froce': 19576, '5-member': 19577, 'redistricting': 19578, 'all-inclusive': 19579, 'incorporating': 19580, 'frenzy': 19581, 'lawmaking': 19582, 'second-round': 19583, 'menas': 19584, 'landholdings': 19585, '154th': 19586, 'général': 19587, 'kountché': 19588, 'blistering': 19589, 'ouwa': 19590, 'shafy': 19591, 'goalkeeper': 19592, 'el-hadary': 19593, 'bordeaux': 19594, 'equalizer': 19595, 'midfield': 19596, 'fathallah': 19597, 'offside': 19598, 'inauspicious': 19599, 'soufriere': 19600, 'sylvester': 19601, 'muscle-building': 19602, 'feb.': 19603, 'laloo': 19604, 'shores': 19605, 'sickness': 19606, 'hemorrhages': 19607, 'dislodged': 19608, 'jabella': 19609, 'hazelnuts': 19610, 'nonalcoholic': 19611, 'merciful': 19612, '18-years-old': 19613, '7.30': 19614, 'tufail': 19615, 'matoo': 19616, "baku-t'bilisi-ceyhan": 19617, "baku-t'bilisi-erzerum": 19618, 'kars-akhalkalaki': 19619, 'mattoo': 19620, '47,000': 19621, 'apprehension': 19622, 'simplified': 19623, 'invincible': 19624, 'neediest': 19625, 'isi': 19626, 'pinning': 19627, 'igniting': 19628, 'discarding': 19629, 'bank-': 19630, 'imf-led': 19631, 'compromises': 19632, 'e.u': 19633, 'sore': 19634, 'regular-season': 19635, 'pre-season': 19636, 'vote-getter': 19637, '1.36': 19638, 'vejjajiva': 19639, 'somchai': 19640, 'wongsawat': 19641, 'yellow-shirted': 19642, 'sainworla': 19643, 'veritas': 19644, 'butty': 19645, 'ring-shaped': 19646, 'beaming': 19647, 'revolutionized': 19648, 'tended': 19649, 'repentance': 19650, 'sensual': 19651, 'culminates': 19652, '60,000-seat': 19653, 'indulgence': 19654, '1853': 19655, '11-style': 19656, '1864': 19657, 'ordeal': 19658, 'video-teleconference': 19659, 'noumea': 19660, 'asad': 19661, '1055': 19662, '1002': 19663, 'wide-body': 19664, 'wafted': 19665, 'persecutors': 19666, 'tanzania-based': 19667, 'anti-tutsi': 19668, 'stationing': 19669, 'unceasing': 19670, 'indissoluble': 19671, 'habitations': 19672, 'edelist': 19673, 'archival': 19674, 'transcript': 19675, '16-team': 19676, 'abound': 19677, 'discern': 19678, 'zaraqawi': 19679, 'sanjaya': 19680, 'baru': 19681, 'jammu-kashmir': 19682, 'stoppages': 19683, 'alitalia': 19684, 'bormio': 19685, 'orchid': 19686, 'cauca': 19687, 'cedatos-gallup': 19688, 'inefficiency': 19689, 'centralize': 19690, '6,40,000': 19691, 'wpmf-tv': 19692, '680': 19693, 'outbursts': 19694, 'long-delayed': 19695, 'loi': 19696, 'rashakai': 19697, 'khata': 19698, 'manhandling': 19699, 'deserting': 19700, 'matti': 19701, 'vanhanen': 19702, 'paraded': 19703, 'maysan': 19704, 'muthana': 19705, 'redeployed': 19706, 'abijan': 19707, 'wandering': 19708, 'nicobari': 19709, 'emaciated': 19710, 'esmatullah': 19711, 'alizai': 19712, 'neshin': 19713, 'chora': 19714, 'unproven': 19715, 'complexes': 19716, 'desalinization': 19717, 'toulouse': 19718, 'esquisabel': 19719, 'urtuzaga': 19720, 'mushir': 19721, 'senseless': 19722, 'dea': 19723, 'gibran': 19724, 'alaina': 19725, 'tepid': 19726, 'dixie': 19727, 'chicks': 19728, 'charisma': 19729, 'goody': 19730, '11.5': 19731, 'payable': 19732, 'n.j.-based': 19733, '9,92,000': 19734, 'anti-takeover': 19735, 'dutch-speaking': 19736, 'flemings': 19737, 'walloons': 19738, 'oases': 19739, 'a.j.': 19740, 'tabaldo': 19741, 'nina': 19742, 'tribally': 19743, 'ashgabat': 19744, 'racy': 19745, 'sensation': 19746, 'redenomination': 19747, 'manat': 19748, 'bulwark': 19749, 'janos': 19750, 'kadar': 19751, 'goulash': 19752, 'marsh': 19753, 'pretend': 19754, 'gait': 19755, 'chink': 19756, 'fatas': 19757, 'synchronize': 19758, 'nadhem': 19759, 'bradman': 19760, 'ricky': 19761, 'ponting': 19762, 'may-84': 19763, 'zaheer': 19764, 'mar-93': 19765, 'saqeb': 19766, 'israeli-built': 19767, 'cordons': 19768, 'rock-throwing': 19769, 'barbed': 19770, 'land-grab': 19771, '1,050': 19772, 'calories': 19773, 'pacifism': 19774, 'speculative': 19775, 'notches': 19776, 'downgrades': 19777, 'quan': 19778, 'caller': 19779, 'pro-syria': 19780, 'arraignment': 19781, 'georgy': 19782, 'maltreat': 19783, 'hastert': 19784, 'centraql': 19785, 'mani': 19786, 'shankar': 19787, 'jadoon': 19788, '57th': 19789, '5,588': 19790, 'kravchenko': 19791, 'tomini': 19792, 'gorontalo': 19793, 'barreled': 19794, '668': 19795, 'tutwiler': 19796, 'rangers': 19797, 'incinerated': 19798, 'caregivers': 19799, 'mlada': 19800, 'fronta': 19801, 'dnes': 19802, 'intake': 19803, 'armchair': 19804, 'tracheotomy': 19805, 'bethelehem': 19806, 'mardan': 19807, 'considerations': 19808, 'parlimentary': 19809, 'duluiyah': 19810, 'insurgent-plagued': 19811, 'extinguished': 19812, 'guardsman': 19813, 'two-member': 19814, 'sawers': 19815, 'last-ditch': 19816, '2,47,000': 19817, 'paktiya': 19818, 'lien': 19819, 'bullhorn': 19820, 'hubbard': 19821, 'feldstein': 19822, '119.48': 19823, '32-million-barrel-a-day': 19824, '357': 19825, 'neelie': 19826, 'kroes': 19827, '633': 19828, 'snarling': 19829, 'sheremetyevo': 19830, 'slicked': 19831, 'encased': 19832, 'ria': 19833, 'taxis': 19834, 'originate': 19835, 'nine-millimeter': 19836, 'touchstone': 19837, 'warms': 19838, 'raziq': 19839, 'alasay': 19840, 'spindleruv': 19841, 'mlyn': 19842, '21.3': 19843, 'kathrin': 19844, 'zettel': 19845, 'marlies': 19846, 'schild': 19847, '2.21.40': 19848, 'zvarych': 19849, 'mae': 19850, 'iranian-made': 19851, 'aborted': 19852, 'abduct': 19853, 'susceptible': 19854, 'affluence': 19855, 'inoculated': 19856, 'state-media': 19857, 'atheist': 19858, 'gregoire': 19859, 'barreling': 19860, 'mayhem': 19861, 'discourages': 19862, 'sidewalks': 19863, 'kilowatt': 19864, 'al-rasul': 19865, 'al-aadham': 19866, 'baucus': 19867, 'montana': 19868, 'exemption': 19869, 'cash-only': 19870, 'rebuked': 19871, 'infringement': 19872, 'state-sanctioned': 19873, 'starving': 19874, 'news-washington': 19875, 'sweet': 19876, '53.88': 19877, 'avi': 19878, 'dichter': 19879, 'mhawesh': 19880, 'al-qadi': 19881, 'naseem': 19882, 'haji': 19883, 'freeskate': 19884, 'hao': 19885, 'liv-grete': 19886, 'half-pipe': 19887, '500-meters': 19888, 'break-in': 19889, 'kogelo': 19890, 'winston': 19891, 'churchill': 19892, 'maneuvering': 19893, 'maroney': 19894, 'tapes': 19895, 'valentino': 19896, '387': 19897, 'kenenisa': 19898, 'bekele': 19899, '81-4': 19900, '965': 19901, 'rumor': 19902, 'tigris': 19903, 'wildly': 19904, 'black-clad': 19905, 'step-grandmother': 19906, 'secessionists': 19907, 'concentrations': 19908, 'zamfara': 19909, 'epidemiologists': 19910, 'kidneys': 19911, 'reproductive': 19912, 'chinhoyi': 19913, 'disinfectant': 19914, 're-erected': 19915, 'masahiko': 19916, 'komura': 19917, 'geelani': 19918, 'quake-damaged': 19919, 'quashed': 19920, 'saga': 19921, 'massively': 19922, 'unvarnished': 19923, 'snuff': 19924, 'integra-a': 19925, 'mailing': 19926, 'entitles': 19927, '13.5': 19928, 'downsized': 19929, '31.7': 19930, 'overstated': 19931, 'anti-independence': 19932, 'supplemented': 19933, 'technology-intensive': 19934, 'oil-and-gas': 19935, '1809': 19936, 'albeit': 19937, 'finns': 19938, 'initiation': 19939, 'clipperton': 19940, 'altars': 19941, 'gently': 19942, 'grasp': 19943, 'mathematics': 19944, 'pharaoh': 19945, 'helium': 19946, 'pencils': 19947, 'elevators': 19948, 'escalators': 19949, 'switches': 19950, 'diapers': 19951, 'keel': 19952, 'recharge': 19953, 'fifty-two': 19954, 'non-food': 19955, 'malnourished': 19956, 'code-named': 19957, 'elmendorf': 19958, 'drik': 19959, 'gassing': 19960, 'glyn': 19961, 'alaeddin': 19962, 'udi': 19963, 'votel': 19964, 'policy-setting': 19965, 'reemergence': 19966, 'mangled': 19967, 'berth': 19968, '0.12': 19969, 'placid': 19970, 'grimmette': 19971, 'martinin': 19972, 'mazdzer': 19973, '0.161': 19974, 'benshoofin': 19975, 'retrosi': 19976, 'hamlin': 19977, 'zablocki': 19978, 'presidium': 19979, 'hyong-sop': 19980, 'paek': 19981, 'veligonda': 19982, 'inayatullah': 19983, 'marshy': 19984, 'brightly': 19985, 'sodden': 19986, 'pitchers': 19987, 'stints': 19988, '09-sep': 19989, '3.76': 19990, 'postseason': 19991, 'wailed': 19992, 'yad': 19993, 'vashem': 19994, 'carriages': 19995, 'roma': 19996, 'gypsies': 19997, 'malawai': 19998, 'luc': 19999, 'chatel': 20000, 'objecting': 20001, 'gauthier': 20002, 'sibling': 20003, 'mouths': 20004, 'anti-polio': 20005, 'unmarried': 20006, 'palin': 20007, 'cotecna': 20008, 'siyam': 20009, 'solovtsov': 20010, 'intermediate': 20011, 'uncalled': 20012, 'mirek': 20013, 'topolanek': 20014, 'mix-up': 20015, 'redskins': 20016, 'doubted': 20017, 'notched': 20018, 'cruised': 20019, 'gimelstob': 20020, 'marach': 20021, 'todd': 20022, 'widom': 20023, 'al-juhyi': 20024, 'purports': 20025, 'embraces': 20026, 'boisvert': 20027, 'winterized': 20028, 'essentially': 20029, 'eritrean-backed': 20030, 'ruphael': 20031, 'amen': 20032, 'benishangul': 20033, 'gumuz': 20034, 'latrines': 20035, 'contaminate': 20036, 'tribune': 20037, 'kuba': 20038, 'rats': 20039, 'limestone': 20040, '19-day': 20041, 'sunflower': 20042, 'subsidize': 20043, 'energy-related': 20044, 'rhine-westphalia': 20045, 'profiled': 20046, 'unauthenticated': 20047, '220-kilogram': 20048, 'bavarian': 20049, 'guenther': 20050, 'beckstein': 20051, 'papillovirus': 20052, 'lesions': 20053, 'andrej': 20054, 'hermlin': 20055, 'gerd': 20056, 'uwe': 20057, 'fleur': 20058, 'dissel': 20059, 'gush': 20060, 'katif': 20061, 'muthmahien': 20062, '183': 20063, 'watchlist': 20064, '244': 20065, 'lower-house': 20066, 'aggressors': 20067, 'fatah-allied': 20068, 'abdulatif': 20069, 'sener': 20070, 'guinean': 20071, 'glades': 20072, 'wading': 20073, 'alligators': 20074, 'two-lane': 20075, 'awesome': 20076, 'sanctioning': 20077, 'chronicles': 20078, 'feminist': 20079, 'p.s.': 20080, 'frode': 20081, 'ruhpolding': 20082, '25.03.05': 20083, 'rösch': 20084, 'greis': 20085, '12.5-kilometer': 20086, 'misses': 20087, 'indonesia-based': 20088, 'nahdlatul': 20089, 'ulama': 20090, 'hasyim': 20091, 'drug-running': 20092, 'abakar': 20093, 'dirk': 20094, 'lever': 20095, 'glen': 20096, 'skyscraper': 20097, 'darien': 20098, 'superintendent': 20099, 'non-environmental': 20100, 'motives': 20101, 'well-timed': 20102, 'bulldozer': 20103, 'french-spanish-swiss': 20104, 'usmani': 20105, 'daynunay': 20106, 'go-betweens': 20107, 'teheran': 20108, 'top-secret': 20109, 'pled': 20110, 'gikoro': 20111, 'arlete': 20112, 'ramaroson': 20113, 'organisation': 20114, 'doncasters': 20115, 'billion-dollar': 20116, 'monteiro': 20117, 'hong-kong-based': 20118, 'contempt-of-court': 20119, 'enshrine': 20120, 'self-determination': 20121, 'yuganskneftgaz': 20122, '148.9': 20123, '153.3': 20124, 'retractable': 20125, 'redeem': 20126, 'coupon': 20127, 'disrepute': 20128, 'well-educated': 20129, '2000s': 20130, 'brake': 20131, 'decelerated': 20132, 'expenditure': 20133, 'hoard': 20134, 'iwf': 20135, '1291': 20136, 'cantons': 20137, '1499': 20138, 'perching': 20139, 'cawed': 20140, 'foreboded': 20141, 'cry': 20142, 'unpleasant': 20143, 'tan-yard': 20144, 'implacable': 20145, 'grain-field': 20146, 'pelt': 20147, 'inning': 20148, 'math': 20149, 'jaji': 20150, 'ratifying': 20151, 'hanan': 20152, 'raufi': 20153, 'u.-s.-led': 20154, 'armor-piercing': 20155, 'anti-mine': 20156, '29-member': 20157, 'sari': 20158, 'pul': 20159, 'inhale': 20160, 'smoker': 20161, 'addictive': 20162, 'lslamic': 20163, 'tham': 20164, 'krabok': 20165, '49-year': 20166, 'msika': 20167, 'muzenda': 20168, 'hand-chosen': 20169, 'emmerson': 20170, 'mnangawa': 20171, 'quentier': 20172, 'iraqi-americans': 20173, 'jordanian-american': 20174, 'cyrus': 20175, 'laywers': 20176, 'prayerful': 20177, 'fenty': 20178, '5.1-magnitude': 20179, 'zhaotong': 20180, 'yunnan-guizhou': 20181, 'bu-young': 20182, 'first-half': 20183, 'khairul': 20184, 'amri': 20185, 'nigerian-born': 20186, 'agu': 20187, 'casmir': 20188, 'mahyadi': 20189, 'panggabean': 20190, 'osako': 20191, 'subpoenas': 20192, 'rossendorf': 20193, 'soviet-built': 20194, 'momir': 20195, 'oilrig': 20196, 'getulio': 20197, 'ironically': 20198, 'p-50': 20199, '1.85': 20200, 'kitgum': 20201, 'bigombe': 20202, 'drinkable': 20203, 'defillo': 20204, 'nonbinding': 20205, '83rd': 20206, 'often-deadly': 20207, 'hashish': 20208, 'genital': 20209, 'genitalia': 20210, 'hemorrhaging': 20211, 'invitations': 20212, 'iftars': 20213, 'el~fasher': 20214, 'afghan-nato': 20215, 'morphine': 20216, 'kibar': 20217, 'delinquency': 20218, 'baume': 20219, 'colonial-era': 20220, 'teikoku': 20221, 'characteristics': 20222, 'plundering': 20223, 'resource-rich': 20224, 'reparations': 20225, 'dedicate': 20226, 'diana': 20227, 'bosnian-serb': 20228, '182': 20229, 'refinement': 20230, '12-country': 20231, '64.08': 20232, '27-member': 20233, 'fuheid': 20234, 'al-muteiry': 20235, 'al-nashmi': 20236, 'muteiry': 20237, 'khobar': 20238, 'racketeering': 20239, 'gang-related': 20240, 'suburban': 20241, 'fists': 20242, 'salute': 20243, 'rachel': 20244, 'armada': 20245, 'mashruh': 20246, 'mahara': 20247, 'dialysis': 20248, 'anat': 20249, 'dolev': 20250, 'zhongwei': 20251, 'farm-raised': 20252, 'sitaula': 20253, 'reappearing': 20254, 'hassem': 20255, 'ramsey': 20256, 'headfirst': 20257, 'snowbank': 20258, 'roughed': 20259, 'marrying': 20260, 'garissa': 20261, 'oranges': 20262, 'commodies': 20263, '197-page': 20264, 'discrepancies': 20265, 'dili': 20266, 'martinho': 20267, 'gusmao': 20268, 'ramos-horta': 20269, 'olo': 20270, 'deregulation': 20271, 'uncontrolled': 20272, 'sudi': 20273, 'yalahow': 20274, 'edgware': 20275, 'hafs': 20276, 'congratulating': 20277, 'concides': 20278, 'notifying': 20279, 'brandenburg': 20280, 'collisions': 20281, 'stefanie': 20282, 'werner': 20283, 'anchorage': 20284, 'bedroom': 20285, 'arjun': 20286, 'disqualifications': 20287, 'compelling': 20288, 'caracas-based': 20289, 'silesia': 20290, '65-to-22': 20291, 'price-per-carat': 20292, 'gemstone': 20293, 'auctioned': 20294, 'six-carat': 20295, 'bested': 20296, 'hancock': 20297, 'fetched': 20298, 'moussaieff': 20299, 'gemstones': 20300, 'vivid': 20301, 'boron': 20302, 'crystal': 20303, 'al-hashemi': 20304, 'revoke': 20305, 'identifications': 20306, 'sarajevo-based': 20307, 'anticipating': 20308, '10,00,000': 20309, 'horbach': 20310, 'ubangi-shari': 20311, 'tumultuous': 20312, 'misrule': 20313, 'ange-felix': 20314, 'patasse': 20315, 'tacit': 20316, '1892': 20317, '1915': 20318, 'makin': 20319, 'tarawa': 20320, 'garrisons': 20321, '1943': 20322, 'oversupply': 20323, 'jilani': 20324, 'euro-zone': 20325, 'contagion': 20326, 'nook': 20327, 'intruded': 20328, 'stranger': 20329, 'antipathy': 20330, 'apologise': 20331, 'madagonians': 20332, 'chagrined': 20333, 'fergana': 20334, 'court-approved': 20335, 'nonexistent': 20336, 'uncontained': 20337, "shi'ite-majority": 20338, 'christiane': 20339, 'berthiaume': 20340, 'navi': 20341, 'pillay': 20342, 'taint': 20343, 'amputation': 20344, 'heckling': 20345, 'anjem': 20346, 'choudary': 20347, 'al-ghurabaa': 20348, 'zirve': 20349, 'wrestling': 20350, 'trabzon': 20351, 'blabague': 20352, 'marboulaye': 20353, 'tram': 20354, 'low-intensity': 20355, 'pro-pakistani': 20356, 'diluting': 20357, 'inhibit': 20358, 'roving': 20359, 'panorama': 20360, '800-meter-wide': 20361, '21-month': 20362, 'six-wheeled': 20363, 'descend': 20364, 'adolescent': 20365, 'woo-ik': 20366, 'obudu': 20367, 'oshiomole': 20368, 'fagih': 20369, 'healthier': 20370, 'cardiovascular': 20371, 'yong-nam': 20372, 'aggressor': 20373, 'estimating': 20374, 'adjusters': 20375, 'ravage': 20376, 'underestimating': 20377, 'dhafra': 20378, 'u-2s': 20379, '380th': 20380, 'expeditionary': 20381, 'full-range': 20382, 'mizhir': 20383, 'yousi': 20384, 'mukhtaran': 20385, 'gang-raped': 20386, 'contractual': 20387, 'liberian-flagged': 20388, 'biscaglia': 20389, 'overtook': 20390, 'triple-jump': 20391, 'ashia': 20392, 'american-born': 20393, '15.16': 20394, 'circulatory': 20395, 'scriptures': 20396, 'camillo': 20397, 'ruini': 20398, 'cardio-circulatory': 20399, 'urinary': 20400, 'soh': 20401, 'antonov': 20402, 'sok': 20403, 'chol': 20404, 'doo-hwan': 20405, "ya'akov": 20406, 'lamp': 20407, 'wayward': 20408, 'almanza': 20409, 'logarda': 20410, 'dror': 20411, 'soku': 20412, 'females': 20413, 'interviewers': 20414, 'yoram': 20415, 'haham': 20416, 'mobsters': 20417, 'vibrant': 20418, 'roemer': 20419, 'exchange-rate': 20420, 'kazem': 20421, 'vaziri-hamaneh': 20422, 'momammad': 20423, 'amy': 20424, 'katz': 20425, 're-deployment': 20426, '14-thousand': 20427, 'fabian': 20428, 'adolphus': 20429, 'wabara': 20430, '273': 20431, 'welsh': 20432, 'hlinethaya': 20433, 'roses': 20434, 'orphan': 20435, 'liz': 20436, 'rosenberg': 20437, '412': 20438, 'athanase': 20439, 'seromba': 20440, 'nandurbar': 20441, 'navapur': 20442, '30-kilometer': 20443, 'heihe': 20444, 'blagoveshchensk': 20445, 'undesirable': 20446, 'pardoning': 20447, 'scouring': 20448, 'angelus': 20449, 'loves': 20450, 'actor-turned-musician': 20451, 'llc': 20452, 'qualifies': 20453, 'takatoshi': 20454, 'imf-world': 20455, 'toufik': 20456, 'hanouichi': 20457, 'mohcine': 20458, 'bouarfa': 20459, 'jew': 20460, 'meknes': 20461, 'fez': 20462, 'trilateral': 20463, 'gadahn': 20464, 'nairobi-based': 20465, 'tibaijuka': 20466, 'semi-permanent': 20467, '12,500': 20468, 'zimbabweans': 20469, 'charting': 20470, 'mentality': 20471, 'daioyu': 20472, 'senkaku': 20473, 'japanto': 20474, 'f': 20475, 'scoreless': 20476, 'half-time': 20477, 'jean-paul': 20478, 'abalo': 20479, '53rd': 20480, 'chun-soo': 20481, 'ahn': 20482, 'jung-hwan': 20483, 'sputtered': 20484, 'kember': 20485, 'loney': 20486, 'harmeet': 20487, 'sooden': 20488, 'damiao': 20489, 'fradique': 20490, 'awarding': 20491, 'doubtful': 20492, 'gisagara': 20493, 'braved': 20494, 'complemented': 20495, 'cubana': 20496, 'c.': 20497, 'walbrecher': 20498, 'francisco-based': 20499, 'a.': 20500, '1.61': 20501, 'write-downs': 20502, '45.75': 20503, 'capital-intensive': 20504, 'debt-restructuring': 20505, 'ineligible': 20506, 'debt-relief': 20507, 'feedstock': 20508, 'ceramics': 20509, 'fabrics': 20510, 'untaxed': 20511, 'harmonizing': 20512, 'occupier': 20513, 'exploit': 20514, 'overstaffed': 20515, 'afloat': 20516, 'senegambia': 20517, 'envisaged': 20518, 'mfdc': 20519, 'snared': 20520, 'homewards': 20521, 'leap': 20522, 'cofer': 20523, 'installs': 20524, 'builder': 20525, 'bribing': 20526, 'janez': 20527, 'jansa': 20528, 'member-countries': 20529, 'roused': 20530, 'largely-muslim': 20531, 'el-obeid': 20532, 'war-induced': 20533, 'booby': 20534, 'retrieving': 20535, 'sub-continent': 20536, 'hewlett-packard': 20537, 'carly': 20538, 'printer': 20539, 'compaq': 20540, 'wayman': 20541, 'syarhei': 20542, 'antonchyk': 20543, 'unsanctioned': 20544, 'insincere': 20545, 'shanwei': 20546, 'hostage-taking': 20547, 'soleiman': 20548, 'tolima': 20549, 'antioquia': 20550, 'simba': 20551, 'banadir': 20552, 'landscape': 20553, 'shatt': 20554, 'al-arab': 20555, 'bulba': 20556, 'payback': 20557, 'mid-july': 20558, 're-scheduled': 20559, 'hill-top': 20560, 'khogyani': 20561, 'multi-million-selling': 20562, 'espino': 20563, 'day-time': 20564, 'thapa': 20565, 'examiner': 20566, 'grills': 20567, 'castelgandolfo': 20568, 'nuremberg': 20569, 'frankenstadion': 20570, 'lukas': 20571, 'podolski': 20572, 'ballack': 20573, '48th': 20574, 'loser': 20575, 'mexico-argentina': 20576, 'leipzig': 20577, 'ukrgazenergo': 20578, 'co-owned': 20579, 'rosukrenergo': 20580, 'bychkova': 20581, 'shahar': 20582, 'scruff': 20583, 'czink': 20584, 'berkofsky': 20585, 'firebrand': 20586, 'virtuosity': 20587, 'berkovsky': 20588, 'scot': 20589, 'riddlesberger': 20590, 'solicit': 20591, 'colonels': 20592, 'islamic-rooted': 20593, 'eighty-six': 20594, 'solidify': 20595, 'nationalistic': 20596, 'tendencies': 20597, 'russian-speaking': 20598, 'ukrainian-speaking': 20599, 'afl-cio': 20600, 'unionists': 20601, 'vocation': 20602, 'pope2you': 20603, 'two-and-half': 20604, 'no-balls': 20605, 'ringwald': 20606, 'malaria-fighting': 20607, 'artemisia': 20608, 'artemisian-based': 20609, 'mefloquine': 20610, 'firat': 20611, 'hakurk': 20612, 'sub-contractor': 20613, 'warmongering': 20614, 'plymouth': 20615, 'djalil': 20616, 'careless': 20617, 'discarded': 20618, 'novaya': 20619, 'gazeta': 20620, 'volvo': 20621, 'allocating': 20622, 'paya': 20623, 'lebar': 20624, 'refuel': 20625, 'starye': 20626, 'novye': 20627, 'disinformation': 20628, 'timelines': 20629, 'iraqi-american': 20630, 'monaf': 20631, 'u.n.-protected': 20632, 'purnomo': 20633, 'yusgiantoro': 20634, '770': 20635, 'plum': 20636, 'nasir': 20637, 'dhakla': 20638, 'juster': 20639, 'micro-miniature': 20640, 'christina': 20641, 'rocca': 20642, 'one-point-three': 20643, 'mathematically-based': 20644, 'mundane': 20645, 'poitou-charentes': 20646, 'respublika': 20647, 'disparaging': 20648, 'kazakhs': 20649, 'dyrdina': 20650, '11th-hour': 20651, '700-billion': 20652, '96.26': 20653, 'desecrating': 20654, 'yosef': 20655, 'eliashiv': 20656, 'closed-circuit': 20657, 'refraining': 20658, '100-year-old': 20659, 'half-hearted': 20660, 'iftaar': 20661, 'kabal': 20662, 'fatah-linked': 20663, 'danzhou': 20664, 'ripe': 20665, '89.7': 20666, '141.9': 20667, '94.8': 20668, '149.9': 20669, '118': 20670, '130.6': 20671, '388': 20672, 'necessitate': 20673, 'amounting': 20674, 'projection': 20675, 'bundesbank': 20676, '0.35': 20677, 'annum': 20678, 'tapered': 20679, 'olav': 20680, 'tryggvason': 20681, '994': 20682, '1397': 20683, 'cession': 20684, 'outset': 20685, 'state-orchestrated': 20686, 'rwandans': 20687, 'rpf-led': 20688, 'retaking': 20689, 'post-genocide': 20690, 'comparative': 20691, 'non-eu': 20692, 'mimics': 20693, 'pupils': 20694, 'arrayed': 20695, 'courtiers': 20696, 'courtier': 20697, 'mischief': 20698, 'amidst': 20699, 'longing': 20700, 'game-bag': 20701, 'plunder': 20702, 'wiled': 20703, 'scar': 20704, 'cavity': 20705, 'annyaso': 20706, 'onulak': 20707, 'passionate': 20708, 'charlie': 20709, 'influences': 20710, 'hillside': 20711, 'thunderstorms': 20712, 'qada': 20713, 'eastbourne': 20714, 'strapping': 20715, 'shortness': 20716, 'accomplishments': 20717, 'smoky': 20718, 'recommit': 20719, 'ozone': 20720, 'acres': 20721, 'chide': 20722, 'sidestep': 20723, 'anti-pollution': 20724, 'hector': 20725, 'robberies': 20726, 'belo': 20727, 'horizonte': 20728, 'fortaleza': 20729, '80-meter': 20730, 'landscaping': 20731, '161-seat': 20732, '944': 20733, 'anti-free': 20734, 'bamboo': 20735, 'kyu-hyung': 20736, 'interfered': 20737, 'aflame': 20738, '204': 20739, 'permitting': 20740, 'amiens': 20741, 'savigny-sur-orge': 20742, 'firebomb': 20743, 'zamoanga': 20744, 'compressed': 20745, 'liquified': 20746, 'stand-up': 20747, 'mccullogh': 20748, 'letterman': 20749, 'cnn-ibn': 20750, '64th': 20751, 'battleship': 20752, 'waver': 20753, 'then-specialist': 20754, 'mess': 20755, 'antiviral': 20756, 'kara-suu': 20757, 'fossils': 20758, 'fossilized': 20759, '25-meter-long': 20760, 'eromanga': 20761, 'weighs': 20762, 'prehistoric': 20763, 'roam': 20764, 'plant-eating': 20765, 'sauropods': 20766, 'sajida': 20767, 'dreadfully': 20768, '5,500-member': 20769, 'politically-dominant': 20770, 'barmer': 20771, 'anchorwoman': 20772, '60-kilometer': 20773, 'twice-weekly': 20774, 'chilpancingo': 20775, 'sos': 20776, 'dissatisified': 20777, 'soderling': 20778, 'eight-man': 20779, 'finale': 20780, 'finalist': 20781, 'gael': 20782, 'monfils': 20783, 'third-round': 20784, 'hard-court': 20785, 'potro': 20786, 'photography': 20787, 'jovicevic': 20788, 'spaceport': 20789, 'heavier': 20790, 'high-resolution': 20791, 'precise': 20792, '5400': 20793, 'homicides': 20794, '1.48': 20795, '87.61': 20796, '88.2': 20797, 'turkish-iraqi': 20798, 'al-badri': 20799, 'al-askari': 20800, 'anti-al-qaida': 20801, 'silencers': 20802, 'neftaly': 20803, 'deceptive': 20804, 'tar': 20805, 'cigarette-labeling': 20806, 'aspect': 20807, 'altria': 20808, 'weary': 20809, 'funneled': 20810, 'counterterrorist': 20811, 'mansur': 20812, 'model-turned-activist': 20813, 'contestant': 20814, 'limb': 20815, 'sleeve': 20816, 'prosthesis': 20817, 'viva': 20818, 'curtains': 20819, 'underfoot': 20820, '55-to-45': 20821, 'akwa': 20822, 'ibom': 20823, 'afren': 20824, '3600': 20825, 'below-average': 20826, 'mukhtar': 20827, 'below-normal': 20828, 'yilmaz': 20829, 'resit': 20830, 'isik': 20831, 'hawija': 20832, 'combat-related': 20833, 'scarcity': 20834, 'condolezza': 20835, 'non-partisan': 20836, 'cbo': 20837, 'threefold': 20838, 'negar': 20839, 'kerman': 20840, 'mid-morning': 20841, 'torbat-e-heydariyeh': 20842, 'rehab': 20843, 'wonderland': 20844, '570': 20845, 'check-in': 20846, 'dui-related': 20847, 'alternative-medicine': 20848, 'unconventional': 20849, 'journal-constitution': 20850, '53-country': 20851, 'iraq-style': 20852, 'defensible': 20853, 'fallback': 20854, 'league-nawaz': 20855, 'khair': 20856, 'shuja': 20857, 'malian': 20858, 'sid-ahmed': 20859, 'nouadhibou': 20860, '235': 20861, '113.2': 20862, '2.46': 20863, '2.42': 20864, 'high-end': 20865, 'villas': 20866, 'inhibits': 20867, 'hadj': 20868, 'ondimba': 20869, 'stonemason': 20870, 'marinus': 20871, '301': 20872, 'marxism': 20873, 'un-negotiated': 20874, 'renamo': 20875, 'joaquim': 20876, 'chissano': 20877, 'emilio': 20878, 'silverstone': 20879, 'well-established': 20880, 'dollarization': 20881, 'non-traditional': 20882, 'uphill': 20883, 'startup': 20884, 'saltillo': 20885, 'cameroonian': 20886, 'kilted': 20887, 'playlists': 20888, 'mwiraria': 20889, 'energy-saving': 20890, '9,60,000': 20891, 'unveiling': 20892, 'bellamy': 20893, 'baixinglou': 20894, 'chaoyang': 20895, 'diners': 20896, 'waiters': 20897, 'backtrack': 20898, 'mind-set': 20899, 'hausas': 20900, 'ibos': 20901, 'anibal': 20902, 'correo': 20903, 'caroni': 20904, 'styled': 20905, 'deploring': 20906, 'al-khalifa': 20907, 're-launch': 20908, 'jela': 20909, 'franceschi': 20910, 'hampton': 20911, 'mineral-rich': 20912, 'alexi': 20913, 'nenets': 20914, 'ria-novosti': 20915, 'whites': 20916, 'two-and-a-half': 20917, 'flu-like': 20918, 'acquittal': 20919, 'kristof': 20920, '0-3': 20921, '02-may': 20922, 'downpour': 20923, 'konstantinos': 20924, 'iaaf': 20925, 'pentastar': 20926, 'v-6': 20927, 'dodge': 20928, 'multimillion': 20929, 'al-amiri': 20930, 'pro-shiite': 20931, 'retract': 20932, '250,000-member': 20933, 'safwat': 20934, 'sherif': 20935, 'hamstring': 20936, 'athletic': 20937, '17-meters': 20938, 'marian': 20939, 'oprea': 20940, 'dmitrij': 20941, 'valukevic': 20942, 'flocked': 20943, 'televisions': 20944, 'staggering': 20945, 'overture': 20946, '69-43': 20947, 'scuttle': 20948, '120-seat': 20949, 'out-of-work': 20950, 'u.n.-led': 20951, 'dimension': 20952, 'priesthood': 20953, 'hierarchy': 20954, 'swiftly': 20955, '8.8': 20956, '950': 20957, 'pilings': 20958, 'ninewa': 20959, '2900': 20960, '4,40,000': 20961, 'jeered': 20962, 'pro-bush': 20963, 'joan': 20964, 'baez': 20965, 'vigils': 20966, 'incumbents': 20967, 'damiri': 20968, 'tayyeb': 20969, 'abel-rahim': 20970, 'ayoub': 20971, 'fewest': 20972, 'bank-based': 20973, 'cfco': 20974, 'pointe': 20975, 'noire': 20976, 'aab-e-ghum': 20977, 'non-ethnic': 20978, 'fnj': 20979, 'worsens': 20980, '423-1': 20981, 'dhabi': 20982, 'trusts': 20983, 'painkiller': 20984, 'diploma': 20985, 'haggling': 20986, 'immigrate': 20987, 'unveils': 20988, 'market-share': 20989, 'carmakers': 20990, 'badran': 20991, 'wheel': 20992, 'suv': 20993, '8-year-old': 20994, 'zookeepers': 20995, 'incubator': 20996, 'jingguo': 20997, 'nurse': 20998, 'gomoa': 20999, 'buduburam': 21000, 'nations-funded': 21001, 'rabiah': 21002, 'cousins': 21003, 'catalina': 21004, 'perpetuate': 21005, 'ingestion': 21006, 'syria-based': 21007, 'excludes': 21008, 'al-haiman': 21009, 'netzarim': 21010, 'morag': 21011, 'ateret': 21012, 'unamid': 21013, 'listens': 21014, 'jurists': 21015, '60-year': 21016, 'arbitrarily': 21017, 'fayssal': 21018, 'mekdad': 21019, 'stone-throwers': 21020, 'veracity': 21021, 'al-tikriti': 21022, 'disc': 21023, '690': 21024, '551': 21025, '2,68,000': 21026, 'balkh': 21027, 'mihtarlam': 21028, 'openings': 21029, 'stringers': 21030, 'rockford': 21031, 'fasteners': 21032, '13.625': 21033, 'outweighs': 21034, 'denominated': 21035, 'creditor': 21036, 'dwindled': 21037, 'dilma': 21038, 'rousseff': 21039, 'greek-speaking': 21040, 'anti-communists': 21041, 'euro-denominated': 21042, 'confine': 21043, 'souvenir': 21044, 'remitted': 21045, 'khurasan': 21046, 'merv': 21047, 'boon': 21048, 'plaintiff': 21049, 'fishing-rich': 21050, 'gurbanguly': 21051, 'accumulating': 21052, 'heathens': 21053, 'peking': 21054, 'translate': 21055, 'pang': 21056, 'ki': 21057, 'devils': 21058, 'barbarity': 21059, 'incensed': 21060, 'wager': 21061, 'janitor': 21062, 'organist': 21063, 'pews': 21064, 'de-orbiting': 21065, 'conakry': 21066, 'squeezed': 21067, 'space.com': 21068, "o'keefe": 21069, 'earth-orbiting': 21070, 'incredibly': 21071, 'celestial': 21072, 'astronomical': 21073, 'overrule': 21074, 'haitian-american': 21075, 'mulgueta': 21076, 'debalk': 21077, 'newcastle': 21078, 'drownings': 21079, 'unknowns': 21080, 'doves': 21081, 'origami': 21082, 'folded': 21083, 'occassion': 21084, 'picnics': 21085, 'cemeteries': 21086, 'xian': 21087, 'han': 21088, 'wuhan': 21089, 'peugeot-citroen': 21090, 'yakaghund': 21091, 'wheelchairs': 21092, 'extricate': 21093, 'bilingual': 21094, 'ashti': 21095, 'anti-iranian': 21096, 'dangtu': 21097, 'identical': 21098, 'win-loss': 21099, 'stampa': 21100, 'parisi': 21101, 'trajectory': 21102, 'miliant': 21103, 'iskandariya': 21104, 'graduated': 21105, 'sulaymaniyeh': 21106, 'emerli': 21107, 'maref': 21108, 'mirpur': 21109, 'sacramento': 21110, 'scurrying': 21111, 'conditioners': 21112, 'mahmood': 21113, 'developmentaly-oriented': 21114, 'moghaddasi': 21115, '640': 21116, 'stockpiled': 21117, 'michaelle': 21118, 'adrienne': 21119, 'turnover': 21120, 'awacs': 21121, 'margraten': 21122, 'erased': 21123, 'vermilion': 21124, 'kohat': 21125, 'mizuki': 21126, 'tomiaki': 21127, 'marathoner': 21128, 'radcliffe': 21129, 'lancaster': 21130, '25-years-old': 21131, 'tzachi': 21132, 'hanegbi': 21133, 'anarchist': 21134, 'kaesong': 21135, 'bride-to-be': 21136, 'drum': 21137, 'bugle': 21138, 'pageant': 21139, 'twilight': 21140, 'feuer': 21141, 'entrapment': 21142, 'close-ally': 21143, 'somali-born': 21144, 'post-assassination': 21145, 'u.s.-european': 21146, 'plaza': 21147, 'encircled': 21148, 'revava': 21149, '1.17': 21150, '50.56': 21151, 'demarcating': 21152, 'abyei': 21153, 'kashmirs': 21154, 'saffir-simpson': 21155, '17.2': 21156, 'viewership': 21157, 'record-holder': 21158, 'six-to-11-year-old': 21159, 'musicalphenomenon': 21160, 'zac': 21161, 'efron': 21162, 'tisdale': 21163, 'spawning': 21164, 'sold-out': 21165, 'defeatism': 21166, 'mockery': 21167, 'turkish-armenian': 21168, 'higüey': 21169, 'faxas': 21170, '216': 21171, 'quand': 21172, 'quoc': 21173, 'kabardino-balkaria': 21174, 'exploiters': 21175, 'exponential': 21176, 'tehrik-e-taliban': 21177, 'baghdadi': 21178, 'hadithah': 21179, 'blitz': 21180, 'natonski': 21181, 'adamantios': 21182, 'vassilakis': 21183, 'recommitted': 21184, 'skeptics': 21185, 'hatta': 21186, 'radjasa': 21187, 'microsystems': 21188, 'beitar': 21189, 'illit': 21190, 'efrat': 21191, 'marcelo': 21192, 'antezana': 21193, 'disposing': 21194, '8.30': 21195, '9.30': 21196, 'cheated': 21197, 'user': 21198, 'castle': 21199, 'sleet': 21200, 'bout': 21201, 'decimate': 21202, 'dhusamareb': 21203, 'marergur': 21204, 'waljama': 21205, 'strewn': 21206, 'jeny': 21207, 'frias': 21208, 'attachés': 21209, 'evangelicals': 21210, 'apprehend': 21211, 'al-aqili': 21212, 'mainz': 21213, 'aqili': 21214, 'wis.': 21215, '1.92': 21216, '352.9': 21217, 'motor-home': 21218, 'deere': 21219, 'larger-than-normal': 21220, 'al-said': 21221, 'omanis': 21222, 'hafiz': 21223, "ba'th": 21224, 'alawite': 21225, 'jarome': 21226, 'iginla': 21227, 'dany': 21228, 'heatley': 21229, 'doan': 21230, "da'ra": 21231, 'legalization': 21232, 'quelling': 21233, 'tjarnqvist': 21234, 'henrik': 21235, 'sedin': 21236, 'corruption-free': 21237, '14.7': 21238, 'hemispheres': 21239, 'e.g.': 21240, 'aragonite': 21241, 'sands': 21242, 'mechanic': 21243, 'imprudent': 21244, 'imprudence': 21245, 'thoughtful': 21246, 'vicissitudes': 21247, 'nap': 21248, 'cosily': 21249, 'slumber': 21250, 'barked': 21251, 'muttering': 21252, 'betsy': 21253, 'gallop': 21254, 'jinan': 21255, 'pillar': 21256, 'low-flying': 21257, 'caliph': 21258, 'caliphate': 21259, 'lobbies': 21260, 'alireza': 21261, 'jamshidi': 21262, 'soheil': 21263, 'farshad': 21264, 'scout': 21265, 'bastani': 21266, 'ghalawiya': 21267, 'mauritanian-algerian': 21268, 'sleeper': 21269, 'isselmou': 21270, 'illegally-built': 21271, 'communist-run': 21272, 'four-decade-long': 21273, 'liberate': 21274, "o'hare": 21275, 'u.s.-hosted': 21276, 'lago': 21277, 'agrio': 21278, 'valiant': 21279, '423': 21280, 'counter-terrorist': 21281, 'prd': 21282, '598-seat': 21283, 'gores': 21284, '673': 21285, 'melchior': 21286, 'ndadaye': 21287, 'kalenga': 21288, 'ramadhani': 21289, '12-year-civil': 21290, 'analyzed': 21291, 'telephoning': 21292, 'maverick': 21293, 'venezuelan-owned': 21294, 'embarrass': 21295, 'al-nabaie': 21296, '76.4': 21297, 'inspects': 21298, 'desolation': 21299, 'currents': 21300, 'dianne': 21301, 'sawyer': 21302, '8,956': 21303, 'dune': 21304, 'four-kilometer': 21305, 'sainct': 21306, 'thessaloniki': 21307, 'cocaine-producing': 21308, 'four-tenths': 21309, 'kyprianou': 21310, 'wanthana': 21311, 'air-dropping': 21312, 'self-sufficient': 21313, 'co-called': 21314, 'rebel-backed': 21315, 'jared': 21316, 'cotter': 21317, 'sloan': 21318, 'vie': 21319, 'jumper': 21320, 'mitterndorf': 21321, 'faultless': 21322, '207.5': 21323, '762.4': 21324, 'morgenstern': 21325, '752.2': 21326, 'planica': 21327, 'overpayments': 21328, '6,78,000': 21329, 'unlocked': 21330, 'footlocker': 21331, 'gambled': 21332, 'boxing': 21333, '18,000-strong': 21334, 'e-commerce': 21335, 'e-gaming': 21336, 'baylesa': 21337, 'boyfriend': 21338, 'pricey': 21339, '4th': 21340, 'coffee-drinkers': 21341, 'avoiding': 21342, 'briskly': 21343, 'ballad': 21344, 'pharrell': 21345, 'downloads': 21346, 'environment-themed': 21347, 'dakhil': 21348, 'yazidis': 21349, 'kurdish-speaking': 21350, 'pre-islamic': 21351, 'post-invasion': 21352, 'gentry': 21353, '1772': 21354, '279-a-ton': 21355, 'comparatively': 21356, 'burundian': 21357, 'onyango': 21358, 'omollo': 21359, 'ochami': 21360, 'marzieh': 21361, 'dastjerdi': 21362, 'underclass': 21363, 'indulge': 21364, 'seeduzzaman': 21365, 'elahi': 21366, 'bakhsh': 21367, 'soomro': 21368, 'fakhar': 21369, 'dignitary': 21370, 'undated': 21371, 'euro-atlantic': 21372, 'computer-chip': 21373, 'krzanich': 21374, 'chi': 21375, 'minh': 21376, '300-million-dollar': 21377, 'world-leading': 21378, 'evaluates': 21379, 'ten-man': 21380, 'presenter': 21381, 'broad-based': 21382, 'ohn': 21383, 'bo': 21384, 'zin': 21385, 'khun': 21386, 'sai': 21387, 'nib': 21388, 'bused': 21389, 'communiqué': 21390, 'consumption-driven': 21391, 'halemi': 21392, 'pathology': 21393, 'autopsies': 21394, 'fukusho': 21395, 'shinobu': 21396, 'hasegawa': 21397, 'sight-seeing': 21398, 'disillusioned': 21399, 'run-through': 21400, 'zonen': 21401, 'rodney': 21402, 'melville': 21403, 'cyclical': 21404, 'ganey': 21405, 'firimbi': 21406, 'base-closing': 21407, 'b-1': 21408, '111th': 21409, 'rendell': 21410, 'daxing': 21411, 'meizhou': 21412, 'domineering': 21413, 'narrow-minded': 21414, 'whittaker': 21415, 'notting': 21416, '94.2': 21417, 'pre-tax': 21418, '306': 21419, '415': 21420, 'underperforming': 21421, 'basotho': 21422, 'mineworkers': 21423, 'canning': 21424, 'apparel-assembly': 21425, 'herding': 21426, 'drawback': 21427, '362.5': 21428, 'precipitously': 21429, 'sinchulu': 21430, 'ceding': 21431, 'whereby': 21432, 'krone': 21433, 'indo-bhutanese': 21434, 'singye': 21435, 'khesar': 21436, 'namgyel': 21437, 'thimphu': 21438, 'loosening': 21439, 'self-employment': 21440, 'venturing': 21441, 'moulting': 21442, 'strutted': 21443, 'cheat': 21444, 'striding': 21445, 'pecked': 21446, 'plucked': 21447, 'plumes': 21448, 'jays': 21449, 'behaviour': 21450, 'seagull': 21451, 'gullet-bag': 21452, 'spendthrift': 21453, 'pawned': 21454, 'cloak': 21455, 'solemnly': 21456, 'bougainville': 21457, 'oil-laden': 21458, 'oily': 21459, 'skimmers': 21460, 'then-hurricane': 21461, 'hookup': 21462, '1890s': 21463, 'pensacola': 21464, 'leased': 21465, 'gushing': 21466, '6,322': 21467, 'dues': 21468, 'guesthouse': 21469, 'tegua': 21470, 'manifestation': 21471, 'executes': 21472, 'time-stamped': 21473, 'newscaster': 21474, 'malfeasance': 21475, '28.9': 21476, '19.7': 21477, 'allan': 21478, 'kemakeza': 21479, 'reestablishing': 21480, 'ramsi': 21481, 'fossum': 21482, 'garan': 21483, '14-day': 21484, 'olive-tree': 21485, 'akihiko': 21486, 'shower': 21487, 'foliage': 21488, 'despoiling': 21489, 're-occupied': 21490, 'recently-halted': 21491, 'often-stalled': 21492, 'denuded': 21493, 'deceive': 21494, 'democratizing': 21495, 'bossaso': 21496, 'vaunting': 21497, 'fearlessness': 21498, 'pained': 21499, 'silvestre': 21500, 'afable': 21501, '60-member': 21502, 'malaysian-led': 21503, 'three-decade-old': 21504, 'enterprising': 21505, 'fearless': 21506, 'modernizes': 21507, 'schearf': 21508, '20/20': 21509, 'reasoned': 21510, 'swicord': 21511, 'signatories': 21512, 'rui': 21513, 'zhu': 21514, 'houze': 21515, 'pu': 21516, 'endeavoured': 21517, 'lundestad': 21518, '168': 21519, 'sheep-and-cattle': 21520, 'roswell': 21521, 'dong': 21522, 'anh': 21523, 'often-fatal': 21524, 'ha': 21525, 'tay': 21526, 'duong': 21527, 'pondicherry': 21528, 'rails': 21529, 'mehrabpur': 21530, 'welded': 21531, 'darkness': 21532, 'meterologists': 21533, 'roaring': 21534, 'spyglass': 21535, 'outdo': 21536, 'triumphs': 21537, 'icahn': 21538, 'plavia': 21539, 'levar': 21540, "jam'iyyat": 21541, 'ul-islam': 21542, 'is-saheeh': 21543, 'samana': 21544, 'latina': 21545, 'scapegoating': 21546, 'criminalizes': 21547, 'alienating': 21548, 'flocking': 21549, 'facelift': 21550, 'modernity': 21551, 'cubanization': 21552, 'amhed': 21553, 'osbek': 21554, 'castillo': 21555, 'diamondbacks': 21556, 'francisely': 21557, 'bueno': 21558, 'braves': 21559, 'ilyas': 21560, 'repressed': 21561, 'hernan': 21562, 'khetaguri': 21563, 'inguri': 21564, '0-6': 21565, '05-jan': 21566, 'jaw': 21567, 'extensively': 21568, 'violence-ridden': 21569, 'signings': 21570, 'contract-style': 21571, 'betrayal': 21572, 'definite': 21573, 'shana': 21574, 'garhi': 21575, 'ransacking': 21576, 'bula': 21577, 'hawo': 21578, 'photographic': 21579, 'maoist-led': 21580, 'baran': 21581, 'illusion': 21582, 'helmet': 21583, 'hoisting': 21584, 'aftab': 21585, 'sherpao': 21586, 'lodi': 21587, 'two-week-old': 21588, 'zazai': 21589, 'taliban-dominated': 21590, 'muniz': 21591, 'maywand': 21592, 'hosada': 21593, 'mobbed': 21594, 'post-taleban': 21595, 'butts': 21596, 'canes': 21597, 'oncologist': 21598, 'state-nominee': 21599, 'meddles': 21600, 'interferes': 21601, 'suppresses': 21602, 'counsel-general': 21603, 'checks-and-balances': 21604, 'fathers': 21605, 'judith': 21606, 'latham': 21607, 'dateline': 21608, 'isro': 21609, 'g.': 21610, 'madhavan': 21611, 'nair': 21612, 'array': 21613, 'salvation': 21614, 'sayidat': 21615, 'al-nejat': 21616, 're-structuring': 21617, 'mcguffin': 21618, '10-person': 21619, 'non-cooperation': 21620, 'harun': 21621, 'kushayb': 21622, 'uzair': 21623, 'solicited': 21624, 'india-us': 21625, '2.5-year': 21626, 'ecolog': 21627, 'summon': 21628, 'al-mustaqbal': 21629, 'liuguantun': 21630, 'tangshan': 21631, 'helland': 21632, 'meadows': 21633, 'stanilas': 21634, 'warwrinka': 21635, 'all-spanish': 21636, 'four-sets': 21637, 'gulbis': 21638, 'henin': 21639, '3,25,000': 21640, 'spellings': 21641, 'compilation': 21642, 'banco': 21643, 'championed': 21644, 'zabi': 21645, 'ul-taifi': 21646, 'taifi': 21647, "gm's": 21648, 'credit-rating': 21649, 'industry-wide': 21650, 'nanhai': 21651, 'bombardier': 21652, 'wiretapped': 21653, 'semana': 21654, 'pilar': 21655, 'thugs': 21656, 'shengyou': 21657, 'feigning': 21658, 'oda': 21659, '976': 21660, 'hok': 21661, 'buro': 21662, 'happold': 21663, 'multiple-use': 21664, 'aquatics': 21665, '476': 21666, 'balfour': 21667, 'beatty': 21668, 'reconfigured': 21669, 'chattisgarh': 21670, 'anti-maoist': 21671, 'newly-signed': 21672, 'rocco': 21673, 'buttiglione': 21674, 'sin': 21675, '472': 21676, '0.82': 21677, 'delisted': 21678, '23.25': 21679, '28.25': 21680, 'duluth': 21681, 'ga.': 21682, 'adept': 21683, 'landholders': 21684, 'jubilee': 21685, 'unmasking': 21686, 'macro-economic': 21687, 'israeli-imposed': 21688, 'unrwa': 21689, 'hamas-regulated': 21690, 'non-approved': 21691, 'astrology': 21692, 'underestimate': 21693, 'handicapped': 21694, '300-km': 21695, 'astute': 21696, 'divide-and-rule': 21697, 'esteem': 21698, 'mortals': 21699, 'messenger': 21700, 'fling': 21701, 'journeying': 21702, 'loosen': 21703, 'faggot': 21704, 'anticipations': 21705, 'strand': 21706, 'pursuer': 21707, 'recollecting': 21708, 'tumultous': 21709, 'heartless': 21710, 'skipper': 21711, "'t": 21712, 'convergent': 21713, 'murmured': 21714, 'marooned': 21715, 'shareman': 21716, 'hauls': 21717, 'pudong': 21718, 'deportees': 21719, 'qatada': 21720, 'lerner': 21721, 'sufa': 21722, 'nahal': 21723, '26-31': 21724, 'antihistamine': 21725, 'heredity': 21726, 'batter': 21727, 'vero': 21728, 'southward': 21729, 'ticketing': 21730, 'rong': 21731, 'utilized': 21732, 'oversold': 21733, 'blackhawk': 21734, 'etienne': 21735, 'tshisekedi': 21736, 'weakness': 21737, 'coahuila': 21738, 'charai': 21739, 'persists': 21740, 'tacked': 21741, 'additions': 21742, 'earmarks': 21743, 'gauteng': 21744, 'surprises': 21745, 'manama': 21746, 'dialog': 21747, 'consulting-security': 21748, 'frederic': 21749, 'piry': 21750, 'kwazulu-natal': 21751, 'sino-russian': 21752, 'once-bitter': 21753, 'hajim': 21754, 'mento': 21755, 'tshabalala-msimang': 21756, 'blacked': 21757, 'underscoring': 21758, 'palestinian-owned': 21759, 'ghassan': 21760, 'daglas': 21761, 'blazing': 21762, 'retaliating': 21763, 'caravan': 21764, 'hardcore': 21765, 'conscious': 21766, 'hayabullah': 21767, 'rafiqi': 21768, 'qabail': 21769, 'solders': 21770, '15-second': 21771, 'repsol': 21772, 'norma': 21773, 'nonspecific': 21774, 'uncorroborated': 21775, 'hermann': 21776, 'apollo': 21777, 'ohno': 21778, 'skates': 21779, '1,500-meters': 21780, 'pechstein': 21781, 'anni': 21782, 'friesinger': 21783, '3,000-meters': 21784, 'armin': 21785, 'zoeggeler': 21786, 'resolves': 21787, 'host-country': 21788, 'sciri': 21789, 'echoes': 21790, 'anti-proliferation': 21791, 'norinco': 21792, 'zibo': 21793, 'chemet': 21794, 'aero-technology': 21795, 'import-export': 21796, 'hongdu': 21797, 'ounion': 21798, 'metallurgy': 21799, 'walt': 21800, '660': 21801, 'nonrefundable': 21802, '40-million': 21803, 'tutor': 21804, 'iger': 21805, 'pixar': 21806, 'marvel': 21807, 'drawn-out': 21808, 'angeles-based': 21809, 'harvey': 21810, 'bloodthirsty': 21811, 'hated': 21812, 'monster': 21813, 'mccellan': 21814, 'bar-on': 21815, 're-arm': 21816, 'mombasa-based': 21817, '740': 21818, 'receving': 21819, 'ecuadoreans': 21820, 'tachilek': 21821, 'noel': 21822, 'imb': 21823, 'pirate-infested': 21824, 'citroen': 21825, 're-arming': 21826, 'gronholm': 21827, 'immunizing': 21828, 'duaik': 21829, 'mitsubishi': 21830, 'gigi': 21831, 'galli': 21832, 'turbo': 21833, 'charger': 21834, 'valves': 21835, 'maracaibo': 21836, '225-kilometer-long': 21837, '3:34:33.2': 21838, 'gazan': 21839, 'constituted': 21840, 'guilt': 21841, 'stripped-down': 21842, 'bickering': 21843, 'near-monoply': 21844, 'xp': 21845, 'lado': 21846, 'grigol': 21847, 'mgalobishvili': 21848, 'markko': 21849, 'technocrat': 21850, 'newhouse': 21851, 'sezibera': 21852, 'al-qassim': 21853, 'branco': 21854, 'scathing': 21855, 'giliani': 21856, 'commerical': 21857, 'traumatized': 21858, 'untreated': 21859, 'earthquake-hit': 21860, 'al-adwa': 21861, 'wlodzimierz': 21862, 'w?odzimierz': 21863, 'visa-free': 21864, 'leopoldo': 21865, 'taiana': 21866, 'dhiren': 21867, 'qaisar': 21868, 'shaffi': 21869, 'nadeem': 21870, 'tarmohammed': 21871, 'esa': 21872, 'citicorp': 21873, '1,64,000': 21874, 'foreign-born': 21875, 'graffiti': 21876, 'turner': 21877, 'durand': 21878, 'force-fed': 21879, 'strikers': 21880, 'noses': 21881, 'stomachs': 21882, 'force-feeding': 21883, 'lauder': 21884, 'schneider': 21885, 'inforadio': 21886, 'oswaldo': 21887, 'jarrin': 21888, 'unaffiliated': 21889, '3.625': 21890, '16.2': 21891, 'computer-services': 21892, 'datapoint': 21893, '2.75': 21894, 'buy-back': 21895, 'above-market': 21896, 'finfish': 21897, 'presidential-parliamentary': 21898, 'instabilities': 21899, 'martinique': 21900, 'mayotte': 21901, '05-jun': 21902, 'overpopulated': 21903, 'inefficiently-governed': 21904, 'single-most-important': 21905, '12.3': 21906, 'fy09': 21907, 'fy10': 21908, 'pulses': 21909, '42,000': 21910, 'mw': 21911, 'grasping': 21912, 'idf': 21913, 'strelets': 21914, 'anatoliy': 21915, 'theoretically': 21916, 'insurgency-plagued': 21917, 'tishrin': 21918, 'sincerity': 21919, 'israel-syria': 21920, 'co-sponsored': 21921, '1,254': 21922, 'zhouqu': 21923, '60.45': 21924, '60.95': 21925, '59.85': 21926, 'u.s.-iranian': 21927, 'watchers': 21928, 'jammed': 21929, 'eighty': 21930, 'bomb-shaped': 21931, '7.75': 21932, 'gymnasium': 21933, 'jianlian': 21934, 'fouled': 21935, 'narvaez': 21936, 'locker': 21937, 'shielding': 21938, 'wore': 21939, 't-shirts': 21940, "dvd's": 21941, 'sola': 21942, 'chilitepic': 21943, 'medan': 21944, '50.65': 21945, 'tahseen': 21946, 'poul': 21947, 'nielsen': 21948, 'farabaugh': 21949, 'thai-burmese': 21950, 'reebok': 21951, 'nord': 21952, 'eclair': 21953, 'gobernador': 21954, 'valadares': 21955, 'biloxi': 21956, 'expletive': 21957, 'bashkortostan': 21958, 'oskar': 21959, 'alekseyeva': 21960, 'rewriting': 21961, 'andrade': 21962, 'gereida': 21963, 'east-northeast': 21964, 'formalizing': 21965, 'odd': 21966, 'totalitarian': 21967, 'pinchuk': 21968, 'guilin': 21969, 'wolong': 21970, 'nestrenko': 21971, 'farsi-language': 21972, 'paseo': 21973, 'reforma': 21974, 'vasconcelos': 21975, 'nadi': 21976, 'tahab': 21977, 'hizb-ul-mujahedeen': 21978, 'snarled': 21979, '20,00,000': 21980, '40,00,000': 21981, 'inception': 21982, '30,00,000': 21983, 'congestion': 21984, 'conjunction': 21985, 'simulated': 21986, 'tarin': 21987, 'insurgent-related': 21988, 'purim': 21989, 'galloway': 21990, 'barter': 21991, 'disrepair': 21992, 'denpasar': 21993, 'czugaj': 21994, 'stephens': 21995, 'renae': 21996, 'lambasting': 21997, 'adoring': 21998, 'divisiveness': 21999, 'twenty-one-year-old': 22000, 'canister': 22001, 'sparingly': 22002, '236': 22003, 'domenici': 22004, 'reap': 22005, 're-invested': 22006, 'india-controlled': 22007, 'pro-pakistan': 22008, 'western-backed': 22009, 'vetoed': 22010, '18-a-share': 22011, 'est': 22012, '576': 22013, '19.5': 22014, 'barron': 22015, 'gem-quality': 22016, '35-40': 22017, 'distributions': 22018, 'gini': 22019, 'coefficient': 22020, 'one-to-one': 22021, 'allotments': 22022, 'germanic': 22023, '1866': 22024, 'anti-money-laundering': 22025, 'practiced': 22026, 'supranational': 22027, 'annals': 22028, 'country-level': 22029, 'nation-states': 22030, 'overarching': 22031, 'yemenis': 22032, 'delimitation': 22033, 'huthi': 22034, 'zaydi': 22035, 'revitalized': 22036, "sana'a": 22037, 'sprain': 22038, 'racquetball': 22039, 'loosed': 22040, 'coil': 22041, 'irritated': 22042, 'rustic': 22043, 'ignorant': 22044, 'induce': 22045, 'semi-annual': 22046, 'mused': 22047, 'ilyushin': 22048, 'rosoboronexport': 22049, 'nightline': 22050, 'balboa': 22051, 'embarking': 22052, '100-year': 22053, '232nd': 22054, 'risking': 22055, 'juste': 22056, 'moradi': 22057, 'self-promotion': 22058, 'avoidance': 22059, 'yoshimasa': 22060, 'hayashi': 22061, 'led-invasion': 22062, 'jinping': 22063, 'undervalued': 22064, 'sunni-dominated': 22065, '2nd': 22066, '502nd': 22067, 'scholarships': 22068, 'thousand-strong': 22069, 'specter': 22070, 'contradicted': 22071, 'government-subsidized': 22072, 'nsanje': 22073, 'italian-run': 22074, 'pepfar': 22075, 'maroua': 22076, 'rania': 22077, 'joyous': 22078, 'eni': 22079, 'wintry': 22080, 'dictates': 22081, 'pro-u.s.-immigration': 22082, 'truck-mounted': 22083, 'ivanovo': 22084, 'teikovo': 22085, 'one-ton': 22086, 'countering': 22087, 'macarthur': 22088, 'parcels': 22089, 'make-shift': 22090, 'el-zahraa': 22091, 'etman': 22092, 'twenty-seven': 22093, 'compel': 22094, 'fuel-making': 22095, '92-years-old': 22096, 'exuberant': 22097, 'guys': 22098, 'can-can': 22099, 'brides': 22100, 'chanthalangsy': 22101, 'non-sawang': 22102, 'frigid': 22103, 'dikweneh': 22104, 'ambulances': 22105, 'paddick': 22106, 'cell-phone': 22107, 'russell': 22108, 'madaen': 22109, 'intermediaries': 22110, 'kuwaitis': 22111, 'al-mutairi': 22112, 'bloemfontein': 22113, '2,720': 22114, 'lieu': 22115, 'conjured': 22116, 'gasparovic': 22117, 'wholesalers': 22118, '162-page': 22119, 'koufax': 22120, 'wilpon': 22121, '70-year-old': 22122, 'years-long': 22123, 'causality': 22124, 'baladiyat': 22125, 'medically-induced': 22126, 'shlomo': 22127, 'mor-yosef': 22128, 'anesthesia': 22129, 'dosage': 22130, 'sedated': 22131, 'unconsciousness': 22132, '481-': 22133, 'vase': 22134, 'sutham': 22135, 'saengprathum': 22136, 'kelantan': 22137, '2.7-million': 22138, 'polytechnic': 22139, 'backcountry': 22140, 'babri': 22141, 'god-king': 22142, 'rama': 22143, 'shiv': 22144, 'sena': 22145, 'food-borne': 22146, '784': 22147, 'heishan': 22148, 'seyoum': 22149, 'mesfin': 22150, 'awan': 22151, 'cayenne': 22152, 'khanun': 22153, 'talim': 22154, 'disfigured': 22155, 'mats': 22156, 'talal': 22157, 'sattar': 22158, 'qassem': 22159, 'suha': 22160, 'jean-jacques': 22161, 'dordain': 22162, 'chemical-plant': 22163, 'shoigu': 22164, 'foul': 22165, 'interruption': 22166, 'outed': 22167, 'sad': 22168, 'revelation': 22169, 'defiled': 22170, 'diligence': 22171, 'cross-strait': 22172, 'choco': 22173, '1.31': 22174, 'record-low': 22175, 'wines': 22176, '7.00e+07': 22177, 'airasia': 22178, 'al-gomhouria': 22179, 'trumped-up': 22180, 'ralia': 22181, 'thrived': 22182, 'hand-held': 22183, '100-million': 22184, '92.4': 22185, '104.1': 22186, '1030': 22187, 'cranks': 22188, 'thayer': 22189, 'kadhim': 22190, 'al-suraiwi': 22191, 'suraiwi': 22192, 'newly-released': 22193, 'defacate': 22194, 'u.s.-launched': 22195, 'finely': 22196, 'armchairs': 22197, 'punitive': 22198, 'theorize': 22199, 'verbytsky': 22200, 'sandbag-reinforced': 22201, 'winfield': 22202, 'inundate': 22203, 'embankments': 22204, 'rain-flooded': 22205, 'ruining': 22206, 'soybeans': 22207, 'skyrocketed': 22208, 'anti-royalists': 22209, 'sahafi': 22210, 'all-cash': 22211, 'liquidation': 22212, 'dividends': 22213, 'redeemed': 22214, '1652': 22215, 'spice': 22216, 'trekked': 22217, 'subjugation': 22218, 'encroachments': 22219, 'boer': 22220, 'afrikaners': 22221, 'whites-only': 22222, 'multi-racial': 22223, 'anc-led': 22224, 'kgalema': 22225, 'motlanthe': 22226, 'general-secretary': 22227, 'promarket': 22228, 'export-led': 22229, 'narcotrafficking': 22230, 'us-colombia': 22231, 'delimited': 22232, 'tokugawa': 22233, 'shogunate': 22234, 'flowering': 22235, 'kanagawa': 22236, 'industrialize': 22237, 'formosa': 22238, 'sakhalin': 22239, '1931': 22240, 'manchuria': 22241, 'strongest-ever': 22242, 'honshu': 22243, 'saibou': 22244, 'madhuri': 22245, 'duration': 22246, 'subsistence-based': 22247, 'sahel': 22248, 'nigerien': 22249, 'growling': 22250, 'snapping': 22251, 'wrathful': 22252, 'tyrannical': 22253, 'gentle': 22254, 'panther': 22255, 'amity': 22256, 'coils': 22257, 'life-and-death': 22258, 'spat': 22259, 'drinking-horn': 22260, 'slake': 22261, 'draught': 22262, 'juror': 22263, 'housecat': 22264, 'de-furred': 22265, 'yong-chun': 22266, 'unimaginably': 22267, 'black-hearted': 22268, 'lurking': 22269, 'yitzhak': 22270, 'leah': 22271, 'land-for-peace': 22272, 'herzl': 22273, 'defaced': 22274, 'neo': 22275, 'beilin': 22276, 'gurion': 22277, 'graveyard': 22278, 'rooting-out': 22279, 'sadah': 22280, '240-8': 22281, 'shivnarine': 22282, '297': 22283, '214': 22284, 'makhaya': 22285, 'feb-63': 22286, 'www.gov.cn': 22287, 'bucca': 22288, '64.7': 22289, 'biographies': 22290, '10-man': 22291, 'editor-in-chief': 22292, 'miscommunication': 22293, 'offloaded': 22294, 'anti-whaling': 22295, 'cetacean': 22296, 'cabeza': 22297, 'vaca': 22298, 'us-mexican': 22299, 'shoko': 22300, 'muallem': 22301, 'sheiria': 22302, 'union-sponsored': 22303, 'iraq-syria': 22304, 'over-the': 22305, 'riad': 22306, 'dovonou': 22307, 'adjarra': 22308, 'porto': 22309, 'novo': 22310, 'good-bye': 22311, 'scheussel': 22312, 'pelting': 22313, 'incapacitating': 22314, 'poors': 22315, 'well-qualified': 22316, 'paused': 22317, 'streetcar': 22318, 'departs': 22319, 'tecnica': 22320, 'loja': 22321, 'universidad': 22322, 'andes': 22323, '2,633-meter': 22324, '0.068472222': 22325, '0.17': 22326, 'nike': 22327, '1.38.53': 22328, 'downhills': 22329, 'second-place': 22330, '882': 22331, '197': 22332, '592': 22333, 'presses': 22334, 'jensen': 22335, 'phuket': 22336, 'yazoo': 22337, 'incomplete': 22338, 'stay-order': 22339, 'anup': 22340, 'raj': 22341, 'co-director': 22342, 'jabril': 22343, 'abdulle': 22344, 'mauritian': 22345, 'grouper': 22346, 'eels': 22347, 'medically': 22348, 'chaka': 22349, 'fattah': 22350, 'fills': 22351, 'gunfights': 22352, 'copacabana': 22353, 'ipanema': 22354, 'legalize': 22355, 'taiwanese-american': 22356, 'shen': 22357, 'gregg': 22358, 'jawid': 22359, 'three-phased': 22360, 'tichaona': 22361, '68-year-old': 22362, 'traitors': 22363, 'all-race': 22364, 'withered': 22365, 'helpe': 22366, 'chasseurs-volontaires': 22367, 'saint-domingue': 22368, '1779': 22369, 'turb': 22370, 'ulent': 22371, 'passy': 22372, '2035': 22373, 'mid-century': 22374, 'sahrawi': 22375, 'bart': 22376, 'stupak': 22377, 'over-pricing': 22378, 'grammyawards': 22379, 'domm': 22380, 'samo': 22381, 'parra': 22382, 'mientes': 22383, 'dejarte': 22384, 'amar': 22385, 'guerraas': 22386, 'bachata': 22387, 'fukuoko': 22388, 'univision': 22389, 'actress-singer': 22390, 'judgmental': 22391, 'starlet': 22392, 'richie': 22393, 'councilor': 22394, 'jiaxuan': 22395, 'lehman': 22396, 'ubs': 22397, 'stearns': 22398, 'renowed': 22399, 'farka': 22400, 'genre': 22401, 'timbuktu': 22402, 'belet': 22403, 'weyn': 22404, 'over-threw': 22405, 'in-fighting': 22406, 'mud-brick': 22407, 'heat-shielding': 22408, 'compartment': 22409, 'latifiya': 22410, 'al-hafidh': 22411, 'al-khalis': 22412, 'deferring': 22413, 'amani': 22414, 'wreaking': 22415, 'laissez-faire': 22416, 'archaic': 22417, 'service-oriented': 22418, 'entrepot': 22419, 'ballooning': 22420, 'israeli-hizballah': 22421, 'conditioned': 22422, 'mayen': 22423, '2-2.5': 22424, '27-year-long': 22425, 'accrued': 22426, 'kwanza': 22427, 'depreciated': 22428, 'fishery': 22429, 'working-age': 22430, 'rutile': 22431, 'protectorates': 22432, 'british-ruled': 22433, 'malaya': 22434, 'sarawak': 22435, 'armourer': 22436, 'dart': 22437, 'fangs': 22438, 'insensible': 22439, 'salesmen': 22440, 'dina': 22441, 'cody': 22442, 'month-to-month': 22443, 'slumps': 22444, '7,000-member': 22445, 'min-soon': 22446, 'asia-oceania': 22447, 'incidences': 22448, 'forcible': 22449, 'al-mabhouh': 22450, 'circulate': 22451, 'deyda': 22452, 'berezovsky': 22453, 'akhmed': 22454, 'zakayev': 22455, 'greenest': 22456, 'soothe': 22457, 'counter-proposal': 22458, 'h9n2': 22459, 'subtypes': 22460, '..': 22461, 'mansehra': 22462, 'emerges': 22463, 'lots': 22464, 'rubies': 22465, 'auctions': 22466, 'nishin': 22467, '2,012': 22468, 'asphyxiated': 22469, 'salvatore': 22470, 'sagues': 22471, 'prosecutions': 22472, 'kandill': 22473, 'faultline': 22474, 'chenembiri': 22475, 'bhunu': 22476, 'ranged': 22477, 'obiang': 22478, 'nguema': 22479, 're-route': 22480, 'unspoiled': 22481, 'maciej': 22482, 'nowicki': 22483, 'rospuda': 22484, 'pristine': 22485, 'bog': 22486, 'lynx': 22487, 'augustow': 22488, 'fayoum': 22489, 'michalak': 22490, 'thuan': 22491, 'congressional-executive': 22492, 'clouded': 22493, 'harass': 22494, 'hofstad': 22495, 'nansan': 22496, 'brader': 22497, 'insurgency-hit': 22498, 'soothing': 22499, 'teng-hui': 22500, 'well-equipped': 22501, 'perpetrating': 22502, 'erkki': 22503, 'tuomioja': 22504, 'haisori': 22505, 'destabilization': 22506, 'rugigana': 22507, 'faustin': 22508, 'kayumba': 22509, 'multimillion-dollar': 22510, 'develops': 22511, 'izzat': 22512, 'al-douri': 22513, '80-percent': 22514, 'ration': 22515, '23.30': 22516, 'cosmonaut': 22517, '11,500': 22518, 'grujic': 22519, 'decadence': 22520, 'aspire': 22521, 'perfection': 22522, 'oberhausen': 22523, 'eight-legged': 22524, 'mussels': 22525, 'peacemaker': 22526, 'reconstructing': 22527, 'wayaobao': 22528, 'zichang': 22529, 'howells': 22530, '277': 22531, 'kantathi': 22532, 'supamongkhon': 22533, 'spicy': 22534, 'garlicky': 22535, 'epitomizes': 22536, 'zero-gravity': 22537, 'pre-launch': 22538, 'alzouma': 22539, 'yada': 22540, 'adamou': 22541, 'cuvette': 22542, 'heath': 22543, 'anura': 22544, 'bandaranaike': 22545, 'amunugama': 22546, 'mediocre': 22547, 'peril': 22548, 'isaiah': 22549, 'unpublished': 22550, 'authored': 22551, '1.294': 22552, '12-nation': 22553, 'surpasses': 22554, '1.2927': 22555, 'polarize': 22556, 'interlagos': 22557, '0.060092593': 22558, '0.050474537': 22559, 'kimi': 22560, '0.193': 22561, 'realistically': 22562, 'schumacher': 22563, '76,000': 22564, 'quails': 22565, 'prolific': 22566, 'next-to-last': 22567, 'non-functional': 22568, 'joao': 22569, 'billboard': 22570, 'million-selling': 22571, 'vocals': 22572, 'most-successful': 22573, 'lamm': 22574, 'loughnane': 22575, 'sandagiri': 22576, 'malam': 22577, 'bacai': 22578, 'tamil-dominated': 22579, 'toddler': 22580, '2,698': 22581, 'ethnically-tibetan': 22582, 'yushu': 22583, 'guangrong': 22584, 'kamau': 22585, 'artur': 22586, 'marina': 22587, 'otalora': 22588, 'condoleeza': 22589, 'ludicrous': 22590, 'barricaded': 22591, 'shangla': 22592, 'sarfraz': 22593, 'naeemi': 22594, 'kingsbridge': 22595, '45-a-share': 22596, 'donuts': 22597, '50.1': 22598, 'dunkin': 22599, '468': 22600, '38.5': 22601, 'randolph': 22602, 'mass.': 22603, 'locally-controlled': 22604, 'gaming-related': 22605, '5,52,300': 22606, 'casinos': 22607, 'cepa': 22608, 'macau-made': 22609, 'pataca': 22610, 'multi-sector': 22611, 'riskier': 22612, 'revisions': 22613, '1889': 22614, '1,26,976': 22615, '1,04,586': 22616, 'euphausia': 22617, 'superba': 22618, '12,027': 22619, 'dissostichus': 22620, 'eleginoides': 22621, 'bass': 22622, '1,27,910': 22623, '1,06,591': 22624, '9.7': 22625, '12,396': 22626, 'ccamlr': 22627, 'unregulated': 22628, '8,376': 22629, '45,213': 22630, '35,552': 22631, '29,799': 22632, 'iaato': 22633, 'sideline': 22634, 'uniosil': 22635, 'furthering': 22636, 'fatherland': 22637, 'wintertime': 22638, 'curled': 22639, 'tee': 22640, 'olden': 22641, 'enchanted': 22642, 'imitate': 22643, 'dull': 22644, 'industrious': 22645, 'seventy': 22646, 'poetry': 22647, 'honoured': 22648, 'compiler': 22649, 'volumes': 22650, 'tabulated': 22651, 'lustily': 22652, 'behold': 22653, 'goeth': 22654, 'hiding-place': 22655, 'calamitously': 22656, 'prowl': 22657, 'dwell': 22658, 'whichever': 22659, 'a-quarrelling': 22660, '138.28': 22661, 'pep': 22662, 'redirect': 22663, 'groomed': 22664, 'sim': 22665, 'fixed-line': 22666, 'unicom': 22667, '4.11': 22668, '628': 22669, 'westerly': 22670, 'siegler': 22671, 'fouls': 22672, 'nrc': 22673, 'handelsblad': 22674, 'henk': 22675, 'forested': 22676, 'lays': 22677, 'iron-ore': 22678, 'orlando': 22679, 'cachaito': 22680, 'buena': 22681, 'compay': 22682, 'segundo': 22683, '100.8': 22684, 'nurja': 22685, 'once-popular': 22686, 'wide-opened': 22687, 're-invest': 22688, 'chief-of-staff': 22689, 'euthanized': 22690, 'pets': 22691, 'cares': 22692, 'al-hindawi': 22693, 'duluiya': 22694, 'pakistani-afghan': 22695, 'machinea': 22696, 'jaatanni': 22697, 'taadhii': 22698, 'upright': 22699, 'two-wheeled': 22700, 'one-meter-long': 22701, 'gyroscopes': 22702, 'tricky': 22703, 'scooters': 22704, 'warrantless': 22705, 'non-opec': 22706, 'mascots': 22707, 'floribert': 22708, 'integrationist': 22709, 'precedence': 22710, 'rigoberta': 22711, 'menchu': 22712, 'jokers': 22713, 'forcers': 22714, 'alliance-head': 22715, 'contradicting': 22716, 'cheats': 22717, 'dhia': 22718, 'assimilating': 22719, 'visually': 22720, 'impaired': 22721, 'ordina': 22722, 'den': 22723, 'taping': 22724, 'klara': 22725, 'ljube': 22726, 'ljubotno': 22727, 'skopje': 22728, 'pre-2007': 22729, 'radisson': 22730, 'militarization': 22731, 'telecast': 22732, 'hizb-e-islami': 22733, 'weaponization': 22734, 'singnaghi': 22735, 'tugboats': 22736, 'mavi': 22737, 'marmaraout': 22738, 'neuilly': 22739, 'marja': 22740, 'jalani': 22741, 'kish': 22742, 'duffle': 22743, '218-5': 22744, 'ashwell': 22745, '87-run': 22746, '131-5': 22747, 'dwayne': 22748, 'jerome': 22749, 'feb-45': 22750, 'germ': 22751, 'rihab': 22752, 'huda': 22753, 'amash': 22754, 'marwa': 22755, 'high-value': 22756, 'brookings': 22757, 'camaraderie': 22758, 'outdoors': 22759, 'monaliza': 22760, 'noormohammadi': 22761, 'porous': 22762, 'iraqi-syrian': 22763, 'thunderous': 22764, '58,000': 22765, 'al-uthmani': 22766, 'funnel': 22767, 'luciano': 22768, 'modena': 22769, 'polyclinic': 22770, 'pancreatic': 22771, 'aficionados': 22772, 'tenor': 22773, 'four-million': 22774, 'minghe': 22775, '80-kilometer-long': 22776, 'salad': 22777, 'pakistan-bound': 22778, 'sixty-eight': 22779, 'crooner': 22780, 'shrugged': 22781, 'suichuan': 22782, '69.54': 22783, 'mala': 22784, 'shorish': 22785, 'akre-bijeel': 22786, 'shaikan': 22787, 'rovi': 22788, 'sarta': 22789, 'dihok': 22790, 'chimneys': 22791, 'embarrased': 22792, 'halftime': 22793, 'performer': 22794, 'jacksonville': 22795, 'bared': 22796, 'mccarthy': 22797, 'facets': 22798, '12-minute': 22799, 'contenders': 22800, 'baya': 22801, 'rahho': 22802, 'archeologist': 22803, 're-energize': 22804, 'verifiably': 22805, 'pontchartrain': 22806, 'megumi': 22807, 'faking': 22808, 'anti-pyongyang': 22809, 'mocks': 22810, 'three-thousand': 22811, 'hamesh': 22812, 'koreb': 22813, 'trawler': 22814, '61,000': 22815, 'hmawbi': 22816, 'six-kilometers': 22817, 'gracious': 22818, 'bijbehera': 22819, 'amor': 22820, 'almagro': 22821, 'buyat': 22822, 'aridi': 22823, 'iranian-': 22824, 'bugojno': 22825, 'twa': 22826, 'stethem': 22827, 'lovers': 22828, '4,410-kilogram': 22829, 'guinness': 22830, 'yerevan': 22831, '3,587-kilogram': 22832, 'elah': 22833, 'dufour-novi': 22834, 'alessandria': 22835, 'piemonte': 22836, '35.2': 22837, 'stamford': 22838, 'magnified': 22839, 'nonrecurring': 22840, 'asset-valuation': 22841, '85.7': 22842, '93.3': 22843, 'geodynamic': 22844, 'kos': 22845, 'astypaleia': 22846, 'excels': 22847, 'self-sufficiency': 22848, 'contractions': 22849, 'post-recession': 22850, 'khmers': 22851, 'angkor': 22852, 'cham': 22853, '1887': 22854, 'pol': 22855, 'pot': 22856, 'normalcy': 22857, 'un-cambodian': 22858, 'contending': 22859, 'sihanouk': 22860, 'sihamoni': 22861, 'ageing': 22862, 'fy06/07': 22863, 'earns': 22864, 'penning': 22865, 'antagonist': 22866, 'leisurely': 22867, 'sauntering': 22868, 'cheer': 22869, 'schemed': 22870, 'tailless': 22871, 'brush': 22872, 'spun': 22873, 'effects-laden': 22874, 'shahab-3': 22875, 'opening-day': 22876, '31.4': 22877, 'salma': 22878, 'valentina': 22879, 'paloma': 22880, 'francois-henri': 22881, 'ventanazul': 22882, 'metro-goldwyn-mayer': 22883, 'ppr': 22884, 'labels': 22885, 'gucci': 22886, 'balenciaga': 22887, 'puma': 22888, '159th': 22889, 'paynesville': 22890, 'congotown': 22891, 'needlessly': 22892, 'antagonizes': 22893, 'uncalculated': 22894, 'melih': 22895, 'gokcek': 22896, 'drapchi': 22897, 'otton': 22898, '351': 22899, '187': 22900, 'heraldo': 22901, 'munoz': 22902, 'imphal': 22903, 'okram': 22904, 'ibobi': 22905, '282-4': 22906, 'rain-interrupted': 22907, 'hussey': 22908, 'lbw': 22909, 'record-tying': 22910, 'meng': 22911, 'diuretic': 22912, 'by-products': 22913, 'hua': 22914, 'swimmer': 22915, 'ouyang': 22916, 'kunpeng': 22917, '281-4': 22918, 'easy-paced': 22919, '267': 22920, 'authorites': 22921, 'outlaws': 22922, 'sea-based': 22923, 'bulava': 22924, 'stavropol': 22925, 'delanoe': 22926, 'coe': 22927, 'suite': 22928, 'yimou': 22929, 'three-and-one-half': 22930, 'rocket-launched': 22931, 'perina': 22932, 'abdul-hamid': 22933, 'non-smoking': 22934, 'anti-smoking': 22935, 'artibonite': 22936, 'refocus': 22937, 'azimbek': 22938, 'beknazarov': 22939, 'nepotism': 22940, 'adolescence': 22941, 'adolescents': 22942, 'full-page': 22943, 'reciprocate': 22944, 'cold-weather': 22945, 'verifiable': 22946, '14-kilometer-long': 22947, 'yichang': 22948, 'wanzhou': 22949, 'downpours': 22950, 'mujhava': 22951, 'cuba-venezuela': 22952, 'commonplace': 22953, 'reshape': 22954, 'broadening': 22955, 'g.a': 22956, 'ojedokun': 22957, 'screeners': 22958, 'baggage': 22959, '9,15,000': 22960, 'glut': 22961, 'unsold': 22962, 'giovanni': 22963, 'fava': 22964, 'lawmakes': 22965, '39-29': 22966, 'overdrafts': 22967, 'kallenberger': 22968, 'bellinger': 22969, '103.9': 22970, 'ammonium': 22971, 'anfo': 22972, 'deactivate': 22973, 'backups': 22974, 'maccabi': 22975, 'qualifying-round': 22976, 'second-leg': 22977, 'betar': 22978, 'dinamo': 22979, 'nikolas': 22980, 'clichy-sous-bois': 22981, 'counter-radicalization': 22982, 'businesswomen': 22983, '8,900': 22984, 'memento': 22985, 'sherpa': 22986, 'unfurled': 22987, 'rededicate': 22988, 'cia-backed': 22989, 'commemorated': 22990, 'fundraisers': 22991, 're-sales': 22992, 'itinerary': 22993, 's.p.': 22994, 'thamilselvan': 22995, 'vidar': 22996, 'russian-georgian': 22997, 'gagra': 22998, 'subzero': 22999, 'slippery': 23000, 'blizzard-like': 23001, 'motels': 23002, 'exclusion': 23003, 'haruna': 23004, 'idrissu': 23005, 're-imposition': 23006, 'softened': 23007, 'adopts': 23008, 'jaua': 23009, 'speculating': 23010, 'climate-friendly': 23011, 'paralyzing': 23012, 'tri': 23013, 'daklak': 23014, 'suicidal': 23015, 'allots': 23016, 'immorality': 23017, 'non-islamic': 23018, 're-impose': 23019, 'repealed': 23020, 'shoichi': 23021, 'nakagawa': 23022, 'penn': 23023, 'then-director': 23024, 'roundly': 23025, 'testifies': 23026, 'natural-gas': 23027, 'fixed-price': 23028, 'reimbursed': 23029, 'amortization': 23030, '169.9': 23031, 'sugar-based': 23032, 'lingered': 23033, 'precipitated': 23034, '1623': 23035, 'rebelled': 23036, 'sekou': 23037, 'sekouba': 23038, 'konate': 23039, 'conde': 23040, 'trillion-dollar': 23041, 'us-canada': 23042, 'nafta': 23043, 'absorbs': 23044, 'buffeted': 23045, 'capitalization': 23046, 'stuffing': 23047, 'goose-quills': 23048, 'aux': 23049, 'dames': 23050, 'wearily': 23051, 'wallets': 23052, 'etc': 23053, 'plaster': 23054, 'noticeable': 23055, 'confidently': 23056, 'rowdy': 23057, 'busied': 23058, 'flap': 23059, 'stapler': 23060, 'stapled': 23061, 'debilitating': 23062, 'addendum': 23063, 'debriefing': 23064, 'weapons-related': 23065, 'mekorot': 23066, 'bursa': 23067, 'compatriots': 23068, 'modifies': 23069, 'enhances': 23070, 'detriment': 23071, 'ishac': 23072, 'diwan': 23073, 'qusai': 23074, 'abdul-wahab': 23075, 'bahrainian': 23076, 'satisfies': 23077, 'verhofstadt': 23078, 'sol': 23079, 'tuzla': 23080, 'gijs': 23081, 'vries': 23082, 'muriel': 23083, 'degauque': 23084, 'anyama': 23085, 'gendarmes': 23086, 'newman': 23087, 'flagrantly': 23088, 'semester': 23089, 'tulane': 23090, 'trailers': 23091, 'flood-damaged': 23092, 'moinuddin': 23093, 'five-star': 23094, 'm.c.': 23095, 'puri': 23096, 'affirming': 23097, 'fayyum': 23098, 'giza': 23099, 'filip': 23100, 'tkachev': 23101, 'krasnodar': 23102, 'beatrice': 23103, 'norton': 23104, 'toby': 23105, 'harnden': 23106, 'simmonds': 23107, 'workings': 23108, 'vetoes': 23109, 'sorting': 23110, 'sabotaging': 23111, 'schmit': 23112, 'zabayda': 23113, 'chitchai': 23114, 'wannasathit': 23115, 'navarro': 23116, 'durango': 23117, 'resembled': 23118, 'incendiary': 23119, 'pardo': 23120, 'al-sharjee': 23121, 'frangiskos': 23122, 'ragoussis': 23123, 'castoff': 23124, 'cinderella': 23125, 'dreamgirls': 23126, 'miro': 23127, 'yousifiyah': 23128, 'simplifying': 23129, 'ler': 23130, 'minesweepers': 23131, 'airdrops': 23132, 'karkh': 23133, 'single-worst': 23134, 'al-qaida-in-iraq': 23135, 'ogero': 23136, '40.3': 23137, '4,56,000': 23138, '2,55,000': 23139, 'compromised': 23140, 'regaining': 23141, 'ping-kun': 23142, 'ghangzhou': 23143, 'yat': 23144, 'nangjing': 23145, 'slideshow': 23146, 'sgt.': 23147, 'mounts': 23148, '74th': 23149, 'researching': 23150, 'malwiya': 23151, 'jagged': 23152, 'brick': 23153, 'abbassid': 23154, 'sunni-led': 23155, 'alps': 23156, 'closeness': 23157, 'star-studded': 23158, 'multitudes': 23159, 'transfusion': 23160, 'vincennes': 23161, 'rockers': 23162, 'lynyrd': 23163, 'skynyrd': 23164, 'butch': 23165, 'kievenaar': 23166, 'blondie': 23167, 'trumpeter': 23168, 'herb': 23169, 'alpert': 23170, 'moss': 23171, 'founders': 23172, 'a&m': 23173, 'ethnically-divided': 23174, 'lafayette': 23175, 'novosibirsk': 23176, 'stymied': 23177, 'inspires': 23178, 'hubris': 23179, 'anonymously': 23180, 'on-scene': 23181, 'festive': 23182, 'decked': 23183, 'zhanjun': 23184, 'accredited': 23185, 'binjie': 23186, 'brazilian-made': 23187, 'aircraft-maker': 23188, 'embraer': 23189, 'ramping': 23190, "sh'ite": 23191, 'sheng': 23192, 'huaren': 23193, 'simmons': 23194, 'fayyaz': 23195, 'cobbled': 23196, '150-member': 23197, 'mortgages': 23198, 'industry-backed': 23199, 'refinancing': 23200, '2.35': 23201, '14.75': 23202, 'customized': 23203, 'simulates': 23204, 'juliana': 23205, 'weisgan': 23206, 'tourism-driven': 23207, 'bahamian': 23208, 'tenth': 23209, 'nyasaland': 23210, 'hastings': 23211, 'kamuzu': 23212, 'sprat': 23213, 'umpire': 23214, 'orchard': 23215, 'ripening': 23216, 'quench': 23217, 'paces': 23218, 'tempting': 23219, 'despise': 23220, 'butlers': 23221, 'chauffeurs': 23222, 'circle-k': 23223, 'backfired': 23224, 'obscene': 23225, 'tourniquet': 23226, 'noticing': 23227, 't-shirt': 23228, 'slyly': 23229, 'fractions': 23230, 'andrea': 23231, 'nahles': 23232, 'kajo': 23233, 'wasserhoevel': 23234, 'solbes': 23235, 'constructing': 23236, 'firehouses': 23237, 'non-government': 23238, 'saudi-syrian': 23239, 'pre-condition': 23240, 'mm': 23241, 'wesson': 23242, 'visitor': 23243, 'unfastened': 23244, 'journeyed': 23245, 'russian-born': 23246, 'sergey': 23247, 'tiffany': 23248, 'stiegler': 23249, 'rebecca': 23250, 'naturalization': 23251, 'tanith': 23252, 'belbin': 23253, 'shagh': 23254, 'initialed': 23255, '15.6': 23256, 'seydou': 23257, 'diarra': 23258, 'thundershowers': 23259, 'rounder': 23260, 'nel': 23261, 'boeta': 23262, 'dippenaar': 23263, 'wai': 23264, 'stylized': 23265, 'sci-fi': 23266, '2047': 23267, 'bernando': 23268, 'vavuniya': 23269, 'beatle': 23270, 'bogdanchikov': 23271, 'allegra': 23272, 'anorexia': 23273, 'zimmerman': 23274, 'estanged': 23275, 'correspond': 23276, '5,64,000': 23277, '3,268': 23278, 'akhalkalaki': 23279, 'nations-hosted': 23280, 'inject': 23281, '60-to-36': 23282, '2,40,000': 23283, 'otri': 23284, 'anti-iraqi': 23285, 'singapore-flagged': 23286, 'pramoni': 23287, 'frechette': 23288, 'plaque': 23289, '50-1': 23290, '616-5': 23291, 'delighted': 23292, 'holing': 23293, 'sohail': 23294, 'tanvir': 23295, 'dhoni': 23296, 'three-test': 23297, 'waw': 23298, 'barge': 23299, 'trudging': 23300, 'jogged': 23301, 'biked': 23302, '5-0': 23303, 'emilie': 23304, 'loit': 23305, 'ruano': 23306, 'bleach': 23307, 'russia-austria': 23308, 'kick-start': 23309, 'cereals': 23310, 'smerdon': 23311, 'al-fahd': 23312, '2.3-million-strong': 23313, 'awaited': 23314, 'mrksic': 23315, 'radic': 23316, 'veselin': 23317, 'sljivancanin': 23318, '534': 23319, 'energy-producing': 23320, 'reforestation': 23321, 'near-total': 23322, 'robby': 23323, 'yeu-tzuoo': 23324, 'sung-ho': 23325, 'neurosurgeon': 23326, 'nihon': 23327, 'keizai': 23328, 'pioneered': 23329, 'fuel-powered': 23330, 'economical': 23331, 'gwangju': 23332, '3,800': 23333, '206': 23334, 'tomislav': 23335, 'nikolic': 23336, 'anti-western': 23337, 'qua-ha': 23338, 'subspecies': 23339, 'reinhold': 23340, 'rau': 23341, 'non-nuclear': 23342, 'pakitka': 23343, 'thrust': 23344, 'submitting': 23345, 'diocese': 23346, 'wash': 23347, 'liturgy': 23348, 'mexican-american': 23349, 'disapproved': 23350, 'clamping': 23351, 'rightists': 23352, 'coaltion': 23353, 'ceramic': 23354, 'four-month-old': 23355, 'murugupillai': 23356, 'jenita': 23357, 'jeyarajah': 23358, 'crested': 23359, 'galam': 23360, 'playground': 23361, 'aviaries': 23362, 'ghanem': 23363, 'deflecting': 23364, 'nuria': 23365, 'llagostera': 23366, 'vives': 23367, 'fourth-seed': 23368, 'aryanto': 23369, 'boedihardjo': 23370, 'aip': 23371, 'hidayat': 23372, 'ciamis': 23373, 'regency': 23374, 'salik': 23375, 'firdaus': 23376, 'misno': 23377, 'lauro': 23378, 'iraq-poverty': 23379, 'savors': 23380, 'martinis': 23381, 'chef': 23382, 'rep.': 23383, 'd-ohio': 23384, 'freer': 23385, '558-page': 23386, 'clotting': 23387, 'co-defendents': 23388, 'al-zubeidi': 23389, '65.32': 23390, 'hasbrouk': 23391, 'n.j.': 23392, 'cash-flow': 23393, 'debenture': 23394, '158': 23395, '666': 23396, '26,956': 23397, 'zalkin': 23398, '608': 23399, '413': 23400, '967': 23401, '809': 23402, 'muscovy': 23403, 'absorb': 23404, 'romanov': 23405, '1682': 23406, '1725': 23407, 'hegemony': 23408, 'defeats': 23409, 'lenin': 23410, 'iosif': 23411, 'glasnost': 23412, 'perestroika': 23413, 'semi-authoritarian': 23414, 'buttressed': 23415, 'seaports': 23416, 'dislocation': 23417, '122.8': 23418, 'barbuda': 23419, 'dual-island': 23420, 'enclave-type': 23421, 'spencer': 23422, 'antiguan': 23423, 'yawning': 23424, 'disadvantages': 23425, 'abject': 23426, 'equaling': 23427, 'gardening': 23428, 'reengagement': 23429, 'mis-2009': 23430, 'outlays': 23431, 'lime': 23432, 'coconut': 23433, 'pitied': 23434, 'bough': 23435, 'fowling-piece': 23436, 'lambs': 23437, 'pupil': 23438, 'lookout': 23439, 'famishing': 23440, 'babe': 23441, 'sneered': 23442, "i'd": 23443, 'compliment': 23444, 'riek': 23445, 'homestead': 23446, 'nyongwa': 23447, 'ankunda': 23448, 'remarked': 23449, 'weakly': 23450, '8.7-magnitude': 23451, 'jolt': 23452, 'fy08/09': 23453, 'tartus': 23454, 'decimated': 23455, 'doodipora': 23456, 'azad': 23457, 'baqer': 23458, 'constitutions': 23459, 'ecumenical': 23460, 'chrysostom': 23461, 'nazianzen': 23462, '1054': 23463, 'sacking': 23464, '1204': 23465, 'salome': 23466, 'zurabishvili': 23467, 'inforamtion': 23468, 'petroleum-based': 23469, 'institutionalizing': 23470, 'squibb': 23471, 'single-dose': 23472, 'sustiva': 23473, 'viread': 23474, 'emtriva': 23475, 'replication': 23476, '140-kilometers': 23477, 'civilian-to-civilian': 23478, 'malaki': 23479, 'hiker': 23480, 'untied': 23481, 'bittersweet': 23482, 'concertacion': 23483, 'foxley': 23484, '1824': 23485, 'notari': 23486, 'self-appointed': 23487, 'involuntarily': 23488, 'tubes': 23489, 'murals': 23490, 'excavation': 23491, '4,000-year-old': 23492, 'mummies': 23493, 'scribe': 23494, 'motorized': 23495, 'rickshaw': 23496, 'hello': 23497, 'collectivization': 23498, 'foreign-run': 23499, 'prodemocracy': 23500, 'best-selling': 23501, 'chronicled': 23502, 'skated': 23503, 'moallem': 23504, 'appalled': 23505, 'barbarism': 23506, 'composition': 23507, 'docks': 23508, 'banbury': 23509, 'seventy-five': 23510, 'lenghty': 23511, 'al-hayam': 23512, 'zahri': 23513, 'borroto': 23514, 'contemplating': 23515, 'glittering': 23516, 'gemelli': 23517, '84-year': 23518, 'inflammation': 23519, 'minya': 23520, 'unsanitary': 23521, 'simulation': 23522, 'followup': 23523, 'solidifies': 23524, 'oft': 23525, "o'er": 23526, 'descending': 23527, 'mick': 23528, 'jagger': 23529, 'hostess-producer': 23530, 'oprah': 23531, 'winfrey': 23532, 'golfer': 23533, 'consitutional': 23534, 'spearker': 23535, 'reversible': 23536, 'state-designate': 23537, 'minimally': 23538, 'one-and-a-half': 23539, 'embolization': 23540, 'hysterectomy': 23541, 'shrinks': 23542, 'rowed': 23543, 'fibroids': 23544, 'alimport': 23545, 'qilla': 23546, 'arrogance': 23547, 'deep-rooted': 23548, '6,54,000': 23549, 'inga': 23550, 'marchand': 23551, 'run-in': 23552, 'spitting': 23553, 'counseling': 23554, 'manicurists': 23555, 'rapidly-growing': 23556, 'fixed-asset': 23557, 'slightest': 23558, 'plotter': 23559, 'reassessed': 23560, '384': 23561, 'much-maligned': 23562, 'pro-european': 23563, 'supremacy': 23564, 'federally-mandated': 23565, 'bereavement': 23566, 'h-2a': 23567, 'haswa': 23568, 'jib': 23569, 'dial': 23570, 'stefan': 23571, 'mellar': 23572, 'deniers': 23573, 'mini-bus': 23574, 'ball-point': 23575, 'hikmet': 23576, '°c': 23577, 'mashonaland': 23578, 'ambassador-designate': 23579, 'godfrey': 23580, 'dzvairo': 23581, 'itai': 23582, 'marchi': 23583, 'karidza': 23584, 'tendai': 23585, 'matambanadzo': 23586, 'naad': 23587, 'mangwana': 23588, '1,125': 23589, 'krygyz': 23590, '3-point': 23591, 'dubrovka': 23592, '1.35': 23593, '10.2': 23594, 'haden': 23595, 'maclellan': 23596, 'surrey': 23597, 'feniger': 23598, 'xenophobic': 23599, 'combative': 23600, 'lsi': 23601, 'sp': 23602, 'wallachia': 23603, 'moldavia': 23604, 'suzerainty': 23605, '1859': 23606, '1862': 23607, 'transylvania': 23608, 'axis': 23609, 'soviets': 23610, 'abdication': 23611, 'nicolae': 23612, 'securitate': 23613, 'concessional': 23614, 'rescheduling': 23615, '3-year': 23616, '60-40': 23617, '1876': 23618, 'tsarist': 23619, 'one-sixth': 23620, 'akaev': 23621, 'otunbaeva': 23622, 'interethnic': 23623, 'lark': 23624, 'uninterred': 23625, 'crest': 23626, 'grave-hillock': 23627, 'monstrous': 23628, 'refrained': 23629, 'non-lethal': 23630, 'posthumous': 23631, 'deft': 23632, 'degenerative': 23633, 'bowie': 23634, 'merle': 23635, 'haggard': 23636, 'jessye': 23637, 'chinooks': 23638, 'twin-rotor': 23639, 'wadajir': 23640, 'hodan': 23641, 'reconciling': 23642, 'seacoast': 23643, 'ezzedin': 23644, 'smoldering': 23645, 'parched': 23646, 'smog': 23647, 'sonthi': 23648, 'boonyaratglin': 23649, 'shaima': 23650, 'privately-run': 23651, 'char': 23652, 'bowing': 23653, 'al-shaer': 23654, 'bombard': 23655, 'booked': 23656, 'fingerprinted': 23657, 'shrunk': 23658, 'layoff': 23659, 'citi': 23660, 'walks': 23661, 'maoist-called': 23662, '79.43': 23663, 'fifteen-year': 23664, 'waning': 23665, 'naha': 23666, 'mint': 23667, 'mouknass': 23668, 'conspirators': 23669, 'brig.': 23670, 'sexy': 23671, 'pertains': 23672, 'nasty': 23673, 'fsg': 23674, 'spessart': 23675, 'rheinland-pfalz': 23676, 'one-sentence': 23677, 'ui-chun': 23678, 'strutting': 23679, 'catwalk': 23680, 'mellon': 23681, 'siddiqi': 23682, 'quake-ravaged': 23683, 'beechcraft': 23684, 'zhulyany': 23685, 'question-and-answer': 23686, 'intoning': 23687, 'mica': 23688, 'adde': 23689, 'slovakian': 23690, 'zsolt': 23691, 'grebe': 23692, 'peregrine': 23693, 'gabcikovo': 23694, 'upsets': 23695, 'kohlschreiber': 23696, 'densely-populated': 23697, 'transformer': 23698, 'kayettuli': 23699, 'tightly-packed': 23700, 'gloomier': 23701, 'haris': 23702, 'ehsanul': 23703, 'sadequee': 23704, 'positioning': 23705, 'recently-completed': 23706, 'think-tank': 23707, 'bheri': 23708, 'surkhet': 23709, '325-member': 23710, 'takirambudde': 23711, 'cross-ethnic': 23712, 'water-dropping': 23713, 'firefighting': 23714, 'carpentry': 23715, 'kuomintang': 23716, 'hsiao': 23717, 'bi-khim': 23718, 'bombmaking': 23719, 'perwiz': 23720, 'reexamine': 23721, 'stipulated': 23722, 'baribari': 23723, 'alhurra': 23724, 'repel': 23725, 'grammy-winning': 23726, 'percussionist': 23727, 'conga': 23728, 'dizzy': 23729, 'gillespie': 23730, 'puente': 23731, 'salsa': 23732, 'celia': 23733, 'forty-nine': 23734, 'hopeless': 23735, 'refloating': 23736, 'refloat': 23737, 'four-day-long': 23738, 'settings': 23739, 'mythical': 23740, 'winding': 23741, 'borg': 23742, 'risky': 23743, 'consul-general': 23744, 'farris': 23745, 'redrawing': 23746, 'separates': 23747, 'indian-': 23748, 'pierced': 23749, 'attributes': 23750, 'as-sahab': 23751, '25-million': 23752, 'europrean': 23753, 'renate': 23754, 'kuenast': 23755, 'co-leader': 23756, 'kuhn': 23757, 'batasuna': 23758, 'qinglin': 23759, 'cnn-turk': 23760, 'mersin': 23761, 'counterkidnapping': 23762, 'anti-gang': 23763, 'thatcher': 23764, 'pathe': 23765, 'productions': 23766, 'dj': 23767, 'frears': 23768, 'oscar-winning': 23769, 'mirren': 23770, 're-awakened': 23771, 'acquisition-minded': 23772, 'seattle-based': 23773, '57.5': 23774, '62.1': 23775, 'outbid': 23776, 'ratner': 23777, '1.26': 23778, '1.64': 23779, 'sweetened': 23780, 'acceptances': 23781, '87-store': 23782, 'near-subsistence': 23783, 'upper-middle': 23784, '07-aug': 23785, 'tourism-related': 23786, 'rupee': 23787, 'overvalued': 23788, 'amortizing': 23789, 'eff': 23790, 'erm2': 23791, 'bounce': 23792, 'supplanted': 23793, 'overstaffing': 23794, 'mortgaged': 23795, 'oil-backed': 23796, 'near-term': 23797, 'specializes': 23798, 'knowledge-based': 23799, 'conformity': 23800, '245': 23801, 'qamdo': 23802, 'zero-interest': 23803, 'bern': 23804, 'prized': 23805, 'double-taxation': 23806, 'behest': 23807, 'account-holders': 23808, 'mururoa': 23809, 'microenterprises': 23810, 'buffetings': 23811, 'reproaches': 23812, 'calmness': 23813, 'fury': 23814, '94.13': 23815, 'issawiya': 23816, 'criminalizing': 23817, 'tainting': 23818, 'erich': 23819, 'circumventing': 23820, 'rumbek': 23821, 'fidler': 23822, 'cutler': 23823, 'tynychbek': 23824, 'rice-growing': 23825, 'transferable': 23826, 'tula': 23827, 'escravos': 23828, 'entitlements': 23829, 'chookiat': 23830, 'ophaswongse': 23831, 'sixth-seed': 23832, 'grosjean': 23833, 'danai': 23834, 'udomchoke': 23835, '43rd-ranked': 23836, 'oudsema': 23837, 'rice-producing': 23838, 'scan': 23839, 'proclaim': 23840, 'accordingly': 23841, 'purification': 23842, 'sunrise-to-sunset': 23843, 'thirteenth': 23844, 'jailings': 23845, 'batted': 23846, 'shaharyar': 23847, 'c.w.': 23848, 'moveon.org': 23849, 'inzamam-ul-haq': 23850, 'mina': 23851, 'jamarat': 23852, 'grueling': 23853, 'able-bodied': 23854, 'communion': 23855, 'wasit': 23856, 'mwesige': 23857, 'shaka': 23858, 'ssali': 23859, 'sutalinov': 23860, 'objectionable': 23861, 'cancelation': 23862, 'herman': 23863, 'rompuy': 23864, 'edinburgh': 23865, 'g-eight': 23866, 'anti-military': 23867, 'anti-nuclear': 23868, 'pro-reunification': 23869, 'chhattisgarh': 23870, 'raipur': 23871, 'jharkhand': 23872, 'fratto': 23873, 'sell-off': 23874, 'indices': 23875, 'half-a-percent': 23876, 'dehui': 23877, 'tulsi': 23878, 'giri': 23879, 'mostafavi': 23880, 'jude': 23881, 'genes': 23882, 'proteins': 23883, 'josefa': 23884, 'baeza': 23885, 'mexico-u.s.': 23886, 'aston': 23887, 'mulally': 23888, 'cost-reduction': 23889, 'khamis': 23890, 'tawhid': 23891, 'wal': 23892, 'dahab': 23893, '352-3': 23894, 'double-century': 23895, 'sachin': 23896, 'kartik': 23897, 'second-wicket': 23898, '175-run': 23899, 'tallied': 23900, '117-ball': 23901, 'j.s.': 23902, 'brar': 23903, 'shamsbari': 23904, 'slowly-implemented': 23905, 'bush-clinton': 23906, 're-locate': 23907, 'wreaked': 23908, 'blockage': 23909, 'anti-monopoly': 23910, 'sidakan': 23911, 'seven-point': 23912, 'rex': 23913, 'walheim': 23914, 'orbital': 23915, 'outing': 23916, 'six-and-a-half': 23917, 'crewmates': 23918, 'ex-generals': 23919, 'leopold': 23920, 'eyharts': 23921, 'plundered': 23922, 'handsets': 23923, 'umpires': 23924, 'hashimzai': 23925, 'obeyed': 23926, 'twenty-five': 23927, 'nsc': 23928, 'selecting': 23929, 'land-use': 23930, 'explicit': 23931, 'central-leftist': 23932, 'theatergoers': 23933, 'suspense': 23934, 'disturbiaunseated': 23935, 'peeping': 23936, 'labeouf': 23937, 'morse': 23938, 'dreamworks': 23939, 'distributor': 23940, 'lebeouf': 23941, 'transformers': 23942, 'hualan': 23943, 'satan': 23944, 'interwoven': 23945, 'salif': 23946, 'ousamane': 23947, 'ngom': 23948, 'taraba': 23949, 'suspends': 23950, '10-week': 23951, 'encampment': 23952, 'afflicting': 23953, 'imperative': 23954, 'oblivious': 23955, 'destitute': 23956, 'three-meter': 23957, 'huts': 23958, '352': 23959, 'post-monsoon': 23960, '1,35,000': 23961, 'supplemental': 23962, 'recounted': 23963, 'rhythm-and-blues': 23964, 'blueberry': 23965, 'erasmus': 23966, 'kremlin-controlled': 23967, 'long-suffering': 23968, 'avenging': 23969, '246': 23970, 'oleksiy': 23971, 'ivchenko': 23972, 'ukrainy': 23973, 'predrag': 23974, "shi'te": 23975, 'dulles': 23976, 'omission': 23977, 'muhajer': 23978, 'rostock': 23979, 'heiligendamm': 23980, 'rostok': 23981, 'pig-borne': 23982, 'gargash': 23983, 'sift': 23984, 'displaying': 23985, 'passages': 23986, 'bara': 23987, '89.9': 23988, '12.09': 23989, 'nuys': 23990, 'calif.': 23991, '1,32,000': 23992, 'realestate': 23993, 'fluctuation': 23994, '361.8': 23995, 'communist-style': 23996, '38th': 23997, 'young-sam': 23998, 'punctuated': 23999, 'bartolomeo': 24000, '1784': 24001, 'gustavia': 24002, 'repurchased': 24003, 'appellations': 24004, 'three-crown': 24005, 'populace': 24006, 'collectivity': 24007, 'hereditary': 24008, 'premiers': 24009, 'ten-year': 24010, 'promulgation': 24011, 'plurality': 24012, 'overruled': 24013, 'nepal-united': 24014, 'gridlock': 24015, 'jhala': 24016, 'khanal': 24017, 'indentured': 24018, 'ethnocultural': 24019, 'socialist-oriented': 24020, 'cheddi': 24021, 'bharrat': 24022, 'jagdeo': 24023, 'rhodesia': 24024, '1923': 24025, 'spratly': 24026, 'overlaps': 24027, 'reef': 24028, 'dry-goods': 24029, 'vucelic': 24030, 'mira': 24031, 'markovic': 24032, 'remembrances': 24033, 'germany-based': 24034, 'trauma': 24035, 'blockbuster': 24036, 'thermopylae': 24037, '480': 24038, 'illegally-copied': 24039, 'amnesties': 24040, 'traffic-law': 24041, 'absolve': 24042, 'watchful': 24043, 'ziyang': 24044, 'magnolia': 24045, 'counter-attacks': 24046, 'one-man': 24047, 'fatmir': 24048, 'sejdiu': 24049, 'nations-mediated': 24050, 'waseem': 24051, 'sohrab': 24052, 'goth': 24053, 'cartographers': 24054, 'haziri': 24055, 'ethic': 24056, 'puntland': 24057, 'inter-parliamentary': 24058, 'elyor': 24059, 'ganiyev': 24060, 'alvarezes': 24061, 'party-led': 24062, '261': 24063, '543-member': 24064, 'kwanzaa': 24065, 'wintery': 24066, 'over-crowded': 24067, 'qader': 24068, 'jassim': 24069, 'sherwan': 24070, 'gaulle': 24071, 'qarabaugh': 24072, 'klerk': 24073, 'al-jafaari': 24074, 'commend': 24075, 'torshin': 24076, 'prosecutor-general': 24077, 'shepel': 24078, 'highly-enriched': 24079, 'disapproves': 24080, 'left-': 24081, 'gimble': 24082, 'papa': 24083, 'u.s.-german': 24084, 'conflict-related': 24085, 'co-finance': 24086, 'mariann': 24087, 'boel': 24088, 'zhenchuan': 24089, 'breakers': 24090, 'hurling': 24091, "n'guesso": 24092, 'permissible': 24093, 'gvero': 24094, 'trans-afghan': 24095, '2,400-kilometer': 24096, 're-enter': 24097, 'us-visit': 24098, 'mocny': 24099, 'green-card': 24100, 'accosted': 24101, 'anne-marie': 24102, 'idrac': 24103, 'sustains': 24104, 'landfills': 24105, 'afeworki': 24106, 'rethink': 24107, 'car-free': 24108, 'organic': 24109, 'cookies': 24110, 'nasariyah': 24111, 'lattes': 24112, 'ex-beatle': 24113, 'still-untitled': 24114, 'feature-length': 24115, 'cobos': 24116, '36-36': 24117, 'schudrich': 24118, 'scrolls': 24119, 'rabinnical': 24120, 'merbau': 24121, 'undisturbed': 24122, 'anti-smuggling': 24123, 'iranian-sponsored': 24124, 'admhaiyah': 24125, 'al-momen': 24126, 'sunni-ruled': 24127, 'dawei-ye': 24128, 'heliodoro': 24129, 'reclassify': 24130, 'bloating': 24131, 'dot': 24132, 'riverbanks': 24133, 'profiteers': 24134, 'essentials': 24135, 'lecturers': 24136, 'lashkar-e-islam': 24137, 'knu': 24138, 'mostly-sunni': 24139, 'ghazaliyah': 24140, 'policing': 24141, 'hovers': 24142, 'chequers': 24143, 'kalamazoo': 24144, 'mich.-based': 24145, 'maumoon': 24146, 'embark': 24147, 'majlis': 24148, 'sea-level': 24149, 'km-wide': 24150, 'badme': 24151, 'eebc': 24152, 'anti-ethiopian': 24153, 'immersed': 24154, 'advent': 24155, 'soviet-led': 24156, 'expended': 24157, 'heel': 24158, 'mitigating': 24159, 'radicova': 24160, 'agonies': 24161, 'fiercer': 24162, 'unto': 24163, 'gasping': 24164, 'grudges': 24165, 'tusks': 24166, 'growled': 24167, 'cowards': 24168, 'majesty': 24169, 'marry': 24170, 'proliferating': 24171, 'ichiro': 24172, 'sino-japanese': 24173, 'audiotapes': 24174, 'doug': 24175, 'pathologically': 24176, 'liar': 24177, '179-run': 24178, '343': 24179, 'csongrad': 24180, 'apr-46': 24181, '196': 24182, '170-meter': 24183, 'abnormally': 24184, 'ratnayke': 24185, 'bruised': 24186, '28-member': 24187, 'cosmopolitan': 24188, 'multiculturalism': 24189, 'multi-sport': 24190, 'etihad': 24191, 'bastille': 24192, 'inalienable': 24193, 'instructor': 24194, 'basij': 24195, 'vigilantes': 24196, 'azov': 24197, 'volganeft-139': 24198, 'sulfur': 24199, 'cary': 24200, 'porpoises': 24201, 'u.s.-coalition': 24202, 'afghan-coalition': 24203, 'tuneup': 24204, 'fallouj': 24205, 'petrodar': 24206, 'conglomerate': 24207, 'arbab': 24208, 'basir': 24209, 'democracy-building': 24210, 'rajah': 24211, 'zamboanga': 24212, 'largest-ever': 24213, '80-kilometer': 24214, 'inlets': 24215, 'fumes': 24216, 'petrechemical': 24217, 'interpreting': 24218, 'taitung': 24219, 'plowing': 24220, 'guardedly': 24221, 'grips': 24222, 'regrettably': 24223, 'channeled': 24224, 'culls': 24225, '555': 24226, 'wisely': 24227, 'krugman': 24228, 'half-measures': 24229, 'trichet': 24230, 'boao': 24231, 'mowaffaq': 24232, 'rubaie': 24233, 'formulas': 24234, 'phillips': 24235, 'battleground': 24236, 'sinaloa': 24237, '27-28': 24238, 'emitted': 24239, 'nosedive': 24240, 'holiday-shortened': 24241, 'venezuela-based': 24242, 'creators': 24243, '890': 24244, 'u.s.-imposed': 24245, 'popov': 24246, 'poncet': 24247, 'barcode': 24248, 'clerks': 24249, 'barcoded': 24250, 'chewing': 24251, 'invalidated': 24252, 'redoing': 24253, 'nails': 24254, '70-day': 24255, 'jam': 24256, 'refiling': 24257, 'sacrilegious': 24258, 'majority-muslim': 24259, 'valdivia': 24260, 'ultrasound': 24261, '17.09': 24262, 'unicameral': 24263, 'verkhovna': 24264, 'rada': 24265, 'immobility': 24266, 'rocket-powered': 24267, 'nycz': 24268, 'dreaded': 24269, 'uthai': 24270, 'burying': 24271, 'afghan-pakistan': 24272, 'petrovka': 24273, 'bakyt': 24274, 'seitov': 24275, 'ramechhap': 24276, 'trailed': 24277, 'schiavone': 24278, 'kasparov': 24279, 'kasyanov': 24280, 'magnet': 24281, 'disparate': 24282, 'transcends': 24283, 'superseded': 24284, 'high-rise': 24285, 'self-named': 24286, 'blond-haired': 24287, 'telephones': 24288, '04-feb': 24289, 'jokhang': 24290, 'gongga': 24291, 'rabdhure': 24292, 'inflate': 24293, 'finalists': 24294, 'fourth-place': 24295, 'high-flying': 24296, 'stockholders': 24297, 'ranger': 24298, 'inc': 24299, '2,25,000': 24300, 'avon': 24301, 'glitches': 24302, 'patch': 24303, 'craze': 24304, 'bankruptcy-law': 24305, 'moheli': 24306, 'fomboni': 24307, 'rotates': 24308, 'sambi': 24309, 'bacar': 24310, 'anjouanais': 24311, 'comoran': 24312, 'curia': 24313, 'mementos': 24314, 'non-budgetary': 24315, 'countercoups': 24316, 'widest': 24317, 'empower': 24318, 'non-indigenous': 24319, 'lowlands': 24320, 'entrant': 24321, 'wavered': 24322, 'pro-market': 24323, 'tallinn': 24324, 'forefront': 24325, '%-plus': 24326, 'outperformed': 24327, 'anti-nato': 24328, 'notch': 24329, 'boediono': 24330, 'continuity': 24331, 'near-poor': 24332, 'peatlands': 24333, 'trailblazing': 24334, 'redd+': 24335, 'suppose': 24336, 'err': 24337, 'testament': 24338, 'anti-alliance': 24339, 'interred': 24340, 'no-fly': 24341, 'el-kheir': 24342, 'terrorizing': 24343, 'morally': 24344, 'land-to-ship': 24345, 'il.': 24346, 'commute': 24347, 'basing': 24348, 'grigory': 24349, 'lighthouses': 24350, 'bachelor': 24351, 'rewarded': 24352, '93rd': 24353, 'redirected': 24354, 'non-controversial': 24355, 'redirecting': 24356, 'hack': 24357, 'drummond': 24358, 'firecracker': 24359, 'pallipat': 24360, 'sweets': 24361, 'snacks': 24362, 'lamps': 24363, '518': 24364, '346': 24365, 'unverified': 24366, 'aids-related': 24367, 'h.i.v./aids': 24368, 'rizkar': 24369, 'pawn': 24370, 'steadfast': 24371, 'h.i.v': 24372, 'state-radio': 24373, 'covenant': 24374, 'deposited': 24375, 'byelection': 24376, 'disbarred': 24377, 'rollout': 24378, 'hennes': 24379, 'mauritz': 24380, 'ab': 24381, 'h&m': 24382, 'purses': 24383, 'risque': 24384, 'conical': 24385, 'bras': 24386, 'stockholm-based': 24387, 'fellowships': 24388, 'orphanages': 24389, 'recalls': 24390, 'reliability': 24391, 'honda': 24392, 'non-farm': 24393, 'tax-relief': 24394, 'bomb-sniffing': 24395, 'near-continuous': 24396, 'sherry': 24397, 'fining': 24398, 'classification': 24399, 'u.s.-born': 24400, 'washington-area': 24401, 'goddess': 24402, 'beijings': 24403, 'rain-swollen': 24404, 'hacksaws': 24405, 'transports': 24406, 'muse': 24407, '4,600': 24408, '7,620': 24409, '777': 24410, 'kochi': 24411, 'baziv': 24412, 'face-off': 24413, 'slauta': 24414, 'kharkiv': 24415, 'yevhen': 24416, 'kushnaryov': 24417, 'rousing': 24418, 'ryzhkov': 24419, 'far-fetched': 24420, 'yevgeny': 24421, 'primakov': 24422, 'phan': 24423, 'khai': 24424, 'tran': 24425, '2,18,000': 24426, '1,72,000': 24427, 'small-': 24428, 'giustra': 24429, 'small-and': 24430, 'self-sustainable': 24431, '2,17,000': 24432, 'osterholm': 24433, 'worst-case': 24434, 'katsuya': 24435, 'okada': 24436, 'naoto': 24437, 'kan': 24438, 'laurence': 24439, 'boring': 24440, 'literally': 24441, 'feather': 24442, 'boas': 24443, 'connick': 24444, 'jr': 24445, 'chao': 24446, 'congratuatlions': 24447, 'posht-e-rud': 24448, 'artemisinin': 24449, 'bala': 24450, 'boluk': 24451, '9,50,000': 24452, 'midestern': 24453, 'newport': 24454, 'hydroxide': 24455, 'reverses': 24456, 'definitely': 24457, 'falciparum': 24458, 'plastered': 24459, "ba'athists": 24460, 're-ignite': 24461, 'mosquito-born': 24462, 'sui': 24463, 'nawab': 24464, 'al-maliky': 24465, 'jaghato': 24466, 'nahim': 24467, 'malawian': 24468, 'willy': 24469, 'mwaluka': 24470, 'artisans': 24471, 'hotspots': 24472, 'adijon': 24473, 'warrior': 24474, 'tendering': 24475, '99.3': 24476, '3.55': 24477, 'taaf': 24478, 'archipelagos': 24479, 'crozet': 24480, 'saint-paul': 24481, 'fauna': 24482, 'adelie': 24483, 'slice': 24484, "mutharika's": 24485, 'exhibited': 24486, 'goodall': 24487, 'gondwe': 24488, 'oresund': 24489, 'denmark-sweden': 24490, 'bosporus': 24491, 'morocco-spain': 24492, 'seaway': 24493, 'canada-us': 24494, 'jomo': 24495, 'toroitich': 24496, 'multiethnic': 24497, 'rainbow': 24498, 'uhuru': 24499, 'conciliatory': 24500, 'israeli-occupied': 24501, 'odm': 24502, 'powersharing': 24503, 'eliminates': 24504, 'willis': 24505, 'islets': 24506, 'beacons': 24507, 'gladly': 24508, 'traitor': 24509, 'nay': 24510, 'wallow': 24511, 'converged': 24512, 'owning': 24513, 'jeungsan': 24514, 'dhi': 24515, 'nautical': 24516, 'listen': 24517, '7115': 24518, '9885': 24519, '11705': 24520, '11725': 24521, 'voanews.com': 24522, '102.4': 24523, '104.6': 24524, 'depressing': 24525, 'argentinean': 24526, 'angie': 24527, 'sanclemente': 24528, 'valenica': 24529, 'airat': 24530, '14-year': 24531, '2.43': 24532, 'hae-sung': 24533, 'panmunjom': 24534, 'meadowbrook': 24535, 'ozlam': 24536, 'sanabel': 24537, 'obsessed': 24538, 'alexeyenko': 24539, 'ordnance': 24540, 'swiergosz': 24541, 'vaguely': 24542, 'fortinet': 24543, 'irawaddy': 24544, 'valery': 24545, 'sitnikov': 24546, 'bio-terrorism': 24547, 'enacts': 24548, 'fitzpatrick': 24549, 'habbaniya': 24550, 'two-degrees': 24551, 'half-a-degree': 24552, 'mandates': 24553, 'airlifts': 24554, 'benghazi': 24555, 'two-point-seven': 24556, 'heightening': 24557, 'thae': 24558, 'bok': 24559, 'geo-services': 24560, 'chinese-language': 24561, 'mandarin': 24562, 'tuo': 24563, 'bpa': 24564, 'ilhami': 24565, 'gafir': 24566, 'abdelkader': 24567, 'loyalties': 24568, 'al-nimnim': 24569, 'palestinian-ruled': 24570, 'grieves': 24571, 'talangama': 24572, 'pro-tamil': 24573, 'eight-nation': 24574, 'shuai': 24575, 'stosur': 24576, 'arthurs': 24577, '1850s': 24578, 'disrespectful': 24579, 'dissipated': 24580, 'tookie': 24581, 'inspirational': 24582, 'playmaker': 24583, 'export-import': 24584, 'wrecking': 24585, 'trade-in': 24586, 'fulayfill': 24587, 'geyer': 24588, 'cavalry': 24589, 'near-empty': 24590, 'jeb': 24591, 'rené': 24592, 'sanderson': 24593, 'rony': 24594, 'trooper': 24595, 'sese': 24596, 'seko': 24597, 'shajoy': 24598, 'insurgent-hit': 24599, 'jesper': 24600, 'helsoe': 24601, 'feyzabad': 24602, 'yazdi': 24603, 'mahendranagar': 24604, 'bager': 24605, 'uspi': 24606, 'guardians': 24607, 'make-up': 24608, 'one-meter-tall': 24609, 'lntelligence': 24610, 'altcantara': 24611, 'vsv-30': 24612, 'voto': 24613, 'bernales': 24614, 'under-funded': 24615, 'likening': 24616, 'gear': 24617, 'practitioners': 24618, 'tarnish': 24619, 'detractors': 24620, 'ill-wishers': 24621, 'paralympics': 24622, '16-stop': 24623, 'statisticians': 24624, 'gymnasiums': 24625, 'chrome': 24626, 'oversized': 24627, 'velocity': 24628, '575': 24629, 'v-150': 24630, 'streaked': 24631, '515': 24632, '581': 24633, 'levitates': 24634, 'alstom': 24635, 'horsepower': 24636, '786': 24637, 'burned-out': 24638, 'alleviation': 24639, 'citibank': 24640, 'bank-funded': 24641, '295': 24642, 'revelations': 24643, 'war-damaged': 24644, 'weight-loss': 24645, 'mayo': 24646, 'environments': 24647, '\x91': 24648, '\x92': 24649, 'u.n.-supervised': 24650, 'decathlon': 24651, 'limping': 24652, 'mccloud': 24653, 'al-ani': 24654, 'lectures': 24655, 'trevor': 24656, 'jianchao': 24657, 'lounderma': 24658, 'unnerve': 24659, 'hanun': 24660, 'spoiled': 24661, 'explanations': 24662, 'cristobal': 24663, 'casas': 24664, 'uribana': 24665, 'mask-wearing': 24666, 'panzhihua': 24667, 'tit-for-tat': 24668, 'gaspard': 24669, 'bulldozing': 24670, 'nyange': 24671, 'schooled': 24672, '492': 24673, '4.55': 24674, '12.97': 24675, 'cutthroat': 24676, '1767': 24677, '1790': 24678, 'outmigration': 24679, '233': 24680, 'cemented': 24681, 'melanesian': 24682, 'melanesians': 24683, 'indo-fijian': 24684, 'civilian-led': 24685, 'laisenia': 24686, 'commodore': 24687, 'voreqe': 24688, 'islamic-oriented': 24689, 'famine-related': 24690, 'influxes': 24691, 'sprang': 24692, 'snatch': 24693, 'brawny': 24694, 'refrigerator': 24695, 'doorbell': 24696, 'pin-kung': 24697, 'el-sibaie': 24698, 'moustafa': 24699, 'over-interpreted': 24700, 'yafei': 24701, 'google-based': 24702, 'condone': 24703, 'amil': 24704, 'al-adamiya': 24705, 'villa': 24706, 'andré': 24707, 'nesnera': 24708, 'korea\x92s': 24709, 'pectoral': 24710, 'booed': 24711, 'domachowska': 24712, 'two-meter': 24713, 'berms': 24714, 'surfboard': 24715, 'emphasize': 24716, 'swells': 24717, 'posses': 24718, 'gholamhossein': 24719, 'convenient': 24720, 'check-up': 24721, 'theoretical': 24722, '6000': 24723, 'daimler': 24724, 'desi': 24725, 'varanasi': 24726, 'meerut': 24727, 'endeavourhas': 24728, 'undocked': 24729, 'exterior': 24730, 'three-person': 24731, 'goodbyes': 24732, 'hatches': 24733, 'recycling': 24734, 'dollar-denominated': 24735, 'expectant': 24736, 'umbilical': 24737, 'suna': 24738, 'afewerki': 24739, 'power-': 24740, '7,000-strong': 24741, 'ontario': 24742, 'rebel-linked': 24743, 'worn-out': 24744, 'coastguard': 24745, 'spyros': 24746, 'kristina': 24747, 'marit': 24748, 'bjoergen': 24749, 'hilde': 24750, 'pedersen': 24751, 'state-approved': 24752, 'state-monitored': 24753, 'genders': 24754, 'speedskating': 24755, 'ebrahim': 24756, 'third-country': 24757, 'multi-target': 24758, 'supersedes': 24759, 'defer': 24760, 'textbook': 24761, 'overlooks': 24762, 'asia-africa': 24763, 'calculation': 24764, 'post-castro': 24765, 'characterization': 24766, 'cynical': 24767, '880': 24768, 'azzedine': 24769, 'belkadi': 24770, 'pakistani-owned': 24771, 'kerimli': 24772, 'pflp': 24773, 'saadat': 24774, 'rehavam': 24775, 'zeevi': 24776, 'splattered': 24777, 'pettigrew': 24778, 'bowling': 24779, 'rolando': 24780, 'otoniel': 24781, 'guevara': 24782, 'samahdna': 24783, 'runup': 24784, 'libertador': 24785, "o'higgins": 24786, 'valparaiso': 24787, '8.8-magnitude': 24788, 'intelcenter': 24789, 'heeding': 24790, 'newly-energized': 24791, 'then-justice': 24792, 'kiraitu': 24793, 'anglo-leasing': 24794, '1,600-kilometer': 24795, 'million-dollar': 24796, 'payday': 24797, 'divine': 24798, 'summed': 24799, 'barbecues': 24800, 'bulandshahr': 24801, 'el-leil': 24802, '454-member': 24803, 'atheists': 24804, 'lambeth': 24805, 'misappropriation': 24806, 'patient-specific': 24807, 'veterinarian': 24808, 'veils': 24809, 'khori': 24810, '4,480': 24811, 'lauberhorn': 24812, '30.54': 24813, 'alleys': 24814, 'bib': 24815, '706': 24816, '589': 24817, 'even-odd': 24818, 'rittipakdee': 24819, 'unvetted': 24820, 'gorges': 24821, 'yuanmu': 24822, 'overpopulation': 24823, 'sixty-six': 24824, 'delegitimize': 24825, '45,000-person': 24826, 'amrullah': 24827, 'militaristic': 24828, 'motoyasu': 24829, 'candlelit': 24830, 'lantern': 24831, 'itahari': 24832, 'astonished': 24833, 'hongyao': 24834, 'arafura': 24835, 'merauke': 24836, 'headcount-control': 24837, 'gelles': 24838, 'wertheim': 24839, 'schroder': 24840, 'co': 24841, 'world-wide': 24842, '87.5': 24843, '38.875': 24844, 'shareholder-rights': 24845, 'suitors': 24846, 'cost-control': 24847, 'igad': 24848, 'staff-reduction': 24849, 'trimmed': 24850, 'kubilius': 24851, 'cutback': 24852, '17.9': 24853, 'eskom': 24854, 'necessitating': 24855, 'load-shedding': 24856, 'empowerment': 24857, 'guano': 24858, 'navassa': 24859, 'rested': 24860, 'panamanians': 24861, '1811': 24862, 'lowland': 24863, '35-year': 24864, 'stroessner': 24865, 'omra': 24866, 'norsemen': 24867, 'boru': 24868, '1014': 24869, 'anglo-irish': 24870, 'repressions': 24871, 'andrews': 24872, 'trotted': 24873, 'scudded': 24874, 'optimist': 24875, 'pessimist': 24876, '13.857': 24877, '22.772': 24878, '107.41': 24879, '107.25': 24880, 'loder': 24881, 'crusader': 24882, 'misdemeanor': 24883, 'sobriety': 24884, '1.10': 24885, 'u-turn': 24886, 'misdmeanor': 24887, '4.00': 24888, 'keifer': 24889, 'hui': 24890, 'metropolis': 24891, 'baishiyao': 24892, 'feminism': 24893, 'psychology': 24894, 'housewife': 24895, 'unfulfilled': 24896, 'best-seller': 24897, 'abdouramane': 24898, 'conviasa': 24899, 'thirty-six': 24900, 'sidor': 24901, 'margarita': 24902, 'u.s.-organized': 24903, 'defections': 24904, 'portraits': 24905, 'yvon': 24906, 'neptune': 24907, 'jocelerme': 24908, 'privert': 24909, 'protestors': 24910, 'naryn': 24911, 'dutch-shell': 24912, 'pour': 24913, 'onshore': 24914, 'changbei': 24915, 'petrochina': 24916, 'unreleased': 24917, 'pneumonic': 24918, '2500': 24919, 'foot-and-mouth': 24920, 'tawilla': 24921, 'widens': 24922, 'first-trimester': 24923, 'incest': 24924, 'addington': 24925, 'hannah': 24926, 'canaries': 24927, 'utor': 24928, 'whipping': 24929, 'huddled': 24930, 'sludge': 24931, 'walayo': 24932, 'hepatitis': 24933, 'cuban-engineered': 24934, 'iranian-cuban': 24935, 'cerda': 24936, 'disappearances': 24937, 'belkhadem': 24938, 'ugalde': 24939, 'magistrates': 24940, '24-year': 24941, 'milky': 24942, 'magellanic': 24943, 'fragmentary': 24944, 'parnaz': 24945, 'haleh': 24946, 'esfandiari': 24947, 'woodrow': 24948, 'kian': 24949, 'tajbakhsh': 24950, 'conflict-resolution': 24951, 'malibu': 24952, 'concentrating': 24953, 'ohlmert': 24954, 'qasr': 24955, 'uncomplicated': 24956, 'patu': 24957, 'cull': 24958, 'extolling': 24959, 'eighty-eight': 24960, 'ad-diyar': 24961, 'quake-zone': 24962, 'nourished': 24963, 'wine-making': 24964, 'sonoma': 24965, 'receded': 24966, 'rained': 24967, 'shieh': 24968, 'jhy-wey': 24969, 'tu': 24970, 'cheng-sheng': 24971, 'shu-bian': 24972, 'ezzedine': 24973, 'al-qassem': 24974, 'contemplate': 24975, '11-year': 24976, '8.7': 24977, 'prithvi-ii': 24978, '500-kilogram': 24979, 'aggravate': 24980, 'idriz': 24981, 'balaj': 24982, 'lahi': 24983, 'brahimaj': 24984, 'pune': 24985, '1,34,503': 24986, '816': 24987, 'extra-judicial': 24988, 'mixed-race': 24989, 'misquoted': 24990, 'magnitudes': 24991, '71-person': 24992, '60.3': 24993, 'israeli-': 24994, 'tidjane': 24995, 'thiam': 24996, 'exiting': 24997, 'at-risk': 24998, 'recife': 24999, 'gabgbo': 25000, 'sculpture': 25001, 'gadgets': 25002, 'matteo': 25003, 'duran': 25004, '8,50,000': 25005, 'ferrero-walder': 25006, 'brendan': 25007, '2.08.50': 25008, 'kosuke': 25009, 'kitajima': 25010, '2.07.51': 25011, 'omaha': 25012, '2.11.37': 25013, '400-meter': 25014, 'medley': 25015, 'katie': 25016, 'hoff': 25017, 'freestyle': 25018, 'regents': 25019, '31,500': 25020, 'square-meter': 25021, 'verez': 25022, 'bencomo': 25023, 'innovations': 25024, 'themba-nyathi': 25025, 'demirbag': 25026, 'forgave': 25027, 'bette': 25028, 'caesar': 25029, 'hotel-casino': 25030, 'bawdy': 25031, 'five-night-a-week': 25032, 'french-canadian': 25033, '4,100-seat': 25034, 'show-per-year': 25035, 'meglen': 25036, 'probable': 25037, 'insiders': 25038, 'cher': 25039, '10-years-ago': 25040, 'a.w.b': 25041, 'a.w.b.': 25042, '545': 25043, 'trumpets': 25044, 'reduces': 25045, 'ambient': 25046, 'racers': 25047, 'experimenting': 25048, 'underage': 25049, 'underfed': 25050, 'jockey': 25051, 'johnny': 25052, 'unpremeditated': 25053, 'garbage': 25054, 'akhrorkhodzha': 25055, 'tolipkhodzhayev': 25056, 'militant-appointed': 25057, 'offline': 25058, 'taliban-appointed': 25059, 'statesmen': 25060, 'denial-of-service': 25061, 'tearfully': 25062, '#name?': 25063, 'doaba': 25064, 'hangu': 25065, 'klaus': 25066, 'toepfer': 25067, 'riverside': 25068, 'profiles': 25069, '377': 25070, 'mushtaq': 25071, 'bechuanaland': 25072, 'preserves': 25073, '1895': 25074, 'reverted': 25075, 'democratized': 25076, 'watermelons': 25077, 'quarry': 25078, 'subsuming': 25079, 'legislated': 25080, 'reasoning': 25081, 'overgrazing': 25082, 'linen': 25083, 'travelling': 25084, 'superb': 25085, 'unconcern': 25086, 'characteristic': 25087, 'contemptuously': 25088, 'bark': 25089, 'birch-tree': 25090, 'gump': 25091, 'overheard': 25092, 'pie': 25093, 'disbursed': 25094, 'akitaka': 25095, 'hammering': 25096, 'trans-saharan': 25097, 'creatively': 25098, 'k-4': 25099, 'v-for-victory': 25100, 'tankbuster': 25101, 'kheyal': 25102, 'baaz': 25103, 'gardez': 25104, 'cantarell': 25105, 'administrators': 25106, 'anti-narcotic': 25107, 'anti-poppy': 25108, 'al-ilayan': 25109, 'duda': 25110, 'mendonca': 25111, 'floors': 25112, '12-story': 25113, 'crumbled': 25114, 'inferior': 25115, 'correcting': 25116, 'dengue': 25117, 'larvae': 25118, 'breed': 25119, 'opium-based': 25120, 'jong-nam': 25121, 'jong-chul': 25122, 'jong-woon': 25123, '626': 25124, "n'zerekore": 25125, 'yentai': 25126, 'bhaikaji': 25127, 'ghimire': 25128, 'prostitutes': 25129, 'handicaps': 25130, 'lebanese-syrian': 25131, 'pioneering': 25132, 'lagham': 25133, 'chaudry': 25134, 'sebastiao': 25135, 'veloso': 25136, 'abdali': 25137, 'powder': 25138, 'keg': 25139, 'servicmen': 25140, 'jennings': 25141, '88,000': 25142, '8,100': 25143, 'borrows': 25144, 'cuomo': 25145, 'brokers': 25146, 'wemple': 25147, 'stoffel': 25148, 'clarified': 25149, 'kerem': 25150, 'yaracuy': 25151, 'lapi': 25152, 'carabobo': 25153, 'henrique': 25154, 'salas': 25155, 'jailbreak': 25156, 'banghazi': 25157, 'unbeatable': 25158, 'abdi': 25159, 'dagoberto': 25160, 'swirled': 25161, '8,200': 25162, 'privately-held': 25163, 'narrowing': 25164, 'redistributed': 25165, 'hectare': 25166, 'witch-hunting': 25167, 'discriminate': 25168, 'muthaura': 25169, 'kospi': 25170, 'interest-rate': 25171, 'housing-loan': 25172, 'noureddine': 25173, 'marshmallow': 25174, 'yam': 25175, 'sibghatullah': 25176, 'marshmallows': 25177, 'hilario': 25178, 'davide': 25179, 'alluded': 25180, 'psychiatrist': 25181, 'carreno': 25182, 'backyards': 25183, 'al-talhi': 25184, 'sughayr': 25185, 'al-aziz': 25186, 'al-khashiban': 25187, 'freezes': 25188, 'frantic': 25189, 'chosun': 25190, 'kuduna': 25191, 'abdulhamid': 25192, 'cough': 25193, 'lubroth': 25194, '11.9': 25195, '64.5': 25196, 'collateral': 25197, 'actor-comedian': 25198, 'demise': 25199, 'awe': 25200, 'wolde-michael': 25201, 'meshesha': 25202, 'ozcan': 25203, 'disrupts': 25204, 'aena': 25205, 'belching': 25206, 'on-and-off': 25207, 'abrasive': 25208, 'minar-e-pakistan': 25209, 'lecture': 25210, 'yongbyong': 25211, 'temptations': 25212, 'drop-in': 25213, 'parachute': 25214, 'zydeco': 25215, 'cajun': 25216, 'okie': 25217, 'dokie': 25218, 'stomp': 25219, 'dandy': 25220, 'zaw': 25221, 'recently-disbanded': 25222, 'masud': 25223, 'mir-kazemi': 25224, 'murshidabad': 25225, 'obeid': 25226, 'el-fasher': 25227, 'greatness': 25228, 'bullring': 25229, 'theofilos': 25230, 'malenchenko': 25231, 'kula': 25232, 'under-20': 25233, 'babel': 25234, 'psv': 25235, 'afellay': 25236, 'sota': 25237, 'hirayama': 25238, '68th': 25239, 'omotoyossi': 25240, '32nd': 25241, '59th': 25242, '5th': 25243, 'hilltops': 25244, 'evictions': 25245, 'thuggish': 25246, 'fitch': 25247, 'aa': 25248, 'bbb': 25249, 'shidiac': 25250, 'lbc': 25251, 'pre-judging': 25252, 'configuring': 25253, 'longest-held': 25254, 'brining': 25255, 'eloy': 25256, '820': 25257, 'designate': 25258, 'talbak': 25259, 'nazarov': 25260, 'rakhmonov': 25261, 'purina': 25262, '45.2': 25263, '84.9': 25264, '1.55': 25265, '422.5': 25266, '6.44': 25267, '387.8': 25268, '5.63': 25269, '70.2': 25270, 'phase-out': 25271, 'greenville': 25272, 'n.c.': 25273, 'cincinnati': 25274, 'eveready': 25275, '80.5': 25276, '1494': 25277, '1655': 25278, 'mohmmed': 25279, 'republic-central': 25280, '461': 25281, 'counter-cyclical': 25282, 'distortions': 25283, 'rigidities': 25284, 'private-sector-led': 25285, 'ahmadi-nejad': 25286, 'investment-grade': 25287, 'fended': 25288, 'ace': 25289, 'decaying': 25290, 'neglected': 25291, 'telecoms': 25292, 'kazakhstani': 25293, 'tenge': 25294, 'overreliance': 25295, 'petrochemicals': 25296, 'lucayan': 25297, '1647': 25298, 'inca': 25299, '1533': 25300, '1717': 25301, 'mid-term': 25302, 'flat-nosed': 25303, 'hairless': 25304, 'ill-featured': 25305, 'laugh': 25306, 'dearest': 25307, 'additives': 25308, 'block-by-block': 25309, 'recede': 25310, '828-kilometer': 25311, 'jixi-nehe': 25312, 'floruan': 25313, 'barbey': 25314, 'fataki': 25315, 'pradip': 25316, 'carmo': 25317, 'fernandes': 25318, 'morale': 25319, 'deandre': 25320, 'crank': 25321, 'itunes': 25322, 'jay-z': 25323, 'globalized': 25324, 'pinpoint': 25325, '52.1': 25326, '2.29': 25327, 'abductee': 25328, 'admited': 25329, 'shanghai-bound': 25330, 'contrite': 25331, 'funk': 25332, 'khayrat': 25333, 'el-shater': 25334, 'linguists': 25335, 'theoneste': 25336, 'codefendants': 25337, 'crj-200s': 25338, 'giordani': 25339, 'communist-led': 25340, 'recoverable': 25341, 'azeglio': 25342, 'ciampi': 25343, 'prestige': 25344, '264': 25345, 'inhumanely': 25346, '5,83,000': 25347, 'aaron': 25348, 'galindo': 25349, 'nandrolone': 25350, 'somebody': 25351, 'fiber': 25352, 'rebirth': 25353, 'unwavering': 25354, 'arinze': 25355, 'easterly': 25356, '1800': 25357, 'lawyer-activist': 25358, 'aktham': 25359, 'ennals': 25360, 'somalian': 25361, 'hatching': 25362, 'naise': 25363, 'embodied': 25364, 'bussereau': 25365, 'perfectly': 25366, 'properly-cooked': 25367, 'plummets': 25368, 'j.p.': 25369, 'owner-occupied': 25370, 'finalizes': 25371, 'modification': 25372, 'barney': 25373, '1,275': 25374, 'disseminating': 25375, 'multi-platinum': 25376, 'r&b': 25377, 'stylist': 25378, 'tameka': 25379, 'smalls': 25380, '311': 25381, 'carme': 25382, '01-feb': 25383, 'yoadimadji': 25384, 'wilds': 25385, 'silicon': 25386, 'english-speaking': 25387, 'outsourcing': 25388, 'numbering': 25389, 'capua': 25390, 'shih-chien': 25391, 'wahid': 25392, 'golo': 25393, 'rebel-controlled': 25394, 'moun': 25395, 'afghan-based': 25396, '65.25': 25397, 'omari': 25398, '73,276': 25399, '57,600': 25400, 'sociology': 25401, 'razumkov': 25402, 'moscow-leaning': 25403, 'croatians': 25404, 'zadar': 25405, 'zabihullah': 25406, 'mujahed': 25407, 'canyons': 25408, 'naxalites': 25409, 'burmese-run': 25410, 'exemplifies': 25411, 'welcometousa.gov': 25412, 'interagency': 25413, 'guides': 25414, 'leaner': 25415, 'trimming': 25416, 'underreporting': 25417, 'european-african': 25418, 'moratinos': 25419, 'kiran': 25420, 'inheritance': 25421, 'magnificent': 25422, 'breadth': 25423, 'acuteness': 25424, 'crumbling': 25425, 'granddaughter': 25426, 'anita': 25427, 'genocidal': 25428, 'embera': 25429, 'pro-al-qaida': 25430, 'jihadists': 25431, 'blend': 25432, 'norfolk': 25433, 'receptive': 25434, 'dictators': 25435, 'multiply': 25436, 'auditorium': 25437, 'decontaminating': 25438, 't.': 25439, 'boone': 25440, 'oilman': 25441, 'reno': 25442, 'then-democratic': 25443, 'laser': 25444, 'sadrists': 25445, 'loach': 25446, 'shakes': 25447, 'palme': 25448, "d'or": 25449, 'flanders': 25450, 'inarritu': 25451, 'volver': 25452, 'collectively': 25453, 'long-dominant': 25454, 'narino': 25455, 'cao': 25456, 'gangchuan': 25457, 'butheetaung': 25458, '2,100': 25459, 'hulk': 25460, 'supra': 25461, 'extricated': 25462, 'bayfront': 25463, 'al-rawi': 25464, 'al-jibouri': 25465, 'champ': 25466, 'johann': 25467, 'koss': 25468, 'hidalgo': 25469, 'heriberto': 25470, 'lazcano': 25471, 'retaken': 25472, 'primitive': 25473, 'life-bearing': 25474, 'caroline': 25475, 'tutsi-dominated': 25476, 'bulls': 25477, 'pastured': 25478, 'guileful': 25479, 'feasted': 25480, '41,53,237': 25481, 'bentegeat': 25482, 'dutch-based': 25483, 'refusals': 25484, 'granville': 25485, 'abdelrahman': 25486, 'rahama': 25487, 'confessing': 25488, 'ireju': 25489, 'bares': 25490, 'furious': 25491, '122': 25492, 'age-old': 25493, 'messaging': 25494, '@dalailama': 25495, 'evan': 25496, 'floundering': 25497, 'khartoum-based': 25498, 'kye-gwan': 25499, 'ex-mexican': 25500, 'script': 25501, 'xijing': 25502, "xi'an": 25503, 'eyebrow': 25504, '14-hour': 25505, 'recluse': 25506, 'disfigurement': 25507, 'tirelessly': 25508, 'denuclearize': 25509, 'gianfranco': 25510, 'bezoti': 25511, 'marshburn': 25512, '6-hour': 25513, '53-minute': 25514, '40th': 25515, '5.5-hour': 25516, '10-million': 25517, 'fine-tuning': 25518, 'at-home': 25519, 'macchiavello': 25520, 'overthrowing': 25521, 'narrows': 25522, 'maud': 25523, '64-page': 25524, 'chastelain': 25525, 'unitary': 25526, 'borys': 25527, '247': 25528, 'lutsenko': 25529, 'tsushko': 25530, 'staunchest': 25531, 'despises': 25532, 'tabloids': 25533, 'soap': 25534, 'operas': 25535, 'alien': 25536, '982': 25537, 'straits': 25538, 'vovk': 25539, 'rs-12m': 25540, 'kapustin': 25541, 'yar': 25542, 'strings': 25543, 'lenten': 25544, 'packs': 25545, 'grassland': 25546, 'demostrators': 25547, 'activist-journalist': 25548, 'skate': 25549, 'evgeny': 25550, 'plushenko': 25551, 'sokolova': 25552, 'viktoria': 25553, 'volchkova': 25554, 'navka': 25555, 'kostomarov': 25556, 'dancers': 25557, 'bio-diesel': 25558, 'greener': 25559, 'equalize': 25560, 'sixth-largest': 25561, 'shota': 25562, 'utiashvili': 25563, 'abkhaz-controlled': 25564, 'turbine': 25565, 'poti': 25566, 'antics': 25567, 'widely-publicized': 25568, 'g-capp': 25569, 'wooten': 25570, 'raleigh': 25571, 'manchin': 25572, 'gazette': 25573, 'sago': 25574, 'fractious': 25575, 'define': 25576, 'euro-mediterranean': 25577, '09-nov': 25578, 'marker': 25579, 'estonian': 25580, 'ruutel': 25581, 'andrus': 25582, 'juhan': 25583, 'andriy': 25584, 'shevchenko': 25585, 'soviet-sponsored': 25586, 'balking': 25587, 'lightly-populated': 25588, 'qeshm': 25589, 'us-backed': 25590, 'unhitches': 25591, 'three-stage': 25592, '1600': 25593, 'friendly-fire': 25594, 'unintentional': 25595, 'demonized': 25596, 'molded': 25597, 'ex-yukos': 25598, 'managerial': 25599, 'plouffe': 25600, 'misallocation': 25601, 'renegotiate': 25602, 'military-related': 25603, 'wmd': 25604, 'rocket-launching': 25605, 'adhamiyah': 25606, 'waziriyah': 25607, 'muscular': 25608, 'shafi': 25609, 'vahiuddin': 25610, 'inward': 25611, 'cartoonists': 25612, 'previously-scheduled': 25613, 'civilizations': 25614, 'digit': 25615, 'bandage': 25616, 'quarter-on-quarter': 25617, 'retreating': 25618, 'forty-one': 25619, 'four-point': 25620, 'dissipate': 25621, 'coasts': 25622, 'coppola': 25623, 'limon': 25624, 'lawmakers-elect': 25625, 'services-based': 25626, 'sirisia': 25627, 'war-like': 25628, 'sabaot': 25629, 'heavy-handed': 25630, 'yisrael': 25631, 'beitenu': 25632, 'goverment': 25633, 'calf': 25634, '51,000': 25635, 'displeased': 25636, 'newfoundland': 25637, '12,348': 25638, 'sq': 25639, 'cashew': 25640, 'hut': 25641, 'haunch': 25642, 'mutton': 25643, 'kernels': 25644, 'senegalese-backed': 25645, 'undp': 25646, '107': 25647, 'faction-ridden': 25648, 'dahomey': 25649, 'clamor': 25650, 'outsider': 25651, 'eu-25': 25652, 'cumulative': 25653, 'imf/eu/world': 25654, 'bank-arranged': 25655, 'nigh': 25656, 'squid': 25657, 'furnish': 25658, 'self-financing': 25659, '200-mile': 25660, 'dampen': 25661, 'abated': 25662, 'perch': 25663, 'entangled': 25664, 'fleece': 25665, 'fluttered': 25666, 'clipped': 25667, 'woodcutter': 25668, 'expedient': 25669, 'importunities': 25670, 'suitor': 25671, 'cheerfully': 25672, 'toothless': 25673, 'clawless': 25674, 'woodman': 25675, 'repent': 25676, 'aisle': 25677, 'messed': 25678, 'crapping': 25679, 'semi': 25680, 'tayr': 25681, 'filsi': 25682, '18-wheeler': 25683, 'solemly': 25684, 'obsolete': 25685, 'shalikashvili': 25686, 'tacitly': 25687, 'cardboard': 25688, 'deliberating': 25689, 'savo': 25690, 'comprehend': 25691, 'distracted': 25692, 'trumped': 25693, 'balikesir': 25694, 'antoin': 25695, 'dom': 25696, 'horn-afrik': 25697, 'tutankhamun': 25698, 'skyline': 25699, 'british-sponsored': 25700, 'demerits': 25701, 'misused': 25702, 'four-time': 25703, 'blanked': 25704, 'emmen': 25705, 'under-age-20': 25706, 'otalvaro': 25707, '52nd': 25708, 'lionel': 25709, 'messi': 25710, 'doetinchem': 25711, 'taye': 25712, 'taiwo': 25713, '80th-minute': 25714, 'kolesnikov': 25715, '331': 25716, 'wirajuda': 25717, 'canadian-born': 25718, 'sassi': 25719, '347': 25720, '202.7': 25721, 'oviedo': 25722, '73.85': 25723, 'norbert': 25724, 'lammert': 25725, 'girona': 25726, 'bigots': 25727, 'apure': 25728, '2,200-kilometer': 25729, 'catalonians': 25730, 'rocio': 25731, 'esteban': 25732, 'thirty-three': 25733, 'cooperativa': 25734, 'boniface': 25735, 'localized': 25736, 'aftenposten': 25737, 'terrorism-related': 25738, 'noneducational': 25739, 'proliferate': 25740, 'balances': 25741, 'regina': 25742, 'hans-jakob': 25743, 'reichenm': 25744, 'reassurance': 25745, 'infomation': 25746, '3,04,569': 25747, 'mohean': 25748, 'nodar': 25749, 'khashba': 25750, 'voided': 25751, 'gali': 25752, 'kharj': 25753, 'majmaah': 25754, 'assorted': 25755, 'rodon': 25756, 'non-natives': 25757, 'hargeisa': 25758, 'maigao': 25759, 'paksitan': 25760, '1718': 25761, 'safta': 25762, 'naysayers': 25763, 'cynics': 25764, 'strenuously': 25765, 'democrat-controlled': 25766, 'balancing': 25767, 'counter-proliferation': 25768, 'assumptions': 25769, 'disturbingly': 25770, 'anti-flu': 25771, 'relays': 25772, 'merajudeen': 25773, 'three-party': 25774, 'russian-iranian': 25775, 'squander': 25776, 'hospitalizations': 25777, 'gripped': 25778, 'hurry': 25779, '50th': 25780, 'ysidro': 25781, 'diameter': 25782, 'jump-start': 25783, 'sinuiju': 25784, 'portman': 25785, 'think-tanks': 25786, 'iiss': 25787, 'hails': 25788, 'tipping': 25789, 'badaun': 25790, 'organiation': 25791, 'shahawar': 25792, 'matin': 25793, 'brooklyn': 25794, 'stoked': 25795, 'impress': 25796, '37-million': 25797, "qur'an": 25798, 'scud': 25799, 'showering': 25800, 'thankful': 25801, 'agendas': 25802, 'bradford': 25803, 'ernst': 25804, 'disburses': 25805, 'kasami': 25806, 'shab-e-barat': 25807, '1621': 25808, 'sophistication': 25809, 'cpsc': 25810, 'trailer': 25811, 'chieftain': 25812, 'darfuri': 25813, 'eissa': 25814, 'al-ghazal': 25815, 'tandem': 25816, 'jebaliya': 25817, 'podemos': 25818, 'akylbek': 25819, 'ricans': 25820, 'popularly-elected': 25821, 'plebiscites': 25822, 'guei': 25823, 'blatantly': 25824, 'disaffected': 25825, 'linas-marcoussis': 25826, 'guillaume': 25827, 'ouagadougou': 25828, 'reintegration': 25829, 'dramane': 25830, '6-month': 25831, 'pleasant': 25832, 'high-value-added': 25833, 'nonpolluting': 25834, 'thrives': 25835, 'jurisdictions': 25836, 'monopolies': 25837, 'computerized': 25838, 'tamper-proof': 25839, 'biometric': 25840, 'reassert': 25841, 'optician': 25842, 'telescopes': 25843, 'munificent': 25844, 'patronage': 25845, 'kangaroo': 25846, 'awkwardly': 25847, 'pouch': 25848, 'desirous': 25849, 'deceitful': 25850, 'insupportable': 25851, 'cheerless': 25852, 'unappreciated': 25853, 'implored': 25854, 'endow': 25855, 'resentment': 25856, 'fulness': 25857, 'thereof': 25858, 'incaudate': 25859, 'chin': 25860, 'wags': 25861, 'gratification': 25862, '650-million': 25863, 'charon': 25864, 'geology': 25865, 'kuiper': 25866, 'pluto-like': 25867, 'payenda': 25868, 'guerrilla-style': 25869, 'rocketed': 25870, 'caves': 25871, 'subverting': 25872, '65.5': 25873, 'mahamadou': 25874, 'issoufou': 25875, 'sarturday': 25876, 'hassas': 25877, 'checkpost': 25878, 'khela': 25879, 'barbershops': 25880, 'jabber': 25881, 'belgaum': 25882, 'mohamuud': 25883, 'musse': 25884, 'gurage': 25885, 'dastardly': 25886, 'responsibilty': 25887, 'ossetian': 25888, 'katsina': 25889, 'acquittals': 25890, 'convict': 25891, 'mazda': 25892, 'outsold': 25893, 'volkswagen': 25894, 'agadez': 25895, 'mnj': 25896, 'snow-crusted': 25897, 'immobilized': 25898, 'drifts': 25899, 'death-row': 25900, 'siding': 25901, 'six-to-three': 25902, 'antonin': 25903, 'scalia': 25904, 'hermogenes': 25905, 'esperon': 25906, 'dulmatin': 25907, 'patek': 25908, 'suspiciously': 25909, 'behaving': 25910, 'cover-up': 25911, '`80s': 25912, 'stopper': 25913, 'evin': 25914, 'hats': 25915, 'cobs': 25916, 'recognizable': 25917, 'pests': 25918, 'monsanto': 25919, 'frattini': 25920, 'colllins': 25921, 'strangely': 25922, 'ineffectively': 25923, 'liners': 25924, 'galveston': 25925, 'aurangabad': 25926, 'berrones': 25927, '57,000': 25928, 'handkerchiefs': 25929, 'colors': 25930, 'dazzled': 25931, 'documentaries': 25932, 'restless': 25933, 'controller': 25934, 'sunda': 25935, 'volcanos': 25936, 'encircling': 25937, 'val-de-grace': 25938, 'gaule': 25939, 'al-dabaan': 25940, '28-point': 25941, 'baalbek': 25942, 're-build': 25943, 'ural': 25944, 'distorted': 25945, 'pro-american': 25946, 'unbiased': 25947, 'cost-sharing': 25948, '9-percent': 25949, '32,500': 25950, 'migrate': 25951, 'roshan': 25952, '350-58': 25953, '71-26': 25954, 'pepsi': 25955, 'indra': 25956, 'madras': 25957, 'undergraduate': 25958, 'graduate': 25959, 'yale': 25960, 'skillfully': 25961, 'jordan-based': 25962, 'sightings': 25963, 'congregate': 25964, 'grower': 25965, '17th-century': 25966, 'masjid': 25967, 'eyewitness': 25968, 'forgiving': 25969, 'geologist': 25970, 'piotr': 25971, 'stanczak': 25972, 'geofizyka': 25973, 'krakow': 25974, 'contributors': 25975, 'rehearsing': 25976, 'penitence': 25977, 'kidwa': 25978, 'unami': 25979, 'muga': 25980, 'delamere': 25981, 'harbored': 25982, 'nonassociated': 25983, '2022': 25984, 'qatar-bahrain': 25985, 'causeway': 25986, 'amerindians': 25987, 'annihilated': 25988, 'sugar-related': 25989, 'revolted': 25990, "l'ouverture": 25991, 'inaugurate': 25992, 'neighbours': 25993, 'avarice': 25994, 'avaricious': 25995, 'envious': 25996, 'vices': 25997, 'spectator': 25998, 'scratched': 25999, 'jog': 26000, 'enables': 26001, 'ivaylo': 26002, 'kalfin': 26003, 'calculated': 26004, 'preparatory': 26005, 'convenience': 26006, 'afghani': 26007, 'magloire': 26008, 'retailing': 26009, 'viral': 26010, 'davit': 26011, 'nomura': 26012, 'military-imposed': 26013, 'shoot-on-sight': 26014, 'kasai': 26015, 'occidental': 26016, 'jean-constatin': 26017, '217': 26018, 'u.n.-funded': 26019, 'irin': 26020, 'burials': 26021, 'dominating': 26022, '05-mar': 26023, '15-feb': 26024, 'ilya': 26025, 'kovalchuk': 26026, 'goaltender': 26027, 'evgeni': 26028, 'nabokov': 26029, 'shutout': 26030, 'suleimaniya': 26031, 'abolishes': 26032, 'tier': 26033, '482': 26034, 'russian-built': 26035, 'salami': 26036, 'tor-m1': 26037, 'belfort': 26038, 'whistler': 26039, '44.92': 26040, '33/100ths': 26041, 'zurbriggen': 26042, '7/100ths': 26043, 'super-g': 26044, 'nkosazana': 26045, 'dlamini-zuma': 26046, 'transitioning': 26047, 'fast-track': 26048, 'extraditions': 26049, 'revert': 26050, 'mamoun': 26051, 'german-syrian': 26052, '26-member': 26053, 'pro-u.s.': 26054, 'unstoppable': 26055, 'marie-paule': 26056, 'kieny': 26057, 'pakistani-ruled': 26058, 'irreversible': 26059, 'zuckerberg': 26060, 'billionaires': 26061, 'less-visible': 26062, 'bacterium': 26063, 'ubiquitous': 26064, 'frenk': 26065, 'guni': 26066, 'rusere': 26067, '51.4': 26068, '41.6': 26069, 'otto': 26070, 'finishers': 26071, 'cpj': 26072, '36-year': 26073, '13,107': 26074, '13,090': 26075, 'prosper': 26076, 'alcoa': 26077, 'blue-chip': 26078, '552': 26079, 'santuario': 26080, 'guatica': 26081, 'restrepo': 26082, 'reintegrate': 26083, 'downbeat': 26084, '41.67': 26085, 'hausa': 26086, 'impressions': 26087, 'showered': 26088, 'unprotected': 26089, 'intercourse': 26090, 'wind-blown': 26091, 'contingency': 26092, 'five-seat': 26093, 'aid-distribution': 26094, 'verge': 26095, 'clinching': 26096, 'huckabee': 26097, 'sayyed': 26098, 'tantawi': 26099, 'bioterrorism': 26100, 'nations-coordinated': 26101, 'purifying': 26102, 'accommodating': 26103, 'navtej': 26104, 'sarna': 26105, 'fascists': 26106, 'neon': 26107, '71.79': 26108, '43rd': 26109, '72.64': 26110, 'sizes': 26111, 'underfunding': 26112, 'baltasar': 26113, 'garzon': 26114, 'ramzi': 26115, 'binalshibh': 26116, 'tahar': 26117, 'ezirouali': 26118, 'heptathlon': 26119, 'natalia': 26120, 'dobrynska': 26121, 'alwan': 26122, 'baquoba': 26123, 'two-bomb': 26124, '76.7': 26125, 'after-hours': 26126, '78.4': 26127, '77.95': 26128, 'hourmadji': 26129, 'doumgor': 26130, 'chadians': 26131, 'khaleda': 26132, 'tarique': 26133, 'taka': 26134, 'anti-graft': 26135, 'zhuang': 26136, 'collaborative': 26137, 'cabezas': 26138, 'unsatisfactory': 26139, 'anti-inflationary': 26140, 'zeros': 26141, 'dzhennet': 26142, 'abdurakhmanova': 26143, 'umalat': 26144, 'magomedov': 26145, 'markha': 26146, '102-member': 26147, 'recently-unsealed': 26148, 'leashes': 26149, 'haidari': 26150, 'cairo-based': 26151, 'illustrates': 26152, 'week-and-a-half': 26153, '275-members': 26154, 'independent-minded': 26155, 'relented': 26156, 'reappointing': 26157, 'aberrahman': 26158, 'dubai-based': 26159, 'intensity': 26160, 'catastrophes': 26161, 'happening': 26162, 'earth-penetrating': 26163, 'bunker-busters': 26164, 'jean-louis': 26165, 'gmt': 26166, 'andrey': 26167, 'denisov': 26168, 'underline': 26169, "al-madai'ni": 26170, 'u.s.-proposed': 26171, 'overrunning': 26172, '249-seat': 26173, 'particulate': 26174, 'cavic': 26175, 'azizullah': 26176, 'lodin': 26177, 'evangelical': 26178, 'zelenak': 26179, 'european-owned': 26180, 'nipayia': 26181, 'norwegian-owned': 26182, 'asir': 26183, 'handwriting': 26184, 'shorja': 26185, 'inflaming': 26186, 'uedf': 26187, 'vaccinating': 26188, 'jeanet': 26189, 'goot': 26190, 'infectiousness': 26191, 'thereby': 26192, 'inoculation': 26193, '1765': 26194, 'manx': 26195, 'gaelic': 26196, 'idi': 26197, 'promulgated': 26198, 'non-party': 26199, 'amending': 26200, '698': 26201, 'respectable': 26202, 'antananarivo': 26203, '15-month': 26204, 'simplification': 26205, 'non-textile': 26206, 'mitigated': 26207, 'graduates': 26208, 'vineyards': 26209, 'spades': 26210, 'mattocks': 26211, 'repaid': 26212, 'superabundant': 26213, 'mechanisms': 26214, 'surety': 26215, 'pleases': 26216, 'begging': 26217, 'coward': 26218, 'cookstown': 26219, 'tyrone': 26220, 'paralysed': 26221, 'waist': 26222, 'negligent': 26223, 'bernazzani': 26224, 'nacion': 26225, 'mladjen': 26226, 'vojkovici': 26227, 'fraught': 26228, '10,000-strong': 26229, 'nato-eu': 26230, 'somali-based': 26231, 'kiwayu': 26232, 'ta': 26233, 'tuol': 26234, 'sleng': 26235, 'duch': 26236, '53-year': 26237, 'adan': 26238, 'nine-and-a-half': 26239, 'mirrored': 26240, 'yet-to-be-announced': 26241, 'despondent': 26242, 'boulevard': 26243, 'infested': 26244, 'vasilily': 26245, 'filipchuk': 26246, 'thespians': 26247, 'shakespeareans': 26248, 'zaher': 26249, 'lifeline': 26250, 'guideline': 26251, 'vaccinated': 26252, 'rosal': 26253, 'deserts': 26254, 'greenery': 26255, 'sonntags-blick': 26256, 'reassigned': 26257, 'millerwise': 26258, 'dyck': 26259, 'hierarchical': 26260, 'staffed': 26261, 'pleshkov': 26262, 'krasnokamensk': 26263, 'kremlin-driven': 26264, 'forays': 26265, 'harcharik': 26266, 'angioplasty': 26267, 'khyber-pakhtunkhwa': 26268, 'impeding': 26269, 'schedules': 26270, 'mid-2007': 26271, 'al-hadjiya': 26272, 'terrorize': 26273, 'fldr': 26274, 'handler': 26275, 'canine': 26276, 'soiling': 26277, '24.5': 26278, 'ketzer': 26279, 'pre-emptively': 26280, 'halfun': 26281, 'thirty-two': 26282, '30s': 26283, 'niyazov': 26284, 'schulz': 26285, 'corroborate': 26286, 'chevallier': 26287, 'shek': 26288, 'jarrah': 26289, 'danny': 26290, 'ayalon': 26291, 'coldplay': 26292, 'rude': 26293, 'paolo': 26294, 'opined': 26295, 'nicest': 26296, '20-minute': 26297, 'testy': 26298, 'ringtone': 26299, 'edging': 26300, '0.04': 26301, '0.068171296': 26302, '0.068217593': 26303, 'aufdenblatten': 26304, '0.068263889': 26305, '0.01': 26306, '0.068275463': 26307, '782': 26308, '685': 26309, 'redesigned': 26310, 'slope': 26311, 'near-capacity': 26312, 'sangju': 26313, '4000': 26314, 'mexicali': 26315, 'marghzar': 26316, 'underfunded': 26317, 'unprepared': 26318, 'comrade': 26319, 'nigerla': 26320, 'four-story': 26321, '18-story': 26322, 'shoppertrak': 26323, 'rct': 26324, 'quietly': 26325, 'elaborating': 26326, 'chaparhar': 26327, 'radioactivity': 26328, 'anti-kurd': 26329, 'narim': 26330, 'alcolac': 26331, 'vwr': 26332, 'thermo': 26333, 'typhoon-triggered': 26334, '50.05': 26335, 'colder-than-normal': 26336, 'maruf': 26337, 'bakhit': 26338, 'kheir': 26339, 'faisal': 26340, 'fayez': 26341, 'al-turk': 26342, '23-member': 26343, '630': 26344, '18-month-old': 26345, 'minia': 26346, '22-kilometer': 26347, 'sights': 26348, '3.29': 26349, 'exxon': 26350, 'reaping': 26351, 'hers': 26352, 'cheadle': 26353, 'remixed': 26354, 'benham': 26355, 'natagehi': 26356, 'paratroopers': 26357, 'excursion': 26358, 'landale': 26359, 'mirko': 26360, 'norac': 26361, 'ademi': 26362, 'medak': 26363, 'gall': 26364, 'bladder': 26365, 'reims': 26366, 'grace': 26367, 'penjwin': 26368, 'durham': 26369, 'taxiways': 26370, 'hornafrik': 26371, 'abdirahman': 26372, 'dinari': 26373, 'sreten': 26374, 'vascular': 26375, '1726': 26376, '1828': 26377, 'tupamaros': 26378, 'frente': 26379, 'amplio': 26380, 'freest': 26381, 'single-digit': 26382, 'standard-of-living': 26383, 'salam': 26384, 'fayyad': 26385, 'uptick': 26386, 'israeli-controlled': 26387, 'high-cost': 26388, 'mandeb': 26389, 'djibouti-yemen': 26390, 'hormuz': 26391, 'iran-oman': 26392, 'malacca': 26393, 'indonesia-malaysia': 26394, 'cranes': 26395, 'plowlands': 26396, 'brandishing': 26397, 'forsook': 26398, 'liliput': 26399, 'earnest': 26400, 'suffice': 26401, 'groan': 26402, 'finger': 26403, 'misdeeds': 26404, 'whitewash': 26405, 'mortification': 26406, 'unsettling': 26407, 'buyout': 26408, 'breakout': 26409, 'inter-religious': 26410, 'kuru': 26411, 'wareng': 26412, 'barakin': 26413, 'ladi': 26414, 'pakistan-afghanistan': 26415, 'anwarul': 26416, 'u.s.-educated': 26417, 'usman': 26418, '454-seat': 26419, 'unseated': 26420, 'railed': 26421, 'w.t.o': 26422, 'wadia': 26423, 'jaish-e-mohammad': 26424, 'srinagar-based': 26425, 'hafsa': 26426, '64.35': 26427, 'sixth-straight': 26428, '203.1': 26429, '3,65,000': 26430, 'hiked': 26431, 'busier': 26432, 'co-exist': 26433, 'quelled': 26434, 'pan-africa': 26435, 'yaounde': 26436, 'cancels': 26437, 'fasted': 26438, 'nidal': 26439, 'saada': 26440, 'amona': 26441, 'overstaying': 26442, 'great-granddaughter': 26443, 'one-car': 26444, 'culpable': 26445, 'great-grandchildren': 26446, 'hrd': 26447, 'touch-screen': 26448, 'kapil': 26449, 'ipad-like': 26450, 'linux-based': 26451, 'perito': 26452, 'glacieres': 26453, 'spectacular': 26454, 'abdolsamad': 26455, 'khorramshahi': 26456, 'shatters': 26457, 'interception': 26458, 'arjangi': 26459, 'rain-swept': 26460, 'elliott': 26461, 'abrams': 26462, 'welch': 26463, 'catandunes': 26464, 'al-qaida-affiliated': 26465, 'crusade': 26466, 'satirical': 26467, '15,000-ton': 26468, 'rockefeller': 26469, 'u.s.a': 26470, 'zaldivar': 26471, 'gusting': 26472, 'treacherous': 26473, 'ice-covered': 26474, 'm-1': 26475, 'multi-car': 26476, 'pile-up': 26477, 'bratislava-trnava': 26478, '40-car': 26479, 'compensating': 26480, 'nine-day': 26481, 'tuition': 26482, 'wielding': 26483, 'biak': 26484, 'tremendously': 26485, 'substantiate': 26486, 'supposition': 26487, 'active-duty': 26488, 'involuntary': 26489, 'callup': 26490, 'chimango': 26491, '188': 26492, 'asgiriya': 26493, 'kandy': 26494, 'seam': 26495, 'spinner': 26496, 'monty': 26497, 'apr-29': 26498, 'collingwood': 26499, 'prasanna': 26500, 'jayawardene': 26501, 'alastair': 26502, 'gana': 26503, 'kingibe': 26504, 'intrude': 26505, 'unintended': 26506, 'conception': 26507, 'liken': 26508, 'latifullah': 26509, 'zukang': 26510, 'shawel': 26511, 'mosaic': 26512, 'farthest': 26513, 'snapshot': 26514, 'bang': 26515, 'turbans': 26516, 'mature': 26517, 'galactic': 26518, 'sorts': 26519, 'clarity': 26520, 'nom': 26521, 'guerre': 26522, 'masons': 26523, 'muqataa': 26524, 'heads-of-state': 26525, 'morality': 26526, 'headdress': 26527, 'pat-downs': 26528, 'wearers': 26529, 'escalates': 26530, 'shaaban': 26531, 'test-fire': 26532, 'plantings': 26533, 'headgear': 26534, 'preventative': 26535, 'pronounces': 26536, 'ex-priest': 26537, 'protégé': 26538, 'well-suited': 26539, 'jusuf': 26540, 'kalla': 26541, 'al-mukmin': 26542, 'mammy': 26543, 'fare': 26544, 'mohtarem': 26545, 'manzur': 26546, 'europe-wide': 26547, 'twice-serving': 26548, '2,46,000': 26549, 'disengage': 26550, 'adamant': 26551, 'ducked': 26552, 'masterminded': 26553, 'hangings': 26554, 'khazaee': 26555, 'tightens': 26556, 'reluctantly': 26557, '1713': 26558, 'utrecht': 26559, 'gibraltarians': 26560, 'tripartite': 26561, 'cooperatively': 26562, 'noncolonial': 26563, 'sporadically': 26564, 'durrani': 26565, '1747': 26566, 'notional': 26567, 'experiment': 26568, 'counter-coup': 26569, 'tottering': 26570, 'relentless': 26571, 'pakistani-sponsored': 26572, 'ladin': 26573, 'bonn': 26574, 'anti-sandinista': 26575, 'saavedra': 26576, 'mitch': 26577, 'pro-investment': 26578, 'march-may': 26579, 'buoyant': 26580, 'strange': 26581, 'lacerated': 26582, 'belabored': 26583, 'tonight': 26584, 'toil': 26585, 'waggon': 26586, 'rut': 26587, 'exertion': 26588, 'tuba': 26589, 'firmn': 26590, 'hotbeds': 26591, 'japanese-owned': 26592, 'mitsumi': 26593, 'foreign-owned': 26594, 'puli': 26595, 'obscure': 26596, 'mizhar': 26597, '68.9': 26598, '618': 26599, 'varema': 26600, 'tell-all': 26601, 'pummeling': 26602, 'east-west': 26603, 'west-northwesterly': 26604, 'vitamin': 26605, 'therapies': 26606, 'matthias': 26607, 'bounnyang': 26608, 'vorachit': 26609, 'ome': 26610, 'taj': 26611, 'macedonians': 26612, '216-3': 26613, '463': 26614, 'counterattack': 26615, 'four-match': 26616, 'yuganskeneftegaz': 26617, 'ramin': 26618, 'mahmanparast': 26619, 'adhaim': 26620, 'emissary': 26621, 'censors': 26622, 'golkar': 26623, 'jonah': 26624, 'brass': 26625, '4,123': 26626, '3,153': 26627, '4,532': 26628, '7,864': 26629, '13,406': 26630, '780.47': 26631, 'electricidad': 26632, 'edc': 26633, 'nationalizations': 26634, 'bad-mouthing': 26635, 'articulated': 26636, 'iadb': 26637, 'crimping': 26638, 'quarter-point': 26639, '4.75': 26640, 'worse-than-usual': 26641, '62.47': 26642, '61.25': 26643, 'mikasa': 26644, 'concubines': 26645, 'c.i.a': 26646, 'us-india': 26647, 'recently-detained': 26648, 'saidi': 26649, 'tohid': 26650, 'bighi': 26651, 'henghameh': 26652, 'somaieh': 26653, 'nosrati': 26654, 'matinpour': 26655, '239-million': 26656, '300-kilometer': 26657, 'espirito': 26658, 'catu': 26659, 'bahia': 26660, 'gasene': 26661, 'dejan': 26662, 'bugsy': 26663, 'nebojsa': 26664, 'instrumental': 26665, 'hajime': 26666, 'massaki': 26667, 're-join': 26668, 'mutalib': 26669, 'convening': 26670, 'procedural': 26671, 'gamble': 26672, 'andar': 26673, 'rejoining': 26674, 'long-ailing': 26675, '86-year-old': 26676, 'madelyn': 26677, 'impregnating': 26678, 'matabeleland': 26679, 'counterfeiting': 26680, 'geologists': 26681, 'confessions': 26682, 'mid-2003': 26683, 'badly-needed': 26684, 'censored': 26685, 'piot': 26686, 'lesser-known': 26687, 'pears': 26688, 'shoe-bombing': 26689, 'baitullah': 26690, 'perpetuating': 26691, 'a-320': 26692, 'ericq': 26693, 'les': 26694, 'cayes': 26695, 'attainable': 26696, 'augustin': 26697, 'twenty-eight-year-old': 26698, 'welt': 26699, 'saroki': 26700, 'eight-tenths': 26701, 'bolder': 26702, 'coal-producing': 26703, 'herve': 26704, 'morin': 26705, 'areva': 26706, 'silbert': 26707, 'californian': 26708, 'ditch': 26709, 'social-conservative': 26710, '51.8': 26711, 'chuck': 26712, 'hamel': 26713, 'corrosion': 26714, 'prudhoe': 26715, 'implanted': 26716, 'manufactures': 26717, 'syncardia': 26718, 'eskisehir': 26719, 'kiliclar': 26720, 'nazer': 26721, '3,300': 26722, 'osprey': 26723, 'mine-hunting': 26724, 'reacts': 26725, 'rules-based': 26726, 'unfold': 26727, 'weblogs': 26728, 'blogs': 26729, 'amazon.com': 26730, 'aleim': 26731, 'gothenburg': 26732, 'scandinavia': 26733, 'gungoren': 26734, 'savage': 26735, 'olusegan': 26736, 'trouncing': 26737, 'prefers': 26738, 'hostage-taker': 26739, 'younes': 26740, 'megawatt': 26741, 'anti-opium': 26742, 'deutchmark': 26743, 'dinar': 26744, 'infusion': 26745, 'criminality': 26746, 'chernobyl': 26747, 'riverine': 26748, 'semidesert': 26749, 'imf-recommended': 26750, 'devaluation': 26751, 'predominate': 26752, 'ranches': 26753, 'melons': 26754, '17,500': 26755, 'exemptions': 26756, 'booty': 26757, 'witnessing': 26758, 'learns': 26759, 'beg': 26760, 'requite': 26761, 'kindness': 26762, 'flattered': 26763, 'go.': 26764, 'messes': 26765, 'eats': 26766, 'behaves': 26767, '60-thousand': 26768, 'cubapetroleo': 26769, '10,000-square-kilometer': 26770, 'cnbc': 26771, 'coercive': 26772, '98,000': 26773, 'conniving': 26774, 'dismisses': 26775, '58-to-42': 26776, 'heat-trapping': 26777, 'doubting': 26778, 'bomb-mangled': 26779, 'z': 26780, 'chests': 26781, 'beltran': 26782, 'shyloh': 26783, 'entangle': 26784, '405': 26785, 'v-e': 26786, 've': 26787, 'falluja': 26788, 'conduit': 26789, 'remissaninthe': 26790, 'roared': 26791, 'off-shore': 26792, 'formulations': 26793, 'differing': 26794, '2050': 26795, 'polluting': 26796, 'bitar': 26797, 'bitarov': 26798, 'adilgerei': 26799, 'magomedtagirov': 26800, '68-34': 26801, 'xianghe': 26802, 'salang': 26803, '3,400-meter-high': 26804, 'belgians': 26805, 'statoil': 26806, 'krasnoyarsk': 26807, 'assortment': 26808, 'subtracted': 26809, 'mawlawi': 26810, 'masah': 26811, 'ulema': 26812, 'cambridge': 26813, 'azocar': 26814, 'bollywood': 26815, 'hindi': 26816, 'kiteswere': 26817, 'silverman': 26818, 'deifies': 26819, 'militarist': 26820, 'demonstrably': 26821, 'anti-narcotics': 26822, 'antony': 26823, 'm.h.': 26824, 'ambareesh': 26825, 'actor-turned': 26826, 'jayprakash': 26827, 'narain': 26828, 'rashtriya': 26829, 'dal': 26830, 'then-minister': 26831, 'three-level': 26832, 'jabel': 26833, 'creutzfeldt-jakob': 26834, 'bovine': 26835, 'spongiform': 26836, 'encephalopathy': 26837, 'alienation': 26838, 'victimize': 26839, 'loosely': 26840, 'smash': 26841, 'delivers': 26842, 'samho': 26843, 'qater': 26844, 'late-morning': 26845, 'heavily-fortified': 26846, 'frontal': 26847, 'semi-official': 26848, 'aboutorabi-fard': 26849, 'asylum-seeker': 26850, 'joongang': 26851, 'esfandiar': 26852, 'exits': 26853, 'multi-count': 26854, 'china-based': 26855, 'fang': 26856, 'llam-mi': 26857, 'impassioned': 26858, 'vandemoortele': 26859, 'defies': 26860, '1004': 26861, 'fifty-four': 26862, 'maghazi': 26863, 'thun': 26864, 'pre-natal': 26865, 'statistical': 26866, 'lindy': 26867, 'makubalo': 26868, 'olesgun': 26869, 'anti-deby': 26870, 'miserable': 26871, 'turkana': 26872, 'rongai': 26873, 'marigat': 26874, 'mogotio': 26875, 'corresponding': 26876, 'sledge': 26877, 'hammers': 26878, 'structurally': 26879, 'unsound': 26880, 'seabees': 26881, 'shujaat': 26882, 'zamir': 26883, '54-page': 26884, 'asef': 26885, 'shawkat': 26886, 'brother-in-law': 26887, 'us-run': 26888, 'bumpy': 26889, 'firebase': 26890, 'ripley': 26891, 'baskin': 26892, 'bypassed': 26893, 'sunni-dominant': 26894, 'inroads': 26895, '40-member': 26896, 'al-wefaq': 26897, 'botanic': 26898, 'partnered': 26899, 'orchids': 26900, 'splash': 26901, 'brilliance': 26902, 'shahid': 26903, 'durjoy': 26904, 'bangla': 26905, 'bogra': 26906, 'demographer': 26907, 'spouse': 26908, 'mu': 26909, 'guangzhong': 26910, 'fetus': 26911, 'non-believers': 26912, 'heartened': 26913, 'life-spans': 26914, 'shies': 26915, 'kanan': 26916, 'duffy': 26917, 'balah': 26918, 'swerve': 26919, 'aweil': 26920, 'al-hanifa': 26921, '733': 26922, 'g20': 26923, '70-page': 26924, 'azizi': 26925, 'jalali': 26926, 'benoit': 26927, 'maimed': 26928, 'mine-affected': 26929, 'ceel': 26930, 'waaq': 26931, 'campsites': 26932, 'schoolgirls': 26933, 'motorbikes': 26934, 'bundled': 26935, 'iapa': 26936, 'freedom-loving': 26937, 'less-developed': 26938, 'welfare-dependent': 26939, 'overhauling': 26940, 'entitlement': 26941, 'quota-driven': 26942, 'exceedingly': 26943, 'samoan': 26944, 'togoland': 26945, 'facade': 26946, 'rpt': 26947, 'continually': 26948, 're-welcomed': 26949, 'state-majority-owned': 26950, 'ninth-largest': 26951, 'france-albert': 26952, 'tile-maker': 26953, 'prospering': 26954, 'tilemaker': 26955, 'shine': 26956, 'bolting': 26957, 'sugar-bowl': 26958, 'abstraction': 26959, 'pickle-fork': 26960, 'fork': 26961, 'spectacles': 26962, 'needless': 26963, 'lens': 26964, 'wharf': 26965, 'stumped': 26966, 'predator': 26967, 'brewed': 26968, 'adhesive': 26969, 'togetherness': 26970, 'monosodium': 26971, 'glue': 26972, 'shimbun': 26973, 'valero': 26974, 'prejudice': 26975, 'tarcisio': 26976, 'furor': 26977, 'priestly': 26978, 'celibacy': 26979, 'galo': 26980, 'chiriboga': 26981, 'dries': 26982, 'buehring': 26983, 'gujri': 26984, 'gopal': 26985, 'slams': 26986, 'tolerating': 26987, 'animists': 26988, 'heineken': 26989, 'pingdingshan': 26990, 'coded': 26991, 'correspondence': 26992, 'deciphering': 26993, 'endeavouris': 26994, 'endeavourundocked': 26995, 'educate': 26996, 'hell': 26997, 'razor-thin': 26998, '49.8': 26999, 'jong-wook': 27000, 'well-cooked': 27001, 'gainers': 27002, 'performace': 27003, 'highest-ever': 27004, 'concacaf': 27005, 'hamarjajab': 27006, 'bridged': 27007, '63.22': 27008, '61.3': 27009, 'two-fifths': 27010, 'khatibi': 27011, 'flourish': 27012, 'abilities': 27013, 'rangeen': 27014, 'flu-infected': 27015, 'milomir': 27016, 'h7n3': 27017, 'droppings': 27018, 'thyroid': 27019, 'ak-47s': 27020, 'slayings': 27021, 'zvyagintsev': 27022, 'parulski': 27023, 'smolensk': 27024, 'fasher': 27025, 'djinnit': 27026, 'unacceptably': 27027, 'grossman': 27028, 'sliced': 27029, 'hacked': 27030, 'canons': 27031, 'glad': 27032, 'unravel': 27033, 'fabric': 27034, 'calculators': 27035, 'bookmaking': 27036, '63.95': 27037, 'khurul': 27038, 'kalmykia': 27039, 'expectation': 27040, 'ipsos': 27041, '03-may': 27042, 'khouna': 27043, 'haidallah': 27044, 'maaouiya': 27045, 'advertisement': 27046, 'expansive': 27047, 'communicated': 27048, 'fares': 27049, 'firmest': 27050, 'pan-iraqi': 27051, 'mediating': 27052, '1,600-member': 27053, 'pepper': 27054, 'truncheons': 27055, 'beyazit': 27056, 'sushi': 27057, 'kung': 27058, 'jinling': 27059, 'spiraled': 27060, 'hagino': 27061, 'uji': 27062, 'ethnic-albanian': 27063, 'shfaram': 27064, 'renen': 27065, 'schorr': 27066, 'shooter': 27067, 'kach': 27068, 'belhadj': 27069, 'inoculate': 27070, '33.5': 27071, 'then-yukos': 27072, 'ziyad': 27073, 'desouki': 27074, 'vaccinators': 27075, 'macheke': 27076, 'aeneas': 27077, 'chigwedere': 27078, 're-open': 27079, 'lci': 27080, 'al-shaalan': 27081, 'react': 27082, 'relate': 27083, 'incur': 27084, 'directives': 27085, '23.7': 27086, '73.83': 27087, '2.00': 27088, 'raiser': 27089, 'picnic': 27090, 'kickball': 27091, 'volleyball': 27092, 'crafts': 27093, 'creek': 27094, 'roast': 27095, 'fixings': 27096, '241-2661': 27097, 'jcfundrzr@aol.com': 27098, 'colonizers': 27099, '1906': 27100, 'anglo-french': 27101, 'condominium': 27102, '963': 27103, 'knife': 27104, 'benelux': 27105, 'agrochemicals': 27106, 'aral': 27107, 'stagnation': 27108, 'curtailment': 27109, '1935': 27110, 'thai-burma': 27111, 'vice-president': 27112, 'insurgencies': 27113, 'on-again': 27114, 'off-again': 27115, 'maoist-inspired': 27116, 'conqueror': 27117, 'flapped': 27118, 'exultingly': 27119, 'pounced': 27120, 'undisputed': 27121, 'comission': 27122, 'sportsmen': 27123, 'coho': 27124, 'walleye': 27125, 'reproduced': 27126, 'muskie': 27127, 'kowalski': 27128, 'france-africa': 27129, 'laotian': 27130, 'harkin': 27131, 'dispose': 27132, 'al-salami': 27133, 'waiver': 27134, 'suicides': 27135, 'farm-dependent': 27136, 'uday': 27137, 'numaniya': 27138, 'kilometer-long': 27139, 'domed': 27140, 'far-reaching': 27141, 'cykla': 27142, 'assailant': 27143, 'fathi': 27144, 'al-nuaimi': 27145, 'void': 27146, 'firmer': 27147, 'fifty-one': 27148, 'jiang': 27149, 'jufeng': 27150, 'bildnewspaper': 27151, 'fastening': 27152, 'ornament': 27153, 'align': 27154, 'propped': 27155, 'johnson-morris': 27156, 'robustly': 27157, 'bigger-than-expected': 27158, 'vote-getters': 27159, 'mizarkhel': 27160, '52.82': 27161, '737s': 27162, 'rashad': 27163, '356.6': 27164, 'defaulting': 27165, 'disbursements': 27166, 're-enactment': 27167, 'pontificate': 27168, 'delegated': 27169, 'urbi': 27170, 'orbi': 27171, 'supremacist': 27172, 'mahlangu': 27173, 'afrikaner': 27174, 'bludgeoned': 27175, '3-month': 27176, 'fenced': 27177, 'least-developed': 27178, '259': 27179, 'labs': 27180, 'kaletra': 27181, 'generic': 27182, 'overpricing': 27183, 'muted': 27184, 'malaysians': 27185, 'vacations': 27186, 'caucasian': 27187, 'violence-torn': 27188, 'teresa': 27189, 'borcz': 27190, 'limited-over': 27191, 'mailed': 27192, 'harkat-ul-zihad': 27193, 'ranbir': 27194, 'one-dayers': 27195, 'then-prime': 27196, 'mathematically': 27197, 'acrimonious': 27198, 'chrysostomos': 27199, 'heretic': 27200, 'poachers': 27201, 'skins': 27202, 'imperialist': 27203, 'crucified': 27204, 'neiva': 27205, 'jebii': 27206, '1582': 27207, 'pakhtunkhwa': 27208, 'noncompliant': 27209, 'yunlin': 27210, 'anti-secessionist': 27211, '8-year': 27212, 'mariane': 27213, 'african-based': 27214, 'odierno': 27215, 'bobbie': 27216, 'marie': 27217, 'vihemina': 27218, 'prout': 27219, 'wrought': 27220, 'singling': 27221, 'chibebe': 27222, 'seige': 27223, 'jean-tobie': 27224, '12-percent': 27225, 'homosexual': 27226, 're-examining': 27227, 'strasbourg-based': 27228, 'discriminated': 27229, 'insights': 27230, 'sulik': 27231, 'rebut': 27232, '44-yard': 27233, 'broncos': 27234, '30-oct': 27235, '41,000': 27236, 'undrafted': 27237, 'jake': 27238, 'plummer': 27239, 'one-yard': 27240, 'kicker': 27241, 'elam': 27242, 'receiver': 27243, 'samie': 27244, 'corpse': 27245, 'seven-man': 27246, 'squid-fishing': 27247, 'kits': 27248, 'arch-rivals': 27249, 'u.s.-iraq': 27250, 'luzhkov': 27251, 'satanic': 27252, 'htay': 27253, 'ashin': 27254, 'hkamti': 27255, 'sagaing': 27256, 'willian': 27257, '167-seat': 27258, 'sadiq': 27259, 'hand-picked': 27260, 'not-guilty': 27261, 'slimy': 27262, 'messiest': 27263, 'dispenser': 27264, 'goo': 27265, 'sandler': 27266, 'stiller': 27267, 'wannabe': 27268, 'youthful': 27269, 'lighthearted': 27270, 'imprisioned': 27271, 'aids-prevention': 27272, '1745': 27273, 'xinqian': 27274, 'adjusted': 27275, 'asadollah': 27276, 'obligates': 27277, 'leftwing': 27278, 'relates': 27279, 'three-phase': 27280, 'buraydah': 27281, 'cleric-supported': 27282, 'singur': 27283, 'uttarakhand': 27284, 'government-appointed': 27285, 'chapfika': 27286, 'zvakwana': 27287, '.tv': 27288, '5,200': 27289, '1,311': 27290, 'anguillan': 27291, 'intercommunal': 27292, 'cypriot-occupied': 27293, 'trnc': 27294, 'impetus': 27295, 'acquis': 27296, 'forbade': 27297, 'projecting': 27298, 'tunes': 27299, 'leaping': 27300, 'perverse': 27301, 'merrily': 27302, 'goblet': 27303, 'jarring': 27304, 'zeal': 27305, 'good-naturedly': 27306, 'benefactor': 27307, 'gnawed': 27308, 'yankees-orioles': 27309, 'baltimore': 27310, 'outfield': 27311, 'glowing': 27312, 'proboscis': 27313, 'pinna': 27314, '34-member': 27315, 'multi-story': 27316, 'near-by': 27317, 'stefano': 27318, 'pre-olympic': 27319, 'predazzo': 27320, 'belluno': 27321, 'spleen': 27322, 'cavalese': 27323, 'four-person': 27324, 'someday': 27325, 'zwelinzima': 27326, 'vavi': 27327, 'said-uz': 27328, 'mushahid': 27329, 'league-n': 27330, 'tu-160': 27331, 'squadron': 27332, 'shawn': 27333, 'brightest': 27334, 'enming': 27335, 'aspiring': 27336, 'olympian': 27337, 'sagging': 27338, '3,72,000': 27339, 'gdps': 27340, '1672': 27341, 'inevitably': 27342, 'simmers': 27343, 'policharki': 27344, 'drone-fired': 27345, 'khaisor': 27346, 'symbolism': 27347, 'heighten': 27348, 'enviable': 27349, 'xinmiao': 27350, 'qing': 27351, '1644': 27352, '6,50,000': 27353, 'ill-informed': 27354, 'taiwan-grown': 27355, 'cross-straits': 27356, 'adwar': 27357, 'rabiyaa': 27358, 'knife-wielding': 27359, 'shehzad': 27360, 'tanweer': 27361, 'hasib': 27362, 'rebel-blockade': 27363, 'far-western': 27364, 'road-block': 27365, 'slid': 27366, 'uncovering': 27367, '180,000-employee': 27368, 'taser': 27369, 'giuliani': 27370, 'hema': 27371, 'serge': 27372, 'acutely': 27373, 'ilna': 27374, 'sajedinia': 27375, 'karoubi': 27376, '29-year': 27377, 'carry-on': 27378, 'liner': 27379, "ma'ariv": 27380, 'heidelberg': 27381, 'intestine': 27382, 'hoekstra': 27383, 'diane': 27384, 'feinstein': 27385, 'dozing': 27386, 'over-rule': 27387, 'bailey': 27388, 'videolink': 27389, 'neil': 27390, 'horrifying': 27391, 'counseled': 27392, 'acorn': 27393, 'heroism': 27394, 'ignace': 27395, 'schops': 27396, "voa's": 27397, 'acorns': 27398, 'mistletoe': 27399, 'irremediable': 27400, 'insure': 27401, 'mikerevic': 27402, 'dismissals': 27403, 'flax': 27404, 'boded': 27405, "pe'at": 27406, 'sadeh': 27407, 'lastly': 27408, 'darts': 27409, 'dormitories': 27410, 'disobey': 27411, 'parliament-in-exile': 27412, 'credence': 27413, 'stubb': 27414, 'wisest': 27415, 'odde': 27416, 'leaflet': 27417, 'solitude': 27418, 'mixing': 27419, 'ayyub': 27420, 'khakrez': 27421, '14,706': 27422, 'touts': 27423, 'pilot-less': 27424, 'haleem': 27425, '121.79': 27426, '1.329': 27427, '0.65': 27428, '3.913': 27429, '2,781': 27430, '41-9': 27431, 'four-game': 27432, 'cowboys': 27433, '21-jul': 27434, 'julius': 27435, 'madhya': 27436, 'buzzard': 27437, 'wait-and-see': 27438, 'party-line': 27439, 'outstrip': 27440, '202-205-9942': 27441, 'vaeidi': 27442, 'soltanieh': 27443, 'provincal': 27444, 'crazy': 27445, 'pest': 27446, 'fouling': 27447, 'insect': 27448, 'one-sided': 27449, 'extracting': 27450, 'cesar': 27451, 'abderrahaman': 27452, 'infidel': 27453, 'swear-in': 27454, 'confiscate': 27455, 'florida-based': 27456, 'ashland': 27457, 'front-running': 27458, 'ex-partner': 27459, 'gyeong-jun': 27460, 'nuptials': 27461, 'bryant': 27462, 'sita': 27463, 'miliatry': 27464, 'garikoitz': 27465, 'pyrenees': 27466, 'wireless': 27467, 'european-built': 27468, '106.5': 27469, 'txeroki': 27470, 'myricks': 27471, 'demotion': 27472, 'samsun': 27473, 'full-length': 27474, 'vertigo': 27475, 'heralds': 27476, 'filmmaking': 27477, 'replicating': 27478, 'physiology': 27479, '715': 27480, '1278': 27481, 'andorrans': 27482, 'co-principality': 27483, '1607': 27484, 'onward': 27485, 'seu': 27486, "d'urgell": 27487, 'titular': 27488, 'benefitted': 27489, 'hellenic': 27490, '20-month': 27491, 'inequities': 27492, 'keeling': 27493, 'clunie-ross': 27494, 'cocos': 27495, '1841': 27496, 'skillful': 27497, 'meshes': 27498, 'exacted': 27499, 'promissory': 27500, '747s': 27501, 'chopper': 27502, 'locator': 27503, 'renouncing': 27504, 'crisp': 27505, 'overpaid': 27506, 'dilemma': 27507, 'athar': 27508, '17-member': 27509, 'chalit': 27510, 'phukphasuk': 27511, 'army-led': 27512, 'thirapat': 27513, 'serirangsan': 27514, 'paulino': 27515, 'matip': 27516, 'ventura': 27517, 'dans': 27518, 'mon': 27519, 'jobim': 27520, 'acronym': 27521, 'expending': 27522, 'deviated': 27523, 'forty-seven': 27524, 'fragility': 27525, 'debt-ridden': 27526, 'enrollment': 27527, 'seminars': 27528, 'confuse': 27529, 'dubai-owned': 27530, 'mckiernan': 27531, 'unites': 27532, 'taza': 27533, 'dutton': 27534, 'buchan': 27535, 'curbed': 27536, 'victors': 27537, 'peacekeeers': 27538, 'demented': 27539, 'invoke': 27540, 'emission': 27541, 'full-fledged': 27542, 'still-classified': 27543, 'two-dozen': 27544, 'kien': 27545, 'premiering': 27546, 'dramatizes': 27547, 'integrating': 27548, 'indifferent': 27549, 'dishonor': 27550, 'pararajasingham': 27551, 'arab-controlled': 27552, 'bi-partisan': 27553, 'undertake': 27554, 'yuli': 27555, 'pro-settlement': 27556, 'ronit': 27557, 'tirosh': 27558, 'reconstructed': 27559, 'pieced': 27560, 'presidencies': 27561, 'goran': 27562, 'half-staff': 27563, '3,17,000': 27564, 'tempered': 27565, 're-issued': 27566, 'gag': 27567, 'colby': 27568, 'vokey': 27569, 'mcnuggets': 27570, 'abdurrahman': 27571, 'yalcinkaya': 27572, 'obesity-related': 27573, 'unger': 27574, 'budny': 27575, 'marxists': 27576, 'jacqui': 27577, 'headless': 27578, 'mi-8': 27579, 'jamaat-i-islami': 27580, 'fitna': 27581, 'geert': 27582, 'wilders': 27583, 'quotations': 27584, 'car-bombing': 27585, 'bloom': 27586, 'narthiwat': 27587, 'sungai': 27588, 'kolok': 27589, '165-seat': 27590, 'carrasquero': 27591, 'kumgang': 27592, 'reunions': 27593, 'mclaughlin': 27594, 'recently-appointed': 27595, 'disrespectfully': 27596, 'tabulation': 27597, 'moines': 27598, 'intuitive': 27599, 'talker': 27600, 'credit-card': 27601, '3,70,000': 27602, 'dismal': 27603, 'arab-dominated': 27604, 'shuttered': 27605, 'humphreys': 27606, 'kozloduy': 27607, 'reinsurer': 27608, 'zurich-based': 27609, 'reinsurers': 27610, 'jetliner': 27611, 'cargolux': 27612, '747-8': 27613, 'japan-based': 27614, 'utilize': 27615, 'quieter': 27616, 'a-380': 27617, 'justly': 27618, 'nad-e-ali': 27619, 'militaries': 27620, 'dahoud': 27621, 'withstood': 27622, '17.7': 27623, '85.5': 27624, '14.2': 27625, 'month-on-month': 27626, '16.3': 27627, '15.1': 27628, 'tristan': 27629, 'cunha': 27630, 'agriculturally': 27631, 'tasalouti': 27632, 'expectancy': 27633, 'much-improved': 27634, 'vertical': 27635, 'horizontal': 27636, 'clusters': 27637, 'mitigate': 27638, 'outreach': 27639, 'mid-17th': 27640, '1667': 27641, 'nominally': 27642, '1650': 27643, 'saeedlou': 27644, 'precipice': 27645, 'endeavoring': 27646, 'willful': 27647, 'funny': 27648, 'mane': 27649, 'windy': 27650, 'charming': 27651, 'sisters': 27652, 'blandly': 27653, 'gust': 27654, 'glistening': 27655, 'billiard': 27656, 'embarrassed': 27657, 'woodchopper': 27658, 'besought': 27659, 'mahsouli': 27660, 'thoughtless': 27661, 'deity': 27662, 'salivated': 27663, 'amphitheatre': 27664, 'devour': 27665, 'honourably': 27666, 'claimant': 27667, 'berra': 27668, 'swear': 27669, 'stocky': 27670, 'catcher': 27671, 'miraculous': 27672, 'bothered': 27673, 'miscreants': 27674, 'toad': 27675, 'evicting': 27676, 'query': 27677, "ha'aretz": 27678, 'utilization': 27679, 'interaction': 27680, 'unsolicited': 27681, 'server': 27682, 'disabling': 27683, 'al-faraa': 27684, 'al-yamoun': 27685, 'african-inhabited': 27686, 'agha-mohammadi': 27687, 'parental': 27688, 'hamstrung': 27689, 'workhorse': 27690, 'dhanush': 27691, 'subhadra': 27692, 'prithvi': 27693, 'world-renowned': 27694, 'balletic': 27695, 'acrobatic': 27696, 'rock-and-roll': 27697, 'souq': 27698, 'al-ghazl': 27699, 'inoculations': 27700, 'rye': 27701, 'bluegrass': 27702, 'mown': 27703, 'fertilized': 27704, 'billed': 27705, 'flares': 27706, 'group-e': 27707, 'sprained': 27708, 'saddamists': 27709, 'gianni': 27710, '34,000': 27711, '36,000': 27712, 'piles': 27713, 'co-hosted': 27714, 'kiosks': 27715, 'expensively': 27716, 'rafidain': 27717, 'finsbury': 27718, 'notoriety': 27719, 'earl': 27720, 'columbiadisintegred': 27721, 're-entering': 27722, '10-dec': 27723, 'showings': 27724, 'disciplines': 27725, 'drugged': 27726, '22-nation': 27727, 're-assessed': 27728, 'headlining': 27729, 'differed': 27730, 'second-': 27731, 'el-erian': 27732, 'complains': 27733, 'poster': 27734, 'ezzat': 27735, 'itv1': 27736, 'now-retired': 27737, 'suleimaniyah': 27738, 'salah': 27739, 'reservist': 27740, 'genitals': 27741, 'kerkorian': 27742, 'shedding': 27743, 'ghosn': 27744, 'french-funded': 27745, 'contry': 27746, 'transporation': 27747, '263': 27748, '299': 27749, 'virus-sharing': 27750, 'diagnostic': 27751, 'utterly': 27752, 'irrelevant': 27753, 'simpson': 27754, 'hudson-dean': 27755, 'orjiako': 27756, 'hassanpour': 27757, 'abdolvahed': 27758, 'hiva': 27759, 'botimar': 27760, 'asou': 27761, 'guocong': 27762, '8,700': 27763, 'grossly': 27764, 'well-placed': 27765, 'jamuna': 27766, '1,500-meter': 27767, '0.051655093': 27768, 'camille': 27769, 'pin': 27770, 'karolina': 27771, 'sprem': 27772, 'military-age': 27773, 'house-by-house': 27774, 'harkleroad': 27775, 'mujahadeen': 27776, 'adjourns': 27777, 'gaston': 27778, 'gaudio': 27779, 'kanaan': 27780, 'u.s.-allied': 27781, 'parlemannews': 27782, 'nato-funded': 27783, 'alyaty': 27784, 'melange': 27785, '2-million': 27786, 'edison': 27787, 'natama': 27788, 'unavoidable': 27789, 'wannian': 27790, 'whereas': 27791, 'hanyuan': 27792, 'zhengyu': 27793, 'dung': 27794, 'structuring': 27795, 'halo': 27796, 'trenches': 27797, '1,82,000': 27798, 'lining': 27799, 'specialize': 27800, 'validated': 27801, 'si': 27802, 'cumple': 27803, 'mvr': 27804, '126.2': 27805, 'depressions': 27806, 'west-southwest': 27807, 'fourth-busiest': 27808, '1851': 27809, 'rustlers': 27810, 'pokot': 27811, 'letimalo': 27812, 'combing': 27813, 'lower-than-normal': 27814, 'bend': 27815, 'wilderness': 27816, '4,10,000': 27817, 'archipelagoes': 27818, 'aquarium': 27819, 'vlado': 27820, 'cathy': 27821, 'majtenyi': 27822, 'uwezo': 27823, 'theatre': 27824, 'colonize': 27825, 'anglo-egyptian': 27826, 'rum': 27827, 'distilling': 27828, 'croix': 27829, 'mindaugas': 27830, '1386': 27831, '1569': 27832, 'abortive': 27833, 'variations': 27834, 'lower-middle-income': 27835, 'gonsalves': 27836, 'limbs': 27837, 'notarized': 27838, 'hint': 27839, 'pinions': 27840, 'biceps': 27841, 'pugiliste': 27842, 'ladders': 27843, 'stepstools': 27844, 'macedonian-american': 27845, 'kurdish-majority': 27846, 'demontrators': 27847, 'h.w': 27848, 'iran-afghanistan': 27849, 'arafat-allies': 27850, 'al-qidwa': 27851, 'long-condemned': 27852, 'pall': 27853, 'sugars': 27854, 'diet': 27855, 'shelley': 27856, 'schlender': 27857, 'hammond': 27858, 'obamas': 27859, 'al-tayeb': 27860, 'al-musbah': 27861, 'f.b.i': 27862, 'pilgrimages': 27863, 'inflicting': 27864, 'khar': 27865, 'hacari': 27866, 'santander': 27867, 'honked': 27868, 'circled': 27869, 'unleaded': 27870, 'third-highest': 27871, '4,38,000': 27872, 'kalameh': 27873, 'sabz': 27874, 'refiner': 27875, 'bazian': 27876, 'dahuk': 27877, 'injustices': 27878, 'impulses': 27879, 'disloyalty': 27880, '20-year-olds': 27881, 'juristictions': 27882, 'kimonos': 27883, 'jamaat-ud-dawa': 27884, 'quake-affected': 27885, 'gabla': 27886, 'anti-eritrean': 27887, 'concocted': 27888, 'animosity': 27889, 'aid-to-lebanon': 27890, 'appropriations': 27891, 'jalalzadeh': 27892, 'cai': 27893, 'guo-quiang': 27894, 'gunpowder': 27895, 'designing': 27896, 'computer-controlled': 27897, 'firework': 27898, 'guggenheim': 27899, 'yiru': 27900, 'war-weary': 27901, '7000-member': 27902, 'bian': 27903, 'tanoh': 27904, 'bouna': 27905, 'abderahmane': 27906, 'vezzaz': 27907, 'tintane': 27908, 'california-mexico': 27909, 'otay': 27910, 'interrogate': 27911, 'press-ipsos': 27912, 'moken': 27913, 'banged': 27914, 'offerings': 27915, 'loftis': 27916, 'schilling': 27917, 'asafa': 27918, 'usain': 27919, 'haniyah': 27920, 'farhat': 27921, 'cautioning': 27922, 'rematch': 27923, '9.88': 27924, 'gels': 27925, 'cabins': 27926, 'one-100th': 27927, 'aidar': 27928, '9.72': 27929, 'two-100ths': 27930, 'sequence': 27931, 'tyson': 27932, 'sonora': 27933, 'wind-assisted': 27934, '9.68': 27935, 'anxiously': 27936, 'rubber-stamp': 27937, '1,69,000': 27938, 'liwei': 27939, 'naser': 27940, 'earthen': 27941, 'raba': 27942, 'borderline': 27943, 'spearheading': 27944, 'tarso': 27945, 'genro': 27946, 'treasurer': 27947, 'dirceu': 27948, 'reshuffling': 27949, 'non-contracted': 27950, 'timer': 27951, '87,000': 27952, 'magnitude-7.0': 27953, 'carnival-like': 27954, 'gaudy': 27955, 'props': 27956, 'muqudadiyah': 27957, 'silo': 27958, 'earthquake-generated': 27959, 'kicks': 27960, 'relais': 27961, 'prommegger': 27962, 'heinz': 27963, 'inniger': 27964, '1,630': 27965, 'jaquet': 27966, '1,460': 27967, 'kohli': 27968, 'daniela': 27969, 'meuli': 27970, 'pomagalski': 27971, '1,950': 27972, 'isabelle': 27973, 'blanc': 27974, 'ibb': 27975, 'jibla': 27976, 'bruno': 27977, 'pro-musharraf': 27978, 'acholi': 27979, 'aphaluck': 27980, 'bhatiasevi': 27981, '292': 27982, 'vomit': 27983, 'saliva': 27984, 'dhess': 27985, 'shalabi': 27986, 'al-dagma': 27987, 'al-tawhid': 27988, 'djamel': 27989, 'jewish-owned': 27990, 'dusseldorf': 27991, 'abdalla': 27992, 'trademarks': 27993, 'most-widely': 27994, 'counterfeiters': 27995, 'alois': 27996, 'soldaten': 27997, 'cavernous': 27998, 'armory': 27999, '300-member': 28000, '8-billion': 28001, 'lekki': 28002, '4,50,000': 28003, '6,500': 28004, 'decommission': 28005, 'acehnese': 28006, 'resent': 28007, 'burdzhanadze': 28008, '42nd': 28009, 'murphy': 28010, 'translates': 28011, 'shaleyste': 28012, 'cart': 28013, 'reconcilation': 28014, 'eritrean-based': 28015, 'alimov': 28016, 'berkin': 28017, 'chaika': 28018, 'emigre': 28019, 'pavel': 28020, 'ryaguzov': 28021, 'annualized': 28022, 'nine-point-two': 28023, 'jalbire': 28024, 'retires': 28025, 'shearer': 28026, '6.8': 28027, 'initiators': 28028, 'middleman': 28029, 'lupolianski': 28030, 'marshal': 28031, 'slavonia': 28032, 'long-successful': 28033, 'interlarded': 28034, 'downturns': 28035, 'vagaries': 28036, '9,34,000': 28037, 'regulated': 28038, 'statutory': 28039, 'gateways': 28040, 'vulnerabilities': 28041, 'alleviated': 28042, 'straw-yard': 28043, 'eater': 28044, 'ashamed': 28045, 'admirer': 28046, 'hairs': 28047, 'zealous': 28048, 'lena': 28049, 'el-bashir': 28050, 'dazzling': 28051, 'boisterous': 28052, 'one-week': 28053, 'tilt': 28054, 'kalachen': 28055, 'rapporteur': 28056, '557': 28057, '522': 28058, 'a-h1n1': 28059, 'enroute': 28060, 'whatsoever': 28061, 'xiamen': 28062, 'kaohsiung': 28063, 'lambasted': 28064, 'unsavory': 28065, 'dunking': 28066, 'filthy': 28067, 'workable': 28068, 'jocic': 28069, 'storm-ravaged': 28070, 'floodwater': 28071, 'aceh-based': 28072, 'marble': 28073, 'sculpted': 28074, 'voulgarakis': 28075, 'popovkin': 28076, 're-start': 28077, 'jabir': 28078, 'jubran': 28079, 'yemen-based': 28080, 'hezbollah-operated': 28081, 'machinists': 28082, 'insulin': 28083, 'glucose': 28084, 'snipers': 28085, 'carrizales': 28086, 'community-owned': 28087, 'atal': 28088, 'bihari': 28089, 'bhartiya': 28090, 'earle': 28091, 'prosecutorial': 28092, 'pornographic': 28093, 'pre-arranged': 28094, 'passwords': 28095, 'proliferators': 28096, 'teshome': 28097, 'toga': 28098, 'structur': 28099, '2,000-years-old': 28100, 'trans': 28101, 'unsaturated': 28102, 'clog': 28103, 'applauding': 28104, 'hubs': 28105, 'artistic': 28106, 'sahar': 28107, 'sepehri': 28108, 'evangelina': 28109, 'elizondo': 28110, 'phantom': 28111, '12.45': 28112, '915': 28113, 'konarak': 28114, 'domestically-produced': 28115, 'furnished': 28116, 'spaces': 28117, 'wmo': 28118, 'factored': 28119, 'salaheddin': 28120, 'talib': 28121, 'nader': 28122, 'shaban': 28123, 'seaport': 28124, '1750': 28125, '4,27,000': 28126, 'self-employed': 28127, 'lithuanians': 28128, 'chipaque': 28129, 'policymaking': 28130, 'sabirjon': 28131, 'kwaito': 28132, 'derivative': 28133, 'mbalax': 28134, 'youssou': 28135, "n'dour": 28136, '100th': 28137, 'stiffly': 28138, 'fracturing': 28139, 'inequalities': 28140, 'batna': 28141, '2.5-trillion': 28142, 'roulette': 28143, 'l': 28144, 'chinese-u.s.': 28145, 'irresponsibly': 28146, 'sittwe': 28147, 'birdflu': 28148, 'sparrowhawk': 28149, 'jubilation': 28150, 'unfunded': 28151, 'specializing': 28152, 'tightly-guarded': 28153, 'government-linked': 28154, 'anti-monarchy': 28155, "jama'at": 28156, 'ul': 28157, "al-qu'ran": 28158, 'al-nahda': 28159, 'rusafa': 28160, 'passport-forging': 28161, 'latifiyah': 28162, 'aps': 28163, 'intercede': 28164, '151': 28165, 'austere': 28166, 'pachyderm': 28167, 'foments': 28168, 'perception': 28169, 'favoritism': 28170, 'u.s.-israeli': 28171, '66.15': 28172, 'crimp': 28173, '58.8': 28174, '17.6': 28175, 'anti-cocaine': 28176, 'drug-fueled': 28177, '27-nov': 28178, 'arbour': 28179, 'pastoral': 28180, 'nomadism': 28181, 'sedentary': 28182, 'quarreled': 28183, 'literate': 28184, 'indebtedness': 28185, 'saa': 28186, 'duhalde': 28187, '1-to-1': 28188, 'bottomed': 28189, 'audacious': 28190, 'understating': 28191, 'exacerbating': 28192, 'dominica': 28193, 'geothermal': 28194, 'two-decade': 28195, 'ethnic-based': 28196, 'lightening': 28197, 'luxuries': 28198, 'overburdened': 28199, 'ever-changing': 28200, 'intensely': 28201, 'shining': 28202, 'anti-denmark': 28203, 'quarreling': 28204, 'hopped': 28205, 'antagonists': 28206, 'comb': 28207, 'fahim': 28208, 'emroz': 28209, 'didadgah': 28210, 'rodong': 28211, 'wenesday': 28212, 'organizational': 28213, 'faulted': 28214, 'responders': 28215, 'gijira': 28216, 'fdic': 28217, 'tartous': 28218, 'rainstorm': 28219, 'togolese-flagged': 28220, 'bean': 28221, 'flannel': 28222, 'sweaters': 28223, 'smells': 28224, 'patat': 28225, 'jabor': 28226, 'dalian': 28227, 'tian': 28228, 'neng': 28229, 'meishan': 28230, 'vent': 28231, 'west-leaning': 28232, 'reformer': 28233, 'lukewarm': 28234, 'recently-repaired': 28235, 'mosleshi': 28236, '7.37': 28237, '637': 28238, 'texmelucan': 28239, 'turkish-ruled': 28240, '20-percent': 28241, 'her-2': 28242, 'ayar': 28243, 'postpones': 28244, 'lazio': 28245, 'fiorentina': 28246, 'serie': 28247, 'handpicking': 28248, 'relegation': 28249, 'demotions': 28250, '164': 28251, 'fulani': 28252, 'herdsmen': 28253, 'all-around': 28254, 'creationism': 28255, 'gymnastics': 28256, 'tsunami-warning': 28257, 'healed': 28258, 'molucca': 28259, 'ternate': 28260, 'vivanews.com': 28261, '9.08': 28262, 'rotator': 28263, 'cuff': 28264, 'keywords': 28265, 'chat': 28266, 'syrian-dominated': 28267, 'taliban-allied': 28268, 'accessed': 28269, 'storing': 28270, 'wiretap': 28271, 'pbsnewshour': 28272, 'alternates': 28273, 'mighty': 28274, 'najjar': 28275, 'meza': 28276, 'oleksander': 28277, 'horobets': 28278, 'six-tenths': 28279, 'despicable': 28280, 'wallet': 28281, 'rfid': 28282, 'biographical': 28283, 'bezsmertniy': 28284, 'terrence': 28285, 'corina': 28286, 'morariu': 28287, 'mashona': 28288, 'u.s.-russia': 28289, 'qutbi': 28290, 'aero': 28291, 'toluca': 28292, 'alistair': 28293, 'left-handed': 28294, '297-for': 28295, '367': 28296, '323': 28297, '30.5': 28298, 'aristolbulo': 28299, 'isturiz': 28300, 'timergarah': 28301, 'sibbi': 28302, 'streamlines': 28303, 'siphon': 28304, 'palestinian-israeli': 28305, 'pullouts': 28306, 'rouen': 28307, 'haze-hit': 28308, 'manifestations': 28309, 'disproportionally': 28310, 'overrepresented': 28311, 'low-paying': 28312, 'pekanbara': 28313, 'islamophobic': 28314, 'underdocumented': 28315, 'underreported': 28316, 'hexaflouride': 28317, 'tackling': 28318, 'tindouf': 28319, 'yudhyono': 28320, 'lekima': 28321, 'guanglie': 28322, 'mesopotamia': 28323, 'abnormalities': 28324, 'pre-katrina': 28325, 'haves': 28326, 'have-nots': 28327, 'keith': 28328, 'snorted': 28329, 'ingesting': 28330, 'cremated': 28331, 'cared': 28332, 'bert': 28333, 'famously': 28334, 'hard-living': 28335, '11-judge': 28336, '10-jan': 28337, 'petitions': 28338, 'dissenting': 28339, 'edmund': 28340, 'vacated': 28341, 'nations-sponsored': 28342, 'strikingly': 28343, 'sews': 28344, 'devra': 28345, 'robitaille': 28346, 'fliers': 28347, 'washingtonians': 28348, 'mousa': 28349, "'80s": 28350, 'irrelevent': 28351, 'valuables': 28352, 'statments': 28353, 'baghaichhari': 28354, 'gravesite': 28355, 'khanty-mansiisk': 28356, 'padded': 28357, 'burdensome': 28358, 'po/psl': 28359, 'business-friendly': 28360, 'second-fastest': 28361, 'straddling': 28362, 'overexploitation': 28363, 'deepwater': 28364, '2.38.46': 28365, '0.19': 28366, 'kjetil': 28367, 'aamodt': 28368, '0.110474537': 28369, 'concessionary': 28370, 'ladder': 28371, 'deepened': 28372, 'parity': 28373, 'debt-driven': 28374, 'inflationary': 28375, '0.110625': 28376, 'maaouya': 28377, 'sid': 28378, 'afro-mauritanians': 28379, 'moor': 28380, 'arab-berber': 28381, "al-qa'ida": 28382, 'aqim': 28383, 'super-combi': 28384, 'jar': 28385, 'smeared': 28386, 'suffocated': 28387, 'expiring': 28388, 'fondles': 28389, 'nurtures': 28390, 'hates': 28391, 'neglects': 28392, 'caressed': 28393, 'smothered': 28394, 'nurtured': 28395, 'banjo': 28396, 'plotters': 28397, 'alishar': 28398, 'mirajuddin': 28399, 'giro': 28400, 'millimeter': 28401, 'rignot': 28402, 'kenedy': 28403, 'neal': 28404, 'lieutenant-colonel': 28405, 'unmonitored': 28406, 'josslyn': 28407, 'aberle': 28408, 'incarceration': 28409, 'readied': 28410, 'homayun': 28411, 'khamail': 28412, 'reassess': 28413, 'election-day': 28414, 'wnba': 28415, 'playoff': 28416, 'hawi': 28417, 'peeling': 28418, 'rides': 28419, 'khao': 28420, 'lak': 28421, 'trumpeting': 28422, 'trunks': 28423, 'calamities': 28424, 'non-perishable': 28425, 'dodger': 28426, '85-years-old': 28427, 'ex-far': 28428, 'alison': 28429, 'forges': 28430, 'sportscar': 28431, 'irwindale': 28432, 'speedway': 28433, 'deuce': 28434, 'bigalow': 28435, 'sadek': 28436, 'enzos': 28437, 'brewing': 28438, 'bottler': 28439, 'drinkers': 28440, 'brewer': 28441, 'coltan': 28442, 'cassiterite': 28443, 'social-christian': 28444, 'diaoyu': 28445, 'unguarded': 28446, 'perimeter': 28447, '83-year-old': 28448, 'olive-green': 28449, 'leftover': 28450, 'palestinian-claimed': 28451, 'retroactively': 28452, 'shuffle': 28453, '541': 28454, 'proportions': 28455, 'dharmendra': 28456, 'rajaratnam': 28457, 'sasa': 28458, 'radak': 28459, 'ovcara': 28460, 'saidati': 28461, 'publishers': 28462, 'agnes': 28463, 'uwimana': 28464, 'disobedience': 28465, '8,462': 28466, '13,543': 28467, '745.71': 28468, 'arenas': 28469, 'non-smokers': 28470, 'uladi': 28471, 'mazlan': 28472, 'jusoh': 28473, 'resonated': 28474, 'deepens': 28475, 'katarina': 28476, 'shinboue': 28477, 'leanne': 28478, 'lubiani': 28479, '87th': 28480, 'fully-loaded': 28481, 'a380': 28482, 'midair': 28483, 'rolls': 28484, 'royce': 28485, 'turbines': 28486, 'volleys': 28487, 'katyusha-type': 28488, 'bani': 28489, 'navarre': 28490, 'corpsman': 28491, 'kor': 28492, '25-meter-high': 28493, '150-meter-long': 28494, 'storm-related': 28495, 'affirmation': 28496, 'yankham': 28497, 'kengtung': 28498, 'countdown': 28499, 'local-level': 28500, 'tor': 28501, 'manan': 28502, '452': 28503, 'edible': 28504, 'non-electrical': 28505, 'kavkazcenter.com': 28506, 'abdul-khalim': 28507, 'sadulayev': 28508, 'rivaling': 28509, '30-thousand': 28510, 'telethons': 28511, '60-million': 28512, 'policy-making': 28513, 'unesco': 28514, 'parthenon': 28515, 'canadian-donated': 28516, 'pro-junta': 28517, '392': 28518, '485': 28519, 'clamped': 28520, 'drywall': 28521, 'helipad': 28522, 'mart': 28523, 'kruif': 28524, 'bedfordshire': 28525, 'bedford': 28526, 'skyjackers': 28527, '726': 28528, 'self-inflicted': 28529, 'bestseller': 28530, 'nanking': 28531, 'pesticides': 28532, 'residue': 28533, 'statistically': 28534, 'kennedys': 28535, 'jacqueline': 28536, 'onassis': 28537, 'stillborn': 28538, 'repose': 28539, 'hkd': 28540, 'konstanz': 28541, 'suitcases': 28542, 'dortmund': 28543, 'koblenz': 28544, 'kazmi': 28545, 'outnumbering': 28546, '8000': 28547, '6,23,000': 28548, 'cisplatin': 28549, 'cervix': 28550, 'small-cell': 28551, 'life-prolonging': 28552, 'hina': 28553, 'jutarnji': 28554, 'bats': 28555, 'diligently': 28556, 'shy': 28557, 'casual': 28558, 'grinstein': 28559, 'highly-competitive': 28560, 'more-profitable': 28561, 'budget-price': 28562, 'rafat': 28563, 'costliest': 28564, 'burdens': 28565, 'downsizing': 28566, 'non-export': 28567, 'cooled': 28568, 'production-sharing': 28569, 'psa': 28570, 'baku-tbilisi-ceyhan': 28571, 'nagorno-karabakh': 28572, 'chiefdoms': 28573, 'springboard': 28574, '1844': 28575, 'dominicans': 28576, 'non-representative': 28577, 'leonidas': 28578, 'trujillo': 28579, '1870': 28580, 'circumscribed': 28581, 'lateran': 28582, 'concordat': 28583, 'primacy': 28584, 'profess': 28585, 'ungrudgingly': 28586, 'commiserated': 28587, 'surely': 28588, 'financially-troubled': 28589, 'celma': 28590, 'illustrious': 28591, 'inhabitant': 28592, 'great-uncle': 28593, 'mc': 28594, "'70s": 28595, 'bullitt': 28596, 'papillon': 28597, '23-25': 28598, 'memorabilia': 28599, 'boyhood': 28600, 'yung-woo': 28601, 'meaningless': 28602, 'graz': 28603, 'nozari': 28604, 'hamshahri': 28605, 'shargh': 28606, '10-story': 28607, 'quarter-million': 28608, 'chow': 28609, 'chilled': 28610, 'high-speed': 28611, 'fuzhou': 28612, 'jessica': 28613, 'fiance': 28614, 'xtr': 28615, 'as-sultan': 28616, 'ludford': 28617, 'hanjiang': 28618, 'mashjid': 28619, 'jalna': 28620, 'osaid': 28621, 'al-falluji': 28622, 'al-najdi': 28623, 'newsletter': 28624, 'sawt': 28625, 'al-jihad': 28626, 'disseminate': 28627, 'jihadist': 28628, 'khaleeq': 28629, 'tumbling': 28630, 'kel': 28631, 'seun': 28632, 'afro-beat': 28633, 'fela': 28634, 'multi-cultural': 28635, 'nnamdi': 28636, 'moweta': 28637, 'leo': 28638, 'cheaply': 28639, 'undercutting': 28640, 'bomblets': 28641, 'hyper-inflation': 28642, 'clearances': 28643, 'assistants': 28644, 'dubs': 28645, 'monk-led': 28646, 'discriminating': 28647, 'disregarding': 28648, 'laguna': 28649, 'al-aynin': 28650, 'haaretzthat': 28651, 'zigana': 28652, 'gumushane': 28653, 'surround': 28654, 'intersect': 28655, 'semiarid': 28656, 'arla': 28657, '10-point': 28658, 'loses': 28659, 'chretien': 28660, 'secularist': 28661, 'koksal': 28662, 'toptan': 28663, 'neo-liberal': 28664, 'korean-flagged': 28665, 'rim': 28666, 'libyan-owned': 28667, 'disowning': 28668, 'notre': 28669, 'dame': 28670, 'patriarchal': 28671, 'half-page': 28672, 'al-khalayleh': 28673, 'doomsday': 28674, '70-year': 28675, 'gobe': 28676, 'squadrons': 28677, 'uprooting': 28678, 'one-million': 28679, 'conclusive': 28680, 'waded': 28681, 'unprovoked': 28682, 'hamidreza': 28683, 'estefan': 28684, 'thalia': 28685, 'jamrud': 28686, 'venevision': 28687, 'ovidio': 28688, 'cuesta': 28689, 'baramullah': 28690, 'srinigar': 28691, 'pro-freedom': 28692, 'indian-administered': 28693, 'assembling': 28694, '1,17,000': 28695, 'bikaner': 28696, 'sketch': 28697, 'amos': 28698, 'esmat': 28699, 'klain': 28700, '8,599': 28701, '90.41': 28702, 'emanate': 28703, 'dahl': 28704, 'froeshaug': 28705, 'four-man': 28706, '4x5-kilometer': 28707, 'atiku': 28708, 'lai': 28709, 'immoral': 28710, 'yambio': 28711, 'seyi': 28712, 'memene': 28713, 'kerosene': 28714, 'veil': 28715, 'ramdi': 28716, 'deepak': 28717, 'gurung': 28718, 'voyages': 28719, 'bago': 28720, 'kyauk-ein-su': 28721, 'sar-dan': 28722, 'kanyutkwin': 28723, 'allert': 28724, '428': 28725, 'sounding': 28726, 'kleiner': 28727, 'caufield': 28728, 'byers': 28729, 'computer-related': 28730, 'blossomed': 28731, 'third-in-command': 28732, 'lower-level': 28733, 'machakos': 28734, 'drank': 28735, 'makutano': 28736, 'methanol': 28737, "chang'aa": 28738, 'state-of-the-nation': 28739, 'palpa': 28740, 'bitlis': 28741, 'pejak': 28742, 'cessation': 28743, 'hawaiian': 28744, 'papahanaumokuakea': 28745, 'marine-': 28746, 'terrestrial-life': 28747, 'corals': 28748, 'shellfish': 28749, 'seabirds': 28750, 'insects': 28751, 'dollar-based': 28752, 'colon': 28753, 'traverse': 28754, 'gray-list': 28755, 'u.s.-china': 28756, 'firsts': 28757, 'aruban': 28758, 'dip': 28759, 'frequenting': 28760, 'lamentation': 28761, 'fuss': 28762, '351-7': 28763, 'batsmen': 28764, '493': 28765, 'run-chase': 28766, '6,500-strong': 28767, '70-member': 28768, 'intimidating': 28769, 'intra-palestinian': 28770, 'sucre': 28771, 'olivier': 28772, 'rochus': 28773, '49.5': 28774, 'ramstein': 28775, 'pontifical': 28776, 'poupard': 28777, 'castel': 28778, 'gandolfo': 28779, '14th-century': 28780, 'theorized': 28781, 'othman': 28782, 'swank': 28783, 'supporting-actress': 28784, 'portrayal': 28785, 'katherine': 28786, 'hepburn': 28787, 'freeman': 28788, 'supporting-actor': 28789, 'shaloub': 28790, 'orbach': 28791, 'ensemble': 28792, 'quirky': 28793, 'road-trip': 28794, 'jpmorgan': 28795, 'midfielder': 28796, 'head-butt': 28797, 'ramming': 28798, 'wael': 28799, 'al-rubaei': 28800, 'izzariya': 28801, 'embryos': 28802, 'preclude': 28803, 'underlings': 28804, 'coercion': 28805, 'impropriety': 28806, '127': 28807, 'fadela': 28808, 'chaib': 28809, 'sufferer': 28810, 'non-aggression': 28811, 'lt.-gen.': 28812, '315-member': 28813, 'union-u.n.': 28814, '26,000-member': 28815, 'kandili': 28816, 'anatolianews': 28817, 'erzurum': 28818, 'governorates': 28819, 'qods': 28820, 'shorten': 28821, 'editorials': 28822, 'gye': 28823, 'gwan': 28824, 'wafd': 28825, 'vote-buying': 28826, 'baleno': 28827, 'fatigues': 28828, 'revote': 28829, 'picnicking': 28830, 'aleg': 28831, 'qadis': 28832, 'na': 28833, 'number-eight': 28834, 'jarmila': 28835, 'gajdosova': 28836, 'ting': 28837, 'tiantian': 28838, 'opposition-controlled': 28839, 'scandalized': 28840, 'anymore': 28841, 'deeply-flawed': 28842, 'mains': 28843, 'mechanics': 28844, 'ana': 28845, 'rauchenstein': 28846, 'test-launch': 28847, 'al-sabaa': 28848, 'spelled': 28849, 'gas-guzzling': 28850, '13-member': 28851, '12-member': 28852, 'lobes': 28853, 'sedatives': 28854, 'french-egyptian': 28855, 'soir': 28856, 'acquaint': 28857, 'inflammable': 28858, '481': 28859, 'atta-mills': 28860, 'consumes': 28861, 'post-civil': 28862, 'philanthropy': 28863, 'loopholes': 28864, 'developing-world': 28865, 'hutton': 28866, 'forgivable': 28867, 'faras': 28868, 'al-jabouri': 28869, 'jabouri': 28870, 'al-aqidi': 28871, 'joanne': 28872, '57.79': 28873, '57.25': 28874, '57.7': 28875, 'fahad': 28876, 'al-ahmad': 28877, 'goldman-sachs': 28878, 'cured': 28879, 'melanne': 28880, 'ambassador-at-large': 28881, 'epidemics': 28882, 'nanjing': 28883, 'haerbaling': 28884, 'divulge': 28885, 'crafted': 28886, 'pennies': 28887, '54.95': 28888, 'tsholotsho': 28889, 'chinotimba': 28890, 'day-visit': 28891, 'inbalance': 28892, 'madero': 28893, 'otis': 28894, 'powerlines': 28895, 'whiskey': 28896, 'abolishing': 28897, 'commutes': 28898, 'footprints': 28899, 'khuzestan': 28900, 'saudia': 28901, 'jules': 28902, 'guere': 28903, 'influnce': 28904, 'majzoub': 28905, 'samkelo': 28906, 'mokhine': 28907, 'jehangir': 28908, 'mirza': 28909, 'kyoto-style': 28910, 'enthusiastically': 28911, 'acknowledgment': 28912, 'iraq-based': 28913, 'thrifty': 28914, 'handmade': 28915, 'polynesians': 28916, 'micronesians': 28917, 'vanilla': 28918, 'tongan': 28919, 'upturn': 28920, 'tanganyika': 28921, 'minimizing': 28922, 'negation': 28923, '1010': 28924, 'lower-priced': 28925, '2,80,000': 28926, 'us-cambodia': 28927, 'tifa': 28928, 'fashioning': 28929, 'poverty-ridden': 28930, 'thrush': 28931, 'intently': 28932, 'upwards': 28933, 'swoon': 28934, 'purposed': 28935, 'unawares': 28936, 'snares': 28937, 'ahmedou': 28938, 'al-ashayier': 28939, 'kouk': 28940, "sa'dun": 28941, 'hamduni': 28942, 'hmas': 28943, 'fitted': 28944, 'f-35': 28945, 'turnbull': 28946, 'doctor-patient': 28947, 'awantipora': 28948, 'sealed-off': 28949, 'saldanha': 28950, 'lateef': 28951, 'adegbite': 28952, 'tenets': 28953, 'darkest': 28954, 'al-zarqwai': 28955, 'insurgent-held': 28956, 'scrubbed': 28957, 'hangar': 28958, 'annex': 28959, 'sabri': 28960, 'annexing': 28961, '335': 28962, '1.34': 28963, 'treetops': 28964, 'compliments': 28965, 'shorter-range': 28966, 'obaidullah': 28967, 'lal': 28968, 'subcontinent': 28969, 'patched': 28970, '80-year': 28971, 'vassily': 28972, 'slandered': 28973, 'yoshie': 28974, 'sato': 28975, 'yokosuka': 28976, 'kitty': 28977, 'assessors': 28978, 'vimpelcom': 28979, 'jaunpur': 28980, 'patna': 28981, 'hexogen': 28982, 'unclaimed': 28983, 'bavaria': 28984, 'allay': 28985, 'u.s.-nigerian': 28986, 'aggressions': 28987, '59.4': 28988, '58th': 28989, 'wrecked': 28990, 'shadows': 28991, 'shines': 28992, 'aijalon': 28993, 'six-point': 28994, 'unhindered': 28995, 'pre-conflict': 28996, 'kasab': 28997, 'faizullah': 28998, 'akhbar': 28999, 'al-youm': 29000, 'sects': 29001, 'drug-making': 29002, 'narcotic': 29003, 'two-month-long': 29004, '72-hour': 29005, 'jamia': 29006, 'anti-americanism': 29007, '12,012': 29008, '12,049': 29009, 'soft-drink': 29010, 'live-fire': 29011, 'stepan': 29012, 'georgi': 29013, 'parvanov': 29014, 'prolongs': 29015, 'normalized': 29016, 'hennadiy': 29017, 'vasylyev': 29018, 'epa': 29019, 'pro-environment': 29020, 'pro-jobs': 29021, '7.3': 29022, 'ecweru': 29023, 'swahili': 29024, 'mulongoti': 29025, 'bududa': 29026, 'header': 29027, 'khedira': 29028, '82nd': 29029, 'mesut': 29030, 'oezil': 29031, 'edinson': 29032, 'cavani': 29033, 'equalized': 29034, 'forlan': 29035, 'marcell': 29036, 'jansen': 29037, 'resettling': 29038, 'unsubstantiated': 29039, 'categorized': 29040, 'shegag': 29041, 'karo': 29042, 'sixty-seven': 29043, 'al-hudaidah': 29044, '411': 29045, 'polio-free': 29046, 'badoer': 29047, 'f-430': 29048, 'spider': 29049, 'maranello': 29050, 'reggio': 29051, 'emilia': 29052, 'venice': 29053, 'ignites': 29054, 'scoop': 29055, 'footprint-shaped': 29056, 'subsurface': 29057, '679': 29058, '447': 29059, 'buyukanit': 29060, '105.97': 29061, 'dollar-priced': 29062, 'oil-importing': 29063, 'billion-member': 29064, 'wadowice': 29065, 'knees': 29066, 'worshipped': 29067, 'readies': 29068, 'meters-deep': 29069, 'seventy-two': 29070, 'gurirab': 29071, 'redistribution': 29072, 'speed-sensing': 29073, 'petting': 29074, 'binyamina': 29075, 'symptom': 29076, 'some-500': 29077, 'chilling': 29078, '22-member': 29079, 'non-local': 29080, 'bullying': 29081, 'deprive': 29082, 'mastering': 29083, 'wabho': 29084, 'shabab-held': 29085, 'electricial': 29086, 'life-long': 29087, 'trusting': 29088, 'triumf': 29089, 'vikings': 29090, 'easternmost': 29091, 'kwajalein': 29092, 'usaka': 29093, '1881': 29094, 'paroled': 29095, 'offender': 29096, 'joachim': 29097, 'ruecker': 29098, 'repressing': 29099, 'unmatched': 29100, 'zine': 29101, 'abidine': 29102, 'tunis': 29103, 'ghannouchi': 29104, "m'bazaa": 29105, 'fluctuated': 29106, 'terrible': 29107, '20-25': 29108, 'zoological': 29109, 'tales': 29110, 'graze': 29111, 'tidings': 29112, 'whence': 29113, 'entertained': 29114, 'boatmen': 29115, 'emotion': 29116, 'yielding': 29117, 'gasped': 29118, 'wretched': 29119, 'perilous': 29120, 'renoun': 29121, 'elves': 29122, 'subordinate': 29123, 'leonardo': 29124, 'overaggressive': 29125, '92,000': 29126, 'ching-hsi': 29127, 'mid-1999': 29128, '1,247': 29129, 'mistreats': 29130, 'baltim': 29131, 'kafr': 29132, 'zigazag': 29133, 'entebbe': 29134, 'jovica': 29135, 'simatovic': 29136, 'insidious': 29137, 'nihilism': 29138, 'fanatic': 29139, 'half-hour': 29140, '?al-qaida': 29141, 'copts': 29142, 'nightly': 29143, 'renegotiating': 29144, '77,000': 29145, '36.8': 29146, 'remote-control': 29147, 'cone': 29148, 'foam': 29149, 'disintegration': 29150, 'goldsmith': 29151, "al-u'zayra": 29152, 'larkin': 29153, 'khayber': 29154, 'oil-based': 29155, 'soo-hyuck': 29156, 'format': 29157, 'worthless': 29158, 'newly-appointed': 29159, 'dae': 29160, 'al-hamash': 29161, 'levied': 29162, 'prescriptions': 29163, 'somavia': 29164, 'co-educational': 29165, 'colder-than-usual': 29166, 'spelling': 29167, 'sayings': 29168, 'borjomi': 29169, 'jaca': 29170, 'pyeongchang': 29171, 'salzburg': 29172, '25-point': 29173, 'questionnaire': 29174, 'accommodations': 29175, '6th': 29176, 'buddhism': 29177, 'mid-third': 29178, 'anuradhapura': 29179, 'polonnaruwa': 29180, '1070': 29181, '1200': 29182, 'mcchrystal': 29183, '07-apr': 29184, 'pinerolo': 29185, '09-aug': 29186, '1500s': 29187, 'glides': 29188, 'circular': 29189, 'briquettes': 29190, 'aluminum-maker': 29191, 'norpro': 29192, 'xingguang': 29193, '3,200': 29194, '1796': 29195, '1802': 29196, 'alain': 29197, 'pellegrini': 29198, 'ceylon': 29199, 'lubero': 29200, 'guehenno': 29201, 'eelam': 29202, 'overshadow': 29203, 'kofoworola': 29204, 'anguish': 29205, 'cual': 29206, 'ojeda': 29207, 'borges': 29208, 'proficiency': 29209, 'indoctrinate': 29210, 'bounds': 29211, '128-member': 29212, 'weekends': 29213, 'meter-long': 29214, 'gravel': 29215, 'kihonge': 29216, 'supervising': 29217, 'at-large': 29218, 'wreaths': 29219, 'yugoslavian': 29220, 'dacic': 29221, 'ever-increasing': 29222, 'colleen': 29223, 'drillings': 29224, 'pedraz': 29225, 'taras': 29226, 'portsyuk': 29227, '40-man': 29228, 'buenaventura': 29229, 'chieftains': 29230, 'waitangi': 29231, 'npa': 29232, '1843': 29233, 'tsotne': 29234, 'zviad': 29235, 'pro-india': 29236, 'hajan': 29237, 'pre-judge': 29238, 'lapsed': 29239, 'printemps': 29240, 'public-opinion': 29241, '464': 29242, 'five-percent': 29243, 'forty~three': 29244, 'dispatching': 29245, 'vampire': 29246, 're-nationalized': 29247, 'refill': 29248, 'anti-settlement': 29249, 'four-thousand': 29250, 'trespassing': 29251, 'seini': 29252, 'leniency': 29253, 'rebuke': 29254, 'khamas': 29255, 'hajem': 29256, 'standby-agreement': 29257, 'bole': 29258, 'nihal': 29259, 'denigrates': 29260, 'ruble': 29261, 'steward': 29262, 'codes': 29263, 'super-majority': 29264, 'göteborg': 29265, 'somalia-linked': 29266, 'krisztina': 29267, 'nagy': 29268, '4,800': 29269, 'tiechui': 29270, '2,900': 29271, 'paramedics': 29272, 'irrawady': 29273, 'fazul': 29274, '5-million': 29275, 'stanezai': 29276, 'impervious': 29277, 'buster': 29278, 'lioness': 29279, 'harden': 29280, 'drift': 29281, '400-kilometer': 29282, 'wild-boar': 29283, 'easley': 29284, 'laszlo': 29285, 'solyom': 29286, 'soviet-imposed': 29287, 'mufti': 29288, 'shuttles': 29289, 'jalula': 29290, 'yourselves': 29291, 'amerli': 29292, 'petrol': 29293, 'two-term': 29294, 'invariably': 29295, 'passers-by': 29296, 'leveraged': 29297, 'maltese': 29298, 'risk-management': 29299, 'faroese': 29300, 'moorish': 29301, 'dynasties': 29302, "sa'adi": 29303, 'al-mansur': 29304, '1578': 29305, 'alaouite': 29306, '1860': 29307, 'internationalized': 29308, 'tangier': 29309, 'v': 29310, 'bicameral': 29311, 'moderately': 29312, 'hobbles': 29313, 'overspending': 29314, 'off-budget': 29315, 'overborrowing': 29316, '14-month': 29317, 'stables': 29318, 'licked': 29319, 'dainty': 29320, 'blinking': 29321, 'stroked': 29322, 'commenced': 29323, 'prancing': 29324, 'pitchforks': 29325, 'clumsy': 29326, 'jesting': 29327, 'joke': 29328, 'summers': 29329, 'clinton-era': 29330, 'good-natured': 29331, 'shibis': 29332, 'llasa': 29333, '1,157': 29334, 'radhia': 29335, 'achouri': 29336, 'no-go': 29337, 'trily': 29338, 'showdowns': 29339, 'reprimands': 29340, 'sanaa': 29341, 'coleand': 29342, 'supertanker': 29343, 'limburg': 29344, 're-training': 29345, 'supervises': 29346, 'dmitrivev': 29347, 'prestwick': 29348, 'jagdish': 29349, 'sakhalin-2': 29350, '20-kilometer': 29351, 'yongbyon': 29352, 'renaissance': 29353, 'kedallah': 29354, 'younous': 29355, 'video-sharing': 29356, 'purging': 29357, 'demeans': 29358, 'gratuitous': 29359, 'instructional': 29360, 'kuta': 29361, 'taormina': 29362, 'sexually-suggestive': 29363, 'newly-declassified': 29364, 'taboos': 29365, 'menstrual': 29366, 'unidentifed': 29367, 'provocateurs': 29368, '600-meter-long': 29369, '385': 29370, 'european-north': 29371, 'falsifying': 29372, 'pakistani-based': 29373, '60-hour': 29374, 'shootouts': 29375, 'eastward': 29376, 'military-political': 29377, 'alternately': 29378, 'ill-treatment': 29379, 'hendrik': 29380, 'taatgen': 29381, 'ambrosio': 29382, 'american-run': 29383, 'intakes': 29384, 'wastewater': 29385, 'yingde': 29386, 'reservoirs': 29387, 'neurological': 29388, 'saudi-owned': 29389, 'knockout': 29390, 'u.n.-mediated': 29391, 'strongmen': 29392, 'yardenna': 29393, 'dwelling': 29394, 'courtyard': 29395, 'annunciation': 29396, 'non-interference': 29397, 'copenhagen': 29398, 'enacting': 29399, 'snub': 29400, 'people-to-people': 29401, 'siachen': 29402, '6,000-meter': 29403, 'kalikot': 29404, 'anti-japan': 29405, 'seiken': 29406, 'spouses': 29407, 'torres': 29408, 'yake': 29409, '4,218': 29410, '3,348': 29411, '4,867': 29412, 'headbands': 29413, '8,577': 29414, '13,969': 29415, '729.88': 29416, 'peace-loving': 29417, 'jamaican-born': 29418, 'germaine': 29419, 'fanatics': 29420, 'twisted': 29421, 'anti-taleban': 29422, 'holdouts': 29423, 'drug-combination': 29424, 'aurobindo': 29425, 'pharma': 29426, 'lamivudine': 29427, 'zidovudine': 29428, 'regimen': 29429, 'patents': 29430, 'sharpened': 29431, 'ex-clinton': 29432, 'daley': 29433, 'rahm': 29434, '1.66': 29435, 'repays': 29436, 'liempde': 29437, 'nuriye': 29438, 'kesbir': 29439, 'climbs': 29440, 'cot': 29441, '70.8': 29442, 'state-of-the-union': 29443, 'breasts': 29444, 'self-exam': 29445, 'malakhel': 29446, 'osmani': 29447, '54.15': 29448, 'mid-day': 29449, 'aubenas': 29450, '57.6': 29451, 'reopens': 29452, 'xinjuan': 29453, 'bruises': 29454, 'georg': 29455, 'trondheim': 29456, 'vague': 29457, 'bernt': 29458, 'pedophilia': 29459, 'senegal-based': 29460, 'takeifa': 29461, 'album-release': 29462, 'keita': 29463, 'shyrock': 29464, '1,675': 29465, 'architectural': 29466, 'burgeoning': 29467, 'hinting': 29468, 'managua': 29469, 'disavowed': 29470, 'murr': 29471, 'oil-giant': 29472, 'auction-off': 29473, 'middle-level': 29474, 'thuggery': 29475, 'michuki': 29476, 'bitten': 29477, 'neskovic': 29478, 'bijeljina': 29479, 'seven-car': 29480, 'amagasaki': 29481, 'osaka': 29482, 'china-africa': 29483, '114.8': 29484, 'milososki': 29485, 'beneficial': 29486, 'mia': 29487, 'crvenkovski': 29488, 'gruevski': 29489, 'swerved': 29490, 'althing': 29491, '930': 29492, 'askja': 29493, 'pattle': 29494, 'woody': 29495, '1788': 29496, 'islanders': 29497, 'mechanized': 29498, 'mortgage-backed': 29499, 'rutte': 29500, 'industrialization': 29501, 'gas-based': 29502, 'soles': 29503, 'idols': 29504, 'unlucky': 29505, 'pedestal': 29506, 'hardly': 29507, 'sultans': 29508, 'splendour': 29509, 'dukeness': 29510, 'liege': 29511, 'gorgeous': 29512, 'jewel': 29513, 'gleaming': 29514, 'badgesty': 29515, 'catarrh': 29516, 'rabbit': 29517, 'obstructs': 29518, 'cheyenne': 29519, 'showers': 29520, 'wednesdays': 29521, 'upholding': 29522, 'abbasali': 29523, 'kadkhodai': 29524, 'al-tahreer': 29525, 'commercially-available': 29526, 'peshmerga': 29527, 'india-u.s.': 29528, 'christman': 29529, 'energy-starved': 29530, 'menoufia': 29531, '43nd': 29532, 'anti-genetic': 29533, 'unasur': 29534, 'basile': 29535, 'casmoussa': 29536, 'guenael': 29537, 'rodier': 29538, 'attributing': 29539, 'colonia': 29540, 'cachet': 29541, 'burrow': 29542, 'gino': 29543, 'casassa': 29544, 'repeals': 29545, 'segun': 29546, 'odegbami': 29547, 'nfa': 29548, 'salisu': 29549, 'hendarso': 29550, 'sumatran': 29551, 'religiously': 29552, '269': 29553, 'alienate': 29554, 'hazel': 29555, 'blears': 29556, 'khalidiyah': 29557, 'fifth-seeded': 29558, '22-month': 29559, 'bakwa': 29560, 'ziba': 29561, 'sharq': 29562, 'awsat': 29563, 'tempers': 29564, 'designating': 29565, 'world-number-one': 29566, 'excited': 29567, 'motel': 29568, 'dietary': 29569, 'hauled': 29570, 'anti-china': 29571, 'pragmatists': 29572, 'copes': 29573, 'revisited': 29574, 'stargazers': 29575, '50-meter': 29576, 'wd5': 29577, 'one-in-75': 29578, 'three-megaton': 29579, 'exhumation': 29580, 'glorious': 29581, 'white-clad': 29582, 'johns': 29583, 'fabrizio': 29584, 'kiffa': 29585, 'motorcyclists': 29586, 'charter97': 29587, 'cyril': 29588, 'despres': 29589, 'ktm': 29590, '8956-kilometer': 29591, 'ceremon': 29592, 'flabbergasted': 29593, 'earth-bound': 29594, 'asparagus': 29595, 'acidic': 29596, 'vapor': 29597, '500-million': 29598, 'unocal-chevron': 29599, 'cassettes': 29600, 'taliban-style': 29601, 'augment': 29602, 'mcbride': 29603, 'vocalist': 29604, 'nichols': 29605, 'craig': 29606, 'wiseman': 29607, 'peformed': 29608, 'mcgraw': 29609, 'gretchen': 29610, 'horizon': 29611, 'kris': 29612, 'kristofferson': 29613, 'zamani': 29614, 'speedboats': 29615, 'coherent': 29616, 'seven-year': 29617, 'nyein': 29618, 'min': 29619, 'ko': 29620, 'naing': 29621, 'appropriated': 29622, 'pro-al-sadr': 29623, 'fadhila': 29624, 'perverted': 29625, 'cctv': 29626, 'mi-5': 29627, 'fuel-pump': 29628, 'nozzles': 29629, 't': 29630, 'salomi': 29631, 'reads': 29632, 'blackwater': 29633, 'ire': 29634, 'benedicto': 29635, 'el~commercio': 29636, 'zevallos': 29637, 'dee': 29638, 'boersma': 29639, 'indian-based': 29640, 'serfdom': 29641, 'dumitru': 29642, 'newsroom': 29643, 'formalities': 29644, 'dens': 29645, 'airdrop': 29646, 'snow-packed': 29647, 'ingmar': 29648, 'actresses': 29649, 'ullman': 29650, 'bibi': 29651, 'andersson': 29652, 'faro': 29653, 'strawberries': 29654, 'madness': 29655, 'tinged': 29656, 'melancholy': 29657, 'humor': 29658, 'iconic': 29659, 'knight': 29660, 'bylaw': 29661, 'thg': 29662, 'tetrahydrogestrinone': 29663, '100-meters': 29664, 'boa': 29665, 'american-israel': 29666, 'mailbox': 29667, 'measurer': 29668, 'machining': 29669, 'invent': 29670, 'braille': 29671, 'sight-impaired': 29672, 'nanotechnology': 29673, 'cybersecurity': 29674, 'grid': 29675, 'standardized': 29676, 'hoses': 29677, 're-measured': 29678, 'brand-new': 29679, 'mummified': 29680, 'earthenware': 29681, 'well-preserved': 29682, 'teir': 29683, 'atoun': 29684, 'third-holiest': 29685, 'open-air': 29686, 'habsburg': 29687, 'anne': 29688, 'emmerick': 29689, 'visions': 29690, 'mel': 29691, 'gibson': 29692, 'ludovica': 29693, 'angelis': 29694, 'mid-20th': 29695, 'beatification': 29696, 'directorate': 29697, 'touted': 29698, 'handovers': 29699, 'third-ranked': 29700, 'vigilante': 29701, 'edwin': 29702, 'workload': 29703, 'principlists': 29704, 'reem': 29705, 'zeid': 29706, 'khazaal': 29707, 'gheorghe': 29708, 'flutur': 29709, 'caraorman': 29710, 'huy': 29711, 'nga': 29712, 'locus': 29713, 'moldovan': 29714, 'dniester': 29715, 'slavic': 29716, 'transnistria': 29717, 'pcrm': 29718, 'aie': 29719, 'reconstituted': 29720, 'aie-coalition': 29721, 'barren': 29722, 'sub-antarctic': 29723, 'heyday': 29724, '1755': 29725, 'guizhou': 29726, 'crocodile': 29727, 'murderer': 29728, 'bosom': 29729, 'coldness': 29730, 'grin': 29731, 'coals': 29732, 'pleasures': 29733, 'thawed': 29734, 'civilly': 29735, '2,38,000': 29736, 'foreshadows': 29737, '940': 29738, 'bedingfield': 29739, 'dreaming': 29740, 'beatboxers': 29741, 'dispenses': 29742, 'aids-causing': 29743, 'dose': 29744, 'anybody': 29745, 'kapondi': 29746, 'joshua': 29747, 'kutuny': 29748, 'wilfred': 29749, 'machage': 29750, 'dragmulla': 29751, 'hurl': 29752, 'wal-marts': 29753, 'elvira': 29754, 'nabiullina': 29755, 'gucht': 29756, 'shoaib': 29757, 'pan-african': 29758, 'festus': 29759, 'okonkwo': 29760, 'attends': 29761, 'u.s.-sub': 29762, 'amed': 29763, 'snag': 29764, 'exclusively': 29765, 'loophole': 29766, 'kordofan': 29767, 'sla': 29768, 'ghubaysh': 29769, 'kilometers-per-hour': 29770, 'mudslide-prone': 29771, 'federations': 29772, 'edel': 29773, 'checkup': 29774, 'poet': 29775, 'rivero': 29776, 'espinoso': 29777, 'chepe': 29778, 'conspired': 29779, 'gulzar': 29780, 'nabeel': 29781, 'shamin': 29782, 'uddin': 29783, 'treasures': 29784, 'dakhlallah': 29785, 'ideals': 29786, 'brice': 29787, 'flashlight': 29788, 'interrogator': 29789, 'azmi': 29790, 'enkhbayar': 29791, 'five-million': 29792, 'cali': 29793, 'barco': 29794, 'bogata': 29795, 'headache': 29796, 'tagging': 29797, 'mastered': 29798, 'earth-like': 29799, 'doused': 29800, 'blizzards': 29801, 'herders': 29802, 'late-year': 29803, 'pre-famine': 29804, 'afder': 29805, 'liban': 29806, 'gode': 29807, 'deteriorate': 29808, 'atomstroyexport': 29809, 'blowtorch': 29810, 'deteriorates': 29811, 'kisangani': 29812, 'shake-up': 29813, '62.3': 29814, '60.98': 29815, 'kidnapper': 29816, 'temur': 29817, 'pasted': 29818, 'krivtsov': 29819, 'iraqna': 29820, 'posh': 29821, 'makati': 29822, 'deluxe': 29823, 'ghorian': 29824, 'thailand-flagged': 29825, 'navfor': 29826, 'harardhere': 29827, 'sacrificial': 29828, 'ubt': 29829, 'trampling': 29830, 'keck': 29831, 'vortices': 29832, 'teenaged': 29833, 'mastung': 29834, 'voltage': 29835, 'naushki': 29836, 'inter-faith': 29837, 'wellhead': 29838, 'cawthorne': 29839, 'rica-based': 29840, 'ein': 29841, 'azhar': 29842, 'dignified': 29843, 'emails': 29844, 'flakus': 29845, 'weird': 29846, 'mingora': 29847, 'neolithic': 29848, 're-tried': 29849, 'republished': 29850, 'alok': 29851, '10-to-eight': 29852, 'lauded': 29853, 'palocci': 29854, 'rato': 29855, 'nations-protected': 29856, 'ringing': 29857, '15-day': 29858, 'tet': 29859, 'hermosa': 29860, 'myitsone': 29861, 'kachin': 29862, 'humam': 29863, 'hammoudi': 29864, 'al-kharj': 29865, 'narayan': 29866, 'huerta': 29867, 'knut': 29868, 'elfriede': 29869, 'jelinek': 29870, 'irreparably': 29871, 'horace': 29872, 'engdahl': 29873, 'madison': 29874, 'rose-tinted': 29875, 'glasses': 29876, 'lennon': 29877, 'lyricist': 29878, 'taupin': 29879, 'serenaded': 29880, 'whoopie': 29881, 'goldberg': 29882, 'union-latin': 29883, 'portraying': 29884, '1,59,000': 29885, 'shaheen': 29886, 'nexus': 29887, 'cheecha': 29888, 'watni': 29889, 'c-295': 29890, 'corvettes': 29891, 'overruns': 29892, 'sipah-e-sahaba': 29893, 'husbandry': 29894, 'conscientious': 29895, 'entrepreneurship': 29896, 'supplements': 29897, 'pretended': 29898, 'lameness': 29899, 'pulse': 29900, 'mouthful': 29901, 'vexation': 29902, 'sup': 29903, 'flagon': 29904, 'insert': 29905, 'requital': 29906, 'pager': 29907, 'anesthetist': 29908, 'stethoscope': 29909, 'dangling': 29910, 'hardfought': 29911, 'opium-producing': 29912, '286': 29913, '252': 29914, 'tuncay': 29915, 'heroic': 29916, 'tabinda': 29917, 'double-bombing': 29918, 'kamiya': 29919, 'british-dutch': 29920, 'e-mailed': 29921, '18-million': 29922, 'outfielders': 29923, 'sammy': 29924, 'sosa': 29925, 'fourth-highest': 29926, 'leagues': 29927, 'hander': 29928, '2.98': 29929, 'strikeouts': 29930, 'cy': 29931, 'rippled': 29932, 'called-off': 29933, 'deauville': 29934, 'kouguell': 29935, 're-building': 29936, 'padang': 29937, 'daraga': 29938, 'durian': 29939, 'catanduanes': 29940, 'al-enezi': 29941, 'kut': 29942, 'rescind': 29943, '57-year-old': 29944, 'hong-ryon': 29945, 'dawoud': 29946, 'rasheed': 29947, 'thawing': 29948, 'predictor': 29949, 'stump': 29950, 'carryover': 29951, 'candlemas': 29952, '400-page': 29953, 'wwii': 29954, 'blared': 29955, 'al-dabbagh': 29956, 'keynote': 29957, 'complementary': 29958, 'genome': 29959, 'dripped': 29960, 'notion': 29961, 'atttacks': 29962, 'hurriya': 29963, 'kufa': 29964, 'boodai': 29965, 'flood-swamped': 29966, 'tearfund': 29967, 'andrhra': 29968, 'ar-rutbah': 29969, 'khurram': 29970, 'mehran': 29971, 'msn': 29972, 'bing': 29973, 'sufi': 29974, 'sufis': 29975, 'heretics': 29976, 'abdul-hussein': 29977, 'turkman': 29978, 'qaratun': 29979, 'khairuddin': 29980, '590': 29981, 'yair': 29982, 'landlords': 29983, 'mykolaiv': 29984, 'chasing': 29985, 'loyalist': 29986, 'lvf': 29987, 'tyrants': 29988, 'oppress': 29989, 'rocker': 29990, 'springsteen': 29991, 'repertoire': 29992, 'idolwinner': 29993, 'no-man': 29994, 'refrigerated': 29995, 'abass': 29996, '58.5': 29997, 'non-manufacturing': 29998, 'barbero': 29999, 'neutralizing': 30000, 'then-iraqi': 30001, 'tessa': 30002, 'jowell': 30003, 'expose': 30004, 'baojing': 30005, 'jianhao': 30006, 'blackmailing': 30007, 'maurice': 30008, '88th': 30009, 'somme': 30010, 'triomphe': 30011, 'pongpa': 30012, 'wasser': 30013, 'backup': 30014, 'irreconcilable': 30015, '22-month-old': 30016, 'tracy': 30017, '90-81': 30018, 'second-straight': 30019, 'lebron': 30020, 'pacers': 30021, '99-89': 30022, 'croshere': 30023, 'steals': 30024, 'dunleavy': 30025, 'faridullah': 30026, 'kani': 30027, 'wam': 30028, 'four-point-five': 30029, 'interivew': 30030, 'jianzhong': 30031, 'surkh': 30032, 'nangahar': 30033, 'pro-immigrant': 30034, 'guest-worker': 30035, 'penalize': 30036, 'baptismal': 30037, 'arranges': 30038, 'shutters': 30039, 'portico': 30040, 'rites': 30041, 'anti-capitalist': 30042, 'asserting': 30043, 'sarabjit': 30044, 'moods': 30045, 'retirements': 30046, '8,17,000': 30047, 'politicized': 30048, 're-submit': 30049, 'derg': 30050, 'selassie': 30051, 'wide-scale': 30052, 'kingdoms': 30053, 'colchis': 30054, 'kartli-iberia': 30055, 'a.d': 30056, '330s': 30057, 'oceanographic': 30058, 'circumpolar': 30059, 'mingle': 30060, 'discrete': 30061, 'ecologic': 30062, 'concentrates': 30063, 'approximates': 30064, '1631': 30065, 'retook': 30066, '1633': 30067, 'amongst': 30068, 'roamed': 30069, 'amused': 30070, 'frighten': 30071, 'perched': 30072, 'beak': 30073, 'wily': 30074, 'stratagem': 30075, 'complexion': 30076, 'deservedly': 30077, 'deceitfully': 30078, 'refute': 30079, 'reproached': 30080, 'luxuriate': 30081, 'depicted': 30082, 'zuhair': 30083, 'al-chalabi': 30084, 'reapprove': 30085, 'upside-down': 30086, 'abiya': 30087, 'rizvan': 30088, 'chitigov': 30089, 'shali': 30090, 'rescinded': 30091, 'judo': 30092, 'repatriation': 30093, 'esperanza': 30094, 'copiapo': 30095, 'shadowed': 30096, 'nazran': 30097, 'zyazikov': 30098, 'jafri': 30099, 'venomous': 30100, 'vindicated': 30101, 'samara': 30102, 'arsamokov': 30103, '2,70,000': 30104, 'ingush': 30105, '15-nation': 30106, 'condominiums': 30107, 'newly-constructed': 30108, 'equipping': 30109, 'interacted': 30110, 'kayhan': 30111, 'anti-kidnapping': 30112, 'pinas': 30113, 'schoof': 30114, 'ayala': 30115, 'alabang': 30116, '5,50,000': 30117, 'baikonur': 30118, 'cosmodrome': 30119, 'pythons': 30120, 'kandal': 30121, 'wedded': 30122, 'kroung': 30123, 'pich': 30124, 'cage': 30125, 'subscribe': 30126, 'animism': 30127, 'inhabit': 30128, 'inanimate': 30129, 'snakes': 30130, 'neth': 30131, 'vy': 30132, 'then-tiny': 30133, '45-thousand': 30134, 'dharmsala': 30135, 'acknowledging': 30136, 'worthwhile': 30137, 'rosaries': 30138, 'scorpion': 30139, 'venom': 30140, 'selectively': 30141, 'trial-taking': 30142, '516': 30143, 'league-wide': 30144, 'howell': 30145, 'enlarged': 30146, 'franchises': 30147, 'echocardiograms': 30148, 'heart-related': 30149, 'knicks': 30150, 'eddy': 30151, 'curry': 30152, 'irregular': 30153, 'heartbeat': 30154, 'hoiberg': 30155, 'guenter': 30156, 'verheugen': 30157, 'five-judge': 30158, 'sheeting': 30159, 'coveted': 30160, 'entitle': 30161, 'salar': 30162, 'uyuni': 30163, 'spanish-chinese': 30164, 'dahir': 30165, 'aweys': 30166, 'powerless': 30167, 'tvn': 30168, 'stainmeier': 30169, 'el-masri': 30170, 'butmir': 30171, 'eufor': 30172, 'karameh': 30173, 'anti-syria': 30174, 'dwarfed': 30175, 'pressewednesday': 30176, 'bud': 30177, 'selig': 30178, 'fehr': 30179, 'sox': 30180, 'sama': 30181, 'murillo': 30182, 'hokkaido': 30183, 'stiffer': 30184, 'carphedon': 30185, 'murcia': 30186, 'gerolsteiner': 30187, 'issori': 30188, 'k.': 30189, 'scourge': 30190, 'buoys': 30191, 'equipment-2': 30192, 'thammararoj': 30193, 'killer': 30194, 'dario': 30195, 'kordic': 30196, 'ganiyu': 30197, 'adewale': 30198, 'backed-militias': 30199, 'badoosh': 30200, 'babaker': 30201, 'zibari': 30202, 'lajcak': 30203, 'clamps': 30204, 'anand': 30205, 'sterilized': 30206, 'nimal': 30207, 'siripala': 30208, 'salvaging': 30209, 'norway-brokered': 30210, 'four-at-a-time': 30211, 'cabs': 30212, 'fortunate': 30213, 'feasts': 30214, 'costan': 30215, 'border-crossing': 30216, 'second-lowest': 30217, '10.8': 30218, '5,10,000': 30219, '6,10,000': 30220, 'emphatically': 30221, 'sakineh': 30222, 'mohammadi': 30223, '43-year-old': 30224, 'starve': 30225, 'inexplicable': 30226, '5\xa0storm': 30227, 'low-pressure': 30228, 'measurement': 30229, 'measurements': 30230, 'bullet-ridden': 30231, 'outcries': 30232, 'shinsei': 30233, 'maru': 30234, 'ben-david': 30235, 'compensatiion': 30236, 'bayu': 30237, 'krisnamurthi': 30238, 'parviz': 30239, 'davoodi': 30240, '59-year-old': 30241, 'vollsmose': 30242, 'odense': 30243, 'conversions': 30244, 'evangelization': 30245, 'three-million': 30246, 'volcanologist': 30247, 'lene': 30248, 'jean-bernard': 30249, '185-million': 30250, 'anti-aids': 30251, 'understates': 30252, 'thankot': 30253, 'bhairahawa': 30254, 'geophysical': 30255, 'moroccan-born': 30256, 'sparse': 30257, 'congratulation': 30258, 'summarizes': 30259, 'pollutes': 30260, 'afghanis': 30261, 'waistcoat': 30262, 'stained': 30263, 'inter-linked': 30264, 'biakand': 30265, 'al-sunnah': 30266, '215': 30267, 'aulnay-sous-bois': 30268, 'eker': 30269, 'mellot': 30270, 'korangal': 30271, 'shon': 30272, 'meckfessel': 30273, 'fadlallah': 30274, 'dispel': 30275, 'koyair': 30276, '64-year': 30277, 'drunkedness': 30278, 'ac-130': 30279, 'crosses': 30280, 'hossam': 30281, 'zaki': 30282, 'sadat': 30283, '10.7': 30284, 'outstripped': 30285, 'legia': 30286, 'football-related': 30287, 'elijah': 30288, 'missteps': 30289, 'front-page': 30290, 'maas': 30291, 'muhamed': 30292, 'garza': 30293, 'anti-inflammatory': 30294, 'hurricane-': 30295, 'superdome': 30296, 'workouts': 30297, 'trinity': 30298, 'alamodome': 30299, 'herbal': 30300, 'palnoo': 30301, 'kulgam': 30302, 'siasi': 30303, 'high-voltage': 30304, 'unprincipled': 30305, 'coldest': 30306, 'mikulski': 30307, 'shyster': 30308, 'uhstad': 30309, 'four-star': 30310, 'hardliner': 30311, 'indicative': 30312, '3700': 30313, 'bermel': 30314, 'slumped': 30315, 'bargains': 30316, 'resales': 30317, 'kibembi': 30318, 'shryock': 30319, 'carcasses': 30320, 'manifest': 30321, 'soviet-made': 30322, 'portoroz': 30323, 'obita': 30324, 'de-escalate': 30325, 'rostov-on-don': 30326, 'checkups': 30327, 'block-to-block': 30328, 'dhangadi': 30329, 'reclaimed': 30330, 'fragmentation': 30331, 'nesterenk': 30332, 'nuclear-powered': 30333, 'cruiser': 30334, 'chabanenko': 30335, 'rearming': 30336, 'al-harith': 30337, 'tayseer': 30338, 'endeavouron': 30339, 'spacewalking': 30340, 'reaffirms': 30341, 'bergner': 30342, 'rish': 30343, 'u.s.-pakistani': 30344, 'ashfaq': 30345, 'parvez': 30346, 'kayani': 30347, 'fatou': 30348, 'bensouda': 30349, 'hemingway': 30350, 'judge-and-jury': 30351, 'brighton': 30352, 'everado': 30353, 'tobasco': 30354, 'montiel': 30355, 'handpicked': 30356, 'higher-level': 30357, 'ofakim': 30358, 'non-residents': 30359, '376-7': 30360, 'feb-88': 30361, 'resurrected': 30362, 'tajouri': 30363, 'anti-violence': 30364, 'campaigner': 30365, 'polska': 30366, 'diminishes': 30367, 'obsession': 30368, 'urdu': 30369, 'american-based': 30370, 'ethnomusicologist': 30371, 'q.': 30372, 'nabarro': 30373, 'poorly-equipped': 30374, 'mirwaiz': 30375, 'line-of-control': 30376, 'vilsack': 30377, 'kucinich': 30378, 'al-dulaymi': 30379, '4.25': 30380, 'nationalizes': 30381, 'eugenio': 30382, 'anez': 30383, 'nunez': 30384, 'romani': 30385, 'dulaymi': 30386, 'non-political': 30387, 'lately': 30388, 'distract': 30389, 'morshed': 30390, 'ldp': 30391, '53.2': 30392, 'responsive': 30393, 'jawed': 30394, 'ludin': 30395, 'sordid': 30396, 'horticulture': 30397, 'evolving': 30398, 'derive': 30399, 'onerous': 30400, 'mid-to-late': 30401, 'jdx': 30402, 'high-priced': 30403, '1.27': 30404, '27-month': 30405, 'multilaterals': 30406, 'exacerbates': 30407, 'tarmiyah': 30408, 'judaidah': 30409, 'ratios': 30410, 'mingling': 30411, 'taleban-linked': 30412, 'kurdish-dominated': 30413, 'zalinge': 30414, 'minimized': 30415, 'demoralized': 30416, 'anti-labor': 30417, 'disinvest': 30418, 'suleyman': 30419, 'aysegul': 30420, 'esenler': 30421, 'bourgas': 30422, 'egebank': 30423, 'joblessness': 30424, 'haridy': 30425, 'mirdamadi': 30426, 'pakhtoonkhaw': 30427, 'iteere': 30428, 'tracing': 30429, 'almazbek': 30430, 'appointee': 30431, 'hashem': 30432, 'ashibli': 30433, '33-member': 30434, 'arabiya': 30435, 'u.n.-proposed': 30436, 'manie': 30437, 'clerq': 30438, 'torching': 30439, 'jumbled': 30440, 'acclaimed': 30441, 'avant-garde': 30442, 'yakov': 30443, 'chernikhov': 30444, '274': 30445, 'antiques': 30446, 'hermitage': 30447, 'gilded': 30448, 'ladle': 30449, 'f.s.b': 30450, 'gastrostomy': 30451, 'zoba': 30452, 'yass': 30453, 'al-aalam': 30454, 'saloumi': 30455, 'nkunda': 30456, 'rutshuru': 30457, 'disproportionate': 30458, 'cordero': 30459, 'montezemolo': 30460, 'corriere': 30461, 'della': 30462, 'sera': 30463, 'u.s.-occupied': 30464, '6,26,000': 30465, 'information-collecting': 30466, 'disregards': 30467, 'bajur': 30468, 'el~arish': 30469, 'resealed': 30470, 'squeezing': 30471, 'workforces': 30472, 'pro': 30473, 'democratia': 30474, 'snow-covered': 30475, 'then-georgian': 30476, 'channu': 30477, 'torchbearers': 30478, '64-day': 30479, '11,300': 30480, 'massaua': 30481, 'turin-born': 30482, 'livio': 30483, 'berruti': 30484, 'palazzo': 30485, 'citta': 30486, 'kissem': 30487, 'tchangai-walla': 30488, 'lome': 30489, 'emmanual': 30490, 'one-percent': 30491, '0.55': 30492, 'townsend': 30493, 'suicide-bomber': 30494, 'jackets': 30495, 'france-3': 30496, 'frenchmen': 30497, 'asylum-seeking': 30498, 'choi': 30499, 'youn-jin': 30500, 'spilt': 30501, 'unjustifiable': 30502, 'beruit': 30503, 'long-armed': 30504, 'odyssey': 30505, 'indignation': 30506, 'boulder': 30507, 're-form': 30508, 'zwally': 30509, 'bioethics': 30510, 'sperm': 30511, 'kompas': 30512, 'iwan': 30513, 'rois': 30514, 'underlined': 30515, 'ridding': 30516, 'untrue': 30517, '50.83': 30518, '1.224': 30519, '1452': 30520, '1.459': 30521, 'decried': 30522, 'vadym': 30523, 'chuprun': 30524, 'preach': 30525, 'namik': 30526, 'infuriated': 30527, 'russian-backed': 30528, 'abdulla': 30529, 'al-shahin': 30530, 'pegs': 30531, 'free-floating': 30532, 'pagonis': 30533, 'habila': 30534, 'bluefields': 30535, 'steelers': 30536, '20-17': 30537, 'chargers': 30538, '17-jun': 30539, '27-20': 30540, 'seahawks': 30541, '34-17': 30542, '55-nation': 30543, 'tarija': 30544, 'gas-separation': 30545, 'one-billion-dollars': 30546, '212-to-206': 30547, '308-to-106': 30548, 'afari': 30549, 'djan': 30550, 'brong': 30551, 'ahafo': 30552, 'npp': 30553, 'duping': 30554, 'fleeced': 30555, 'ziauddin': 30556, 'gilgit': 30557, 'conoco': 30558, 'gela': 30559, 'bezhuashvili': 30560, 'abiding': 30561, 'rebuffing': 30562, '8,206': 30563, '602': 30564, 'western-style': 30565, 'universally': 30566, 'originates': 30567, 'latgalians': 30568, 'ca.': 30569, 'panabaj': 30570, 'moviegoers': 30571, 'irate': 30572, 'propriety': 30573, 'thirty-one-year-old': 30574, 'bhagwagar': 30575, 'sensationalizing': 30576, '100-day': 30577, 'revalue': 30578, 'churn': 30579, '345': 30580, 'anatol': 30581, 'liabedzka': 30582, 'krasovsky': 30583, '1776': 30584, 'confederacy': 30585, 'buoyed': 30586, 'desolate': 30587, 'indisputably': 30588, '1614': 30589, 'trappers': 30590, 'beerenberg': 30591, '1,75,000': 30592, 'winters': 30593, 'victorian': 30594, 'awkward': 30595, 'two-part': 30596, 'wajed': 30597, 'harangued': 30598, 'nineva': 30599, 'pushpa': 30600, 'dahal': 30601, 'dasain': 30602, 'risk-sharing': 30603, 'predetermined': 30604, 'stunned': 30605, 'reverberated': 30606, 'downfall': 30607, 'prabtibha': 30608, 'patil': 30609, 'quarter-of-a-million': 30610, 'fallujans': 30611, 'swazis': 30612, 'grudgingly': 30613, 'backslid': 30614, '1,55,000': 30615, 'fisher-price': 30616, 'gagging': 30617, 'cesme': 30618, 'ksusadasi': 30619, 'bassel': 30620, 'sanitary': 30621, 'typhoid': 30622, 'morelos': 30623, 'ex-diplomat': 30624, 'disdain': 30625, 'lugansville': 30626, 'espiritu': 30627, 'non-elected': 30628, 'bankroll': 30629, '1845': 30630, 'chandpur': 30631, 'apologies': 30632, 'giorgio': 30633, 'exhort': 30634, 'darwin': 30635, 'resurrect': 30636, 'adana': 30637, 'disparage': 30638, 'enraged': 30639, 'idolatry': 30640, 'eventide': 30641, 'selective': 30642, 'snowed': 30643, 'strangers': 30644, 'abundantly': 30645, 'qubad': 30646, 'ingratitude': 30647, 'restriction': 30648, 'hindrance': 30649, 'iwc': 30650, '31-30': 30651, 'hunts': 30652, '1,07,000': 30653, 'nahrawan': 30654, 'becher': 30655, "sa'eed": 30656, 'comptroller': 30657, 'feeble': 30658, 'compassionately': 30659, 'advertisers': 30660, 'abayi': 30661, 'kourou': 30662, 'moans': 30663, 'tonic': 30664, 'nutrient': 30665, 'helios': 30666, '2a': 30667, 'optical': 30668, 'parasol': 30669, 'halts': 30670, '95.8': 30671, 'mistrust': 30672, 'fixation': 30673, 'reminder': 30674, 'hedi': 30675, 'yousseff': 30676, 'planners': 30677, 'five-kilometer': 30678, '10-meter': 30679, '20-meter': 30680, '30-meter': 30681, '25-meter': 30682, 'fanned': 30683, 'andreu': 30684, 'hearse': 30685, 'lytvyn': 30686, 'nearly-naked': 30687, 'russian-ukrainian': 30688, 'ulrich': 30689, 'wilhelm': 30690, 'juventud': 30691, 'rebelde': 30692, 'conjwayo': 30693, 'el~tiempo': 30694, 'mauricio': 30695, 'zapata': 30696, 'seven-month': 30697, 're-stating': 30698, 'sununu': 30699, 'palestinian-americans': 30700, 'dangabari': 30701, 're-bidding': 30702, 'svalbard': 30703, 'wainganga': 30704, 'jens': 30705, 'stoltenberg': 30706, 'nobel-prize': 30707, 'environmentalist': 30708, 'wangari': 30709, 'maathai': 30710, 'capsize': 30711, 'bhandara': 30712, '925': 30713, 'woo': 30714, 'janan': 30715, 'souray': 30716, 'gloom': 30717, 'lashkargah': 30718, '1,000-year': 30719, 'tables': 30720, 'pham': 30721, 'thrive': 30722, 'painstakingly': 30723, 'anchor': 30724, 'ulsan': 30725, 'much-honored': 30726, 'mobster': 30727, 'hangout': 30728, 'satriale': 30729, 'manny': 30730, 'condominum': 30731, 'serial': 30732, 'authentication': 30733, 'demolishing': 30734, 'pals': 30735, 'episode': 30736, '130-thousand': 30737, '100-thousand': 30738, 'gref': 30739, 'german-owned': 30740, 'grief-stricken': 30741, 'pai': 30742, 'f15': 30743, 'dohuk': 30744, 'okay': 30745, 'insurers': 30746, 'caters': 30747, 'caymanians': 30748, 'preoccupied': 30749, 'sudeten': 30750, 'ruthenians': 30751, 'reich': 30752, 'truncated': 30753, 'ruthenia': 30754, 'anti-soviet': 30755, 'mid-14th': 30756, 'siam': 30757, 'chinnawat': 30758, 'anti-thaksin': 30759, 'yellow-shirts': 30760, 'shahdi': 30761, 'mohanna': 30762, 'wetchachiwa': 30763, 'red-shirts': 30764, 'confiscating': 30765, 'arson-related': 30766, 'cleavages': 30767, 'politic': 30768, 'malay-muslim': 30769, 'marginal': 30770, 'underinvestment': 30771, 'highly-developed': 30772, 'wood-processing': 30773, 'vitality': 30774, 'dombrovskis': 30775, 'meekness': 30776, 'gentleness': 30777, 'temper': 30778, 'altogether': 30779, 'boldness': 30780, 'bridle': 30781, 'dread': 30782, 'break-ins': 30783, 'valynkin': 30784, 'mainichi': 30785, 'nancy-amelia': 30786, 'de-radicalization': 30787, 'renting': 30788, 'lehava': 30789, 'rentals': 30790, 'lifestyles': 30791, 'su-24': 30792, '1.18': 30793, 'oil-drilling': 30794, 'gulf-area': 30795, 'muttur': 30796, 'hanssen-bauer': 30797, 'indescribable': 30798, 'horrors': 30799, 'miracles': 30800, 'off-duty': 30801, 'military-owned': 30802, 'askari': 30803, 'e2-k': 30804, 'pingtung': 30805, 'flattening': 30806, 'shanties': 30807, 'browne': 30808, 'inductee': 30809, 'masser': 30810, 'burgie': 30811, 'bobby': 30812, 'teddy': 30813, 'randazzo': 30814, 'dylan': 30815, 'catalog': 30816, 'induction': 30817, 'heavily-armed': 30818, 'vindicates': 30819, 'septum': 30820, 'cerebral': 30821, 'piling': 30822, 'orenburg': 30823, '2017': 30824, 'polo': 30825, 'rugby': 30826, 'roller': 30827, 'equestrian': 30828, 'equine': 30829, 'hallams': 30830, 'muammar': 30831, 'divas': 30832, 'nina-maria': 30833, 'potts': 30834, 're-set': 30835, 'rapid-reaction': 30836, 'angelina': 30837, 'naankuse': 30838, 'jolie-pitt': 30839, 'dedication': 30840, 'baboons': 30841, 'leopards': 30842, 'dalia': 30843, 'itzik': 30844, 'seasonally': 30845, 'tropics': 30846, 'semi-arid': 30847, 'rain-fed': 30848, 'climates': 30849, 'modifying': 30850, 'acidity': 30851, 'salinity': 30852, 'second-ranked': 30853, 'round-robin': 30854, 'federer-roddick': 30855, 'idaho': 30856, 'taepodong-2': 30857, 'geoscience': 30858, 'spewing': 30859, '12-kilometer-high': 30860, 'plume': 30861, 'gulfport': 30862, 'on-the-ground': 30863, 'sancha': 30864, 'waterworks': 30865, 'changqi': 30866, 'turkish-dominated': 30867, 'jubilant': 30868, 'neve': 30869, 'dekalim': 30870, 'commissioning': 30871, 'gaza-egyptian': 30872, 'state-engineered': 30873, 'mo': 30874, 'penetrating': 30875, 'multi-layered': 30876, 'hospice': 30877, 'radiotherapy': 30878, 'shelve': 30879, 'iskan': 30880, 'harithiya': 30881, 'brazen': 30882, 'nourollah': 30883, 'niaraki': 30884, 'dousing': 30885, 'fuselage': 30886, 'tupolev': 30887, 'airtour': 30888, 'logjams': 30889, 'kilogram': 30890, 'ballenas': 30891, 'tenths': 30892, '115.2': 30893, 'high-powered': 30894, 'defrauding': 30895, 'plea-bargain': 30896, 'pro-tibetan': 30897, 'tibetan-american': 30898, 'psychologically': 30899, 'unfurling': 30900, 'ayham': 30901, 'mujaheddin': 30902, 'distinction': 30903, 'radhika': 30904, 'coomaraswamy': 30905, 'cormac': 30906, "murphy-o'connor": 30907, 'delmas': 30908, 'challengerspace': 30909, 'donovan': 30910, 'al-watan': 30911, 'blackburn': 30912, 'basically': 30913, 'sneaking': 30914, 'burqa-clad': 30915, 'yoshinori': 30916, 'ono': 30917, '2,60,000': 30918, 'yvelines': 30919, 'hauts-de-seine': 30920, 'seine-saint-denis': 30921, "d'": 30922, 'oise': 30923, '1,700-year-old': 30924, 'zheijiang': 30925, 'inscriptions': 30926, '256-ad': 30927, 'best-preserved': 30928, 'phoenixes': 30929, 'etched': 30930, 'porcelain': 30931, 'pop-rock': 30932, 'derek': 30933, 'whibley': 30934, 'sum-41': 30935, 'bedrooms': 30936, 'three-story': 30937, 'sauna': 30938, '132-seat': 30939, 'disarms': 30940, 'noranit': 30941, 'setabutr': 30942, 'amass': 30943, 'sigou': 30944, 'solidaria': 30945, 'salvadorans': 30946, 'vulgar': 30947, '18-day': 30948, 'gniljane': 30949, 'selim': 30950, 'transocean': 30951, 'trident': 30952, 'channeling': 30953, 'muslim-run': 30954, 'indus': 30955, 'fused': 30956, 'indo-aryan': 30957, 'scythians': 30958, 'satisfactorily': 30959, 'marginalization': 30960, 'rocky': 30961, 'brightens': 30962, 'microchips': 30963, '15-20': 30964, 'idiot': 30965, 'drunkard': 30966, 'clinging': 30967, 'us-central': 30968, 'american-dominican': 30969, 'chinchilla': 30970, 'chinggis': 30971, 'khaan': 30972, 'eurasian': 30973, 'steppe': 30974, 'mongolians': 30975, 'waemu': 30976, 'non-disbursing': 30977, 'bakers': 30978, 'concealment': 30979, 'nibble': 30980, 'tendrils': 30981, 'rustling': 30982, 'arrow': 30983, 'maltreated': 30984, 'partridges': 30985, 'recompense': 30986, 'scruple': 30987, 'pondered': 30988, 'matched': 30989, 'seventh-century': 30990, '425': 30991, 'touches': 30992, 'closings': 30993, 'nasab': 30994, 'haqooq-i-zan': 30995, 'vujadin': 30996, 'popovic': 30997, 'milutinovic': 30998, 'sainovic': 30999, 'dragoljub': 31000, 'ojdanic': 31001, 'moses': 31002, 'lottery': 31003, 'refinances': 31004, 'cyanide-based': 31005, 'cashed': 31006, 'tavis': 31007, 'smiley': 31008, 'imagine': 31009, 'evading': 31010, 'conscription': 31011, 'asmara': 31012, 'qadir': 31013, 'sinak': 31014, 'al-muadham': 31015, 'brillantes': 31016, 'kilju': 31017, 'long-haul': 31018, 'medium-haul': 31019, 'cabin': 31020, 'jean-cyril': 31021, 'spinetta': 31022, 'cricketer-turned-politician': 31023, 'mauro': 31024, 'vechchio': 31025, 'sharks': 31026, 'stalking': 31027, 'crittercam': 31028, 'saola': 31029, 'hachijo': 31030, 'habibur': 31031, 'gaddafi': 31032, 'thilan': 31033, 'samaraweera': 31034, 'tharanga': 31035, 'paranavithana': 31036, 'misleading': 31037, 'devise': 31038, 'sisli': 31039, 'obeidi': 31040, 'compounding': 31041, 'agence': 31042, 'presse': 31043, 'riskiest': 31044, 'half-a-day': 31045, 'womb': 31046, 'veterinarians': 31047, 'farhatullah': 31048, 'kypriano': 31049, 'kristin': 31050, 'halvorsen': 31051, 'fund-global': 31052, 'crave': 31053, 'leong': 31054, 'lazy': 31055, 'mandera': 31056, 'prep': 31057, 'skelleftea': 31058, 'carli': 31059, 'dribbled': 31060, 'left-footed': 31061, 'ricocheted': 31062, 'pia': 31063, 'sundhage': 31064, 'cups': 31065, 'severity': 31066, 'kazemeini': 31067, 'disapora': 31068, 'speeds': 31069, 'parallels': 31070, 'lagunas': 31071, 'chacahua': 31072, 'cabo': 31073, 'corrientes': 31074, 'guatemalans': 31075, 'tabasco': 31076, 'mengal': 31077, 'salem': 31078, 'respectfully': 31079, '355': 31080, 'overtake': 31081, 'latvians': 31082, 'erase': 31083, '16-goal': 31084, 'clinch': 31085, 'kyat': 31086, 'u.n.-appointed': 31087, 'bomb-laden': 31088, 'mine-clearing': 31089, 'de-miners': 31090, 'ronco': 31091, 'doled': 31092, 'highness': 31093, 'brussels-based': 31094, 'year-on-year': 31095, 'royals': 31096, 'front-line': 31097, 'scimitar': 31098, 'graduating': 31099, 'sandhurst': 31100, 'accompany': 31101, 'isna': 31102, 'nie': 31103, 'frailty': 31104, 'polish-born': 31105, 'decriminalizing': 31106, 'sweltering': 31107, '37-degree': 31108, 'tursunov': 31109, 'edgardo': 31110, 'massa': 31111, 'gremelmayr': 31112, 'bjorn': 31113, 'phau': 31114, 'teesta': 31115, 'river-linking': 31116, 'ecology': 31117, 'ganges': 31118, 'damadola': 31119, 'scold': 31120, 'cuzco': 31121, 'reckon': 31122, 'incwala': 31123, 'swazi': 31124, 'nhlanhla': 31125, 'nhlabatsi': 31126, 'mbabane': 31127, 'hms': 31128, 'seawater': 31129, 'yunus': 31130, 'newly-formed': 31131, 'worst-ever': 31132, 'tadjoura': 31133, 'loudspeakers': 31134, 'committee-chairman': 31135, '1975-to-1990': 31136, 'ophir': 31137, 'pines': 31138, 'relinquish': 31139, 'rosemond': 31140, 'pradel': 31141, 'indiscretion': 31142, 'shoplifting': 31143, 'husseindoust': 31144, 'destablize': 31145, 'one-stop': 31146, 'streamlined': 31147, 'weiner': 31148, 'pro-israel': 31149, 'lobbyists': 31150, 'conservative-dominated': 31151, 'semi-retirement': 31152, 'tenzin': 31153, 'taklha': 31154, 'attribute': 31155, 'vials': 31156, 'profiting': 31157, 'ciampino': 31158, 'martha': 31159, 'goldenberg': 31160, 'kendrick': 31161, 'middle-class': 31162, 'cong.': 31163, 'superhighway': 31164, 'shady': 31165, 'loitering': 31166, 'boyle': 31167, '1860s': 31168, 'sughd': 31169, 'deluge': 31170, 'prison-break': 31171, 'khujand': 31172, 'rasht': 31173, 'oil-': 31174, 'gas-producing': 31175, 'sedimentary': 31176, 'basins': 31177, 'garments': 31178, 'overflowing': 31179, '0': 31180, 'damsel': 31181, 'bride': 31182, 'reclining': 31183, 'forgetting': 31184, 'nurture': 31185, 'ox-stall': 31186, 'curious': 31187, 'sandbags': 31188, 'learnt': 31189, 'escapes': 31190, 'chevy': 31191, 'gallegos': 31192, '20-thousand': 31193, '55.65': 31194, 'sheldon': 31195, '232': 31196, 'honesty': 31197, '728': 31198, 'arnoldo': 31199, 'aleman': 31200, 'discredited': 31201, 'sandanista': 31202, '54th': 31203, 'dikes': 31204, 'island-wide': 31205, 'vicinity': 31206, 'flashpoints': 31207, 'osnabrueck': 31208, 'processor': 31209, 'westland': 31210, 'hallmark': 31211, 'usda': 31212, 'vani': 31213, 'contreras': 31214, 'pre-marked': 31215, 'averting': 31216, 'jeopardizes': 31217, 'fresh-cut': 31218, 'anti-abortion': 31219, '34th': 31220, 'socially-conservative': 31221, 'hatim': 31222, '1657': 31223, 'basset': 31224, 'al-megrahi': 31225, 'mans': 31226, 'saydiyah': 31227, 'neighbourhood': 31228, 'south-west': 31229, 'street-to-street': 31230, '19-month': 31231, 'lightened': 31232, 'lateral': 31233, 'limped': 31234, '2100': 31235, 'edt': 31236, 'doubly': 31237, '2,75,000': 31238, 'aoun': 31239, 'sulieman': 31240, 'ivana': 31241, 'lisjak': 31242, 'gaz': 31243, 'byes': 31244, 'virginie': 31245, 'kveta': 31246, 'peschke': 31247, '1,31,000': 31248, 'tsvetana': 31249, 'pironkova': 31250, '11-nation': 31251, 'rae': 31252, 'bareli': 31253, '70th': 31254, 'italian-born': 31255, 'rajiv': 31256, 'china-north': 31257, 'safeguarding': 31258, '27,000': 31259, 'refurbished': 31260, 're-planted': 31261, 'bike': 31262, 'walkways': 31263, 'spring-break': 31264, 'sorted': 31265, 'crawl': 31266, 'unidentifiied': 31267, 'gruesome': 31268, "'n": 31269, 'large-caliber': 31270, 'chairperson': 31271, 'jaoko': 31272, 'recounting': 31273, 'episodes': 31274, 'us-based': 31275, 'pheasants': 31276, 'townships': 31277, 'reinvigorate': 31278, 'five-decade': 31279, 'doi': 31280, 'talaeng': 31281, 'burma-thailand': 31282, 'quadriplegic': 31283, 'clips': 31284, 'never-before-seen': 31285, 'streaming': 31286, 'outward': 31287, 'sunspots': 31288, 'high-definition': 31289, 'lra..': 31290, '489': 31291, 'forgotten': 31292, 'jaber': 31293, 'wafa': 31294, 'month-old': 31295, 'saboor': 31296, 'al-hadhar': 31297, 'ceyhan': 31298, 'leatherback': 31299, 'fascinate': 31300, 'roams': 31301, 'nesting': 31302, 'leatherbacks': 31303, 'tubigan': 31304, 'joko': 31305, 'suyono': 31306, 'intensifies': 31307, 'apostates': 31308, 'qais': 31309, 'shameri': 31310, 'sabotages': 31311, 'rekindle': 31312, 'lawfully': 31313, 'pre-empt': 31314, 'seche': 31315, 'al-otari': 31316, 'unleash': 31317, 'miscalculated': 31318, 'zap': 31319, 'ararat': 31320, 'istanbul-based': 31321, 'mazlum-der': 31322, 'cuban-born': 31323, 'double-standards': 31324, 'seventy-three': 31325, 'mattei': 31326, 'al-nuimei': 31327, 'half-century': 31328, 'listened': 31329, 'republican-proposed': 31330, '\x97': 31331, 'monuc': 31332, 'anyplace': 31333, 'catapulted': 31334, 'showcased': 31335, 'sino-tibetan': 31336, 'madhoun': 31337, 'kara': 31338, '13,100': 31339, '70-80': 31340, 'leveling': 31341, 'kumtor': 31342, 'cis': 31343, 'mid-1995': 31344, 'crisis-related': 31345, 'recapitalization': 31346, 'cycles': 31347, 'shackleton': 31348, 'ill-fated': 31349, 'aviary': 31350, 'wont': 31351, 'peacock': 31352, 'sharm-el-sheik': 31353, 'opium-traffickers': 31354, 'bajura': 31355, 'accham': 31356, 'kohistani': 31357, 'ul-haq': 31358, 'ahadi': 31359, 'locusts': 31360, 'zinder': 31361, 'castes': 31362, 'fortis': 31363, 'sudan-chad': 31364, 'stella': 31365, 'artois': 31366, 'radek': 31367, 'johansson': 31368, 'tulkarm': 31369, 'maclang': 31370, 'batad': 31371, 'banaue': 31372, 'ifugao': 31373, 'legaspi': 31374, 'starmagazine': 31375, 'distorts': 31376, 'al-muasher': 31377, 'non-jordanians': 31378, 'safehaven': 31379, 'wachira': 31380, 'waruru': 31381, '73rd': 31382, 'holodomor': 31383, 'anti-chinese': 31384, 'morale-boosting': 31385, 'limited-overs': 31386, 'kiwis': 31387, 'scorers': 31388, '203-4': 31389, '42.1': 31390, '201': 31391, '46.3': 31392, 'ashraful': 31393, 'napier': 31394, 'detective': 31395, 'colt': 31396, 'cobra': 31397, 'revolver': 31398, 'definitively': 31399, 'holster': 31400, 'semi-finals': 31401, 'aussie': 31402, 'h5-type': 31403, 'flyway': 31404, 'yuanshi': 31405, 'xianliang': 31406, 'earth-moving': 31407, 'sprees': 31408, 'gedo': 31409, 'bakili': 31410, 'muluzi': 31411, '365-5': 31412, 'yuvraj': 31413, 'career-best': 31414, 'araft': 31415, 'goodbye': 31416, 'saca': 31417, 'shaowu': 31418, 'second-ranking': 31419, 'chinese-sanctioned': 31420, 'chairmen': 31421, 'mumps': 31422, 'rubella': 31423, 'autism': 31424, 'upali': 31425, 'tamil-majority': 31426, 'ealier': 31427, 'affiliations': 31428, 'hypocrisy': 31429, 'snow-shortened': 31430, 'gardena': 31431, '27.99': 31432, '0.02': 31433, '1.28.01': 31434, 'guay': 31435, '1.28.19': 31436, 'super-giant': 31437, '442': 31438, '420': 31439, 'aksel': 31440, 'svindal': 31441, '417': 31442, 'configured': 31443, 'nahum': 31444, 'repeating': 31445, 'pera': 31446, 'petrasevic': 31447, 'vukov': 31448, 'csatia': 31449, 'metullah': 31450, 'teeple': 31451, 'unspent': 31452, 'helmund': 31453, 'hushiar': 31454, 'facilitates': 31455, '17-jan': 31456, 'loveland': 31457, 'meat-cutting': 31458, 'butchers': 31459, 'unionize': 31460, 'acetylene': 31461, 'basement': 31462, 'monsoons': 31463, 'bong-jo': 31464, '26,500': 31465, '1,850': 31466, 'saqlawiyah': 31467, 'nampo': 31468, 'permanent-status': 31469, 'samad': 31470, 'kojak': 31471, 'sulaiman': 31472, 'pashtoonkhwa': 31473, 'milli': 31474, 'cholily': 31475, '221': 31476, 'chakul': 31477, 'consent': 31478, 'abune': 31479, 'squares': 31480, 'sicko': 31481, 'humaneness': 31482, 'harshest': 31483, 'wealthier': 31484, 'heeds': 31485, 'https://www.celebritiesforcharity.org/raffles/netraffle_main.cfm': 31486, 'round-trip': 31487, 'airfare': 31488, 'china-usa': 31489, 'costumed': 31490, 'urchins': 31491, 'porch': 31492, 'trick-or-treaters': 31493, 'gravelly': 31494, 'beggars': 31495, 'decorating': 31496, 'upshot': 31497, 'dusk': 31498, 'ghouls': 31499, 'make-believe': 31500, 'starlets': 31501, 'distasteful': 31502, 'abd-al-nisr': 31503, 'khantumani': 31504, '178': 31505, 'moravia': 31506, 'magyarization': 31507, 'soviet-dominated': 31508, 'issas': 31509, 'gouled': 31510, 'aptidon': 31511, 'issa-dominated': 31512, 'guelleh': 31513, 'socializing': 31514, 'terrified': 31515, 'slew': 31516, 'pounce': 31517, 'remedy': 31518, 'doosnoswair': 31519, 'oliveira': 31520, '567': 31521, 'onion': 31522, 'chili': 31523, 'eggplant': 31524, 'smoked': 31525, 'trout': 31526, 'spaghetti': 31527, 'parmesan': 31528, 'spinach': 31529, 'avocado': 31530, 'custodian': 31531, 'long-handled': 31532, 'squeegee': 31533, 'educators': 31534, 'believer': 31535, 'séances': 31536, 'merriment': 31537, 'fooling': 31538, 'gullible': 31539, 'poke': 31540, 'mansions': 31541, 'scenic': 31542, '99.29': 31543, '314': 31544, 'i.a.e.a': 31545, 'long-denied': 31546, 'amharic-language': 31547, 'stinging': 31548, 'misled': 31549, 'rotfeld': 31550, 'surrenders': 31551, 'onal': 31552, 'ulus': 31553, 'cabello': 31554, 'youssifiyeh': 31555, 'kamdesh': 31556, 'half-english': 31557, 'abbreviations': 31558, 'translations': 31559, 'transplanting': 31560, 'casualities': 31561, 'rawhi': 31562, 'fattouh': 31563, 'corners': 31564, '407': 31565, 'francs': 31566, 'islamist-controlled': 31567, 'jokic': 31568, 'murghab': 31569, 'conductor': 31570, 'mstislav': 31571, 'moderating': 31572, '39th': 31573, 'adrien': 31574, 'houngbedji': 31575, 'vishnevskaya': 31576, '72-year-old': 31577, 'laughable': 31578, 'well-planned': 31579, '90-year-old': 31580, '19,000': 31581, 'laiyan': 31582, 'jiaotong': 31583, 'tp': 31584, 'anti-constitution': 31585, 'rovnag': 31586, 'abdullayev': 31587, '200-kilometer': 31588, 'novo-filya': 31589, 'implications': 31590, 'eu-backed': 31591, 'nabucco': 31592, 'bypassing': 31593, 'colchester': 31594, 'shabat': 31595, 'al-basra': 31596, 'jabbar': 31597, 'jeremy': 31598, 'hobbs': 31599, 'khasavyurt': 31600, 'longyan': 31601, 'mensehra': 31602, '72-39': 31603, 'dialed': 31604, 'gibb': 31605, 'guadalupe': 31606, 'escamilla': 31607, 'mota': 31608, 'rajapakshe': 31609, 'kentung': 31610, 'ui': 31611, 'leonella': 31612, 'sgorbati': 31613, 'disable': 31614, 'amerzeb': 31615, 'well-trained': 31616, 'beats': 31617, '135.6': 31618, 'web-slinging': 31619, 'irresistable': 31620, 'shrek': 31621, 'grosses': 31622, '59.9': 31623, 'bourj': 31624, 'abi': 31625, 'haidar': 31626, 'puccio': 31627, 'vince': 31628, 'spadea': 31629, 'straight-sets': 31630, 'hyung-taik': 31631, 'melli': 31632, 'proliferator': 31633, 'agartala': 31634, 'indian-pakistani': 31635, 'sébastien': 31636, 'season-opening': 31637, 'citroën': 31638, 'xsara': 31639, 'wrc': 31640, '18.46.09': 31641, 'duval': 31642, '32.7': 31643, 'grönholm': 31644, 'auriol': 31645, '1,12,000': 31646, 'land-based': 31647, 'submachine': 31648, 'ex-wife': 31649, 'estranged': 31650, 'hailie': 31651, 'remarried': 31652, 'recouped': 31653, 'eu-latin': 31654, 'continents': 31655, 'ferrero-waldner': 31656, 'tycze': 31657, 'substantiated': 31658, 'u.s.-held': 31659, 'pick-up': 31660, 'establishments': 31661, 'ventilation': 31662, 'passive': 31663, 'petropavlosk-kamchatskii': 31664, 'fedora-wearing': 31665, 'adventurer': 31666, 'spielberg': 31667, 'paramount': 31668, 'indy': 31669, 'mangos': 31670, 'avocados': 31671, '1634': 31672, 'bonaire': 31673, 'isla': 31674, 'refineria': 31675, 'lied': 31676, 'charcoal-burner': 31677, 'housekeeping': 31678, 'whiten': 31679, 'blacken': 31680, 'harnessed': 31681, 'tormented': 31682, 'yoke': 31683, 'smile': 31684, 'idleness': 31685, 'lighten': 31686, 'labors': 31687, 'hive': 31688, 'tasted': 31689, 'honeycomb': 31690, 'self-interest': 31691, 'brute': 31692, 'brayed': 31693, 'cudgelling': 31694, 'fright': 31695, 'irrationality': 31696, 'lakeside': 31697, 'vijay': 31698, 'silly': 31699, 'clapping': 31700, 'luay': 31701, 'zaid': 31702, 'sander': 31703, 'eight-month': 31704, 'maddalena': 31705, 'sardinian': 31706, 'storm-damaged': 31707, 'recently-elected': 31708, 'humberto': 31709, 'valbuena': 31710, 'caqueta': 31711, '1890': 31712, 'dorsey': 31713, 'mahalia': 31714, '227-kilogram': 31715, '13010': 31716, 'majdal': 31717, 'dutch-built': 31718, 'matara': 31719, 'nguesso': 31720, 'muhsin': 31721, 'matwali': 31722, 'atwah': 31723, 'theme-park-like': 31724, 'intoxicating': 31725, 'five-month-long': 31726, 'sunni-backed': 31727, 'maysoon': 31728, 'al-damaluji': 31729, 'cross-sectarian': 31730, 'massum': 31731, '163-seat': 31732, 'homeowner': 31733, 'micky': 31734, 'rosenfeld': 31735, 'talansky': 31736, 'congratulations': 31737, 'barka': 31738, 'e.': 31739, 'kousalyan': 31740, 'shipyards': 31741, 'disillusionment': 31742, 'pictured': 31743, 'kira': 31744, 'plastinina': 31745, 'mikhailova': 31746, 'emma': 31747, 'half-sister': 31748, 'sarrata': 31749, 'tractor-trailer': 31750, 'chronicle': 31751, 'jamahiriyah': 31752, 'pariah': 31753, 'walled': 31754, 'southern-based': 31755, 'sevilla': 31756, 'maresca': 31757, '78th': 31758, '84th': 31759, 'fabiano': 31760, 'frédéric': 31761, 'kanouté': 31762, 'anglo-spanish': 31763, 'moscow-led': 31764, 'partially-collapsed': 31765, 'collemaggio': 31766, 'tarsyuk': 31767, 'adriatic': 31768, 'slovenians': 31769, 'dagger': 31770, 'stashes': 31771, 'unleashing': 31772, 'atilla': 31773, 'onna': 31774, 'shkin': 31775, 'abruzzo': 31776, 'cruelty': 31777, 'low-ranking': 31778, 'health-care': 31779, 'one-china': 31780, 'book-signing': 31781, 'amicable': 31782, 'hard-core': 31783, 'remnant': 31784, 'fades': 31785, 'animist': 31786, 'gil': 31787, 'hoffman': 31788, '8,88,000': 31789, 'ex-chancellor': 31790, 'helmut': 31791, 'reunifying': 31792, 'swearing': 31793, '238': 31794, 'member-nations': 31795, 'european-iranian': 31796, 'verbeke': 31797, 'intercepts': 31798, 'symbolize': 31799, 'ishaq': 31800, 'alako': 31801, 'rahmatullah': 31802, 'nazari': 31803, '68.4': 31804, 'robinsons': 31805, 'quentin': 31806, 'tarantino': 31807, 'grindhouse': 31808, 'much-publicized': 31809, 'debuted': 31810, '11.6': 31811, 'tracker': 31812, 'dergarabedian': 31813, 'takhar': 31814, 'icelanders': 31815, 'bootleg': 31816, 'reykjavik': 31817, 'marginalize': 31818}
    31819
    

인덱스 0에는 패딩을 위한 'PAD'를 집어놓고, 인덱스 1에는 모르는 단어를 의미하는 'OOV'를 집어넣습니다. 그 이후에는 빈도수 순대로 인덱스를 부여합니다. 여기서는 빈도수 1인 단어를 별도로 제거하지 않았는데, 만약 제거 한다면 뒤에서 각 문장에 대해서 정수 인코딩을 진행할 때 단어 집합이 모르는 단어가 발생하므로 그런 상황에서 'OOV'로 판단하여 인덱스 1을 사용할 수 있습니다.

이제 태깅 정보에도 인덱스를 부여해보겠습니다.


```python
tag_to_index={'PAD' : 0}
i=0
for tag in tag_set:
    i=i+1
    tag_to_index[tag]=i
print(tag_to_index)
```

    {'PAD': 0, 'B-gpe': 1, 'B-org': 2, 'I-geo': 3, 'I-per': 4, 'B-nat': 5, 'I-eve': 6, 'B-per': 7, 'B-geo': 8, 'I-art': 9, 'B-eve': 10, 'O': 11, 'I-org': 12, 'B-tim': 13, 'I-nat': 14, 'I-gpe': 15, 'B-art': 16, 'I-tim': 17}
    

이제 임의의 단어와 태깅 정보에 대해서 인덱스가 출력되는지 시험해봅시다.


```python
print(word_to_index["walled"])
print(tag_to_index["B-geo"])
```

    31754
    8
    

이제 정수 인코딩을 진행해보겠습니다.


```python
data_X = []

for s in sentences: # 전체 데이터에서 하나의 데이터. 즉, 하나의 문장씩 불러옵니다.
    temp_X = []
    for w in s: # 각 문장에서 각 단어를 불러옵니다.
        temp_X.append(word_to_index.get(w,1)) # 각 단어를 맵핑되는 인덱스로 변환합니다.
    data_X.append(temp_X)
```


```python
len(data_X)
```




    47959



총 샘플의 개수는 47,959으로 정수 인코딩이 되기 전과 동일합니다. 첫번째 샘플을 출력해보겠습니다.


```python
print(data_X[0])
```

    [254, 6, 967, 16, 1795, 238, 468, 7, 523, 2, 129, 5, 61, 9, 571, 2, 833, 6, 186, 90, 22, 15, 56, 3]
    

이제 데이터의 y에 해당되는 개체명 태깅 정보 또한 정수 인코딩을 진행하겠습니다.


```python
data_y = []
for s in ner_tags:
    temp_y = []
    for w in s:
            temp_y.append(tag_to_index.get(w))
    data_y.append(temp_y)
print(data_y[0]) # 첫번째 데이터 출력
```

    [11, 11, 11, 11, 11, 11, 8, 11, 11, 11, 11, 11, 8, 11, 11, 11, 11, 11, 1, 11, 11, 11, 11, 11]
    

이제 X 데이터와 y 데이터가 구성되었습니다.

이제 패딩 작업을 진행해봅시다. 앞서 확인하였듯이 대부분의 데이터의 길이는 40~60에 분포되어져 있습니다. 그러므로 가장 긴 샘플의 길이인 104가 아니라 70정도로 max_len을 정해보겠습니다.


```python
max_len=70
from keras.preprocessing.sequence import pad_sequences
pad_X = pad_sequences(data_X, padding='post', maxlen=max_len)
# data_X의 모든 샘플의 길이를 패딩할 때 뒤의 공간에 숫자 0으로 채움.
pad_y = pad_sequences(data_y, padding='post', value=tag_to_index['PAD'], maxlen=max_len)
# data_y의 모든 샘플의 길이를 패딩할 때 'PAD'에 해당되는 인덱스로 채움.
# 결과적으로 'PAD'의 인덱스값인 0으로 패딩된다.
```

모든 샘플의 길이가 70이 되었습니다.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(pad_X, pad_y, test_size=.1, random_state=777)
```

훈련 데이터와 테스트 데이터를 분리합니다. 모델을 설계해보겠습니다.


```python
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
```


```python
n_words = len(word_to_index)
n_labels = len(tag_to_index)
```


```python
model = Sequential()
model.add(Embedding(input_dim=n_words, output_dim=20, input_length=max_len, mask_zero=True))
model.add(Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1)))
model.add(TimeDistributed(Dense(50, activation="relu")))
crf = CRF(n_labels)
model.add(crf)
```

모델에 양방향 LSTM을 사용하고, 모델의 출력층에 CRF 층을 배치합니다.


```python
from keras.utils import np_utils
y_train2 = np_utils.to_categorical(y_train) #원-핫 인코딩
y_train2[0][0]
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0.], dtype=float32)



훈련 데이터 y_train에 대해서 원-핫 인코딩을 수행하고 y_train2에 저장합니다.


```python
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
history = model.fit(X_train, y_train2, batch_size=32, epochs=5, validation_split=0.1, verbose=1)
```

    C:\Users\joon2\Anaconda3\lib\site-packages\keras_contrib\layers\crf.py:346: UserWarning: CRF.loss_function is deprecated and it might be removed in the future. Please use losses.crf_loss instead.
      warnings.warn('CRF.loss_function is deprecated '
    C:\Users\joon2\Anaconda3\lib\site-packages\keras_contrib\layers\crf.py:353: UserWarning: CRF.accuracy is deprecated and it might be removed in the future. Please use metrics.crf_accuracy
      warnings.warn('CRF.accuracy is deprecated and it '
    

    Train on 38846 samples, validate on 4317 samples
    Epoch 1/5
    38846/38846 [==============================] - 544s 14ms/step - loss: 8.3192 - crf_viterbi_accuracy: 0.9036 - val_loss: 8.1421 - val_crf_viterbi_accuracy: 0.9383
    Epoch 2/5
    38846/38846 [==============================] - 543s 14ms/step - loss: 8.1242 - crf_viterbi_accuracy: 0.9508 - val_loss: 8.0810 - val_crf_viterbi_accuracy: 0.9552
    Epoch 3/5
    38846/38846 [==============================] - 542s 14ms/step - loss: 8.0939 - crf_viterbi_accuracy: 0.9588 - val_loss: 8.0685 - val_crf_viterbi_accuracy: 0.9580
    Epoch 4/5
    38846/38846 [==============================] - 549s 14ms/step - loss: 8.0817 - crf_viterbi_accuracy: 0.9625 - val_loss: 8.0621 - val_crf_viterbi_accuracy: 0.9595
    Epoch 5/5
    38846/38846 [==============================] - 591s 15ms/step - loss: 8.0749 - crf_viterbi_accuracy: 0.9649 - val_loss: 8.0599 - val_crf_viterbi_accuracy: 0.9593
    


```python
y_test2 = np_utils.to_categorical(y_test)
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test2)[1]))
```

    4796/4796 [==============================] - 34s 7ms/step
    
     테스트 정확도: 0.9605
    

96%의 정확도를 얻어냅니다. 뒤에서 좀 더 알맞은 방법으로 정확도를 측정해볼겁니다.


```python
index_to_word={}
for key, value in word_to_index.items():
    index_to_word[value] = key

index_to_tag={}
for key, value in tag_to_index.items():
    index_to_tag[value] = key


i=13 # 확인하고 싶은 테스트용 샘플의 인덱스.
y_predicted = model.predict(np.array([X_test[i]])) # 입력한 테스트용 샘플에 대해서 예측 y를 리턴
y_predicted = np.argmax(y_predicted, axis=-1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.
true = np.argmax(y_test2[i], -1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.

print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
print(35 * "-")

for w, t, pred in zip(X_test[i], true, y_predicted[0]):
    if w != 0: # PAD값은 제외함.
        print("{:17}: {:7} {}".format(index_to_word[w], index_to_tag[t], index_to_tag[pred]))
```

    단어             |실제값  |예측값
    -----------------------------------
    the              : O       O
    statement        : O       O
    came             : O       O
    as               : O       O
    u.n.             : B-org   B-org
    secretary-general: I-org   I-org
    kofi             : B-per   B-per
    annan            : I-per   I-per
    met              : O       O
    with             : O       O
    officials        : O       O
    in               : O       O
    amman            : B-geo   B-geo
    to               : O       O
    discuss          : O       O
    wednesday        : B-tim   B-tim
    's               : O       O
    attacks          : O       O
    .                : O       O
    


```python
epochs = range(1, len(history.history['val_loss']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
```


![png](NLP_basic_11_Tagging_Task_files/NLP_basic_11_Tagging_Task_294_0.png)


### 3) 시퀀스 레이블링 평가하기

개체명 인식에서는 그 어떤 개체도 아니라는 의미의 'O'라는 태깅이 존재합니다. 그런데 이런 정보는 보통 대다수의 레이블을 차지하기 때문에 기존에 사용했던 정확도 평가 방법을 사용하는 것이 적절하지 않을 수 있습니다.

실제로는 PER, MISC, PER, MISC, PER이라는 총 5개의 개체가 존재함에도 불구하고 예측값인 predicted는 단 1개의 개체도 맞추지 못한 상황을 시뮬레이션을 해봅시다. 개체를 하나도 맞추지 못했다는 가정하에 전부 'O'로만 채워진 예측값 predicted를 생성합니다.


```python
true=['B-PER', 'I-PER', 'O', 'O', 'B-MISC', 'O','O','O','O','O','O','O','O','O','O','B-PER','I-PER','O','O','O','O','O','O','B-MISC','I-MISC','I-MISC','O','O','O','O','O','O','B-PER','I-PER','O','O','O','O','O']
# 실제값
predicted=['O'] * len(true) #실제값의 길이만큼 전부 'O'로 채워진 리스트 생성. 예측값으로 사용.
print(predicted)
```

    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    

이제 시뮬레이션에 대한 정확도를 계산해봅시다.


```python
hit = 0 # 정답 개수
for t, p in zip(true, predicted):
    if t == p:
        hit +=1 # 정답인 경우에만 +1
accuracy = hit/len(true) # 정답 개수를 총 개수로 나눈다.
print("정확도: {:.1%}".format(accuracy))
```

    정확도: 74.4%
    

#### 문제점 (모델의 성능의 오해를 불러 일으킬 수 있다.)

실제값에서도 대부분의 값이 'O'이기 때문에 그 어떤 개체도 찾지 못하였음에도 74%의 정확도를 얻습니다. 이는 정확도가 뻥튀기되어 모델의 성능을 오해할 수 있다는 문제가 있습니다.

그래서 여기서는 위와 같은 상황에서 더 적절한 평가 방법을 도입하고자 합니다. 윈도우의 명령 프롬프트나 UNIX의 터미널에서 아래의 명령을 수행하여 파이썬 패키지 seqeval를 설치합니다.

#### 해경방법 (더 적절한 평가방법 사용, 패키지 seqeval사용)


```python
#pip install seqeval
```

평가방법으로 정밀도(precision)과 재현률(recall)를 사용한다.

$정밀도 = \frac{TP}{TP + FP} = \text{특정 개체라고 예측한 경우 중에서 실제 특정 개체로 판명되어 예측이 일치한 비율}$

$재현률 = \frac{TP}{TP + FN} = \text{전체 특정 개체 중에서 실제 특정 개체라고 정답을 맞춘 비율}$

정밀도와 재현률로부터 조화 평균(harmonic mean)을 구한 것을 f1-score라고 합니다.

$f1\ score = 2 × \frac{\text{정밀도 × 재현률}}{\text{정밀도 + 재현률}}$

predicted의 성능을 평가하기 위해서 정밀도, 재현률, f1-score를 계산해보도록 하겠습니다.


```python
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
print(classification_report(true, predicted))
```

               precision    recall  f1-score   support
    
         MISC       0.00      0.00      0.00         2
          PER       0.00      0.00      0.00         3
    
    micro avg       0.00      0.00      0.00         5
    macro avg       0.00      0.00      0.00         5
    
    

이러한 측정 방법을 사용하면 PER과 MISC 두 특정 개체 중에서 실제 predicted가 맞춘 것은 단 하나도 없는 것을 확인할 수 있습니다. 

이번에는 어느 정도는 정답을 맞추었다고 가정하고 예측값인 predicted를 수정하여 정밀도, 재현률, f1-score를 확인해봅시다.


```python
true=['B-PER', 'I-PER', 'O', 'O', 'B-MISC', 'O','O','O','O','O','O','O','O','O','O','B-PER','I-PER','O','O','O','O','O','O','B-MISC','I-MISC','I-MISC','O','O','O','O','O','O','B-PER','I-PER','O','O','O','O','O']
predicted=['B-PER', 'I-PER', 'O', 'O', 'B-MISC', 'O','O','O','O','O','O','O','O','O','O','B-PER','I-PER','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O']
print(classification_report(true, predicted))
```

               precision    recall  f1-score   support
    
         MISC       1.00      0.50      0.67         2
          PER       1.00      0.67      0.80         3
    
    micro avg       1.00      0.60      0.75         5
    macro avg       1.00      0.60      0.75         5
    
    

특정 개체로 예측한 경우에 대해서는 모두 제대로 예측을 하였으므로 정밀도는 1이 나옵니다. 하지만 재현률에서는 MISC는 실제로는 4개임에도 2개만을 맞추었으므로 0.5, PER은 실제로는 3개임에도 2개만을 맞추었으므로 0.67이 나온 것을 볼 수 있습니다.

### 4) 실제 모델에 대해서 f1 score 구하기

이제 앞서 구현한 모델에 대해서 위에서 배운 f1-score를 적용해봅시다.

모델이 리턴하는 예측값은 숫자로 구성되어져 있으므로, 이를 먼저 태깅이 나열되어 있는 리스트로 치환하는 작업이 필요합니다.

 sequences_to_tag (숫자 시퀀스로부터 태깅 정보로 치환)하는 함수를 만듭니다.


```python
def sequences_to_tag(sequences): # 예측값을 index_to_tag를 사용하여 태깅 정보로 변경하는 함수.
    result = []
    for sequence in sequences: # 전체 시퀀스로부터 시퀀스를 하나씩 꺼낸다.
        temp = []
        for pred in sequence: # 시퀀스로부터 예측값을 하나씩 꺼낸다.
            pred_index = np.argmax(pred) # 예를 들어 [0, 0, 1, 0 ,0]라면 1의 인덱스인 2를 리턴한다.
            temp.append(index_to_tag[pred_index].replace("PAD", "O")) # 'PAD'는 'O'로 변경
        result.append(temp)
    return result
```


```python
y_predicted = model.predict(X_test)
pred_tags = sequences_to_tag(y_predicted)
test_tags = sequences_to_tag(y_test2)
```


```python
print(classification_report(test_tags, pred_tags))
```

               precision    recall  f1-score   support
    
          org       0.72      0.54      0.61      2037
          per       0.76      0.78      0.77      1732
          tim       0.88      0.85      0.86      2016
          geo       0.83      0.87      0.85      3832
          gpe       0.97      0.94      0.96      1600
          art       0.00      0.00      0.00        32
          eve       0.00      0.00      0.00        28
          nat       0.00      0.00      0.00        17
    
    micro avg       0.83      0.80      0.81     11294
    macro avg       0.82      0.80      0.81     11294
    
    


```python
print("F1-score: {:.1%}".format(f1_score(test_tags, pred_tags)))
```

    F1-score: 81.4%
    

### 5) 새로운 입력에 대해서 예측하기

이제 임의로 만든 새로운 문장에 대해서 앞서 만든 개체명 인식 모델을 수행해보겠습니다.


```python
test_sentence='Hong Gildong starts over with the Justice League of Joseon at the country of Yul'.split()
```


```python
new_X=[]
for w in test_sentence:
    try:
      new_X.append(word_to_index.get(w,1))
    except KeyError:
      new_X.append(word_to_index['OOV'])
      # 모델이 모르는 단어에 대해서는 'OOV'의 인덱스인 1로 인코딩

print(new_X)
```

    [1, 1, 4027, 80, 17, 2, 1, 1, 6, 1, 20, 2, 56, 6, 1]
    


```python
pad_new = pad_sequences([new_X], padding="post", value=0, maxlen=max_len)
```


```python
p = model.predict(np.array([pad_new[0]]))
p = np.argmax(p, axis=-1)
print("{:15}||{}".format("Word", "Prediction"))
print(30 * "=")
for w, pred in zip(test_sentence, p[0]):
    print("{:15}: {:5}".format(w, index_to_tag[pred]))
```

    Word           ||Prediction
    ==============================
    Hong           : B-org
    Gildong        : I-org
    starts         : O    
    over           : O    
    with           : O    
    the            : O    
    Justice        : B-org
    League         : I-org
    of             : I-org
    Joseon         : I-org
    at             : O    
    the            : O    
    country        : O    
    of             : O    
    Yul            : B-geo
    

CRF 사용 참고 :<br/>
https://www.depends-on-the-definition.com/sequence-tagging-lstm-crf/<br/>
https://medium.com/@rohit.sharma_7010/a-complete-tutorial-for-named-entity-recognition-and-extraction-in-natural-language-processing-71322b6fb090<br/>
http://blog.shurain.net/2013/04/crf.html<br/>
http://www.stokastik.in/bilstm-crf-sequence-tagging-for-e-commerce-attribute-extraction/<br/>


CRF 이론 참고 :<br/>
https://createmomo.github.io/2017/09/12/CRF_Layer_on_the_Top_of_BiLSTM_1/<br/>
