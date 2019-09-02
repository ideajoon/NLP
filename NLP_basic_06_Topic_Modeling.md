
written by ideajoon<br/>
※ 참고 : 딥 러닝을 이용한 자연어 처리 입문 (https://wikidocs.net/book/2155) 자료를 공부하고 정리함

# 06. 토픽 모델링(Topic Modeling)

토픽 모델링(Topic Modeling)이란 기계 학습 및 자연어 처리 분야에서 토픽이라는 문서 집합의 추상적인 주제를 발견하기 위한 통계적 모델 중 하나로, 텍스트 본문의 숨겨진 의미 구조를 발견하기 위해 사용되는 텍스트 마이닝 기법입니다.

BoW에 기반한 DTM이나 TF-IDF는 기본적으로 단어의 빈도 수를 이용한 수치화 방법이기 때문에 단어의 의미를 고려하지 못한다는 단점이 있었습니다. (이를 토픽 모델링 관점에서는 단어의 토픽을 고려하지 못한다고도 합니다.)

## 목차
1. 잠재 의미 분석(Latent Semantic Analysis, LSA)
2. 잠재 디리클레 할당(Latent Dirichlet Allocation, LDA)
3. 잠재 디리클레 할당(LDA) 실습2

## 1. 잠재 의미 분석(Latent Semantic Analysis, LSA)

LSA는 정확히는 토픽 모델링을 위해 최적화 된 알고리즘은 아니지만, 토픽 모델링이라는 분야에 아이디어를 제공한 알고리즘이라고 볼 수 있습니다.

뒤에서 배우게 되는 LDA는 LSA의 단점을 개선해가며 탄생한 알고리즘으로 토픽 모델링에 보다 적합한 알고리즘입니다.

DTM의 잠재된(Latent) 의미를 이끌어내는 방법으로 잠재 의미 분석(Latent Semantic Analysis, LSA)이라는 방법이 있습니다. 잠재 의미 분석(Latent Semantic Indexing, LSI)이라고 부르기도 합니다.

### 1) 특이값 분해(Singular Value Decomposition, SVD)

특이값 분해(Singular Value Decomposition, SVD)는 실수 벡터 공간에 한정하여 내용을 설명함을 명시합니다. SVD란 A가 m × n 행렬일 때, 다음과 같이 3개의 행렬의 곱으로 분해(decomposition)하는 것을 말합니다.

$A=UΣV^\text{T}$

여기서 각 3개의 행렬은 다음과 같은 조건을 만족합니다.

$U: m × m\ \text{직교행렬}\ (AA^\text{T}=U(ΣΣ^\text{T})U^\text{T})$<br/>
$V: n × n\ \text{직교행렬}\ (A^\text{T}A=V(Σ^\text{T}Σ)V^\text{T})$<br/>
$Σ: m × n\ \text{직사각 대각행렬}$

* 직교행렬(orthogonal matrix)이란 자신과 자신의 전치 행렬(transposed matrix)의 곱 또는 이를 반대로 곱한 결과가 단위행렬(identity matrix)이 되는 행렬을 말합니다.
* 대각행렬(diagonal matrix)이란 주대각선을 제외한 곳의 원소가 모두 0인 행렬을 의미합니다.

이때 SVD로 나온 대각 행렬의 대각 원소의 값을 행렬 A의 특이값(singular value)라고 합니다.

#### (1) 전치 행렬(Transposed Matrix)

전치 행렬(transposed matrix)은 원래의 행렬에서 행과 열을 바꾼 행렬입니다. 즉, 주대각선을 축으로 반사 대칭을 하여 얻는 행렬입니다.

![](https://lh3.googleusercontent.com/G7Cp9VWMpLUX_hb-QsP4Faz61Pjo_XO4su4Z1s2ZNBAmSCJ4PKOuJXCujGVkOBRz51U5CcNRwc-Z3gJTo9fLD0Fs8fn8Q11Tn7Hao3dwf_WkIDwHIJZqTt6gGLsjMUNN9IlGJxkxcOm9Met8ylH9yAdiq5ifxVmJkbl4V0RkzxR2urPe4GQRs68xkUPhUwWxsnEbtnlAXfUEewSX_iAzQRkgXv4iis9GE4pqfC4K5dvKwUxYInY2r7tx5CLjsSNhHwpm5T3m0yHB2l9Shv0aPPCwpjPJNnd8Y_rkBcKP4gB9BFh2nh4SuB8Ho5gV7m9h9XM3E-2KQjGQ0i6nRpJtNiulIlJ-zIRWJgqlud2YXD4Z7CRP5Ai4rfe_6AsrLremyW84PzauLTEQwZZsDvZSi9yYfsl3Xl2lsbIuaCW6ITy7Em3JQIx2fuK9Vp64iPyXpszMi0wo0KxV-oydehF9EC3yVTfRO_UEcNkHker3ys28_Omu3RGQT07f2HZGFD1Yz9lgiLcMX8saLehdg3_F7Jl6nQqNIaCT4-JQb7pP5FfHFwbBlMMA_k-AZRz9JpxtgmDIV4PLMzKjXXDLMPrPT0rdhDSwzJz-Aa1H8NlPmyImwCJMB0EA4JiKOOxYv4ExyJ78suR1ZYnLi3HB6ZQb4UL8pzT-lYU=w348-h117-no)

#### (2) 단위 행렬(Identity Matrix)

단위 행렬(identity matrix)은 주대각선의 원소가 모두 1이며 나머지 원소는 모두 0인 정사각 행렬을 말합니다.

![](https://lh3.googleusercontent.com/nWDvpgjUOxnQDou1_pZw2TNrjC6kexqKHe9NzSqU6TSKUWxWCZxbvk0ajRpQmEFIwo2OeJ-eLZDvEDJw4_2HZkHsy9fILDXPzxn7wO0pvBEWV82N0TRfP-Cf4iGJFXpE3l5PRMKGPzTCNVdCzl56ph-5-S1makrStnxlYqihJMOf3gagYEnbNZcdRVHbDvacYy_5sLndPuC3MSXaGoJo3ubktR5Mr4AhPN-47hlyCJC4EaxzuEQsO2JK6XMimcwHCUlx1lV-cul7DjUGJxoXIuwFlvZN9tMc4zFSROvEDgeuVM0KaJuothNw7YU4HpOTAycVz-T7yGEfrjOzBeeuZ8PP5bSskdfPLeDkolZYSKET1uBuaOPjzPPqjvkPWLaLkVHA7jjpPRASWA8TOrxf7TSjXhtkzwXIN6A_4rl2xR6t8qVnL6vB0JPvxS_w2LEuuJXatp0wTpKgcMpKM5SUA2qJQtyS-4Xb8_R4xeCpNeR0-EHuaFMWXys7bdMbFh2x2H57nJXAHPyA3KjSRO9CCKlxWHpU4Tk9UN9VK-HZX3NqPqODX9t9iM4SwnhGrupnhVIRSxulN23Z_51fLW4nW6ckdpsSmhHVaV483mJGtXXjM3WbZNyCkZXUAeBbLR7B2CweglYCeCALG1FKRuFHMd4M7Vjzexg=w348-h117-no)

#### (3) 역행렬(Inverse Matrix)

행렬 A와 어떤 행렬을 곱했을 때, 결과로서 단위 행렬이 나온다면 이때의 어떤 행렬을 A의 역행렬이라고 하며, $A^{-1}$라고 표현합니다.

![](https://lh3.googleusercontent.com/PxagU0aXcnHvHA_I95SrAqzdm1SlbThhrQ_Gy55vEcMLxudKPtcowcOr2vagmR7GlMbF13X_U8flfY9VVE-DTn7FK3BPdxCKnKozFJGvr5rrDcsRaxXjmH4FDPm-hmsme6Q7OXmlHArDeGjRWzYXYQ0ZpYzSBA5IJjXaS_LZgNLmX0usgK4r-OoldZGhEVwgcOhh4olbkl5Fl4Gmuxr2Xj5CElFCmhytJdA2J0IoCeKAo16dkWgdMo-rcF6RFNHx7qEKW6wIK4QSTMYxVHPxctOan_ZEbNnOQRy98hXbMNpmTr_3mkZQeTtz5BlbmQcfgUZToQs5cD-bn4ZEldJpCn9E2z_pTn8XCy-kQc1xywznWrM7gMJm6mD6VhRm5lsHIUCoPcfWAw63rYrqKRxx120cNERm7mVJOvliI6dMhxQs9YMta2o_B0jHB3gqqs92RSesr7ZwWBvfkkDmwK872ebq70ytMYdh-KvdX09Eyg54EvlRZoj6LDlAVRrtSho1TR2kiWqMDrZQqg5x_g9uBomOeNjaNw8ajzDV36Gt3U6EAVPTj5kgGA9g8MKbsH8f6yalKSzoTIXLeZ9GfzGbXbGHzjvix2z9GNupwJTeBMp8nKStFTvUUO4cnpvrplfRuXeeQY4GPuBLwebbN39eB2729BG7WR8=w376-h178-no)

#### (4) 직교 행렬(Orthogonal matrix)

실수 n×n행렬 A에 대해서 $A\ ×\ A^{T} = I$를 만족하면서 $A^{T}\ ×\ A = I$을 만족하는 행렬 A를 직교 행렬이라고 합니다. 그런데 역행렬의 정의를 다시 생각해보면, 결국 직교 행렬은 $A^{-1}=A^{T}$를 만족합니다.

#### (5) 대각 행렬(Diagonal matrix)

대각행렬(diagonal matrix)은 주대각선을 제외한 곳의 원소가 모두 0인 행렬을 말합니다.

![](https://lh3.googleusercontent.com/Xwab-AIipD0U2bn2DghGulcplEarx4R1KhgnKBw0zUblEqENk2kk6Z1af-S6oAugVKcUa86pv4ZoYrXleCC5CWWyuqsIJbGdZCAHPu_2AoAfYHpMmOKL9loxgqdAmxUsdhp5lhv9Tx1VQmqRerbdp7ysgSuRbRBa4oRFIY9AMFzjeZ8sveNeduNudF4YoeEHnxMyk0pisnn3KFb27xT72CrzjHZI9T9uSOelABKrd2fFqb2dGhgavQ2hvwuE-ZoOtbopDF65W9NidV6aC_Vvtoxg8w4Z_L1pKuAEeFzV8Pv6M_MMJTuwQT2wfgrODxUxhv5bCjomchXoVOwrb1a4RgrriFT2v3KO-pqeff6iebGY7gad-a0E0JzDIRqwUzfxHrcd6pkU3SiyyioLr4GvBn9oG-G7XzBmD3TDCErRC9C7560XXP8LbdzFAzoO3UFCitQ5NtaqVuW0VesyNfyuT-HVb84LghO3iSW326JvHRyR4la13aa6pDq2sThg8drYvhXzUiJngUxOhU81--xJgrOj1awJ2TLwRmDGScQf7rcXy09YHp6OGrudTf77JOVB55LHAvcGk-zvosWpAzNpe7QtNmJCdSRC8WXoQrOd0pl73jm_PgupRV0QtJ0l3C-ad0xJqdP28fWVqUD03XxGA7hCAzYtY3s=w174-h127-no)

만약 행의 크기가 열의 크기보다 크다면 다음과 같은 모양을 가집니다. 즉, m × n 행렬일 때, m > n인 경우입니다.

![](https://lh3.googleusercontent.com/qt1wxqVy0a9kfT4DVDTgiJVhtXKfI5TkRKMrQDMnalStqLewut4zybluypuBMwxUUgZilb_KT-I0Mt5vKVwkPosRI8S98bUPvTvOhUFJFRHv7o9FZhRmXw2ZAISqamkp919WOTShZjImJgWRKtvQZVI4DLXMct2U9IocS0EN22-sGU0-MqKqpVyr7Eooyg9ReeYfKse3k3KF0BfigjRlcTyl_MeZ7dQf2olzTiY3YEjytxscH7xd9ViTuLccgqT5G1zFSIoN8kMP13JwkeELbLckoR-B0Yd3x5NORniDmm6kJfh37bx4xmLvb69MJArp0sSfw2FPUDy7fmCwyCigidAEDPa9IktVegqNqQ1S9r8J9KG8I1-ZNQF0UNL0BliloP6Dsx2yD698JE4Qm0dU80vaeQ6o6SuIVHBeuYm9S-vKU4WHYUL2fj888rVTU3u5u2-aBStPMFM9V--g71rx74Zb-D8clagTTiDBoVjA77tR9DiBUt1NBV6B5sSIhmP6GrL7r8TauJjM6xIJti5mxlojpgkPlOHNffte0GFVDAZ85touvy2_9ZXAh8VSPlAfLFD53eYQLwfh49EDAsgHD5L4jUpSrTpOoDF4yjFc300pE7aqlmqe11fvKuue9qm4QbtahEFpi33SYrdPHXPbUuGIRDeFaQk=w174-h173-no)

반면 n > m인 경우에는 다음과 같은 모양을 가집니다.

![](https://lh3.googleusercontent.com/LXr2U1idlgU6NwnU-b79T8uhLrHBzFfZL5unZp2us-M2qWMriqpBOIJh06m0yEJzwoIAfAoqRBRARWenWxN-ZW65M8WxpYKtXb4aaS2K60Cg6GeGCpfo1RhhZEMo5Hh4SUk2BVqFxcdAr6r5affaCB-jz1ji0Op6UjPHON-gj7EM2uOqKqLF5g7cW2Jsm7tgQAEd6woyHfawokgFN2v_a8toloq5VuyK_wxS9u16X-7momz_dsz4cCC_h33LkCvzYe1b2uHqmhOnFm4dLKZoeNFKrHpX1pJSY1X_uBmjwXsGtSFKg3A3TPZXuirN-c2klbb1AQSnQING5IqxwkilhYNsC0jr5Cbe6pRXEpDBoAT-9XXL6TcrjhlJ5NUgSv5ApjCCRWhXWEsJICAxv9Z8tdpAaJovp3qnozwaU6XNksW0CIp_lqTyXh6sMwHbRCATmTraqfK4YjMs2ugT-BYBNtKLdMp-RZbe2INpnCXTJB0NiI3kkt4umLciumfMad2FOlYU2_83Sl0NpuXFuFRjlyMlbjJrP5UgqwkfMO1oZE9BjGJCjUSETQQr7IiohT-VQahjdyhWhhiwnv5H40eIyRlVasj9TtPx6FLON4buEewd99oZwoWvMv_a5_rs3TspkoWet8YFnfVf4PJHFKJRMw7bEAOEOJ8=w201-h130-no)

SVD를 통해 나온 대각 행렬 Σ는 추가적인 성질을 가지는데, 대각 행렬 Σ의 주대각원소를 행렬 A의 특이값(singular value)라고 하며, 이를 σ1, σ2, ...,σr라고 표현한다고 하였을 때 특이값 σ1, σ2, ...,σr은 내림차순으로 정렬되어 있다는 특징을 가집니다.

아래의 그림은 특이값 12.4, 9.5, 1.3이 내림차순으로 정렬되어져 있는 모습을 보여줍니다.

![](https://lh3.googleusercontent.com/oCkPt2FhAswCf9xnLDc-ByMbC6yzPb4m5KPIb94yiGtmRjRntDSdvUjWpPhuuwvJt_nCGb-awV8zsRfLNtSH1Vyn2sJRzeysGkJS4WLm0ey514BRH2M14ccIH8v--Yh3FZyTAnMW0kR_5MYQrbdyl-SUggrlu7WOK0Qpxo54FhfhR_-XSImMGiacaOfae0Hq63DqQo08BCgoOvBjxDFF_SXQDPGZRK_OtfaH48ivBGnVclg919vEmh6XdYsaxNEzCrb9bEnJxI0PpnSLVOFOtXgARDh8WkFmFqu8lzNmkvt-SHE0z9VCs3gy4A8NxcfxWhaXZLimJb6DbcnRPgomcETltTzrVCufRpcqC0f_c1wzjZ-YiH9V8TPEtx_VDddcML9bdGTIIihqo3gzn78c4ojwvzgK_hTsvNI96bRZPSwGttYgN9rb_xToxOJH2KmGNZ6VaDPw1okia6ehbAZNq0A_Pq67ZVuy-j0XGmSFjjfCRl0f6DtGSXRq6JW5q7mVGZi4ihyxsViR61P1SralwhUcTFEBv0s5BT80lWwjot0MWd60jpQeohKEkie1qAz-OIVceRpI2MbeBDZI1d0FTFwWcBkQDDfVARL7XATEFIGirR-6MDJJjHHxTOehn_2fLuvhbPwRCx2tHKVC2SCNvUVxY7NkYFM=w201-h130-no)

### 2) 절단된 SVD(Truncated SVD)

LSA의 경우 풀 SVD에서 나온 3개의 행렬에서 일부 벡터들을 삭제시킨 절단된 SVD(truncated SVD)를 사용하게 됩니다.

![](https://wikidocs.net/images/page/24949/svd%EC%99%80truncatedsvd.PNG)

절단된 SVD는 대각 행렬 Σ의 대각 원소의 값 중에서 상위값 t개만 남게 됩니다.

또한, U행렬과 V행렬의 t열까지만 남깁니다. 여기서 t는 우리가 찾고자하는 토픽의 수를 반영한 하이퍼파라미터값이 되는 것입니다. 하이퍼파라미터란 사용자가 직접 값을 선택하며 성능에 영향을 주는 매개변수를 말합니다. t를 선택하는 것은 쉽지 않은 일입니다. t를 크게 잡으면 기존의 행렬 A로부터 다양한 의미를 가져갈 수 있지만, t를 작게 잡아야만 노이즈를 제거할 수 있기 때문입니다.

### 3) 잠재 의미 분석(Latent Semantic Analysis, LSA)

LSA는 기본적으로 DTM이나 TF-IDF 행렬에 절단된 SVD(truncated SVD)를 사용하여 차원을 축소시키고, 단어들의 잠재적인 의미를 끌어낸다는 아이디어를 갖고 있습니다.

![](https://lh3.googleusercontent.com/kVIj_n7PozQ4YPPzjMqy3WaBcIQreNsOUjBMpRPVO15Yltam0Ex_Xvc3unJyI_unpEu1QySK6DQvwxBR_NaV4G6iUOMCK9TH6wKdAGkyLEf_f5Aekcc3q9S5G0k3Lk9z_ycMnTBHZ48ZuoxeF3pFvxdLXM5v7bXfz9OMtSXcttRDdSm9fhjriaCv0dVrLkdburALulx1sHei7Uem2xwsOxUBIAABC9MLHm3j0oxjQ8rOCMlqSN4F2veJWPHrVAKT3qq7U2RGMulIG9BbB0A2C5mPNRSBXhJaqldIPBVWblA30ZoPh-EbavnT-MkkljP4ZXHFvW-TLFYGqc_cLdvwEcUyyrMqXMZE_iYC0E-boieRnsEOgrQtsZTL-vqfFqHyERI1PB-6fLiRGRU6gLn7LItA5ZQjPDHSM-Q4neTR0AkMAUskzAP-7-fDE7QvWtmlOY6Yq33fOg-5Bgfht1AlqQukY7Cz5MoFrZoUiDUUaepcK1UwcExOxFce0_nSaM9rWPglgJ2W29T76Uf2ipsZCQnRQUu9l4BX-46vmck9FilqIA3_BIcS7iK5a9EANOlmFJM0IHDlL1RoSVOWV0KAowL6ZxI-asKbrdfQCshLE-EG6X6YvhXl7Wgr7V0OTh_wf5a8KVCICVZRs0rq9OVpts26PL_m_SM=w945-h276-no)

위와 같은 DTM을 실제로 파이썬을 통해서 만들면 다음과 같습니다.


```python
import numpy as np
A=np.array([[0,0,0,1,0,1,1,0,0],[0,0,0,1,1,0,1,0,0],[0,1,1,0,2,0,0,0,0],[1,0,0,0,0,0,0,1,1]])
np.shape(A)
```




    (4, 9)



이에 대해서 풀 SVD(full SVD)를 수행해보겠습니다. 단, 여기서는 대각 행렬의 변수명을 Σ가 아니라 S를 사용합니다. 또한 V의 전치 행렬을 VT라고 하겠습니다.


```python
U, s, VT = np.linalg.svd(A, full_matrices = True)
```


```python
print(U.round(2))
np.shape(U)
```

    [[-0.24  0.75  0.   -0.62]
     [-0.51  0.44 -0.    0.74]
     [-0.83 -0.49 -0.   -0.27]
     [-0.   -0.    1.    0.  ]]
    




    (4, 4)



4 × 4의 크기를 가지는 직교 행렬 U가 생성되었습니다. 

소수점의 길이가 너무 길게 출력하면 보기 힘들어서 두번째 자리까지만 출력하기위해서 .round(2)를 사용합니다.

이제 대각 행렬 S를 확인해봅시다.


```python
print(s.round(2))
np.shape(s)
```

    [2.69 2.05 1.73 0.77]
    




    (4,)



Numpy의 linalg.svd()는 특이값 분해의 결과로 대각 행렬이 아니라 특이값의 리스트를 반환합니다. 그러므로 앞서 본 수식의 형식으로 보려면 이를 다시 대각 행렬로 바꾸어 주어야 합니다. 우선 특이값을 s에 저장하고 대각 행렬 크기의 행렬을 생성한 후에 그 행렬에 특이값을 삽입해도록 하겠습니다.


```python
S = np.zeros((4, 9)) # 대각 행렬의 크기인 4 x 9의 임의의 행렬 생성
S[:4, :4] = np.diag(s) # 특이값을 대각행렬에 삽입
print(S.round(2))
np.shape(S)
```

    [[2.69 0.   0.   0.   0.   0.   0.   0.   0.  ]
     [0.   2.05 0.   0.   0.   0.   0.   0.   0.  ]
     [0.   0.   1.73 0.   0.   0.   0.   0.   0.  ]
     [0.   0.   0.   0.77 0.   0.   0.   0.   0.  ]]
    




    (4, 9)



4 × 9의 크기를 가지는 대각 행렬 S가 생성되었습니다. 2.69 > 2.05 > 1.73 > 0.77 순으로 값이 내림차순을 보이는 것을 확인할 수 있습니다.


```python
print(VT.round(2))
np.shape(VT)
```

    [[-0.   -0.31 -0.31 -0.28 -0.8  -0.09 -0.28 -0.   -0.  ]
     [ 0.   -0.24 -0.24  0.58 -0.26  0.37  0.58 -0.   -0.  ]
     [ 0.58 -0.    0.    0.   -0.    0.   -0.    0.58  0.58]
     [ 0.   -0.35 -0.35  0.16  0.25 -0.8   0.16 -0.   -0.  ]
     [-0.   -0.78 -0.01 -0.2   0.4   0.4  -0.2   0.    0.  ]
     [-0.29  0.31 -0.78 -0.24  0.23  0.23  0.01  0.14  0.14]
     [-0.29 -0.1   0.26 -0.59 -0.08 -0.08  0.66  0.14  0.14]
     [-0.5  -0.06  0.15  0.24 -0.05 -0.05 -0.19  0.75 -0.25]
     [-0.5  -0.06  0.15  0.24 -0.05 -0.05 -0.19 -0.25  0.75]]
    




    (9, 9)



9 × 9의 크기를 가지는 직교 행렬 VT(V의 전치 행렬)가 생성되었습니다. 즉, U × S × VT를 하면 기존의 행렬 A가 나와야 합니다.

 Numpy의 allclose()는 2개의 행렬이 동일하면 True를 리턴합니다. 이를 사용하여 정말로 기존의 행렬 A와 동일한지 확인해보겠습니다.


```python
np.allclose(A, np.dot(np.dot(U,S), VT).round(2))
```




    True



지금까지 수행한 것은 풀 SVD(Full SVD)입니다. 

이제 t를 정하고, 절단된 SVD(Truncated SVD)를 수행해보도록 합시다. 여기서는 t=2로 하겠습니다. 우선 대각 행렬 S 내의 특이값 중에서 상위 2개만 남기고 제거해보도록 하겠습니다.


```python
S=S[:2,:2]
print(S.round(2))
```

    [[2.69 0.  ]
     [0.   2.05]]
    

상위 2개의 값만 남기고 나머지는 모두 제거된 것을 볼 수 있습니다.

이제 직교 행렬 U에 대해서도 2개의 열만 남기고 제거합니다.


```python
U=U[:,:2]
print(U.round(2))
```

    [[-0.24  0.75]
     [-0.51  0.44]
     [-0.83 -0.49]
     [-0.   -0.  ]]
    

이제 행렬 V의 전치 행렬인 VT에 대해서 2개의 행만 남기고 제거합니다. 


```python
VT=VT[:2,:]
print(VT.round(2))
```

    [[-0.   -0.31 -0.31 -0.28 -0.8  -0.09 -0.28 -0.   -0.  ]
     [ 0.   -0.24 -0.24  0.58 -0.26  0.37  0.58 -0.   -0.  ]]
    

이제 축소된 행렬 U, S, VT에 대해서 다시 U × S × VT연산을 하면 기존의 A와는 다른 결과가 나오게 됩니다.

값이 손실되었기 때문에 이 세 개의 행렬로는 이제 기존의 A행렬을 복구할 수 없습니다.


```python
A_prime=np.dot(np.dot(U,S), VT)
print(A) # 기존행렬 A
print(A_prime.round(2))
```

    [[0 0 0 1 0 1 1 0 0]
     [0 0 0 1 1 0 1 0 0]
     [0 1 1 0 2 0 0 0 0]
     [1 0 0 0 0 0 0 1 1]]
    [[ 0.   -0.17 -0.17  1.08  0.12  0.62  1.08 -0.   -0.  ]
     [ 0.    0.2   0.2   0.91  0.86  0.45  0.91  0.    0.  ]
     [ 0.    0.93  0.93  0.03  2.05 -0.17  0.03  0.    0.  ]
     [ 0.    0.    0.    0.    0.   -0.    0.    0.    0.  ]]
    

축소된 U는 4 × 2의 크기를 가지는데, 이는 잘 생각해보면 문서의 개수 × 토픽의 수 t의 크기입니다. 단어의 개수인 9는 유지되지 않는데 문서의 개수인 4의 크기가 유지되었으니 4개의 문서 각각을 2개의 값으로 표현하고 있습니다. 즉, U의 각 행은 잠재 의미를 표현하기 위한 수치화 된 각각의 문서 벡터라고 볼 수 있습니다.

축소된 VT는 2 × 9의 크기를 가지는데, 이는 잘 생각해보면 토픽의 수 t × 단어의 개수의 크기입니다. VT의 각 열은 잠재 의미를 표현하기 위해 수치화된 각각의 단어 벡터라고 볼 수 있습니다.

### 4) 실습을 통한 이해

사이킷 런에서는 Twenty Newsgroups이라고 불리는 20개의 다른 주제를 가진 뉴스 데이터를 제공합니다.

앞서 언급했듯이 LSA가 토픽 모델링에 최적화 된 알고리즘은 아니지만, 토픽 모델링이라는 분야의 시초가 되는 알고리즘입니다.

여기서는 LSA를 사용해서 문서의 수를 원하는 토픽의 수로 압축한 뒤에 각 토픽당 가장 중요한 단어 5개를 출력하는 실습으로 토픽 모델링을 수행해보도록 하겠습니다.

#### (1) 뉴스 데이터에 대한 이해


```python
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data
len(documents)
```




    11314



훈련에 사용할 뉴스는 총 11,314개입니다. 이 중 첫번째 훈련용 뉴스를 출력해보겠습니다.


```python
documents[1]
```




    "\n\n\n\n\n\n\nYeah, do you expect people to read the FAQ, etc. and actually accept hard\natheism?  No, you need a little leap of faith, Jimmy.  Your logic runs out\nof steam!\n\n\n\n\n\n\n\nJim,\n\nSorry I can't pity you, Jim.  And I'm sorry that you have these feelings of\ndenial about the faith you need to get by.  Oh well, just pretend that it will\nall end happily ever after anyway.  Maybe if you start a new newsgroup,\nalt.atheist.hard, you won't be bummin' so much?\n\n\n\n\n\n\nBye-Bye, Big Jim.  Don't forget your Flintstone's Chewables!  :) \n--\nBake Timmons, III"



뉴스 데이터에는 특수문자가 포함된 다수의 영어문장으로 구성되어져 있습니다.

사이킷 런이 제공하는 뉴스 데이터에서 target_name에는 본래 이 뉴스 데이터가 어떤 20개의 카테고리를 갖고있었는지가 저장되어져 있습니다. 이를 출력해보겠습니다.


```python
print(dataset.target_names)
```

    ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
    

#### (2) 텍스트 전처리

기본적인 아이디어는 알파벳을 제외한 구두점, 숫자, 특수 문자를 제거하는 것입니다. 이는 텍스트 전처리 챕터에서 정제 기법으로 배웠던 정규 표현식을 통해서 해결할 수 있습니다.

짧은 단어는 유용한 정보를 담고있지 않다고 가정하고, 길이가 짧은 단어도 제거합니다.

모든 알파벳을 소문자로 바꿔서 단어의 개수를 줄이는 작업을 합니다.


```python
news_df = pd.DataFrame({'document':documents})
# 특수 문자 제거
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z]", " ")
# 길이가 3이하인 단어는 제거 (길이가 짧은 단어 제거)
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
# 전체 단어에 대한 소문자 변환
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())
```


```python
news_df['clean_doc'][1]
```




    'yeah expect people read actually accept hard atheism need little leap faith jimmy your logic runs steam sorry pity sorry that have these feelings denial about faith need well just pretend that will happily ever after anyway maybe start newsgroup atheist hard bummin much forget your flintstone chewables bake timmons'



우선 특수문자가 제거되었으며, if나 you와 같은 길이가 3이하인 단어가 제거된 것을 확인할 수 있습니다. 뿐만 아니라 대문자가 전부 소문자로 바뀌었습니다.

이제 뉴스 데이터에서 불용어를 제거합니다. 불용어를 제거하기 위해서 토큰화를 우선 수행합니다.


```python
from nltk.corpus import stopwords
stop_words = stopwords.words('english') # NLTK로부터 불용어를 받아옵니다.
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split()) # 토큰화
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
# 불용어를 제거합니다.
```


```python
print(tokenized_doc[1])
```

    ['yeah', 'expect', 'people', 'read', 'actually', 'accept', 'hard', 'atheism', 'need', 'little', 'leap', 'faith', 'jimmy', 'logic', 'runs', 'steam', 'sorry', 'pity', 'sorry', 'feelings', 'denial', 'faith', 'need', 'well', 'pretend', 'happily', 'ever', 'anyway', 'maybe', 'start', 'newsgroup', 'atheist', 'hard', 'bummin', 'much', 'forget', 'flintstone', 'chewables', 'bake', 'timmons']
    

기존에 있었던 불용어에 속하던 your, about, just, that, will, after 단어들이 사라졌을 뿐만 아니라, 토큰화가 수행된 것을 확인할 수 있습니다.

#### (3) TF-IDF 행렬 만들기

불용어 제거를 위해 토큰화 작업을 수행하였지만, TfidfVectorizer(TF-IDF 챕터 참고)는 기본적으로 토큰화가 되어있지 않은 텍스트 데이터를 입력으로 사용합니다.

그렇기 때문에 TfidfVectorizer를 사용해서 TF-IDF 행렬을 만들기 위해서 다시 토큰화 작업을 역으로 취소하는 작업을 수행해보도록 하겠습니다. 이를 역토큰화(Detokenization)라고 합니다.


```python
# 역토큰화 (토큰화 작업을 역으로 되돌림)
detokenized_doc = []
for i in range(len(news_df)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)

news_df['clean_doc'] = detokenized_doc
```

역토큰화가 제대로 되었는지 다시 첫번째 훈련용 뉴스를 출력하여 확인해보겠습니다.


```python
news_df['clean_doc'][1]
```




    'yeah expect people read actually accept hard atheism need little leap faith jimmy logic runs steam sorry pity sorry feelings denial faith need well pretend happily ever anyway maybe start newsgroup atheist hard bummin much forget flintstone chewables bake timmons'



이제 사이킷런의 TfidfVectorizer를 통해 단어 1,000개에 대한 TF-IDF 행렬을 만들 것입니다. 


```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', 
max_features= 1000, # 상위 1,000개의 단어를 보존 
max_df = 0.5, 
smooth_idf=True)

X = vectorizer.fit_transform(news_df['clean_doc'])
X.shape # TF-IDF 행렬의 크기 확인
```




    (11314, 1000)



11,314 × 1,000의 크기를 가진 TF-IDF 행렬이 생성되었음을 확인할 수 있습니다.

#### (4) 토픽 모델링(Topic Modeling)

이제 TF-IDF 행렬을 다수의 행렬로 분해해보도록 하겠습니다.

여기서는 사이킷 런의 절단된 SVD(Truncated SVD)를 사용합니다. 절단된 SVD를 사용하면 차원을 축소할 수 있습니다.

원래 기존 뉴스 데이터가 20개의 뉴스 카테고리를 갖고있었기 때문에, 20개의 토픽을 가졌다고 가정하고 토픽 모델링을 시도해보겠습니다. 토픽의 숫자는 n_components의 파라미터로 지정이 가능합니다.


```python
from sklearn.decomposition import TruncatedSVD
svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(X)
len(svd_model.components_)
```




    20



여기서 svd_model.componets_는 앞서 배운 LSA에서 VT에 해당됩니다.


```python
np.shape(svd_model.components_)
```




    (20, 1000)



정확하게 토픽의 수 t × 단어의 수의 크기를 가지는 것을 볼 수 있습니다.


```python
terms = vectorizer.get_feature_names() # 단어 집합. 1,000개의 단어가 저장됨.

def get_topics(components, feature_names, n=5):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (idx+1), [(feature_names[i], topic[i].round(5)) for i in topic.argsort()[:-n - 1:-1]])
get_topics(svd_model.components_,terms)
```

    Topic 1: [('like', 0.21386), ('know', 0.20046), ('people', 0.19293), ('think', 0.17805), ('good', 0.15128)]
    Topic 2: [('thanks', 0.32888), ('windows', 0.29088), ('card', 0.18069), ('drive', 0.17455), ('mail', 0.15111)]
    Topic 3: [('game', 0.37064), ('team', 0.32443), ('year', 0.28154), ('games', 0.2537), ('season', 0.18419)]
    Topic 4: [('drive', 0.53324), ('scsi', 0.20165), ('hard', 0.15628), ('disk', 0.15578), ('card', 0.13994)]
    Topic 5: [('windows', 0.40399), ('file', 0.25436), ('window', 0.18044), ('files', 0.16078), ('program', 0.13894)]
    Topic 6: [('chip', 0.16114), ('government', 0.16009), ('mail', 0.15625), ('space', 0.1507), ('information', 0.13562)]
    Topic 7: [('like', 0.67086), ('bike', 0.14236), ('chip', 0.11169), ('know', 0.11139), ('sounds', 0.10371)]
    Topic 8: [('card', 0.46633), ('video', 0.22137), ('sale', 0.21266), ('monitor', 0.15463), ('offer', 0.14643)]
    Topic 9: [('know', 0.46047), ('card', 0.33605), ('chip', 0.17558), ('government', 0.1522), ('video', 0.14356)]
    Topic 10: [('good', 0.42756), ('know', 0.23039), ('time', 0.1882), ('bike', 0.11406), ('jesus', 0.09027)]
    Topic 11: [('think', 0.78469), ('chip', 0.10899), ('good', 0.10635), ('thanks', 0.09123), ('clipper', 0.07946)]
    Topic 12: [('thanks', 0.36824), ('good', 0.22729), ('right', 0.21559), ('bike', 0.21037), ('problem', 0.20894)]
    Topic 13: [('good', 0.36212), ('people', 0.33985), ('windows', 0.28385), ('know', 0.26232), ('file', 0.18422)]
    Topic 14: [('space', 0.39946), ('think', 0.23258), ('know', 0.18074), ('nasa', 0.15174), ('problem', 0.12957)]
    Topic 15: [('space', 0.31613), ('good', 0.3094), ('card', 0.22603), ('people', 0.17476), ('time', 0.14496)]
    Topic 16: [('people', 0.48156), ('problem', 0.19961), ('window', 0.15281), ('time', 0.14664), ('game', 0.12871)]
    Topic 17: [('time', 0.34465), ('bike', 0.27303), ('right', 0.25557), ('windows', 0.1997), ('file', 0.19118)]
    Topic 18: [('time', 0.5973), ('problem', 0.15504), ('file', 0.14956), ('think', 0.12847), ('israel', 0.10903)]
    Topic 19: [('file', 0.44163), ('need', 0.26633), ('card', 0.18388), ('files', 0.17453), ('right', 0.15448)]
    Topic 20: [('problem', 0.33006), ('file', 0.27651), ('thanks', 0.23578), ('used', 0.19206), ('space', 0.13185)]
    

각 20개의 행의 각 1,000개의 열 중 가장 값이 큰 5개의 값을 찾아서 단어로 출력합니다.

### 5) LSA의 장단점(Pros and Cons of LSA)

- 장점<br/>
LSA는 쉽고 빠르게 구현이 가능할 뿐만 아니라 단어의 잠재적인 의미를 이끌어낼 수 있어 문서의 유사도 계산 등에서 좋은 성능을 보여준다.


- 단점<br/>
SVD의 특성상 이미 계산된 LSA에 새로운 데이터를 추가하여 계산하려고하면 보통 처음부터 다시 계산해야 합니다. 즉, 새로운 정보에 대해 업데이트가 어렵습니다.

## 2. 잠재 디리클레 할당(Latent Dirichlet Allocation, LDA)

토픽 모델링은 문서의 집합에서 토픽을 찾아내는 프로세스를 말합니다.

LDA는 문서들은 토픽들의 혼합으로 구성되어져 있으며, 토픽들은 확률 분포에 기반하여 단어들을 생성한다고 가정합니다.

### 1) 잠재 디리클레 할당(Latent Dirichlet Allocation, LDA) 개요

LDA에 문서 집합을 입력하면, 어떤 결과를 보여주는지 간소화 된 예를 들어 보겠습니다.

- 문서1 : 저는 사과랑 바나나를 먹어요 
- 문서2 : 우리는 귀여운 강아지가 좋아요
- 문서3 : 저의 깜찍하고 귀여운 강아지가 바나나를 먹어요

LDA를 수행할 때 문서 집합에서 토픽이 몇 개가 존재할지 가정하는 것은 사용자가 해야 할 일입니다. 

토픽의 개수를 의미하는 변수를 k라고 하였을 때, k를 2로 한다는 의미입니다.

LDA가 위의 세 문서로부터 2개의 토픽을 찾은 결과는 아래와 같습니다.

전처리 과정을 거친 DTM이 LDA의 입력이 되었다고 가정합니다.

LDA는 각 문서의 토픽 분포와 각 토픽 내의 단어 분포를 추정합니다.

<각 문서의 토픽 분포>
- 문서1 : 토픽 A 100%
- 문서2 : 토픽 B 100%
- 문서3 : 토픽 B 60%, 토픽 A 40%

<각 토픽의 단어 분포>
- 토픽A : 사과 20%, 바나나 40%, 먹어요 40%, 귀여운 0%, 강아지 0%, 깜찍하고 0%, 좋아요 0%
- 토픽B : 사과 0%, 바나나 0%, 먹어요 0%, 귀여운 33%, 강아지 33%, 깜찍하고 16%, 좋아요 16%

LDA는 토픽의 제목을 정해주지 않지만, 이 시점에서 알고리즘의 사용자는 두 토픽이 각각 과일에 대한 토픽과 강아지에 대한 토픽이라고 판단해볼 수 있습니다.

이제 LDA에 대해서 알아봅시다.

### 2) LDA의 가정

LDA는 문서의 집합으로부터 어떤 토픽이 존재하는지를 알아내기 위한 알고리즘입니다. 

LDA는 앞서 배운 빈도수 기반의 표현 방법인 BoW의 행렬 DTM 또는 TF-IDF 행렬을 입력으로 하는데, 이로부터 알 수 있는 사실은 LDA는 단어의 순서는 신경쓰지 않겠다는 겁니다.

각각의 문서는 다음과 같은 과정을 거쳐서 작성되었다고 가정합니다.

1. 문서에 사용할 단어의 개수 N을 정합니다.
 - Ex) 5개의 단어를 정하였습니다.
2. 문서에 사용할 토픽의 혼합을 확률 분포에 기반하여 결정합니다.
 - Ex) 위 예제와 같이 토픽이 2개라고 하였을 때 강아지 토픽을 60%, 과일 토픽을 40%와 같이 선택할 수 있습니다.
3. 문서에 사용할 각 단어를 (아래와 같이) 정합니다.<br/>
 3-1) 토픽 분포에서 토픽 T를 확률적으로 고릅니다.
 - Ex) 60% 확률로 강아지 토픽을 선택하고, 40% 확률로 과일 토픽을 선택할 수 있습니다.<br/>
 
 3-2) 선택한 토픽 T에서 단어의 출현 확률 분포에 기반해 문서에 사용할 단어를 고릅니다.<br/>
 - Ex) 강아지 토픽을 선택하였다면, 33% 확률로 강아지란 단어를 선택할 수 있습니다. 이제 3)을 반복하면서 문서를 완성합니다.

이러한 과정을 통해 문서가 작성되었다는 가정 하에 LDA는 토픽을 뽑아내기 위하여 위 과정을 역으로 추적하는 역공학(reverse engneering)을 수행합니다.

### 3) LDA의 수행하기

LDA의 수행 과정

#### (1) 사용자는 알고리즘에게 토픽의 개수 k를 알려줍니다.

LDA는 토픽의 개수 k를 입력받으면, k개의 토픽이 M개의 전체 문서에 걸쳐 분포되어 있다고 가정합니다.

#### (2) 모든 단어를 k개 중 하나의 토픽에 할당합니다.

LDA는 모든 문서의 모든 단어에 대해서 k개 중 하나의 토픽을 랜덤으로 할당합니다. 

#### (3) 이제 모든 문서의 모든 단어에 대해서 아래의 사항을 반복 진행합니다. (iterative)

#### (3-1) 어떤 문서의 각 단어 w는 자신은 잘못된 토픽에 할당되어져 있지만, 다른 단어들은 전부 올바른 토픽에 할당되어져 있는 상태라고 가정합니다. 이에 따라 단어 w는 아래의 두 가지 기준에 따라서 토픽이 재할당됩니다.

p(topic t | document d) : 문서 d의 단어들 중 토픽 t에 해당하는 단어들의 비율<br/>
p(word w | topic t) : 단어 w를 갖고 있는 모든 문서들 중 토픽 t가 할당된 비율

이를 반복하면, 모든 할당이 완료된 수렴 상태가 됩니다.

![](https://wikidocs.net/images/page/30708/lda1.PNG)

위의 그림은 두 개의 문서 doc1과 doc2를 보여줍니다. 여기서는 doc1의 세번째 단어 apple의 토픽을 결정하고자 합니다.

![](https://wikidocs.net/images/page/30708/lda3.PNG)

우선 첫번째로 사용하는 기준은 문서 doc1의 단어들이 어떤 토픽에 해당하는지를 봅니다. doc1의 모든 단어들은 토픽 A와 토픽 B에 50 대 50의 비율로 할당되어져 있으므로, 이 기준에 따르면 단어 apple은 토픽 A 또는 토픽 B 둘 중 어디에도 속할 가능성이 있습니다.

![](https://wikidocs.net/images/page/30708/lda2.PNG)

두번째 기준은 단어 apple이 전체 문서에서 어떤 토픽에 할당되어져 있는지를 봅니다. 이 기준에 따르면 단어 apple은 토픽 B에 할당될 가능성이 높습니다. 이러한 두 가지 기준을 참고하여 LDA는 doc1의 apple을 어떤 토픽에 할당할지 결정합니다.

### 4) 잠재 디리클레 할당과 잠재 의미 분석의 차이

- LSA : DTM을 차원 축소 하여 축소 차원에서 근접 단어들을 토픽으로 묶는다.
- LDA : 단어가 특정 토픽에 존재할 확률과 문서에 특정 토픽이 존재할 확률을 결합확률로 추정하여 토픽을 추출한다.

### 5) 실습을 통한 이해

이제 gensim을 사용하여 LDA를 실습하겠습니다.

#### (1) 정수 인코딩과 단어 집합 만들기

Twenty Newsgroups이라고 불리는 20개의 다른 주제를 가진 뉴스 데이터를 다시 사용합니다.

동일한 전처리 과정을 거친 후에 tokenized_doc으로 저장한 상태라고 합시다. 훈련용 뉴스를 5개만 출력해보겠습니다.


```python
tokenized_doc[:5]
```




    0    [well, sure, story, seem, biased, disagree, st...
    1    [yeah, expect, people, read, actually, accept,...
    2    [although, realize, principle, strongest, poin...
    3    [notwithstanding, legitimate, fuss, proposal, ...
    4    [well, change, scoring, playoff, pool, unfortu...
    Name: clean_doc, dtype: object



이제 각 단어에 정수 인코딩을 하는 동시에, 각 뉴스에서의 단어의 빈도수를 기록해보겠습니다. 여기서는 각 단어를 (word_id, word_frequency)의 형태로 바꾸고자 합니다.

word_id는 단어가 정수 인코딩된 값이고, word_frequency는 해당 뉴스에서의 해당 단어의 빈도수를 의미합니다.

이는 gensim의 corpora.Dictionary()를 사용하여 손쉽게 구할 수 있습니다. 전체 뉴스에 대해서 정수 인코딩을 수행하고, 두번째 뉴스를 출력해봅시다.


```python
from gensim import corpora
dictionary = corpora.Dictionary(tokenized_doc)
corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
print(corpus[1]) # 수행된 결과에서 두번째 뉴스 출력. 첫번째 문서의 인덱스는 0
```

    [(52, 1), (55, 1), (56, 1), (57, 1), (58, 1), (59, 1), (60, 1), (61, 1), (62, 1), (63, 1), (64, 1), (65, 1), (66, 2), (67, 1), (68, 1), (69, 1), (70, 1), (71, 2), (72, 1), (73, 1), (74, 1), (75, 1), (76, 1), (77, 1), (78, 2), (79, 1), (80, 1), (81, 1), (82, 1), (83, 1), (84, 1), (85, 2), (86, 1), (87, 1), (88, 1), (89, 1)]
    

두번째 뉴스의 출력 결과를 봅시다.

위의 출력 결과 중에서 (66, 2)는 정수 인코딩이 66으로 할당된 단어가 두번째 뉴스에서는 두 번 등장하였음을 의미합니다.

66이라는 값을 가지는 단어가 정수 인코딩이 되기 전에는 어떤 단어였는지 확인하여봅시다.


```python
print(dictionary[66])
```

    faith
    

기존에는 단어 'faith'이었음을 알 수 있습니다.


```python
len(dictionary)
```




    64281



총 65,284개의 단어가 학습되었습니다. 이제 LDA 모델을 훈련시켜보겠습니다.

#### (2) LDA 모델 훈련시키기

기존의 뉴스 데이터가 총 20개의 카테고리를 가지고 있었으므로 토픽의 개수를 20으로 하여 LDA 모델을 학습시켜보도록 하겠습니다.


```python
import gensim
NUM_TOPICS = 20 #20개의 토픽, k=20
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)
```

    (0, '0.023*"science" + 0.012*"scientific" + 0.009*"disease" + 0.008*"objective"')
    (1, '0.011*"year" + 0.010*"game" + 0.009*"team" + 0.008*"said"')
    (2, '0.016*"people" + 0.012*"would" + 0.008*"think" + 0.006*"believe"')
    (3, '0.028*"space" + 0.011*"nasa" + 0.008*"center" + 0.006*"data"')
    (4, '0.034*"file" + 0.019*"output" + 0.019*"entry" + 0.018*"window"')
    (5, '0.020*"government" + 0.013*"president" + 0.011*"public" + 0.008*"states"')
    (6, '0.017*"would" + 0.011*"like" + 0.010*"know" + 0.009*"time"')
    (7, '0.011*"militia" + 0.010*"military" + 0.010*"nuclear" + 0.009*"weapon"')
    (8, '0.026*"armenian" + 0.023*"israel" + 0.023*"armenians" + 0.014*"israeli"')
    (9, '0.023*"drive" + 0.019*"card" + 0.017*"thanks" + 0.013*"disk"')
    (10, '0.008*"like" + 0.007*"bike" + 0.006*"much" + 0.005*"cars"')
    (11, '0.017*"sale" + 0.015*"shipping" + 0.013*"price" + 0.013*"offer"')
    (12, '0.027*"jesus" + 0.013*"church" + 0.012*"bible" + 0.011*"christ"')
    (13, '0.015*"turkish" + 0.012*"greek" + 0.010*"history" + 0.008*"germany"')
    (14, '0.017*"windows" + 0.013*"available" + 0.012*"software" + 0.012*"version"')
    (15, '0.028*"period" + 0.019*"play" + 0.017*"chicago" + 0.015*"power"')
    (16, '0.023*"apartment" + 0.016*"father" + 0.016*"sumgait" + 0.014*"woman"')
    (17, '0.021*"water" + 0.018*"filename" + 0.008*"pitcher" + 0.008*"stealth"')
    (18, '0.032*"mail" + 0.020*"send" + 0.020*"list" + 0.016*"internet"')
    (19, '0.017*"chip" + 0.012*"keys" + 0.011*"system" + 0.010*"data"')
    

각 단어 앞에 붙은 수치는 단어의 해당 토픽에 대한 기여도를 보여줍니다.

passes는 알고리즘의 동작 횟수를 말하는데, 알고리즘이 결정하는 토픽의 값이 적절히 수렴할 수 있도록 충분히 적당한 횟수를 정해주면 됩니다. 여기서는 총 15회를 수행하였습니다.

여기서는 num_words=4로 총 4개의 단어만 출력하도록 하였습니다.

#### (3) LDA 시각화 하기

LDA 시각화를 위해서는 pyLDAvis의 설치가 필요합니다.


```python
# import pyLDAvis.gensim
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
# pyLDAvis.display(vis)
```

![](https://wikidocs.net/images/page/30708/visualization_final.PNG)

좌측의 원들은 각각의 20개의 토픽을 나타냅니다. 각 원과의 거리는 각 토픽들이 서로 얼마나 다른지를 보여줍니다. 만약 두 개의 원이 겹친다면, 이 두 개의 토픽은 유사한 토픽이라는 의미입니다. 위의 그림에서는 10번 토픽을 클릭하였고, 이에 따라 우측에는 10번 토픽에 대한 정보가 나타납니다.

#### (4) 문서 별 토픽 분포 보기

우선 문서 별 토픽 분포를 확인하는 방법을 보겠습니다.

각 문서의 토픽 분포는 이미 훈련된 LDA 모델인 ldamodel[]에 전체 데이터가 정수 인코딩 된 결과를 넣은 후에 확인이 가능합니다. 


```python
for i, topic_list in enumerate(ldamodel[corpus]):
    if i==5:
        break
    print(i,'번째 문서의 topic 비율은',topic_list)
```

    0 번째 문서의 topic 비율은 [(1, 0.025709333), (2, 0.46393046), (8, 0.16392322), (13, 0.15981175), (18, 0.17451078)]
    1 번째 문서의 topic 비율은 [(1, 0.041393768), (2, 0.8739314), (3, 0.031111782), (12, 0.032486286)]
    2 번째 문서의 topic 비율은 [(2, 0.21293025), (6, 0.28951982), (8, 0.2635365), (10, 0.19721931), (13, 0.024664909)]
    3 번째 문서의 topic 비율은 [(2, 0.12250288), (5, 0.2543938), (6, 0.3057989), (10, 0.23378237), (12, 0.039818056), (18, 0.03272659)]
    4 번째 문서의 topic 비율은 [(1, 0.8298844), (4, 0.09122815), (18, 0.04739592)]
    

위의 출력 결과에서 (숫자, 확률)은 각각 토픽 번호와 해당 토픽이 해당 문서에서 차지하는 분포도를 의미합니다.

위의 코드를 응용하여 좀 더 깔끔한 형태인 데이터프레임 형식으로 출력해보겠습니다.


```python
def make_topictable_per_doc(ldamodel, corpus, texts):
    topic_table = pd.DataFrame()

    # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
    for i, topic_list in enumerate(ldamodel[corpus]):
        doc = topic_list[0] if ldamodel.per_word_topics else topic_list            
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.
        # EX) 정렬 전 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%), 
        # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)
        # 48 > 25 > 21 > 5 순으로 정렬이 된 것.

        # 모든 문서에 대해서 각각 아래를 수행
        for j, (topic_num, prop_topic) in enumerate(doc): #  몇 번 토픽인지와 비중을 나눠서 저장한다.
            if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                topic_table = topic_table.append(pd.Series([int(topic_num), round(prop_topic,4), topic_list]), ignore_index=True)
                # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
            else:
                break
    return(topic_table)
```


```python
topictable = make_topictable_per_doc(ldamodel, corpus, tokenized_doc)
topictable = topictable.reset_index() # 문서 번호을 의미하는 열(column)로 사용하기 위해서 인덱스 열을 하나 더 만든다.
topictable.columns = ['문서 번호', '가장 비중이 높은 토픽', '가장 높은 토픽의 비중', '각 토픽의 비중']
topictable[:10]
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
      <th>문서 번호</th>
      <th>가장 비중이 높은 토픽</th>
      <th>가장 높은 토픽의 비중</th>
      <th>각 토픽의 비중</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2.0</td>
      <td>0.4639</td>
      <td>[(1, 0.025710873), (2, 0.46391582), (8, 0.1639...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2.0</td>
      <td>0.8739</td>
      <td>[(1, 0.041391823), (2, 0.8739293), (3, 0.03111...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>6.0</td>
      <td>0.2895</td>
      <td>[(2, 0.21294913), (6, 0.28949916), (8, 0.26353...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>6.0</td>
      <td>0.3056</td>
      <td>[(2, 0.12268029), (5, 0.25439763), (6, 0.30563...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1.0</td>
      <td>0.8299</td>
      <td>[(1, 0.82987374), (4, 0.09124688), (18, 0.0473...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>6.0</td>
      <td>0.4436</td>
      <td>[(0, 0.04766284), (1, 0.11438124), (2, 0.10096...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>6.0</td>
      <td>0.3288</td>
      <td>[(0, 0.039850496), (3, 0.023102334), (6, 0.328...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>2.0</td>
      <td>0.3585</td>
      <td>[(1, 0.024587026), (2, 0.35846967), (6, 0.2213...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>10.0</td>
      <td>0.5348</td>
      <td>[(2, 0.13225864), (6, 0.19812517), (10, 0.5348...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>6.0</td>
      <td>0.5712</td>
      <td>[(6, 0.57116574), (10, 0.211909), (11, 0.20494...</td>
    </tr>
  </tbody>
</table>
</div>



## 3. 잠재 디리클레 할당(LDA) 실습2

앞서 gensim을 통해서 LDA를 수행하고, 시각화를 진행해보았습니다. 이번에는 LSA 챕터에서처럼 사이킷런(sklearn)을 사용하여 LDA를 수행하여 보겠습니다.

### 1) 실습을 통한 이해

#### (1) 뉴스 기사 제목 데이터에 대한 이해

여기서 사용할 데이터는 약 15년 동안 발행되었던 뉴스 기사 제목을 모아놓은 영어 데이터입니다.


```python
import pandas as pd
data = pd.read_csv(r'abcnews-date-text.csv', error_bad_lines=False)
```


```python
print(len(data))
```

    1103663
    

해당 데이터는 약 100만개의 샘플을 갖고 있습니다. 상위 5개의 샘플만 출력해봅시다.


```python
print(data.head(5))
```

       publish_date                                      headline_text
    0      20030219  aba decides against community broadcasting lic...
    1      20030219     act fire witnesses must be aware of defamation
    2      20030219     a g calls for infrastructure protection summit
    3      20030219           air nz staff in aust strike for pay rise
    4      20030219      air nz strike to affect australian travellers
    

각각 뉴스가 나온 날짜와 뉴스 기사 제목을 의미합니다.

필요한 데이터는 이 중에서 headline_text 열. 즉, 뉴스 기사 제목이므로 이 부분만 별도로 저장합니다.


```python
text = data[['headline_text']]
text.head(5)
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
      <th>headline_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>aba decides against community broadcasting lic...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>act fire witnesses must be aware of defamation</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a g calls for infrastructure protection summit</td>
    </tr>
    <tr>
      <th>3</th>
      <td>air nz staff in aust strike for pay rise</td>
    </tr>
    <tr>
      <th>4</th>
      <td>air nz strike to affect australian travellers</td>
    </tr>
  </tbody>
</table>
</div>



이제 뉴스 기사 제목만 저장이 된 데이터를 가지고 텍스트 전처리를 수행해보겠습니다.

#### (2) 텍스트 전처리

토큰화, 불용어 제거, 표제어 추출이라는 전처리를 사용해보겠습니다.


```python
import nltk
text['headline_text'] = text.apply(lambda row: nltk.word_tokenize(row['headline_text']), axis=1)
```

    C:\Users\joon2\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      
    

NLTK의 word_tokenize를 통해 단어 토큰화를 수행합니다.


```python
print(text.head(5))
```

                                           headline_text
    0  [aba, decides, against, community, broadcastin...
    1  [act, fire, witnesses, must, be, aware, of, de...
    2  [a, g, calls, for, infrastructure, protection,...
    3  [air, nz, staff, in, aust, strike, for, pay, r...
    4  [air, nz, strike, to, affect, australian, trav...
    

상위 5개의 샘플만 출력하여 단어 토큰화 결과를 확인합니다. 이제 불용어를 제거합니다.


```python
from nltk.corpus import stopwords
stop = stopwords.words('english')
text['headline_text'] = text['headline_text'].apply(lambda x: [word for word in x if word not in (stop)])
```

    C:\Users\joon2\Anaconda3\lib\site-packages\ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    


```python
print(text.head(5))
```

                                           headline_text
    0   [aba, decides, community, broadcasting, licence]
    1    [act, fire, witnesses, must, aware, defamation]
    2     [g, calls, infrastructure, protection, summit]
    3          [air, nz, staff, aust, strike, pay, rise]
    4  [air, nz, strike, affect, australian, travellers]
    

against, be, of, a, in, to 등의 단어가 제거되었습니다.

이제 표제어 추출을 수행합니다. 표제어 추출로 3인칭 단수 표현을 1인칭으로 바꾸고, 과거 현재형 동사를 현재형으로 바꿉니다.


```python
from nltk.stem import WordNetLemmatizer
text['headline_text'] = text['headline_text'].apply(lambda x: [WordNetLemmatizer().lemmatize(word, pos='v') for word in x])
print(text.head(5))
```

                                           headline_text
    0       [aba, decide, community, broadcast, licence]
    1      [act, fire, witness, must, aware, defamation]
    2      [g, call, infrastructure, protection, summit]
    3          [air, nz, staff, aust, strike, pay, rise]
    4  [air, nz, strike, affect, australian, travellers]
    

    C:\Users\joon2\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      
    

표제어 추출이 된 것을 확인할 수 있습니다.

이제 길이가 3이하인 단어에 대해서 제거하는 작업을 수행합니다. 그리고 이번에는 결과를 tokenized_doc이라는 새로운 변수에 저장합니다.


```python
tokenized_doc = text['headline_text'].apply(lambda x: [word for word in x if len(word) > 3])
print(tokenized_doc[:5])
```

    0       [decide, community, broadcast, licence]
    1      [fire, witness, must, aware, defamation]
    2    [call, infrastructure, protection, summit]
    3                   [staff, aust, strike, rise]
    4      [strike, affect, australian, travellers]
    Name: headline_text, dtype: object
    

길이가 3이하인 단어들에 대해서 제거가 된 것을 볼 수 있습니다.

이제 TF-IDF 행렬을 만들어보겠습니다.

#### (3) TF-IDF 행렬 만들기

TfidfVectorizer(TF-IDF 챕터 참고)는 기본적으로 토큰화가 되어있지 않은 텍스트 데이터를 입력으로 사용합니다.

역토큰화(Detokenization)작업을 수행해보겠습니다.


```python
# 역토큰화 (토큰화 작업을 되돌림)
detokenized_doc = []
for i in range(len(text)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)

text['headline_text'] = detokenized_doc # 다시 text['headline_text']에 재저장
```

    C:\Users\joon2\Anaconda3\lib\site-packages\ipykernel_launcher.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      import sys
    

역토큰화가 되었는지 text['headline_text']의 5개의 샘플을 출력해보겠습니다.


```python
text['headline_text'][:5]
```




    0       decide community broadcast licence
    1       fire witness must aware defamation
    2    call infrastructure protection summit
    3                   staff aust strike rise
    4      strike affect australian travellers
    Name: headline_text, dtype: object



정상적으로 역토큰화가 수행되었음을 확인할 수 있습니다

이제 사이킷런의 TfidfVectorizer를 TF-IDF 행렬을 만들 것입니다. 텍스트 데이터에 있는 모든 단어를 가지고 행렬을 만들 수도 있겠지만, 여기서는 간단히 1,000개의 단어로 제한하겠습니다.


```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', 
max_features= 1000) # 상위 1,000개의 단어를 보존 
X = vectorizer.fit_transform(text['headline_text'])
X.shape # TF-IDF 행렬의 크기 확인
```




    (1103663, 1000)



1,103,663 × 1,000의 크기를 가진 가진 TF-IDF 행렬이 생겼습니다.

이제 이에 LDA를 수행합니다.

#### (4) 토픽 모델링


```python
from sklearn.decomposition import LatentDirichletAllocation
lda_model=LatentDirichletAllocation(n_components=10,learning_method='online',random_state=777,max_iter=1)
```


```python
lda_top=lda_model.fit_transform(X)
```


```python
print(lda_model.components_)
print(lda_model.components_.shape) 
```

    [[1.00000703e-01 1.00000829e-01 1.00003578e-01 ... 1.00004871e-01
      1.00003129e-01 1.00002930e-01]
     [1.00001421e-01 8.66862951e+02 1.00008903e-01 ... 1.00004224e-01
      1.00005598e-01 7.01841034e+02]
     [1.00000648e-01 1.00000545e-01 1.00002661e-01 ... 1.00005158e-01
      1.00008596e-01 1.00001987e-01]
     ...
     [1.00001636e-01 1.00000889e-01 2.68570402e+03 ... 1.00003039e-01
      1.00010511e-01 1.00004475e-01]
     [1.00001352e-01 1.00000852e-01 1.00003353e-01 ... 1.00003378e-01
      1.00005211e-01 1.00003635e-01]
     [1.00002244e-01 1.00000967e-01 1.00003675e-01 ... 1.00002444e-01
      1.00003580e-01 1.00004738e-01]]
    (10, 1000)
    


```python
terms = vectorizer.get_feature_names() # 단어 집합. 1,000개의 단어가 저장됨.

def get_topics(components, feature_names, n=5):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (idx+1), [(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n - 1:-1]])
get_topics(lda_model.components_,terms)
```

    Topic 1: [('government', 8658.95), ('queensland', 8134.58), ('perth', 6332.45), ('year', 5981.93), ('change', 5833.07)]
    Topic 2: [('world', 7026.33), ('house', 6217.97), ('donald', 5757.52), ('open', 5620.39), ('years', 5563.76)]
    Topic 3: [('police', 12140.34), ('kill', 6091.65), ('interview', 5921.12), ('live', 5657.67), ('rise', 4162.16)]
    Topic 4: [('court', 6173.46), ('crash', 5497.33), ('state', 4857.9), ('tasmania', 4443.89), ('accuse', 4300.92)]
    Topic 5: [('australia', 13994.07), ('south', 6253.18), ('woman', 5614.31), ('coast', 5465.23), ('warn', 5155.11)]
    Topic 6: [('charge', 8440.62), ('election', 7650.47), ('adelaide', 6839.75), ('murder', 6418.61), ('make', 6198.2)]
    Topic 7: [('help', 5372.6), ('miss', 4601.06), ('people', 4561.71), ('2016', 4212.58), ('family', 4149.3)]
    Topic 8: [('sydney', 8597.95), ('melbourne', 7603.52), ('canberra', 6285.91), ('plan', 5606.37), ('power', 4198.99)]
    Topic 9: [('attack', 6818.74), ('market', 5094.55), ('council', 3854.2), ('share', 3811.79), ('national', 3793.87)]
    Topic 10: [('trump', 13043.55), ('australian', 11389.92), ('north', 6241.08), ('school', 5726.25), ('report', 5560.48)]
    
