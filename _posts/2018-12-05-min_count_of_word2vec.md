---
title: (Gensim) Word2Vec 의 최소 빈도수 설정
date: 2018-03-26 21:00:00
categories:
- nlp
- representation
tags:
- word representation
---

## Brief reviews of Word2Vec

Word2Vec 은 Softmax regression 을 이용하여 단어의 의미적 유사성을 보존하는 embedding space 를 학습합니다. 문맥이 유사한 (의미가 비슷한) 단어는 비슷한 벡터로 표현됩니다. 'cat' 과 'dog' 은 서로 비슷한 문맥에서 이용될 수 있기 때문에 비슷한 embedding vector 를 지닙니다.

Word2Vec 은 softmax regression 을 이용하여 문장의 한 스냅샷에서 기준 단어의 앞/뒤에 등장하는 다른 단어들 (context words) 이 기준 단어를 예측하도록 classifier 를 학습합니다. 그 과정에서 단어의 embedding vectors 가 학습됩니다. Context vector 는 앞/뒤 단어들의 평균 임베딩 벡터 입니다. [a, little, cat, sit, on, the, table] 문장에서 context words [a, little, sit, on] 를 이용하여 cat 을 예측합니다.

![]({{ "/assets/figures/word2vec_logistic_structure.png" | absolute_url }})

이는 cat 의 임베딩 벡터를 context words 의 평균 임베딩 벡터에 가깝도록 이동시키는 역할을 합니다. 비슷한 문맥을 지니는 dog 도 비슷한 context words 의 평균 임베딩 벡터에 가까워지기 때문에 cat 과 dog 의 벡터가 비슷해집니다. 

![]({{ "/assets/figures/word2vec_softmax.png" | absolute_url }})

그런데 Softmax regression 을 이용하면 infrequent words 에 대해서는 좋은 embedding vector 가 학습되기 어렵습니다. 한 context 에 여러 단어가 나올 경우 infreqeunt words 는 positive samples 로 선택되는 횟수가 적기 때문입니다. 그래서 저는 Word2Vec 으로 학습된 단어 공간은 단어의 빈도수에 영향을 받아 infrequent words 가 한 공간에 모여있다고 생각합니다.

![]({{ "/assets/figures/word2vec_odyssey_my_assumption.png" | absolute_url }})

이에 대한 자세한 설명은 [이전 포스트][oddysay]에 있습니다.


