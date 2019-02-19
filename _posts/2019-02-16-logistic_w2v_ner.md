---
title: Word2Vec 과 Logistic Regression 을 이용한 (Semi-supervised) Named Entity Recognition
date: 2019-02-16 19:00:00
categories:
- nlp
tags:
- ner
---

Named Entity Recognition 을 위하여 Conditional Random Field (CRF) 나 Recurrent Neural Network (RNN) 과 같은 sequential labeling 이 이용될 수 있습니다. 하지만 Richard Socher 의 강의노트에서 window classification 만으로도 가능하다는 내용이 있습니다. 또한 sequential labeling 알고리즘은 잘 구축된 학습 데이터가 필요하다는 단점도 있습니다. 이번 포스트에서는 학습 데이터셋이 전혀 없는 상황에서 한국어 Named Entity Recognizer 를 만드는 과정을 정리합니다. 이를 위하여 Word2Vec 으로 최소한의 seed set 을 구축하고, logistic regression 을 이용하여 window classification 을 하는 알고리즘을 만듭니다. 

## Named Entity Recognition

Named Entity Recognition (NER) 은 문장에서 특정한 종류의 단어를 찾아내는 information extraction 문제 중 하나입니다. '디카프리오가 나온 영화 틀어줘'라는 문장에서 '디카프리오'를 사람으로 인식하는 것을 목표로 합니다. 단어열로 표현된 문장에 각 단어의 종류를 인식하는 sequential labeling 방법이 주로 이용되었습니다. 최근에는 LSTM-CRF 와 같은 Recurrent Neural Network 계열 방법도 이용되지만, 오래전부터 Conditional Random Field (CRF) 가 이용되었습니다. 특히 CRF 모델은 named entities 를 판별하는 규칙을 해석할 수 있다는 점에서 유용합니다. 

Sequential labeling 은 pos tagging 에 이용되는 알고리즘이기도 합니다. 주어진 형태소 열에서 각 형태소의 품사를 추정하는 것이 품사 판별이라면, 각 단어의 class 를 추정하는 것이 named entity recognition 입니다. 목적에 따라 tag set 의 크기가 pos tagging 보다 클 수도, 작을 수도 있습니다.

```
Word sequence : [디카프리오, 가, 나온, 영화, 틀어줘]
Tag sequence  : [명사, 조사, 동사, 명사, 동사]
NER tagging   : [People, X, X, X, Request]
```

CoNLL 의 shared task 로 CoNLL 2002, CoNLL 2003 에서 스페인어, 네델란드어, 영어, 독일어에 대한 NER dataset 이 공개되기도 했습니다. CRF 를 이용하여 CoNLL 2002 작업을 하는 내용은 [이전의 포스트][crfner]를 살펴보시기 바랍니다.

Named entity recognition 은 챗봇에서 이용자의 의도를 판단하는 intention classificaion 의 주요한 features 이기도 합니다. 그렇기 때문에 최근까지도 여전히 중요한 문제입니다. 하지만 앞서 언급한 sequential labeling algorithm 은 잘 구축된 학습데이터가 필요합니다. 그러나 우리가 이용할 named entity recognition task 는 학습할 데이터가 없습니다. 챗봇의 intention classification 용 NER tagger 와 영화 추천 시스템의 NER tagger 는 서로 다른 학습데이터를 이용합니다.

물론 학습 데이터를 잘 구축하면 높은 학습 능력을 지닌 모델들을 이용할 수 있습니다. 하지만 일단 학습 데이터를 잘 만들어야 합니다. 그런데 모든 데이터가 학습에 적합한 것도 아닐 것입니다. 우리는 간단한 partially positive labeled dataset 을 만들고, 해당 raw text 를 NER 용 데이터를 구축하는데 쓸만한지 확인하는 과정도 살펴봅니다.

[crfner]: {{ site.baseurl }}{% link _posts/2018-06-22-crf_based_ner.md %}


## Window classification for Named Entity Recognition

Named Entity Recogntion 은 위의 예시처럼 sequential labeling 을 이용할 수도 있습니다. 하지만 문장 전체가 아닌, named entity 주변의 정보만을 이용할 수도 있습니다. 영화 도메인에서는 `[People] + 가 + 나온` 이라는 표현에서 `가` 앞에 출연 배우가 위치하기 때문입니다. Named entity 를 기술하는 정보는 앞/뒤에 등장하는 몇 개의 단어만으로도 충분합니다.

```
Word sequence : [디카프리오, 가, 나온]
Target word   : 디카프리오 / People
Features      : X[1] = '가' & X[2] = '나온'
```

[이전의 포스트][crfner]를 살펴보면 실제로 CRF 가 학습하는 정보도 앞/뒤에 등장하는 단어입니다. 문장 내에서 멀리 떨어진 단어 간에 상관성이 없다면 labeling 작업에 이를 굳이 이용하지 않아도 괜찮습니다. [Richard Socher 의 강의노트]에서도 neural network 를 이용하여 named entity recognition 용 window classfier 를 만드는 내용이 나오기도 합니다.

또한 window classification 으로부터 얻을 수 있는 결과물 중 하나는 templates 입니다. `[People] + 가 + 나온` 이라는 pattern 을 얻을 수 있습니다. Logistic regression 나 softmax 를 이용하는 neural network 를 이용한다면 template 의 score 까지도 얻을 수 있습니다. 반대로 추출된 templates 을 확인함으로써, NER model 을 학습하기에 좋은 데이터인지 확인할 수 있다는 장점도 있습니다.

이번 포스트에서는 학습데이터를 구축하는 과정을 간소화하고, NER 을 할 수 있는 데이터인지 살펴보기 위하여 logistic regression 을 이용한 window classification model 을 만들어 봅니다.

[socher]: https://nlp.stanford.edu/~socherr/pa4_ner.pdf

## Dataset

데이터셋은 [이전 데이터셋 포스트][dataset]에 설명한 `lovit_textmining_dataset` 을 이용합니다. 영화평 데이터는 (영화 id, 영화 평, 영화 평점) 의 3 columns 으로 이뤄진 파일이므로 split 을 한 뒤, text 만 yield 합니다. 토크나이징은 완료되었다 가정하여 띄어쓰기 기준으로 단어들을 나눈 형태로 yield 를 합니다.

```python
from navermovie_comments import get_movie_comments_path

class Comments:
    def __init__(self, path):
        self.path = path
    def __iter__(self):
        with open(self.path, encoding='utf-8') as f:
            for i, doc in enumerate(f):
                idx, text, rate = doc.split('\t')
                yield text.split()

path = get_movie_comments_path(large=True, tokenize='soynlp_unsup')
comments = Comments(path)
```

Comments 의 세 문장의 예시입니다.

```
['명불허전']
['왠지', '고사', '피의', '중간', '고사', '보다', '재미', '가', '없을듯', '해요', '만약', '보게', '된다면', '실망', '할듯']
['티아라', '사랑', '해', 'ㅜ']
```

[dataset]: {{ site.baseurl }}{% link _posts/2018-06-22-textmining_dataset.md %}

## Vocabulary scan

Dataset 에 등장한 모든 단어를 이용할 수는 없습니다. min count 이상 등장한 단어만 학습에 이용합니다. `scan_vocabulary` 함수는 단어의 빈도수를 계산한 뒤, 빈도수의 역순으로 단어를 index 로 바꿉니다. `idx_to_vocab` 에는 index 별 단어가 포함되어 있으며, `vocab_to_idx` 는 {str:int} 형식의 indexer 입니다.

```python
from collections import defaultdict

def scan_vocabulary(sents, min_count, verbose=False):
    counter = defaultdict(int)
    for i, sent in enumerate(sents):
        if verbose and i % 100000 == 0:
            print('\rscanning vocabulary .. from %d sents' % i, end='')
        for word in sent:
            counter[word] += 1
    counter = {word:count for word, count in counter.items()
               if count >= min_count}
    idx_to_vocab = [vocab for vocab in sorted(counter,
                    key=lambda x:-counter[x])]
    vocab_to_idx = {vocab:idx for idx, vocab in enumerate(idx_to_vocab)}
    idx_to_count = [counter[vocab] for vocab in idx_to_vocab]
    if verbose:
        print('\rscanning vocabulary was done. %d terms from %d sents' % (len(idx_to_vocab), i+1))
    return vocab_to_idx, idx_to_vocab, idx_to_count
```

`min_count = 10` 으로 `scan_vocabulary` 함수를 실행시켜 모델링에 이용할 단어를 선택합니다.

```python
vocab_to_idx, idx_to_vocab, idx_to_count = scan_vocabulary(
    comments, min_count=10, verbose=True)
```

3,280,685 개의 문장으로부터 69,541 개의 단어가 선택되었습니다.

```
scanning vocabulary was done. 69541 terms from 3280685 sents
```

`idx_to_vocab` 과 `idx_to_count` 를 살펴봅니다. `영화` 라는 단어는 총 1,128,809 번 등장하였습니다.

```python
print(idx_to_vocab[:5]) # ['영화', '이', '관람', '객', '의']
print(idx_to_count[:5]) # [1128809, 866305, 600351, 526070, 489950]
```

## Features

우리는 window = 2 를 이용하여 한 단어 X[0] 의 앞, 뒤 각각 두 개의 단어 (총 4개의 단어)를 X[0] 의 features 로 이용합니다. 예를 들어 (a, b, c, d, e) 라는 단어가 등장하였고, X[0]=c 라면 X[-2]=a, X[-1]=b, X[1]=d, X[2]=e 입니다.

그리고 scan vocabulary 함수를 통하여 학습된 단어가 총 5 개라면 이들의 위치를 보존하면 X[0] 에 대한 feature space 를 20 차원으로 만들 수 있습니다. 만약 각 단어의 index 가 {a:0, b:1, c:2, d:3, e:4} 라면 X[0]=c 는 [0, 5+1, 10+3, 15+4] 를 features 로 가진다 표현할 수 있습니다.

`feature_to_idx` 는 이를 만드는 함수입니다. 문장 내에서 현재 단어 X[0] 의 위치를 i, 현재 단어 앞, 뒤 단어인 X[-1] 이나 X[1] 의 위치를 j, j 위치의 단어의 index 를 vocab_idx 라 할 때, 이 값의 feature index 를 출력합니다.

```python
def feature_to_idx(i, j, vocab_idx, window, n_terms):
    if j < i:
        return n_terms * (j - i + window) + vocab_idx
    else:
        return n_terms * (j - i + window - 1) + vocab_idx

feature_to_idx(i=2, j=3, vocab_idx=3, window=2, n_terms=5) # 13
```

`idx_to_feature` 는 반대로 feature index 를 feature 로 decode 합니다. feature idx 를 vocabulary 의 개수로 나눈 몫은 상대적 위치값이 되고, 나머지는 vocabulary idx 입니다.


```python
def idx_to_feature(feature_idx, idx_to_vocab, window):
    # 몫
    position = feature_idx // len(idx_to_vocab)
    if position < window:
        feature = 'X[-%d] = ' % (window - position)
    else:
        feature = 'X[%d] = ' % (position - window + 1)
    # 나머지
    vocab_idx = feature_idx % len(idx_to_vocab)
    feature += idx_to_vocab[vocab_idx]
    return feature

idx_to_feature(13, 'a b c d e'.split(), window=2) # 'X[1] = d'
```

이를 이용하여 학습데이터로부터 window classification 용 데이터를 만듭니다. `create_window_cooccurrence_matrix` 함수는 X[0] 을 기준으로 X[-2], X[-1], X[1], X[2] 의 co-occurrence 를 계산하는 matrix 를 만듭니다. Sparse matrix 형식이기 때문에 rows, columns 를 따로 모읍니다. words 는 각 rows 에 해당하는 단어를 넣어둡니다.

`create_window_cooccurrence_matrix` 함수에서 scan vocabulary 의 결과에 포함되지 않은 단어는 건너 띄며, context words 의 범위는 문장의 맨 앞에서 문장의 맨 뒷 단어가 되도록 index 의 범위를 확인합니다.

```python
for i, word in enumerate(sent):
    if not (word in vocab_to_idx):
        continue

    b = max(0, i - window)
    e = min(i + window, n_words)
```

아래 구문을 통하여 sent[j] 의 단어 역시 scan vocabulary 의 결과에 포함되지 않으면 이를 이용하지 않습니다.

```python
for j in range(b, e):
    if i == j:
        continue
    j_idx = vocab_to_idx.get(sent[j], -1)
    if j_idx == -1:
        continue
```

위 내용이 포함된 `create_window_cooccurrence_matrix` 함수입니다.

```python
import numpy as np
from scipy.sparse import csr_matrix

def create_window_cooccurrence_matrix(vocab_to_idx, sentences, window=2, verbose=True):

    n_terms = len(vocab_to_idx)

    rows = []
    cols = []
    words = []

    row_idx = 0
    col_idx = window * 2 * n_terms

    for i_sent, sent in enumerate(sentences):

        if verbose and i_sent % 10000 == 0:
            print('\rcreating train dataset {} rows from {} sents'.format(row_idx, i_sent), end='')

        n_words = len(sent)

        for i, word in enumerate(sent):
            if not (word in vocab_to_idx):
                continue

            b = max(0, i - window)
            e = min(i + window, n_words)

            features = []
            for j in range(b, e):
                if i == j:
                    continue
                j_idx = vocab_to_idx.get(sent[j], -1)
                if j_idx == -1:
                    continue
                features.append(feature_to_idx(i, j, j_idx, window, n_terms))

            if not features:
                continue

            # sparse matrix element
            for col in features:
                rows.append(row_idx)
                cols.append(col)

            # words element
            words.append(word)

            row_idx += 1

    if verbose:
        print('\rtrain dataset {} rows from {} sents was created    '.format(row_idx, i_sent))

    # to csr matrix
    rows = np.asarray(rows, dtype=np.int)
    cols = np.asarray(cols, dtype=np.int)
    data = np.ones(rows.shape[0], dtype=np.int)
    X = csr_matrix((data, (rows, cols)), shape=(row_idx, col_idx))

    return X, words
```

이를 이용하여 co-occurrence matrix 와 각 rows 에 해당하는 단어 리스트를 학습합니다.

```python
window = 2

X, words = create_window_cooccurrence_matrix(
    vocab_to_idx, comments, window)
```

만들어진 데이터는 row 의 크기가 42,981,576 입니다. 문장으로부터 snapshot 을 만들었기 때문에 그 개수가 매우 커집니다. 그리고 feature size 는 278,164 입니다. 이는 단어 개수 69,541 의 4 배 ($$2 \times window$$) 입니다. Scikit-learn 의 logistic regression 을 이용하기 위하여 메모리에 데이터를 모두 올렸을 뿐, minibatch 형식으로 구현한다면 메모리를 절약할 수 있습니다.

```python
print(X.shape) # (42981576, 278164)
```

## Word2Vec 을 이용한 seed set 만들기

Gensim 으로 미리 학습해둔 Word2Vec model 을 로딩합니다. 우리는 사람 이름을 인식하는 named entity recognizer 를 만들겁니다. 학습된 Word2Vec model 역시 앞서 소개한 textmining dataset 에 올려뒀습니다. `송강호`와 `디카프리오`의 Word2Vec 유사어는 사람 이름임을 알 수 있습니다. 각각 100 개씩의 유사어를 선택하여 이의 합집합을 seed_words 로 선택합니다. 총 172 개의 단어가 seeds 로 선택되었습니다.

```python
from navermovie_comments import load_trained_embedding

word2vec = load_trained_embedding(tokenize='soynlp_unsup')

seed_words = {word for word, _ in word2vec.most_similar('송강호', topn=100)}
seed_words.update({word for word, _ in word2vec.most_similar('디카프리오', topn=100)})

print(len(seed_words)) # 172
```

토크나이저에 따라서 `안성기 + 씨` 자체가 단어로 인식되어 `송강호` 의 유사어로 학습되기도 했습니다. 대부분이 배우 이름임을 확인할 수 있습니다.

`송강호`의 Word2Vec 기준 유사한 단어들

| 하정우 (0.908) | 조진웅 (0.797) | 김민희 (0.765) | 송중기 (0.738) | 이경영 (0.721) |
| 한석규 (0.882) | 조정석 (0.797) | 정우성 (0.764) | 라미란 (0.736) | 조재현 (0.720) |
| 오달수 (0.856) | 안성기씨 (0.794) | 김정태 (0.763) | 배두나 (0.736) | 이범수씨 (0.719) |
| 이정재 (0.855) | 류승룡 (0.793) | 브래드피트 (0.759) | 정진영 (0.733) | 강동원 (0.718) |
| 김명민 (0.846) | 정재영씨 (0.790) | 류승범 (0.758) | 권상우 (0.732) | 박철민씨 (0.717) |
| 이범수 (0.842) | 진구 (0.789) | 심은경 (0.755) | 차태현 (0.732) | 박유천 (0.716) |
| 설경구 (0.842) | 손예진 (0.786) | 이선균 (0.753) | 엄태구 (0.732) | 송새벽 (0.716) |
| 황정민 (0.838) | 이하늬 (0.782) | 김태리 (0.753) | 유아인 (0.731) | 김옥빈 (0.714) |
| 손현주 (0.837) | 이제훈 (0.782) | 임지연 (0.750) | 김해숙씨 (0.731) | 차인표 (0.714) |
| 김윤석 (0.833) | 감우성 (0.782) | 박소담 (0.750) | 문소리 (0.730) | 앤해서웨이 (0.713) |
| 유해진 (0.830) | 정재영 (0.781) | 김윤식 (0.749) | 김남길 (0.730) | 조인성 (0.713) |
| 주진모 (0.828) | 박신양 (0.778) | 박해일 (0.748) | 차승원 (0.729) | 한예리 (0.711) |
| 공유 (0.816) | 고수 (0.777) | 라미란씨 (0.748) | 톰크루즈 (0.728) | 박희순씨 (0.709) |
| 이병헌 (0.812) | 윌스미스 (0.777) | 전지현 (0.747) | 서교 (0.726) | 앤헤서웨이 (0.708) |
| 문정희 (0.811) | 마동석 (0.776) | 김인권 (0.746) | 박희순 (0.726) | 유혜진 (0.707) |
| 정우 (0.809) | 곽도원 (0.776) | 임달화 (0.745) | 박시후 (0.726) | 안소희 (0.706) |
| 최민식 (0.808) | 김혜수 (0.772) | 박성웅 (0.744) | 신하균 (0.725) | 주지훈 (0.705) |
| 안성기 (0.808) | 박신혜 (0.770) | 김인권씨 (0.744) | 하지원 (0.725) | 이민기 (0.702) |
| 김윤진 (0.805) | 한석규씨 (0.770) | 장윤주 (0.742) | 송지효 (0.724) | 류승용 (0.701) |
| 성동일 (0.798) | 김원해 (0.766) | 박중훈 (0.741) | 이병현 (0.723) | 신세경 (0.701) |

`레저`는 `히스 레저` (배트맨 다크나이트의 조커 역), `틸다` 는 `틸다 스윈튼` (설국열차의 메이슨 역) 입니다. 한국인의 이름은 unigram 으로 표현되는 경우가 많으나, 외국인의 이름은 bigram, trigram 으로 표현되어 띄어쓰기가 포함되는 경우들이 있습니다.

`디카프리오`의 Word2Vec 기준 유사한 단어들

| 레오 (0.840) | 레저 (0.702) | 권상우 (0.680) | 정우 (0.664) | 케이트 (0.653) |
| 톰하디 (0.830) | 마크러팔로 (0.698) | 한석규 (0.679) | 동원오빠 (0.664) | 드니로 (0.652) |
| 앤해서웨이 (0.775) | 숙희 (0.697) | 틸다 (0.678) | 이준기 (0.664) | 이중구 (0.652) |
| 앤헤서웨이 (0.764) | 아놀드 (0.696) | 천우희 (0.676) | 하녀 (0.664) | 김인권씨 (0.652) |
| 브래드피트 (0.750) | 베니 (0.696) | 이범수 (0.675) | 슈왈제네거 (0.663) | 마고로비 (0.652) |
| 로다주 (0.749) | 안성기씨 (0.694) | 공유 (0.675) | 주진모 (0.662) | 톰아저씨 (0.652) |
| 로버트드니로 (0.749) | 컴버배치 (0.692) | 히스 (0.673) | 김범수 (0.662) | 강혜정 (0.651) |
| 콜린퍼스 (0.737) | 조커 (0.692) | 자베르 (0.673) | 홀트 (0.661) | 임달화 (0.651) |
| 히스레저 (0.733) | 정진영씨 (0.691) | 유코 (0.671) | 김태리 (0.661) | 벤 (0.651) |
| 윌스미스 (0.730) | 러셀크로우 (0.691) | 진구 (0.671) | 김해숙씨 (0.660) | 하시모토 (0.651) |
| 니콜라스홀트 (0.730) | 김혜수 (0.690) | 엄태구 (0.671) | 샤를리즈 (0.660) | 기럭지 (0.650) |
| 안성기 (0.727) | 고수 (0.689) | 아저씨 (0.670) | 다니엘 (0.659) | 신하균 (0.649) |
| 히스레져 (0.723) | 레이놀즈 (0.689) | 배두나 (0.670) | 토니스타크 (0.658) | 스미스 (0.648) |
| 레오나르도 (0.716) | 샤오위 (0.689) | 태리 (0.668) | 퓨리오사 (0.658) | 맥스 (0.648) |
| 에디 (0.713) | 휴잭맨 (0.687) | 하쿠 (0.667) | 에드워드 (0.657) | 레토 (0.648) |
| 피트 (0.710) | 윈슬렛 (0.687) | 브래드 (0.667) | 김민희 (0.656) | 주걸륜 (0.647) |
| 베네딕트 (0.710) | 이정재 (0.683) | 중기 (0.666) | 해서웨이 (0.655) | 아가씨 (0.647) |
| 시저 (0.708) | 콜린 (0.682) | 히데코 (0.666) | 테론 (0.655) | 미모 (0.647) |
| 톰크루즈 (0.705) | 감우성 (0.681) | 로버트다우니주니어 (0.665) | 해리 (0.655) | 효진이 (0.647) |
| 멧데이먼 (0.704) | 정진영 (0.680) | 송중기 (0.665) | 스파이디 (0.654) | 죠니뎁 (0.646) |

## Word2Vec 유사어를 이용하여 label vector 만들기

앞서 만든 X 의 row 에 해당하는 단어가 seed_words 에 포함될 경우, 이 rows 의 값을 1 로, 그렇지 않은 경우 0 으로 지정합니다.

172 개의 단어가 361,394 번 등장하였습니다.

```python
y = np.zeros(X.shape[0], dtype=np.int)
for i, word in enumerate(words):
    if word in seed_words:
        y[i] = 1

y.sum() # 361394
```

이 데이터는 partially positive labeled imbalanced data 입니다. Negative 로 레이블링 된 데이터는 실제로 negative 일 경우도 있지만, positive 가 잘못 레이블링 된 경우도 있습니다. 그리고 positive 의 비율이 0.841 % (= 361394 / 42981576)밖에 되지 않습니다. 극심한 imbalanced data 입니다.

## Logistic Regression 을 이용한 window classifier 만들기

Logistic regression 을 학습합니다. seed words 를 positive class 로 예측하는 모델을 만듭니다. 그런데 실제로는 사람 이름이면서도 label 을 0 으로 가지는 데이터도 존재합니다. Logistic regression 은 이들에 대해서는 큰 확률값을 지닐 가능성이 높습니다. Training error 를 named entity 의 힌트로 이용하는 것입니다. 이 때의 training error 는 우리가 seed words 를 이용하여 엉성하게 데이터를 준비했기 때문에 발생하는 error 이기 때문입니다.

```python
from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression()
logistic.fit(X, y)
y_pred = logistic.predict(X)
y_prob = logistic.predict_proba(X)[:,1]
```

모델은 softmax probability 가 0 에 가까운 값이면 일단은 negative class 로 분류하여 loss 가 작을 것입니다. 그리고 사람이 등장하는 문맥에서 등장하는 negative class 의 softmax probability 를 지나치게 줄이려 하면 positive class 의 확률값이 매우 작게 되기 때문에 negative class 이면서 사람인 단어들에 대해서는 0 보다는 크되, 매우 작은 확률을 부여합니다. 이 점을 이용하여 prediction probability 가 0.05 보다 큰 snapshot (row) 들을 `pred_pos` 에 카운팅 합니다.

이후 해당 단어가 등장한 횟수로 0.05 보다 큰 prediction probability 를 받은 횟수를 나눠 named entity score 를 계산합니다.

예를 들어 배우 `백윤식` 은 seed words 에 포함되지 않았지만 총 100 번 등장하였고, 그 중 95 번을 0.05 보다 큰 prediction probability 를 받았다면, 이 단어의 named entity score 는 0.95 가 됩니다.

```python
from collections import Counter

# word count
word_counter = Counter(words)

# prediction count
pred_pos = defaultdict(int)
for row in np.where(y_prob >= 0.05)[0]:
    pred_pos[words[row]] += 1
pred_pos = {word:pos/word_counter[word] for word, pos in pred_pos.items()}
```

## 결과 확인하기

Named entity score 가 큰 순서대로 상위 1000 개의 단어를 선택합니다. 그 중 seed words 에 포함된 단어는 출력하지 않습니다. (단어, 빈도수), score 를 확인합니다. 실제로 `백윤식` 은 293 번 등장했으며, 약 222 번 0.05 보다 큰 probability 를 얻었습니다.

`앤 헤서웨이`의 경우 다양한 오탈자들이 존재합니다. 또 대부분은 앞의 이름을 붙여서 `앤헤서웨이` 로 쓰는 경우가 많기 때문에 다양한 `헤서웨이` 들이 등장합니다. 그리고 그 빈도수가 작은 (최소 빈도수 10 으로 학습) 경우에도 사람 이름으로 인식됨을 볼 수 있습니다.

```python
for word, prob in sorted(pred_pos.items(), key=lambda x:-x[1])[:1000]:
    if word in seed_words:
        continue
    idx = vocab_to_idx[word]
    count = idx_to_count[idx]
    # print ...
```

| 해서워이 (10, 1.000) | 그브가 (12, 1.000) | 장현성 (11, 1.000) | 왕이고싶었고 (26, 0.962) | 공지영작가 (13, 0.923) |
| 신정근 (11, 0.909) | 달화 (10, 0.900) | 헤더웨이 (10, 0.900) | 전국환 (10, 0.900) | 헤서웨이 (261, 0.893) |
| 틸타 (10, 0.889) | 박원상 (23, 0.870) | 패틴슨 (286, 0.864) | 천의 (42, 0.857) | 와저 (188, 0.849) |
| 동해물 (46, 0.848) | 곽동원 (12, 0.833) | 류승수 (12, 0.833) | 레져 (78, 0.808) | 고슬링 (195, 0.805) |
| 참바다 (81, 0.802) | 김동욱씨 (10, 0.800) | 동명수 (15, 0.800) | 해써웨 (10, 0.800) | 진짫 (10, 0.800) |
| 헤스 (10, 0.800) | 김소담 (10, 0.800) | 마형 (15, 0.800) | 계두식 (10, 0.800) | 윤지혜 (26, 0.800) |
| 유이인 (15, 0.800) | 종석 (183, 0.796) | 하저우 (14, 0.786) | 임현식 (13, 0.769) | 전혜진씨 (13, 0.769) |
| 크루주 (17, 0.765) | 희순 (21, 0.762) | 백윤식 (293, 0.758) | 손현주아저씨 (37, 0.757) | ㅋㅋ이민기 (80, 0.750) |

...

| 볼드모트 (220, 0.447) | 달수 (131, 0.446) | 동원이형 (66, 0.446) | 미스봉 (65, 0.446) | 의발 (18, 0.444) |
| 해진씨 (18, 0.444) | 보영님 (28, 0.444) | 하우어 (18, 0.444) | 김환희 (54, 0.444) | 다니엘헤니 (27, 0.444) |
| 병헌씨 (36, 0.444) | 우성씨 (18, 0.444) | JB (18, 0.444) | 최진혁씨 (27, 0.444) | 윤진언니 (18, 0.444) |
| 리빙빙 (27, 0.444) | 우에노주리 (36, 0.444)


## Named Entity Filter (Feature) 확인하기

우리는 window instance 를 하나의 row 로 만들었기 때문에 prediction probability 가 높은 instance 를 확인하면, 어떤 context 에서 X[0] 가 사람 이름인지를 확인할 수 있습니다.

```python
top_instances = np.where(0.7 <= y_prob)[0]
top_probs = y_prob[top_instances]

print(top_instances.shape) # (32775,)
print(top_probs.shape) # (32775,)
```

32,775 개의 rows 중에는 중복된 것들도 많습니다. 중복된 경우를 정리하여 각 instance 와 count, 그리고 prediction probability 를 정리하는 함수를 만듭니다.

```python
from collections import defaultdict

def get_unique_top_instances(sample_idxs, probs):

    # slice samples
    X_samples = X[sample_idxs]
    rows, cols = X_samples.nonzero()

    # find unique instance
    instance_prob = {}
    instance_count = defaultdict(int)
    before_row = None

    def update_dict(features, prob):
        features = sorted(features, key=lambda x:x[0])
        instance = ', '.join(f[1] for f in features)
        instance_prob[instance] = prob
        instance_count[instance] += 1
        return []

    features = [] # temporal variable
    for row, feature_idx in zip(rows, cols):
        # update unique dictionary
        if row != before_row and features:
            features = update_dict(features, probs[row])
        # update temporal variable
        before_row = row
        feature = idx_to_feature(feature_idx, idx_to_vocab, window)
        features.append((feature_idx, feature))

    # last elements
    if features:
        update_dict(features, probs[row])

    return instance_prob, instance_count

instance_prob, instance_count = get_unique_top_instances(top_instances, top_probs)    
```

빈도수 기준으로 상위 500 개의 instance 를 출력합니다 (probability, count), instance 입니다.

뒤에 X[1] = '씨' 가 등장하는 경우가 가장 많았으며, 아래쪽에는 다음과 같은 표현도 있습니다. `믿고보는 송강호` 와 같은 전형적인 영화평 도메인에서의 표현입니다.

```
(0.8705, 16)	X[-2] = 역시, X[-1] = 믿고보는, X[1] = 과
```

아래의 표현은 `브래드`가 seed words 에 포함되었기 때문에 `브래드`라는 단어 뒤의 단어를 사람 이름으로 인식한 경우입니다.

```
(0.7439, 50)	X[1] = 피트
```

또한 영화평에서는 배우의 이름을 나열하는 경우들도 있습니다. 배우의 이름을 다 나열한 뒤, '이 캐스팅봐라' 식의 표현들이 있어서 배우 이름 역시 유의미한 context 로 선택됩니다.

```
(0.8263, 18)	X[-2] = 전지현, X[-1] = 하정우, X[1] = 조진웅
```

```
...
(0.7965, 56)	X[-2] = 게, X[-1] = 봤습니다, X[1] = 씨
(0.8211, 53)	X[-2] = 있, X[-1] = 었어요, X[1] = 씨
(0.7439, 50)	X[1] = 피트
(0.8999, 47)	X[-1] = 역시, X[1] = 님
(0.8373, 47)	X[-2] = 영화, X[-1] = 였습니다, X[1] = 씨
(0.7439, 47)	X[-1] = 믿고보는, X[1] = 하정우
(0.7819, 46)	X[-2] = 영화, X[-1] = 입니다, X[1] = 씨
(0.9073, 46)	X[-2] = 객, X[-1] = 믿고보는, X[1] = 황정민
(0.9318, 46)	X[1] = 레드메인
(0.7679, 45)	X[-2] = 역시, X[-1] = 믿고보는, X[1] = 황정민
(0.8713, 43)	X[-1] = 역시, X[1] = 형님
...
(0.7135, 18)	X[-2] = 전지현, X[-1] = 이정재, X[1] = 오달수
(0.8263, 18)	X[-2] = 전지현, X[-1] = 하정우, X[1] = 조진웅
(0.8267, 18)	X[-2] = 송강호, X[-1] = 랑, X[1] = 연기
(0.8037, 18)	X[-2] = 송강호, X[-1] = 씨와, X[1] = 씨의
...
(0.8379, 13)	X[-2] = 강동원, X[-1] = 이랑, X[1] = 때문
(0.7941, 13)	X[-2] = 들, X[-1] = 었습니다, X[1] = 씨의
...
(0.7006, 7)	X[-2] = 객, X[-1] = 믿고보는, X[1] = 연기력
```

그런데 아래의 templates 는 word embedding 입장에서는 비슷한 벡터를 지닐 것입니다. 앞, 뒤에 등장하는 features 들이 모두 사람 이름이기 때문입니다. N-gram 을 features 로 이용하는 Convolutional Neural Network 를 이용한다면 훨씬 효율적으로 features 를 학습할 가능성이 높습니다.

```
(0.7135, 18)	X[-2] = 전지현, X[-1] = 이정재, X[1] = 오달수
(0.8263, 18)	X[-2] = 전지현, X[-1] = 하정우, X[1] = 조진웅
```

또한 앞서 실험한 방법은 각 features 가 독립이라 가정하였기 때문에 unigram 으로 학습한 모델입니다. 이 실험으로 확인할 수 있는 점은 unigram 이어도 named entity 를 찾기에 충분한 정보를 얻을 수 있으며, 오히려 n-gram 을 이용한다면 더 정확한 문맥을 학습할 수도 있다는 점입니다. 또한 이 데이터를 이용하여 영화 리뷰 도메인에서 사람 이름을 인식하는 모델을 학습할 수 있다는 확인도 할 수 있습니다. 즉 이 방법은 데이터의 활용도에 대한 확인과 named entity recognition 의 base model 로 이용할 수 있습니다.

