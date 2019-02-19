---
title: 한국어 텍스트마이닝 실습용 데이터셋 (lovit textmining dataset) 과 실습 코드 (python ml4nlp)
date: 2019-06-22 15:00:00
categories:
- dataset
tags:
- dataset
---

최근에 한국어 텍스트 분석을 연습할 수 있는 데이터를 공유하고 있습니다. 데이터 분석 연습의 첫 코드가 분석이 아닌 데이터 수집이지 않기를 바래서 였습니다. 데이터셋의 구성과 앞으로 추가하려는 종류의 데이터, 그리고 연습 가능한 문제들을 살펴봅니다.

## Textmining dataset

성능 평가까지 가능한 공개된 데이터들은 머신러닝과 자연어처리 분야의 발전에 큰 기여를 하였습니다. 분명 컴퓨터 비전 분야의 경우에는 ImageNet 데이터를 이용한 competition 이 모델의 발전을 촉진했습니다. 특정 tasks 를 위한 labeled data 가 있다면 기존 알고리즘과 비교하며 새로운 알고리즘의 성능을 객관적으로 평가할 수 있습니다. NLP 에서도 [CoNLL][conll] 의 shared task dataset 이 공개되어 있으며, 질의 응답 분야도 SQuAD dataset 이 발전에 큰 기여를 하였습니다. 그리고 각 테스크 별로 잘 정의된 데이터셋이 공유되고 있습니다. 이런 정리들은 여러 사람들이 정리하여 [블로그][datasetblog]나 [github][datasetgit] 에 공유되고 있기도 합니다.

[conll]: http://www.conll.org/2019-shared-task
[datasetblog]: https://machinelearningmastery.com/datasets-natural-language-processing/
[datasetgit]: https://github.com/niderhoff/nlp-datasets

최근에는 한국어 질의 응답용 [KorQuAD][korquad] 데이터도 공유되었습니다. 이전에도 한국어 감성 분석 (sentiment analysis) 을 위하여 네이버 영화 리뷰를 정리해둔 [Naver sentiment movie corpus v1.0][nsmc] 이 공유되기도 했습니다.

[korquad]: https://korquad.github.io/
[nsmc]: https://github.com/e9t/nsmc

이처럼 객관적인 성능을 평가할 수 있는 데이터도 필요하지만, 데이터 분석 공부를 시작할 때에도 손쉽게 접근할 수 있는 데이터셋이 여러 종류가 있다면 좋을 것입니다. 영화 리뷰, 뉴스, 뉴스 댓글 등 우리가 접근할 수 있는 데이터들은 많지만, 이 데이터들은 우리가 수집을 해야만 합니다. 처음 데이터 분석을 시작하려 할 때, 분석 코드가 아닌 데이터 수집 코드부터 작성하는 모습을 종종 보곤 했습니다. 데이터 분석을 하고 싶다면 첫 코드가 데이터 분석이었으면 좋겠습니다.

이러한 목적에서 수집된 한글 텍스트 데이터를 공유하는 작업을 진행하고 있습니다. 그리고 이 데이터를 이용한 usages 도 공유하려 하고 있습니다. 데이터셋의 repository 는 [github.com/lovit/textmining_dataset][tmd] 입니다. 그리고 이를 이용하는 [textmining tutorial][tutorial] 을 [github.com/lovit/python_ml4nlp][tutorial] 에 올리고 있습니다.

[tmd]: https://github.com/lovit/textmining_dataset
[tutorial]: https://github.com/lovit/python_ml4nlp

## 구성 요소

위 repository 에는 분석용 샘플 데이터가 포함되어 있습니다. `lovit_textmining_dataset` 의 하위 폴더들은 각 데이터셋의 이름이며 현재 정리된 데이터셋의 이름은 아래와 같습니다. 데이터 별 특징은 각 데이터 폴더 안의 README 에 기록하였습니다.

| Dataset name | Description |
| --- | --- |
| navermovie_comments | 네이버영화에서 수집한 영화별 사용자 작성 커멘트와 평점 |
| navernews_10days | 네이버뉴스에서 수집한 2016-10-20 ~ 2016-10-29 (10일) 간의 뉴스와 댓글 |

폴더는 아래처럼 구성되어 있습니다. 데이터셋 폴더 아래의 data 는 각 데이터셋별 raw data 이며, models 는 raw data 를 이용하여 학습을 한 모델들이 저장된 폴더 입니다. 예를 들어 영화 평점 분류 문제를 위하여 텍스트들로부터 Bag-of-Words Models 를 만들고, 이를 이용하여 학습데이터 (X, Y) 를 미리 만들어 둘 수 있습니다. 이러한 모델들을  models 안에 모아뒀습니다.

각 데이터셋 안에는 데이터셋 핸들링에 관련된 Python 파일들이 포함되어 있습니다. 사용법은 각 데이터 폴더 안의 README 를 참고하세요.

```
lovit_textmining_dataset
    |-- navermovie_comments
        |-- __init__.py
        |-- loader.py
        |-- README.md
        |-- data
            |-- data_large.txt
            |-- ...
        | models
            |-- ...
    |-- navernews_10days
        |-- __init__.py
        |-- loader.py
        |-- README.md
        |-- data
            |-- 2016-10-20.txt
            |-- ...
        | models
            |-- ...
```

이 repository 에는 데이터셋을 이용할 수 있는 Python codes 만 포함되어 있습니다. 실제 데이터들은 외부 서버에 나눠서 저장중이며, 파일을 다운로드 받기 위해서는 fetch 함수를 이용합니다. `version_check` 함수를 이용하면 현재 다운로드 된 데이터셋의 버전이 확인됩니다.

```python
from lovit_textmining_dataset import version_check

version_check()
```

`fetch` 함수를 이용하면 latest version 이 아닌 데이터들을 다운로드 합니다.

```python
from lovit_textmining_dataset import fetch

fetch()
```

특정 데이터셋의 data 나 models 만 다운로드 하고 싶다면 fetch 함수의 argument 로 그 값을 넣어줄 수 있습니다.

```python
fetch(dataset='navernews_10days', content='models')
fetch(dataset='navernews_10days')
```

## 사용 예시

예를 들어 soynlp 의 명사 추출기를 테스트하고 싶을 경우, 다음처럼 데이터셋과 soynlp 를 활용할 수 있습니다.

```python
from lovit_textmining_dataset.navernews_10days import get_news_paths
from soynlp.noun import LRNounExtractor_v2
from soynlp.utils import DoublespaceLineCorpus


corpus_path = get_news_paths(date='2016-10-20')
corpus = DoublespaceLineCorpus(corpus_path, iter_sent=True)

noun_extractor = LRNounExtractor_v2()
nouns = noun_extractor.train_extract(corpus)
```

또는 KoNLPy 를 이용하여 미리 토크나이징 한 데이터를 가지고 올 수도 있습니다. 빠른 분석 실습을 하려는데 데이터를 토크나이징 하는 시간도 줄여보기 위해서입니다. Raw data 가 아닌 경우에는 앞에서 언급한 [tutorial][tutorial] 에서 사용하는 모델들만 미리 학습해 두었습니다.

```python
corpus_path = get_news_paths(date='2016-10-20', tokenize='komoran')
corpus = DoublespaceLineCorpus(corpus_path, iter_sent=True)
```

또는 라라랜드 영화 리뷰를 읽은 뒤, [soyspacing][soyspacing] 을 이용하여 띄어쓰기 모델을 학습할 수도 있습니다.

```python
from soyspacing.countbase import CountSpace
from lovit_textmining_dataset.navermovie_comments import load_movie_comments

# store texts as a file
idxs, texts, rates = load_movie_comments(idxs='134963')
corpus_path = 'lalaland_comments.txt'
with open('lalaland_comments.txt', 'w', encoding='utf-8') as f:
    for text in texts:
        f.write('{}\n'.format(text.strip()))

# train model
model = CountSpace()
model.train(corpus_path)

# correct sentence
sent = '이건진짜좋은영화 라라랜드진짜좋은영화'
sent_corrected, tags = model.correct(sent)
```

또는 Gensim 을 이용하여 LDA 를 학습하기 위한 Bag of words model 을 가져올 수도 있습니다.

```python
import gensim
from gensim.models import LDAModel
from lovit_textmining_dataset.navernews_10days import get_bow

# load data
x, idx_to_vocab, vocab_to_idx = get_bow(date='2016-10-20', tokenize='noun')

# transform for gensim
corpus = gensim.matutils.Sparse2Corpus(x, documents_columns=False)
id2word = dict(enumerate(idx_to_vocab))

# train LDA model
ldamodel = LdaModel(corpus=corpus, num_topics=100, id2word=id2word)
```

[soyspacing]: https://github.com/lovit/soyspacing/

그 외에도 다양한 튜토리얼용 데이터셋을 공유할 것이며, 일부 큰 데이터는 다른 repository 에 공유할 계획입니다.

## 앞으로 정리할 예정인 데이터셋

### 아이돌 관련 뉴스

특정 tasks 용 데이터셋을 만드려 합니다. 시계열 분석용 (topic detection & tracking) 용 데이터도 준비중입니다. 예를 들어 `2013-01-01` 부터 최근까지의 뉴스 중에서 특정 단어 (아이돌 이름) 들이 포함된 뉴스를 수집중입니다. 아이돌은 활동 시기에 따라 생성되는 뉴스의 양이 다릅니다. 또한 각 시기별로 뉴스의 키워드들도 다릅니다. 그렇기 때문에 topic detection & tracking 을 연습하기가 좋습니다. 게다가 나무위키와 같은 곳에 활동 내역도 잘 정리되어 있기 때문에 labeled data 역시 만들기 쉬울 것으로 생각됩니다.

이 데이터를 수집할 때 특정 단어가 포함된 뉴스를 모두 수집하고 있습니다. 이 경우에는 주제와 관련없는 뉴스가 함께 수집되기도 합니다. 예를 들어 `여자친구`는 아이돌 그룹의 이름이기도 하지만, 일반적으로 사용되는 명사이기도 합니다. 수집된 뉴스 집합에서 목적에 맞지 않는 뉴스를 걸러내는 연습도 할 수 있습니다.

### 청와대 국민청원 데이터

문재인정부는 청와대 홈페이지에 국민청원 게시판을 만들었습니다. 국민청원에 게시된 질문 혹은 요청에 대하여 국민 20만명 이상의 동의를 얻으면 정부는 반드시 해당 게시물에 대한 답변을 해야하는 시스템입니다. 이 청원 데이터에는 각 시점의 사회적 이슈들이 일부 기록되어 있습니다. 청원의 국민 참여 기간은 게시후 한 달 이며, 청원이 종료된 게시물에 대하여 주기적으로 수집중입니다. 수집된 데이터는 [청원데이터셋 repository][petitions] 와 [청원데이터셋 archive][petitions_archive] 에 저장중이며, [textmining dataset][tmd] 에서도 곧 이용할 수 있도록 작업할 예정입니다.

[petitions]: https://github.com/lovit/petitions_dataset
[petitions_archive]: https://github.com/lovit/petitions_archive

그 외에도 의미있는 분석을 할 수 있거나 특정 tasks 를 할 수 있는 데이터가 정리된다면 지속적으로 업데이트할 예정입니다.

