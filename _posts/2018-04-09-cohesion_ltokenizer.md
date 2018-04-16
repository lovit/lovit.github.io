---
title: Cohesion score + L-Tokenizer. 띄어쓰기가 잘 되어있는 한국어 문서를 위한 unsupervised tokenizer
date: 2018-04-09 22:00:00
categories:
- nlp
tags:
- preprocessing
- word
- tokenizer
---

다양한 언어에서 미등록 단어를 데이터 기반으로 추출하려는 시도가 있었습니다. 단어는 연속으로 등장한 글자이며, 그 글자들은 서로 연관성이 높습니다. Characters 간의 co-occurrence 정보를 이용하면 단어를 찾을 수 있습니다. Cohesion score 는 한국어의 단어 추출을 위하여 character n-gram 을 이용합니다. 또한 한국어 어절의 구조인 L + [R] 특성을 함께 이용하면 간단한 unsupervised tokenizer 도 만들 수 있습니다. 


## Out of vocabulary problem

말은 언제나 변화합니다. 새로운 개념을 설명하기 위해 새로운 단어가 만들어지기 때문에 모든 단어를 포함하는 사전은 존재할 수 없습니다. 학습데이터를 이용하는 supervised algorithms 은 가르쳐주지 않은 단어를 인식하기가 어렵습니다. 

KoNLPy 의 트위터 한국어 분석기를 이용하여 아래 문장을 분석합니다.

{% highlight python %}
from konlpy.tag import Twitter

twitter = Twitter()
twitter.pos('너무너무너무는 아이오아이의 노래입니다')
{% endhighlight %}

사전에 등제되지 않은 '너무너무너무'와 '아이오아이'는 형태소 분석이 제대로 이뤄지지 않습니다. 

	[('너무', 'Noun'),
	 ('너무', 'Noun'),
	 ('너무', 'Noun'),
	 ('는', 'Josa'),
	 ('아이오', 'Noun'),
	 ('아이', 'Noun'),
	 ('의', 'Josa'),
	 ('노래', 'Noun'),
	 ('입니', 'Adjective'),
	 ('다', 'Eomi')]

하지만 데이터를 분석하기 전까지 '너무너무너무', '아이오아이' 라는 단어가 등장할 것이라 예상하지 못할 수 있습니다. 통계 기반 단어 추출 기법은 '우리가 분석하려는 데이터에서 최대한 단어를 인식'하여 학습데이터를 기반으로 하는 supervised approach 를 보완하기 위한 방법입니다.


## Cohesion score

Branching Entropy와 Accessor Variety는 단어의 좌/우의 경계에 등장하는 글자들의 정보를 이용하여 단어의 경계를 판단하는 exterior boundary scoring 방법입니다. 단어를 구성하는 글자들 외의 정보를 이용합니다. 반대로 단어를 구성하는 글자들만의 정보를 이용하는 방법을 interior boundary scoring 이라 합니다. Character n-gram 이나 mutual information 방법이 이에 해당합니다. 

먼저 한국어 어절의 구조에 대하여 다시 한 번 살펴봅니다. 한국어의 어절은 L + [R] 구조입니다. 띄어쓰기가 제대로 되어있다면 한국어는 의미를 지니는 단어 명사, 동사, 형용사, 부사, 감탄사가 어절의 왼쪽 (L part)에 등장합니다. 문법 기능을 하는 조사나 어미는 어절의 오른쪽 (R part)에 등장합니다. 

새롭게 만들어지는 단어들은 문법 기능이 아닌 의미를 표현하는 경우가 많습니다. 이들은 L part 에 위치합니다. 우리가 데이터 기반으로 추출해야 하는 단어는 L part 에 있습니다. R part 의 어미도 새롭게 만들어지는 단어 (정확히는 형태소) 입니다. 다양한 말투 때문에 새로운 어미가 만들어집니다. 이 문제는 잠시 제쳐두고 L part 에 집중합니다. 단, 어절의 중간에 위치한 글자는 추출할 필요가 없습니다. '아이오아이'의 가운데의 '이오아'는 단어 추출의 후보에서 제외해도 됩니다. L part 도 R part 도 아니면 단어가 아닙니다. 정말로 '이오아'가 단어라면 다른 어절에서 L part 일 것입니다. 

Character n-gram 을 계산하기 위하여 substring counting 을 합니다. 데이터에 아래 다섯 개의 어절이 각각의 빈도수로 존재합니다. 

	노래가 (50)
	노래는 (30)
	노래를 (20)
	노란색 (90)
	노란색을 (10)

이로부터 우리는 어절의 왼쪽에서 시작하는 모든 substrings 의 빈도수를 계산할 수 있습니다. 

    노 (200)
    노래 (100)
    노란 (100)
    노래가 (50)
    노래는 (30)
    노래를 (20)
    노란색 (100)
    노란색을 (10)
    
래가, 래는, 래를, 란색 과 같이 어절의 왼쪽에서 시작하지 않은 substrings는 카운팅할 필요가 없습니다. 왜냐면 애초에 단어가 아니기 때문입니다. 

한가지 더, 우리는 길이가 1인 글자에 관심이 없습니다. 1음절 단어는 해석이 어렵습니다. 표의문자의 성격을 많이 지닌 한국어는 자주 쓰이는 1음절 자체가 하나의 단어로 이미 존재합니다. 그리고 앞, 뒤의 문맥 정보 없이는 해석이 어렵습니다. 그렇기 때문에 단어 추출의 관점에서는 1음절 단어는 무시합니다. 

$$P(노래 \vert 노)$$를 정의합니다. Frequentist 처럼 관찰한 정보를 그대로 확률을 정의합니다. '노'라는 글자가 나온 다음, '노래'가 나온 경우가 100 / 200 번 이므로 $$P(노래 \vert 노) = 0.5$$ 입니다. 이처럼 다른 substring 도 한 글자가 더 늘어날 확률을 각각 계산합니다. 

	P(노래|노) = 0.5
	P(노란|노) = 0.5
	P(노란색|노란) = 1
	P(노란색을|노란색) = 0.1

'노'는 context 로써 명확한 정보를 주지 못합니다. 어절의 종류가 많아지면 $$P(노x \vert 노)$$ 의 확률은 매우 작아집니다. 하지만 '노란' 이라는 substring 은 '노란색' 을 지칭합니다. '노란'은 '노란색'이라는 정보를 명확히 가르쳐주는 context 입니다. 단어의 경계 이전에는 다음 글자를 예측하기가 쉽습니다. 그러나 단어의 경계를 넘어서 존재하는 글자는 이전 글자가 명확한 context 가 되지 않습니다. '노란색'은 '노란색을'의 명확한 context 가 아닙니다. 

단어의 경계에 가까워질수록 $$P(xy \vert x)$$ 의 값이 커지고, 단어의 경계를 넘어서면 $$P(xy \vert x)$$의 값이 줄어듭니다. 이 현상을 이용하여 L part 에서 단어를 추출할 수 있는 character n-gram 기반 score 를 정의합니다. 

$$cohesion(c_{0:n}) = \left( \prod P(c_{0:i+1} \vert c_{0:i}) \right) ^{n-1}$$

어절의 왼쪽에서부터 $$n-1$$ 개의 $$P(xy \vert x)$$ 를 누적하여 곱한 뒤 $$\frac{1}{n-1}$$ 승을 취합니다. $$P(xy \vert x)$$ 는 확률이기 때문에 1 이하입니다. 길이가 길수록 누적곲이 같거나 줄어듭니다. 더 긴 글자를 선호 (preference) 하기 위하여 root 를 취합니다. '노란색'의 cohesion 은 아래와 같습니다. 

$$cohesion(노란색) = \left( P(노란 \vert 노) \times P(노란색 \vert 노란) \right) ^{(0.5)} = (0.5 \times 1) ^{(0.5)} = 0.707$$

Cohesion score 를 학습하는 과정은 corpus 의 substring counting 입니다. list of str 의 docs 에 대하여 띄어쓰기 기준으로 어절을 나눈 뒤 어절 왼쪽에 위치하는 모든 substrings 의 빈도수를 계산합니다. 

{% highlight python %}
from collections import defaultdict
count= defaultdict(lambda: 0)

for doc in docs:
    for word in doc.split():
        n = len(word)
        for e in range(1, n+1):
            count[word[:e]] += 1
{% endhighlight %}

Cohesion score 는 다음처럼 계산합니다.

{% highlight python %}
def cohesion(w):
    return pow(count[w]/count[w[0]], 1/(len(w)-1))
{% endhighlight %}

위 코드는 확률의 누적곲이 아닙니다. 아래처럼 $$P(x_{0:1})$$ 과 $$P(x_{0:n})$$ 을 제외한 모든 부분이 삭제되기 때문입니다. 

$$cohesion(c_{0:4}) = \frac{ c_{0:2} }{ c_{0:1} } \times \frac{ c_{0:3} }{ c_{0:2} } \times \frac{ c_{0:4} }{ c_{0:3} }$$


## L-Tokenizer

Cohesion score 를 이용하여 어절의 왼쪽에 위치한 substrings 중 가장 단어스러운 부분을 찾는 방식으로 간단한 비지도학습 기반 토크나이저를 만들 수 있습니다. 

뉴스처럼 띄어쓰기가 잘 되어 있는 한국어 텍스트의 어절 구조는 명확히 L + [R] 입니다. 어절의 왼쪽에 위치한 substring 중 단어 점수가 가장 높은 부분을 선택하면 어절을 L + [R] 로 나눌 수 있습니다. 

2016-10-20 의 뉴스 데이터를 이용한 실험입니다. 어절 '아이오아이는'의 L substrings 중 chesion score 가 가장 높은 substring 은 '아이오아이' 이기 때문에 이 어절은 '아이오아이 + 는' 으로 나뉘어집니다. 

| subword | frequency | $$P(AB \vert A)$$ | Cohesion score |
| --- | --- | --- | --- |
| 아이 | 4,910 | 0.15 | 0.15 |
| 아이오 | 307 | 0.06 | 0.1 |
| 아이오아 | 270 | 0.88 | 0.2 |
| 아이오아이 | 270 | 1 | 0.3 |
| 아이오아이는 | 40 | 0.15 | 0.26 |

이 과정을 다음의 알고리즘으로 만들 수 있습니다. 

{% highlight python %}
def ltokenize(w):
    n = len(w)
    if n <= 2: return (w, '') tokens = []
    for e in range(2, n+1):
        tokens.append(w[:e], w[e:], cohesion(w[:e]))
    tokens = sorted(tokens, key=lambda x:-x[2])
    return tokens[0][:2]
{% endhighlight %}


## L-Tokenizer + Naver news + Word2Vec

Cohesion score 와 L-Tokenizer 를 이용하여 문서를 토크나이징 한 뒤, Word2Vec 모델을 학습하였습니다. soynlp의 WordExtractor + L-Tokenizer chapter 에서 설명하였던 예시의 ltokenizer 를 이용하여 30,091 개 문서, 22 만 여개의 문장을 토크나이징 하였습니다. 

{% highlight python %}
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from gensim.models import Word2Vec

corpus = DoublespaceLineCorpus(corpus_path, iter_sent=True)
word_extractor = WordExtractor(min_count=5)
word_extractor.train(corpus)
word_scores = word_extractor.extract()

cohesion_scores = {word:score.cohesion_forward for word, score in word_scores.items()}
ltokenizer = LTokenizer(scores = cohesion_scores)

word2vec_corpus = [ltokenizer.tokenize(sent) for sent in corpus]
word2vec = Word2Vec(word2vec_corpus)
{% endhighlight %}

그 뒤 유사어 검색을 하였습니다. '방송'의 유사어로 프로그램을 지칭하는 말들과 방송 프로그램의 이름들이 검색됩니다. 

{% highlight python %}
word2vec.most_similar('방송')

# [('예능프로그램', 0.6981078386306763),
#  ('예능', 0.688145637512207),
#  ('방영', 0.660905659198761),
#  ('라디오스타', 0.6552329659461975),
#  ('라디오', 0.6498782634735107),
#  ('엠카운트다운', 0.6143872141838074),
#  ('파워타임', 0.586585521697998),
#  ('식사하셨어요', 0.5865078568458557),
#  ('한끼줍쇼', 0.5841958522796631),
#  ('황금어장', 0.5715700387954712)]
{% endhighlight %}

'뉴스'의 유사어는 뉴스 체널과 관련된 단어가 학습되었습니다. 

{% highlight python %}
word2vec.most_similar('뉴스')

# [('연예', 0.6419150233268738),
#  ('화면', 0.6120390892028809),
#  ('뉴스부장', 0.5935080051422119),
#  ('뉴스부', 0.5910232067108154),
#  ('라디오', 0.5880887508392334),
#  ('채널', 0.5730870962142944),
#  ('정보팀', 0.5722399950027466),
#  ('속보팀', 0.5669897794723511),
#  ('앵커', 0.5661150217056274),
#  ('토크쇼', 0.5654663443565369)]
{% endhighlight %}

'영화'의 유사어는 영화 용어 및 영상 작품을 지칭하는 단어들이 학습되었습니다. 

{% highlight python %}
word2vec.most_similar('영화')

# [('작품', 0.7261432409286499),
#  ('드라마', 0.7252553701400757),
#  ('흥행', 0.6903091669082642),
#  ('독립영화', 0.6875913143157959),
#  ('걷기왕', 0.654923677444458),
#  ('감독', 0.6543536186218262),
#  ('럭키', 0.6453847885131836),
#  ('다큐멘터리', 0.6425734758377075),
#  ('블록버스터', 0.6288458704948425),
#  ('주연', 0.6285911798477173)]
{% endhighlight %}

'아이오아이'의 유사어로 '에이핑크', '샤이니', '다이아', '트와이스'와 같은 동시대에 함께 활동한 다른 아이돌 그룹이 거색됩니다. '너무너무'는 타이틀곡 '너무너무너무'가 잘못 토크나이징 된 결과이며, '몬스'는 '몬스터 엑스'입니다. 이 부분은 아쉽습니다. '잠깐만'은 '아이오아이'의 노래 제목이며, '불독'은 그 당시 데뷔하였던 다른 아이돌그룹 이름입니다. '불독의'로 토크나이징 된 점이 아쉽습니다.

{% highlight python %}
word2vec.most_similar('아이오아이')

# [('타이틀곡', 0.8425650000572205),
#  ('에이핑크', 0.8301823139190674),
#  ('샤이니', 0.8209962844848633),
#  ('너무너무', 0.8158787488937378),
#  ('다이아', 0.8083204030990601),
#  ('트와이스', 0.8079981803894043),
#  ('파이터', 0.8070152997970581),
#  ('잠깐만', 0.7995353937149048),
#  ('몬스', 0.7901242971420288),
#  ('불독의', 0.7892482280731201)]
{% endhighlight %}

어떠한 언어자원을 이용하지 않고서 통계 기반 방법 만으로도 여기까지 자연어처리를 할 수 있습니다. 이 정보를 기존의 supervised algorithm 과 더하여 이용할 수도 있습니다. 반드시 pure unsupervised approach 를 고집할 이유가 없습니다. 어떤 수단을 써서라도 문제를 잘 푸는 것이 우리의 목적입니다.
