---
title: Unsupervised noun extraction (3). Usage of extractor and tokenizer
date: 2018-05-09 19:00:00
categories:
- nlp
tags:
- preprocessing
- word extraction
---

이전 포스트에서 한국어 어절의 특징인 L + [R] 구조를 이용하여 통계 기반으로 명사를 추출하는 두 가지 방법을 제안하였습니다. Unsupervised noun extraction (2) 는 첫번째 포스트에서 제안되었던 soynlp.noun.LRNounExtractor 의 문제점을 유형화하고 이를 해결한 방법에 대한 포스트입니다. 두번째 포스트의 방법은 soynlp.noun.LRNounExtractor_v2 에 구현되어 있습니다. 명사 추출기의 역할은 명사 리스트를 만드는 것 까지입니다. 그 다음으로 필요한 기능은 주어진 문장들에서 명사를 잘라내는 토크나이징입니다. 이를 위해서 분석할 도메인의 데이터에 포함된 명사 리스트를 만든 뒤, 이를 Komoran 이나 soynlp.pos 와 같은 품사 판별기에 사용자 사전으로 추가할 수도 있습니다. 혹은 우리가 구축한 명사 리스트만을 이용하여 간단한 토크나이저를 만들 수도 있습니다. 이번 포스트에서는 두번째 포스트에서 제안된 명사 추출기를 이용하여 명사 리스트를 만든 뒤, string match 를 이용하여 토크나이징을 하는 과정을 공유합니다.

## Noun extractor vs. Noun tokenizer

이번 포스트는 이전 포스트의 명사 추출기에 대한 내용을 전제로 합니다. 통계와 한국어 어절 구조의 특징을 이용하여 데이터로부터 명사 리스트를 구축하는 과정에 대해서는 이전 [포스트 1][noun_v1] 과 [포스트 2][noun_v2]를 참고하세요. [포스트 2][noun_v2] 는 [soynlp][soynlp] 프로젝트의 명사 추출기인 LRNounExtractor_v2 에 대한 명세서입니다. 이 패키지를 이용하여 우리는 데이터에 등장한 명사 리스트를 확보할 수 있습니다.

명사 추출기의 역할은 명사 리스트를 만드는 것 까지입니다. 하지만 데이터 분석을 위하여 우리가 해야하는 일은 문장에서 명사들을 잘라내는 일입니다. 토크나이징을 통하여 문장을 단어열로 만들어야 합니다. 이를 위해서 Komoran 이나 soynlp.pos 등 다양한 형태소 분석기를 이용할 수도 있습니다. 하지만 우리가 확보한 명사 리스트가 충분히 유의미하고 데이터 분석에서 명사 외에 다른 단어들은 필요하지 않다면, 더 간단한 방법으로 토크나이징을 할 수도 있습니다.

이번 포스트에서는 확보된 명사 리스트와 string match 를 이용하여 간단한 명사 토크나이저를 만들어 봅니다.


## L part string match, Simplest way

정밀한 토크나이징은 길이가 $$n$$ 인 문장에 대하여 $$m$$ 개의 단어를 나누는 확률 $$P(w_{1:m} \vert c_{1:n})$$ 를 최대화 하여야 합니다. 이 과정에는 명사 외의 다른 단어의 단어 가능성도 살펴보아야 합니다.

그러나 아래와 같은 명사 리스트가 주어졌을 때 아래의 문장에서 명사를 찾는 과정은 string match 만으로도 충분합니다. 특히나 띄어쓰기가 잘 지켜졌다면 명사는 어절의 왼쪽에 위치합니다. 왼쪽의 subword, L part 가 명사 리스트에 포함되었는지를 확인하면 됩니다.

	sent = '뉴스에서 명사를 인식하는 것은 어렵지 않습니다.'
	nouns = {'뉴스', '명사', '인식', '것', '데이터', '명사사전', '사전', '복합', '사사'}

	expected = ['뉴스', '명사', '인식', '것']

몇 가지 예상되는 상황을 고려합니다. 

아래처럼 한 어절에서 매칭되는 명사들이 여러 개가 존재할 수 있습니다. '명사사전은'이란 어절에서는 ['명사', '사사', '사전', '명사사전']이 매칭 되었습니다. 하지만 '사사'는 어절의 왼쪽이 아닌 중앙에 위치합니다. 이는 제외합니다. 

['명사', '사전', '명사사전'] 은 복합명사와 이를 구성하는 두 개의 명사입니다. 만약 복합명사가 명사 사전에 추가되어 있다면 이를 단일한 명사로 이용합니다.

'복합명사'는 ['복합', '명사'] 처럼 연속된 두 개의 명사로 이뤄진 어절입니다. '복합명사'가 명사 사전에 등록되어 있지 않다면 연속된 명사는 복합명사로 취급할지 단일 명사로 분해할지는 사용자 옵션으로 두면 좋을 것 같습니다.

	sent = '명사사전은 명사와 사전의 복합명사 입니다.'

이 과정을 다음처럼 구현할 수 있습니다. 처음에는 주어진 어절, token 에 대하여 명사 후보를 만듭니다.

{% highlight python %}
# string match for generating candidats

_nouns = {'명사', '사전', '명사사전', '사사'}
token = '명사사전은'
n = len(token)

nouns = []
for b in range(n):
    for e in range(b, n+1):
        subword = token[b:e]
        if subword in _nouns:
            # (word, begin, length)
            nouns.append((subword, b, e - b))
{% endhighlight %}

첫 시작의 begin index 를 0 으로 설정합니다. 정렬의 1번 기준은 begin index 로, 2번 기준은 명사의 길이로 이용합니다. 그 뒤 begin index 와 end index 가 같은지 확인하며, 연속된 명사를 계속하여 찾습니다. 

{% highlight python %}
# sort. fisrt order: begin index, second order: length (desc)
nouns = sorted(nouns, key=lambda x:(x[1], -x[2]))

nouns_ = []
e = 0

while nouns:
    # pop first element
    noun, b, len_ = nouns.pop(0)
    # only concatenate nouns
    if not (b == e):
        return nouns_
    # append noun and update end index
    nouns_.append(noun)
    e = b + len_
    nouns = [noun for noun in nouns if noun[1] >= e]

return nouns_
{% endhighlight %}

이 과정을 각 어절별로 수행하면 어절의 왼쪽에 존재하는 명사를 잘라낼 수 있습니다.

앞서 연속된 단일 명사들을 복합명사로 만들지에 대한 논의를 하였습니다. noun_ 에는 연속된 명사가 포함되어 있기 때문에 이를 복합명사로 만들기 위해서는 string join 만으로도 충분합니다.

{% highlight python %}
''.join(nouns_)
{% endhighlight %}

## L, R 위치에 관계없는 string match

어절의 L part 에 위치한 subword 중 명사 리스트에 포함되어 있는지를 확인하는 방식은 문장에 띄어쓰기 오류가 없을 때 이용할 수 있는 방법입니다. 위 방법은 띄어쓰기 오류가 존재하여 여러 개의 어절이 붙어 있는 경우에는 첫 어절의 명사 밖에 확인할 수 없습니다.

하지만 이전의 [tokenizer post][tokenizer] 에서 제안한 MaxScoreTokenizer 의 원리처럼 사람은 띄어쓰기가 되어있지 않은 문장에서 자신이 잘 아는 명사부터 눈에 들어옵니다. MaxScoreTokenizer 를 그대로 이용할 수 있습니다. 마침 이전 포스트에서 다뤘던 명사 추출기는 명사 점수도 포함되어 있습니다. 

MaxScoreTokenizer 에서 flatten=False 를 설정하면 단어 외에도 (word, begin index, end index, word score, length) 의 list 가 return 되었습니다.

{% highlight python %}
from soynlp.tokenizer import MaxScoreTokenizer

scores = {'파스': 0.3, '파스타': 0.7, '좋아요': 0.2, '좋아':0.5}
tokenizer = MaxScoreTokenizer(scores=scores)

print(tokenizer.tokenize('난파스타가 좋아요', flatten=False))
# [[('난', 0, 1, 0.0, 1), ('파스타', 1, 4, 0.7, 3), ('가', 4, 5, 0.0, 1)],
#  [('좋아', 0, 2, 0.5, 2), ('요', 2, 3, 0.0, 1)]]
{% endhighlight %}

반드시 보호하고 싶거나 확신을 가지는 명사가 있다면 그들의 점수를 다른 명사보다 더 높게 설정하면 됩니다. 그리고 모든 명사들은 0 보다 큰 점수를 부여합니다. MaxScoreTokenizer 가 tokenizing 을 한 결과 중에서 단어 점수가 0 인 단어들을 제거하면 명사만 남기 때문입니다.

{% highlight python %}
from soynlp.tokenizer import MaxScoreTokenizer

base_tokenizer = MaxScoreTokenizer(scores=noun_scores)
words = base_tokenizer(eojeol, flatten=False)[0]

# remove non-noun words
words = [word for word in words if word[3] > 0]
{% endhighlight %}

대신 한 가지 기능을 추가하였습니다. 앞서 복합명사를 하나의 단일 명사로 취급할 수 있도록 사용자 설정 option 을 주자는 이야기를 했습니다. 이 부분은 begin index 와 end index 를 확인하여 이들이 연속하면 하나의 명사로 묶습니다.

{% highlight python %}
def concatenate(eojeol, words):
    words_, b, e, score = [], 0, 0, 0
    for noun_, b_, e_, score_, _ in words:
        if e == b_:
            e, score = e_, max(score, score_)
        else:
            words_.append((eojeol[b:e], b, e, score, e-b))
            b, e = b_, e_
    if e > b:
        words_.append((eojeol[b:e], b, e, score, e-b))
    return words_
{% endhighlight %}


## Package. soynlp.tokenizer

위 코드들은 soynlp.tokenizer 의 NounLMatchTokenizer 와 NounMatchTokenizer 에 정리해두었습니다.

compose_compound=False 로 설정하면 연속된 명사 각각을 명사로 return 하고 compose_compound=True 로 설정하면 연속된 명사를 하나의 단일 명사로 묶을 수 있도록 하였습니다.

{% highlight python %}
from soynlp.tokenizer import NounLMatchTokenizer

noun_scores = {'파이썬': 1, '패키지': 1, '파이썬패키지':0.5}
l_match_tokenizer = NounLMatchTokenizer(noun_scores)

sent = '패키지파이썬패키지는 파이썬으로 만들어진 패키지입니다'

print(l_match_tokenizer.tokenize(sent))
# ['패키지파이썬패키지', '파이썬', '패키지']

print(l_match_tokenizer.tokenize(sent, compose_compound=False))
# ['패키지', '파이썬패키지', '파이썬', '패키지']

print(l_match_tokenizer.tokenize(sent.replace(' ',''), compose_compound=False))
# ['패키지', '파이썬패키지']
{% endhighlight %}

띄어쓰기가 잘 되어있지 않는 경우에 점수가 높은 명사부터 잘라내기 위한 NounMatchTokenizer 를 만들었습니다. 여기에도 compose_compound option 을 두어 선택적으로 연속된 명사열을 하나의 명사로 묶을 수 있도록 하였습니다.

{% highlight python %}
from soynlp.tokenizer import NounMatchTokenizer

noun_scores = {'파이썬': 1, '패키지': 1}
match_tokenizer = NounLMatchTokenizer(noun_scores)

sent = '패키지파이썬패키지는 파이썬으로 만들어진 패키지입니다'

print(match_tokenizer.tokenize(sent))
# ['패키지파이썬패키지', '파이썬', '패키지']

print(match_tokenizer.tokenize(sent, compose_compound=False))
# ['패키지', '파이썬', '패키지', '파이썬', '패키지']

print(match_tokenizer.tokenize(sent.replace(' ',''), compose_compound=False))
# ['패키지', '파이썬', '패키지', '파이썬', '패키지']
{% endhighlight %}

## Discussion

물론 어절의 왼쪽 부분의 string match 로만 토크나이징을 할 경우 모호성이 발생하거나 사전에 등록되지 않는 명사들에 대해서 제대로 인식하지 못하는 문제가 있습니다. 특히 위의 방법들은 각 어절에 대하여 서로 독립적입니다. 하지만 모호성이 발생할 경우에는 앞, 뒤의 어절을 함께 살펴봐야 하는 경우들이 발생합니다.

심광섭 교수님은 [논문 (심광섭, 2016)][cloning]에서 앞, 뒤 어절의 문맥을 모두 고려하는 형태소 분석기의 결과에 대하여 어절 단위로 복제를 한 뒤, 복제된 형태소 분석기를 이용하여 형태소 분석을 하였습니다. 복제된 형태소 분석기는 아래와 같은 정보가 학습되어 있습니다.

	pre_analyzed_eojeol_dictionary = {
		'명사사전은':(('명사', 'Noun'), ('사전', 'Noun'), ('은', 'Josa')),
		'명사와': (('명사', 'Noun'), ('와', 'Josa')),
		...
	}

복제된 어절 단위의 형태소 분석기를 이용한다는 것은 각 어절에 대하여 독립적인 형태소 분석을 수행한다는 의미입니다. 원 형태소 분석기와 복제된 형태소 분석기에서의 형태소 분석 결과의 96.80 % 가 일치하였습니다. 이 말의 의미는 어절 단위의 분석을 수행하여도 96.80 % 는 모호하지 않다고도 해석할 수 있습니다. 물론 기 분석된 어절이 존재하지 않을 경우, 이를 처리하기 위한 backoff module 이 있어서 수치는 정확하지 않겠지만, 다수의 어절은 단일 어절만으로도 모호성이 적다는 것은 틀림없습니다.

조금 거칠게 표현하자면 한국어 품사 판별 분석의 96.80 % 는 쉬운 case 이고, 3.20 % 가 어려운 case 라고도 말할 수 있습니다. 쉬운 부분은 쉽게 풀자는 컨셉으로 명사 리스트를 이용한 명사 토크나이저를 만들어 보았습니다.

## Demo

### Context vector + PMI 를 이용한 유사어 검색

2016-10-20 의 뉴스를 이용하여 명사를 추출하였습니다. 그리고 이를 이용하여 토크나이징과 context vector + PMI 를 이용한 유사어 검색을 하였습니다. Context vector + PMI 의 유사어 검색은 [이전 포스트][w2v_pmi]를 참고하세요. 

{% highlight python %}
from soynlp.utils import DoublespaceLineCorpus
from soynlp.noun import LRNounExtractor_v2

corpus_path = '2016-10-20-news'
sents = DoublespaceLineCorpus(corpus_path, iter_sent=True)

noun_extractor = LRNounExtractor_v2()
nouns = noun_extractor.train_extract(sents)
{% endhighlight %}

길이가 1 보다 긴 명사에 대해서만 명사 리스트를 만든 뒤, NounLMatchTokenizer 를 이용합니다. 분석할 데이터가 뉴스이기 때문에 명사는 어절의 왼쪽에 위치합니다.

{% highlight python %}
from soynlp.tokenizer import NounLMatchTokenizer

noun_scores = {noun:score[0] for noun, score in nouns.items() if len(noun) > 1}
tokenizer = NounLMatchTokenizer(noun_scores)
{% endhighlight %}

이를 이용하여 context - word matrix 를 만듭니다.

{% highlight python %}
from soynlp.vectorizer import sent_to_word_context_matrix

corpus.iter_sent=True
x, idx2vocab = sent_to_word_context_matrix(
    corpus,
    windows=3,
    min_tf=10,
    tokenizer=tokenizer, # (default) lambda x:x.split(),
    verbose=True)

vocab2idx = {vocab:idx for idx,vocab in enumerate(idx2vocab)}
{% endhighlight %}

그리고 이를 soynp.word.pmi 를 이용하여 term weighting 을 수행합니다.

{% highlight python %}
from soynlp.word import pmi

pmi_dok = pmi(
    x,
    min_pmi=0,
    alpha=0.0001,
    verbose=True)
{% endhighlight %}

Scikit-learn 의 Singular Value Decomposition 을 이용하여 이를 300 차원으로 차원을 축소합니다. 23,029 개의 단어에 대한 embedding vector 를 학습하였습니다.

{% highlight python %}
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=300)
y = svd.fit_transform(pmi_dok)
print(y.shape) # (23029, 300)
{% endhighlight %}

유사어를 찾는 함수를 만듭니다. 이를 이용하여 유사어 탐색을 하였습니다.

{% highlight python %}
from sklearn.metrics import pairwise_distances

def most_similar_words(query, topk=10):
    
    if not (query in vocab2idx):
        return []

    query_idx = vocab2idx[query]
    dist = pairwise_distances(y[query_idx].reshape(1,-1), y, metric='cosine')[0]
    similars = []
    
    # sorting
    for similar_idx in dist.argsort():
        
        # filtering query term
        if similar_idx == query_idx:
            continue

        if len(similars) >= topk:
            break

        # decoding index to word
        similar_word = idx2vocab[similar_idx]
        similars.append((similar_word, 1-dist[similar_idx]))

    return similars
{% endhighlight %}

LTokenizer 를 이용하였을 때에는 '너무너무너무'가 '너무너무'로 잘못 토크나이징이 되었습니다. 명사 추출기와 명사 토크나이저를 이용하였더니 이 부분은 해결되었습니다. Cohesion score 는 (L, R) 구조가 아닌 L part 의 단어 가능성 점수만을 이용합니다. 토픽모델링이나 키워드 추출처럼 명사만 이용해도 되는 작업이라면 명사 추출과 명사 토크나이징을 통하여 전처리를 하는 것도 좋다고 생각됩니다.

[이전 포스트][w2v_pmi]에서 아프리카는 '대륙 아프리카'와 '아프리카TV' 가 하나의 단어로 합쳐졌습니다. 여기서도 그러한 현상이 보입니다. 대도서관, 윰댕, 밴쯔는 아프리카TV 의 BJ 입니다. 대도서관과 중남미는 각각의 비슷한 토픽의 단어들과 유사함을 확인할 수 있습니다. '새마을운동글로벌리그' 같은 단어도 눈에 뜁니다.

[이전 포스트][w2v_pmi]에서 재밌던 단어 중 하나는 '아프리카발톱개구리'였습니다. 이런 개구리가 정말 있었습니다. 생물학 관련 단어들이 유사어로 검색됩니다.

| 아이오아이 | 아프리카 | 대도서관 | 중남미 | 아프리카발톱개구리 |
|  -- |  -- |  -- | -- | -- |
| [('카운트다운', 0.877), | [('대도서관', 0.528), | [('윰댕', 0.735), | [('아프리카', 0.514), | [('유전체', 0.958), |
|  ('너무너무너무', 0.847), | ('중남미', 0.514), | ('밴쯔', 0.559), | ('아시아', 0.479), | ('해독', 0.930), |
|  ('완전체', 0.828), | ('국가들', 0.500), | ('아프리카', 0.528), | ('진출', 0.475), | ('염색체', 0.907), |
|  ('전소미', 0.812), | ('윰댕', 0.480), | ('유튜브', 0.483), | ('동남아시아', 0.470), | ('개구리', 0.894), |
|  ('타이틀곡', 0.805), | ('이전', 0.477), | ('블로그', 0.443), | ('남미', 0.449), | ('서양발톱개구리', 0.879), |
|  ('멤버들', 0.793), | ('일본', 0.474), | ('개인방송', 0.434), | ('인도네시아', 0.434), | ('유전', 0.846), |
|  ('오블리스', 0.782), | ('아시아', 0.472), | ('유투브', 0.409), | ('국내외', 0.422), | ('2배체', 0.835), |
|  ('신용재', 0.747), | ('유럽', 0.469), | ('라온', 0.403), | ('각국', 0.413), | ('권태', 0.827), |
|  ('음악방송', 0.720), | ('전쟁', 0.465), | ('네티즌', 0.397), | ('싱가포르', 0.413), | ('공동연구진', 0.823), |
|  ('걸그룹', 0.715)] | ('남미', 0.462)] | ('생중계', 0.395)] | ('새마을운동글로벌리그', 0.405)] | ('생명과학부', 0.795)] |

유사어와 명사로 추출된 결과들이 대체로 납득이 됩니다.


### NounMatchTokenizer usage

테스트 삼아 NounMatchTokenizer 도 이용해봅니다. 띄어쓰기를 하지 않거나 중간에 '훗'같은 글자르 넣어도 어느 정도 명사가 추출되기는 합니다. 하지만, '해도'는 명사가 아닙니다. 이처럼 string match 만을 이용하면 문맥이 고려되지 않아서 명사가 아닌 글자가 명사로 잘못 잡힐 수도 있습니다.

{% highlight python %}
from soynlp.tokenizer import NounMatchTokenizer

tokenizer2 = NounMatchTokenizer(noun_scores)
sent = '두바이월드빌딩훗행사에는 여러 유명인들이 초대되었습니다띄어쓰기를안해도어느정도명사는추출됩니다'

print(tokenizer(sent2, compose_compound=False))
# ['두바이', '월드', '빌딩', '행사', '유명인들', '초대', '해도', '어느정도', '명사', '추출']
{% endhighlight %}

또한 연속된 명사를 하나로 합치면 명사가 아닌 단어가 잘못 추출된 여파로 엉뚱한 단어가 튀어나오기도 합니다. 복합명사를 하나의 단일명사로 취급하는 것은 L parts 에 대해서 명사를 추출할 때만 적용하는 것이 좋다는 생각이 듭니다. 그리고 복합명사가 그 자체로 자주 이용되었다면 명사 추출 단계에서 하나의 명사로 이미 추출되어 있습니다. 

{% highlight python %}
print(tokenizer(sent2))
# ['두바이월드빌딩', '행사', '유명인들', '초대', '해도어느정도명사', '추출']
{% endhighlight %}

## References
- 심광섭. (2016). [기분석 어절 사전과 음절 단위의 확률 모델을 이용한 한국어 형태소 분석기 복제.][cloning] 정보과학회 컴퓨팅의 실제 논문지, 22(3), 119-126.

[soynlp]: https://github.com/lovit/soynlp/
[cloning]: http://kiise.or.kr/e_journal/2016/3/KTCP/pdf/01.pdf
[tokenizer]: {{ site.baseurl }}{% link _posts/2018-04-09-three_tokenizers_soynlp.md %}
[w2v_pmi]: {{ site.baseurl }}{% link _posts/2018-04-22-context_vector_for_word_similarity.md %}
[noun_v1]: {{ site.baseurl }}{% link _posts/2018-05-07-noun_extraction_ver1.md %}
[noun_v2]: {{ site.baseurl }}{% link _posts/2018-05-08-noun_extraction_ver2.md %}
