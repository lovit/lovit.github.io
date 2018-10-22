---
title: Hidden Markov Model 기반 품사 판별기의 decode 함수
date: 2018-10-23 23:00:00
categories:
- nlp
tags:
- tokenizer
- sequential labeling
---

Part of Speech tagging 과 같은 sequential labeling 을 위하여 Hidden Markov Model (HMM), Conditional Random Field (CRF), 혹은 Recurrent Neural Network (RNN) 이 이용될 수 있습니다. 이러한 sequential labeling 알고리즘은 주어진 sequence 에 대한 적합성 (score, probability) 을 할 수 있으며 이를 evaluation 이라 합니다. 그리고 품사 판별이나 형태소 분석에서 evaluation 은 주어진 문장에 대한 단어, 품사열이 주어졌다고 가정할 때 이용할 수 있는 방법입니다. 이전 [포스트][hmm_tagger]에서는 HMM 기반 품사 판별기의 구조와 장단점, 그리고 evaluation 에 대하여 다뤘습니다. 이번 포스트에서는 문장이 주어졌을 때 단어, 품사열 후보를 만들고, 적합한 후보를 선정하는 과정까지 살펴봅니다.


## Hidden Markov Model (HMM)

Hidden Markov Model (HMM) 은 길이가 $$n$$ 인 sequence $$x_{1:n} = [x_1, x_2, \dots, x_n]$$ 에 대하여 $$P(y_{1:n} \vert x_{1:n})$$ 가 가장 큰 $$y_{1:n}$$ 를 찾습니다. 품사 판별의 문제에서는 $$n$$ 개의 단어로 구성된 단어열에 대하여 각 단어의 품사 $$y$$ 를 부여하는 것입니다. 이 과정을 HMM 의 decoding 이라 합니다. 그리고 우리가 찾아야 하는 label, $$y$$ 를 HMM 에서는 state 라 합니다.

이 때 $$P(y_{1:n} \vert x_{1:n})$$ 는 다음처럼 계산됩니다.

$$P(y_{1:n} \vert x_{1:n}) = P(x_1 \vert y_1) \times P(y_1 \vert START) \times P(y_2 \vert y_1) \times P(x_2 \vert y_2) \cdots$$

위의 식은 현재 시점 $$i$$ 의 state 인 $$y_i$$ 를 판별 (classification, or labeling) 하기 위하여 이전 시점의 state 인 $$y_{i-1}$$ 이 이용됩니다. 이처럼 이전의 한 단계 전의 state 정보를 이용하는 모델을 first-order Markov Model 이라 합니다. 만약 이전 두 단계의 states 정보를 모두 이용한다면 $$P(y_i \vert y_{i-2}, y_{i-1}$$ 이 학습되어야 하며, 이는 second-order Markov Model 이라 합니다. 그 외의 멀리 떨어진 state 정보는 이용하지 않습니다.

이처럼 state 간의 변화 확률을 transition probability (전이 확률) 라 합니다. HMM 은 각 state 에서 우리가 관측 가능한 값 (observation) 이 발생할 확률이 있다고 가정합니다. 이를 emission probability, $$P(x_i \vert y_i)$$ 라 합니다. 품사 판별에서는 명사 집합에서 '아이오아이'라는 단어가 존재할 확률 입니다.

숫자 계산에서 곱셈은 덧셈보다 비싼 작업입니다. 그렇기 때문에 확률을 곱하는 작업들은 주로 log 를 씌워 덧셈으로 변환합니다. 위 수식은 아래처럼 변환됩니다.

$$log P(y_{1:n} \vert x_{1:n}) = log P(x_1 \vert y_1)+ log P(y_1 \vert S) + log P(y_2 \vert y_1) + log P(x_2 \vert y_2) \cdots$$

## Supervised training

학습 말뭉치를 이용하여 품사 판별이나 형태소 분석용 tagger 를 학습하기도 합니다. 학습 말뭉치는 다음과 같은 데이터로 구성되어 있습니다.

아래는 세종 말뭉치의 예시입니다. 세종 말뭉치는 한국어 형태소 분석을 위하여 국립국어원에서 배포한 말뭉치입니다. 한 문장이 (morpheme, tag) 형식으로 기술되어 있습니다. 아래는 네 개의 문장의 스냅샷입니다.

    [['뭐', 'NP'], ['타', 'VV'], ['고', 'EC'], ['가', 'VV'], ['ㅏ', 'EF']]
    [['지하철', 'NNG']]
    [['기차', 'NNG']]
    [['아침', 'NNG'], ['에', 'JKB'], ['몇', 'MM'], ['시', 'NNB'], ['에', 'JKB'], ['타', 'VV'], ['고', 'EC'], ['가', 'VV'], ['는데', 'EF']]

우리는 first-order HMM tagger 를 만들어봅니다. 이는 state 관점에서 bigram 을 이용하기 때문에 bigram HMM tagger 라 불려도 괜찮습니다.

bigram HMM tagger 를 학습하려면 transition probability $$P(y_i \vert y_{i-1})$$ 와 emittion probability $$P(x_i \vert y_i)$$ 를 학습해야 합니다. 직관적으로 다음처럼 확률을 정의할 수 있습니다.

- transition prob: $$P(y_i \vert y_{i-1}) = \frac{f(y_{i-1}, y_i)}{f(y_{i-1})}$$
- emittion prob:   $$P(x_i \vert y_i) = \frac{f(x_i, y_i)}{f(y_i)}$$

그런데 위의 식이 직관적일 뿐 아니라 Maximum Likelihood Estimation (MLE) 관점에서도 확률을 학습하는 solution 입니다. 즉, 학습 말뭉치에서의 빈도수를 계산하는 것만으로 학습을 할 수 있습니다. 그리고 앞서 언급한 것처럼 float 는 곱셈보다 덧셈이 계산이 빠르기 때문에, 미리 math.log 를 이용하여 확률값을 log probability 로 변환하여 줍니다.

학습 말뭉치를 이용하는 supervised training 에 대한 자세한 내용은 이전의 [HMM tagger 포스트][hmm_tagger] 를 참고하세요. 우리는 학습이 끝난 HMM 의 parameters 인 emission prob. 와 transition prob. 의 log 값이 주어졌다고 가정합니다.

{% highlight python %}
class TrainedHMMTagger:
    def __init__(self, transition, emission, begin,
        begin_state='BOS', end_state='EOS', unk_state='Unk'):

        self.transition = transition
        self.emission = emission
        self.begin = begin

    def tag(self, sentence):
        raise NotImplemented
{% endhighlight %}

## Developing your HMM based pos tagger

### Structure

우리는 [이전의 포스트][ford_for_pos]에서 shortest path 를 찾는 Ford algorithm 이 HMM based pos tagger 의 가장 적합한 단어, 품사열 결과를 찾는 과정과 동일하다는 이야기를 하였습니다. 이번에는 Ford algorithm 을 이용하는 HMM based pos tagger 를 만듭니다.

Tagger 의 구조는 다음과 같습니다. 첫단계는 generate_edge 함수에서 학습된 사전을 바탕으로 (단어, 품사) 그래프를 만듭니다. 이 함수는 문장에서 우리가 알고 있는 단어들을 확인한 뒤, 그래프의 마디들로 만듭니다. 이를 shortest path 문제로 만들기 위하여 그래프에서의 시작 마디 (bos) 와 끝 마디 (eos) 도 지정합니다. 두번째 단계는 Ford algorithm 을 이용하여 최적의 경로를 찾는 것입니다. 세번째 단계는 그래프의 shortest path 형식으로 되어 있는 결과를 우리가 원하는 품사 판별의 결과로 만들고, 마지막으로 미등록 단어에 대한 품사 추정도 수행합니다.

{% highlight python %}
class TrainedHMMTagger:
    def __init__(self, transition, emission, begin):
        ...

    def tag(self, sentence):
        # generate candidates
        edges, bos, eos = self._generate_edge(sentence)
        edges = self._add_weight(edges)
        nodes = {node for edge in edges for node in edge[:2]}

        # choose optimal sequence
        path, cost = ford_list(edges, nodes, bos, eos)

        # postprocessing
        pos = self._postprocess(path)

        if inference_unknown:
            pos = self._inference_unknown(pos)

        return pos
{% endhighlight %}

다음 장에서 각각의 함수에 대하여 살펴봅니다. 일단은 클래스로 정리하지 않고, 기능별로 함수를 만들어봅니다.

이를 위해 다음의 데이터를 이용합니다. max word len, min emission, min transition 은 모델이 이용할 parameters 입니다. 미등록 단어에 대한 emission score 와 transition score 입니다. 또한 학습된 사전에서 어느 길이까지 단어를 찾아볼지 정의하기 위하여, 사전에 등록된 단어의 길이 중 가장 긴 길이를 max word len 에 저장합니다.

{% highlight python %}
from pprint import pprint

emission = {
    'Adjective': {'이': 0.1, '짧': 0.1},
    'Eomi': {'다': 0.1, '았다': 0.1, '었다': 0.1},
    'Josa': {'는': 0.15, '다': 0.05, '도': 0.1, '은': 0.2, '이': 0.1},
    'Noun': {'시작': 0.1, '예시': 0.1, '이': 0.15, '이것': 0.1},
    'Verb': {'입': 0.1, '하': 0.1}
}

transition = {
    ('Adjective', 'Noun'): 0.1,
    ('Josa', 'Adjective'): 0.1,
    ('Josa', 'Noun'): 0.2,
    ('Josa', 'Verb'): 0.1,
    ('Noun', 'Adjective'): 0.05,
    ('Noun', 'Josa'): 0.1,
    ('Noun', 'Noun'): 0.1,
    ('Noun', 'Verb'): 0.05,
    ('Verb', 'Noun'): 0.1
}

begin = {
    'Noun': 0.2,
    'Verb': 0.1,
    'Adjective': 0.1
}

_max_word_len = max(len(w) for words in emission.values() for w in words)
_min_emission = min(s for words in emission.values() for s in words.values()) - 0.05
_min_transition = min(transition.values()) - 0.05

print(_max_word_len) # 2
print(_min_emission) # 0.0
print(_min_transition) # 0.0
{% endhighlight %}

### Lookup

문장이 입력되면 띄어쓰기 기준으로 이를 분리합니다. 어절 단위로 우리가 알고 있는 단어들이 존재하는지 확인합니다. offset 은 현재까지 탐색한 문장에서의 글자수 입니다. 문장에서의 다음 어절의 시작점을 아려줍니다.

{% highlight python %}
def sentence_lookup(sentence):
    sent = []
    for eojeol in sentence.split():
        sent += eojeol_lookup(eojeol, offset=len(sent))
    return sent
{% endhighlight %}

어절에서의 단어 시작점 b 로부터 사전에 등록된 단어의 최대 길이 _max_word_len 까지 확장하며 substring 을 surface 에 잘라둡니다. 이 substring 이 사전에 존재하는지 get_pos 함수를 이용하여 확인합니다. 품사가 존재한다면 이는 우리가 학습한 단어라는 의미입니다.

그런데 단어를 (substring, tag, tag, begin, end) 형식으로 저장합니다. 이는 용언처럼 형태가 변하는 (활용, conjugation) 단어의 정보를 손쉽게 저장하기 위함입니다. 일단은 substring lookup 만 이뤄지도록 함수를 구현합니다.

{% highlight python %}
def eojeol_lookup(eojeol, offset=0):
    n = len(eojeol)
    pos = [[] for _ in range(n)]
    for b in range(n):
        for r in range(1, _max_word_len+1):
            e = b+r
            if e > n:
                continue
            surface = eojeol[b:e]
            for tag in get_pos(surface):
                pos[b].append((surface, tag, tag, b+offset, e+offset))
    return pos

def get_pos(sub):
    tags = []
    for tag, words in emission.items():
        if sub in words:
            tags.append(tag)
    return tags
{% endhighlight %}

eojeol_lookup 함수를 이용하면 어절의 길이만큼의 nested list 가 return 됩니다. list 의 각 위치는 어절에서의 글자의 시작지점입니다.

    eojeol_lookup('예시이다')

    [[('예시', 'Noun', 'Noun', 0, 2)],
     [],
     [('이', 'Noun', 'Noun', 2, 3), ('이', 'Josa', 'Josa', 2, 3),('이', 'Adjective', 'Adjective', 2, 3)],
     [('다', 'Josa', 'Josa', 3, 4), ('다', 'Eomi', 'Eomi', 3, 4)]]

문장에 대한 lookup 을 수행한 결과 입니다. '예시'는 문장에서 네번째로 등장하는 단어이기 때문에 begin index 가 3 임을 확인할 수 있습니다. 눈으로 확인하기 쉽도록 아래의 예시를 적을 때 문장의 글자 위치를 기준으로 줄바꿈을 하였습니다.

    sentence_lookup('이것은 예시이다')

    [[('이', 'Noun', 'Noun', 0, 1),
      ('이', 'Josa', 'Josa', 0, 1),
      ('이', 'Adjective', 'Adjective', 0, 1),
      ('이것', 'Noun', 'Noun', 0, 2)],

     [],

     [('은', 'Josa', 'Josa', 2, 3)],

     [('예시', 'Noun', 'Noun', 3, 5)],

     [],

     [('이', 'Noun', 'Noun', 5, 6),
      ('이', 'Josa', 'Josa', 5, 6),
      ('이', 'Adjective', 'Adjective', 5, 6)],

     [('다', 'Josa', 'Josa', 6, 7), ('다', 'Eomi', 'Eomi', 6, 7)]]

그러나 우리가 가지고 있는 사전이 용언의 표현형에 대한 정보가 아니라 어간과 어미의 원형이라면 lemmatization 과정을 거쳐야 합니다.

'했다'를 동사로 알고 있다면 그 자체를 동사로 인식하여도 됩니다. 하지만 이를 '하/Verb + 았다/Eomi' 로 인식할 수도 있습니다. 이 기능을 추가하기 위하여 앞서 개발한 규칙 기반 lemmatizer 를 이용합니다. 이 lemmatizer 의 개발 과정은 이전의 [포스트][lemmatizer]를 참고하세요. 부분어절을 자른 뒤, 이 부분 어절이 어간과 어미로 분리가 되면 이를 lemmas 에 추가합니다.

{% highlight python %}
from soynlp.lemmatizer import _lemma_candidate

def lemmatize(word, i):
    l = word[:i]
    r = word[i:]
    lemmas = []
    len_word = len(word)
    for l_, r_ in _lemma_candidate(l, r):
        word_ = l_ + ' + ' + r_
        if (l_ in emission['Verb']) and (r_ in emission['Eomi']):
            lemmas.append((word_, 'Verb', 'Eomi'))
        if (l_ in emission['Adjective']) and (r_ in emission['Eomi']):
            lemmas.append((word_, 'Adjective', 'Eomi'))
    return lemmas
{% endhighlight %}

그 결과 부분어절 '했다'의 가운데 지점을 어간과 어미의 교차점으로 가정하면 '하/Verb + 았다/Eomi' 로 인식합니다.

    lemmatize('했다', 1)
    [('하 + 았다', 'Verb', 'Eomi')]

이 과정을 eojeol_lookup 에 추가합니다. 그리고 다시 sentence_lookup 함수를 적용하면 '였다'가 '이/Adjective + 었다/Eomi' 로 인식됨을 확인할 수 있습니다.

    sentence_lookup('이것은 예시였다')

    [[('이', 'Adjective', 'Adjective', 0, 1),
      ('이', 'Josa', 'Josa', 0, 1),
      ('이', 'Noun', 'Noun', 0, 1),
      ('이것', 'Noun', 'Noun', 0, 2)],
     [],
     [('은', 'Josa', 'Josa', 2, 3)],
     [('예시', 'Noun', 'Noun', 3, 5)],
     [],
     [('이 + 었다', 'Adjective', 'Eomi', 5, 7)],
     [('다', 'Eomi', 'Eomi', 6, 7), ('다', 'Josa', 'Josa', 6, 7)]]

### Generate (word, tag) graph

### 그래프 만들기

앞서 어절과 문장에서 알려진 단어를 찾는 과정을 구현하였습니다. 이 단어들 중에서 앞의 단어의 end index 와 뒤의 단어의 begin index 가 같은 경우, 이 둘을 연결할 수 있습니다.

이는 다음의 과정으로 구현할 수 있습니다.

    links = []
    for words in sent[:-1]:
        for word in words:
            begin = word[3]
            end = word[4]
            for adjacent in sent[end]:
                links.append((word, adjacent))

그러나 한 단어의 end index 에서부터 시작하는 단어가 없을 경우에는 링크가 만들어지지 않습니다. 이 때에는 가능한 가장 가까운 다음 단어의 begin index 를 찾아야 합니다. 이를 위하여 다음의 함수를 구현합니다. offset 이후의 지점에서 sent[i] 가 empty 가 아닌 가장 빠른 지점을 return 합니다.

    def get_nonempty_first(sent, end, offset=0):
        for i in range(offset, end):
            if sent[i]:
                return i
        return offset

만약 문장의 끝부분까지 아는 단어가 존재하지 않을수도 있습니다. 이를 방지하기 위해 문장의 끝을 표시하는 eos 를 sent 에 추가합니다.

    sent = sentence_lookup(sentence)
    n_char = len(sent) + 1

    eos = ('EOS', 'EOS', 'EOS', n_char-1, n_char)
    sent.append([eos])

그리고 end index 로부터 시작하는 단어가 없을 경우, 가장 가까운 단어의 시작지점까지 부분어절을 잘라 'Unk' 태그를 부여합니다. 

    links = []
    for words in sent[:-1]:
        for word in words:
            begin = word[3]
            end = word[4]
            if not sent[end]:
                b = get_nonempty_first(sent, n_char, end)
                unk = (chars[end:b], 'Unk', 'Unk', end, b)
                links.append((word, unk))
            for adjacent in sent[end]:
                links.append((word, adjacent))

하지만 모르는 단어로부터 아는 단어까지의 링크가 아직 만들어지지 않았습니다. 품사가 'Unk' 인 단어들을 찾아 그 단어의 end index 로부터 시작하는 단어들과 링크를 만듭니다.

    unks = {to_node for _, to_node in links if to_node[1] == 'Unk'}
    for unk in unks:
        for adjacent in sent[unk[3]]:
            links.append((unk, adjacent))

마지막으로 그래프의 시작점에 bos 를 추가합니다. 문장의 맨 앞에 있는 단어들과 bos 간에 링크를 추가합니다.

    bos = ('BOS', 'BOS', 'BOS', 0, 0)
    for word in sent[0]:
        links.append((bos, word))

Shortest path 에서 bos 와 eos 는 시작 마디와 끝 마디로 이용됩니다. 이 값을 함께 return 합니다.

{% highlight python %}
def generate_link(sentence):

    def get_nonempty_first(sent, end, offset=0):
        for i in range(offset, end):
            if sent[i]:
                return i
        return offset

    chars = sentence.replace(' ','')
    sent = sentence_lookup(sentence)
    n_char = len(sent) + 1

    eos = ('EOS', 'EOS', 'EOS', n_char-1, n_char)
    sent.append([eos])

    i = get_nonempty_first(sent, n_char)
    if i > 0:
        sent[0].append((chars[:i], 'Unk', 'Unk', 0, i))

    links = []
    for words in sent[:-1]:
        for word in words:
            begin = word[3]
            end = word[4]
            if not sent[end]:
                b = get_nonempty_first(sent, n_char, end)
                unk = (chars[end:b], 'Unk', 'Unk', end, b)
                links.append((word, unk))
            for adjacent in sent[end]:
                links.append((word, adjacent))

    unks = {to_node for _, to_node in links if to_node[1] == 'Unk'}
    for unk in unks:
        for adjacent in sent[unk[3]]:
            links.append((unk, adjacent))

    bos = ('BOS', 'BOS', 'BOS', 0, 0)
    for word in sent[0]:
        links.append((bos, word))
    links = sorted(links, key=lambda x:(x[0][3], x[1][4]))

    return links, bos, eos
{% endhighlight %}

이 함수를 적용하여 만든 링크입니다. '것'이라는 단어는 알려지지 않았기 때문에 'Unk' 품사로 인식되었습니다. 또한 '였다'는 '이/Adjective + 었다/Eomi' 로 인식되어 하나의 마디를 이루고 있습니다. 이는 앞 단어 '예시'와 문장의 끝인 'EOS' 와 연결되어 있습니다.

    links, bos, eos = generate_link('이것은 예시였다')

    pprint(links)

    [(('BOS', 'BOS', 'BOS', 0, 0), ('이', 'Noun', 'Noun', 0, 1)),
     (('BOS', 'BOS', 'BOS', 0, 0), ('이', 'Josa', 'Josa', 0, 1)),
     (('BOS', 'BOS', 'BOS', 0, 0), ('이', 'Adjective', 'Adjective', 0, 1)),
     (('이', 'Noun', 'Noun', 0, 1), ('것', 'Unk', 'Unk', 1, 2)),
     (('이', 'Josa', 'Josa', 0, 1), ('것', 'Unk', 'Unk', 1, 2)),
     (('이', 'Adjective', 'Adjective', 0, 1), ('것', 'Unk', 'Unk', 1, 2)),
     (('BOS', 'BOS', 'BOS', 0, 0), ('이것', 'Noun', 'Noun', 0, 2)),
     (('이것', 'Noun', 'Noun', 0, 2), ('은', 'Josa', 'Josa', 2, 3)),
     (('은', 'Josa', 'Josa', 2, 3), ('예시', 'Noun', 'Noun', 3, 5)),
     (('예시', 'Noun', 'Noun', 3, 5), ('이 + 었다', 'Adjective', 'Eomi', 5, 7)),
     (('이 + 었다', 'Adjective', 'Eomi', 5, 7), ('EOS', 'EOS', 'EOS', 7, 8)),
     (('다', 'Josa', 'Josa', 6, 7), ('EOS', 'EOS', 'EOS', 7, 8)),
     (('다', 'Eomi', 'Eomi', 6, 7), ('EOS', 'EOS', 'EOS', 7, 8))]

만든 링크에 가중치를 부여합니다. HMM 의 모델을 그래프로 표현하기 위해서는 현재 마디로 유입되는 앞 마디의 단어로부터 지금 단어로 이동하는 transition probability 와 현재 마디의 단어, 품사가 발생할 emission probability 의 곱, 혹은 두 확률의 log 값의 합을 가중치로 이용하면 됩니다. HMM 학습 시 보지 못했던 단어와 transition 일 수 있기 때문에 _min_emission 과 _min_transition 을 이용하여 Key Error 를 방지합니다.

    w = emission.get(to_node[2], {}).get(morphs[0], _min_emission)
    w += transition.get((from_node[3], to_node[2]), _min_transition)

단, '였다 = 이/Adjective + 었다/Eomi' 의 경우, 하나의 마디에 두 개의 형태소가 포함된 구조이므
로, 이 때에는 마디 안에서의 transition 도 고려해야 합니다. tuple 로 표현된 마디의 첫번째 요소를 ' + '로 나눕니다. 이 길이가 2 라면 이는 '이 + 었다'처럼 두 형태소가 하나의 마디를 이루는 경우입니다.

    if len(morphs) == 2:
        w += emission.get(to_node[3], {}).get(morphs[1], _min_emission)
        w += transition.get(to_node[2], {}).get(to_node[3], _min_transition)

그리고 이 과정에서 다른 품사보다 명사를 더 선호한다거나, 길이가 짧은 단어에 페널티를 부여할 수도 있습니다.

{% highlight python %}
def add_weight(links):

    def weight(from_node, to_node):
        morphs = to_node[0].split(' + ')

        # score of first word
        w = emission.get(to_node[2], {}).get(morphs[0], _min_emission)
        w += transition.get((from_node[3], to_node[2]), _min_transition)

        # score of second word
        if len(morphs) == 2:
            w += emission.get(to_node[3], {}).get(morphs[1], _min_emission)
            w += transition.get(to_node[2], {}).get(to_node[3], _min_transition)
        return w

    graph = []
    for from_node, to_node in links:
        edge = (from_node, to_node, weight(from_node, to_node))
        graph.append(edge)
    return graph
{% endhighlight %}

링크에 가중치를 더한 결과입니다. 미등록 단어인 '것'과 연결된 경우에는 가장 작은 가중치가 부여되었습니다.

    graph = add_weight(links)

    pprint(graph)

    [(('BOS', 'BOS', 'BOS', 0, 0), ('이', 'Noun', 'Noun', 0, 1), 0.15),
     (('BOS', 'BOS', 'BOS', 0, 0), ('이', 'Josa', 'Josa', 0, 1), 0.1),
     (('BOS', 'BOS', 'BOS', 0, 0), ('이', 'Adjective', 'Adjective', 0, 1), 0.1),
     (('이', 'Noun', 'Noun', 0, 1), ('것', 'Unk', 'Unk', 1, 2), 0.0),
     (('이', 'Josa', 'Josa', 0, 1), ('것', 'Unk', 'Unk', 1, 2), 0.0),
     (('이', 'Adjective', 'Adjective', 0, 1), ('것', 'Unk', 'Unk', 1, 2), 0.0),
     (('BOS', 'BOS', 'BOS', 0, 0), ('이것', 'Noun', 'Noun', 0, 2), 0.1),
     (('이것', 'Noun', 'Noun', 0, 2), ('은', 'Josa', 'Josa', 2, 3), 0.2),
     (('은', 'Josa', 'Josa', 2, 3), ('예시', 'Noun', 'Noun', 3, 5), 0.1),
     (('예시', 'Noun', 'Noun', 3, 5), ('이 + 었다', 'Adjective', 'Eomi', 5, 7), 0.0),
     (('이 + 었다', 'Adjective', 'Eomi', 5, 7), ('EOS', 'EOS', 'EOS', 7, 8), 0.0),
     (('다', 'Josa', 'Josa', 6, 7), ('EOS', 'EOS', 'EOS', 7, 8), 0.0),
     (('다', 'Eomi', 'Eomi', 6, 7), ('EOS', 'EOS', 'EOS', 7, 8), 0.0)]

### 포드 알고리즘을 이용한 최단경로 찾기

앞서 shortest path 와 HMM decoding 이 같은 문제임을 이야기했던 [포스트][ford_for_pos]에서 만든 ford_list 함수를 이용합니다. 포드 알고리즘에 대한 설명은 이전 [포스트][ford_for_pos]를 참고하세요.

Shortest path 는 d[u] + w(u,v) < d[v] 이면 d[v] = d[u] + w(u,v) 로 대체합니다. 이를 d[u] + w(u,v) > d[v] 이면 d[v] = d[u] + w(u,v) 이 되도록 변경하면 longest path 를 구하는 함수로 바꿀 수 있습니다. HMM 은 log 확률 합의 쵀대값을 지니는 sequence 를 찾습니다. 이를 위하여 아래의 ford_list 함수는 longest path 를 찾도록 식을 변형하였습니다.

{% highlight python %}
def ford_list(E, V, S, T):

    ## Initialize ##
    # (max weight + 1) * num of nodes
    inf = (min((weight for from_, to_, weight in E)) - 1) * len(V)

    # distance
    d = {node:0 if node == S else inf for node in V}
    # previous node
    prev = {node:None for node in V}

    ## Iteration ##
    # preventing infinite loop
    for _ in range(len(V)):
        # for early stop
        changed = False
        for u, v, Wuv in E:
            d_new = d[u] + Wuv
            if d_new > d[v]:
                d[v] = d_new
                prev[v] = u
                changed = True
        if not changed:
            break

    # Checking negative cycle loop
    for u, v, Wuv in E:
        if d[u] + Wuv > d[v]:
            raise ValueError('Cycle exists')

    # Finding path
    prev_ = prev[T]
    if prev_ == S:
        return {'paths':[[prev_, S][::-1]], 'cost':d[T]}

    path = [T]
    while prev_ != S:
        path.append(prev_)
        prev_ = prev[prev_]
    path.append(S)

    return path[::-1], d[T]

nodes = {node for edge in graph for node in edge[:2]}

# choose optimal sequence
path, cost = ford_list(graph, nodes, bos, eos)
{% endhighlight %}

앞서 생성한 그래프를 ford_list 함수에 입력하면 품사 판별이 된 문장이 선택됩니다.

    path

    [('BOS', 'BOS', 'BOS', 0, 0),
     ('이것', 'Noun', 'Noun', 0, 2),
     ('은', 'Josa', 'Josa', 2, 3),
     ('예시', 'Noun', 'Noun', 3, 5),
     ('이 + 었다', 'Adjective', 'Eomi', 5, 7),
     ('EOS', 'EOS', 'EOS', 7, 8)]

### As morphological analysis results

위 결과에서 '이 + 었다'를 각각 하나의 형태소로 분해하면 형태소 분석의 결과가 됩니다. 반대로 '이 + 었다' 를 '였다/Adjective' 로 인식하면 품사 판별이 됩니다. 형태소 분석 결과로 만들기 위해서 ' + ' 를 기준으로 단어를 나눕니다.

{% highlight python %}
def flatten(path):
    pos = []
    for word, tag0, tag1, b, e in path:
        morphs = word.split(' + ')
        pos.append((morphs[0], tag0))
        if len(morphs) == 2:
            pos.append((morphs[1], tag1))
    return pos

pos = flatten(path)
{% endhighlight %}

    pos

    [('BOS', 'BOS'),
     ('이것', 'Noun'),
     ('은', 'Josa'),
     ('예시', 'Noun'),
     ('이', 'Adjective'),
     ('었다', 'Eomi'),
     ('EOS', 'EOS')]

이번에는 미등록 단어 'tt' 가 포함된 문장을 넣어 형태소 분석을 한 결과까지 한 번에 얻어봅니다.

{% highlight python %}
links, bos, eos = generate_link('tt도예시였다')
graph = add_weight(links)
nodes = {node for edge in graph for node in edge[:2]}
path, cost = ford_list(graph, nodes, bos, eos)
pos = flatten(path)
{% endhighlight %}

'tt' 가 미등록 단어로 인식되었습니다.

    pos

    [('BOS', 'BOS'),
     ('tt', 'Unk'),
     ('도', 'Josa'),
     ('예시', 'Noun'),
     ('이', 'Adjective'),
     ('었다', 'Eomi'),
     ('EOS', 'EOS')]

### Inferring tag of unknown words

이번에는 tt 의 품사를 추정하는 함수를 만듭니다. 'tt' 는 문장의 맨 앞에 시작했으며 뒤에 '-도/Josa'가 위치하기 때문에 명사일 가능성이 높습니다. 이를 HMM 의 식으로 표현하면 앞 단어의 품사에서 tt 가 가질 수 있는 품사로의 transtion probability 와 tt 가 가질 수 있는 품사에서 다음 단어로의 transition probability 의 곱이 가장 큰 품사를 찾으면 'Noun' 이라는 의미입니다.

    for i, pos_i in enumerate(pos[:-1]):

        ...
        # previous -> current transition
        if i == 1:
            tag_prob = {tag:prob for tag, prob in begin.items()}
        else:
            tag_prob = {
                tag:prob for (prev_tag, tag), prob in transition.items()
                if prev_tag == pos[i-1][1]
            }

        # current -> next transition
        for (tag, next_tag), prob in transition.items():
            if next_tag == pos[i+1][1]:
                tag_prob[tag] = tag_prob.get(tag, 0) + prob

만약 앞, 뒤 단어의 품사들조차 이용할 수 없는 상황이라면 이를 명사로 추정합니다. 한국어 단어 중 명사가 가장 많은 단어를 보유하고 있기 때문에 이러한 추정은 자연스럽습니다.

    for i, pos_i in enumerate(pos[:-1]):
        if not tag_prob:
            infered_tag = 'Noun'
        else:
            infered_tag = sorted(tag_prob, key=lambda x:-tag_prob[x])[0]

{% highlight python %}
def inference_unknown(pos):
    pos_ = []
    for i, pos_i in enumerate(pos[:-1]):
        if not (pos_i[1] == 'Unk'):
            pos_.append(pos_i)
            continue

        # previous -> current transition
        if i == 1:
            tag_prob = {tag:prob for tag, prob in begin.items()}
        else:
            tag_prob = {
                tag:prob for (prev_tag, tag), prob in transition.items()
                if prev_tag == pos[i-1][1]
            }

        # current -> next transition
        for (tag, next_tag), prob in transition.items():
            if next_tag == pos[i+1][1]:
                tag_prob[tag] = tag_prob.get(tag, 0) + prob

        if not tag_prob:
            infered_tag = 'Noun'
        else:
            infered_tag = sorted(tag_prob, key=lambda x:-tag_prob[x])[0]
        pos_.append((pos_i[0], infered_tag))

    return pos_ + pos[-1:]
{% endhighlight %}

추정 함수를 거친 결과 'tt' 는 명사로 인식됩니다.

    inference_unknown(pos)

    [('BOS', 'BOS'),
     ('tt', 'Noun'),
     ('도', 'Josa'),
     ('예시', 'Noun'),
     ('이', 'Adjective'),
     ('었다', 'Eomi'),
     ('EOS', 'EOS')]

위 결과의 앞/뒤에 존재하는 BOS 와 EOS 를 제거하여 형태소 분석 결과를 정리합니다.

{% highlight python %}
def postprocessing(pos):
    return pos[1:-1]
{% endhighlight %}

### 사용자 사전 추가

HMM 기반 모델은 사용자 사전을 추가하기가 쉽습니다. 추가할 (단어, 품사)를 emission probability 에 추가만 하여도 되기 때문입니다.

혹은 존재하는 단어라 하더라도 사용자가 원하는 선호도를 score 로 업데이트 할 수도 있습니다.

{% highlight python %}
def add_user_dictionary(word, tag, score):
    if not (tag in emission):
        emission[tag] = {word: score}
    else:
        emission[tag][word] = score
{% endhighlight %}


## HMMTagger

{% highlight python %}
class HMMTagger:
    def __init__(self, emission, transition, begin):
        self.emission = emission
        self.transition = transition
        self.begin = begin

        self._max_word_len = max(
            len(w) for words in emission.values() for w in words)
        self._min_emission = min(
            s for words in emission.values() for s in words.values()) - 0.05
        self._min_transition = min(transition.values()) - 0.05

    def tag(self, sentence):
        # lookup & generate graph
        links, bos, eos = self._generate_link(sentence)
        graph = self._add_weight(links)

        # find optimal path
        nodes = {node for edge in graph for node in edge[:2]}
        path, cost = ford_list(graph, nodes, bos, eos)
        pos = self._flatten(path)

        # infering tag of unknown words
        pos = self._inference_unknown(pos)

        # post processing
        pos = self._postprocessing(pos)

        return pos

    def _sentence_lookup(self, sentence):
        sent = []
        for eojeol in sentence.split():
            sent += self._eojeol_lookup(eojeol, offset=len(sent))
        return sent

    def _eojeol_lookup(self, eojeol, offset=0):
        n = len(eojeol)
        pos = [[] for _ in range(n)]
        for b in range(n):
            for r in range(1, self._max_word_len+1):
                e = b+r
                if e > n:
                    continue
                surface = eojeol[b:e]
                for tag in self._get_pos(surface):
                    pos[b].append((surface, tag, tag, b+offset, e+offset))
                for i in range(1, r + 1):
                    suffix_len = r - i
                    try:
                        lemmas = self._lemmatize(surface, i)
                        if lemmas:
                            for morphs, tag0, tag1 in lemmas:
                                pos[b].append((morphs, tag0, tag1, b+offset, e+offset))
                    except:
                        continue
        return pos

    def _get_pos(self, sub):
        tags = []
        for tag, words in self.emission.items():
            if sub in words:
                tags.append(tag)
        return tags

    def _lemmatize(self, word, i):
        l = word[:i]
        r = word[i:]
        lemmas = []
        len_word = len(word)
        for l_, r_ in lemma_candidate(l, r):
            word_ = l_ + ' + ' + r_
            if (l_ in self.emission['Verb']) and (r_ in self.emission['Eomi']):
                lemmas.append((word_, 'Verb', 'Eomi'))
            if (l_ in self.emission['Adjective']) and (r_ in self.emission['Eomi']):
                lemmas.append((word_, 'Adjective', 'Eomi'))
        return lemmas

    def _generate_link(self, sentence):

        def get_nonempty_first(sent, end, offset=0):
            for i in range(offset, end):
                if sent[i]:
                    return i
            return offset

        chars = sentence.replace(' ','')
        sent = self._sentence_lookup(sentence)
        n_char = len(sent) + 1

        eos = ('EOS', 'EOS', 'EOS', n_char-1, n_char)
        sent.append([eos])

        i = get_nonempty_first(sent, n_char)
        if i > 0:
            sent[0].append((chars[:i], 'Unk', 'Unk', 0, i))

        links = []
        for words in sent[:-1]:
            for word in words:
                begin = word[3]
                end = word[4]
                if not sent[end]:
                    b = get_nonempty_first(sent, n_char, end)
                    unk = (chars[end:b], 'Unk', 'Unk', end, b)
                    links.append((word, unk))
                for adjacent in sent[end]:
                    links.append((word, adjacent))

        unks = {to_node for _, to_node in links if to_node[1] == 'Unk'}
        for unk in unks:
            for adjacent in sent[unk[3]]:
                links.append((unk, adjacent))

        bos = ('BOS', 'BOS', 'BOS', 0, 0)
        for word in sent[0]:
            links.append((bos, word))
        links = sorted(links, key=lambda x:(x[0][3], x[1][4]))

        return links, bos, eos

    def _add_weight(self, links):

        def weight(from_node, to_node):
            morphs = to_node[0].split(' + ')

            # score of first word
            w = self.emission.get(to_node[2], {}).get(morphs[0], _min_emission)
            w += self.transition.get((from_node[3], to_node[2]), _min_transition)

            # score of second word
            if len(morphs) == 2:
                w += self.emission.get(to_node[3], {}).get(morphs[1], _min_emission)
                w += self.transition.get(to_node[2], {}).get(to_node[3], _min_transition)
            return w

        graph = []
        for from_node, to_node in links:
            edge = (from_node, to_node, weight(from_node, to_node))
            graph.append(edge)
        return graph

    def _flatten(self, path):
        pos = []
        for word, tag0, tag1, b, e in path:
            morphs = word.split(' + ')
            pos.append((morphs[0], tag0))
            if len(morphs) == 2:
                pos.append((morphs[1], tag1))
        return pos

    def _inference_unknown(self, pos):
        pos_ = []
        for i, pos_i in enumerate(pos[:-1]):
            if not (pos_i[1] == 'Unk'):
                pos_.append(pos_i)
                continue

            # previous -> current transition
            if i == 1:
                tag_prob = {tag:prob for tag, prob in self.begin.items()}
            else:
                tag_prob = {
                    tag:prob for (prev_tag, tag), prob in self.transition.items()
                    if prev_tag == pos[i-1][1]
                }

            # current -> next transition
            for (tag, next_tag), prob in self.transition.items():
                if next_tag == pos[i+1][1]:
                    tag_prob[tag] = tag_prob.get(tag, 0) + prob

            if not tag_prob:
                infered_tag = 'Noun'
            else:
                infered_tag = sorted(tag_prob, key=lambda x:-tag_prob[x])[0]
            pos_.append((pos_i[0], infered_tag))

        return pos_ + pos[-1:]

    def _postprocessing(self, pos):
        return pos[1:-1]

    def add_user_dictionary(self, word, tag, score):
        if not (tag in self.emission):
            self.emission[tag] = {word: score}
        else:
            self.emission[tag][word] = score
{% endhighlight %}

정리한 품사 판별기를 이용하여 앞의 예시 문장의 형태소 분석 결과를 다시 확인합니다.

{% highlight python %}
hmm_tagger = HMMTagger(emission, transition, begin)
hmm_tagger.tag('tt도예시였다')

# [('tt', 'Noun'),
#  ('도', 'Josa'),
#  ('예시', 'Noun'),
#  ('이', 'Adjective'),
#  ('었다', 'Eomi')]
{% endhighlight %}

## Package

이 과정에 품사의 선호나 1 음절 단어에 대한 패널티 기능을 추가한 코드를 [github][hmm_postagger_git] 에 올려두었습니다.

## Reference

- Brants, T. (2000, April). [TnT: a statistical part-of-speech tagger][tnt]. In Proceedings of the sixth conference on Applied natural language processing (pp. 224-231). Association for Computational Linguistics.

[ford_for_pos]: {{ site.baseurl }}{% link _posts/2018-08-21-ford_for_pos.md %}
[hmm_tagger]: {{ site.baseurl }}{% link _posts/2018-09-11-hmm_based_tagger.md %}
[lemmatizer]: {{ site.baseurl }}{% link _posts/2018-06-07-lemmatizer.md %}
[sejong_cleaner]: https://github.com/lovit/sejong_corpus_cleaner/
[hmm_postagger_git]: https://github.com/lovit/hmm_postagger/