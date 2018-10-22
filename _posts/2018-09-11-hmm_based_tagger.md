---
title: Hidden Markov Model (HMM) 기반 품사 판별기의 원리와 문제점
date: 2018-09-11 23:00:00
categories:
- nlp
tags:
- tokenizer
- sequential labeling
---

Hidden Markov Model (HMM) 은 Conditional Random Field (CRF) 가 제안되기 이전에 Part of Speech tagging 과 같은 sequential labeling 에 자주 이용되던 알고리즘입니다. HMM 은 CRF 와 비교하여, unsupervised learning 도 할 수 있다는 장점이 있습니다만, tagger 를 만들 때에는 주로 학습 말뭉치를 이용합니다. 그리고 학습 말뭉치를 이용할 때에도 HMM 이 CRF 보다 빠른 학습속도를 보여줍니다. 그러나 HMM 은 단어열의 문맥을 기반으로 품사를 추정하지 못합니다. 이는 unguaranteed independence problem 이라는 HMM 의 구조적 특징 때문입니다. 이 포스트에서는 HMM 기반의 품사 판별기의 원리와 학습 방법에 대하여 이야기합니다. 단, HMM 의 unsupervised learning 과 decoding 과정에 대해서는 다루지 않습니다.

## Tokenization, Part of speech tagging, Morphological analysis, Sequential Labeling ?

**토크나이징 (tokenization)** 은 주어진 문장을 토큰 (tokens) 으로 나누는 과정입니다. 토큰은 상황에 따라 다르게 정의할 수 있습니다. 띄어쓰기를 기준으로 문장을 나눌 수도 있습니다. 영어는 띄어쓰기를 기준으로 문장을 나눠도 단어열로 나뉘어집니다.  

{% highlight python %}
sent = '너무너무너무는 아이오아이의 노래입니다'
tokens = ['너무너무너무는', '아이오아이의', '노래입니다']
{% endhighlight %}

한국어에서 띄어쓰기 기준으로 나뉘어지는 단위는 '어절' 입니다. 어절은 한 개 이상의 단어, 혹은 형태소로 구성될 수 있습니다. 그렇기 때문에 '너무너무너무는', '너무너무너무가' 처럼 같은 단어를 포함함에도 서로 다른 형태의 어절이 등장할 수 있습니다. 또한 띄어쓰기 기준으로 토큰을 나누면 띄어쓰기 오류에 취약합니다. 핸드폰으로 입력되는 텍스트들에서는 띄어쓰기가 잘 지켜지지 않는 경우들이 많습니다. 여러 이유로 한국어 텍스트를 띄어쓰기 기준으로 나누는 것은 좋지 않습니다. 

**품사 판별 (part of speech tagging)**은 토큰을 (단어, 품사)로 정의합니다. 아래처럼 각 단어가 품사와 함께 분리되야 합니다. 

{% highlight python %}
tokens = [
    ('너무너무너무', '명사'),
    ('는', '조사'),
    ('아이오아이', '명사'),
    ('의', '조사'),
    ('노래', '명사'),
    ('입니다', '형용사')
]
{% endhighlight %}

한국어의 품사 체계는 5언 9품사로 구성되어 있습니다. 다른 단어들은 형태가 변하지 않습니다. 하지만 동사, 형용사인 용언은 형태가 변합니다. 이를 '용언의 활용' 이라 합니다. '이다' 라는 형용사는 '이고, 이니까, 이었다' 처럼 본래의 의미는 같지만 형태가 변합니다. 이때 의미를 지니는 부분을 어근 (root), 형태가 변하는 부분을 어미 (ending) 이라 합니다. '이었다 = 이/어근 + 었다/어미' 입니다. '이었다'는 형용사로 단어이지만, '이/어근', '었다/어미'는 단어가 아닙니다. 이들을 형태소라 합니다. 

| 언 | 품사 |
| --- | --- |
| 체언 | 명사, 대명사, 수사 |
| 수식언 | 관형사, 부사 |
| 관계언 | 조사 |
| 독립언 | 감탄사 |
| 용언 | 동사, 형용사 |

**형태소 분석은 (morphological analysis)** 품사 판별과 자주 혼동되는 개념입니다. 형태소란 의미를 지니는 최소 단위로, 형태소들이 모여 단어를 구성합니다. 5언 9품사에서 용언을 제외한 7 품사는 그 자체로 형태소이기도 합니다. 용언은 형태소인 어간과 어미로 구성되어 있습니다. 어절 '따뜻한'은 '따뜻하/어간 + ㄴ/어미'로 구성됩니다.

위 문장에 대하여 [너무너무너무/명사, 는/조사, 아이오아이/명사, 의/조사, 노래/명사, **입니다/형용사**] 가 출력된다면 품사 판별입니다.

하지만 [너무너무너무/명사, 는/조사, 아이오아이/명사, 의/조사, 노래/명사, **이/어간, ㅂ니다/어미**] 가 출력된다면 이는 형태소 분석입니다

품사 판별과 형태소 분석은 **sequential labeling** 문제로 이해할 수도 있습니다. 품사 판별은 문장이 단어열로 나뉘어 진 뒤, 각 단어의 품사를 부여하는 것이며, 형태소 분석은 문장이 형태소열로 나뉘어 진 뒤, 각각의 형태소를 부여하는 것입니다.

이들을 sequential labeing 문제로 이해하기 위해서는 일단 문장이 단어열, 혹은 형태소열로 나뉘어져 있어야 합니다. 그런데 '서울대공원' = ['서울대 + 공원', '서울 + 대공원'] 처럼 문장을 단어열로 나누는 과정에서도 모호성이 발생하며, '은/조사, 은/어미'와 같이 품사에서도 모호성이 발생합니다. 한국어의 문장에 대한 전처리 과정은 문장을 단어/형태소열로 분해하는 segmentation 과정과 sequential labeling 과정을 복합적으로 풀어야 하는 문제입니다. 게다가 우리가 실제로 분석하는 데이터에는 띄어쓰기와 오탈자 오류가 많으며 신조어에 의한 [미등록 단어 문제][oov]도 발생합니다. 그래서 한국어 분석의 전처리 과정이 어려운 문제입니다.

여하튼, 품사 판별 혹은 형태소 분석을 할 때, 문장을 단어/형태소 열로 분해 (lookup)하는 과정을 끝냈다면, input sequence 에 대한 sequential labeling 을 수행해야 합니다. 초기에는 Hidden Markov Model (HMM) 이 이를 위해 자주 이용되었으나, 몇 년 뒤, Maximum Entropy Markov Model (MEMM) 을 거쳐 Conditional Random Field (CRF) 가 제안되면서 locality 를 이용해야 하는 sequential labeling 에는 CRF 가 HMM 을 거의 대체하였습니다.

이번 포스트에서는 HMM 이 CRF 로 대체되는 결정적인 이유인 unguaranteed independency problem 에 대하여 알아보고, 그 이전에 HMM 기반의 tagger 들이 작동하는 방식과 학습 방법에 대해서 알아봅니다.

## Hidden Markov Model (HMM)

Hidden Markov Model (HMM) 은 길이가 $$n$$ 인 sequence $$x_{1:n} = [x_1, x_2, \dots, x_n]$$ 에 대하여 $$P(y_{1:n} \vert x_{1:n})$$ 가 가장 큰 $$y_{1:n}$$ 를 찾습니다. 품사 판별의 문제에서는 $$n$$ 개의 단어로 구성된 단어열에 대하여 각 단어의 품사 $$y$$ 를 부여하는 것입니다. 이 과정을 HMM 의 decoding 이라 합니다. 그리고 우리가 찾아야 하는 label, $$y$$ 를 HMM 에서는 state 라 합니다.

이 때 $$P(y_{1:n} \vert x_{1:n})$$ 는 다음처럼 계산됩니다.

$$P(y_{1:n} \vert x_{1:n}) = P(x_1 \vert y_1) \times P(y_1 \vert START) \times P(y_2 \vert y_1) \times P(x_2 \vert y_2) \cdots$$

위의 식은 현재 시점 $$i$$ 의 state 인 $$y_i$$ 를 판별 (classification, or labeling) 하기 위하여 이전 시점의 state 인 $$y_{i-1}$$ 이 이용됩니다. 이처럼 이전의 한 단계 전의 state 정보를 이용하는 모델을 first-order Markov Model 이라 합니다. 만약 이전 두 단계의 states 정보를 모두 이용한다면 $$P(y_i \vert y_{i-2}, y_{i-1}$$ 이 학습되어야 하며, 이는 second-order Markov Model 이라 합니다. 그 외의 멀리 떨어진 state 정보는 이용하지 않습니다.

이처럼 state 간의 변화 확률을 transition probability (전이 확률) 라 합니다. HMM 은 각 state 에서 우리가 관측 가능한 값 (observation) 이 발생할 확률이 있다고 가정합니다. 이를 emission probability, $$P(x_i \vert y_i)$$ 라 합니다. 품사 판별에서는 명사 집합에서 '아이오아이'라는 단어가 존재할 확률 입니다.

숫자 계산에서 곱셈은 덧셈보다 비싼 작업입니다. 그렇기 때문에 확률을 곱하는 작업들은 주로 log 를 씌워 덧셈으로 변환합니다. 위 수식은 아래처럼 변환됩니다.

$$log P(y_{1:n} \vert x_{1:n}) = log P(x_1 \vert y_1)+ log P(y_1 \vert S) + log P(y_2 \vert y_1) + log P(x_2 \vert y_2) \cdots$$

### Supervised training

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

그런데 위의 식이 직관적일 뿐 아니라 Maximum Likelihood Estimation (MLE) 관점에서도 확률을 학습하는 solution 입니다. 즉, 학습 말뭉치에서의 빈도수를 계산하는 것만으로 학습을 할 수 있습니다.

우리는 corpus 가 nested list 구조라 가정합니다. 각 문장 sent 는 [(word, tag), (word, tag), ... ] 형식입니다. 우리가 해야 할 일은 각 태그 별로 단어가 발생한 횟수와 $$tag_{i-1}, tag_{i}$$ 의 횟수를 세는 것 뿐입니다.

특별히 문장이 시작할 때의 state 횟수를 bos 에 저장합니다. 문장의 마지막 state 의 다음에 문장이 끝남을 저장하기 위하여 trans 에 마지막 state 에서 EOS 로의 횟수를 저장합니다.

{% highlight python %}
from collections import defaultdict

pos2words = defaultdict(lambda: defaultdict(int))
trans = defaultdict(int)
bos = defaultdict(int)

# sent = [(word, tag), (word, tag), ... ] format
for sent in corpus:

    # generation prob
    for word, pos in sent:
        pos2words[pos][word] += 1

    # transition prob
    for bigram in as_bigram_tag(sent):
        trans[bigram] += 1

    # begin prob (BOS -> tag)
    bos[sent[0][1]] += 1

    # end prob (tag -> EOS)
    trans[(sent[-1][1], 'EOS')] += 1
{% endhighlight %}

문장 sent 에서 tag sequence 만 취하여 이를 bigram 으로 묶는 as_bigram_tag 함수를 만듭니다. tag sequence 인 poslist 를 만들고 이를 zip(postlist, poslist[1:]) 로 묶음으로써 (이전 품사, 현재 품사)를 yield 하여 bigram sequence 를 만듭니다. Transition probability 인 trans 에 이 횟수를 계산합니다.

{% highlight python %}
def as_bigram_tag(wordpos):
    poslist = [pos for _, pos in wordpos]
    return [(pos0, pos1) for pos0, pos1 in zip(poslist, poslist[1:])]
{% endhighlight %}

우리가 계산한 것은 빈도수 입니다. 이를 확률로 변형합니다. Emission probability 는 각 품사의 빈도수로 (단어, 품사) 발생 횟수를 나눠주면 됩니다. Transition probability 역시 이전 품사의 빈도수인 base[pos0] 으로 (pos0, pos1) 의 빈도수를 나눠주면 됩니다.

그리고 앞서 언급한 것처럼 float 는 곱셈보다 덧셈이 계산이 빠르기 때문에, 미리 math.log 를 이용하여 확률값을 log probability 로 변환하여 줍니다.

{% highlight python %}
import math

def _to_log_prob(pos2words, transition, bos):

    # observation
    base = {pos:sum(words.values()) for pos, words in pos2words.items()}
    pos2words_ = {pos:{word:math.log(count/base[pos]) for word, count in words.items()}
                  for pos, words in pos2words.items()}

    # transition
    base = defaultdict(int)
    for (pos0, pos1), count in transition.items():
        base[pos0] += count
    transition_ = {pos:math.log(count/base[pos[0]]) for pos, count in transition.items()}

    # bos
    base = sum(bos.values())
    bos_ = {pos:math.log(count/base) for pos, count in bos.items()}
    return pos2words_, transition_, bos_
{% endhighlight %}

Sequential labeling 을 위한 HMM model 의 supervised learning 은 모두 끝났습니다.

### Tagging using HMM

학습한 HMM 모델을 이용하여 word sequence 에 대한 확률값을 계산해봅니다.

사용 가능한 형태소 분석기나 품사 판별기를 만들려면 문장에서 단어열을 만드는 look-up 부분을 구현해야 합니다. 또한 가능한 문장 후보 (단어, 품사)열을 모두 열거한 다음에 각 후보의 확률을 계산하지도 않습니다. 문장이 조금만 길어져도 (단어, 품사) 열의 후보 개수는 기하급수적으로 늘어나며, 중복된 계산이 많아지기 때문입니다. 이때는 Viterbi style 의 decode 함수를 구현해야 합니다. 혹은 [이전 포스트][ford_for_pos]에서 다룬 것처럼 bigram HMM 이라면 shortest path 의 해를 찾는 Ford algorithm 을 이용할 수도 있습니다.

지금은 HMM tagger 의 작동 원리를 이해하는 것이 목적이므로, 주어진 단어열의 확률, 정확히는 log probability 를 계산하는 sentence_log_prob 함수를 구현합니다.

이 함수는 각각 emission, transition probability 를 누적하여 더합니다. 우리는 이전에 모든 확률을 log 로 바꿔서 학습했기 때문에 prob. 의 누적곲 대신 log prob. 의 누적합으로 빠르게 문장의 확률을 계산할 수 있습니다.

한 가지, 모르는 단어가 등장할 때의 penalty 를 줘야 합니다. 학습 데이터를 이용할 때에는 학습 시 보지 못했던 단어나 state transition 의 probability 는 0 이고, 이의 log 값은 negative infinite 입니다. 이 값은 덧셈을 할 수 없으니, 충분히 작은 값으로 이를 대체합니다.

또 한가지, 한 문장에 대하여 (단어, 품사) 열의 길이가 다를 경우, 긴 문장이 손해를 봅니다. 확률은 1 보다 작은 값입니다. 이 값을 누적하여 곱할수록 계속 작아집니다. log prob. 의 경우에는 음수의 값을 더하기 때문에 더 많이 더할수록 그 값이 작아집니다. 일종의 length bias 가 발생합니다. 이를 방지하기 위하여 log prob. 의 누적합을 (단어, 품사)열의 길이로 나눠 length normalization 을 하였습니다.

{% highlight python %}
class TrainedHMM:
    def __init__(self, pos2words, transition, bos):
        self.pos2words = pos2words
        self.transition = transition
        self.bos = bos

        self.unknown_penalty=-15
        self.eos_tag='EOS'

    def sentence_log_prob(self, sent):
        # emission probability
        log_prob = sum(
            (self.pos2words.get(t, {}).get(w,self.unknown_penalty)
             for w, t in sent)
        )

        # bos
        log_prob += tagger.bos.get(sent[0][1], self.unknown_penalty)

        # transition probability
        bigrams = [(t0, t1) for (_, t0), (_, t1) in zip(sent, sent[1:])]
        log_prob += sum(
            (self.transition.get(bigram, self.unknown_penalty)
             for bigram in bigrams))

        # eos
        log_prob += self.transition.get(
            (sent[-1][1], self.eos_tag), self.unknown_penalty
        )

        # length normalization
        log_prob /= len(sent)

        return log_prob

trained_hmm = TrainedHMM(pos2words_, transition_, bos_)
{% endhighlight %}

세종 말뭉치를 이용하여 학습한 bigram HMM tagger 를 이용하여 '뭐 타고가' 라는 예문의 네 가지 (단어, 품사)열 후보에 대한 확률값을 계산합니다.

{% highlight python %}
candidates = [
    [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Eomi'), ('가', 'Noun')],
    [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Noun'), ('가', 'Noun')],
    [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Eomi'), ('가', 'Verb'), ('ㅏ', 'Eomi')],
    [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Noun'), ('가', 'Verb'), ('ㅏ', 'Eomi')]
]

for sent in candidates:    
    print('\n{}'.format(sent))
    print(trained_hmm.sentence_log_prob(sent))
{% endhighlight %}

그 결과 정답인 [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Eomi'), ('가', 'Verb'), ('ㅏ', 'Eomi')] 가 가장 큰 log prob. 값을 지님을 확인할 수 있습니다. Length normalization 을 하지 않으면 [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Eomi'), ('가', 'Noun')] 가 가장 큰 log prob. 를 지닙니다.

    [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Eomi'), ('가', 'Noun')]
    -8.136057979789241

    [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Noun'), ('가', 'Noun')]
    -12.146730944852035

    [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Eomi'), ('가', 'Verb'), ('ㅏ', 'Eomi')]
    -6.575814992086488

    [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Noun'), ('가', 'Verb'), ('ㅏ', 'Eomi')]
    -9.820101102795197

위의 예제로부터 우리는 몇 가지 bigram HMM tagger 의 어려움을 예측할 수 있습니다. 미등록 단어가 등장하면 unknown penalty 에 의하여 매우 작은 값이 부여됩니다. HMM model 은 왠만하면 모르는 단어를 아는 단어로 분해하는 쪽을 선택할 것입니다. 그래서 '아이오아이'라는 신조어를 '아이/명사 + 오/명사 + 아이/명사' 나 '아이오/명사 + 아이/명사' 로 분해하는 것입니다. 이것이 이전의 [pos tagger and oov 포스트][oov] 에서 다뤘던 supervised learning tagger 가 미등록단어 문제를 발생시키는 이유입니다. 물론 이를 어느 정도 해결할 수 있는 트릭들도 존재합니다만, 구조적으로 미등록단어 문제를 야기할 수 밖에 없습니다.

두번째는 state 마다 emission probability 의 분포가 다릅니다. 세종 말뭉치 기준 보통 명사(NNG) 의 경우에는 90k 개의 unique words 가 등록되어 있습니다. 그리고 각 단어마다 평균 60 번 정도 등장합니다. 그러나 부사격 조사 (JKB) 는 125 개의 unique words 로 이뤄져 있으며, 평균 8k 번 등장합니다. 그리고 특정 단어가 그 중에서도 주로 이용됩니다. 그렇다면 $$P(w \vert NNG)$$ 와 $$P(w \vert JKB)$$ 의 크기의 차이가 다를 수 있습니다. 어떤 단어가 보통 명사 (NNG) 이기도 하고 부사격 조사 (JKB) 이기도 하다면, 부사격 조사를 선택하는 편이 문장의 확률이 높게 됩니다. 앞/뒤의 맥락을 고려하지 않고 말이죠.

      NNG: unique= 90298, frequency= 5423295, freq/unique= 60.06
       VV: unique=  6603, frequency= 1847882, freq/unique= 279.85
       EC: unique=  1816, frequency= 1774216, freq/unique= 976.99
      JKB: unique=   125, frequency= 1023071, freq/unique= 8184.57
      ...

또한 보통 명사 (NNG) 는 흔하게 등장하기 때문에 다른 state 로의 transition probability 들도 작습니다. 하지만, 아주 가끔만 등장하는 품사는 transition probability 의 값들도 클 것입니다. 즉, HMM model 에서는 정답이 아니더라도, 희귀한 품사를 선택하는 것에 확률적으로 유리한 경우가 많습니다. 이런 문제를 **label bias**라 합니다.

### Unguaranteed Independency Problem

앞서 label bias 를 이야기하면서 흔하지 않은 품사 (state) 를 택하는 것이 정답이 아니더라도 확률적으로 유리할 수 있다는 설명을 하면서 **앞/뒤의 맥락도 고려하지 않고**라는 표현을 이용하였습니다. 우리가 단어의 품사를 추정할 때에는 앞, 뒤의 단어를 살펴봅니다.

아래의 예시에서처럼 '이'는 명사일 수도 조사일 수도 있습니다. 모호성을 해결해주는 힌트는 '이'라는 단어 다음에 '가/Josa - 빠졌어/Verb' 가 위치하기 때문이며, 두번째 문장에서는 '이' 앞에 '남/Noun', 그 뒤에 '빼줬어/Verb'가 위치하기 때문입니다. 우리가 관측 가능한 단어, 즉 $$x_i$$ 와 $$x_{i-1}, x_{i+1}$$ 혹은 더 넓은 문맥인 $$x_{i-2:i-1}$$ 이나 $$x_{i+1:i+2}$$ 가 서로 상관이 있다는 의미입니다.

    이, 가, 빠졌어 -> [Noun, Josa, Verb]
    남, 이, 빼줬어 -> [Noun, Josa, Verb]

하지만 HMM model 에서 $$x_i$$ 와 $$x_j$$ 는 직접적으로 연관성을 지닐 수가 없습니다. 오로직 $$y_{i-1}$$ 와 $$y_i$$ 만이 transition 을 통하여 상관성을 지닙니다. 단어열이 실제로 상관이 있고, 각 단어의 품사를 추정하는데 유용한 features 로 이용해야 함에도 불구하고 이를 이용할 수 없는 점이 HMM 이 tagger 로 이용되기에 부적절한 결정적인 이유입니다. 이 문제는 미등록 단어처럼 어떤 트릭을 써서 해결하기도 어렵습니다.

반대로 이후에 제안되는 Maximum Entropy Markov Model (MEMM) 이나 Conditional Random Field (CRF) 는 potential function 을 이용하여 단어 간의 관계를 직접적으로 이용함으로써 더욱 정확한 sequential labeling 이 되도록 만들어줍니다.

## TnT - Statistical Part-of-Speech Tagging

HMM based pos tagger 를 공부한다면, 역사적으로 중요한 taggers 도 살펴보면 좋을 것입니다. TnT 는 대표적인 초기 part of speech tagger 입니다. 무려 2000 년에 제안된 알고리즘이며, CRF 가 이용되기 전에 널리 이용되던 tagger 입니다. TnT 는 Trigram'n Tags 의 약자입니다. Trigram, second-order Markov model 을 이용합니다.

TnT 는 word sequence $$w_1, w_2, \dots, w_n$$ 에 대하여 다음의 확률을 최대화 하는 tag sequence $$t_1, t_2, \dots, t_n$$ 을 찾습니다.

$$argmax_{t_{1:n}} \prod_{i=1}^{n} P(t_i \vert t_{i-1}, t_{i-2}) P(w_i \vert t_i)$$

State 인 tag, $$t_i$$ 에서 단어 $$w_i$$ 가 발생하는 확률 모델은 동일합니다. 하지만 state 간의 transtion 의 조건이 이전 품사 1개 ($$t_{i-1}$$) 가 아닌 2개 ($$t_{i-1}, t_{i-2}$$) 입니다.

각 부분의 확률은 다음과 같이 정의됩니다.

- Unigram: $$P(t_i) = \frac{f(t_i)}{N}$$
- Bigram: $$P(t_i \vert t_{i-1}) = \frac{f(t_{i-1}, f_i)}{f(t_{i-1})}$$
- Trigram: $$P(t_i \vert t_{i-1}, t_{i-2}) = \frac{f(t_i, t_{i-1}, t_{i-2})}{f(t_{i-1}, t_{i-2})}$$
- Lexical: $$P(w_i \vert t_i) = \frac{f(w_i, t_i)}{f(t_i)}$$

Bigram 보다는 문맥을 더 잘 이해할 수도 있지만, 여전히 unguaranteed independence problem 이 발생합니다.

### Smoothing

TnT 에서부터도 smoothing 이 이용되었습니다. Trigram 을 이용하다보면 데이터의 sparseness 때문에 잘못된 값이 학습되기도 합니다. 이를 완화하기 위하여 smoothing 이 이용됩니다.

$$P(t_3 \vert t_1, t_2) \leftarrow \lambda_1 P(t_3) + \lambda_2 P(t_3 \vert t_2) + \lambda_3 P(t_3 \vert t_1, t_2)$$

where $$\lambda_1 + \lambda_2 + \lambda_3 = 1$$ 

그러나 back-off 는 TnT 에서 이용되지는 않았습니다.

### Handling unknown words

[미등록 단어 문제][oov]가 발생할 경우, suffix 를 이용하여 미등록 단어의 품사를 추정합니다. 영어의 Wall Streat Journal (WSJ) corpus 에서는 -able 로 끝나는 단어의 98 % 가 형용사였습니다. 학습데이터에 존재하지 않은 단어에 대해서는 suffix 를 이용하여 품사를 추정합니다.

영어에서는 띄어쓰기 기준으로 나뉘어진 단위 (어절)이 하나의 단어입니다. 그렇기 때문에 suffix 를 이용할 수가 있지만, 한국어의 경우에는 어절의 suffix 에 하나의 단어가 위치할 경우가 많습니다. '-가/조사', '-다던/어미'와 같은 단어, 혹은 형태소가 위치하기 때문에 이 규칙을 적용하기는 어렵습니다. 단, 어절의 앞부분에 위치한 substring 이 미등록 단어이고, 그 다음 -가 이 결합되어 있다면 이 substring 을 명사로 추정하는 것 정도로 적용할 수는 있습니다.

### Appending user dictionary

HMM based tagger 에 사용자 사전을 입력하는 것은 상대적으로 쉽습니다. 특정 단어 $$x$$ 가 품사 $$y$$ 가 될 수 있는 가능성은 emission probability 에만 저장되어 있습니다. 사실 $$P(x \vert y)$$ 는 확률 분포이기는 하지만, $$\sum P(x_i \vert y) > 1$$ 이어도 단어/품사열의 log probability 를 계산하는데 전혀 문제가 되지 않습니다. 우리는 여러 단어/품사열 간의 log probability 에 대한 상대비교만 할 것이기 때문입니다. 이긴 후보가 우리 후보입니다.

만약 새로운 단어 (word) 가 다른 어떤 단어들 보다도 품사 (tag) 에서 우위에 있는 명사임을 알고 그 단어는 학습 말뭉치에 등장하지 않았었다면 $$max P(word \vert tag)$$ 혹은 그보다 조금 더 큰 값으로 새로운 단어의 emission probability 를 저장합니다.

{% highlight python %}
class TrainedHMM:
    ...
    def add_user_dictionary(self, tag, words):
        append_score = max(self.pos2words[tag].values())
        for word in words:
            self.pos2words[tag][word] = append_score
{% endhighlight %}

이제 이 단어들은 매우 큰 emission probability 를 지니기 때문에 다른 단어/품사열보다 우선적으로 선택될 가능성이 높아졌습니다.

위 코드들은 [링크의 github][hmm_postagger] 에 올려두었습니다. 여기에는 decode 의 가능까지 구현된 코드가 올라갈 예정입니다. Decode 에 대한 구현 과정은 다음 포스트에서 다룹니다.

## Reference

- Brants, T. (2000, April). [TnT: a statistical part-of-speech tagger][tnt]. In Proceedings of the sixth conference on Applied natural language processing (pp. 224-231). Association for Computational Linguistics.

[ford_for_pos]: {{ site.baseurl }}{% link _posts/2018-08-21-ford_for_pos.md %}
[oov]: {{ site.baseurl }}{% link _posts/2018-04-01-pos_and_oov.md %}
[crf]: {{ site.baseurl }}{% link _posts/2018-04-24-crf.md %}
[tnt]: http://www.coli.uni-saarland.de/~thorsten/publications/Brants-ANLP00.pdf
[sejong_git]: https://github.com/lovit/sejong_corpus/
[hmm_postagger]: https://github.com/lovit/hmm_postagger/