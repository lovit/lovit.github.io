---
title: Reviews of sequential labeling algorithms (Sparse representation model 을 중심으로)
date: 2018-12-05 21:00:00
categories:
- nlp
- machine learning
tags:
- sequential labeling
---

머신 러닝의 분류기 (classifiers) 는 input vector $x$ 에 대하여 클래스를 분류합니다. 만약 입력값이 벡터가 아닌 $$x = [x_1, x_2 ,\ldots, x_n]$$ 시퀀스일 경우, 각 시퀀스에 적절한 클래스 시퀀스 $$y = [y_1, y_2, \ldots, y_n]$$ 들을 분류하는 문제를 sequential labeling 이라 합니다. 이를 위해 처음에는 Hidden Markov Model (HMM) 도 이용되었습니다만, 구조적인 단점이 존재하였습니다. 이후 Conditional Random Field (CRF) 와 같은 maximum entropy classifiers 들이 제안되었고, Word2Vec 이후 단어 임베딩 기술이 성숙하면서 Recurrent Neural Network (RNN) 계열도 이용되고 있습니다. 최근에는 Transformer 를 이용하는 BERT 까지도 sequential labeling 에 이용됩니다. 이번 포스트에서는 이 분야에서 중요한 알고리즘들을 살펴봅니다.

## Sequential labeling

일반적인 머신 러닝의 분류기는, 하나의 입력 벡터 $$x$$ 에 대하여 하나의 label 값 $$y$$ 를 return 합니다. 그런데 입력되는 $$x$$ 가 벡터가 아닌 sequence 일 경우가 있습니다. $$x$$ 를 길이가 $$n$$ 인 sequence, $$x = [x_1, x_2, \ldots, x_n]$$ 라 할 때, 같은 길이의 $$y = [y_1, y_2, \ldots, y_n]$$ 을 출력해야 하는 경우가 있습니다. 각 $$y_i$$ 에 대하여 출력 가능한 label 중에서 적절한 것을 선택하는 것이기 때문에 classification 에 해당하며, 데이터의 형식이 벡터가 아닌 sequence 이기 때문에 sequential data 에 대한 classification 이라는 의미로 sequential labeling 이라 부릅니다.

띄어쓰기 문제나 품사 판별이 대표적인 sequential labeling 입니다. 품사 판별은 주어진 단어열 $$x$$ 에 대하여 품사열 $$y$$ 를 출력합니다. 

- $$x$$ = [이것, 은, 예문, 이다]
- $$y$$ = [명사, 조사, 명사, 조사]

띄어쓰기는 길이가 $$n$$ 인 글자열에 대하여 [띈다, 안띈다] 중 하나로 이뤄진 Boolean sequence $$y$$ 를 출력합니다. 

- $$x$$ = 이것은예문이다
- $$y$$ = $$[0, 0, 1, 0, 1, 0, 1]$$

이 과정을 확률모형으로 표현하면 주어진 $$x$$ 에 대하여 $$P(y \vert x)$$ 가 가장 큰 $$y$$ 를 찾는 문제입니다. 이를 아래처럼 기술하기도 합니다. $$x_{1:n}$$ 은 길이가 $$n$$ 인 sequence 라는 의미입니다. 많은 sequential labeling 알고리즘들은 아래의 식을 어떻게 정의할 것이냐에 따라 다양한 방법으로 발전하였습니다.

$$argmax_y P(y_{1:n} \vert x_{1:n})$$

가장 간단한 방법은 각 $$x_i$$ 별로 별도의 분류기를 이용하는 것입니다. 이 분류기는 Softmax regression, Support Vector Machine 혹은 그 어떤 것이던지 이용할 수 있습니다. $$f$$(은) = 조사 와 같은 함수를 학습할 수 있지만, 이는 모든 '은'이라는 단어가 반드시 '조사'라는 가정이 있어야 합니다. 하지만 한 단어는 다양한 품사를 가질 수 있습니다. 띄어쓰기 교정에서도 $$f$$(은) = $$1$$ 이라는 모델이 학습되는 것인데, 모든 '은' 다음에 띄어쓰는 것이 아니기 때문에 이러한 접근 방법은 옳지 않습니다. 더 좋은 방법은 '은' 이라는 단어나 글자가 **등장한 맥락을 입력값으로 함께 이용**하는 것입니다.

### v.s. sequence segmentation

Sequence 를 다루는 문제 중 하나로, sequence segmentation 이 있습니다. 이는 길이가 $$n$$ 인 input sequence $$x_{1:n}$$ 에 대하여 길이가 $$m \le $n$$ 인 output sequence $$y_{1:m}$$ 을 출력하는 문제입니다. 대표적인 문제는 문장을 단어로 분리하는 토크나이저 입니다. 중국어권 연구에서는 주로 word segmentation 이라 부릅니다. 아래 예시처럼 $$7$$ 개의 글자열이 입력되면 $$4$$ 개의 단어열을 출력하는 것입니다. 띄어쓰기도 어절 단위에서의 sequence segmentation 이기도 합니다.

- $$x$$ = '이것은예문이다'
- $$y$$ = [이것, 은, 예문, 이다]

Sequence segmentation 은 sequential labeling 문제로 생각할 수도 있습니다. Output sequence 가 각 단어의 경계이면 됩니다. 아래의 예시처럼 각 단어의 시작 부분을 B 로, 그 외의 부분을 I 로 표현할 수도 있습니다.

- $$y$$ = [B, I, B, B, I, B, I]

혹은 품사태그까지 한 번에 부여가 가능합니다. 아래처럼 각 단어의 시작 위치 태그 (B, I) 뿐 아니라 각 단어의 품사를 함께 부여할 수도 있습니다. [카카오 형태소 분석기](https://github.com/kakao/khaiii) 역시 음절 단위의 품사 태깅을 수행한다고 설명한 적이 있습니다 ([블로그 참고](https://brunch.co.kr/@kakao-it/308))

- $$y$$ = [B-Noun, I-Noun, B-Josa, B-Noun, I-Noun, B-Adjective, I-Adjective]

단 segmentation 을 위하여 글자 단위의 sequence labeling 을 할 때에는 각 글자가 독립적이지 않게 태깅하는 것이 중요합니다. `예문`의 `예` 와 `문`은 한 단어로부터 등장하는 단어이기 때문입니다. 즉 `예`가 `B-Noun` 로 태깅되는 순간 `문` 역시 `I-Noun` 이 되어야 합니다.

Segmentation 으로의 sequence labeling 은 다른 포스트에서 다뤄보고, 이 포스트에서는 sequential labeling 에 이용된 전통적인 방법들의 발전 과정과 각 알고리즘의 차이점에 대하여 정리합니다.


## Hidden Markov Model (HMM)

HMM (Krogh, 1994)^[1] 을 이용한 품사 판별기에 대해서는 [이전의 포스트][hmmpost]에서 다룬 적이 있습니다. 이 포스트에서는 HMM 의 원리를 간단히 정리하고, 품사 판별과 같은 sequential labeling 관점에서의 HMM 의 문제점에 대하여 정리합니다. HMM 은 $$ P(y_{1:n} \vert x_{1:n})$$ 을 아래처럼 정의합니다. 여기서 $$x$$ 는 단어열, $$y$$ 는 품사열 입니다.

$$P(y_{1:n} \vert x_{1:n}) := P(x_{1:n}, y_{1:n}) = \prod_i P(x_i \vert y_i) \times P(y_i \vert y_{i-1})$$

HMM 은 Naive Bayes rules 을 이용하는데, 주어진 $$x$$ 에 대한 $$y$$ 의 확률이 아닌, 데이터에 $$(x, y)$$ 가 존재할 확률을 계산합니다. 그래서 HMM 을 generative model 이라 말합니다. 우리는 $$x$$ 를 주어주고, 가장 적절한 $$y$$ 를 판별해 달라고 말하지만, HMM 은 이를 간접적으로 계산합니다.

HMM 은 각 단어의 품사를 물어보는 질문에, 학습 데이터의 각 품사에서 해당 단어가 등장했던 확률값으로 단어의 품사를 추정합니다. $$P(이 \vert Josa) \ge P(이 \vert Noun)$$ 이면 `이` 라는 글자를 `Josa` 로 판단합니다. 물론 모든 단어를 독립적으로 판단하지는 않습니다. `Josa` 다음에는 `Josa`가 등장하기 어려우니, 이런 경우는 $$P(y_i \vert y_{i-1})$$ 에 의하여 배제될 가능성이 높습니다.

HMM 을 이용하는 대표적인 품사 판별기는 TnT 입니다. [NLTK 의 엔진](https://www.nltk.org/_modules/nltk/tag/tnt.html)으로도 공개되어 있으며, 품사 판별을 위한 확률 식은 아래와 같습니다. 단어 단위에서는 unigram 의 정보 $$P(w_i \vert t_i)$$ 만을 이용하지만, 품사 단위에서는 trigram 까지 이용하는 모델입니다.

$$P(t_{1:n} \vert w_{1:n}) = \prod_i P(w_i \vert t_i) \times P(t_i \vert t_{i-1}, t_{i-2})$$

TnT 는 영어 단어의 품사를 추정하기 위한 모델입니다. 이 모델은 모르는 단어 (미등록단어 문제) 의 품사를 추정하기 위하여 단어의 끝 부분의 2, 3 글자 (suffix) 정보를 이용했습니다. 단어의 끝 부분이 -ed 라면 동사나 형용사의 과거형일 가능성이 높을 것입니다. 이러한 정보를 규칙 기반으로 이용하였는데, 이는 각 단어의 경계가 띄어쓰기로 나뉘어지는 영어의 특징을 이용한 것입니다.

그러나 HMM 은 **품사 판별기로써의 약점**이 여러 가지가 있습니다. 이 문제들에 대하여 간단히 정리해봅니다. 그리고 이 문제들은 HMM 뿐 아니라 많은 sequential labeling 알고리즘들의 문제이기도 하며, 이후의 모델들은 이를 해결하기 위한 방법들이기도 합니다.

### Out of vocabulary

첫번째 문제는 한 번도 보지 못한 단어는 제대로 인식할 방법이 없다는 것입니다. [이전의 포스트][hmmpost]에서 다룬 것처럼 HMM 은 학습 데이터에 등장한 (단어, 품사) 정보를 확률 모델로 암기합니다. 그렇기 때문에 학습 데이터에 등장한 단어를 제대로 인식할 방법이 없습니다. 즉 모르는 단어 $$x_i$$ 에 대해서는 $$P(x_i \vert y_i) = 0$$ 입니다. 이는 단어를 있는 그대로 외웠기 때문입니다. 다른 모델들은 그 단어가 등장하는 문맥 정보를 학습하기 때문에 모르는 단어라도 비슷한 문맥이 입력되면 해당 단어의 품사를 어느 정도는 추정할 수 있습니다.

그러나 이는 사용자 사전에 단어를 추가함으로써 간단하게 해결할 수 있습니다. 사용자에 의하여 단어와 품사 쌍 $$(w, t)$$ 를 특정 확률로 추가하고, $$\sum_w P(w \vert t) = 1$$ 이 되도록 re-scaling 하면 됩니다.

### Unguaranteed Independency Problem

두번째 문제는 매우 치명적입니다. 우리는 `오늘, 의, A, 는, ... `이라는 문장에서 `A` 라는 단어의 품사를 추정하기 위하여 앞, 뒤의 단어들의 정보를 이용합니다. 문맥 정보는 주로 앞, 뒤에 등장하는 단어들입니다. 하지만 HMM 은 $$P(y_i \vert y_{i-1}$$ 에 대한 정보는 학습하여도 $$(x_{i-1}, x_i)$$ 의 정보는 학습하지 않습니다. 앞선 예시처럼 `이` 라는 단어는 명사, 조사, 형용사 등 다양한 품사를 가질 수 있기 때문에 $$x_i = 이$$ 의 품사 추정을 위해서는 앞, 뒤 단어를 살펴봐야 합니다. 하지만 HMM 은 그 앞에 등장한 단어의 품사 $$y_{i-1}$$ 정보만 이용할 뿐입니다. 

이러한 문제를 unguaranteed indeiendency problem 이라 합니다. 각 단어 $$x_i$$ 가 서로 독립이라는 잘못된 가정을 한다는 의미입니다. 주로 sequential modeling 에서는 한 시점 주변의 스냅샷 정보를 이용하는 경우가 많은데, HMM 은 이러한 능력이 없습니다.

### Number of words (Label bias)

세번째 문제도 치명적입니다. 앞의 예시처럼 실제로 학습데이터의 단어 `이` 의 확률은 $$P(이 \vert Josa) \ge P(이 \vert Noun)$$ 입니다. 이는 명사의 종류가 조사의 종류보다 압도적으로 많기 때문에 명사의 각 단어의 확률 $$P(w \vert Noun)$$ 은 대체로 매우 작은 값입니다. 반대로 $$P(w \vert Josa)$$ 는 매우 큰 값을 지닙니다. 아래는 세종 말뭉치 데이터의 일부에서의 각 품사 별 고유 단어의 개수 예시입니다.

| Tag | Number of unique words |
| --- | --- |
| Noun | 63968 |
| Verb | 3598 |
| Adverb | 3190 |
| Eomi | 1460 |
| Adjective | 849 |
| Exclamation | 464 |
| Josa | 158 |
| Determiner | 123 |

품사, 혹은 output value 에 따라 편향성 (bias) 이 생깁니다. 이때문에 확률적으로 특정 품사를 선호하는 현상이 발생합니다.

### Local normalization (Label bias)

네번째 문제 역시 치명적입니다. 이 역시 output values 의 빈도수 때문에 발생하는 문제입니다. 만약 $$t_1$$ 은 자주 이용되는 품사이기 때문에 그 다음에 등장하는 품사들은 그 종류가 30 가지 정도 된다고 가정합니다. 다른 품사 $$t_2$$ 는 특정한 문맥에서만, 상대적으로 적게 이용되는 품사이기 때문에 그 다음에 등장하는 품사들의 종류 역시 3 가지 정도 작다고 가정합니다. 그러면 위의 문제처럼 일반적으로 $$P(t \vert t_1) \le P(t \vert t_2)$$ 이게 됩니다. 하지만 $$t_2$$ 는 가정한 것처럼 거의 등장하지 않았습니다.

즉, 실제 데이터의 분포를 보면 $$(y_{i-1} = t_2, y_i = t)$$ 의 경우가 매우 작음에도 불구하고 확률값은 더 크게 계산됩니다. 이는 매 시점 $$i$$ 마다 $$P(y_{1:i} \vert x_{1:n}$$ 까지의 확률을 정의하고 가기 때문인데, 이를 local normalization 이라 합니다. Sequence 전체를 보기 전에 이미 sub-sequence 에 대한 확률을 정의한다는 의미입니다.

### Length bias

다섯번째 문제는 sequence segmentation 을 함께 푸는 경우에 발생하는 문제입니다. 아래처럼 $$x_{1:n}$$ 에 대하여 $$y_{1:n}$$ 이 출력되는 문제라면 output sequence 의 길이가 같기 때문에 $$P(y_0 \vert x)$$ 나 $$P(y_1 \vert x)$$ 의 스케일에 큰 차이가 없습니다.

- $$x$$ = '이것은예문이다'
- $$y_0$$ = [B-Noun, I-Noun, **B-Josa**, B-Noun, I-Noun, B-Adjective, I-Adjective]
- $$y_1$$ = [B-Noun, I-Noun, **I-Noun**, B-Noun, I-Noun, B-Adjective, I-Adjective]

하지만 아래처럼 $$y_0$$ 은 4 개의 단어열로, $$y_1$$ 은 3 개의 단어열로 문장을 분해한다면 $$y_1$$ 이 잘못된 문장임에도 불구하고 더 큰 확률을 가질 수 있습니다. HMM 은 $$2n$$ 개의 확률의 곱으로 $$P(y_{1:n} \vert x_{1:n})$$ 의 확률을 정의합니다. 그리고 각 확률은 1 이하의 값으로 정의됩니다. 확률적으로는 짧은 output sequence 에 더 큰 확률이 주어질 가능성이 높습니다. 그래서 HMM 은 길이가 긴 단어로 구성된 문장, 즉 최대한 적은 수의 단어들로 문장을 분해하려는 편향성이 생깁니다.

- $$y_0$$ = [이것/Noun, 은/Josa, 예문/Noun, 이다/Adjective]
- $$y_1$$ = [이것은/Noun, 예문/Noun, 이다/Adjective]

단, 이는 일단 문장이 제대로 된 단어열로 분해되었다는 가정을 할 때 입니다. 한국어는 표의 문자인 한자어를 일부 차용하는 언어이기 때문에 각 음절이 단어인 경우가 많습니다. 게다가 미등록단어 문제까지 발생합니다. 그렇기 때문에 HMM 을 이용하는 한국어 형태소 분석기에서 미등록단어 문제가 발생하면 학습데이터에 등장하는 단어가 포함된 가장 긴 단어를 우선적으로 선호할 가능성이 높습니다.

1, 2 번은 HMM 의 구조적 한계점이며, 3 - 5 번은 특정한 경우에 편향성이 생기는 문제입니다. 좋은 sequential labeling 은 주어진 $$x$$ 에 대하여 편향성 없이 $$y$$ 를 찾을 수 있어야 합니다. TnT 는 2000 년에 제안되었지만 HMM 기반 모델들은 주로 90 년대까지 이용되었습니다. 뒤이어 설명할 MEMM 같은 maximum entropy classifiers 들은 1, 2 번의 한계점을 극복하기 때문에 HMM 기반으로 작업할 이유가 사라진 것입니다.

## Maximum Entropy Markov Model (MEMM)

2000 년에 ICML 에 Maximum Entropy Markov Model 이 제안됩니다 (McCallum et al., 2000) ^[4]. 이는 maximum entropy classifiers 에 속하는 모델로, maximum entropy classifier 는 쉽게 생각하여 softmax regression 형식의 classifier 를 의미한다고 생각해도 좋습니다. 물론 MEMM 이 이런 종류의 첫번째 모델은 아니지만, 분기점 같은 역할을 하는 알고리즘입니다.

MEMM 과 CRF 에 대해서도 [이전의 포스트][memm_crf]에서 다뤘습니다. 그 중에서 중요을 집고 넘어갑니다.

### Potential function

그전에 MEMM 에 대하여 이야기하려면 단어와 같은 category sequence 를 벡터로 표현하는 방법부터 알아야 합니다. HMM 은 (단어, 품사) 의 확률만을 계산하였기 때문에 단어열을 벡터 형식으로 변환할 필요는 없었습니다. 하지만 softmax regression 을 이용하는 MEMM 은 단어열을 벡터로 표현해야 했습니다. 이를 위하여 potential function 이 이용됬습니다. 이는 categorical 뿐 아니라 numerical sequence 도 벡터로 표현할 수 있는 방법입니다.

예를 들어 $$x = [3.2, 2.1, -0.5]$$ 라는 길이가 3 인 sequence 에 대하여 아래의 필터 $$F_1$$ 를 적용할 수 있습니다.

- $$x = [3.2, 2.1, -0.5]$$ . 
- $$F_1 = 1$$ if $$x_i > 0$$ else $$0$$
- $$x_{vec} = [1, 1, 0]$$ .

필터를 여러 개 이용할 수도 있습니다. 각 시점 $$i$$ 에 대한 벡터의 크기는 필터의 개수와 같습니다.

- $$x = [3.2, 2.1, -0.5]$$ .
- $$F_1 = 1$$ if $$x_i > 0$$ else $$0$$
- $$F_2 = 1$$ if $$x_i > 3$$ else $$0$$
- $$x_{vec} = [(1, 1), (1, 0), (0, 0)]$$ .

이 필터가 potential function 입니다. Potential function 은 categorical variable 에 대해서도 적용이 가능합니다. 

- $$x = [이것, 은, 예문, 이다]$$ .
- $$F_1 = 1$$ if $$x_{i-1} =$$ '이것' & $$x_i =$$ '은' else $$0$$
- $$F_2 = 1$$ if $$x_{i-1} =$$ '이것' & $$x_i =$$ '예문' else $$0$$
- $$F_3 = 1$$ if $$x_{i-1} =$$ '은' & $$x_i =$$ '예문' else $$0$$
- $$x_{vec} = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, 0)]$$ .

앞서 달아둔 label $$y_{i-1}$$ 를 함께 이용하기 위한 potential function 도 만들 수 있습니다.

- $$x = [이것, 은, 예문, 이다]$$ .
- $$F_1 = 1$$ if $$x_{i-1} =$$ '이것' & $$x_i =$$ '은' else $$0$$
- $$F_2 = 1$$ if $$x_{i-1} =$$ '이것' & $$x_i =$$ '예문' else $$0$$
- $$F_3 = 1$$ if $$x_{i-1} =$$ '은' & $$x_i =$$ '예문' else $$0$$
- $$F_4 = 1$$ if $$x_{i-1} =$$ '이것' & $$x_i =$$ '은' & $$y_{i-1} =$$ '명사' else $$0$$
- $$F_5 = 1$$ if $$x_{i-1} =$$ '이것' & $$x_i =$$ '예문' & $$y_{i-1} =$$ '명사' else $$0$$
- $$F_6 = 1$$ if $$x_{i-1} =$$ '은' & $$x_i =$$ '예문' & $$y_{i-1} =$$ '조사' else $$0$$

Potential function 은 데이터를 암기하여 Boolean vector 로 표현하는 방법입니다. 단어열의 예시처럼 필터의 종류가 다양할 수 있기 때문에 주로 templates 으로 표현합니다. 예를 들어 $$(x_{-2}, x_{-1}, x_{0})$$ 은 앞의 두 단어와 현재의 단어를 모두 합하여 하나의 $$F_i$$ 로 이용한다는 의미입니다. 그렇기 때문에 품사 판별과 같은 NLP tasks 에서는 potential function 에 의하여 매우 큰 차원의 벡터 공간이 만들어집니다. $$(x_{-2}, x_{-1}, x_{0})$$ 은 trigram space 입니다.

이 정보에 의하여 MEMM 은 단어열의 문맥을 features 로 이용할 수 있게 되었습니다. 이는 미등록단어에 대해서도 대처할 수 있도록 도와주는데, $$x_{0}$$ 을 모르더라도 $$(x_{-2}, x_{-1}, x_{1})$$ 을 이용하면 $$x_0$$ 을 짐작할 수 있습니다. 예를 들어 `(오늘, 의, A, 은)` 에서 A 는 명사일 것이라는 힌트를 앞, 뒤에 등장하는 단어만으로 짐작할 수 있게 된 것입니다. 단, 이때도 문장이 단어열로 제대로 분해되었다는 가정이 필요합니다.

하지만 이 방법은 bag-of-words model 처럼 각 차원이 어떤 의미인지 해석할 수 있다는 장점이 있습니다. 단점은 (오늘, 의, 메뉴) 나 (오늘, 의, 식단) 처럼 두 features 가 비슷한 의미를 지니고 있다는 정보를 학습할 수가 없습니다.

### MEMM as Logistic regression

MEMM 은 potential function 을 이용하여 입력된 단어열 $$x_{1:n}$$ 을 Boolean vector sequence $$h_{1:n}$$ 으로 변환합니다. 그 뒤, 각 $$h_i$$ 에 대하여 $$y_i$$ 의 확률을 계산합니다.

$$P(y_{1:n} \vert x_{1:n}} = \prod_i^n P(y_i \vert h_i)$$

$$P(y_i \vert h_i)$$ 을 아래처럼 표현할 수 있습니다. $$h_i$$ 가 클래스 $$k$$ (품사 $$k$$) 가 될 확률을 계산하고, 각 $$i$$ 에 대하여 독립적인 $$n$$ 번의 softmax regression 을 수행한다는 의미입니다. 이 때 $$\lambda$$ 는 매 시점 $$i$$ 마다 공통으로 이용됩니다.

$$P(y_{1:n} \vert x_{1:n}} = \prod_i^n \frac{exp(\lambda_{k}^T h_i)}{\sum_l exp(\lambda_{l}^T h_i)}$$

이를 아래처럼 더 자세하게 기술할 수도 있습니다. $$f_j(x, i, y_i, y_{i-1})$$ 은 현재 시점 $$i$$ 의 앞의 품사가 $$y_{i-1}$$ 이고 지금 시점의 품사가 $$y_i$$ 라면 이라는 potential function 의 Boolean 값 입니다. 그리고 $$\lambda_j$$ 는 그럴 경우의 점수, 즉 logistic regression 의 coefficient 입니다.

$$P(y \vert x) = \prod_{i=1}^{n} \frac{exp(\sum_{j=1}^{m} \lambda_j f_j (x, i, y_i, y_{i-1}))}{ \sum_{y^{`}} exp(\sum_{j^{`}=1}^{m} \lambda_j f_j (x, i, y_i^{`}, y_{i-1}^{`})) }$$

마지막 수식은 복잡하긴 하지만, 결국 potential function 을 이용하여 $$x$$ 를 sparse vector $$h$$ 로 만든 뒤, softmax regression 을 수행한다는 의미입니다. 그렇기 때문에 **ME**MM 이라는 이름을 가졌습니다.

MEMM 은 discriminative model 인 softmax regression 형식입니다. 확률 모델이 아니기 때문에 $$f_j$$ 의 빈도수의 영향을 덜받습니다. $$f_j$$ 가 학습데이터에 몇 번 등장하지 않은 변수라 하더라도 그 정보가 명확하다면 매우 큰 값의 $$\lambda_j$$ 가 학습될 것입니다. 하지만 이는 그렇게 잘 학습될 수 있다는 가정일 뿐, 현실은 softmax regression 을 근사학습하는 최적화 방법들에 의하여 약간의 frequency bias 가 있습니다.

### MEMM as Markov Model

또한 ME**MM** 은 Markov Model 의 성질을 가지고 있습니다. 이는 $$(y_{i-1}, y_i)$$ 의 정보가 학습된다는 의미입니다. 위의 마지막 식의 $$f_j$$ 는 아래처럼 두 종류의 성분으로 구분할 수 있습니다. $$f_p$$ 는 $$(x_i, y_i)$$ 의 성분에 대한 features 이며, $$g_q$$ 는 $$(y_{i-1}, y_i)$$ 성분에 대한 features 입니다. 두 종류의 features 는 구분될 수 있으며, MEMM 은 HMM 이 학습하는 정보를 모두 (이론적으로는) 학습할 수 있다는 의미입니다.

$$exp(\lambda^T h_j) = exp(\sum_p \mu_p f_p(x, i, y_i) + \sum_q \theta_q g_q(x, i, y_i, y_{i-1}))$$

이처럼 현재 값 $$y_i$$ 가 이전 값 $$y_{i-1}$$ 에만 영향을 받는 모델을 Markov model 이라 합니다. 그렇기 때문에 ME**MM** 이라는 이름을 가지게 되었습니다.

그리고 $$P(y_{i-1}, y_i)$$ 은 HMM 에서 transition probability 라 부릅니다. 이후 transion based model 을 설명할텐데, 이 모델들은 output sequence 의 bigram (혹은 그 이상)의 정보를 이용한다는 의미입니다.

## Conditional Random Field (CRF)

### Local normalization vs global normalization

SyntaxNet

MeCab

## Structured Support Vector Machine (StructuredSVM)

SVM

$$\min_w \frac{1}{2} \rVert w \rVert^2 + \frac{C}{n} \sum_{(x, y) \in S} max \left( 0, 1 - y \left<w, x \right> \right)$$

Structured SVM

$$\min_w \frac{1}{2} \rVert w \rVert^2 + \frac{C}{n} \sum_{i}^{n} \max_{y \in \mathcal{Y}} \left(0, \Delta(y_i, y) - \left( \left<w, \Phi(x_i, y_i) \right> - \left<w, \Phi(x_i, y) \right> \right) \right) $$

With slack variable,

$$\min_{w, \xi} \frac{1}{2} \rVert w \rVert^2 + \frac{C}{n} \sum_{i}^{n} \xi_i $$

$$s.t. \left <w, \Phi(x_i, y_i) \right> - \left<w, \Phi(x_i, y) \right> + \xi_i \ge \Delta(y_i, y)$$

## Average perceptron

Inference

$$f(x) = \arg \max_{y \in \mathcal{Y}} \left< w, \Phi(x, y) \right>$$

## Pegasos: Primal Estimation sub-GrAdient SOlver for SVM


## Transition based sequence labeling


## Recurrent Neural Network

### GRU and LSTM

문맥 정보를 hidden vectors 의 정보로

### LSTM-CRF

Label sequence 에 bigram 을. 이는 transition based labeler 형식

### BERT

문맥을 반영한 semantic word vector sequences 를

## Neural transition based sequence labeling

## Sequence labeling and segmentation








## References
- [1] Krogh, A. (1994). Hidden markov models for labeled sequences. In Proceedings of the 12th IAPR International Conference on Pattern Recognition, Vol. 3-Conference C: Signal Processing (Cat. No. 94CH3440-5), volume 2, pages 140–144. IEEE
- [2] 카카오 형태소 분석기 [github](https://github.com/kakao/khaiii), [blog](https://brunch.co.kr/@kakao-it/308)
- [3] Brants, T. (2000, April). TnT: a statistical part-of-speech tagger. In Proceedings of the sixth conference on Applied natural language processing (pp. 224-231). Association for Computational Linguistics.
- [4] McCallum, A., Freitag, D., and Pereira, F. C. (2000). Maximum entropy markov models for information extraction and segmentation. In Icml, volume 17, pages 591–598.

[hmmpost]: {{ site.baseurl }}{% link _posts/2018-09-11-hmm_based_tagger.md %}
[memm_crf]: {{ site.baseurl }}{% link _posts/2018-04-24-crf.md %}