---
title: 한국어 용언의 원형 복원 (lemmatization)
date: 2018-06-07 22:00:00
categories:
- nlp
tags:
- lemmatization
---

한국어의 단어는 9 품사로 이뤄져 있습니다. 그 중 용언에 해당하는 형용사와 동사는 활용 (conjugation) 이 됩니다. 용언은 어근 (root) 과 어미 (ending) 로 구성되어 있으며, 용언의 원형은 어근의 원형에 종결어미 '-다'가 결합된 형태입니다. 어미 부분이 다른 어미로 교체되기만 한 활용을 규칙활용이라 합니다. 반면 활용 도중 어근의 뒷부분 혹은 어미의 앞부분의 형태가 변하는 활용을 불규칙 활용이라 합니다. 불규칙 활용은 용언의 형태소 분석을 어렵게 하는 원인 중 하나입니다. 이번 포스트에서는 불규칙 활용의 경우를 유형화하여 어근과 어미의 원형을 복원하는 lemmatizer 를 구현합니다.

## Conjugation, Lemmatization, Stemming

용언의 활용과 관련된 단어들의 정의입니다. 활용 (conjutagion) 은 한 동사나 형용사인 용언이 더 정밀한 의미를 표현하기 위하여 다양한 형태로 변하는 현상입니다. '가다/동사'는 '가고, 가니까, 갔었다' 처럼 변할 수 있다. 영어에서도 'act' 는 'acting, acts, acted' 처럼 변합니다. 

활용은 규칙 활용과 불규칙 활용으로 나뉩니다. **규칙 활용**은 규칙에 따라 용언이 변하는 경우로, 영어에서는 과거형을 만들기 위해 '-ed' 라는 suffix 를 붙입니다. 한국어의 용언은 어근 (root) 과 어미 (ending) 라는 형태소로 구성되는되, 어근의 형태는 변하지 않고 어미만 다른 어미로 교체되는 경우입니다. 

	가다/동사 = 가/어근 + 다/어미
	가니까/동사 = 가/어근 + 니까/어미
	가라고/동사 = 가/어근 + 라고/어미

이 경우에는 용언으로 구성된 어절을 L + R 구조로 분해하면 손쉽게 용언의 원형인 '가다'를 복원할 수 있습니다. '가다, 가니까, 가라고'는 모두 '가- + {-다, -니까, -라고}'의 형태로 string split 을 합니다.

**불규칙 활용**은 어근이나 어미의 형태가 변하는 활용입니다. 이때에는 단순한 string split 만으로는 용언의 원형을 복원하기가 어렵습니다. 불규칙 활용도 몇 가지의 문법 규칙이 있습니다. 이 규칙들을 이용하여 lemmatizer 를 만들 수 있습니다.

	갔어/동사 = 가/어근 + ㅆ어/어미
	간거야/동사 = 가/어근 + ㄴ거야/어미
	꺼줘/동사 = 끄/어근 + 어줘/어미

한 단어는 원형 (canonical form) 과 활용되는 형 (surfacial form) 이 있습니다. 사전에 등제된 단어의 형태가 원형입니다. 한국어의 용언은 '어근 + -다/어미' 형태로 원형을 기술합니다. 영어에서는 명사도 단/복수에 따라 surfacial form 이 달라집니다. 'car'의 복수형은 'cars' 입니다.

Stemming 과 lemmatization 은 모두 단어의 canonical form 을 **인식**하기 위한 방법입니다. 둘의 차이는 **stemming** 은 규칙들로 이뤄진 string processing 입니다. '단어의 끝부분 -ed 를 제거한다'는 규칙을 적용하면 -ed 로 끝나는 단어의 원형을 찾을 수 있습니다.

	studies = studi + es
	studying = study + ing

**Lemmatization** 은 단어의 원형 (lemma)으로 복원을 합니다. 

	studies = studi + (-i / +y) + es
	studying = study + ing

Lemmatization 을 위해서는 단어의 품사 추정이 함께 이뤄져야 합니다. 또한 데이터 분석에서 '꺼줘'와 '끈'을 모두 '끄다/동사'로 표현하기 위해서는 lemmatization 이 이뤄져야 합니다.


## 한국어 lemmatization

한국어 용언의 불규칙 활용은 [나무위키][korean_lemmatize]에 예제와 함께 잘 정리가 되어있습니다. 이를 Python 으로 구현합니다.

우리가 구현할 lemmatizer 는 한국어 어절의 L-R 구조를 이용합니다. L-R 구조에 대해서는 [이전 포스트][lr_structure]를 참고하세요. 예를 들어 우리가 다음과 같은 동사 원형 사전을 가지고 있을 때, 다음의 어절을 (L, R) 로 분해한 뒤, L 이 우리가 알고 있는 동사의 어근인지 확인합니다.

	verb_dict = {'깨닫다'}
	eojeol = '깨달아'

	check(l='깨달아', r='')
	check(l='깨달', r='아') # -> '깨닫' + '아'
	check(l='깨', r='달아')

일단 lemmatizer 가 용언의 원형 사전을 이용하기 때문에 이를 class 형태로 만들면 좋습니다. 단어 word 가 입력되었을 때 이를 (L, R) 로 나눠 가능한 원형의 후보들을 출력하는 함수도 만들어 둡니다.

{% highlight python %}
class Lemmatizer:
    def __init__(self, roots):
        self._roots = roots

    def is_root(self, w):
        return w in self._roots

    def lemmatize(self, word):
        return None

    def candidates(self, word):
        candidates = set()
        for i in range(1, len(word)+1):
            l, r = word[:i], word[i:]
            candidates.update(self._candidates(l, r))
        return candidates

    def _candidates(self, l, r):
        candidates = set()
        # TODO
        return candidates
{% endhighlight %}

### 규칙 활용을 따르는 용언의 원형 복원

규칙 활용의 경우에는 L 자체가 어근의 원형이기 때문에 확인이 쉽습니다.

{% highlight python %}
def _candidates(self, l, r):
    candidates = set()

    if self.is_root(l):
        candidates.add((l, r))
{% endhighlight %}

### 불규칙 활용을 따르는 용언의 원형 복원을 위한 준비

'불규칙 활용'이라는 단어는 의미가 잘 전달되지 않는 말 같습니다. 사실 불규칙 활용은 어근과 어미의 형태가 변하는 '문법 규칙' 입니다. 에를 들어 'ㄷ불규칙 활용'은 어근의 ㄷ 받침과 어미의 모음이 만나 ㄷ 이 ㄹ 로 변하는 규칙이기 때문입니다. 규칙을 따르지 않는 불규칙은 '예외 경우'라 합니다.

여하튼 '불규칙 활용'도 문법 규칙이 있습니다. 이를 이용하여 용언의 어근과 어미를 인식하는 lemmatizer 를 만듭니다.

용언의 불규칙 활용은 어근과 어절이 만나는 부분에서 발생합니다. 주어진 어절을 (L, R) 로 나눈 뒤, L 의 끝부분과 R 의 첫부분이 불규칙 활용이 발생하는지 확인합니다. 이를 위하여 주어진 str 인 l 의 마지막 글자와 r 의 첫글자의 초/중/종성을 분해한 l_last, r_first 를 만듭니다. 그리고 l_last 의 종성 (받침)을 없엔 l_last_ 와 r 의 종성을 없엔 r_first_ 도 만듭니다.

{% highlight python %}
l_last = decompose(l[-1])
l_last_ = compose(l_last[0], l_last[1], ' ')
r_first = decompose(r[0]) if r else ('', '', '')
r_first_ = compose(r_first[0], r_first[1], ' ') if r else ' '
{% endhighlight %}

### ㄷ 불규칙 활용

어근의 마지막 글자 종성이 'ㄷ' 이고 어미의 첫글자가 'ㅇ' 이면 (모음으로 시작하면) 'ㄷ' 이 'ㄹ' 로 바뀝니다.

	깨닫 + 아 -> 깨달아
	묻 + 었다 -> 물었다

{% highlight python %}
if l_last[2] == 'ㄹ' and r_first[0] == 'ㅇ':
    l_root = l[:-1] + compose(l_last[0], l_last[1], 'ㄷ')
    if self.is_root(l_root):
        candidates.add((l_root, r))
{% endhighlight %}

### 르 불규칙 활용

어근의 마지막 글자가 '르' 이고 어미의 첫글자가 '-아/-어'이면 어근의 마지막 글자는 'ㄹ' 로 변화하여 앞글자와 합쳐지고, 어미의 첫글자는 '-라/-러'로 바뀝니다.

	구르 + 어 -> 굴러
	들르 + 었다 -> 들렀다

l 의 마지막 글자의 종성이 'ㄹ' 이고 r 의 첫글자의 초/중성이 '-라/-러'인지 확인합니다.

{% highlight python %}
if (l_last[2] == 'ㄹ') and (r_first_ == '러' or (r_first_ == '라')):
    l_root = l[:-1] + compose(l_last[0], l_last[1], ' ') + '르'
    r_canon = compose('ㅇ', r_first[1], r_first[2]) + r[1:]
    if self.is_root(l_root):
        candidates.add((l_root, r_canon))
{% endhighlight %}

### ㅂ 블규칙 (1)

어근의 마지막 글자 종성이 'ㅂ' 이고 어미의 첫글자가 모음으로 시작하면 'ㅂ' 이 'ㅜ/ㅗ' 로 바뀝니다.

	덥 + 어 -> 더워
	우습 + 어 -> 우스워
	곱 + 아 -> 고와

l 의 마지막 글자의 종성이 없고 r 의 첫글자의 초/중성이 '워/와' 이면 l 의 종성에 'ㅂ' 을 추가합니다.

{% highlight python %}
if (l_last[2] == ' ') and (r_first_ == '워' or r_first_ == '와'):
    l_root = l[:-1] + compose(l_last[0], l_last[1], 'ㅂ')
    r_canon = compose('ㅇ', 'ㅏ' if r_first_ == '와' else 'ㅓ', r_first[2]) + r[1:]
    if self.is_root(l_root):
        candidates.add((l_root, r_canon))
{% endhighlight %}

### 어미의 첫글자가 종성일 경우 (-ㄴ, -ㄹ, -ㅂ, -ㅆ)

한국어 문법에서는 규칙 활용으로 분류되지만, lemmatizer 의 관점에서는 불규칙에 해당하는 현상이 하나 더 있습니다. 어미 중에는 'ㅂ니다' 처럼 'ㅂ'으로 시작하는 어미들이 있습니다. 이들은 종성이 없는 어근의 받침에 'ㅂ'을 추가하는 형태로 활용됩니다. 

	이 + ㅂ니다 -> 입니다
	하 + ㅂ니다 -> 합니다

이에 해당하는 글자들은 -ㄴ, -ㄹ, -ㅆ 이 있습니다.

	하 + ㄴ답니다 -> 한답니다
	하 + ㄹ껄 -> 할껄
	이 + ㅆ어요 -> 있어요

l 의 마지막 글자의 종성이 'ㄴ, ㄹ, ㅂ, ㅆ' 이면 이를 제거합니다.

{% highlight python %}
if l_last[2] == 'ㄴ' or l_last[2] == 'ㄹ' or l_last[2] == 'ㅂ' or l_last[2] == 'ㅆ':
    l_root = l[:-1] + compose(l_last[0], l_last[1], ' ')
    r_canon = l_last[2] + r
    if self.is_root(l_root):
        candidates.add((l_root, r_canon))
{% endhighlight %}

### ㅅ 불규칙 활용

어근의 종성이 'ㅅ' 이고 어미가 모음으로 시작하면 'ㅅ' 이 탈락합니다. 단, '벗다' 는 예외입니다.

	낫 + 아 -> 나아
	긋 + 어 -> 그어
	벗 + 어 -> 벗어

l 의 마지막 글자의 종성이 없고 r 의 첫글자의 초성이 'ㅇ' 이면 l 의 마지막 글자의 종성에 'ㅅ' 을 추가합니다.

{% highlight python %}
if (l_last[2] == ' ' and l[-1] != '벗') and (r_first[0] == 'ㅇ'):
    l_root = l[:-1] + compose(l_last[0], l_last[1], 'ㅅ')
    if self.is_root(l_root):
        candidates.add((l_root, r))
{% endhighlight %}

### 우 불규칙

어근의 중/종성이 '우'이고 어미의 첫글자가 '어'일 때 'ㅜ' 가 탈락하는 활용입니다. '푸다'가 유일하며, 그 외에는 'ㅜ + ㅓ = ㅝ'로 규칙 활용이 됩니다. 하지만 lemmatizer 관점에서는 후자 역시 처리가 필요합니다. '푸다'와 그 외를 나눠서 구현합니다.

	푸 + 어갔어 -> 퍼갔어

'푸다'가 합성된 다른 어근이 존재할 수 있기 때문에 l 의 마지막 글자의 초/중성이 '퍼'인지 확인합니다. l 의 마지막 글자를 '푸'로 변환하고 r 의 맨 앞에 l 의 마지막 글자의 중/종성을 추가합니다.

{% highlight python %}
if l_last_ == '퍼':
    l_root = l[:-1] + '푸'
    r_canon = compose('ㅇ', l_last[1], l_last[2]) + r
    if self.is_root(l_root):
        candidates.add((l_root, r_canon))
{% endhighlight %}

'ㅜ + ㅓ = ㅝ'인 활용의 예시는 다음과 같습니다. 

	주 + 었습니다 -> 줬습니다
	누 + 었어 -> 눴어

l 의 마지막 글자의 중성이 'ㅝ' 이면 이 중성과 종성을 'ㅜ'와 ' '로 바꿉니다. r 의 첫글자에 'ㅓ'와 l 의 마지막 글자의 종성을 더합니다. 

{% highlight python %}
if l_last[1] == 'ㅝ':
    l_root = l[:-1] + compose(l_last[0], 'ㅜ', ' ')
    r_canon = compose('ㅇ', 'ㅓ', l_last[2]) + r
    if self.is_root(l_root):
        candidates.add((l_root, r_canon))
{% endhighlight %}

### 오 불규칙 활용 (가제, 본래는 규칙활용)

한국어 문법의 규칙은 아니지만, 우 불규칙 활용과 비슷한 현상이 있습니다. 어근의 중성/종성이 '오' 이고 어미의 첫글자가 '아'이면 'ㅗ + ㅏ = ㅘ'에 의하여 어근의 마지막 글자의 중성이 'ㅘ'로 변합니다.

l 의 마지막 글자의 중성이 'ㅘ' 이면 이 중성과 종성을 'ㅗ'와 ' '로 바꿉니다. r 의 첫글자에 'ㅏ'와 l 의 마지막 글자의 종성을 더합니다. 

{% highlight python %}
if l_last[1] == 'ㅘ':
    l_root = l[:-1] + compose(l_last[0], 'ㅇ', ' ')
    r_canon = compose('ㅇ', 'ㅏ', l_last[2]) + r
    if self.is_root(l_root):
        candidates.add((l_root, r_canon))
{% endhighlight %}

### ㅡ 탈락 불규칙 활용

어근의 중성이 'ㅡ' 이고 받침이 없고 어미가 '-아/-어'로 시작하면 'ㅡ'가 탈락합니다.

	끄 + 었다 -> 껐다
	트 + 었어 -> 텄어

l 의 마지막 글자의 중성이 'ㅓ/ㅏ'이면 중성과 종성을 'ㅡ'와 ' '로 바꿉니다. r 의 앞에 l 의 마지막 글자의 중성과 종성을 더합니다.

{% highlight python %}
if (l_last[1] == 'ㅓ' or l_last[1] == 'ㅏ'):
    l_root = l[:-1] + compose(l_last[0], 'ㅡ', ' ')
    r_canon = compose('ㅇ', l_last[1], l_last[2]) + r
    if self.is_root(l_root):
        candidates.add((l_root, r_canon))
{% endhighlight %}

### 거라, 너라 불규칙 활용

명령형 어미 '-아라/-어라'가 '-거라/-너라'로 바뀌는 활용입니다. '-거라/-너라'를 어미로 취급하면 규칙 활용으로 생각할 수 있습니다.

	가 + 아라 -> 가거라
	오 + 어라 -> 오너라 

### 러 불규칙 활용

어근의 마지막 글자가 '르'이고 어미의 첫글자가 '-어' 일 때 '-어'가 '-러'로 바뀝니다. 이때도 '러'를 포함하는 형태소를 어미로 생각하면 규칙 활용에 해당합니다. 

	이르 + 어 -> 이르러
	푸르 + 어 -> 푸르러

### 여 불규칙 활용

'-하다'로 끝나는 용언에서 어미의 첫글자 '-아'가 '-여'로 바뀌는 활용입니다. 이 역시 독립적인 어미로 생각하면 규칙 활용에 해당합니다.

	아니하 + 았다 -> 아니하였다
	영원하 + 아 -> 영원하여

### ㅎ 불규칙 활용

어근의 마지막 글자의 종성이 'ㅎ'일 경우 'ㅎ'이 탈락하거나 축약되는 활용입니다. 어근의 종성이 'ㅎ'인 형용사 중에서 '좋다'를 제외한 모든 형용사에서 발생합니다.

'ㅎ'이 탈락하는 경우입니다.

	파랗 + 면 -> 파라면
	동그랗 + ㄴ -> 동그란

l 의 마지막 글자의 종성이 없거나 'ㄴ, ㄹ, ㅂ, ㅆ' 이면 이를 'ㅎ'으로 변환합니다.

{% highlight python %}
if (l_last[2] == ' ' or l_last[2] == 'ㄴ' or l_last[2] == 'ㄹ' or l_last[2] == 'ㅂ' or l_last[2] == 'ㅆ'):
    l_root = l[:-1] + compose(l_last[0], l_last[1], 'ㅎ')
    r_canon = r if l_last[2] == ' ' else l_last[2] + r
    if self.is_root(l_root):
        candidates.add((l_root, r_canon))
{% endhighlight %}

어근의 마지막 글자의 종성 'ㅎ'와 어미의 첫글자 'ㅏ/ㅓ'가 합쳐져 'ㅐ/ㅔ'로 변합니다. 

	파랗 + 았다 -> 파랬다
	그렇 + 아 -> 그래
	시퍼렇 + 었다 -> 시퍼렜다

{% highlight python %}
if (l_last[1] == 'ㅐ') or (l_last[1] == 'ㅔ'):
    # exception : 그렇 + 아 -> 그래
    if len(l) >= 2 and l[-2] == '그' and l_last[0] == 'ㄹ':
        l_root = l[:-1] + '렇'
    else:
        l_root = l[:-1] + compose(l_last[0], 'ㅓ' if l_last[1] == 'ㅔ' else 'ㅏ', 'ㅎ')
    r_canon = compose('ㅇ', 'ㅓ' if l_last[1] == 'ㅔ' else 'ㅏ', l_last[2]) + r
    if self.is_root(l_root):
        candidates.add((l_root, r_canon))
{% endhighlight %}


## 구현된 lemmatizer

한 어절이 주어졌을 때 이로부터 가능한 용언의 어근과 어미의 원형 후보를 생성하는 lemmatizer 를 정리하면 아래와 같습니다.

{% highlight python %}
import soynlp
from soynlp.hangle import compose, decompose

class Lemmatizer:
    def __init__(self, roots, predefined=None):
        self._roots = roots
        self._predefined = {}
        if predefined:
            self._predefined.update(predefined)

    def is_root(self, w): return w in self._roots

    def lemmatize(self, word):
        raise NotImplemented

    def candidates(self, word):
        candidates = set()
        for i in range(1, len(word) + 1):
            l = word[:i]
            r = word[i:]
            candidates.update(self._candidates(l, r))
        return candidates

    def _candidates(self, l, r):
        candidates = set()
        if self.is_root(l):
            candidates.add((l, r))

        l_last = decompose(l[-1])
        l_last_ = compose(l_last[0], l_last[1], ' ')
        r_first = decompose(r[0]) if r else ('', '', '')
        r_first_ = compose(r_first[0], r_first[1], ' ') if r else ' '
        
        # ㄷ 불규칙 활용: 깨닫 + 아 -> 깨달아
        if l_last[2] == 'ㄹ' and r_first[0] == 'ㅇ':
            l_root = l[:-1] + compose(l_last[0], l_last[1], 'ㄷ')
            if self.is_root(l_root):
                candidates.add((l_root, r))

        # 르 불규칙 활용: 굴 + 러 -> 구르다
        if (l_last[2] == 'ㄹ') and (r_first_ == '러' or (r_first_ == '라')):
            l_root = l[:-1] + compose(l_last[0], l_last[1], ' ') + '르'
            r_canon = compose('ㅇ', r_first[1], r_first[2]) + r[1:]
            if self.is_root(l_root):
                candidates.add((l_root, r_canon))

        # ㅂ 불규칙 활용: 더러 + 워서 -> 더럽다
        if (l_last[2] == ' ') and (r_first_ == '워' or r_first_ == '와'):
            l_root = l[:-1] + compose(l_last[0], l_last[1], 'ㅂ')
            r_canon = compose('ㅇ', 'ㅏ' if r_first_ == '와' else 'ㅓ', r_first[2]) + r[1:]
            if self.is_root(l_root):
                candidates.add((l_root, r_canon))

#         # 어미의 첫글자가 종성일 경우 (-ㄴ, -ㄹ, -ㅂ, -ㅅ)
#         # 입 + 니다 -> 입니다
        if l_last[2] == 'ㄴ' or l_last[2] == 'ㄹ' or l_last[2] == 'ㅂ' or l_last[2] == 'ㅆ':
            l_root = l[:-1] + compose(l_last[0], l_last[1], ' ')
            r_canon = l_last[2] + r
            if self.is_root(l_root):
                candidates.add((l_root, r_canon))

#         # ㅅ 불규칙 활용: 부 + 었다 -> 붓다
#         # exception : 벗 + 어 -> 벗어
        if (l_last[2] == ' ' and l[-1] != '벗') and (r_first[0] == 'ㅇ'):
            l_root = l[:-1] + compose(l_last[0], l_last[1], 'ㅅ')
            if self.is_root(l_root):
                candidates.add((l_root, r))

        # 우 불규칙 활용: 똥퍼 + '' -> 똥푸다
        if l_last_ == '퍼':
            l_root = l[:-1] + '푸'
            r_canon = compose('ㅇ', l_last[1], l_last[2]) + r
            if self.is_root(l_root):
                candidates.add((l_root, r_canon))

        # 우 불규칙 활용: 줬 + 어 -> 주다
        if l_last[1] == 'ㅝ':
            l_root = l[:-1] + compose(l_last[0], 'ㅜ', ' ')
            r_canon = compose('ㅇ', 'ㅓ', l_last[2]) + r
            if self.is_root(l_root):
                candidates.add((l_root, r_canon))

        # 오 불규칙 활용: 왔 + 어 -> 오다
        if l_last[1] == 'ㅘ':
            l_root = l[:-1] + compose(l_last[0], 'ㅗ', ' ')
            r_canon = compose('ㅇ', 'ㅏ', l_last[2]) + r
            if self.is_root(l_root):
                candidates.add((l_root, r_canon))

        # ㅡ 탈락 불규칙 활용: 꺼 + '' -> 끄다 / 텄 + 어 -> 트다
        if (l_last[1] == 'ㅓ' or l_last[1] == 'ㅏ'):
            l_root = l[:-1] + compose(l_last[0], 'ㅡ', ' ')
            r_canon = compose('ㅇ', l_last[1], l_last[2]) + r
            if self.is_root(l_root):
                candidates.add((l_root, r_canon))

        # 거라, 너라 불규칙 활용
        # '-거라/-너라'를 어미로 취급하면 규칙 활용
        # if (l[-1] == '가') and (r and (r[0] == '라' or r[:2] == '거라')):
        #    # TODO

        # 러 불규칙 활용: 이르 + 러 -> 이르다
        # if (r_first[0] == 'ㄹ' and r_first[1] == 'ㅓ'):
        #     if self.is_root(l):
        #         # TODO

        # 여 불규칙 활용
        # 하 + 였다 -> 하 + 았다 -> 하다: '였다'를 어미로 취급하면 규칙 활용

        # ㅎ (탈락) 불규칙 활용
        # 파라 + 면 -> 파랗다
        if (l_last[2] == ' ' or l_last[2] == 'ㄴ' or l_last[2] == 'ㄹ' or l_last[2] == 'ㅂ' or l_last[2] == 'ㅆ'):
            l_root = l[:-1] + compose(l_last[0], l_last[1], 'ㅎ')
            r_canon = r if l_last[2] == ' ' else l_last[2] + r
            if self.is_root(l_root):
                candidates.add((l_root, r_canon))

        # ㅎ (축약) 불규칙 할용
        # 시퍼렜 + 다 -> 시퍼렇다, 파랬 + 다 -> 파랗다, 파래 + '' -> 파랗다
        if (l_last[1] == 'ㅐ') or (l_last[1] == 'ㅔ'):
            # exception : 그렇 + 아 -> 그래
            if len(l) >= 2 and l[-2] == '그' and l_last[0] == 'ㄹ':
                l_root = l[:-1] + '렇'
            else:
                l_root = l[:-1] + compose(l_last[0], 'ㅓ' if l_last[1] == 'ㅔ' else 'ㅏ', 'ㅎ')
            r_canon = compose('ㅇ', 'ㅓ' if l_last[1] == 'ㅔ' else 'ㅏ', l_last[2]) + r
            if self.is_root(l_root):
                candidates.add((l_root, r_canon))

        ## Pre-defined set
        if (l, r) in self._predefined:
            for root in self._predefined[(l, r)]:
                candidates.add(root)

        return candidates
{% endhighlight %}

## 테스트 코드 및 결과

용언의 어근 사전이 주어졌을 때 이를 이용하여 가능한 어근과 어미의 원형을 찾는 테스트를 수행합니다.

{% highlight python %}
roots = {
    '깨닫', '가', # ㄷ 불규칙
    '구르', '들르', # 르 불규칙
    '더럽',  '곱', '감미롭', # ㅂ 불규칙 (1)
    '이', '하', '푸르', # # 어미의 첫글자가 종성일 경우
    '낫', '긋', '벗', # ㅅ 불규칙
    '푸', '주', '누', # 우 불규칙
    '오', # 오 불규칙 (가제, 규칙 활용 ㅗ + ㅏ = ㅘ)
    '끄', '트', # ㅡ 탈락 불규칙
    '파랗', '하얗', '그렇', '시퍼렇', '노랗' # ㅎ (탈락) 불규칙
}

testset = [
    '깨달아', '가고', # ㄷ 불규칙
    '굴러', '구르라니까', '들러', '들렀다', # 르 불규칙     
    '더러워서', '더럽다', '고와', '감미로워서',  # ㅂ 블규칙 (1)
    '입니다', '합니다', '합니까', '한답니다', '할껄', '있어요', '푸른', # 어미의 첫글자가 종성일 경우
    '나았어', '그어버려', '벗어던져',  # ㅅ 불규칙
    '퍼갔어', '줬습니다', '눴어', # 우 불규칙
    '왔다', # 오 불규칙
    '껐다', '껐어', '텄어', # ㅡ 탈락 블규칙
    '파란', '파라면', '하얀', '노란', # ㅎ (탈락) 불규칙
    '파랬다', '그래', '그랬다', '그랬지', '시퍼렜다', #ㅎ (축약) 불규칙
]

lemmatizer = Lemmatizer(roots = roots)

for word in testset:
    candidates = lemmatizer.candidates(word)
    print('{} : {}'.format(word, candidates))
{% endhighlight %}

결과는 아래와 같습니다. '하얀'의 경우에는 '하다'와 '하얗다'가 어근의 원형의 후보로 생성됩니다. 물론 '-얀'이라는 어미는 존재하지 않기 때문에 '하얗다'가 정답입니다. 이 부분은 lemmatize 함수에서 구현할 부분입니다.

	깨달아 : {('깨닫', '아')}
	가고 : {('가', '고')}
	굴러 : {('구르', '어')}
	구르라니까 : {('구르', '라니까')}
	들러 : {('들르', '어')}
	들렀다 : {('들르', '었다')}
	더러워서 : {('더럽', '어서')}
	더럽다 : {('더럽', '다')}
	고와 : {('곱', '아')}
	감미로워서 : {('감미롭', '어서')}
	입니다 : {('이', 'ㅂ니다')}
	합니다 : {('하', 'ㅂ니다')}
	합니까 : {('하', 'ㅂ니까')}
	한답니다 : {('하', 'ㄴ답니다')}
	할껄 : {('하', 'ㄹ껄')}
	있어요 : {('이', 'ㅆ어요')}
	푸른 : {('푸르', 'ㄴ'), ('푸', '른')}
	나았어 : {('낫', '았어')}
	그어버려 : {('긋', '어버려')}
	벗어던져 : {('벗', '어던져')}
	퍼갔어 : {('푸', '어갔어')}
	줬습니다 : {('주', '었습니다')}
	눴어 : {('누', '었어')}
	왔다 : {('오', '았다')}
	껐다 : {('끄', '었다')}
	껐어 : {('끄', '었어')}
	텄어 : {('트', '었어')}
	파란 : {('파랗', 'ㄴ')}
	파라면 : {('파랗', '면')}
	하얀 : {('하얗', 'ㄴ'), ('하', '얀')}
	노란 : {('노랗', 'ㄴ')}
	파랬다 : {('파랗', '았다')}
	그래 : {('그렇', '아')}
	그랬다 : {('그렇', '았다')}
	그랬지 : {('그렇', '았지')}
	시퍼렜다 : {('시퍼렇', '었다')}

위 코드와 테스트는 soynlp.lemmatizer 에 구현되어 있습니다.

## References
- [나무위키: 용언 활용][lemmatize_wiki]

[lemmatize_wiki]: https://en.wikipedia.org/wiki/Lemmatisation
[korean_lemmatize]: https://namu.wiki/w/%ED%95%9C%EA%B5%AD%EC%96%B4/%EB%B6%88%EA%B7%9C%EC%B9%99%20%ED%99%9C%EC%9A%A9
[lr_structure]: {{ site.baseurl }}{% link _posts/2018-05-07-noun_extraction_ver1.md %}