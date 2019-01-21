---
title: 말뭉치를 이용한 한국어 용언 분석기 (Korean Lemmatizer)
date: 2019-01-22 09:00:00
categories:
- nlp
tags:
- lemmatization
---

한국어의 단어는 9 품사로 이뤄져 있으며, 그 중 동사와 형용사는 목적에 따라 그 형태가 변하기 때문에 용언이라 부릅니다. 동사와 형용사는 중심 의미를 지니는 어간 (stem) 과 시제와 같은 문법적 기능을 하는 어미 (eomi, ending) 가 결합하여 표현형 (surfacial form) 이 이뤄집니다. 때로는 표현형에서 어간과 어미를 분리하거나, 원형 (canonical form, lemma) 으로 복원해야 할 필요가 있습니다. 이번 포스트에서는 사전과 규칙 기반으로 이뤄진 한국어 용언 분석기를 만들어 봅니다.

## 단어와 형태소

언어의 최소 단위는 의미를 지니는 단어 입니다. 한국어의 단어는 5 언 9 품사로 이뤄져 있습니다. 명사, 대명사, 수사는 개념이나 숫자를 표현하는 단어이며, 관형사와 부사는 각각 명사와 용언을 수식합니다. 조사는 명사 뒤에 위치하여 어절을 구성하는데, 체언의 문법적 역할을 규정합니다. 예를 들어 목적격 조사는 체언을 목적어로 만듭니다.

| 언 | 품사 |
| --- | --- |
| 체언 | 명사, 대명사, 수사 |
| 수식언 | 관형사, 부사 |
| 관계언 | 조사 |
| 독립언 | 감탄사 |
| 용언 | 동사, 형용사 |

형태소는 그런 언어를 구성하는 단위 (unit) 입니다. 위의 각 단어들은 그 자체가 형태소이기도 합니다. 조사나 명사는 형태소이면서 단어입니다. 때로 여러 개의 명사가 교착하여 복합명사를 이루기도 합니다. `골목식당`은 `골목` 과 `식당` 이라는 두 개의 명사 형태소가 결합된 복합 형태소 입니다. 하지만 그 자체로 단어 명사이기도 합니다.

명사는 어절을 이루기 위하여 반드시 조사가 필요한 것은 아닙니다. 명사 자체가 어절이 되는 경우는 많습니다.

그런데 문법 기능을 하면서도 단어가 아닌 형태소가 있습니다. 용언의 `어간`과 `어미` 입니다. 미는 그 자체가 단어이지는 않습니다. 그리고 어미는 반드시 용언의 어간과 함께 이용되어야 합니다. 어간과 어미가 결합되어 동사나 형용사가 이뤄집니다.

```
합니다/동사 = 하/동사 어간 + ㅂ니다/어미
```

앞서 언급한 것처럼 조사는 그 역할에 따라 다양한 종류가 있습니다. 세종 말뭉치의 형태소 구성표에는 주격 조사, 목적격 조사, 서술격 조사 등이 존재합니다. 어미 역시 선어말어미, 어말어미 등 다양한 종류가 있습니다. 그러나 구문 분석처럼 한 문장에 대하여 구조적인 분석을 하는 경우가 아니라면 이를 구분하여 이용하지 않아도 됩니다. 그렇기 때문에 형태소, 품사 구성 체계는 그 목적에 따라 자주 변경되곤 합니다. 그 예시로 파이썬 한국어 분석기인 `KoNLPy` 의 다양한 형태소 분석기들의 품사 체계가 다양함을 들 수 있습니다.

저 역시 단어 추출과 데이터 분석의 편의성을 위해 위의 9 품사 + 어간, 어미 정도만 품사 체계로 자주 이용하며, 이는 Open Korean Text 에서도 비슷합니다. 이후 이 포스트에서의 품사 체계는 9 품사 + 어간, 어미로 이야기 합니다.

## Conjugation vs Lemmatization

이전에 [관련 포스트][conjugator_post]에서 활용 (conjugation) 과 원형 복원 (lemmatization) 에 대하여 다뤘습니다. 다시 간단히 용어만 정리하자면 활용은 용언의 기본형 (canonical form) 의 어미가 교체되어 표현형 (surfacial form) 이 되는 것을 의미합니다. 기본형의 어미는 `다/Eomi` 입니다. 이 어미가 `았다/Eomi` 로 교체되면서 표현형이 `했다`로 변하는 것을 활용이라 합니다.

```
기본형 : 하/Verb + 다/Eomi -> 하다
표현형 : 하/Verb + 았다/Eomi -> 했다
```

반대로 표현형이 기본형으로 변화하는 것을 lemmatization 이라 합니다. Conjugation 과 역관계 입니다.

## 용언의 규칙 활용과 불규칙 활용

용언은 `어간`과 `어미`가 결합하여 만들어집니다. 그 과정에서 어떤 경우에는 어간과 어미의 형태가 그대로 유지되며 concatenation 이 일어나고, 어떤 경우에는 서로 결합되는 부분에서 어간 혹은 어미의 형태가 변형되기도 합니다.

이후부터는 형태소 (morpheme) 와 품사 (tag) 를 쉽게 구분하기 위하여 품사는 영어로 기술합니다. 어간의 `Verb` 는 단어의 동사가 아닌 형태소 어간의 동사를 의미합니다.

용언의 규칙 활용은 초/중/종성의 단위에서 concatenation 이 일어나는 현상을 의미합니다. 아래의 예시는 어간과 어미의 형태 변화가 없습니다.

```
시작하/Verb + 는/Eomi  -> 시작하는
```

이러한 경우는 파이썬 코드로 구현하기도 쉽습니다.

```python
def conjugate(stem, eomi):
    return stem + eomi

conjugate('시작하', '는') # '시작하는'
```

그런데 규칙 활용은 음절 단위가 아닌 초/중/종성 기준에서의 concatenation 입니다. 어미 `ㄴ/Eomi` 가 마지막 글자의 종성이 없는 어간 `시작하/Verb` 와 결합될 때에는 빈 종성의 자리를 `ㄴ` 이 매꿉니다.

```
시작하/Verb + ㄴ/Eomi -> 시작한
```

이를 구현하려면 한글의 음절을 초/중/종성으로 분해하는 `decompose` 함수가 필요합니다. 이 함수의 구현에 대해서는 언급하지 않겠습니다. 코드는 [여기][lemmatizer_git]의 `soylemma.hangle`에 있습니다. 코드는 개념적으로 설명하겠습니다.

```python
def conjugate(stem, eomi):
    cho_s, jung_s, jong_s = decompose(stem[-1])
    cho_e, jung_e, jong_e = decompose(eomi[0])
    if jong_s == ' ' and jung_e == ' ':
        return stem[:-1] + compose(cho_s, jung_s, cho_e) + eomi[1:]
    return stem + eomi
```

문법에서는 규칙 활용이지만 구현 측면에서는 concatenation 만으로는 되지 않는 경우입니다. 음절 단위의 글자에서는 형태 변화가 있지만 한글을 자음/모음 sequence 로 변환하면 그 형태가 변하지 않기 때문입니다.

때로는 세종 말뭉치에 `ㅏㅆ/EP` 와 같은 어미도 존재하는데 이는 `았/EP` 로 바꿔 이용하시는게 좋습니다. 세종 말뭉치에 통일성이 없어서 경우에 따라 다르게 기술된 경우들이 있습니다. 그 외에는 어미의 첫 글자가 자음 혹은 한글입니다.

불규칙 활용은 이와 같이 자음/모음 sequence 에 변화가 생기는 경우를 의미합니다. 그리고 이 불규칙 활용도 규칙이 있습니다. 불규칙 활용에 법칙이 있다니. 처음 용언의 활용을 공부할 때, 이 '불규칙 활용'이란 말의 기원이 정말 궁금했습니다 (아직도 모릅니다).

불규칙 활용의 몇 가지 예시입니다. 이들은 초/중/종성의 sequence 에서 `ㄹㅏㅎㄴ` -> `ㄹㅏㄴ` 으로, `ㅎㅏㅇㅏㅆ` -> `ㅎㅐㅆ` 으로 변화하였습니다.

```
파랗/Adjective + ㄴ/Eomi -> 파란
하/Verb + 았다/Eomi -> 했다
```

다행히도 규칙은 있고, 이는 이전의 [lemmatizer post][lemmatizer_post] 나 [conjugator post][conjugator_post] 에 규칙 기반으로 구현하는 방법을 적어뒀습니다. 이 내용이 궁금하신 분들은 해당 포스트를 참고하세요. 규칙 기반으로 이를 구현했던 이유는 학습 데이터가 존재하지 않거나, 말투 때문에 발생하는 새로운 어미들을 추출하기 위해서입니다.

## 말뭉치를 이용한 conjugate, lemmatize 함수

만약 우리가 말뭉치를 가지고 있다면 그 말뭉치로부터 '활용' 혹은 '원형 복원' 규칙을 학습하여 이용할 수도 있습니다. 이번 포스트에서 작업할 내용입니다.

만약 우리가 용언이 활용 될 때 `어간의 마지막 글자`와 `어미의 첫글자`의 형태 변화를 다음과 같은 규칙으로 가지고 있다면 이전 포스트처럼 복잡한 규칙을 이용하지 않으면서도 용언의 형태소를 인식할 수 있습니다. `파랗ㄴ` 을 제거하는 과정은 여기에서는 설명하진 않겠습니다. 표현형은 자음, 모음이 아닌 한글이어야 합니다. 여하튼 규칙을 가지고 있다면 손쉽게 conjugation 을 할 수 있습니다.

```python
lemma_rules = {'란': {('랗', 'ㄴ')}, '했': {('하', '았')}}
conju_rules = {('랗', 'ㄴ'): {'란'}, ('하', '았'): {'했'}}

def conjugate(stem, eomi, rules):
    key = (stem[-1], eomi[0])
    surfaces = [stem + eomi]
    for conjugation in rules.get(key, {}):
        surfaces.append(stem[:-1] + conjugation + eomi[1:])
    return surfaces

conjugate('파랗', 'ㄴ', conju_rules) # ['파란', '파랗ㄴ']
```

이와 비슷하게 lemmatization 도 할 수 있습니다. 단어 `word` 에서 어간의 마지막 글자의 위치를 `i` 로 입력합니다. `파란`의 두 번째 글자가 어간의 마지막이므로 `i=2` 를 입력하면 `란` 이 `lemma_rules` 에 key 로 있는지 확인하여 어간과 어미의 원형 후보를 만들 수 있습니다.

```python
def _lemmatize(word, i, rules):
    key = word[i-1]
    lemmas = [(word[:i], word[i:])]
    for s, e in rules.get(key, {}):
        lemmas.append((word[:i-1] + s, e + word[i:]))
    return lemmas

_lemmatize('파란', 2, lemma_rules) # [('파', '란'), ('파랗', 'ㄴ')]
```

이제 어간, 어미 후보쌍이 모두 사전에 존재하는 단어인지 확인합니다. 그리고 실제로는 단어에서 어간의 마지막 글자의 위치를 알지 못하기 때문에 모든 경우를 탐색합니다. `adjectives` 와 `eomis` 는 각각 형용사아 어미 사전입니다. 물론 동사와 형용사를 한 번에 확인하는 것이 좋기 때문에 `lemmatizer` 의 arguments 에 `verbs` 도 추가합니다.

```python
def lemmatize(word, rules, adjectives, verbs, eomis):
    lemmas = []
    # generate candidates
    for i in range(1, len(word) + 1):
        lemmas += _lemmatize(word, i, rules)
    # check dictionary
    lemmas_ = []
    for stem, eomi in lemmas:
        if not ((stem in adjectives) and (eomi in eomis)):
            continue
        lemmas_.append((stem, eomi))
    return lemmas_
```

## 형태 변화의 종류

앞선 예시에서는 단어의 1 음절만 형태 변화가 있는 경우를 예시로 들었습니다만, 실제는 2 음절과 3음절에서도 형태 변화가 일어납니다. 몇 가지 예시 입니다.

| 형태 변화 음절 길이 | 형태 변화 규칙 | 단어 예시 |
| --- | --- | --- |
| 1 음절 | `했` = `하 + 았` | 시작했으니까 = 시작하 + 았으니까 |
| 1 음절 | `랬` = `랗 + 았` | 파랬던 = 파랗 + 았던 |
| 2 음절 | `추운` = `춥 + 은` | 추운데 = 춥 + 은데 |
| 2 음절 | `했다` = `하 + 았다` | 시작했다 = 시작하 + 았다 |
| 3 음절 | `가우니` = `갑 + 니` | 차가우니까 = 차갑 + 니까 |

이로부터 활용과 원형 복원 규칙을 만들 수 있습니다.

```
lemma_rules = {
    '했' : {('하', '았')}
    '랬' : {('랗', '았')}
    '추운' : {('춥', '은')}
    '했다' : {('하', '았다')}
    '가우니' : {('갑', '니')}
}

conju_rules = {
    ('하', '았'): {'했'}
    ('랗', '았'): {'랬'}
    ('춥', '은'): {'추운'}
    ('하', '았다'): {'했다'}
    ('갑', '니'): {'가우니'}
}
```

그리고 이를 모두 이용하여 어간, 어미의 후보를 만드는 함수를 만들 수 있습니다.

```python
def get_lemma_candidates(word, rules):
    max_i = len(word) - 1
    candidates = []
    for i, c in enumerate(word):
        l = word[:i+1]
        r = word[i+1:]
        l_ = word[:i]
        # concatenation
        if i < max_i:
            candidates.append((l, r))

        # 1 syllable conjugation
        for stem, eomi in rules.get(c, {}):
            for stem, eomi in rules.get(c, {}):
                candidates.append((l_ + stem, eomi + r))

        # 2 or 3 syllables conjugation
        for conj in {word[i:i+2], word[i:i+3]}:
            for stem, eomi in rules.get(conj, {}):
                candidates.append((l_ + stem, eomi + r[1:]))
    return candidates
```

## 형태 변화 규칙과 단어 사전 구성

이제 사전과 형태 변화 규칙을 준비해야 합니다. 이를 위하여 세종 말뭉치를 이용하였습니다. 그러나 세종 말뭉치의 형태소 체계는 매우 복잡하기 때문에 이를 간략화 했습니다. 예를 들어 어절 `로드무비인`은 아래처럼 구성되어 있습니다. `NNG` 는 일반 명사, `VCP` 는 긍정 지정사 (명사와 결합되어 형용사화하는 보조 용언), `ETM` 은 관형형 전성 어미로, 이 어절은 관형사처럼 체언을 수식하는 능력을 지닙니다.

```
로드/NNG + 무비/NNG + 이/VCP + ㄴ/ETM
```

어절이 `로드무비였다` 라면 세종 말뭉치의 품사 체계에서는 아래처럼 분석됩니다. `EP` 는 선어말 어미, `EC` 는 연결 어미 입니다.

```
로드/NNG + 무비/NNG + 이/VCP + 었/EP + 다/EC
```

이는 마치 `로드무비이` 까지가 어간이고 `었다`가 어미처럼 역할을 합니다. `었다` 대신에 `ㄴ`이 결합되면 `로드무비인`이 되니까요. 이처럼 어절 내 복합형태소를 단일한 하나의 형태소로 결합하여 어절 내 구성 요소를 간략화 합니다. [이전의 포스트][lr_post]에서 언급한 어절의 `L + [R]` 구조 입니다. 물론 세종 말뭉치의 어절내 형태소의 개수가 2 개로 간단하다면 이를 이용합니다.

품사 체계도 간략화 합니다. 모든 종류의 어미는 `Eomi` 로 만들어 9 품사 + 어간, 어미가 되도록 합니다. 이를 위해서는 [`sejong_corpus_cleaner`][sejong_cleaner_git] 의 `eojeol_morphtags_to_lr` 함수를 이용합니다. 이를 이용하면 아래와 같은 결과를 얻을 수 있습니다.

```python
from sejong_corpus_cleaner.simplier import eojeol_morphtags_to_lr

eojeol_morphtags_to_lr('로드무비였다', [('로드', 'NNG'), ('무비', 'NNG'), ('이', 'VCP'), ('었', 'EP'), ('다', 'EC')], separate_xsv=False)
```

```
('로드무비이', '었다', 'Adjective', 'Eomi')
```

그리고 어간의 마지막 글자 `이` 와 어미의 첫글자 `었`, 어절에서의 이들의 위치를 고려하여 규칙을 추출합니다. 이는 [korean_lemmatizer github][lemmatizer_git] 의 `extract_rule` 함수에 구현되어 있습니다.

```
from soylemma import extract_rule

eojeol = '로드무비였다'
lw = '로드무비이'
lt = 'Adjective'
rw = '었다'
rt = 'Eomi'

extract_rule(eojeol, lw, lt, rw, rt)
```

```
('였다', ('이', '었다'))
```

세종 말뭉치로부터 eojeol_morphtags_to_lr 함수가 작동하는 어절만 이용하여 (품사에 오류가 있거나, 아직 발견하지 못한 함수의 버그가 있어서) (L, R) 을 만든 뒤, extract_rule 함수를 이용하여 규칙들을 추출합니다.

동사와 형용사 사전은 L + [R] 구조의 복합 형태소가 아닌, 세종 말뭉치 형태소에서 추출했으며, 어미 사전은 L + [R] 구조로 만든 뒤 R 에서 Eomi 인 경우들을 모았습니다.

## 구현체

[`github.com/lovit/korean_lemmatizer`][lemmatizer_git] 에 구현체와 세종 말뭉치로부터 학습한 사전, 형태 변화 규칙을 올려두었습니다.

이 패키지의 사용법은 README 에 적어두었습니다.

한 가지, 구조와 원리를 파악하기 위해서는 규칙이나 사전의 크기가 클 경우 복잡합니다. 이를 위해서 `demo` 용 사전을 따로 만들어뒀습니다. 이를 이용하여 테스트를 하려면 아래처럼 Lemmatizer 를 만듭니다.

```python
from soylemma import Lemmatizer

lemmatizer = Lemmatizer(dictionary_name='demo')
```

세종 말뭉치로 학습된 Lemmatizer 를 이용하려면 아래처럼 Lemmatizer 를 만듭니다. `dictionary_name` 의 기본값인 `default`는 세종말뭉치를 이용하는 모델입니다.

```python
from soylemma import Lemmatizer

lemmatizer = Lemmatizer()

# or
lemmatizer = Lemmatizer(dictionary_name='default')
```


[lemmatizer_git]: https://github.com/lovit/korean_lemmatizer
[sejong_cleaner_git]: https://github.com/lovit/sejong_corpus_cleaner
[lr_post]: {{ site.baseurl }}{% link _posts/2018-04-09-cohesion_ltokenizer.md %}
[lemmatizer_post]: {{ site.baseurl }}{% link _posts/2018-06-07-lemmatizer.md %}
[conjugator_post]: {{ site.baseurl }}{% link _posts/2018-06-11-conjugator.md %}