---
title: Levenshtein (edit) distance 를 이용한 한국어 단어의 형태적 유사성
date: 2017-08-28 05:00:00
categories:
- nlp
tags:
- string distance
---

단어, 혹은 문장과 같은 string 간의 형태적 유사성을 정의하는 방법을 string distance 라 합니다. Edit distance 라는 별명을 지닌 Levenshtein distance 는 대표적인 string distance metric 중 하나입니다. 그러나 Levenshtein distance 는 한국어처럼 각 글자가 요소들 (초/중/종성)로 이뤄진 언어를 고려한 metric 이 아닙니다. 이번 포스트에서는 Levenshtein distance 를 한글에 적합하도록 변형하는 방법에 대하여 알아봅니다.

## Why string distance ?

단어 간 거리는 형태적 거리와 의미적 거리로 분류됩니다. 의미적 거리는 word embedding 과 같은 방법으로 학습할 수 있습니다. 대표적인 word embedding 방법 중 하나인 [Word2Vec][word2vec] 은 단어 간 의미적 유사성 (거리)를 벡터로 표현합니다. Word2Vec 을 통하여 학습된 단어 (영화, 애니메이션)의 벡터 간 cosine distance 는 매우 작습니다. 하지만 두 단어의 형태적 유사성은 없습니다. 2 글자와 5 글자 사이에 공통된 음절이 하나도 존재하지 않습니다. 이처럼 string 의 형태적인 거리를 정의하는 방법을 [string distance][wikipedia] 라 합니다.

String distance metric 은 매우 다양합니다. Jaro-winkler, Levenshtein 과 같이 string 을 이용하여 정의되는 metrics 도 있으며, Hamming, Cosine, TF-IDF distance 와 같이 string 을 벡터로 표현한 다음 거리를 정의하는 방법도 있습니다. 그 중 Levenshtein distance 는 대표적인 metric 입니다. 이 방법의 별명은 edit distance 입니다. 한 string $$s_1$$ 에서 다른 $$s_2$$ 로 교정하는데 드는 최소 횟수를 두 strings 간의 거리로 정의합니다. 이번 포스트에서는 이 방법에 대하여 알아봅니다. 

String distance 는 오탈자 교정에 자주 이용됩니다. 정자 (right words)에 대한 사전이 존재한다면 정자가 아닌 단어들을 string distance 기준으로 가장 가까운 단어로 치환할 수 있습니다. Levenshtein distance 를 다룰 줄 알며, 정자 사전을 보유하였다면 오탈자 교정기를 만들 수 있습니다.

## Levenshtein (Edit) distance

Levenshtein distance 는 한 string $$s_1$$ 을 $$s_2$$ 로 변환하는 최소 횟수를 거리로 정의합니다. $$s_1$$ = '꿈을꾸는아이' 에서 $$s_2$$ = '아이오아이' 로 바뀌기 위해서는 (꿈을꾸 -> 아이오) 로 바뀌고, 네번째 글자 '는' 이 제거되면 됩니다. Levenshtein distance 에서는 이처럼 string 을 변화하기 위한 edit 방법을 세 가지로 분류합니다.

1. delete: '점심**을**먹자 $$\rightarrow$$ 점심먹자' 로 바꾸기 위해서는 **을** 을 삭제해야 합니다.
2. insert: '점심먹자 $$\rightarrow$$ 점심**을**먹자' 로 바꾸기 위해서는 반대로 **을** 을 삽입해야 합니다.
3. substitution: '점심먹**자** $$\rightarrow$$ 점심먹**장**' 로 바꾸기 위해서는 **자**를 **장** 으로 치환해야 합니다.

$$s_1$$ = '꿈을꾸는아이' 에서 $$s_2$$ = '아이오아이' 로 변환하는 방법은 다양합니다. '꿈'을 지우고 '아'를 입력할 수도 있습니다. 하지만 이 때 비용은 2 번의 수정 (edit) 입니다. 이보다는 '꿈'을 '아'로 변환하는 것이 1 의 비용만 들기 때문에 더 쌉니다. 이처럼 가장 적은 비용이 드는 수정 방법을 찾는 것이 Levenshtein distance 의 목표입니다.

이를 위해 동적 프로그래밍 (dynamic programming) 이 이용됩니다. 이는 전체 문제를 작은 문제의 집합으로 정의하고, 작은 문제를 반복적으로 풂으로써 전체 문제의 해법을 찾는 방법론 입니다. 그리고 Levenshtein 은 dynamic programming 의 연습용으로 자주 등장하는 예제이기도 합니다.

## Levenshtein distance 구현하기

### Base version

{% highlight python %}
def levenshtein(s1, s2, debug=False):
    if len(s1) < len(s2):
        return levenshtein(s2, s1, debug)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))

        if debug:
            print(current_row[1:])

        previous_row = current_row

    return previous_row[-1]
{% endhighlight %}


{% highlight python %}
s1 = '꿈을꾸는아이'
s2 = '아이오아이'
levenshtein(s1, s2, debug=True)
{% endhighlight %}

    [1, 2, 3, 4, 5]
    [2, 2, 3, 4, 5]
    [3, 3, 3, 4, 5]
    [4, 4, 4, 4, 5]
    [4, 5, 5, 4, 5]
    [5, 4, 5, 5, 4]

    4


{% highlight python %}
s1 = '아이돌'
s2 = '아이오아이'
levenshtein(s1, s2, debug=True)
{% endhighlight %}

    [0, 1, 2]
    [1, 0, 1]
    [2, 1, 1]
    [3, 2, 2]
    [4, 3, 3]

    3

{% highlight python %}
s1 = '꿈을 꾸는 아이'
s2 = '아이는 꿈을 꿔요'
levenshtein(s1, s2, debug=True)
{% endhighlight %}

    [1, 2, 3, 4, 5, 6, 6, 7]
    [2, 2, 3, 4, 5, 6, 7, 6]
    [3, 3, 3, 4, 4, 5, 6, 7]
    [4, 4, 3, 4, 5, 4, 5, 6]
    [4, 5, 4, 4, 5, 5, 5, 6]
    [5, 4, 5, 5, 5, 6, 6, 6]
    [6, 5, 4, 5, 6, 5, 6, 7]
    [7, 6, 5, 5, 6, 6, 6, 7]
    [8, 7, 6, 6, 6, 7, 7, 7]

    7

{% highlight python %}
# 어절 단위
s1 = '꿈을 꾸는 아이'
s2 = '아이는 꿈을 꿔요'
levenshtein(s1.split(), s2.split(), debug=True)
{% endhighlight %}

    [1, 1, 2]
    [2, 2, 2]
    [3, 3, 3]
    
    3

### User define cost + Levenshtein

{% highlight python %}
def levenshtein(s1, s2, cost=None, debug=False):
    if len(s1) < len(s2):
        return levenshtein(s2, s1, debug=debug)

    if len(s2) == 0:
        return len(s1)

    if cost is None:
        cost = {}

    def substitution_cost(c1, c2):
        if c1 == c2:
            return 0
        return cost.get((c1, c2), 1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            # Changed
            substitutions = previous_row[j] + substitution_cost(c1, c2)
            current_row.append(min(insertions, deletions, substitutions))

        if debug:
            print(current_row[1:])

        previous_row = current_row

    return previous_row[-1]
{% endhighlight %}


{% highlight python %}
s1 = '아이쿠야'
s2 = '아이쿵야'
levenshtein(s1, s2, debug=True)
{% endhighlight %}

    [0, 1, 2, 3]
    [1, 0, 1, 2]
    [2, 1, 1, 2]
    [3, 2, 2, 1]

    1

{% highlight python %}
cost = {('쿠', '쿵'):0.1}
s1 = '아이쿠야'
s2 = '아이쿵야'
levenshtein(s1, s2, cost, debug=True)
{% endhighlight %}


    [0, 1, 2, 3]
    [1, 0, 1, 2]
    [2, 1, 0.1, 1.1]
    [3, 2, 1.1, 0.1]

    0.1

### 한글의 초/중/종성 분리

{% highlight python %}
for char in 'azAZ가힣ㄱㄴㅎㅏ':
    print('{} == {}'.format(char, ord(char)))
{% endhighlight %}

    a == 97
    z == 122
    A == 65
    Z == 90
    가 == 44032
    힣 == 55203
    ㄱ == 12593
    ㄴ == 12596
    ㅎ == 12622
    ㅏ == 12623

{% highlight python %}
for idx in [97, 122, 65, 90, 44032, 55203]:
    print('{} == {}'.format(idx, chr(idx)))
{% endhighlight %}

    97 == a
    122 == z
    65 == A
    90 == Z
    44032 == 가
    55203 == 힣

{% highlight python %}
kor_begin = 44032
kor_end = 55203
chosung_base = 588
jungsung_base = 28
jaum_begin = 12593
jaum_end = 12622
moum_begin = 12623
moum_end = 12643

chosung_list = [ 'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 
        'ㅅ', 'ㅆ', 'ㅇ' , 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 
        'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 
        'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 
        'ㅡ', 'ㅢ', 'ㅣ']

jongsung_list = [
    ' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ',
        'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 
        'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 
        'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jaum_list = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ', 
              'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 
              'ㅃ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

moum_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 
              'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

def compose(chosung, jungsung, jongsung):
    char = chr(
        kor_begin +
        chosung_base * chosung_list.index(chosung) +
        jungsung_base * jungsung_list.index(jungsung) +
        jongsung_list.index(jongsung)
    )
    return char

def decompose(c):
    if not character_is_korean(c):
        return None
    i = ord(c)
    if (jaum_begin <= i <= jaum_end):
        return (c, ' ', ' ')
    if (moum_begin <= i <= moum_end):
        return (' ', c, ' ')
    i -= kor_begin
    cho  = i // chosung_base
    jung = ( i - cho * chosung_base ) // jungsung_base 
    jong = ( i - cho * chosung_base - jung * jungsung_base )    
    return (chosung_list[cho], jungsung_list[jung], jongsung_list[jong])

def character_is_korean(c):
    i = ord(c)
    return ((kor_begin <= i <= kor_end) or
            (jaum_begin <= i <= jaum_end) or
            (moum_begin <= i <= moum_end))
{% endhighlight %}


{% highlight python %}
decompose('감') # ('ㄱ', 'ㅏ', 'ㅁ')
{% endhighlight %}


{% highlight python %}
compose('ㄲ', 'ㅜ', 'ㅁ') # '꿈'
{% endhighlight %}


### 초/중/종성 분리를 적용한 Levenshtein distance

{% highlight python %}
def jamo_levenshtein(s1, s2, debug=False):
    if len(s1) < len(s2):
        return jamo_levenshtein(s2, s1, debug)

    if len(s2) == 0:
        return len(s1)

    def substitution_cost(c1, c2):
        if c1 == c2:
            return 0
        return levenshtein(decompose(c1), decompose(c2))/3

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            # Changed
            substitutions = previous_row[j] + substitution_cost(c1, c2)
            current_row.append(min(insertions, deletions, substitutions))

        if debug:
            print(['%.3f'%v for v in current_row[1:]])

        previous_row = current_row

    return previous_row[-1]
{% endhighlight %}


{% highlight python %}
s1 = '아이쿠야'
s2 = '아이쿵야'
jamo_levenshtein(s1, s2, debug=True)
{% endhighlight %}

    ['0.000', '1.000', '2.000', '3.000']
    ['1.000', '0.000', '1.000', '2.000']
    ['2.000', '1.000', '0.333', '1.333']
    ['3.000', '2.000', '1.333', '0.333']

    0.3333333333333333

{% highlight python %}
s1 = '아이쿵야'
s2 = '훍앜이쿠야'
jamo_levenshtein(s1, s2, debug=True)
{% endhighlight %}

    ['1.000', '2.000', '2.667', '3.667']
    ['1.333', '1.667', '2.667', '3.333']
    ['2.333', '1.333', '2.333', '3.000']
    ['3.333', '2.333', '1.667', '2.667']
    ['4.333', '3.333', '2.667', '1.667']

    1.6666666666666665

{% highlight python %}
s1 = '아이쿠야'
s2 = '아이쿵야'

s1_ = ''.join([comp for c in s1 for comp in decompose(c)])
s2_ = ''.join([comp for c in s2 for comp in decompose(c)])

print(s1_) # ㅇㅏ ㅇㅣ ㅋㅜ ㅇㅑ 
print(s2_) # ㅇㅏ ㅇㅣ ㅋㅜㅇㅇㅑ 
print(levenshtein(s1_, s2_)/3) # 0.3333333333333333
{% endhighlight %}


{% highlight python %}
s1 = '아이쿵야'
s2 = '훍앜이쿠야'

s1_ = ''.join([comp for c in s1 for comp in decompose(c)])
s2_ = ''.join([comp for c in s2 for comp in decompose(c)])

print(s1_) # ㅇㅏ ㅇㅣ ㅋㅜㅇㅇㅑ 
print(s2_) # ㅎㅜㄺㅇㅏㅋㅇㅣ ㅋㅜ ㅇㅑ 
print(levenshtein(s1_, s2_)/3) # 1.6666666666666667
{% endhighlight %}


## soynlp

앞서 언급한 compose, decompose 함수 및 levenshtein, 초/중/종성 분리를 적용한 jamo_levenshtein 를 soynlp 에 구현해 두었습니다.

{% highlight python %}
from soynlp.hangle import levenshtein
from soynlp.hangle import jamo_levenshtein

s1 = '아이쿠야'
s2 = '아이쿵야'

print(levenshtein(s1, s2)) # 1
print(jamo_levenshtein(s1, s2)) # 0.3333333333333333
{% endhighlight %}

{% highlight python %}
from soynlp.hangle import compose
from soynlp.hangle import decompose

decompose('꼭') # ('ㄲ', 'ㅗ', 'ㄱ')
{% endhighlight %}

[wikipedia]: https://en.wikipedia.org/wiki/String_metric
[word2vec]: {{ site.baseurl }}{% link _posts/2018-03-26-word_doc_embedding.md %}
[next]: {{ site.baseurl }}{% link _posts/2018-08-21-ford_for_pos.md %}