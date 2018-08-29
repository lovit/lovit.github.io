---
title: Levenshtein (edit) distance 를 이용한 한국어 단어의 형태적 유사성
date: 2018-08-28 05:00:00
categories:
- nlp
tags:
- string distance
---

단어, 혹은 문장과 같은 string 간의 형태적 유사성을 정의하는 방법을 string distance 라 합니다. Edit distance 라는 별명을 지닌 Levenshtein distance 는 대표적인 string distance metric 중 하나입니다. 그러나 Levenshtein distance 는 한국어처럼 각 글자가 요소들 (초/중/종성)로 이뤄진 언어를 고려한 metric 이 아닙니다. 이번 포스트에서는 Levenshtein distance 를 한글에 적합하도록 변형하는 방법에 대하여 알아봅니다.

## Why string distance ?

단어 간 거리는 형태적 거리와 의미적 거리로 분류됩니다. 의미적 거리는 word embedding 과 같은 방법으로 학습할 수 있습니다. 대표적인 word embedding 방법 중 하나인 [Word2Vec][word2vec] 은 단어 간 의미적 유사성 (거리)를 벡터로 표현합니다. Word2Vec 을 통하여 학습된 단어 (영화, 애니메이션)의 벡터 간 cosine distance 는 매우 작습니다. 하지만 두 단어의 형태적 유사성은 없습니다. 2 글자와 5 글자 사이에 공통된 음절이 하나도 존재하지 않습니다. 이처럼 string 의 형태적인 거리를 정의하는 방법을 [string distance][string_distance_wikipedia] 라 합니다.

String distance metric 은 매우 다양합니다. Jaro-winkler, Levenshtein 과 같이 string 을 이용하여 정의되는 metrics 도 있으며, Hamming, Cosine, TF-IDF distance 와 같이 string 을 벡터로 표현한 다음 거리를 정의하는 방법도 있습니다. 그 중 Levenshtein distance 는 대표적인 metric 입니다. 이 방법의 별명은 edit distance 입니다. 한 string $$s_1$$ 에서 다른 $$s_2$$ 로 교정하는데 드는 최소 횟수를 두 strings 간의 거리로 정의합니다. 이번 포스트에서는 이 방법에 대하여 알아봅니다. 

String distance 는 오탈자 교정에 자주 이용됩니다. 정자 (right words)에 대한 사전이 존재한다면 정자가 아닌 단어들을 string distance 기준으로 가장 가까운 단어로 치환할 수 있습니다. Levenshtein distance 를 다룰 줄 알며, 정자 사전을 보유하였다면 오탈자 교정기를 만들 수 있습니다.

## Levenshtein (Edit) distance

Levenshtein distance 는 한 string $$s_1$$ 을 $$s_2$$ 로 변환하는 최소 횟수를 거리로 정의합니다. $$s_1$$ = '꿈을꾸는아이' 에서 $$s_2$$ = '아이오아이' 로 바뀌기 위해서는 (꿈을꾸 -> 아이오) 로 바뀌고, 네번째 글자 '는' 이 제거되면 됩니다. Levenshtein distance 에서는 이처럼 string 을 변화하기 위한 edit 방법을 세 가지로 분류합니다.

1. delete: '점심**을**먹자 $$\rightarrow$$ 점심먹자' 로 바꾸기 위해서는 **을** 을 삭제해야 합니다.
2. insert: '점심먹자 $$\rightarrow$$ 점심**을**먹자' 로 바꾸기 위해서는 반대로 **을** 을 삽입해야 합니다.
3. substitution: '점심먹**자** $$\rightarrow$$ 점심먹**장**' 로 바꾸기 위해서는 **자**를 **장** 으로 치환해야 합니다.

$$s_1$$ = '꿈을꾸는아이' 에서 $$s_2$$ = '아이오아이' 로 변환하는 방법은 다양합니다. '꿈'을 지우고 '아'를 입력할 수도 있습니다. 하지만 이 때 비용은 2 번의 수정 (edit) 입니다. 이보다는 '꿈'을 '아'로 변환하는 것이 1 의 비용만 들기 때문에 더 쌉니다. 이처럼 가장 적은 비용이 드는 수정 방법을 찾는 것이 Levenshtein distance 의 목표입니다.

이를 위해 동적 프로그래밍 (dynamic programming) 이 이용됩니다. 이는 전체 문제를 작은 문제의 집합으로 정의하고, 작은 문제를 반복적으로 풂으로써 전체 문제의 해법을 찾는 방법론 입니다. 그리고 Levenshtein 은 dynamic programming 의 연습용으로 자주 등장하는 예제이기도 합니다.

'데이터마이닝 $$\rightarrow$$ 데이타마닝'으로 변환하는 예제로 그 원리를 알아봅니다. Levenshtein distance 계산을 위해서 len($$s_1$$) by len($$s_2$$) 의 거리 행렬, d 를 만듭니다. 우리는 맨 윗줄을 0 번째 row 로, 그 아랫줄을 1 번째 row 로 이야기합니다. 

d[0,0] 은 $$s_1, s_2$$ 의 첫 글자가 같으면 0, 아니면 1로 초기화 합니다. 글자가 다르면 substitution cost 가 발생한다는 의미입니다. 그리고 그 외의 d[0,j]에 대해서는 d[0,j] = d[0,j-1] + 1 의 비용으로 초기화 합니다. 한글자씩 insertion 이 일어났다는 의미입니다. 이후에는 좌측, 상단, 좌상단의 값을 이용하여 거리 행렬 d 를 업데이트 합니다. 그 규칙은 아래와 같습니다.

    d[i,j] = min(
                 d[i-1,j] + deletion cost,
                 d[i,j-1] + insertion cost,
                 d[i-1,j-1] + substitution cost
                )

아래 그림은 deletion 이 일어나는 경우입니다. '데이터'의 마지막 글자, '터'를 지우면 '데이'가 되는 겨우입니다.

![]({{ "/assets/figures/string_distance_dp_deletion.png" | absolute_url }}){: width="80%" height="80%"}

아래 그림은 insertion 이 일어나는 경우입니다. '데이'에 '타'를 추가하여 '데이타'가 되는 경우입니다.

![]({{ "/assets/figures/string_distance_dp_insertion.png" | absolute_url }}){: width="80%" height="80%"}

아래 그림은 substitution 이 일어나는 경우입니다. '데이터'에서 '데이타'로 마지막 글자가 변환되는 경우입니다.

![]({{ "/assets/figures/string_distance_dp_substitution.png" | absolute_url }}){: width="80%" height="80%"}

위 세 경우는 위의 식으로 표현 가능하며, 윗 줄의 왼쪽 칸부터 두 개의 for loop 을 돌면서 거리 행렬 d 의 모든 값을 계산합니다. 최종 거리 값은 d[$$len(s_1)$$-1, $$len(s_2)$$-1] 입니다. 

## Levenshtein distance 구현하기

### Base version

Levenshtein distance 는 [Wikipedia][levenshtein_wikipedia] 에 기본 코드가 구현되어 있습니다. 이를 바탕으로 Python 함수로 구현을 하였습니다.

current row 는 $$s_2$$ 길이보다 1 깁니다. 이는 $$s_2$$ 가 empty string 일 때를 가정한 비용을 추가하기 위함입니다. 코드의 current_row = [i + 1] 부분은 첫번째 행에 지금까지의 $$s_1$$ 의 길이만큼 deletion 이 일어났을 때의 비용을 의미합니다.

위 설명처럼 각각 insertion, deletion, substitution 이 일어났을 때의 비용을 계산한 뒤, 그 값의 최소값을 [i,j] 의 비용으로 current_row 에 추가합니다.

Distance 계산에는 필요하지 않지만, 이해를 위해 위 그림의 거리 행렬을 출력하기 위해 debug 라는 변수를 추가합니다. debug = True 일 때, initial 비용인 첫번째 행을 제외한 나머지 행, current_row[1:] 를 출력합니다.

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

위 코드에 대하여 아래와 같은 두 단어의 levenshtein distance 를 계산합니다.

{% highlight python %}
s1 = '꿈을꾸는아이'
s2 = '아이오아이'
levenshtein(s1, s2, debug=True)
{% endhighlight %}

debug = True 에 따라 행렬이 표시됩니다. d[0,0] = 1 인 이유는 첫글자가 각각 '꿈'과 '아'로 다르기 때문에 substitution 비용이 발생하기 때문입니다. 그리고 0 번째 row 의 다른 값은 $$s_1$$ = '꿈' 이 '아'로 바뀐 뒤, 한글자씩 insertion 이 되기 때문에 [1, 2, 3, 4, 5] 로 비용이 증가합니다. 그 이후는 위의 Levenshtein 식과 같이 계산됩니다.

최종적으로 [꿈, 을, 꾸]가 [아, 이, 오]로 substitution 이 된 뒤, '는'이 deletion 되어 4 의 길이가 계산됩니다.

    [1*, 2, 3, 4, 5]
    [2, 2*, 3, 4, 5]
    [3, 3, 3*, 4, 5]
    [4, 4, 4, 4*, 5]
    [4, 5, 5, 4*, 5]
    [5, 4, 5, 5, 4*]

    4

만약 $$s_2$$ 가 $$s_1$$ 보다 길다면, $$s_1, s_2$$ 가 서로 뒤바뀌어 계산됩니다. 이번에는 두 단어의 첫 글자가 같기 때문에 d[0,0] = 0 입니다.

{% highlight python %}
s1 = '아이돌'
s2 = '아이오아이'
levenshtein(s1, s2, debug=True)
{% endhighlight %}

    [0*, 1, 2]
    [1, 0*, 1]
    [2, 1*, 1]
    [3, 2, 2*]
    [4, 3, 3*]

    3

띄어쓰기가 있는 string 에 대하여 Levenshtein 을 계산합니다. '꿈을' 이라는 어절이 같지만, character 기준으로 Levenshtein 을 적용하기 때문에 두 string 의 거리는 7 입니다.

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

Levenshtein 의 단위를 character 가 아닌, 띄어쓰기 기준으로 나뉘어지는 어절로 정의할 수도 있습니다. 위에서 구현한 코드는 $$s_1, s_2$$ 에 대하여 list 의 각 element 를 순서대로 확인합니다. Python 의 str 은 list of characters 이기 때문에 enumerate(str) 은 한 글자씩을 yield 합니다. $$s_1, s_2$$ 에 대하여 str.split() 을 함으로써 list of str 이 되도록 입력하면 어절 단위로 거리가 계산됩니다.

{% highlight python %}
# 어절 단위
s1 = '꿈을 꾸는 아이'
s2 = '아이는 꿈을 꿔요'
levenshtein(s1.split(), s2.split(), debug=True)
{% endhighlight %}

그 결과 어절 단위에서 세 번의 수정으로 두 string 이 변환됩니다.

    [1, 1, 2]
    [2, 2, 2]
    [3, 3, 3]
    
    3

### User define cost + Levenshtein

Insertion, deletion, substitution 의 비용을 글자마다 다르게 적용할 수도 있습니다. 우리는 substitution 비용을 글자마다 다르게 적용할 수 있도록 위의 구현체를 변형합니다. cost 라는 변수를 추가합니다. cost 는 $$c_1$$ 이 $$c_2$$ 로 변하는 비용을 dict 형식으로 저장한 변수입니다.

Python 에서 dict 는 mutable 한 변수이기 때문에 안전한 구현을 위하여 cost=None 으로 초기화 합니다. 사용자가 특별히 substitution 비용을 정의하지 않는다면 levenshtein 함수 내부에서 cost 를 empty dict 로 재정의 합니다.

우리는 substitution_cost 라는 helper 함수를 만들었습니다. $$c_1$$ 이 $$c_2$$ 로 변하는 비용입니다. 두 유닛 (반드시 글자이지 않아도 되니,이후로는 유닛이라고 명합니다) 이 같으면 비용을 0 으로, 그렇지 않다면 cost dict 안에 비용이 있는지 확인합니다. 특별히 정의되지 않았다면 기본 substition cost 를 1 로 정의합니다.

{% highlight python %}
def levenshtein(s1, s2, cost=None, debug=False):
    if len(s1) < len(s2):
        return levenshtein(s2, s1, debug=debug)

    if len(s2) == 0:
        return len(s1)

    if cost is None:
        cost = {}

    # changed
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

변형된 levenshtein 함수를 아래의 예시에 대하여 적용합니다.

{% highlight python %}
s1 = '아이쿠야'
s2 = '아이쿵야'
levenshtein(s1, s2, debug=True)
{% endhighlight %}

특별한 substitution cost 가 적용되지 않았기 때문에 '쿠'와 '쿵'에 대하여 1 의 비용이 들었습니다.

    [0, 1, 2, 3]
    [1, 0, 1, 2]
    [2, 1, 1, 2]
    [3, 2, 2, 1]

    1

이번에는 '쿠 $$\rightarrow$$ 쿵' 으로의 substitution cost 를 0.1 로 설정합니다.

{% highlight python %}
cost = {('쿠', '쿵'):0.1}
s1 = '아이쿠야'
s2 = '아이쿵야'
levenshtein(s1, s2, cost, debug=True)
{% endhighlight %}

그 결과 위의 예시의 Levenshtein distance 는 0.1 이 되었습니다.

    [0, 1, 2, 3]
    [1, 0, 1, 2]
    [2, 1, 0.1, 1.1]
    [3, 2, 1.1, 0.1]

    0.1

어떤 도메인에서는 특정한 글자가 서로 자주 교차되기도 합니다. 예를 들어, '서비스'라는 단어는 '써비스'라고 자주 이용됩니다. 초성이 된소리가 된다거나, 종성에 'ㅇ' 받침이 추가되는 경우가 잦다면 (특히 대화데이터에서 그렇습니다), 이 때의 edit distance 의 비용을 글자에 따라 다르게 부과할 수 있습니다.

그리고 이와 같은 변형은 insertion, deletion cost 에 대해서도 동일하게 적용할 수 있습니다.

### 한글의 초/중/종성 분리

위의 예시에서 '쿠'와 '쿵'을 글자 단위로 비교하였고, 초, 중성이 같지만 종성이 다르기 때문에 비용을 0.1 로 설정하였습니다. 그 외에도 한글을 초/중/종성으로 분리한 뒤, 각각에 대한 Levenshtein distance 를 계산할 수도 있습니다. 우리가 원하는 것은 '쿠'와 '쿵'이 초/중/종성 중 종성만 다르니 그 거리를 1/3 으로 정의하는 것입니다.

이를 위해서 한글의 초/중/종성을 분리할 수 있어야 합니다. 이를 위한 기초 지식에 대해 알아봅니다.

컴퓨터는 각 글자에 대한 숫자가 정의되어 있습니다. 이 체계를 [encoding][encoding_wikipedia] 이라 합니다. 각 글자의 고유 아이디라 생각해도 됩니다. Python 에서 글자를 아이디로 변형하기 위해서는 ord 함수를 이용하면 됩니다. 아래는 각 글자의 ord 값입니다.

{% highlight python %}
for char in 'azAZ가힣ㄱㄴㅎㅏ':
    print('{} == {}'.format(char, ord(char)))
{% endhighlight %}

특히 '가'는 완전한 한글의 첫 글자, '힣'은 완전한 한글의 마지막 글자입니다. 한국어의 완전한 글자에 대한 고유 아이디의 범위는 44032 ~ 55203 입니다. 자음과 모음도 특정 범위 안에 위치합니다.

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

반대로 숫자로 표현된 글자의 고유 아이디를 글자로 변형하기 위해서는 chr 함수를 이용할 수 있습니다.

{% highlight python %}
for idx in [97, 122, 65, 90, 44032, 55203]:
    print('{} == {}'.format(idx, chr(idx)))
{% endhighlight %}

위의 결과와 반대의 결과가 출력됩니다.

    97 == a
    122 == z
    65 == A
    90 == Z
    44032 == 가
    55203 == 힣

그리고 완전 한글과 초/중/종성 사이에는 합성 규칙이 있습니다. 이를 이용하면 완전 한글을 초/중/종성으로 분리하거나, 역으로 결합할 수 있습니다.

초/중/종성으로 분해하기 위해서는 ord 로 글자를 숫자로 변형한 뒤, 완전 한글의 시작값, 44032 를 빼줍니다. 그리고 초성의 기본값 (588) 과 중성의 기본값 (28) 로 각각 나눠주면 그 몫이 초, 중성 list 의 index 가 됩니다. 그리고 그 나머지가 종성 list 의 index 가 됩니다.

Composition 은 그 과정을 반대로 진행하면 됩니다.

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

    # decomposition rule
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

위에서 만든 함수를 적용합니다. '감'이 ('ㄱ', 'ㅏ', 'ㅁ') 로 분해됨을 확인할 수 있습니다.

{% highlight python %}
decompose('감') # ('ㄱ', 'ㅏ', 'ㅁ')
{% endhighlight %}

반대로 ('ㄲ', 'ㅜ', 'ㅁ') 이 '꿈'으로 결합되는 것도 확인할 수 있습니다. 만약 ('a', 'ㅜ', 'ㅁ') 처럼 옳지 않은 글자를 입력하면 list.index 함수가 제대로 작동하지 않기 때문에 exception 이 발생합니다.

{% highlight python %}
compose('ㄲ', 'ㅜ', 'ㅁ') # '꿈'
{% endhighlight %}

### 초/중/종성 분리를 적용한 Levenshtein distance

위에서 만든 composition, decomposition 함수를 이용하여 초/중/종성 단위의 Levenshtein distance 인 jamo_levenshtein 함수를 구현합니다. Base code 에서 substitution_cost 를 변형합니다. 만약 두 글자가 같으면 비용을 0 으로, 그렇지 않다면 각각을 decomposition 하여 초/중/종성 단위에서의 Levenshtein distance 를 계산합니다. 거리 값의 범위가 0 ~ 3 이기 때문에, 이를 3 으로 나눠줍니다.

이제부터는 debug = True 일 때, 소숫점이 출력될테니, '%.3f'%v 을 이용하여 소수점 아래 셋째자리까지만 str 형식으로 출력합니다.

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

아래 예제에 대하여 jamo_levenshtein 함수를 적용합니다.

{% highlight python %}
s1 = '아이쿠야'
s2 = '아이쿵야'
jamo_levenshtein(s1, s2, debug=True)
{% endhighlight %}

d[2,2] = 0.333 입니다. '쿠'에서 '쿵'으로 변하는 비용이 0.333 이었습니다.

    ['0.000', '1.000', '2.000', '3.000']
    ['1.000', '0.000', '1.000', '2.000']
    ['2.000', '1.000', '0.333', '1.333']
    ['3.000', '2.000', '1.333', '0.333']

    0.3333333333333333

맨 앞글자에 전혀 다른 글자를 삽입하면 첫글자에 대하여 deletion 이 발생하기 때문에 d[0,0] = 1 이 됩니다. 그 뒤 '앍 $$\rightarrow$$ 아'로의 substitution cost, 0.333 이 더해집니다. 이후에 '쿠 $$\rightarrow$$ 쿵'이 추가되어 최종적으로 1.666 의 거리가 계산됩니다.

{% highlight python %}
s1 = '훍앜이쿠야'
s2 = '아이쿵야'
jamo_levenshtein(s1, s2, debug=True)
{% endhighlight %}

    ['1.000'*, '2.000', '2.667', '3.667']
    ['1.333'*, '1.667', '2.667', '3.333']
    ['2.333', '1.333'*, '2.333', '3.000']
    ['3.333', '2.333', '1.667'*, '2.667']
    ['4.333', '3.333', '2.667', '1.667'*]

    1.6666666666666665

이는 다르게도 구현할 수 있습니다. 먼저 각 글자에 대하여 초/중/종성을 모두 분해하여 concatenation 을 합니다. 그 결과를 각각 $$s_1\_, s_2\_$$ 라 정의합니다. 그 다음 이에 대한 Levenshtein distance 를 계산합니다. 이때도 한 음절에 대한 거리의 범위가 0 ~ 3 이기 때문에 거리 값을 3으로 나눠줍니다.

'아이쿠야 $$\rightarrow$$ 아이쿵야' 로의 거리가 0.333 으로 계산됩니다.

{% highlight python %}
s1 = '아이쿠야'
s2 = '아이쿵야'

s1_ = ''.join([comp for c in s1 for comp in decompose(c)])
s2_ = ''.join([comp for c in s2 for comp in decompose(c)])

print(s1_) # ㅇㅏ ㅇㅣ ㅋㅜ ㅇㅑ 
print(s2_) # ㅇㅏ ㅇㅣ ㅋㅜㅇㅇㅑ 
print(levenshtein(s1_, s2_)/3) # 0.3333333333333333
{% endhighlight %}

'아이쿵야 $$\rightarrow$$ 훍앜이쿠야' 로의 거리도 위와 같이 1.666 으로 계산됩니다.

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

앞서 언급한 compose, decompose 함수 및 levenshtein, 초/중/종성 분리를 적용한 jamo_levenshtein 를 soynlp 에 구현해 두었습니다. 빠르게 해당 함수를 이용하고 싶으시면 `pip install soynlp` 로 soynlp 를 설치하신 뒤 이를 이용할 수 있습니다.

각각의 경우에 맞게 함수를 변형하고 싶다면, 위의 예시를 base 로 이용할 수도 있습니다.

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

[string_distance_wikipedia]: https://en.wikipedia.org/wiki/String_metric
[levenshtein_wikipedia]: https://en.wikipedia.org/wiki/Levenshtein_distance
[encoding_wikipedia]: https://en.wikipedia.org/wiki/Character_encoding
[word2vec]: {{ site.baseurl }}{% link _posts/2018-03-26-word_doc_embedding.md %}
[next]: {{ site.baseurl }}{% link _posts/2018-08-21-ford_for_pos.md %}