---
title: Inverted index 를 이용한 빠른 Levenshtein (edit) distance 탐색
date: 2018-09-04 20:00:00
categories:
- nlp
tags:
- string distance
---

Levenshtein distance 는 string 간 형태적 유사도를 정의하는 척도입니다. 만약 우리가 단어 사전을 가지고 있고, 사전에 등록되지 않은 단어가 발생한다면 Levenshtein distance 가 가장 가까운 단어로 치환함으로써 오탈자를 교정할 수 있습니다. 그러나 Levenshtein distance 는 계산 비용이 비쌉니다. 이 때 간단한 inverted index 를 이용하여 비슷할 가능성이 있는 단어 후보만을 추린 뒤 몇 번의 Levenshtein distance 를 계산함으로써 효율적으로 오탈자를 교정할 수 있습니다. 이번 포스트에서는 inverted index 를 이용하는 효율적인 Levenshtein distance 기반 오탈자 교정기를 만들어 봅니다.

## Levenshtein distance

String 간의 형태적 유사도를 정의하는 척도를 string distance 라 합니다. Edit distance 라는 별명을 지닌 Levenshtein distance 는 대표적인 string distance 입니다.

Levenshtein distance 는 한 string $$s_1$$ 을 $$s_2$$ 로 변환하는 최소 횟수를 두 string 간의 거리로 정의합니다. $$s_1$$ = '꿈을꾸는아이' 에서 $$s_2$$ = '아이오아이' 로 바뀌기 위해서는 (꿈을꾸 -> 아이오) 로 바뀌고, 네번째 글자 '는' 이 제거되면 됩니다. Levenshtein distance 에서는 이처럼 string 을 변화하기 위한 edit 방법을 세 가지로 분류합니다.

1. delete: '점심**을**먹자 $$\rightarrow$$ 점심먹자' 로 바꾸기 위해서는 **을** 을 삭제해야 합니다.
2. insert: '점심먹자 $$\rightarrow$$ 점심**을**먹자' 로 바꾸기 위해서는 반대로 **을** 을 삽입해야 합니다.
3. substitution: '점심먹**자** $$\rightarrow$$ 점심먹**장**' 로 바꾸기 위해서는 **자**를 **장** 으로 치환해야 합니다.

이를 위해 동적 프로그래밍 (dynamic programming) 이 이용됩니다. d[0,0] 은 $$s_1, s_2$$ 의 첫 글자가 같으면 0, 아니면 1로 초기화 합니다. 글자가 다르면 substitution cost 가 발생한다는 의미입니다. 그리고 그 외의 d[0,j]에 대해서는 d[0,j] = d[0,j-1] + 1 의 비용으로 초기화 합니다. 한글자씩 insertion 이 일어났다는 의미입니다. 이후에는 좌측, 상단, 좌상단의 값을 이용하여 거리 행렬 d 를 업데이트 합니다. 그 규칙은 아래와 같습니다.

    d[i,j] = min(
                 d[i-1,j] + deletion cost,
                 d[i,j-1] + insertion cost,
                 d[i-1,j-1] + substitution cost
                )

아래 그림은 '데이터마이닝'과 '데이타마닝' 과의 Levenshtein distance 를 계산하는 경우의 예시입니다. 세 가지 수정 중 deletion 이 일어나는 경우입니다. '데이터'의 마지막 글자, '터'를 지우면 '데이'가 되는 겨우입니다.

![]({{ "/assets/figures/string_distance_dp_deletion.png" | absolute_url }}){: width="80%" height="80%"}

그 외의 insertion 과 substitution 도 위와 동일한 형태로 계산됩니다. Levenshtein distance 의 구현 및 한글 텍스트의 적용에 관련된 내용은 [이전 포스트][levenshtein]를 참고하시기 바랍니다.

### Computation cost issue

Levenshtein distance 를 이용하여 오탈자 교정기를 만들 수 있습니다. 정자에 대한 사전, reference data 을 미리 구축합니다. 만약 우리가 알지 못하는 (사전에 등록되지 않은) 단어가 나타날 경우, 한 단어에 대하여 정자 단어 사전에 등록된 단어들 중 거리가 가장 가까운 단어로 해당 단어를 치환할 수 있습니다.

그러나 우리는 한 단어에 대해 사전에 등록된 모든 단어와의 거리를 계산해야만 합니다. Levenshtein distance 의 계산 비용은 작지 않습니다. String slicing 과 equal 함수를 실행해야 합니다. 우리가 이용하는 reference data 의 크기가 10 단어 정도라면 계산 비용의 문제를 무시할 수도 있지만, 10 만 단어라면 비용의 문제도 고민해야 합니다.

효율성 관점에서 더 중요한 점은, 오탈자의 거리 범위입니다. 단어 기준에서 오탈자는 주로 1 ~ 2 글자 수준이기 마련입니다. '서비스'를 '써비스'로 적는다면 오탈자라고 고려할만 하지만, '서울시'로 적었다면 '서비스'가 정자일 가능성은 적습니다. 그렇다면 두 string 의 형태가 어느 정도 비슷한 단어에 대해서만 string distance 를 계산해도 될 것입니다.

이번 포스트에서 다룰 이야기는 주어진 단어 $$q$$ 에 대하여 Levenshtein distance 가 $$d$$ 이하일 가능성이 있는 reference words 에 대해서만 거리 계산을 하는 효율적인 Levenshtein distance indexer 를 만드는 것입니다.

### Concept of proposed model

우리의 목표는 한 단어 $$q$$ 의 거리를 $$l_q$$ 라 할 때, 임의의 단어 $$s$$ 와의 Levenshtein distnace 값이 $$d$$ 이하인 단어를 최소한의 비용 (최소한의 Levenshtein distance 계산) 으로 찾는 것입니다.

Levenshtein distance 의 특징을 이용하면 간단한 조건식을 만들 수 있습니다. 첫째로, $$q$$ 와 $$s$$ 의 길이 차이가 $$d$$ 보다 클 경우, Levenshtein distance 또한 반드시 $$d$$ 보다 크게 됩니다. 최소한 길이의 차이만큼 insertion 이나 deletion 이 일어나야 하기 때문입니다.

또한 두 단어 $$q, s$$ 의 길이가 같다고 할 때, $$q$$ 에는 포함되어 있으나, $$s$$ 에 포함되지 않은 글자의 개수가 $$d$$ 보다 크다면 적어도 $$d$$ 번의 substitution 이 일어나야 함을 의미합니다.

위의 두 조건을 정리하면 아래와 같습니다.

1. $$ \vert len(q) - len(s) \vert \le d$$
2. $$len(set(q) set(s)) \le d$$

그리고 위 조건을 만족하는 $$s$$ 를 찾기 위하여 inverted index 를 이용할 수 있습니다.

## Inverted index

Inverted index 는 information retrieval 분야에서 제안되었습니다. 많은 검색 엔진의 기본 indexer 로 이용되는 방법입니다.

Bag-of-words model 로 문서를 표현할 때, 하나의 문서에 대하여 그 문서에 등장한 단어와 빈도수로 문서를 표현할 수 있습니다.

    BOW = {
      d0: [(t1, w01), (t3, w03), ...],
      d1: [(t2, w12), (t3, w13), ... ],
      ...
    }

위 그림은 $$t1, t2, t3$$ 로 이뤄진 두 개의 문서 $$d0, d1$$ 를 표현한 것입니다. $$BOW[d_0][t_3] = w_{0,3}$$ 입니다. 이는 문서 기준으로 단어가 indexing 이 되어 있는 형태입니다.

검색 엔진에 query 가 입력되면 query 에 포함된 단어들을 포함하는 문서들을 query 의 답변 문서 후보로 선택합니다. 즉 우리가 알고 싶은 것은 어떤 문서들이 $$t_1$$ 을 포함하고 있는지 입니다. 위의 BOW 처럼 indexer 를 만들면 모든 문서들을 뒤져가며 query 에 포함된 단어가 포함되어 있는지 확인해야 합니다. 빠른 검색을 위해서는 문서 기준이 아닌, 단어 기준으로 문서를 indexing 할 필요가 있습니다. 문서 - 단어 기준이 아닌, 단어 - 문서 기준으로 인덱싱을 한다는 의미로 inverted index 라 합니다. 위의 예시는 다음과 같은 indexer 를 지닙니다.

    Invertedindex = {
      t1: [(d0, w01), ...],
      t2: [(d1, w12), ...],
      t3: [(d0, w03), (d0, w03), ...],
      ...,
    }

우리는 $$Inverted\_index[t_1]$$ 를 통해서 단어 $$t_1$$ 이 포함된 문서들을 쉽게 가져올 수 있습니다.

물론 대량의 문서 검색을 위한 검색 엔진에는 더 많은 기능들이 들어 있습니다만, 빠른 오탈자 교정기를 위한 inverted index 는 이정도면 충분합니다.

## Implementation

우리는 [앞선 포스트][levenshtein]에서 구현한 levenshtein 함수를 이용합니다.

{% highlight python %}
def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

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
        previous_row = current_row
    return previous_row[-1]
{% endhighlight %}

Inverted index 는 dict 와 set 을 이용하여 손쉽게 만들 수 있습니다. self._index 는 글자 -> 단어의 map 입니다. for c in word 를 이용하여 각 단어에 어떤 글자가 포함되어 있는지를 저장합니다.

이후 defaultdict 에 엉뚱한 값이 key 로 포함되는 것을 방지하기 위하여 dict(self._index) 를 걸어줍니다.

{% highlight python %}
from collections import defaultdict

class LevenshteinIndex:
    def __init__(self):
        self._index = {} # character to words

    def indexing(self, words):
        self._index = defaultdict(lambda: set())

        for word in words:
            for c in word:
                self._index[c].add(word)
        self._index = dict(self._index)
{% endhighlight %}

아래처럼 8 개의 단어를 indexing 합니다.

{% highlight python %}
words = '아이고 어이고 아이고야 아이고야야야야 어이구야 지화자 징화자 쟝화장'.split()

index = LevenshteinIndex()
index.indexing(words)
index._index
{% endhighlight %}

self._index 에는 다음처럼 인덱싱이 되어 있습니다.

    {'고': {'아이고', '아이고야', '아이고야야야야', '어이고'},
     '구': {'어이구야'},
     '아': {'아이고', '아이고야', '아이고야야야야'},
     '야': {'아이고야', '아이고야야야야', '어이구야'},
     '어': {'어이고', '어이구야'},
     '이': {'아이고', '아이고야', '아이고야야야야', '어이고', '어이구야'},
     '자': {'지화자', '징화자'},
     '장': {'쟝화장'},
     '쟝': {'쟝화장'},
     '지': {'지화자'},
     '징': {'징화자'},
     '화': {'쟝화장', '지화자', '징화자'}}

단어 query 와 Levenshtein distance 가 max_distance 이하인 단어를 찾는 부분을 구현합니다.

{% highlight python %}
class LevenshteinIndex:
    def levenshtein_search(self, query, max_distance=1):
        similars = defaultdict(int)
        (n, nc) = (len(query), len(set(query)))
        for c in set(query):
            for item in self._index.get(c, {}):
                similars[item] += 1
            
        similars = {word for word,f in similars.items()
                    if (abs(n-len(word)) <= max_distance) and (abs(nc - f) <= max_distance)}

        dist = {}
        for word in similars:
            dist[word] = levenshtein(query, word)

        filtered_words = filter(lambda x:x[1] <= max_distance, dist.items())
        sorted_words = sorted(filtered_words, key=lambda x:x[1])
        return sorted_words
{% endhighlight %}

n 과 nc 는 각각 query 의 길이와 query 의 unique 단어 개수 입니다. 아래 코드를 통하여 query 에 있는 글자와 같은 글자를 가진 단어 후보들을 찾을 수 있습니다. similars 에는 reference words 와 query 에 공통으로 포함된 글자의 개수가 계산됩니다. query 에 같은 글자가 여러 번 나올 수 있기 때문에 set(query) 를 통하여 unique characters 에 대해서만 index 를 살펴보도록 합니다.

    for c in set(query):
        for item in self._index.get(c, {}):
            similars[item] += 1

아래 코드를 통하여 길이 차이가 max_distance 보다 작게 나거나, 구성하는 글자의 종류의 개수 차이가 max_distance 보다 작은 글자만을 선택할 수 있습니다.

    similars = {c for c,f in similars.items()
                if (abs(n-len(c)) <= max_distance) and (abs(nc - f) <= max_distance)}

이 조건을 만족하는 글자에 대해서만 levenshtein distance 를 계산합니다.

초/중/종성을 분리하여 jamo levenshtein distance 를 계산하려면, 초/중/종성에 대하여 따로 따로 inverted index 를 만들면 됩니다.

{% highlight python %}
class LevenshteinIndex:

    def indexing(self, words):
        self._index = defaultdict(lambda: set())
        self._cho_index = defaultdict(lambda: set())
        self._jung_index = defaultdict(lambda: set())
        self._jong_index = defaultdict(lambda: set())
        
        for word in word_counter:
            # Indexing for levenshtein
            for c in word:
                self._index[c].add(word)
            # Indexing for jamo_levenshtein
            for c in word:
                if not character_is_korean(c):
                    continue
                cho, jung, jong = decompose(c)
                self._cho_index[cho].add(word)
                self._jung_index[jung].add(word)
                self._jong_index[jong].add(word)
        ...
{% endhighlight %}

검색에 관련된 부분도 초/중/종성에 대하여 위 조건문을 적용하면 됩니다.

## Performance

이번에는 더 큰 크기의 단어 사전을 이용하여 테스트 하였습니다. [경제 용어 사전][dict_github]에 포함된 132,864 개의 단어에서 max_distance 이하의 Levenshtein distance 를 지니는 단어를 검색하는 예시입니다.

Verbose 및 초/중/종성 단위의 Levenshtein distance 를 계산하는 구현체는 [github][leven_inv_github]에 구현해 두었습니다.

{% highlight python %}
print(len(nouns)) # 132,864
financial_word_indexer = LevenshteinIndex(nouns)

financial_word_indexer.verbose = True
financial_word_indexer.levenshtein_search('분식회계', max_distance=1)
{% endhighlight %}

Verbose mode 를 이용하면 비슷한 단어의 후보의 개수 변화가 출력됩니다. 같은 글자를 1 개 이상 지니는 10,137 개의 후보를 먼저 선택하였습니다. 그 뒤, 두 개의 조건을 만족하는 후보를 추리면 총 7 개의 단어가 후보로 선정됩니다. 이 단어에 대해서만 Levenshtein distance 를 계산합니다. 그 결과 0.00626 초 만에 Levenshtein distance 가 1 이하인 단어들이 검색됩니다.

    query=분식회계, candidates=10137 -> 7, time=0.00626 sec.
    [('분식회계', 0), ('분식회', 1), ('분식회계설', 1), ('분석회계', 1)]

같은 예시에 대하여 132,864 개의 단어와 Levenshtein distance 를 모두 계산하였습니다.

{% highlight python %}
import time
query = '분식회계'

search_time = time.time()
distance = {word:levenshtein(word, query) for word in nouns}
search_time = time.time() - search_time
print('search time = {} sec'.format('%.2f'%search_time))

similars = sorted(filter(lambda x:x[1] <= 1, distance.items()), key=lambda x:x[1])
print(similars)
{% endhighlight %}

같은 결과를 찾는데 이번에는 2.49 초가 필요합니다.

    search time = 2.49 sec
    [('분식회계', 0), ('분식회', 1), ('분식회계설', 1), ('분석회계', 1)]

Reference data 의 크기가 커지더라도 같은 글자를 지닌 글자의 숫자는 크게 증가하지 않습니다. Inverted index 를 이용하여 최소한 비슷할 수 있는 단어들만을 후보로 추린 뒤, 소수의 후보에 대해서만 계산 비용이 비싼 Levenshtein distance 를 계산함으로써, 효율적인 오탈자 교정을 할 수 있습니다.


[leven_inv_github]: https://github.com/lovit/inverted_index_for_hangle_editdistance
[dict_github]: https://github.com/lovit/sharing_korean_dictionary
[levenshtein]: {{ site.baseurl }}{% link _posts/2018-08-28-levenshtein_hangle.md %}