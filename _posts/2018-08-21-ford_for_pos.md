---
title: Ford algorithm 을 이용한 품사 판별 (Part-of-speech tagging), 그리고 Hidden Markov Model 과의 관계
date: 2017-08-21 09:00:00
categories:
- nlp
- graph
tags:
- shortest path
- tokenizer
---

Hidden Markov Model (HMM) 은 품사 판별을 위하여 이용되기도 하였습니다. 특히 앞의 한 단계의 state 정보만을 이용하는 모델을 first-order HMM 이라 합니다. 그런데 이 first-order HMM 을 이용하는 품사 판별기의 품사열은 최단 경로 문제 기법으로도 구할 수 있습니다. 이번 포스트에서는 앞선 포스트에서 다뤘던 Ford algorithm 을 이용하여 품사를 판별하는 방법에 대하여 알아봅니다. 

## Review of Ford algorithm for shortest path

얼마 전, Hidden Markov Model (HMM) 을 기반으로 하는 품사 판별기의 코드를 공부하던 중, "findPath" 라는 이름의 함수를 보았습니다. HMM 의 decoding 과정을 설명할 때 주로 Dynamic programming 의 관점으로 설명을 하는데, 그 구현체는 최단 경로를 찾는 방법으로 decoding 과정을 설명하고 있었습니다. 한 번도 생각해보지 않았는데, 생각해보니 first-order 를 따르는 HMM 이라면 최단 경로 문제와 동치가 됩니다. 이 이야기를 한 번 정리해야 겠다는 생각을 하였습니다.

이번 포스트에서는 HMM 에 대한 설명은 하지 않습니다. 조만간에 HMM 관련 포스트를 작성하고, 이 부분을 링크로 대체하도록 하겠습니다. 마디 (node) 와 호 (edge) 로 표현된 그래프에서 두 마디를 연결할 수 있는 경로 (path) 는 다양합니다. 그 중 거리가 가장 짧은 경로를 찾는 문제를 최단 경로 문제, shortest path 라 합니다.

Ford algorithm 은 매우 간단합니다. 목적지까지 가는 길에 조금 더 가까운 경로를 발견한다면, 내가 알고 있는 최단 경로를 그 경로로 계속하여 대체합니다. 더 이상 대체할 경로가 없다면 현재 알고 있는 경로가 최단 경로가 됩니다.

이를 위하여 Ford algorithm 의 초기화를 합니다. 출발지와 목적지가 주어지면, 출발지의 거리를 0 으로 설정합니다. 그리고 그 외의 모든 마디의 거리를 무한대로 설정합니다. 아직 어떤 마디까지도 가보지 않았기 때문에 실제로 얼마의 거리가 걸리는지를 알지 못합니다. 이제 모든 마디에 대해서 아래와 같은 조건이 만족하는 마디가 있는지 확인합니다.

if $$C[u] + w(u,v) < C[v]$$, then update $$C[v] \leftarrow C[u] + w(u,v)$$

위 조건을 만족하는 경우가 없을 때 까지 반복하여 각 마디까지의 비용을 업데이트합니다.

[이전 포스트][prev]에서 지하철 최단 경로 탐색을 위해 Ford algorithm 을 만드는 과정을 설명하였습니다. Ford algorithm 에 대한 자세한 설명은 위 포스트를 참고하세요. 이번 포스트는 Ford algorithm 에 대해서 알고 있다는 가정하에, 텍스트 처리에 대한 부분만 설명합니다.

## 예문과 사전

우리가 살펴볼 예문은 아래와 같습니다. 예문은 사실 비문이지만, 일단 쉬운 예시를 이용하기 위해 아래의 예문을 이용합니다.

{% highlight python %}
sentence = '청하는 아이오아이의 출신입니다'
{% endhighlight %}

그리고 우리가 알고 있는 단어 사전은 아래와 같다고 가정합니다.

{% highlight python %}
pos2words = {
    'Noun': set('아이 아이오 아이오아이 청하 출신 청'.split()),
    'Josa': set('은 는 이 가 의 를 을'.split()),
    'Verb': set('청하 이 있 하 했 입'.split()),
    'Eomi': set('다 었다 는 니다'.split())
}
{% endhighlight %}

위 사전을 이용하여 위 문장을 다음과 같은 그래프로 표현할 수 있습니다. 그리고 올바른 품사 판별의 결과는 그림 위의 진한 선으로 연결된 경로로 표현할 수 있습니다. 아래 그림에서 BOS, EOS 는 Begin/End of Sentence 로, 문장의 시작과 끝을 표현하는 일반적인 표현입니다.

![]({{ "/assets/figures/shortestpath_chungha.png" | absolute_url }})

사실 품사 판별, 형태소 분석을 이야기하려면 한국어 용언의 [활용][conjugate]과 [복원][lemmatizer]에 대해서도 다뤄야 합니다. 하지만 이는 최단 경로를 이용한 품사 판별 원리를 설명하는데 반드시 필요한 부분이 아니므로, string slicing 만으로 단어 후보를 만들 수 있다고 가정합니다. 용언의 활용과 복원에 대해 궁금하신 분은 위의 링크를 참고하세요.

## Lookup 을 위한 Dictionary 구현

우리는 어떤 단어가 주어졌을 때, 해당 단어의 품사를 확인하는 Dictionary 라는 class 를 만들것입니다. 이는 품사 판별 과정에서 단어/품사 후보를 만드는 기능을 합니다.

한 가지, 사전에 등록된 단어의 최대 길이를 확인하는 set_max_len 이라는 함수도 구현합니다.

{% highlight python %}
class Dictionary:
    def __init__(self, pos2words=None):
        self.pos2words = pos2words if pos2words else {}
        self.max_len = self._set_max_len()

    def _set_max_len(self):
        if not self.pos2words: return 0
        return max((len(word) for words in self.pos2words.values() for word in words))

    def get_pos(self, word):
        return [pos for pos, words in self.pos2words.items() if word in words]

dictionary = Dictionary(pos2words)
{% endhighlight %}

구현된 Dictionary 를 이용하여 두 단어에 대한 품사를 확인해 봅니다. get_pos 함수는 list of str 을 return 합니다. 만약 단어에 해당하는 품사가 없다면 [] 와 같은 empty list 를 return 합니다.

{% highlight python %}
dictionary.get_pos('이') # ['Josa', 'Verb']
dictionary.get_pos('청하') # ['Noun', 'Verb']
{% endhighlight %}

## Lookup 을 이용한 마디 구성

{% highlight python %}
def word_lookup(eojeol, offset):
    n = len(eojeol)
    words = [[] for _ in range(n)]
    for b in range(n):
        for r in range(1, dictionary.max_len+1):
            e = b+r
            if e > n:
                continue
            sub = eojeol[b:e]
            for pos in dictionary.get_pos(sub):
                words[b].append((sub, pos, b+offset, e+offset))
    return words

word_lookup('청하는', 0)
{% endhighlight %}

    [[('청', 'Noun', 0, 1), ('청하', 'Noun', 0, 2), ('청하', 'Verb', 0, 2)],
     [('하', 'Verb', 1, 2)],
     [('는', 'Josa', 2, 3), ('는', 'Eomi', 2, 3)]]


{% highlight python %}
word_lookup('청하는', 3)
{% endhighlight %}

    [[('청', 'Noun', 3, 4), ('청하', 'Noun', 3, 5), ('청하', 'Verb', 3, 5)],
     [('하', 'Verb', 4, 5)],
     [('는', 'Josa', 5, 6), ('는', 'Eomi', 5, 6)]]

{% highlight python %}
def lookup(sentence):
    sent = []
    for eojeol in sentence.split():
        sent += word_lookup(eojeol, offset=len(sent))
    return sent

lookup(sentence)
{% endhighlight %}

    [[('청', 'Noun', 0, 1), ('청하', 'Noun', 0, 2), ('청하', 'Verb', 0, 2)],
     [('하', 'Verb', 1, 2)],
     [('는', 'Josa', 2, 3), ('는', 'Eomi', 2, 3)],
     [('아이', 'Noun', 3, 5), ('아이오', 'Noun', 3, 6), ('아이오아이', 'Noun', 3, 8)],
     [('이', 'Josa', 4, 5), ('이', 'Verb', 4, 5)],
     [],
     [('아이', 'Noun', 6, 8)],
     [('이', 'Josa', 7, 8), ('이', 'Verb', 7, 8)],
     [('의', 'Josa', 8, 9)],
     [('출신', 'Noun', 9, 11)],
     [],
     [('입', 'Verb', 11, 12)],
     [('니다', 'Eomi', 12, 14)],
     [('다', 'Eomi', 13, 14)]]

## Edges 구성하기

### 연속된 단어 간 edges 와 Unk 간 edges

{% highlight python %}
def draw_edges(sentence):
    chars = sentence.replace(' ', '')
    sent = lookup(sentence)
    n_char = len(chars)
    sent.append([('EOS', 'EOS', n_char + 1, n_char + 1)])

    nonempty_first = get_nonempty_first(sent, offset=0)
    if nonempty_first > 0:
        sent[0].append((chars[:nonempty_first], 'Unk', 0, nonempty_first))

    edges = forward_link(sent, chars)
    edges = backward_link_for_unk(edges, sent)
    edges = add_bos(edges, sent)

    edges = sorted(edges, key=lambda x:(x[0][2], x[0][3], x[1][2]))
    return edges, sent

def get_nonempty_first(sent, offset=0):
    for i in range(offset, len(sent)+1):
        if sent[i]:
            return i
    return offset    
{% endhighlight %}



{% highlight python %}
def forward_link(sent, chars):
    graph = []
    for words in sent[:-1]:
        for word in words:
            begin = word[2]
            end = word[3]
            if not sent[end]:
                next_begin = get_nonempty_first(sent, end)
                unk = (chars[end:next_begin], 'Unk', end, next_begin)
                graph.append((word, unk))
            else:
                for adjacent in sent[end]:
                    graph.append((word, adjacent))
    return graph
{% endhighlight %}



{% highlight python %}
def backward_link_for_unk(edges, sent):
    unks = {node for _, node in edges if node[1] == 'Unk'}
    for unk in unks:
        for adjacent in sent[unk[3]]:
            edges.append((unk, adjacent))
    return edges
{% endhighlight %}



{% highlight python %}
def add_bos(edges, sent):
    bos = ('BOS', 'BOS', 0, 0)
    for word in sent[0]:
        edges.append((bos, word))
    return edges
{% endhighlight %}



{% highlight python %}
edges, sent = draw_edges(sentence)
{% endhighlight %}


    [(('BOS', 'BOS', 0, 0), ('청', 'Noun', 0, 1)),
     (('BOS', 'BOS', 0, 0), ('청하', 'Noun', 0, 2)),
     (('BOS', 'BOS', 0, 0), ('청하', 'Verb', 0, 2)),
     (('청', 'Noun', 0, 1), ('하', 'Verb', 1, 2)),
     (('청하', 'Noun', 0, 2), ('는', 'Josa', 2, 3)),
     (('청하', 'Noun', 0, 2), ('는', 'Eomi', 2, 3)),
     (('청하', 'Verb', 0, 2), ('는', 'Josa', 2, 3)),
     (('청하', 'Verb', 0, 2), ('는', 'Eomi', 2, 3)),
     (('하', 'Verb', 1, 2), ('는', 'Josa', 2, 3)),
     (('하', 'Verb', 1, 2), ('는', 'Eomi', 2, 3)),
     (('는', 'Josa', 2, 3), ('아이', 'Noun', 3, 5)),
     (('는', 'Josa', 2, 3), ('아이오', 'Noun', 3, 6)),
     (('는', 'Josa', 2, 3), ('아이오아이', 'Noun', 3, 8)),
     (('는', 'Eomi', 2, 3), ('아이', 'Noun', 3, 5)),
     (('는', 'Eomi', 2, 3), ('아이오', 'Noun', 3, 6)),
     (('는', 'Eomi', 2, 3), ('아이오아이', 'Noun', 3, 8)),
     (('아이', 'Noun', 3, 5), ('오', 'Unk', 5, 6)),
     (('아이오', 'Noun', 3, 6), ('아이', 'Noun', 6, 8)),
     (('아이오아이', 'Noun', 3, 8), ('의', 'Josa', 8, 9)),
     (('이', 'Josa', 4, 5), ('오', 'Unk', 5, 6)),
     (('이', 'Verb', 4, 5), ('오', 'Unk', 5, 6)),
     (('오', 'Unk', 5, 6), ('아이', 'Noun', 6, 8)),
     (('아이', 'Noun', 6, 8), ('의', 'Josa', 8, 9)),
     (('이', 'Josa', 7, 8), ('의', 'Josa', 8, 9)),
     (('이', 'Verb', 7, 8), ('의', 'Josa', 8, 9)),
     (('의', 'Josa', 8, 9), ('출신', 'Noun', 9, 11)),
     (('출신', 'Noun', 9, 11), ('입', 'Verb', 11, 12)),
     (('입', 'Verb', 11, 12), ('니다', 'Eomi', 12, 14)),
     (('니다', 'Eomi', 12, 14), ('EOS', 'EOS', 15, 15)),
     (('다', 'Eomi', 13, 14), ('EOS', 'EOS', 15, 15))]

### Weight 부여

{% highlight python %}
transition = {
    ('Noun', 'Josa'): 0.7,
    ('Noun', 'Noun'): 0.3,
    ('Verb', 'Eomi'): 0.5,
    ('Verb', 'Noun'): 0.5,
    ('Verb', 'Josa'): -0.1,
}
generation = {
    'Noun': {
        '아이오아이': 0.5,
        '청하': 0.2,
    }
}
{% endhighlight %}


{% highlight python %}
class Weighter:
    def __init__(self, transition, generation):
        self.transition = transition
        self.generation = generation

    def cost(self, edge):
        prob = 0
        prob += self.transition.get((edge[0][1], edge[1][1]), 0)
        for node in edge:
            prob += self.generation.get(node[1], {}).get(node[0], 0)
        return -1 * prob

weighter = Weighter(transition, generation)
{% endhighlight %}


{% highlight python %}
print(edges[4]) # (('청하', 'Noun', 0, 2), ('는', 'Josa', 2, 3))
print(weighter.cost(edges[4])) # -0.8999999999999999
{% endhighlight %}


{% highlight python %}
def attach_weight(edges):
    edges = [(edge[0], edge[1], weighter.cost(edge)) for edge in edges]
    return edges

edges = attach_weight(edges)
{% endhighlight %}

    [(('BOS', 'BOS', 0, 0), ('청', 'Noun', 0, 1), 0),
     (('BOS', 'BOS', 0, 0), ('청하', 'Noun', 0, 2), -0.2),
     (('BOS', 'BOS', 0, 0), ('청하', 'Verb', 0, 2), 0),
     (('청', 'Noun', 0, 1), ('하', 'Verb', 1, 2), 0),
     (('청하', 'Noun', 0, 2), ('는', 'Josa', 2, 3), -0.8999999999999999),
     (('청하', 'Noun', 0, 2), ('는', 'Eomi', 2, 3), -0.2),
     (('청하', 'Verb', 0, 2), ('는', 'Josa', 2, 3), 0.1),
     (('청하', 'Verb', 0, 2), ('는', 'Eomi', 2, 3), -0.5),
     (('하', 'Verb', 1, 2), ('는', 'Josa', 2, 3), 0.1),
     (('하', 'Verb', 1, 2), ('는', 'Eomi', 2, 3), -0.5),
     (('는', 'Josa', 2, 3), ('아이', 'Noun', 3, 5), 0),
     (('는', 'Josa', 2, 3), ('아이오', 'Noun', 3, 6), 0),
     (('는', 'Josa', 2, 3), ('아이오아이', 'Noun', 3, 8), -0.5),
     (('는', 'Eomi', 2, 3), ('아이', 'Noun', 3, 5), 0),
     (('는', 'Eomi', 2, 3), ('아이오', 'Noun', 3, 6), 0),
     (('는', 'Eomi', 2, 3), ('아이오아이', 'Noun', 3, 8), -0.5),
     (('아이', 'Noun', 3, 5), ('오', 'Unk', 5, 6), 0),
     (('아이오', 'Noun', 3, 6), ('아이', 'Noun', 6, 8), -0.3),
     (('아이오아이', 'Noun', 3, 8), ('의', 'Josa', 8, 9), -1.2),
     (('이', 'Josa', 4, 5), ('오', 'Unk', 5, 6), 0),
     (('이', 'Verb', 4, 5), ('오', 'Unk', 5, 6), 0),
     (('오', 'Unk', 5, 6), ('아이', 'Noun', 6, 8), 0),
     (('아이', 'Noun', 6, 8), ('의', 'Josa', 8, 9), -0.7),
     (('이', 'Josa', 7, 8), ('의', 'Josa', 8, 9), 0),
     (('이', 'Verb', 7, 8), ('의', 'Josa', 8, 9), 0.1),
     (('의', 'Josa', 8, 9), ('출신', 'Noun', 9, 11), 0),
     (('출신', 'Noun', 9, 11), ('입', 'Verb', 11, 12), 0),
     (('입', 'Verb', 11, 12), ('니다', 'Eomi', 12, 14), -0.5),
     (('니다', 'Eomi', 12, 14), ('EOS', 'EOS', 15, 15), 0),
     (('다', 'Eomi', 13, 14), ('EOS', 'EOS', 15, 15), 0)]

{% highlight python %}
from collections import defaultdict
from pprint import pprint

def edges_to_dict(edges):
    g = defaultdict(lambda: {})
    for from_, to_, weight in edges:
        g[from_][to_] = weight
    return dict(g)

g = edges_to_dict(edges)
pprint(g)
{% endhighlight %}

    {
        ('BOS', 'BOS', 0, 0): {
            ('청', 'Noun', 0, 1): 0,
            ('청하', 'Noun', 0, 2): -0.2,
            ('청하', 'Verb', 0, 2): 0
         },
        ('는', 'Eomi', 2, 3): {
            ('아이', 'Noun', 3, 5): 0,
            ('아이오', 'Noun', 3, 6): 0,
            ('아이오아이', 'Noun', 3, 8): -0.5
        },
        ....
    }
                  


## Ford algorithm 을 이용한 품사 판별기 만들기

{% highlight python %}
bos = ('BOS', 'BOS', 0, 0)
eos = ('EOS', 'EOS', 15, 15)
ford(g, bos, eos, debug=True)
{% endhighlight %}


    cost[('BOS', 'BOS', 0, 0) -> ('청', 'Noun', 0, 1)] = 24.200000000000003 -> 0
    cost[('BOS', 'BOS', 0, 0) -> ('청하', 'Noun', 0, 2)] = 24.200000000000003 -> -0.2
    cost[('BOS', 'BOS', 0, 0) -> ('청하', 'Verb', 0, 2)] = 24.200000000000003 -> 0
    cost[('청', 'Noun', 0, 1) -> ('하', 'Verb', 1, 2)] = 24.200000000000003 -> 0
    cost[('청하', 'Noun', 0, 2) -> ('는', 'Josa', 2, 3)] = 24.200000000000003 -> -1.0999999999999999
    cost[('청하', 'Noun', 0, 2) -> ('는', 'Eomi', 2, 3)] = 24.200000000000003 -> -0.4
    cost[('청하', 'Verb', 0, 2) -> ('는', 'Eomi', 2, 3)] = -0.4 -> -0.5
    cost[('는', 'Josa', 2, 3) -> ('아이', 'Noun', 3, 5)] = 24.200000000000003 -> -1.0999999999999999
    cost[('는', 'Josa', 2, 3) -> ('아이오', 'Noun', 3, 6)] = 24.200000000000003 -> -1.0999999999999999
    cost[('는', 'Josa', 2, 3) -> ('아이오아이', 'Noun', 3, 8)] = 24.200000000000003 -> -1.5999999999999999
    cost[('아이', 'Noun', 3, 5) -> ('오', 'Unk', 5, 6)] = 24.200000000000003 -> -1.0999999999999999
    cost[('아이오', 'Noun', 3, 6) -> ('아이', 'Noun', 6, 8)] = 24.200000000000003 -> -1.4
    cost[('아이오아이', 'Noun', 3, 8) -> ('의', 'Josa', 8, 9)] = 24.200000000000003 -> -2.8
    cost[('의', 'Josa', 8, 9) -> ('출신', 'Noun', 9, 11)] = 24.200000000000003 -> -2.8
    cost[('출신', 'Noun', 9, 11) -> ('입', 'Verb', 11, 12)] = 24.200000000000003 -> -2.8
    cost[('입', 'Verb', 11, 12) -> ('니다', 'Eomi', 12, 14)] = 24.200000000000003 -> -3.3
    cost[('니다', 'Eomi', 12, 14) -> ('EOS', 'EOS', 15, 15)] = 24.200000000000003 -> -3.3

    'cost': -3.3

     [[('BOS', 'BOS', 0, 0),
       ('청하', 'Noun', 0, 2),
       ('는', 'Josa', 2, 3),
       ('아이오아이', 'Noun', 3, 8),
       ('의', 'Josa', 8, 9),
       ('출신', 'Noun', 9, 11),
       ('입', 'Verb', 11, 12),
       ('니다', 'Eomi', 12, 14),
       ('EOS', 'EOS', 15, 15)]]}

[prev]: {{ site.baseurl }}{% link _posts/2018-08-21-ford_for_shortestpath.md %}
[conjugate]: {{ site.baseurl }}{% link _posts/2018-06-11-conjugator.md %}
[lemmatizer]: {{ site.baseurl }}{% link _posts/2018-06-07-lemmatizer.md %}