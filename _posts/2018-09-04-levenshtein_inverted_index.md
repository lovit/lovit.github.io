---
title: Inverted index 를 이용한 빠른 Levenshtein (edit) distance 탐색
date: 2018-09-04 05:00:00
categories:
- nlp
tags:
- string distance
---

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

그 외의 insertion 과 substitution 도 위와 동일한 형태로 계산됩니다. Levenshtein distance 의 구현 및 한글 텍스트의 적용에 관련된 내용은 [이전의 블로그][levenshtein]를 참고하시기 바랍니다.

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

- Condition 1. $$ \vert len(q) - len(s) \vert \le d$$
- condition 2. $$len(set(q)) - len(set(s)) \le d$$

그리고 위 조건을 만족하는 $$s$$ 를 찾기 위하여 inverted index 를 이용할 수 있습니다.

## Inverted index

Inverted index 는 information retrieval 분야에서 제안되었습니다. 많은 검색 엔진의 기본 indexer 로 이용되는 방법입니다.

Bag-of-words model 로 문서를 표현할 때, 하나의 문서에 대하여 그 문서에 등장한 단어와 빈도수로 문서를 표현할 수 있습니다.

    BOW = {
     $$d_0$$: [($$t_1$$, $$w_{0,1}$$), ($$t_3$$, $$w_{0,3}$$), $$\dots$$],
     $$d_1$$: [($$t_2$$, $$w_{1,2}$$), ($$t_3$$, $$w_{1,3}$$), $$\dots$$ ],
     $$\dots$$
    }

위 그림은 $$t1, t2, t3$$ 로 이뤄진 두 개의 문서 $$d0, d1$$ 를 표현한 것입니다. $$BOW[d_0][t_3] = w_{0,3}$$ 입니다. 이는 문서 기준으로 단어가 indexing 이 되어 있는 형태입니다.

검색 엔진에 query 가 입력되면 query 에 포함된 단어들을 포함하는 문서들을 query 의 답변 문서 후보로 선택합니다. 즉 우리가 알고 싶은 것은 어떤 문서들이 $$t_1$$ 을 포함하고 있는지 입니다. 위의 BOW 처럼 indexer 를 만들면 모든 문서들을 뒤져가며 query 에 포함된 단어가 포함되어 있는지 확인해야 합니다. 빠른 검색을 위해서는 문서 기준이 아닌, 단어 기준으로 문서를 indexing 할 필요가 있습니다. 문서 - 단어 기준이 아닌, 단어 - 문서 기준으로 인덱싱을 한다는 의미로 inverted index 라 합니다. 위의 예시는 다음과 같은 indexer 를 지닙니다.

    Inverted_index = {
      $$t_1$$: [($$d_0$$, $$w_{0,1}$$), $$\dots$$],
      $$t_2$$: [($$d_1$$, $$w_{1,2}$$), $$\dots$$],
      $$t_3$$: [($$d_0$$, $$w_{0,3}$$), ($$d_0$$, $$w_{0,3}$$), $$\dots$$],
      $$\dots$$,
    }

우리는 $$Inverted_index[t_1]$$ 를 통해서 단어 $$t_1$$ 이 포함된 문서들을 쉽게 가져올 수 있습니다.

## Implementation




[leven_inv_github]: https://github.com/lovit/inverted_index_for_hangle_editdistance
[levenshtein]: {{ site.baseurl }}{% link _posts/2018-08-28-levenshtein_hangle.md %}