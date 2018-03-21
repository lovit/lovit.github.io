---
title: Cluster labeling for text data
date: 2018-03-21 11:00:00
categories:
- nlp
- machine learning
tags:
- kmeans
---

문서 군집화는 비슷한 문서들을 하나의 군집으로 묶어줍니다. k-means 는 빠른 속도와 안정성 때문에 문서 군집화에 유용합니다. 하지만, 학습된 군집이 어떤 주제의 문서들이 모여있는지를 확인하기는 어렵습니다. 데이터가 수백만건이라면 각 군집에 속한 문서를 확인하는 것도 불가능합니다. 하지만 데이터 기반으로 각 군집의 레이블을 할당할 수 있습니다. 만약 Bag of words model 로 표현된 문서에 k-means 를 적용시켰다면, 학습된 center vectors 를 이용하여 손쉽게 군집에 레이블을 부여할 수 있습니다. 


## k-means Introduction

k-means 는 다른 군집화 알고리즘과 비교하여 매우 적은 계산 비용을 요구하면서도 안정적인 성능을 보입니다. 그렇기 때문에 큰 규모의 데이터 군집화에 적합합니다. 문서 군집화의 경우에는 문서의 개수가 수만건에서 수천만건 정도 되는 경우가 많기 때문에 다른 알고리즘보다도 k-means 가 더 많이 선호됩니다. k-partition problem 은 데이터를 k 개의 겹치지 않은 부분데이터 (partition)로 분할하는 문제 입니다. 이 때 나뉘어지는 k 개의 partiton 에 대하여, **"같은 partition 에 속한 데이터 간에는 서로 비슷하며, 서로 다른 partition 에 속한 데이터 간에는 이질적"**이도록 만드는 것이 군집화라 생각할 수 있습니다. k-means problem 은 각 군집 (partition)의 평균 벡터와 각 군집에 속한 데이터 간의 거리 제곱의 합 (분산, variance)이 최소가 되는 partition 을 찾는 문제입니다. 

$$\sum _{i=1}^{k}\sum _{\mathbf {x} \in S_{i}}\left\|\mathbf {x} -{\boldsymbol {\mu }}_{i}\right\|^{2}$$

우리가 흔히 말하는 k-means 알고리즘은 Lloyd 에 의하여 제안되었습니다. 이는 다음의 순서로 이뤄져 있습니다. 

1. k 개의 군집 대표 벡터 (centroids) 를 데이터의 임의의 k 개의 점으로 선택합니다. 
2. 모든 점에 대하여 가장 가까운 centroid 를 찾아 cluster label 을 부여하고, 
3. 같은 cluster label 을 지닌 데이터들의 평균 벡터를 구하여 centroid 를 업데이트 합니다. 
4. Step 2 - 3 의 과정을 label 의 변화가 없을때까지 반복합니다. 

Lloyd k-means 는 대체로 빠르고 안정적인 학습 결과를 보여줍니다만, 몇 가지 단점을 가지고 있습니다. 이들은 우리가 데이터에 (Lloyd) k-means 를 적용할 때 의사결정을 해야 하는 부분이기도 합니다. 

    (1) initial points 설정
    (2) iteration 횟수
    (3) distance measure
    (4) 적절한 k 의 개수 설정
    (5) 학습된 클러스터의 레이블 부여

(1) - (4) 는 알고리즘이 학습에 이용하는 패러매터의 이슈입니다만, 학습된 군집에 레이블을 부여하는 것은 해석, 혹은 모델과 사람 사이의 interaction 이슈입니다. 이번 포스트에서는 **(5) 학습된 클러스터의 레이블 부여**에 대하여 다뤄봅니다.

## From text to k-means (Scikit-learn)

scikit-learn 에서 제공하는 k-means 를 이용하여 문서를 군집화 하는 방법은 아래와 같습니다. list of str 형식의 문서 집합을 vectorizer 를 통하여 doc - term matrix 로 변환합니다. tf-idf 형식으로 표현하고 싶다면, CountVectorizer 대신 TfidfVectorizer 를 이용하면 됩니다. normalize() 는 X 의 각 row 를 크기가 1인 unit vector 로 변환하기 위함입니다. 이에 대한 자세한 이야기는 "Spherical k-means for document clustering" 포스트를 참조해주세요. 우리는 일단 scikit-learn 의 k-means 를 이용하여 문서 군집화를 학습했다는 가정하에, 그 뒷 이야기를 할 것입니다. 

{% highlight python %}
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

docs = ['document format', 'list of str like']
k=2

# vectorizing
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

# L2 normalizing
X = normalize(X, norm=2)

# training k-means
kmeans = KMeans(n_clusters=k).fit(X)

# trained labels and cluster centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_
{% endhighlight %}

k-means 의 학습 결과는 두 가지 입니다. (1) 각 row 가 어떤 군집에 속하는지를 표현하는 kmeans.labels_ 과 (2) 각 군집의 평균 벡터 kmeans.cluster_centers_ 입니다. 이 두 가지 정보를 이용하여 데이터 기반으로 각 군집에 레이블을 부여해보겠습니다. 

## Clsutering labeling from keyword extraction

우리가 Bag of words model 을 이용하였다면 k-means 의 학습 결과인 cluster centers 는 각 군집에 속한 문서들에서 어떤 단어가 얼만큼 등장하였는지를 표현하는 벡터입니다. Term frequency vector 와 거의 유사합니다. 물론 L2 정규화를 한다면 정확하게 단어의 등장 비율은 아니겠지만, 자주 등장하는 단어가 큰 가중치 (weight)를 지닌다는 점에서는 변함이 없습니다. 만약 doc2vec 과 같은 distributed representation 으로 문서가 표현되었었다 하더라도, labels 를 이용하면 각 군집에 대한 Bag of words model 로 표현된 cluster center vector 를 얻을 수 있습니다. 

사실 cluster labeling 의 분야는 그렇게 많이 연구되지는 않았습니다. 하지만, 토픽모델링에 자주 이용되는 Latent Dirichlet Allocation (LDA) 의 학습 결과인 topics 을 데이터 기반으로 해석하려는 topic labeling 은 연구들이 많습니다. Topic 을 몇 개의 단어로 레이블링 한다는 것은 각 topic 의 키워드를 선택하는 것과 같습니다. Keyword extraction / Topic labeling / Clustering labeling 은 겹치는 부분이 매우 많습니다. 

그럼 어떤 단어가 키워드로 적합할까요? Topic labeling 의 연구들에서는 대체로 두 가지의 기준에 대하여 이야기합니다. 첫째는 salience 입니다. 한 군집/토픽의 키워드라면, 각 군집/토픽에 속한 많은 문서들에서 등장해야 할 것입니다. 한 단어의 coverage 가 높다는 의미입니다. 둘째는 discriminative power 입니다. coverage 가 가장 높은 단어는 'a, the, -은, -는, -이, -가' 와 같은 문법적인 단어일 것입니다. 하지만 'a'라는 단어는 어떤 군집을 명확히 지시해주지도 않습니다. 차별성이 없는 단어는 키워드로 부적합합니다. 하지만 최고의 discriminative power 를 지닌 단어는 infrequent terms 일 가능성이 높습니다. 한 군집에서만, 사실은 그 군집에서 딱 5번 등장한 단어라면, 그 단어를 들었을 때 어떤 군집을 특정할 수 있기 때문입니다. 즉, salience 와 discriminative power 사이에는 negative correlation 이 있습니다. 앙면을 모두 고려하여 키워드를 선택해야 합니다. 

우리는 cluster center vectors 를 이용하여 salient and discriminative 한 키워드 집합을 선택해 보겠습니다. 일단 salient terms 이라면 각 군집에서 large weight 를 지닌 단어일 것입니다. 그렇다면 우리가 해야 할 작업은 discriminative power 를 수식으로 정의하는 것입니다. Discriminative power 는 레이블을 달고 싶은 군집에서는 자주 등장하지만, 다른 군집에서는 등장하지 않은 단어가 클 것입니다. 그래서 아래와 같이 군집 c 에서의 단어 v 의 키워드 점수를 정의하였습니다. 

$$s(v, c) = \frac{w(v \vert c)}{w(v \vert c) + w(v \vert \tilde{c})}$$

- $$w(v \vert c)$$ : 군집 c 의 center vector 에서의 term v 에 대한 weight
- $$w(v \vert \tilde{c})$$ : 군집 c 를 제외한 다른 문서 집합들에서의 term v 에 대한 weight

위의 점수 공식은 해석이 잘 된다는 장점이 있습니다. 만약 한 군집에서 1% 등장하는 단어가 보통 1% 등장하는 단어였다면 $$s(v, c)$$ 는 0.5 = (1%) / (1% + 1%) 일 것입니다. 만약 군집 c 에만 등장한다면 $$s(v, c) = 1.0$$ 입니다. Cluster center vector weight 는 정확히는 단어의 등장 비율은 아니지만, scaling 이 되었다고 생각하면 됩니다. 경향은 비슷하니까요. 

Saliency 와 discriminative power 를 모두 고려하기 위하여 우리는 각 군집에서 weight 가 큰 순서대로 top k1 개의 단어를 키워드 후보로 선택한 뒤, 이들에 대하여 $$s(v, c)$$를 계산합니다. 그리고 $$s(v, c)$$가 큰 순서대로 top k2 개의 단어를 군집의 레이블로 선택합니다. 

## IMDB review experiment

제안된 방법이 cluster labeling 에 적합한지 실험을 수행하였습니다. 실험에 이용한 데이터는 122M 개의 문서로 이뤄진 IMDB reviews 입니다. 이 데이터에 대하여 100 개의 initial points 를 선택하였습니다. 

| Dataset name | Num of docs | Num of terms | Num of nonzero | Sparsity |
| --- | --- | --- | --- | --- |
| IMDB reviews | 1,228,348 | 68,049 | 181,411,713 | 0.9978 |

문서의 개수가 1.22M 이기 때문에 군집의 개수는 k=1,000 으로 설정하였습니다. 수렴은 0.1 % 수준까지 시켰으며, center vector 의 weight 의 크기 순서로 top 300 개 중에서 keyword score 가 높은 순으로 키워드 (레이블)를 선택하였습니다. '타이타닉'과 같이 한 영화의 리뷰들이 하나의 군집으로 학습되기도 하지만, Marvle comics 의 영화들이 하나의 군집으로 뭉치거나, Clover field, District 9 과 같은 공포영화들이 하나의 군집으로 뭉치는 것도 확인할 수 있습니다. 아니, 그런 영화들이 하나의 군집으로 뭉쳤다고 우리는 군집화 결과를 해석할 수 있습니다. 

<table>
  <colgroup>
    <col width="20%" />
    <col width="80%" />
  </colgroup>
  <thead>
    <tr class="query_and_topic">
      <th>군집의 의미</th>
      <th>키워드 (레이블)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td markdown="span"> 영화 “타이타닉” </td>
      <td markdown="span"> iceberg, zane, sinking, titanic, rose, winslet, camerons, 1997, leonardo, leo, ship, cameron, dicaprio, kate, tragedy, jack, di saster, james, romance, love, effects, special, story, people, best, ever, made </td>
    </tr>
    <tr>
      <td markdown="span"> Marvle comics 의 heros (Avengers) </td>
      <td markdown="span"> zemo, chadwick, boseman, bucky, panther, holland, cap, infinity, mcu, russo, civil, bvs, antman, winter, ultron, airport, ave ngers, marvel, captain, superheroes, soldier, stark, evans, america, iron, spiderman, downey, tony, superhero, heroes </td>
    </tr>
    <tr>
      <td markdown="span"> Cover-field, District 9 등 외계인 관련 영화 </td>
      <td markdown="span"> skyline, jarrod, balfour, strause, invasion, independence, cloverfield, angeles, district, los, worlds, aliens, alien, la, budget, scifi, battle, cgi, day, effects, war, special, ending, bad, better, why, they, characters, their, people </td>
    </tr>
    <tr>
      <td markdown="span"> 살인자가 출연하는 공포 영화 </td>
      <td markdown="span"> gayheart, loretta, candyman, legends, urban, witt, campus, tara, reid, legend, alicia, englund, leto, rebecca, jared, scream, murders, slasher, helen, killer, student, college, students, teen, summer, cut, horror, final, sequel, scary </td>
    </tr>
    <tr>
      <td markdown="span"> 영화 “매트릭스" </td>
      <td markdown="span"> neo, morpheus, neos, oracle, trinity, zion, architect, hacker, reloaded, revolutions, wachowski, fishburne, machines, agents, matrix, keanu, smith, reeves, agent, jesus, machine, computer, humans, fighting, fight, world, cool, real, special, effects </td>
    </tr>
  </tbody>
</table>

물론 lasso regression 을 이용하여 키워드를 추출할 수도 있습니다. 하지만, 이 실험에서는 1,000 개의 군집이 존재합니다. 천 개의 sub matrix 를 만들고, 천 번의 lasss model 을 학습하는 데에는 오랜 시간이 걸립니다. 중복된 단어들이 등장하더라도 빠르게 군집을 해석할 수 있는 레이블을 달고 싶을 때 이 포스트의 방법을 이용하면 좋습니다.

## Packages

이와 관련된 코드는 github 의 [clustering4docs][clustering4docs] repository 에 올려뒀습니다. 사용법은 아래와 같습니다. 

{% highlight python %}
from soyclustering import proportion_keywords

vocabs = ['this', 'is', 'vocab', 'list']
labels = kmeans.labels_
centers = kmeans.cluster_centers_

keywords = proportion_keywords(
    centers,
    labels=labels,
    index2word=vocabs)
{% endhighlight %}

각 군집의 centers 와 labels 를 넣어줍니다. labels 는 각 군집에 속한 문서의 개수를 계산하는데 이용됩니다. 이를 입력하지 않으면 모든 군집의 크기를 동일하게 생각합니다. 또한 index2word 에 list of str 형식의 vocabulary 를 입력하면 키워드로 선택된 단어들을 str 로 변환하여 return 합니다. 이를 입력하지 않으면 (vocab idx, score) 형식으로 return 됩니다. 


[clustering4docs]: https://github.com/lovit/clustering4docs