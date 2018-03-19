---
title: k-means initial points 선택 방법
date: 2018-03-19 17:00:00
categories:
- nlp
- machine learning
tags:
- kmeans
---

k-means 는 다른 군집화 알고리즘과 비교하여 매우 빠른 계산 속도를 보임에도 불구하고 안정적인 성능을 보여주기 때문에 큰 규모의 데이터 군집화에 적합합니다. 특히 문서 군집화의 경우에는 문서의 개수가 수만건에서 수천만건 정도 되는 경우가 많기 때문에 다른 알고리즘보다도 k-means 가 더 많이 선호됩니다. k-means 문제는 정확히는, k-partition problem 입니다. 데이터를 k 개의 겹치지 않은 부분데이터 (partition)로 분할하는 것입니다. 이 때 나뉘어지는 k 개의 partiton 에 대하여, "같은 partition 에 속한 데이터 간에는 서로 비슷하며, 서로 다른 partition 에 속한 데이터 간에는 이질적"이도록 만드는 것이 군집화라 생각할 수 있습니다. k-means problem 은 각 군집 (partition)의 평균 벡터와 각 군집에 속한 데이터 간의 거리 제곱의 합 (분산, variance)이 최소가 되는 partition 을 찾는 문제입니다. 

$$\sum _{i=1}^{k}\sum _{\mathbf {x} \in S_{i}}\left\|\mathbf {x} -{\boldsymbol {\mu }}_{i}\right\|^{2}$$

우리가 잘 알고 있는 k-means 알고리즘 중 하나는 Lloyd k-means 입니다. 이는 다음의 순서로 이뤄져 있습니다. 

1. 임의로 k 개의 점을 cetroid 로 선택한 뒤, 
2. 모든 점에 대하여 가장 가까운 centroid 를 찾아 cluster label 을 부여하고, 
3. 같은 cluster label 을 지닌 데이터들의 평균 벡터를 구하여 centroid 를 업데이트 합니다. 
4. Step 2 - 3 의 과정을 label 의 변화가 없을때까지 반복합니다. 

Lloyd k-means 는 빠르게 k-means problem 을 풀 수 있지만, 몇 가지 단점을 가지고 있습니다. 데이터에 (Lloyd) k-means 를 적용할 때 우리가 의사결정을 해야 하는 부분이기도 합니다. 

    (1) initial points 에 따른 안정성
    (2) iteration 횟수
    (3) distance measure
    (4) 적절한 k 의 개수 설정

이번 포스트에서는 문서 군집화에 대하여 initial points 설정에 대한 이야기를 하려 합니다. k-means 는 initial points 가 제대로 설정되지 않으면 불안정한 군집화 결과를 학습한다 알려져 있습니다. 사실 k-means 의 학습 결과가 좋지 않는 경우는 initial points 로 비슷한 점들이 여러 개 선택 된 경우입니다. 이 경우만 아니라면 k-means 는 빠른 수렴속도와 안정적인 성능을 보여줍니다. 그렇기 때문에 질 좋은 initial points 를 선택하려는 연구들이 진행되었습니다. 그 중에서도 가장 널리 알려진 방법이 k-means++ 입니다 (scalable k-means^2 은 Spark 와 같은 분산 환경 버전의 k-means++ 입니다). Python 의 scikit-learn 의 k_means 에는 사용자가 결정할 다양한 option 이 있습니다. 이 중에서 init='k-means++' 이라는 부분이 보입니다. 다른 옵션으로는, 사용자가 임의로 설정한 seed points 를 이용하던지, random sampling 을 할 수도 있습니다. 

{% highlight python %}
def k_means(X, n_clusters, init='k-means++', precompute_distances='auto',
            n_init=10, max_iter=300, verbose=False,
            tol=1e-4, random_state=None, copy_x=True, n_jobs=1,
            algorithm="auto", return_n_iter=False):
{% endhighlight %}
    
k-means++ 은 다음처럼 작동합니다. 

1. 첫 initial point $$c_1$$ 은 임의로 선택합니다. 
2. 이후의 initial point $$c_t$$ 는 이전에 선택한 $$c_{t-1}$$ 과의 거리인 $$d(c_{t-1}, c_{t})$$ 를 이용합니다. 다음의 확률에 따라 임의로 하나의 점을 선택합니다. $$\frac{d(c_{t-1}, c_t)}{\sum d(c_{t-1}, c_{t^`})}$$
3. k 개의 initial points 를 선택할 때까지 step 2 를 반복합니다. 

Step 2 의 확률의 의미는 이전에 선택한 점 $$c_{t-1}$$ 과 거리가 멀수록 선택될 확률이 높다는 의미입니다. 이렇게 함으로써 비슷한 점들이 initial points 로 선택되지 않게 하려던 것입니다. 그러나 k-means++ 도 문제점을 지니고 있습니다. Cosine 을 이용하는 문서 군집화 과정을 살펴봅시다. 문서 간 거리는 Euclidean distance 보다 Cosine distance 가 더 적합합니다. Bag of words model 을 이용한다면 문서가 sparse vector 로 표현되기 때문에 공통된 단어의 개수에 대한 정보를 포함하는 Jaccard, Cosine 과 같은 metrics 이 적합합니다^3. 그럼 우리는 샘플데이터를 이용하여 문서 간 거리의 분포를 살펴보겠습니다. 샘플데이터는 3만여건의 하루 치 뉴스를 Bag of words model 로 표현한 데이터입니다. 9,774 개의 단어로 표현된 문서 집합입니다.  

{% highlight python %}
print(x.shape)
#(30091, 9774)
{% endhighlight %}

이 데이터에 대하여 1,000 개의 문서를 random sampling 하여 다른 문서 간의 거리를 계산합니다 (모든 문서 간 거리를 계산하면 오래 걸리니까요). 

{% highlight python %}
from sklearn.metrics import pairwise_distances
from numpy.random import permutation
from numpy import histogram

sample_idx = permutation(x.shape[0])[:1000]
dist = pairwise_distances(x, x[sample_idx,:], metric='cosine')
hist, bin_edges = histogram(dist, bins=20)
{% endhighlight %}
    
문서 간 거리 분포를 살펴보면 거리가 0.85 이상인 경우가 91.79 % 에 해당합니다. 이는 고차원 벡터에서의 거리 척도의 특징입니다. 고차원에서는 Euclidean 이던지, Cosine 이던지 "가까운 거리는 의미가 있으나, 먼 거리는 의미가 없습니다". 이에 대해서는 나중에 더 자세히 이야기하겠습니다. 결국, 대부분의 문서 간 거리가 0.85 ~ 1.00 이라는 의미이고, k-means++ 의 step 2 과정에서 계산된 sampling probability 는 사실 uniform distribution 에 가깝습니다. 그런데, 모든 점들 간의 거리를 계산하고, 이를 cumulative distribution 으로 바꾸어 random sampling 을 수행하는 과정은 생각보다도 비싼 계산과정입니다. 즉 문서 군집화 과정에서 k-means++ 을 이용한다는 것은 "매우 비싼 random sampling" 을 수행하는 것입니다.
    
    [distance range]: num of value (percentage)
    -------------------------------------------
    [0.000 ~ 0.050] :      37848   (0.13 %)
    [0.050 ~ 0.100] :       8106   (0.03 %)
    [0.100 ~ 0.150] :      18554   (0.06 %)
    [0.150 ~ 0.200] :      32180   (0.11 %)
    [0.200 ~ 0.250] :      21512   (0.07 %)
    [0.250 ~ 0.300] :      69913   (0.23 %)
    [0.300 ~ 0.350] :      26691   (0.09 %)
    [0.350 ~ 0.400] :      25581   (0.09 %)
    [0.400 ~ 0.450] :      31954   (0.11 %)
    [0.450 ~ 0.500] :      29859   (0.10 %)
    [0.500 ~ 0.550] :      54503   (0.18 %)
    [0.550 ~ 0.600] :      60503   (0.20 %)
    [0.600 ~ 0.650] :      71768   (0.24 %)
    [0.650 ~ 0.700] :     132200   (0.44 %)
    [0.700 ~ 0.750] :     239247   (0.80 %)
    [0.750 ~ 0.800] :     511296   (1.70 %)
    [0.800 ~ 0.850] :    1098302   (3.65 %)
    [0.850 ~ 0.900] :    2469531   (8.21 %)
    [0.900 ~ 0.950] :    7754535   (25.77 %)
    [0.950 ~ 1.000] :   17396917   (57.81 %)
    
    

## Reference    
1. Arthur, D., & Vassilvitskii, S. (2007, January). k-means++: The advantages of careful seeding. In Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms (pp. 1027-1035). Society for Industrial and Applied Mathematics.
2. Bahmani, B., Moseley, B., Vattani, A., Kumar, R., & Vassilvitskii, S. (2012). Scalable k-means++. Proceedings of the VLDB Endowment, 5(7), 622-633.
3. Huang, A. (2008, April). Similarity measures for text document clustering. In Proceedings of the sixth new zealand computer science research student conference (NZCSRSC2008), Christchurch, New Zealand (pp. 49-56).