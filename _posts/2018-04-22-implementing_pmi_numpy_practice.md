---
title: Implementing PMI (Practice handling matrix of numpy & scipy)
date: 2018-04-22 18:00:00
categories:
- nlp
tags:
- preprocessing
---

Point Mutual Information (PMI) 은 두 변수의 상관성을 측정하는 방법 중 하나입니다. 이를 구현하는 방법은 다양합니다. 간단한 개념이기 때문에 numpy 와 scipy 의 matrix handling 을 연습하기에 적절한 예제입니다. 이번 포스트에서는 numpy.ndarray 와 scipy.sparse 의 matrix 를 이용하여 PMI, PPMI 를 계산하는 함수를 구현합니다.


## Brief review of Point Mutial Information

확률에서 두 확률 분포가 서로 독립인가 (두 확률은 서로 상관이 없는가)의 조건은 joint probability 와 각각의 marginal probability 의 곱의 비율이 1인가 입니다. 식으로는 다음처럼 표현됩니다. 

$$\frac{p_{i,j}}{p_i \times p_j}$$

안경을 쓰는 것과 저녁을 먹는 것은 서로 상관없는 문제입니다. 전체 1200 명의 사람 중 저녁을 먹은 사람은 400 명, 1/3 입니다. 1200 명 중 안경을 쓴 사람은 300 명, 1/4 입니다. 그리고 안경을 썼으며, 저녁을 먹은 사람은 1/12 입니다. 각각의 확률의 곱과 같습니다. 

다른 관점으로 살펴봅니다. 안경을 썼을 경우로 제한하면, 저녁을 먹은 사람의 비율은 100 명 / 300 명 = 1/3 입니다. 안경을 쓰지 않은 사람이 저녁을 먹은 비율은 300 / 900 = 1/3 입니다. 안경을 썼는지와 저녁을 먹었는지가 서로 상관없는 일이라면, 안경을 썼는지와 상관없이 저녁을 먹은 사람의 비율이 일정해야 합니다. 위 식은 이 말을 수식으로 표현한 것입니다. $$\frac{p_{i,j}}{p_i \times p_j}$$ 은 $$\frac{p_{j \vert i}}{p_j}$$ 이기 때문입니다.

| 안경 \ 저녁 | 저녁을 먹었다 | 저녁을 안먹었다 | marginal prob |
| 안경을 썼다 |  100 | 200 | $$\frac{1}{4}$$ = $$\frac{300}{1200}$$ |
| 안경을 안썼다 | 300 | 600 | $$\frac{3}{4}$$ = $$\frac{900}{1200}$$ |
| marginal prob | $$\frac{1}{3}$$ = $$\frac{400}{1200}$$ | $$\frac{3}{4}$$ = $$\frac{800}{1200}$$ | .. |

PMI 는 위 수식에 $$log$$ 를 취합니다. $$log(1) = 0$$ 이므로, 서로 상관이 없는 $$i$$ 와 $$j$$ 의 pmi 는 0 입니다. 

$$PMI_{i,j} = log \left( \frac{p_{i,j}}{p_i \times p_j} \right)$$

안경을 썼을수록 저녁을 먹을 가능성이 높다면 (서로 양의 상관성이 있다면) pmi 는 0 보다 큽니다. 

| 안경 \ 저녁 | 저녁을 먹었다 | 저녁을 안먹었다 | marginal prob |
| 안경을 썼다 |  200 | 100 | $$\frac{1}{4}$$ = $$\frac{300}{1200}$$ |
| 안경을 안썼다 | 300 | 600 | $$\frac{3}{4}$$ = $$\frac{900}{1200}$$ |
| marginal prob | $$\frac{5}{12}$$ = $$\frac{500}{1200}$$ | $$\frac{7}{12}$$ = $$\frac{700}{1200}$$ | .. |

$$PMI(안경+, 저녁+) = log \left( \frac{ \frac{200}{1200} }{ \frac{200}{300} \times \frac{200}{500} } \right) = log(1.2) = 0.182$$

음의 상관관계가 있다면 pmi 는 음수입니다. 자연어처리에서의 semantic 에서는 negative correlation 에 관심이 적습니다. PPMI 는 0 보다 작은 PMI 의 값을 모두 0 으로 만들고, positive correlation 에만 초점을 둡니다.

$$PPMI_{i,j} = max(0, PMI_{i,j})$$

PMI 의 단점 중 하나는 infrequent pattern 에 대해서 지나치게 민감하게 반응한다는 것입니다. $$p_j$$ 의 확률이 지나치게 작으면 $$p_{j \vert i}$$ 를 나눴을 때 그 수를 매우 크게 만들 수 있습니다. 이를 해결하는 간단한 방법 중 하나는 infrequent pattern 의 probability $$p_j$$ 에 $$\alpha$$ 를 더하는 것입니다. 이 방법은 language model 에서의 smoothing 과도 비슷합니다. 

$$PMI_{i,j} = log \left( \frac{p_{i,j}}{p_i \times \left( p_j + \alpha \right) } \right)$$


## Implementing PMI with dense matrix (numpy.ndarray)

Numpy 의 dense matrix 로 표현되는 작은 행렬을 이용하여 matrix handling 에 대하여 연습합니다. 눈으로 값의 변화를 확인하며 행렬을 다루는 방법을 익힌 뒤, sparse matrix 에 응용합니다. 

우리가 이용할 행렬은 아래와 같습니다. (4, 3) 의 작은 행렬입니다. $$x$$ 는 rows, $$y$$ 는 columns 입니다.

{% highlight python %}
import numpy as np

x = np.array([[3, 0, 0], 
              [0, 2, 0],
              [1, 0, 1],
              [2, 0, 1]
             ])
{% endhighlight %}

$$p(x)$$ 와 $$p(y)$$ 를 계산합니다. x.sum(axis=0) 은 row sum 이며, x.sum(axis=1) 은 column sum 입니다. 모든 columns 을 하나로 합치면 각 row 의 sum 이 계산됩니다. 이 값을 행렬 전체의 합인 x.sum() 으로 나누면 $$p(x)$$ 를 얻을 수 있습니다. 

{% highlight python %}
# marginal probability
px = x.sum(axis=1) / x.sum()
py = x.sum(axis=0) / x.sum()

print(px) # [0.3 0.2 0.2 0.3]
print(py) # [0.6 0.2 0.2]
{% endhighlight %}

$$p(x,y)$$ 를 계산하기 위해서는 $$x$$ 를 x.sum() 으로 나눕니다.

{% highlight python %}
# convert x to probability matrix
pxy = x / x.sum()
print(pxy)

# [[0.3 0.  0. ]
#  [0.  0.2 0. ]
#  [0.1 0.  0.1]
#  [0.2 0.  0.1]]
{% endhighlight %}

$$p(x,y)$$ 를 $$p(x)$$ 로 나누기 위해서는 행렬곲을 이용할 수 있습니다. $$p(x)$$ 와 $$p(y)$$ 를 diagonal matrix 로 만듭니다. $$p(x)$$ 의 $$i$$ 번째 값은 diagonal matrix 의 $$(i,i)$$ 의 값입니다. 이를 위해 numpy.diag 를 이용합니다. numpy.diag 는 array 의 값을 diagonal elements 로 지니는 diagonal matrix 를 만듭니다.

{% highlight python %}
# diagonalize px & py for matrix multiplication
# (4, 4) by (4, 3) by (3, 3) = (4, 3)
np.diag(px)

# array([[0.3, 0. , 0. , 0. ],
#        [0. , 0.2, 0. , 0. ],
#        [0. , 0. , 0.2, 0. ],
#        [0. , 0. , 0. , 0.3]])
{% endhighlight %}

$$p(x)$$ 를 곱하는 것이 아니라 나누는 것이기 때문에 역수를 취합니다. 이 때 $$p(x)$$ 가 0 인 원소는 그 값을 나누지 않고 0 으로 할당합니다.

{% highlight python %}
# inverse elements if the element is greater than 0
np.diag(np.array([0 if pxi == 0 else 1/pxi for pxi in px]))

# array([[3.33333333, 0.        , 0.        , 0.        ],
#        [0.        , 5.        , 0.        , 0.        ],
#        [0.        , 0.        , 5.        , 0.        ],
#        [0.        , 0.        , 0.        , 3.33333333]])
{% endhighlight %}

위 방법을 이용하여 $$p(x)$$ 의 역수와 $$p(y)$$ 의 역수로 이뤄진 diagonal matrix 를 만듭니다. 이 때 $$p(y)$$ 에 $$\alpha$$ 를 더하는 smoothing 도 할 수 있습니다. $$p(y)_i$$ 가 0 이 아닐 때 $$\alpha$$ 를 더합니다.

{% highlight python %}
# inverse element diagonal matrix of px and py
alpha = 0

px_diag = np.diag(np.array([0 if pxi == 0 else 1/pxi for pxi in px]))
py_diag = np.diag(np.array([0 if pyi == 0 else 1/(pyi + alpha) for pyi in py]))

print(px_diag.shape) # (4, 4)
print(py_diag.shape) # (3, 3)
{% endhighlight %}

행렬 곲은 각 행렬의 .dot 함수를 이용할 수 있습니다. numpy.dot 이 호출되어 두 행렬이 곱해집니다. 

{% highlight python %}
# logarithm is not applied yet
exp_pmi = px_diag.dot(pxy).dot(py_diag)
print(exp_pmi)

# array([[1.66666667, 0.        , 0.        ],
#        [0.        , 5.        , 0.        ],
#        [0.83333333, 0.        , 2.5       ],
#        [1.11111111, 0.        , 1.66666667]])
{% endhighlight %}

우리가 행렬곲으로 계산한 결과와 손으로 직접 계산한 결과가 같은지 확인합니다. 이처럼 계산과정이 제대로 구현되었는지 값을 넣어 직접 확인하는 작업은 매우 중요합니다.

행렬곲으로 얻은 값과 손으로 계산한 값이 다르면 그 값을 출력하도록 합니다. 네 개의 값이 서로 다릅니다. 하지만 그 값 차이를 살펴보면 float truncated error 때문에 발생한 것임을 알 수 있습니다. 다행히도 우리의 구현이 맞았습니다. 

{% highlight python %}
# check value
# difference cause by truncation error
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        exp_pmi_ij = exp_pmi[i,j]
        manually_ij = pxy[i,j] / (px[i] * py[j])
        if not (exp_pmi_ij == manually_ij):
            print('({}, {}), exp_pmi = {}, manually = {}'.format(
                i, j, exp_pmi_ij, manually_ij))

# (1, 1), exp_pmi = 5.0, manually = 4.999999999999999
# (2, 2), exp_pmi = 2.5, manually = 2.4999999999999996
# (3, 0), exp_pmi = 1.1111111111111114, manually = 1.1111111111111112
# (3, 2), exp_pmi = 1.666666666666667, manually = 1.6666666666666667
{% endhighlight %}

아직까지는 $$\frac{p(x,y)}{p(x) \times p(y)}$$ 만 계산했습니다. log 값을 취해야 합니다. Minimum pmi 보다 작은 경우는 제거하고, $$(x, y) = pmi$$ 의 형식으로 dok_matrix 에 저장합니다. Sparse matrix 의 형식 중 하나입니다. dok_matrix 에 대해서는 [이전 포스트][sparse_post]를 참고하세요. 

numpy.where 를 이용하면 해당 조건을 만족하는 rows, columns 가 return 됩니다. zip(rows, cols) 를 이용하여 각 $$(i, j)$$ 의 값에 접근합니다. 

{% highlight python %}
from scipy.sparse import dok_matrix

# PPMI using threshold
min_exp_pmi = 1

rows, cols = np.where(exp_pmi > min_exp_pmi)
pmi_dok = dok_matrix(exp_pmi.shape)

for i, j in zip(rows, cols):
    # apply logarithm
    pmi_dok[i,j] = np.log(exp_pmi[i,j])
{% endhighlight %}

계산된 결과를 확인합니다.

{% highlight python %}
for position, pmi_ij in pmi_dok.items():
    print('{} = {} (exp = {})'.format(
        position, pmi_ij, np.exp(pmi_ij)))

# (0, 0) = 0.5108256237659907 (exp = 1.6666666666666667)
# (1, 1) = 1.6094379124341003 (exp = 4.999999999999999)
# (2, 2) = 0.9162907318741551 (exp = 2.5)
# (3, 0) = 0.10536051565782655 (exp = 1.1111111111111114)
# (3, 2) = 0.5108256237659908 (exp = 1.666666666666667)
{% endhighlight %}

Numpy 를 이용하여 우리의 구현이 pmi 의 계산에 정확한지 확인하였습니다.


## Implementing PMI with sparse matrix (scipy.sparse)

앞서서 logic 을 확인하였으니 이를 응용하여 sparse matrix 에서의 PMI 를 계산하는 과정을 구현합니다. 데이터는 (질문, 답변) pairs 의 단어 간의 co-occurrence matrix 입니다. 질문이 $$x$$, 답변이 $$y$$ 입니다. 17k 개의 단어로 이뤄져 있습니다. 

{% highlight python %}
idx2vocab = [word.strip() for word in f]
vocab2idx = {vocab:idx for idx, vocab in enumerate(idx2vocab)}
print(x.shape) # (17761, 17761)
{% endhighlight %}

sparse matrix 에서도 sum(axis=0) 과 sum(axis=1) 은 같습니다. reshape(-1) 을 이용하여 row vector 를 만듭니다.

{% highlight python %}
# convert x to probability matrix & marginal probability 
px = (x.sum(axis=1) / x.sum()).reshape(-1)
py = (x.sum(axis=0) / x.sum()).reshape(-1)
pxy = x / x.sum()

print(px.shape) # (1, 17761)
print(py.shape) # (1, 17761)
print(pxy.shape) #  (17761, 17761)
{% endhighlight %}

rows 를 list 로 만든 뒤, 이를 diagonal elements 로 지니는 diagonal matrix 로 만듭니다. scipy.sparse 에서도 diagonal matrix 를 제공합니다. 

{% highlight python %}
from scipy.sparse import diags

px_diag = diags(px.tolist()[0])
py_diag = diags(py.tolist()[0])
{% endhighlight %}

scipy.sparse.diag 의 data 는 numpy.ndarray 입니다. Diagonal matrix 는 대각의 원소만 저장하면 되니까요. 

{% highlight python %}
print(type(px_diag.data)) # class 'numpy.ndarray'
print(px_diag.data.shape) # (1, 17761)
{% endhighlight %}

이번에도 $$\alpha$$ 를 $$p(y)_i$$ 에 더하여 smoothing 을 합니다. scipy.sparse.diag.data 의 형식이 ndarray 이기 때문에 이 값의 역수를 취한 뒤, 이를 형식이 같은 numpy.ndarray 로 만들었습니다.

{% highlight python %}
alpha = 0.0001 # acts as p(y) threshold

px_diag.data[0] = np.asarray([0 if v == 0 else 1/v for v in px_diag.data[0]])
py_diag.data[0] = np.asarray([0 if v == 0 else 1/(v + alpha) for v in py_diag.data[0]])
{% endhighlight %}

각 행렬의 type 를 확인합니다.

{% highlight python %}
print(type(px_diag)) # scipy.sparse.dia.dia_matrix
print(type(pxy))     # scipy.sparse.csr.csr_matrix
print(type(py_diag)) # scipy.sparse.dia.dia_matrix
{% endhighlight %}

.dot() 함수를 이용하여 행렬곲을 합니다. 위 세 개 행렬이 모두 sparse matrix 이기 때문에 scipy 의 _safe_sparse_dot 이 호출됩니다. 이 중 하나라도 numpy.ndarray 라면 numpy.dot 이 호출되어 sparse matrix 가 dense matrix 화 됩니다. Sparse matrix 를 다룰 때는 numpy.ndarray 와 함께 이용되는 경우가 없는지 늘 조심합시다.

{% highlight python %}
exp_pmi = px_diag.dot(pxy).dot(py_diag)
print(exp_pmi.shape) # (17761, 17761)
{% endhighlight %}

Minimum pmi 를 이용하여 threshold cutting 을 합니다. Minimum pmi = 0 을 적용합니다. $$exp(0) = 1$$ 이므로, 1 을 threshold 로 이용합니다. 

pxy 가 scipy.sparse.csr.csr_matrix 이기 때문에 exp_pmi 의 형식도 csr_matrix 입니다. 0 이 아닌 값들에 대해서만 minimum pmi 와의 비교를 하면 되기 때문에 data array 의 0 이 아닌 값의 indices 를 가져옵니다. 

{% highlight python %}
# PPMI using threshold
min_exp_pmi = 1

# because exp_pmi is sparse matrix and type of exp_pmi.data is numpy.ndarray
indices = np.where(exp_pmi.data > min_exp_pmi)[0]
{% endhighlight %}

csr_matrix.nonzero() 를 통하여 return 되는 rows, cols 와 csr_matrix.data 는 모두 같은 길이의 numpy.ndarray 입니다. zip(rows, cols, data) 를 이용하면 (i, j, value) 를 얻을 수 있습니다. 

Minimum pmi 값보다 큰 값들에 logarithm 을 적용하여 pmi_dok 에 저장합니다.

{% highlight python %}
pmi_dok = dok_matrix(exp_pmi.shape)

# prepare data (rows, cols, data)
rows, cols = exp_pmi.nonzero()
data = exp_pmi.data

# enumerate function for printing status
for idx in indices:
    # apply logarithm
    pmi_dok[rows[idx], cols[idx]] = np.log(data[idx])
{% endhighlight %}

질문의 '뭐먹', '어디'라는 단어와 답변의 '피자', '치킨', '지하철'의 pmi value 를 확인합니다. '뭐먹느냐'는 질문에는 음식 단어가 상관성이 높고, '어디냐'는 질문의 답변에서는 '지하철'의 높습니다. ('어디', '치킨') 의 상관성이 조금은 있는 걸로 봐서 ('어디야?', '치킨먹고있어') 와 같은 질문 - 답변이 데이터에 있었는지도 모르겠네요.

{% highlight python %}
for term_1 in ['뭐먹', '어디']:
    for term_2 in ['피자', '치킨', '지하철']:
        term_1_idx = vocab2idx[term_1]
        term_2_idx = vocab2idx[term_2]
        pmi_12 = pmi_dok[term_1_idx, term_2_idx]
        print('PMI({}, {}) = {}'.format(term_1, term_2, pmi_12))

# PMI(뭐먹, 피자) = 3.0861874233694397
# PMI(뭐먹, 치킨) = 3.8521326246077767
# PMI(뭐먹, 지하철) = 0.0
# PMI(어디, 피자) = 0.0
# PMI(어디, 치킨) = 0.4396905736037917
# PMI(어디, 지하철) = 2.2598710194560514
{% endhighlight %}


## Implemented function (soynlp)

위 기능들을 정리하여 soynlp.word.pmi 에 구현하였습니다. Sparse matrix 에 대해서만 계산 기능을 제공합니다. 

{% highlight python %}
from soynlp.word import pmi
pmi_dok = pmi(x, verbose=True)
{% endhighlight %}

동일한 데이터에 대하여 동일한 결과를 얻음을 확인하였습니다.

{% highlight python %}
for term_1 in ['뭐먹', '어디']:
    for term_2 in ['피자', '치킨', '지하철']:
        term_1_idx = vocab2idx[term_1]
        term_2_idx = vocab2idx[term_2]
        pmi_12 = pmi_dok[term_1_idx, term_2_idx]
        print('PMI({}, {}) = {}'.format(term_1, term_2, pmi_12))

# PMI(뭐먹, 피자) = 3.0861874233694397
# PMI(뭐먹, 치킨) = 3.8521326246077767
# PMI(뭐먹, 지하철) = 0.0
# PMI(어디, 피자) = 0.0
# PMI(어디, 치킨) = 0.4396905736037917
# PMI(어디, 지하철) = 2.2598710194560514
{% endhighlight %}



[sparse_post]: {{ site.baseurl }}{% link _posts/2018-04-09-sparse_mtarix_handling.md %}