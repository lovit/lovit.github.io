---
title: Scipy sparse matrix handling
date: 2018-03-27 11:00:00
categories:
- nlp
tags:
- keyword
---


## What is Keyword?

키워드 추출이란 말은 매우 익숙한 말입니다. 하지만 키워드 추출은 매우 모호한 말입니다. 키워드의 정의는 분석의 목적이나 데이터의 특징에 따라 다릅니다. 그렇기 때문에 키워드 추출은 키워드에 대한 정의부터 시작해야 합니다. 한 문서에 많이 등장한 단어를 키워드로 정의하는 것은 매우 위험합니다. ‘-은, -는, -이, -가’와 같은 조사는 모든 문서에서 가장 많이 등장할 단어들입니다. 그보다는 어떤 문서집합을 연상시킬 수 있는 몇 개의 단어를 키워드로 정의하면 더 좋을 것 같습니다. 어떤 문서집합을 연상시키려면 해당 단어가 그 문서집합에서만 유독 많이 등장하여야 할 것입니다. 

키워드 추출 관련 연구들에서 키워드의 조건으로 언급되는 기준은 크게 두 가지 입니다. 첫번째는 saliency 입니다. 키워드는 그 대상을 대표한다는 의미이며, 해당 문서 집합에 자주 등장하여 coverage 가 높다는 의미입니다. 한 집합의 1 만개 문서 중 10 개에만 등장한 단어를 키워드로 선택하는 것은 옳지 않습니다. 두번째는 distinctiveness 혹은 discriminative power 입니다. 다른 집합과 구분이 되는 단어를 키워드로 선정해야 대표성이 있습니다. 

그러나 saliency 와 distinctiveness 는 약간의 역관계가 있습니다. 자주 등장한 단어는 여러 집합에 모두 등장할 가능성이 높습니다. 그렇기 때문에 distinctiveness 가 낮아질 수 있습니다. 키워드 추출을 위한 많은 알고리즘들은 이 두 기준을 모두 소화할 수 있는 index 를 만드려 합니다.

[Lasso regression 을 이용하여 keyword extraction][lasso_keyword] 을 할 수도 있습니다. Lasso regression 은 correlation 이 높은 단어는 키워드로 선택하지 않는 단점이 있습니다. 모든 문서에서 '버락'과 '오바마'가 늘 함께 등장한다면 lasso regression 입장에서는 두 단어를 모두 선택하면 cost 가 더 높아지기 때문에 한 단어만 선택합니다. 

이번 포스트에서는 lasso regression 처럼 여러 집합을 구분할 수 있는 단어를 찾는 방법을 제안합니다. 


## Term Proportion Ratio

키워드는 관점이 주어졌을 때, 그 관점에서 더 자주 등장하는 단어로 정의할 수 있습니다. 예를 들어 여름 철 평상시에 뉴스에서 ‘폭우’가 0.1% 등장하는데, 오늘의 뉴스에서 ‘폭우’가 1% 등장하였다면, ‘폭우’는 오늘 뉴스의 키워드입니다. 평상시의 단어 비율과 오늘의 단어 비율의 배율를 수치로 이용할 수도 있습니다. 하지만 scale 이 $$[0, \inf ]$$ 이기 때문에 해석이 어렵습니다. 또한 mutual information 처럼 infrequent words 일 때는 배율이 민감하게 반응합니다.

아래처럼 키워드 점수를 설계할 수 있습니다. 

$$score(w) = \frac{P(w \vert D_t )}{P(w \vert D_t ) + P(w \vert D_r )}$$

- $$P(w \vert D_t )$$: target documents 에서 단어 w 가 출현한 비율
- $$P(w \vert D_r )$$: reference documents 에서 단어 w 가 출현한 비율

Target documents 에서만 등장한 단어는 $$score = 1.0$$ 입니다. Reference documents 에서만 등장한 단어의 점수는 0 입니다. 점수가 0.5 라면 집합에 관계없이 늘 비슷한 비율로 등장함을 의미합니다. 점수의 크기가 해석력을 지닙니다. 또한 infrequent words 라 하여도 scale 이 심하게 변동하지 않습니다. 


## Develop term proportion ratio based keyword extractor (Python)

Term freuqency matrix 가 주어졌을 때 키워드를 추출하는 알고리즘을 Python 으로 만들어봅니다. 

Vectorizer 를 통하여 term frequency matrix 를 만듭니다. word idx 를 str 로 만들기 위한 int2word() 함수도 만듭니다.

{% highlight python %}
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(corpus) # term frequency matrix
_word2int = vectorizer.vocabulary_
_int2word = [word for word, _ in sorted(_word2int.items(), key=lambda x:x[1])]

def word2int(word):
    return _word2int.get(word, -1)

def int2word(idx):
    if 0 <= idx < len(_int2word):
        return _int2word[idx]
    return None

word2int('아이오아이')
{% endhighlight %}

'아이오아이'라는 단어가 포함된 문서 집합을 positive_documents 로, 그렇지 않은 문서 집합을 negative_documents 로 설정합니다. 97 개의 positive documents 가 있습니다. 이 문서 집합의 키워드를 추출해 봅니다.

{% highlight python %}ㅊ
word = '아이오아이'
word_idx = word2int(word)
positive_documents = x[:,word_idx].nonzero()[0].tolist()
negative_documents = [i for i in range(x.shape[0]) if not (i in positive_documents)]

print('n pos = {}, n neg = {}'.format(len(positive_documents), len(negative_documents)))
# n pos = 97, n neg = 29994
{% endhighlight %}

sparse matrix x에서 sum()을 하면 모든 값의 합이 구해집니다. sum(axis=0)을 하면 rows가 하나의 row로 합쳐지는 sum이며, sum(axis=1)을 하면 columns가 하나의 column으로 합쳐지는 sum입니다. 우리의 x는 (document by term) matrix이기 때문에 row sum을 하면 모든 문서에서의 단어들의 빈도수 합이 구해집니다. 그래서 (30091 by 9774)의 term frequency matrix가 9774 차원의 term frequency vector가 되었음을 확인할 수 있습니다. 

{% highlight python %}
x.shape # (30091, 9774)
x.sum(axis=0).shape # (1, 9774)
{% endhighlight %}

scipy.sparse 의 matrix는 slicing이 가능합니다. positive_documents를 list 형식으로 만들었습니다. 이 list를 x에 넣어서 x[list,:] 을 실행하면 list에 해당하는 모든 row들을 잘라서 submatrix를 만듭니다. positive_documents, 즉 '아이오아이'라는 단어가 들어간 문서들만을 잘라내어 submatrix를 만든 뒤, 이를 row sum (= sum(axis=0))을 하였습니다. '아이오아이'라는 단어가 들어간 문서의 단어 빈도수가 만들어집니다. 이를 list로 만든 뒤, 출력해보면 다음과 같이 term frequency list가 만들어졌음을 볼 수 있습니다. 길이는 단어의 개수와 같습니다. 

{% highlight python %}
positive_proportion = x[positive_documents,:].sum(axis=0)
positive_proportion = positive_proportion.tolist()[0]
{% endhighlight %}

총 합을 _sum 이라는 변수로 만든 뒤, 모든 빈도수를 이 _sum으로 나누어주면 positive documents, 즉 '아이오아이'가 포함된 문서에서의 단어들의 출현 비율이 만들어집니다. 

{% highlight python %}
_sum = sum(positive_proportion)
positive_proportion = [v/_sum for v in positive_proportion]
{% endhighlight %}

이 과정을 반복할 것이니 to_proportion(documents_list) 라는 함수로 만들어 둡니다. 

positive proportion은 '아이오아이'가 포함된 문서에서의 단어 출현 비율, negative proportion은 '아이오아이'가 포함되지 않은 문서에서의 단어 출현 비율입니다. 

{% highlight python %}
def to_proportion(documents_list):
    proportion = x[documents_list,:].sum(axis=0)
    proportion = proportion.tolist()[0]
    _sum = sum(proportion)
    proportion = [v/_sum for v in proportion]
    return proportion

positive_proportion = to_proportion(positive_documents)
negative_proportion = to_proportion(negative_documents)
{% endhighlight %}

상대적 출현 비율은 모든 단어들에 대하여 p / (p+n) 을 계산하면 됩니다. p는 한 단어의 positive proportion의 값이며, n은 그 단어의 negative proportion의 값입니다. 

{% highlight python %}
def proportion_ratio(pos, neg):
    assert len(pos) == len(neg)
    ratio = [0 if (p+n == 0) else (p / (p+n)) for p,n in zip(pos, neg)]
    return ratio

keyword_score = proportion_ratio(positive_proportion, negative_proportion)
{% endhighlight %}

이제 proportion ratio가 높은 단어들을 찾아봅니다. enumerate를 이용하면 점수가 높은 단어의 index와 그 점수를 (단어, 점수) pair로 만들 수 있습니다. 이를 점수 기준으로 정렬하면 점수 순 정렬이 됩니다. 

{% highlight python %}
sorted(enumerate(keyword_score), key=lambda x:-x[1])[:30]

# [(4309, 1.0),
#  (5537, 1.0),
#  (2308, 0.9999606090273322),
#  (5333, 0.998991194480233),
#  (6145, 0.9989863725521622),
#  (921, 0.9982710816259988),
#  (2466, 0.9981432513884275),
#  (5880, 0.9978307775631691),
#  (4815, 0.9978210421997837),
#  (3682, 0.9975836317984187),
#  ...
{% endhighlight %}

앞서 term frequency vector를 만들었습니다. 이도 list로 만들어 둡니다. 키워드/연관어를 추출할 때, 최소 빈도수를 설정하기 위해서입니다. 

{% highlight python %}
term_frequency = x.sum(axis=0).tolist()[0]
{% endhighlight %}

이 과정을 proportion ratio keyword로 감싸서 함수로 만들어 둡니다. min count와 단어를 입력받도록 합니다. 

term frequency matrix 에 포함되지 않은 단어면 키워드분석을 하지 않습니다. 

	word_idx = word2int(word)
		if word_idx == -1:
			return None
            
min count cutting을 통해서 최소 빈도수 이상인 단어들만 available terms로 만들어 둡니다. 

	term_frequency = x.sum(axis=0).tolist()[0]
	available_terms = {term:count for term, count in enumerate(term_frequency) if count >= min_count}

그 뒤 positive_documents / negative_documents를 선택하고, positive_proportion / negative_proportion 를 계산한 뒤, proportion_ratio를 계산합니다. 

	positive_documents = x[:,word_idx].nonzero()[0].tolist()
	positive_proportion = to_proportion(positive_documents)
	...
	keyword_score = proportion_ratio(positive_proportion, negative_proportion)

최소빈도수 이상으로 등장한 단어만을 keyword로 남겨두는 filtering을 합니다. filter 함수를 써도 좋습니다.

	keyword_score = [(term, score) for term, score in keyword_score if term in available_terms]

word index로 표현되어 있는 keyword_score = [(idx, score), ... ]를 [(word, score), ...]로 바꿔줍니다. 

	keyword_score = [(int2word(term), score) for term, score in keyword_score]

{% highlight python %}
def proportion_ratio_keyword(word, min_count=10):
    word_idx = word2int(word)
    if word_idx == -1:
        return None
    
    term_frequency = x.sum(axis=0).tolist()[0]
    available_terms = {term:count for term, count in enumerate(term_frequency) if count >= min_count}
    
    positive_documents = x[:,word_idx].nonzero()[0].tolist()
    negative_documents = [i for i in range(x.shape[0]) if not (i in positive_documents)]
    
    positive_proportion = to_proportion(positive_documents)
    negative_proportion = to_proportion(negative_documents)
    
    keyword_score = proportion_ratio(positive_proportion, negative_proportion)
    keyword_score = sorted(enumerate(keyword_score), key=lambda x:-x[1])
    keyword_score = [(term, score) for term, score in keyword_score if term in available_terms]
    keyword_score = [(int2word(term), score) for term, score in keyword_score]
    
    return keyword_score
{% endhighlight %}

직접 만든 함수를 이용하여 30 번 이상 등장한 단어에 대하여 '아이오아이' 관련 문서의 키워드를 선택합니다. 

{% highlight python %}
from pprint import pprint

keywords = proportion_ratio_keyword(word='아이오아이', min_count=30)
pprint(keywords[:30])
{% endhighlight %}

음악방송과 관련된 단어들과 다른 아이돌 그룹, 멤버들의 이름이 키워드로 선택됩니다. 

	[('빅브레인', 1.0),
	 ('아이오아이', 1.0),
	 ('너무너무너무', 0.9999606090273322),
	 ('신용재', 0.998991194480233),
	 ('오블리스', 0.9989863725521622),
	 ('갓세븐', 0.9982710816259988),
	 ('다비치', 0.9981432513884275),
	 ('엠카운트다운', 0.9978307775631691),
	 ('세븐', 0.9978210421997837),
	 ('박진영', 0.9975836317984187),
	 ('완전체', 0.9973594469004617),
	 ('선의', 0.9963128215975839),
	 ('산들', 0.9958319893090414),
	 ('중독성', 0.9948644479894773),
	 ('프로듀스101', 0.9946890725030576),
	 ('열창', 0.9938200380735884),
	 ('펜타곤', 0.9934422266805437),
	 ('잠깐', 0.9929667382454291),
	 ('상큼', 0.9909673401797572),
	 ('소녀들', 0.9908127932033489),
	 ('엠넷', 0.9907514986652862),
	 ('걸크러쉬', 0.99017203825805),
	 ('일산동구', 0.9884164745297143),
	 ('음악방송', 0.9881439461828352),
	 ('사나', 0.9880894585465715),
	 ('선율', 0.9875086307696628),
	 ('타이틀곡', 0.9869906112674688),
	 ('코드', 0.9867835556082788),
	 ('본명', 0.98596911773225),
	 ('깜찍', 0.9853881990008125)]

2016-10-20 은 최순실-박근혜 게이트의 보도가 시작되던 시기입니다. 이 시기의 '최순실' 관련 뉴스의 키워드입니다. 

{% highlight python %}
pprint(proportion_ratio_keyword(word='최순실', min_count=100)[:30])
{% endhighlight %}

	[('최순실', 1.0),
	 ('게이트', 0.9981018054860111),
	 ('정유라', 0.9949748004314443),
	 ('연설문', 0.9900718598746623),
	 ('모녀', 0.9875768099004291),
	 ('승마', 0.9872307511905503),
	 ('개명', 0.986641026457457),
	 ('비선', 0.985018930232134),
	 ('더블루케이', 0.9838995868457685),
	 ('실세', 0.9823312201845503),
	 ('스포츠재단', 0.980984809482314),
	 ('최씨', 0.9802224596736517),
	 ('최경희', 0.980172024643097),
	 ('비덱', 0.9794924174652362),
	 ('이화여대', 0.9792281858488985),
	 ('특혜', 0.9775213977582151),
	 ('미르재단', 0.9774516345256685),
	 ('의혹들', 0.977198367560925),
	 ('학점', 0.976567846725211),
	 ('비선실세', 0.9747618098586102),
	 ('이대', 0.9713049096885505),
	 ('미르', 0.9697354303371427),
	 ('재단', 0.9665692895878129),
	 ('정유라씨', 0.9651208193465403),
	 ('엄정', 0.9635099910913556),
	 ('차은택', 0.9630949366283257),
	 ('이화', 0.962975945484486),
	 ('국정조사', 0.9614360445588696),
	 ('사퇴', 0.961117249105005),
	 ('의혹', 0.9610013059869946)]


## Related words is keyword? 

특정 단어가 포함된 문서 집합의 키워드를 선택하니 연관어들이 등장합니다. 연관어는 대부분 co-occurrence 기반으로 추출합니다. Pointwise Mutual Information (PMI) 이 co-occurrence 기반으로 연관어를 선택하는 대표적인 방법입니다. 

앞서 직접 만든 방법 역시 co-occurrence 를 측정하는 방법입니다. 기준 단어가 등장한 문서 집합에만 등장할 경우에 score = 1.0 입니다. 연관어 추출을 위해서 위의 방법을 이용할 수 있습니다. 


## Clustering labeling with Keyword extraction

이전의 [clustering labeling post][clustering_labeling] 이 지금의 방법으로 만들어졌습니다. 군집은 단어 기준으로 묶은 문서 집합이 아닙니다. 하지만 군집의 키워드를 선택하면 군집의 레이블을 달 수 있습니다. 자세한 내용은 위 포스트를 보시기 바랍니다. 


## Packages (soykeyword)

위의 작업을 packaging 하였습니다. 설치는 pip install 을 이용할 수 있습니다. 현재 버전은 0.0.12 입니다.

	pip install soykeyword

다른 작업을 위해서도 sparse matrix 를 만들어두는 경우가 많습니다. Sparse matrix 와 word list 를 입력하면 위의 기능을 이용할 수 있습니다. 학습은 train() 함수에 matrix 와 list 를 입력합니다.

{% highlight python %}
from soykeyword.proportion import MatrixbasedKeywordExtractor

matrixbased_extractor = MatrixbasedKeywordExtractor(
    min_tf=20, 
    min_df=2,
    verbose=True)

matrixbased_extractor.train(x, index2word)
{% endhighlight %}

연관어는 extract_from_word() 함수를 이용합니다. min_score 는 keyword score 의 threshold 이며, min_count 는 문서 집합 전체에서의 등장 빈도 기준 최소 빈도수입니다. 

{% highlight python %}
keywords = matrixbased_extractor.extract_from_word('아이오아이', min_score=0.8, min_count=100)
keywords[:10]
{% endhighlight %}

KeywordScore 는 collections.namedtuple 입니다. 

	[KeywordScore(word='아이오아이', frequency=270, score=1.0),
	 KeywordScore(word='엠카운트다운', frequency=221, score=0.9978307775631691),
	 KeywordScore(word='펜타곤', frequency=104, score=0.9934422266805437),
	 KeywordScore(word='잠깐', frequency=162, score=0.9929667382454291),
	 KeywordScore(word='엠넷', frequency=125, score=0.9907514986652862),
	 KeywordScore(word='걸크러쉬', frequency=111, score=0.99017203825805),
	 KeywordScore(word='타이틀곡', frequency=311, score=0.9869906112674688),
	 KeywordScore(word='코드', frequency=105, score=0.9867835556082788),
	 KeywordScore(word='본명', frequency=105, score=0.98596911773225),
	 KeywordScore(word='엑스', frequency=101, score=0.9847950780631249)]

문서의 idx list 를 이용하여 문서 집합의 키워드를 추출할 때에는 extract_from_docs() 함수를 이용합니다. 

{% highlight python %}
matrixbased_extractor.extract_from_docs(documents, min_score=0.8, min_count=100)
{% endhighlight %}


## References
- Chuang, J., Manning, C. D., & Heer, J. (2012, May). [Termite: Visualization techniques for assessing textual topic models.][termite] In Proceedings of the international working conference on advanced visual interfaces (pp. 74-77). ACM.
- Mei, Q., Shen, X., & Zhai, C. (2007, August). [Automatic labeling of multinomial topic models.][meilda] In Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 490-499). ACM.
- Sievert, C., & Shirley, K. (2014). [LDAvis: A method for visualizing and interpreting topics.][ldavis] In Proceedings of the workshop on interactive language learning, visualization, and interfaces (pp. 63-70).

[meilda]: http://sifaka.cs.uiuc.edu/xshen/research_files/sigkdd2007_clustering_labeling.pdf
[ldavis]: http://www.aclweb.org/anthology/W14-3110
[termite]: http://www.infomus.org/Events/proceedings/AVI2012/CHAPTER%201/p74-chuang.pdf
[lasso_keyword]: {{ site.baseurl }}{% link _posts/2018-03-24-lasso_keyword.md %}
[clustering_labeling]: {{ site.baseurl }}{% link _posts/2018-03-21-kmeans_cluster_labeling.md %}