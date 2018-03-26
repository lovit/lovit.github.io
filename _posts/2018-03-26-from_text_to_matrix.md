---
title: From text to term frequency matrix (KoNLPy)
date: 2018-03-26 11:00:00
categories:
- nlp
tags:
- preprocessing
---

텍스트마이닝을 수행하기 위해서는 텍스트 형식의 데이터를 머신러닝 알고리즘이 이해할 수 있는 벡터 형식으로 변환해야 합니다. 문서를 벡터로 표현하는 방식은 크게 두 가지로 나뉘어집니다. One hot representation 혹은 Bag of words model 로 불리는 방법은 한 문서 $$i$$에 단어 $$j$$ 가 몇 번 등장했는지를 표현하는 term frequency vector 로 문서를 표현합니다. 우리는 KoNLPy 를 이용하여 한국어 텍스트 문서를 term frequency matrix 로 변환하는 과정에 대하여 알아봅니다.

## Ways of document representation

텍스트마이닝을 수행하기 위해서는 텍스트 형식의 데이터를 머신러닝 알고리즘이 이해할 수 있는 벡터 형식으로 변환해야 합니다. 문서를 벡터로 표현하는 방식은 크게 두 가지로 나뉘어집니다. One hot representation 혹은 Bag of words model 로 불리는 방법은 한 문서 $$i$$에 단어 $$j$$ 가 몇 번 등장했는지를 표현하는 term frequency vector 로 문서를 표현합니다. 이 방식은 한 문서에 어떤 단어들이 등장하였는지를 직접적으로 표현하기 때문에 벡터로부터 의미를 직접적으로 이해할 수 있습니다. 이와 반대로 Doc2Vec 으로 잘 알려진 distributed representation 은 벡터로부터 직접적으로 문서의 의미를 이해할 수는 없습니다. 하지만 비슷한 문서는 비슷한 벡터를 지니기 때문에 의미적으로 유사한 문서를 찾기에 용이합니다. Distributed representation 은 [Word/Document embedding][word2vec_post] 에서 이야기하겠습니다. 우리는 KoNLPy 를 이용하여 한국어 텍스트 문서를 term frequency matrix 로 변환하는 과정에 대하여 알아봅니다.

## KoNLPy

형태소분석은 주어진 문장에 대하여 각 문장을 구성하는 형태소들을 분해/인식하는 과정입니다. 물론 문장 속의 띄어쓰기로 나눠서 인식할 수도 있습니다. 하지만 한국어 텍스트 분석에서는 그리 좋은 방법은 아닙니다. 그 이유에 대해서는 [토크나이저/품사판별기][pos_and_oov] 포스트에서 이야기합니다. 

[KoNLPy][konlpy] 는 다양한 언어 (C++, Scala, Java)로 구현되어 있는 형태소분석기들을 Python 환경에서 이용할 수 있도록 도와줍니다. 특히 분석기들마다 서로 다른 인터페이스를 통일하였기 때문에 매우 편리합니다. KoNLPy (ver 0.4.4) 에는 Hannanum, Kkma, Mecab, Twitter 네 가지 형태소 분석기가 포함되어 있습니다. 문장을 pos() 에 입력하면 문장을 구성하는 형태소들을 확인할 수 있습니다. 특히 '테스트문장' 같은 복합 명사는 단일 명사들로 분해됩니다.

{% highlight python %}
from konlpy.tag import Twitter

twitter = Twitter()
twitter.pos('이건 테스트문장 입니다.')
# [('이건', 'Noun'), ('테스트', 'Noun'), ('문장', 'Noun'), ('입니', 'Adjective'), ('다', 'Eomi'), ('.', 'Punctuation')]
{% endhighlight %}

사실 다른 품사들 보다도 명사들만을 선택하여 데이터 분석을 하는 경우가 많습니다. 이를 위해서 명사만 선택하고 싶다면 nouns() 함수를 이용할 수 있습니다. 

{% highlight python %}
twitter.nouns('이건 테스트문장 입니다.')
# ['이건', '테스트', '문장']
{% endhighlight %}

그러나 형태소분석기들마다 서로 다른 체계의 형태소를 학습에 이용하였기 때문에 tag 가 다를 수 있습니다. Twitter 에서는 ['이건', '테스트', '문장'] 을 모두 Noun 으로 표기하지만, 꼬꼬마 형태소분석기에서는 '이건 / 대명사 (NNP)', '테스트 / 일반명사 (NNG)' 로 표기합니다. 또한 '입니다'의 경우 Twitter 에서는 '입니/형용사 어근' 으로 분해하지만, 꼬꼬마에서는 '이/지정사 + ㅂ니다/평서형 종결 어미'로 인식합니다. 

{% highlight python %}
from konlpy.tag import Kkma

kkma = Kkma()
kkma.nouns('이건 테스트문장 입니다.')
# [('이건', 'NNP'), ('테스트', 'NNG'), ('문장', 'NNG'), ('이', 'VCP'), ('ㅂ니다', 'EFN'), ('.', 'SF')]
{% endhighlight %}

각 형태소 분석기마다 어떤 체계를 이용하는지 확인하기 위해서는 .tagset 을 이용하면 됩니다. 

{% highlight python %}
print(twitter.tagset)
{% endhighlight %}

    {'Adjective': '형용사',
     'Adverb': '부사',
     'Alpha': '알파벳',
     'Conjunction': '접속사',
     'Determiner': '관형사',
     'Eomi': '어미',
     'Exclamation': '감탄사',
     'Foreign': '외국어, 한자 및 기타기호',
     'Hashtag': '트위터 해쉬태그',
     'Josa': '조사',
     'KoreanParticle': '(ex: ㅋㅋ)',
     'Noun': '명사',
     'Number': '숫자',
     'PreEomi': '선어말어미',
     'Punctuation': '구두점',
     'ScreenName': '트위터 아이디',
     'Suffix': '접미사',
     'Unknown': '미등록어',
     'Verb': '동사'}

형태소분석기마다 서로 다른 체계를 이용하는 것은 각 엔진을 이용할 목적이 다르거나 학습에 이용하였던 학습데이터 말뭉치의 품사체계를 따랐기 때문입니다. 각 형태소분석기 별로 pos() 의 결과를 이용하실 때에는 사용자가 tag를 반드시 확인하셔야 합니다. 

저는 개인적으로 형태소분석의 작동 방식과 속도 때문에 Twitter 를 선호합니다. [Komoran][komoran] 도 꾸준히 업데이트가 되고 있으니 Java 를 이용할 때에는 유용합니다. 

## Corpus class

텍스트 분석을 할 경우에 분석의 단위가 "단어 / 문장 / 문서"로 달라질 수 있습니다. 매번 문장과 문서를 구분하는 것이 번거러워 corpus 를 다루는 Python class 하나를 만들어 이용하고 있습니다. 저는 데이터를 정리할 때 한 줄에 하나의 문서가 저장되도록 합니다. 그렇다면 하나의 문서 안에서 문장을 구분하기 어려워지는데, 문장의 구분기호는 두 칸 띄어쓰기를 이용합니다. 그래서 DoublespaceLineCorpus 라는 이름으로 class 를 만들었습니다. 그리고 iter_sent 라는 argument 를 추가하여 iteration 을 문서 단위로 수행할 것인지 문장 단위로 수행할 것인지 조절할 수 있도록 하였습니다. 

{% highlight python %}
class DoublespaceLineCorpus:    
    def __init__(self, corpus_fname, iter_sent = False):
        self.corpus_fname = corpus_fname
        self.iter_sent = iter_sent
            
    def __iter__(self):
        with open(self.corpus_fname, encoding='utf-8') as f:
            for doc in f:
                if not self.iter_sent:
                    yield doc
                    continue
                for sent in doc.split('  '):
                    sent = sent.strip()
                    if not sent:
                        continue
                    yield sent
{% endhighlight %}

2016-10-20 의 뉴스데이터를 이용하여 테스트를 해봅니다. iter_sent = False, True 로 조절함에 따라 for loop 의 iteration 의 단위가 달라집니다. 

{% highlight python %}
corpus = DoublespaceLineCorpus(corpus_path, iter_sent=False)

for n_doc, doc in enumerate(corpus):
    continue
print('num docs = {}'.format(n_doc+1))
# num docs = 30091

corpus.iter_sent = True
for n_sent, sent in enumerate(corpus):
    continue
print('num sent = {}'.format(n_sent+1))
# num sent = 223357
{% endhighlight %}

length 나 몇 개의 samples 만 iteration 에 이용하는 기능을 추가하여 [soynlp][soynlp] 에 완성된 DoublespaceLineCorpus 를 올려두었습니다. 

{% highlight python %}
from soynlp import DoublespaceLineCorpus

corpus = DoublespaceLineCorpus(corpus_path, iter_sent=False)
print(len(corpus))
# 30091

corpus.iter_sent = True
print(len(corpus))
# 223357
{% endhighlight %}

만약 100 개 문서에 대해서만 iteration 을 수행하려면 num_doc=100 으로 설정하면 됩니다. 100 개 문서에는 1087 개의 문장이 포함되어 있습니다.

{% highlight python %}
corpus = DoublespaceLineCorpus(corpus_path, num_doc=100, iter_sent=False)
for n_doc, doc in enumerate(corpus):
    continue
print('num docs = {}'.format(n_doc+1))
# num docs = 100

corpus.iter_sent = True
print(len(corpus))
# 1087
{% endhighlight %}

## Scikit-learn vectorizer

Scikit-learn 에서는 텍스트를 sparse matrix 로 변환하기 위한 기능들을 제공합니다. CountVectorizer 는 term frequency matrix 를 만들어줍니다. CountVectorizer 에는 유용한 몇 가지 옵션들이 있습니다. 아래의 예시들은 모두 기본값입니다. 각각의 의미에 대하여 알아봅니다. 

CountVectorizer 에 입력되는 min_df, max_df 는 특정 단어를 포함한 문서의 비율입니다. 만약 min_df=0.02 로 설정한다면, 100 개의 문서에서 1 번만 등장한 (0.01) 단어는 term frequency matrix 에서 제외됩니다. 마찬가지로 max_df=0.5 로 설정한다면, 51 개 이상의 문서에서 등장하는 단어들은 제외됩니다. ngram_range 는 term frequency matrix 에서 이용할 n-gram 의 범위입니다. 기본은 unigram 으로 되어 있으며, (1, 3) 으로 설정하면 uni / bi / trigram 을 모두 이용한다는 의미입니다. lowercaes=True 는 영어의 경우 모든 글자를 소문자로 변환한다는 의미입니다. 한국어 텍스트를 sparse matrix 로 만들 때 가장 중요한 것은 tokenizer 입니다. lambda x:x.split() 이 기본입니다. 띄어쓰기 기준으로 단어를 나눕니다. 

fit() 혹은 fit_transform() 에 입력되는 데이터의 형식은 list of str (like) 입니다. 우리가 만든 DoublespaceLineCorpus 역시 list of str 처럼 iteration loop 을 돌며 str 을 yield 하기 때문에 이용할 수 있습니다. 

{% highlight python %}
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(
    min_df=0,
    max_df=1,
    ngram_range=(1,1),
    lowercase=True,
    tokenizer=lambda x:x.split())

corpus.iter_sent=False
X = vectorizer.fit_transform(corpus)
print(X.shape) # (100, 7624)
{% endhighlight %}

만약 tf-idf 형식으로 weight 를 변환한다면 두 가지 방법이 있습니다. 첫번째는 TfidfVectorizer 를 이용합니다. Vectorizer 의 옵션은 동일합니다.

{% highlight python %}
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    min_df=0,
    max_df=1,
    ngram_range=(1,1),
    lowercase=True,
    tokenizer=lambda x:x.split())

X = vectorizer.fit_transform(corpus)
{% endhighlight %}

혹은 term frequency matrix 를 tf-idf 형식으로 변환합니다. 둘 모두 결과는 같습니다. 저는 머신러닝 알고리즘 학습 이후에 각 문서에 어떤 말들이 등장하였는지 정확히 살펴보기 위하여 CountVectorizer 를 이용하여 term frequency matrix 를 먼저 만들어 저장을 해두고, 필요시에 따라 tf-idf 로 변환하여 이용합니다.

{% highlight python %}
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

transformer = TfidfTransformer()
X = transformer.fit_transform(X)
{% endhighlight %}

Vectorizer 에는 sparse matrix 의 각 column 이 어떤 단어에 해당하는지에 대한 index 가 저장되어 있습니다. Vectorizer 의 fit() 함수가 하는 일입니다. .vocabulary_ 에는 {str:int} 형식으로 각 단어가 어떤 idx 에 해당하는지를 나타내는 dict 가 저장되어 있습니다. 

{% highlight python %}
vectorizer.vocabulary_
{% endhighlight %}

    {'19': 129,
     '1990': 149,
     '52': 478,
     '22': 265,
     '오패산터널': 5944,
     '총격전': 8150,
     ...
    }

이로부터 0, 1, 2, ...  순서대로 각 idx 가 어떤 단어인지를 저장하는 list of str 을 만들 수 있습니다. 우리가 띄어쓰기 기준으로 tokenizer 를 이용하였기 때문에 비슷한 어절들이 모두 다른 단어로 학습되었습니다.

{% highlight python %}
idx2vocab = [vocab for vocab, idx in sorted(vectorizer.vocabulary_.items(), key=lambda x:x[1])]
print(idx2vocab[5537:5542])
# ['어려운', '어려움을', '어려움이', '어려워', '어려웠다']
{% endhighlight %}

Vectorizer 를 저장하거나, 학습된 vectorizer 를 불러올 때는 pickling 을 이용할 수 있습니다. 

{% highlight python %}
import pickle
with open('./vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
{% endhighlight %}

## Matrix I/O

Vectorizer 에 의하여 만들어지는 term frequency matrix 는 scipy.sparse.csr.csr_matrix 형식입니다. Sparse matrix 는 다양한 형식이 있습니다. 이에 대해서는 [sparse matrix handling][sparse_matrix] 에서 이야기합니다. 

{% highlight python %}
X = vectorizer.fit_transform(corpus)

print(type(X))
# scipy.sparse.csr.csr_matrix
{% endhighlight %}

학습된 matrix 는 scipy 의 io 를 이용하여 저장합니다. 

{% highlight python %}
from scipy.io import mmwrite

mtx_path = './x.mtx'
mmwrite(mtx_path, X)
{% endhighlight %}

저장된 matrix 를 읽어올 때도 비슷합니다. 하지만, mmread() 를 이용하여 읽은 matrix 의 형식은 COO matrix 입니다. 만약 csr 로 변환하고 싶다면 tocsr() 을 이용합니다. 
{% highlight python %}
from scipy.io import mmread

X = mmread(mtx_path)
print(type(X))
# scipy.sparse.coo.coo_matrix

X = X.tocsr()
print(type(X))
# scipy.sparse.csc.csc_matrix
{% endhighlight %}

## TF vs TF-IDF

tf-idf 가 tf 보다 더 좋은 표현방법이라 말하기도 하지만, 그것은 틀린 말입니다. tf-idf 가 항상 tf 보다 좋은 방법은 아닙니다. 먼저 tf-idf 가 어떤 목적으로 만들어졌는지를 이해해야 합니다. 처음 tf-idf 는 검색 엔진 (information retrieval) 분야에서 제안되었습니다. 문서와 query 의 관계를 정의하기 위해서였습니다. ['아이오아이', '콘서트'] 라는 query가 입력되었을 때 상관있는 문서의 후보들을 만들려면 '아이오아이'나 '콘서트' 단어가 포함된 문서를 먼저 가져와야 합니다. term frequency vector 로 문서와 query 를 모두 표현한 다음, cosine 과 같은 방법을 이용하여 두 벡터간의 유사도를 계산합니다. 팁으로, cosine 은 포함된 단어 수가 적은 벡터 기준으로 for loop 을 돌면 좋습니다. 또한 모든 벡터가 unit vector 라면 norm 으로 나누어 주는 부분도 필요없습니다. 

$$cos(q, d) = \frac{\sum_{v \in q} w(q, v) * w(d, v)}{\sqrt{\sum_{v \in q} w(q, v)^2}\sqrt{\sum_{v \in d} w(d, v)^2}}$$

결국 한 단어 기준으로 두 벡터의 weight 를 곱한 만큼 두 벡터 (query, 문서)가 유사하다는 의미입니다. 하지만 어떤 단어는 그리 중요하게 여기지 않아도 되는 단어가 있습니다. '-은, -는, -의' 와 같은 단어는 많은 문서에서 등장하기 때문에 '없는 단어 취급'을 해도 좋습니다. 이들을 stop words 라 합니다. 검색에 딱히 도움은 되지 않는 단어입니다. 하지만 이들은 자주 등장하기 때문에 term frequency + cosine 으로 벡터 간 유사도를 계산하였다가는 ['아이오아이', '의', '콘서트'] 에서 '-의'의 영향력이 너무 커집니다. 물론 stop words 를 term frequency matrix 를 만들 때 제거할 수도 있습니다. Scikit-learn 의 vectorizer 에서도 stop words 기능을 제공하고 있습니다. 

{% highlight python %}
CountVectorizer(stop_words=None)
{% endhighlight %}

하지만 도메인마다 stop words 가 다르기도 합니다. 대신에 많은 문서에서 등장하는 단어는 덜 중요한 단어로 그 weight 를 줄이는 것도 벡터 간 유사도를 cosine 으로 이용할 때 사용할 수 있는 방법입니다. Vectorizer 에 존재는 할 수 있도록 하되, 힘을 쭉 빼버리는 겁니다. 문서 전체 집합을 $$D$$, 문서 전체 집합의 크기 (문서 개수)를 $$N$$ 이라 할 때, tf-idf 는 다음처럼 정의할 수 있습니다. $$df(D, w)$$ 는 문서 집합 $$D$$ 에서 단어 $$w$$ 가 등장한 문서의 개수입니다. 여기서 $$log \left( \frac{N}{df(D, w)} \right)$$ 에 주목해야 합니다. 만약 한 단어가 모든 문서에 등장하였다면 $$log(1) = 0$$ 이 되어 해당 단어는 유사도 계산에 영향력을 미치지 않습니다. 

$$tf-idf(d, w) = tf(d, w) \times log \left( \frac{N}{df(D, w)} \right)$$

조금씩 tf-idf 를 다르게 정의하기도 합니다만, 공통된 철학은 **문서 집합에서 자주 등장하는 단어의 영향력을 줄인다**입니다. 

$$tf-idf(d, w) = \frac{tf(d, w)}{log\left(1 + \frac{N}{df(D, w)}\right)}$$

Term frequency vector 의 weight 를 tf-idf 로 변환한 다음에 cosine 을 동일하게 적용합니다.

$$cos(q, d) = \frac{\sum_{v \in q} tf-idf(q, v) * tf-idf(d, v)}{\sqrt{\sum_{v \in q} tf-idf(q, v)^2}\sqrt{\sum_{v \in d} tf-idf(d, v)^2}}$$

그런데 어떤 경우에는 문서 집합에서 자주 등장한다하여 중요하지 않는 단어가 될 수 있을 때 입니다. 어떤 상품에 대한 소비자들의 상품평을 모아둔 데이터를 예로 들어보면, '좋다, 싫다'와 같은 단어는 매우 빈번하게 등장할 것입니다. 우리가 sentiment classification 을 위한 판별기를 학습한다면, 이 단어들은 자주 등장하였음에도 불구하고 매우 중요한 terms 이 될 것입니다. 이때에는 tf-idf 로 변환하기 보다 tf 를 그대로 이용하는 것이 더 좋습니다. 

tf-idf 의 철학은 '문서 집합에서 자주 등장하는 단어의 중요도를 낮춘다'입니다. 이 철학이 필요한 경우에 tf-idf 를 적용해야 합니다. 그리고 이 외에도 다양한 term weighting 방법이 존재합니다. 각각의 문제와 데이터에 적합한 weighting 방법은 다를 수 있습니다. 이를 생각하고 weighting 을 디자인해야 합니다. 

## Scikit-learn vectorizer + KoNLPy

한국어 텍스트 전처리를 위하여 KoNLPy 의 형태소분석기들을 이용할 수 있음을 확인하였습니다. 그리고 scikit-learn 제공하는 Vectorizer 를 이용하면 bag of words model 형태의 matrix 를 얻을 수 있음도 알아보았습니다. 이번에는 이 둘을 함께 이용하는 방법에 대하여 이야기합니다. 

Vectorizer 의 argument 에는 tokenizer = lambda x:x.split() 이 있었습니다. 이 부분을 KoNLPy 의 pos() 함수로 대체합니다. pos() 의 return type 은 list of tuple 입니다. 이를 list of str 로 변환하는 함수를 추가하여 vectorizer 를 customize 합니다. 형태소분석기가 교체될 수 있기 때문에 MyTokenizer class 를 만들었습니다. init argument 로 tagger 를 입력받습니다. __call__() 을 구현하여 함수 형식으로 호출할 수 있도록 해두었습니다. [('이건', 'Noun'), ('테스트', 'Noun'), ...] 형식으로 출력되던 부분이 ['이건/Noun', '테스트/Noun', ...] 으로 바뀌었습니다. 

{% highlight python %}
from konlpy.tag import Twitter

class MyTokenizer:
    def __init__(self, tagger):
        self.tagger = tagger
    def __call__(self, sent):
        pos = self.tagger.pos(sent)
        pos = ['{}/{}'.format(word,tag) for word, tag in pos]
        return pos

my_tokenizer = MyTokenizer(Twitter())

sent = '이건테스트문장입니다.'
print(my_tokenizer(sent))
# ['이건/Noun', '테스트/Noun', '문장/Noun', '입니/Adjective', '다/Eomi', './Punctuation']
{% endhighlight %}

이 부분을 Vectorizer 의 tokenizer 에 입력합니다. 띄어쓰기 기준으로 토크나이징을 할 경우에 (100, 7624) 모양이던 X 가 (100, 6094) 로 바뀌었습니다. 

{% highlight python %}
from konlpy.tag import Twitter
from sklearn.feature_extraction.text import CountVectorizer

my_tokenizer = MyTokenizer(Twitter())
vectorizer = CountVectorizer(tokenizer = my_tokenizer)
X = vectorizer.fit_transform(corpus)

print(x.shape) # (100, 6094)
{% endhighlight %}

Vectorizer 에 학습된 vocab 을 살펴보면, 어절 단위가 아닌 'term/tag' 형식임을 알 수 있습니다. 

{% highlight python %}
idx2vocab = [vocab for vocab, idx in sorted(vectorizer.vocabulary_.items(), key=lambda x:x[1])]
print(idx2vocab[4300:4400])
{% endhighlight %}

    ['인도/Noun',
     '인도주의/Noun',
     '인들/Josa',
     '인력/Noun',
     '인류/Noun',
     '인방/Noun',
     '인사/Noun',
     '인사하고/Verb',
     ...
    ]

[word2vec_post]: {{ site.baseurl }}{ link _posts/2018-03-26_word_doc_embedding.md }
[konlpy]: http://konlpy.org/
[pos_and_oov]: {{ site.baseurl }}{ link _posts/2018-03-26-pos_and_oov.md }
[komoran]: https://github.com/shin285/KOMORAN
[soynlp]: https://github.com/lovit/soynlp/
[sparse_matrix]: {{ site.baseurl }}{ link _posts/2018-03-27-sparse_mtarix_handling.md }