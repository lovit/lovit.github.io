---
title: Unsupervised tokenizers in soynlp project, (Max Score Tokenizer, L-Tokenizer, RegexTokenizer)
date: 2018-04-09 23:00:00
categories:
- nlp
tags:
- preprocessing
---

soynlp 는 제가 작업하는 한국어 정보처리를 위한 비지도학습 기반 자연어처리 라이브러리 입니다. 현재 (ver 0.0.4) 세 가지 통계 기반 단어 추출 기법과 이를 이용하는 두 종류의 unsupervised tokenizers 를 제공합니다. WordExtractor 는 세 가지 단어 추출 기법인 Cohesion score, Branching Entropy, Accessor Variety 를 동시에 학습합니다. 학습된 통계 기반 단어 추출 기법들을 조합하여 Max Score Tokenizer 와 L-Tokenizer 를 만들 수 있습니다. 또한 규칙 기반으로 작동하는 Regex Tokenizer 도 제공합니다. 이에 대한 설명과 사용기입니다. 

## soynlp 

한국어 텍스트의 분석을 위해서 형태소 분석이나 품사 판별 같은 토크나이징 과정이 필요합니다. 정확한 단어로 토크나이징이 될 필요가 없다면 [Word Piece Model][wpm] 도 토크나이징에 이용될 수 있습니다. WPM 과 다르게 KoNLPy 에 등록된 형태소 분석기를 이용할 경우, [미등록단어 문제][pos_and_oov]가 발생할 수 있습니다. 

KoNLPy 의 트위터 한국어 분석기를 이용하여 다음의 예문의 형태소를 분석하면 미등록 단어가 잘못 인식됩니다. 
{% highlight python %}
from konlpy.tag import Twitter

twitter = Twitter()
twitter.pos('너무너무너무는 아이오아이의 노래입니다')
{% endhighlight %}

'너무너무너무'와 '아이오아이'는 하나의 단어이지만, 여러 개의 복합단어로 나뉘어졌습니다. 이를 해결하기 위하여 많은 형태소 분석기와 품사 판별기는 사용자 사전 추가 기능을 제공합니다. 

	[('너무', 'Noun'),
	 ('너무', 'Noun'),
	 ('너무', 'Noun'),
	 ('는', 'Josa'),
	 ('아이오', 'Noun'),
	 ('아이', 'Noun'),
	 ('의', 'Josa'),
	 ('노래', 'Noun'),
	 ('입니', 'Adjective'),
	 ('다', 'Eomi')]

WPM 은 일종의 데이터 압축 알고리즘으로, 정해진 크기의 벡터 공간으로 모든 문서를 표현하는데 그 목적이 있습니다. 미등록 단어 문제를 근본적으로 해결하는 방법은 아닙니다. [soynlp][soynlp] 는 미등록 단어 문제를 직접 풀기 위한 목적으로 시작된 한국어 처리 프로젝트입니다. 단어 추출 / 토크나이저 빌더 / 품사 추출 / 품사 판별기 빌더 / 전처리에 필요한 utils 를 제공하고 있습니다. 

soynlp 의 의미는 콩nlp 가 아닙니다. 오래전 스페인어를 공부하였을 때, gmail 계정을 soy.lovit 으로 만들었습니다. 'lovit' 은 자주 쓰는 저의 필명입니다. 'soy lovit'은 스페인어이며, 영어로 번역하면 'I am lovit' 입니다. 이후 프로젝트 이름 앞에 soy 를 붙였습니다. 굳이 번역하자면 "저는 NLP 입니다" 입니다.

오랫동안 작업 중이지만, 제대로 집중한 기간이 짧아 아직 완성되지 않은 프로젝트입니다. 현재 버전은 0.0.4 이며, 현재 그리고 있는 기능들이 모두 구현되면 0.1 로 명할 계획입니다. 그 전에는 API interface 가 바뀔 가능성이 있습니다. 


## Install soynlp

pip install 이 가능합니다. 

	pip install soynlp==0.0.4

git clone 으로 이용하셔도 됩니다. numpy 와 psutil 에만 dependency 가 있습니다. 


## Word Extractor

Word Extractor 에서 학습하는 [Cohesion score][cohesion], [Branching Entropy, Accessor Variety][beav] 는 각각의 포스트에서 알고리즘을 설명합니다. 이 포스트에서는 soynlp.word.WordExtractor 의 사용법에 대하여 이야기 합니다. 

soynlp.DoublespaceLineCorpus 은 [이전의 포스트][text_to_matrix]에서 설명하였습니다. iter_sent=True 이면 문장 단위로, iter_sent=False 이면 문서 단위로 iteration 을 수행합니다. 실험에 이용한 데이터는 2016-10-20 의 뉴스 30,091 건 입니다. 

{% highlight python %}
from soynlp import DoublespaceLineCorpus

corpus = DoublespaceLineCorpus(corpus_path, iter_sent=True)
print(len(corpus)) # 223,357
{% endhighlight %}

단어 추출 방법들은 문장 단위로 학습합니다. 학습을 위하여 WordExtractor 를 만든 뒤 corpus 를 train() 함수에 넣습니다. Subwords 에 대한 통계 수치를 학습합니다. extract() 함수는 {str:Scores} 를 return 합니다. 

{% highlight python %}
%%time
from soynlp.word import WordExtractor

word_extractor = WordExtractor(min_count=5)
word_extractor.train(corpus)
word_scores = word_extractor.extract()

# Wall time: 54.9 s
{% endhighlight %}

word_scores 는 substring 을 key 로 지닌 dict 입니다. value 는 Scores 의 collections.namedtuple 입니다. 

{% highlight python %}
word_scores['아이오아이']
{% endhighlight %}

Scores 에는 [Cohesion][cohesion], [Branching Entropy][beav], [Accessor Variety][beav] 의 값이 학습되어 있습니다. 각각에 대한 설명은 링크의 포스트를 참고하세요. forward 는 왼쪽에서 오른쪽 방향으로 바라보았을 때의 score 입니다. cohesion_forward 는 어절의 왼쪽에 위치한 subword 의 cohesion score 입니다. 

	Scores(cohesion_forward=0.30063636035733476,
	   cohesion_backward=0,
	   left_branching_entropy=3.0548011243339506,
	   right_branching_entropy=2.7422160443312897,
	   left_accessor_variety=32,
	   right_accessor_variety=27,
	   leftside_frequency=270,
	   rightside_frequency=0
	   )

'아이오아이' 라는 단어의 왼쪽부터 오른쪽 끝까지의 cohesion score 와 subword frequency 를 확인합니다. 

{% highlight python %}
for e in range(2, 6):
    word = '아이오아이'[:e]
    if word in word_scores:
    score = word_scores[word].cohesion_forward
    frequency = word_scores[word].leftside_frequency
    else:
    score = 0
    frequency = 0
    print('word = {}, frequency = {}, cohesion_forward={}'.format(word, frequency, score))
{% endhighlight %}

'아이' 라는 단어의 빈도수가 급격히 떨어지며, '아이오아이'가 될 때 cohesion score 가 가장 큽니다. '아이오아'가 word_scores 에 포함되지 않은 이유는 빈도수가 거의 같은 subwords 는 긴 subword 만 남기도록 되어있기 때문입니다. 실제로 '아이오아'의 빈도수는 270 으로 '아이오아이'와 같습니다. 

	word = 아이, frequency = 4910, cohesion_forward=0.1485537940215418
	word = 아이오, frequency = 307, cohesion_forward=0.09637631475495469
	word = 아이오아, frequency = 0, cohesion_forward=0
	word = 아이오아이, frequency = 270, cohesion_forward=0.30063636035733476



## 한국어 어절의 구조 L + [R]

한국어 어절의 구조는 L + [R] 입니다. 한국어 단어의 품사는 아래와 같은 5언 9품사 입니다. 어절은 하나의 단어로 구성될 수 있습니다. 명사 / 부사 / 동사 / 형용사 / 감탄사는 독립적으로 어절이 될 수 있습니다. 

관형사는 독립적으로 어절이 되기도, 명사 앞에 위치하기도 합니다. '그 사람'을 띄어쓸 경우 '그'는 관형사 입니다. 때론 '그사람'으로 붙여 쓰기도 합니다.

조사는 반드시 체언 뒤에 위치합니다. '조사/명사 + 는/조사' 이렇게 위치합니다.

용언은 분명 독립적으로 하나의 어절을 이뤄야 하지만, 명사 뒤에 위치하는 것처럼 보이기도 합니다. '시작했다'는 동사입니다. 그러나 형태소 분석을 하면 '시작/명사 + 하/동사형전성어미 + 었다/종결어미' 입니다. 명사 뒤에 전성어미가 결합되어 동사나 형용사화 됩니다. 이와 같은 전성어미로는 '-하, -이, -되' 등이 있습니다. '하다, 이다, 되다' 등의 용언 취급을 하면 '시작했다 = 시작/명사 + 했다/동사'로 처리할 수도 있습니다. 어문법에 맞는 것은 아니지만, 데이터 분석의 관점에서는 이처럼 처리할 수 있습니다. 

| 언 | 품사 |
| --- | --- |
| 체언 | 명사, 대명사, 수사 |
| 수식언 | 관형사, 부사 |
| 관계언 | 조사 |
| 독립언 | 감탄사 |
| 용언 | 동사, 형용사 |

그렇다면 한국어 어절의 구조는 L + [R] 입니다. L 에는 '체언 / 부사 / 동사 / 형용사 / 감탄사'가 올 수 있습니다. R 에는 '조사 / 동사 / 형용사'가 올 수 있습니다. 그러나 R 이 반드시 필요하지는 않습니다.


## L-Tokenizer

띄어쓰기가 잘 지켜진 데이터에서는 어절은 띄어쓰기 기준으로 구분됩니다. 그리고 어절의 구성은 L+[R] 입니다. 그렇다면 어절의 왼쪽에 위치한 subwords 중 가장 단어스러운 부분을 잘라내면 어절이 L + [R] 로 나뉩니다. '예시는 = 예시 + 는'으로 나뉘어집니다. L-Tokenizer 는 어절의 왼쪽에 위치한 길이가 2 이상인 subwords 중에서 단어 점수가 가장 높은 subword 를 L 로 자릅니다. 만약 어절 전체가 가장 단어 점수가 높다면 R 은 empty str 입니다. 

L-Tokenizer 는 토크나이징에 이용할 score dict 를 argument 로 받습니다. 

{% highlight python %}
from soynlp.tokenizer import LTokenizer

ltokenizer = LTokenizer(scores = cohesion_scores)
{% endhighlight %}

토크나이징을 할 문장을 .tokenize() 에 입력합니다. 

{% highlight python %}
ltokenizer.tokenize('아이오아이의 무대가 방송에 중계되었습니다')
# ['아이오아이', '의', '무대', '가', '방송', '에', '중계', '되었습니다']
{% endhighlight %}

각 어절 별로 (L, R) 의 구조를 살펴보기 위해서 flatten=False 로 입력합니다. Default value 는 flatten=True 입니다. 

{% highlight python %}
ltokenizer.tokenize('아이오아이의 무대가 방송에 중계되었습니다', flatten=False)
# [('아이오아이', '의'), ('무대', '가'), ('방송', '에'), ('중계', '되었습니다')]
{% endhighlight %}

때로는 R parts 가 불필요하기도 합니다. 이를 제거하기 위해서는 remove_r=True 를 이용합니다. 

{% highlight python %}
ltokenizer.tokenize('아이오아이의 무대가 방송에 중계되었습니다', remove_r=True)
# ['아이오아이', '무대', '방송', '중계']
{% endhighlight %}


## Max Score Tokenizer

뉴스 데이터는 띄어쓰기가 잘 지켜져 있습니다. 공식 문서들도 띄어쓰기가 잘 지켜져 있습니다. 그러나 웹공간에서 만날 수 있는 수많은 문서들은 띄어쓰기가 잘 지켜지지 않습니다. 철자까지도 틀리며 ['야민정음'][yaminjungum]까지도 등장하는데, 띄어쓰기가 잘 지켜져 있기를 기대할 수도 없습니다. 띄어쓰기가 잘 지켜지지 않는 경우에 이용하기 위하여 Max Score Tokenizer 를 만들었습니다. 

Max Score Tokenizer 의 원리는 사람이 띄어쓰기가 지켜지지 않은 문장을 인식하는 원리와 같습니다. 사람도 잘 알고 있는 단어부터 눈에 들어옵니다. 아래 문장을 단어들로 직접 나눠보세요.

	이런문장을직접토크나이징을해볼게요

우리는 지금 토크나이징을 이야기 하고 있기 때문에, '토크나이징'이라는 단어가 눈에 잘 들어옵니다. 그 다음으로는 '문장', '직접', '볼게요' 순으로 단어가 눈에 들어옵니다. 일단 그렇다고 가정하면, 아래 순서대로 단어를 마킹할 수 있습니다. 

	이런문장을직접 [토크나이징] 을해볼게요
	이런 [문장] 을직접 [토크나이징] 을해볼게요
	이런 [문장] 을 [직접] [토크나이징] 을해볼게요
	이런 [문장] 을 [직접] [토크나이징] 을해 [볼게요]

단어로 인식되지 않은 부분들은 그대로 이어서 하나의 단어로 취급합니다.

	[이런] [문장] [을] [직접] [토크나이징] [을해] [볼게요]

그 결과 [이런, 문장, 을, 직접, 토크나이징, 을해, 볼게요]라는 토큰을 얻을 수 있습니다. 아쉬운 점은 [토크나이징, 을, 해, 볼게요]이지만, -을, 해-를 제대로 인식하지 못할 수 있습니다. 이는 계속 개선해야 할 점이긴 합니다. 그보다 중요한 점은, 단어라고 확신이 드는 부분부터 연속된 글자집합에서 잘라내어도 토크나이징이 된다는 점입니다. 

Word Piece Model 과 비슷한 원리이기도 합니다. WPM 처럼 가장 빈번한 연속된 substring 을 단어로 인식하는 방법은 substring frequency 를 단어 점수로 이용하는 것과 같습니다. 그러나 substring frequency 보다도 단어의 경계를 명확히 표현할 수 있는 방법들이 있습니다. Cohesion 이나 Branching Entropy 는 좀 더 명확한 방법으로 단어의 경계를 찾기 위한 방법이며, L-Tokenizer 와 Max Score Tokenizer 는 이를 이용한 토크나이저 입니다. 그리고 WPM 의 목적은 정해진 크기의 subword units 으로 모든 단어를 표현하는 것입니다. 목적이 다릅니다. 

다시 Max Score Tokenizer 로 돌아와서, 위 아이디어를 알고리즘으로 구현해 봅니다. 우리에게 아래의 네 가지 subwords 의 점수표와 예문이 있다고 합니다. 

	sent = '파스타가좋아요'
	scores = {'파스': 0.3, '파스타': 0.7, '좋아요': 0.2, '좋아':0.5}

단어 길이의 범위를 [2, 3]이라고 가정하면 아래와 같은 subword score를 얻을 수 있습니다. 아래는 (subword, begin, end, score) 입니다.

	[('파스', 0, 2, 0.3),
	('파스타', 0, 3, 0.7),
	('스타', 1, 3, 0),
	('스타가', 1, 4, 0),
	('타가', 2, 4, 0),
	('타가좋', 2, 5, 0),
	('가좋', 3, 5, 0),
	('가좋아', 3, 6, 0),
	('좋아', 4, 6, 0.5),
	('좋아요', 4, 7, 0.2),
	('아요', 5, 7, 0)]
 
이를 점수 순서로 정렬하면 아래와 같습니다. 사람도 아는 단어부터 잘 인식된다는 점을 sorting 으로 잘 아는 subword 를 찾는 과정으로 구현하였습니다. 

	[('파스타', 0, 3, 0.7),
	 ('좋아', 4, 6, 0.5),
	 ('파스', 0, 2, 0.3),
	 ('좋아요', 4, 7, 0.2),
	 ('스타', 1, 3, 0),
	 ('스타가', 1, 4, 0),
	 ('타가', 2, 4, 0),
	 ('타가좋', 2, 5, 0),
	 ('가좋', 3, 5, 0),
	 ('가좋아', 3, 6, 0),
	 ('아요', 5, 7, 0)]

파스타라는 subword 의 점수가 가장 높으니, 이를 토큰으로 취급합니다. 파스타의 범위인 [0, 3)과 겹치는 다른 subwords 을 리스트에서 지워주면 아래와 같은 토큰 후보들이 남습니다. 

파스타가좋아요 > [파스타]가좋아요

	[('좋아', 4, 6, 0.5),
	 ('좋아요', 4, 7, 0.2),
	 ('가좋', 3, 5, 0),
	 ('가좋아', 3, 6, 0),
	 ('아요', 5, 7, 0)]

다음으로 '좋아'를 단어로 인식하면 남은 토큰 후보가 없기 때문에 아래처럼 토크나이징이 되며, 남는 글자들 역시 토큰으로 취급하여 토크나이징을 종료합니다. 

	파스타가좋아요 > [파스타]가[좋아]요 > [파스타, 가, 좋아, 요]

이처럼 단어 점수만을 이용하여도 손쉽게 토크나이징을 할 수 있습니다. 이 방법의 장점은 각 도메인에 적절한 단어 점수를 손쉽게 변형할 수 있다는 것입니다. 도메인에서 반드시 단어로 취급되어야 하는 글자들이 있다면, 그들의 점수를 scores에 최대값으로 입력합니다. Score tie-break 는 글자가 오버랩이 되어 있다면, 좀 더 긴 글자를 선택하는 것으로 구현하였습니다. 합성명사 역시 처리 가장 긴 명사를 선택하게 됩니다. 

	scores = {'서울': 1.0, '대학교': 1.0, '서울대학교': 1.0} 

위처럼 단어 점수가 부여된다면 '서울대학교'를 [서울, 대학교]로 분리하지는 않을 것입니다. 대신 '서울'이나 '대학교'가 등장한 다른 어절에서는 이를 단어로 분리합니다. 

Max Score Tokenizer 는 이러한 컨셉으로, 단어 점수를 토크나이저에 입력하여 원하는 단어를 잘라냅니다. 이는 띄어쓰기가 제대로 이뤄지지 않은 텍스트를 토크나이징하기 위한 방법이며, 단어 점수를 잘 정의하는 것은 단어 추출의 몫입니다. 

Max Score Tokenizer의 사용법은 아래와 같습니다. class instance 를 만들 때 scores에 {str:float} 형태의 단어 점수 사전을 입력합니다. 

{% highlight python %}
from soynlp.tokenizer import MaxScoreTokenizer

maxscoretokenizer = MaxScoreTokenizer(scores = cohesion_scores)
{% endhighlight %}

토크나이징은 L-Tokenizer 처럼 tokenize() 함수에 str 을 입력합니다. cohesion 이 잘 학습되었기 때문에 아래의 문장이 올바른 단어들로 잘 나뉘어졌습니다. 이를 단어 사전 없이 온전히 통계만 이용해도 이뤄낼 수 있습니다. 

{% highlight python %}
maxscoretokenizer.tokenize('아이오아이의무대가방송에중계되었습니다')
# ['아이오아이', '의', '무대', '가', '방송', '에', '중계', '되었습니다']
{% endhighlight %}

각 토큰의 상세 정보가 보고 싶을 때에는 flatten=False 로 설정합니다. list of list of tuple 가 return 됩니다. 바깥 list 는 띄어쓰기 단위의 어절입니다. 아래 예시에서는 하나의 어절 뿐이어서 list 안에 list 가 하나만 들어있습니다. 

안쪽 list 는 한 어절 내의 subword 에 대한 정보를 지닌 list of tuple 입니다. tuple 에는 (subword, begin index, end index, score, length) 가 입력되어 있습니다. 

{% highlight python %}
maxscoretokenizer.tokenize('아이오아이의무대가방송에중계되었습니다', flatten=False)

# [[('아이오아이', 0, 5, 0.30063636035733476, 5),
#   ('의', 5, 6, 0.0, 1),
#   ('무대', 6, 8, 0.042336645588678112, 2),
#   ('가', 8, 9, 0.0, 1),
#   ('방송', 9, 11, 0.31949135704351284, 2),
#   ('에', 11, 12, 0.0, 1),
#   ('중계', 12, 14, 0.0019356503785271852, 2),
#   ('되었습니다', 14, 19, 0.2762976357271788, 5)]]
{% endhighlight %}

## Regex Tokenizer

단어를 추출하지 않아도 기본적으로 토크나이징이 되어야 하는 부분들이 있습니다. 언어의 종류가 바뀌는 부분입니다. 

	이것은123이라는숫자

물론 숫자와 한글이 합쳐져서 하나의 단어가 되기도 합니다. 6.25전쟁이 '6.25', '전쟁'으로 나뉘어진 다음에, 이를 '6.25 - 전쟁'으로 묶는 건 ngram extraction 으로 할 수 있다. 이 부분은 일단 다루지 않습니다. 

'6.25전쟁'과 같은 경우는 소수이며, 대부분의 경우에는 한글|숫자|영어(라틴)|기호가 바뀌는 지점에서 토크나이징이 되어야 합니다. 위의 예제는 적어도 [이것은, 123, 이라는숫자]로 니뉘어져야 합니다. 그 다음에 단어 추출에 의하여 [이것, 은, 123, 이라는, 숫자]라고 나뉘어지는 것이 이상적입니다.

또한 한국어에서 자음/모음이 단어 중간에 단어의 경계를 구분해주는 역할을 합니다. 우리는 문자 메시지를 주고 받을 때 자음으로 이뤄진 이모티콘들로 띄어쓰기를 대신하기도 합니다. 

	아이고ㅋㅋ진짜? = [아이고, ㅋㅋ, 진짜, ?]

'ㅋㅋ' 덕분에 '아이고'와 '진짜'가 구분이 되며, 'ㅠㅠ'와 함께 붙어있는 'ㅋㅋ'는 서로 다른 이모티콘으로 구분이 될 수 있습니.

	아이고ㅋㅋㅜㅜ진짜? = [아이고, ㅋㅋ, ㅜㅜ, 진짜, ?]

이를 분리하는 손쉬운 방법은 'ㅋㅋ'를 찾아내어 앞/뒤에 빈 칸을 하나씩 추가하는 것입니다. 

	str.replace('ㅋㅋ', ' ㅋㅋ ')

str에서의 어떤 pattern 을 찾아내기 위하여 regular expression 을 이용합니다. 

	re.compile('[가-힣]+')

위 regular expression은 초/중/종성이 완전한 한국어의 시작부터 끝까지라는 의미입니다. 

	re.compile('[ㄱ-ㅎ]+')

위 regular expression은 ㄱ부터 ㅎ까지 자음의 범위를 나타냅니다. 

Regex Tokenizer 는 regular extression 을 이용하여 언어가 달라지는 순간에 띄어쓰기를 추가합니다. 영어의 경우에는 움라우트가 들어가는 경우들이 있어서 알파벳 뿐 아니라 라틴까지 포함하였습니다. 

{% highlight python %}
from soynlp.tokenizer import RegexTokenizer

tokenizer = RegexTokenizer()
{% endhighlight %}

띄어쓰기가 전혀 없고, 언어가 바뀌지 않는다면 도저히 방법이 없습니다. 

{% highlight python %}
sent = '이렇게연속된문장은잘리지않습니다만'
tokenizer.tokenize(sent)

# ['이렇게연속된문장은잘리지않습니다만']
{% endhighlight %}

하지만 숫자, 영어, 자음, 모음, 완전글자가 뒤섞여 있다면 그 경계로 토크나이징을 할 수 있습니다. 

{% highlight python %}
sent = '숫자123이영어abc에섞여있으면ㅋㅋ잘리겠죠'
tokenizer.tokenize(sent)

# ['숫자', '123', '이영어', 'abc', '에섞여있으면', 'ㅋㅋ', '잘리겠죠']
{% endhighlight %}

일부라도 존재하는 띄어쓰기 정보까지 이용하면 더욱 좋습니다. 

{% highlight python %}
sent = '띄어쓰기가 포함되어있으면 이정보는10점!꼭띄워야죠'
tokenizer.tokenize(sent)

# ['띄어쓰기가', '포함되어있으면', '이정보는', '10', '점', '!', '꼭띄워야죠']
{% endhighlight %}

이는 다른 토크나이저를 이용하기 전, 최대한 띄어쓰기를 수행하여 다른 토크나이저의 계산 비용을 줄이고 오류를 막기 위한 용도로 이용됩니다. 

## L-Tokenizer / KoNLPy Twitter + Naver news + Word2Vec

Cohesion score 와 L-Tokenizer 를 이용하여 문서를 토크나이징 한 뒤, Word2Vec 모델을 학습시켜봅니다. 

WordExtractor + L-Tokenizer chapter 에서 설명하였던 예시의 ltokenizer 를 이용하여 30,091 개 문서, 22 만 여개의 문장을 토크나이징 하였습니다. 16.7 초가 소비되었습니다. 

{% highlight python %}
%%time
words = [ltokenizer.tokenize(sent) for sent in corpus]

# Wall time: 16.7 s
{% endhighlight %}

KoNLPy 의 엔진을 이용할 때에는 engine loading time 이 걸립니다. 로딩은 처음 pos() 함수가 실행될 때 이뤄집니다. 그래서 예문 하나를 입력하여 미리 로딩을 완료하였습니다. 

{% highlight python %}
from konlpy.tag import Twitter
twitter = Twitter()
twitter.pos('로딩시간을제외하기위한예문')
{% endhighlight %}

같은 데이터에 대하여 토크나이징을 위해 262 초를 이용하였습니다. L-Tokenizer 가 약 15.7 배 빠르게 작업을 완료하였습니다. C code 로 변환한다면 더 빠른 작업이 가능할 것 같습니다.

{% highlight python %}
%%time
twitter_words = [twitter.pos(sent) for sent in corpus]

# Wall time: 4min 22s
{% endhighlight %}

twitter.pos() 의 결과는 list of tuple 입니다. 이를 list of list of str (like) 로 변환합니다. 

{% highlight python %}
twitter_words = [['%s/%s' % (w,t) for w,t in sent] for sent in twitter_words]
{% endhighlight %}

L-Tokenizer 와 KoNLPy 의 트위터 한국어 분석기를 이용하여 각각 토크나이징 한 결과를 Word2Vec 모델에 학습시켰습니다. 

{% highlight python %}
from gensim.models import Word2Vec

word2vec = Word2Vec(words)
word2vec_twitter = Word2Vec(twitter_words)
{% endhighlight %}

그 뒤 유사어 검색을 하였습니다. '방송'의 유사어로 프로그램을 지칭하는 말들과 방송 프로그램의 이름들이 검색됩니다. 

{% highlight python %}
word2vec.most_similar('방송')

# [('예능프로그램', 0.6981078386306763),
#  ('예능', 0.688145637512207),
#  ('방영', 0.660905659198761),
#  ('라디오스타', 0.6552329659461975),
#  ('라디오', 0.6498782634735107),
#  ('엠카운트다운', 0.6143872141838074),
#  ('파워타임', 0.586585521697998),
#  ('식사하셨어요', 0.5865078568458557),
#  ('한끼줍쇼', 0.5841958522796631),
#  ('황금어장', 0.5715700387954712)]
{% endhighlight %}

'뉴스'의 유사어는 뉴스 체널과 관련된 단어가 학습되었습니다. 

{% highlight python %}
word2vec.most_similar('뉴스')

# [('연예', 0.6419150233268738),
#  ('화면', 0.6120390892028809),
#  ('뉴스부장', 0.5935080051422119),
#  ('뉴스부', 0.5910232067108154),
#  ('라디오', 0.5880887508392334),
#  ('채널', 0.5730870962142944),
#  ('정보팀', 0.5722399950027466),
#  ('속보팀', 0.5669897794723511),
#  ('앵커', 0.5661150217056274),
#  ('토크쇼', 0.5654663443565369)]
{% endhighlight %}

'영화'의 유사어는 영화 용어 및 영상 작품을 지칭하는 단어들이 학습되었습니다. 

{% highlight python %}
word2vec.most_similar('영화')

# [('작품', 0.7261432409286499),
#  ('드라마', 0.7252553701400757),
#  ('흥행', 0.6903091669082642),
#  ('독립영화', 0.6875913143157959),
#  ('걷기왕', 0.654923677444458),
#  ('감독', 0.6543536186218262),
#  ('럭키', 0.6453847885131836),
#  ('다큐멘터리', 0.6425734758377075),
#  ('블록버스터', 0.6288458704948425),
#  ('주연', 0.6285911798477173)]
{% endhighlight %}

'아이오아이'의 유사어로 '에이핑크', '샤이니', '다이아', '트와이스'와 같은 동시대에 함께 활동한 다른 아이돌 그룹이 거색됩니다. '너무너무'는 타이틀곡 '너무너무너무'가 잘못 토크나이징 된 결과이며, '몬스'는 '몬스터 엑스'입니다. 이 부분은 아쉽습니다. '잠깐만'은 '아이오아이'의 노래 제목이며, '불독'은 그 당시 데뷔하였던 다른 아이돌그룹 이름입니다. '불독의'로 토크나이징 된 점이 아쉽습니다.

{% highlight python %}
word2vec.most_similar('아이오아이')

# [('타이틀곡', 0.8425650000572205),
#  ('에이핑크', 0.8301823139190674),
#  ('샤이니', 0.8209962844848633),
#  ('너무너무', 0.8158787488937378),
#  ('다이아', 0.8083204030990601),
#  ('트와이스', 0.8079981803894043),
#  ('파이터', 0.8070152997970581),
#  ('잠깐만', 0.7995353937149048),
#  ('몬스', 0.7901242971420288),
#  ('불독의', 0.7892482280731201)]
{% endhighlight %}

하지만, 여기까지 오는데 우리는 사전을 전혀 이용하지 않았습니다. 이러한 통계 정보를 더 잘 활용한다면 미등록 단어의 많은 문제들이 해결될 가능성이 있겠네요. 가능성이 보입니다. 

KoNLPy 의 트위터 한국어 분석기를 이용한 토크나이징 결과를 Word2Vec 모델에 학습한 뒤, 유사어 검색을 하였습니다. '영화, 뉴스, 방송'에 대하여 유사한 단어가 잘 학습되었음을 확인할 수 있습니다. 

{% highlight python %}
word2vec_twitter.most_similar('영화/Noun')

# [('드라마/Noun', 0.6787333488464355),
#  ('작품/Noun', 0.677376389503479),
#  ('독립영화/Noun', 0.6677168607711792),
#  ('영화로/Noun', 0.6049726009368896),
#  ('주연/Noun', 0.5995980501174927),
#  ('럭키/Noun', 0.5985063314437866),
#  ('감독/Noun', 0.5851325988769531),
#  ('다큐멘터리/Noun', 0.5788881778717041),
#  ('청춘/Noun', 0.5751833915710449),
#  ('강동원/Noun', 0.5721665620803833)]
{% endhighlight %}

{% highlight python %}
word2vec_twitter.most_similar('뉴스/Noun')

# [('연합뉴스/Noun', 0.6110204458236694),
#  ('연예/Noun', 0.5120304822921753),
#  ('기자/Noun', 0.4866550862789154),
#  ('김도연/Noun', 0.48150816559791565),
#  ('렬/Noun', 0.47369953989982605),
#  ('권현진/Noun', 0.47151216864585876),
#  ('방지영/Noun', 0.46529433131217957),
#  ('2580/Number', 0.46096765995025635),
#  ('한대욱/Noun', 0.4602851867675781),
#  ('김영선/Noun', 0.4595600962638855)]
{% endhighlight %}

{% highlight python %}
word2vec_twitter.most_similar('방송/Noun')

# [('방영/Noun', 0.6817507147789001),
#  ('엠카운트다운/Noun', 0.6040224432945251),
#  ('예능/Noun', 0.5873677134513855),
#  ('라디오스타/Noun', 0.5550553202629089),
#  ('방송사/Noun', 0.5497354865074158),
#  ('첫방송/Noun', 0.5472666621208191),
#  ('방송한/Verb', 0.5442488789558411),
#  ('수목드라마/Noun', 0.5297601222991943),
#  ('라디오/Noun', 0.5036641955375671),
#  ('준영/Noun', 0.49247831106185913)]
{% endhighlight %}

[soynlp]: https://github.com/lovit/soynlp/
[yaminjungum]: https://namu.wiki/w/%EC%95%BC%EB%AF%BC%EC%A0%95%EC%9D%8C
[wpm]: {{ site.baseurl }}{% link _posts/2018-04-02-wpm.md %}
[text_to_matrix]: {{ site.baseurl }}{% link _posts/2018-03-26-from_text_to_matrix.md %}
[pos_and_oov]: {{ site.baseurl }}{% link _posts/2018-04-01-pos_and_oov.md %}
[cohesion]: {{ site.baseurl }}{% link _posts/2018-04-09-cohesion_ltokenizer.md %}
[beav]: {{ site.baseurl }}{% link _posts/2018-04-09-branching_entropy_accessor_variety.md %}