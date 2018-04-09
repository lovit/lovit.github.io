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


[soynlp]: https://github.com/lovit/soynlp/
[wpm]: {{ site.baseurl }}{% link _post/2018-04-02-wpm.md %}
[text_to_matrix]: {{ site.baseurl }}{% link _post/2018-03-26-from_text_to_matrix.md %}
[pos_and_oov]: {{ site.baseurl }}{% link _post/2018-04-01-pos_and_oov.md %}
[cohesion]: {{ site.baseurl }}{% link _post/2018-04-09-cohesion_ltokenizer.md %}
[beav]: {{ site.baseurl }}{% link _post/2018-04-09-branching_entropy_accessor_variety.md %}