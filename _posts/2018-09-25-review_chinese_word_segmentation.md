---
title: (Review) Incorporating Global Information into Supervised Learning for Chinese Word Segmentation
date: 2018-09-25 20:00:00
categories:
- nlp
tags:
- tokenizer
- sequential labeling
- word extraction
---

중국어의 문장에는 띄어쓰기가 존재하지 않습니다. 대신 한국어처럼 용언의 활용이 일어나지 않습니다. 그렇기 때문에 중국어의 품사 판별을 하기 위해서는 문장을 단어열로 분해하는 word segmentation 을 수행합니다. Conditional Random Field 는 학습 데이터를 이용하는 supervised word segmentation 에 적합하여 자주 이용되었습니다. 그러나 중국어에서도 미등록단어와 모호성 문제는 발생합니다. 이를 해결하기 위하여 unsupervised features 를 supervised model 에 결합하기 위한 시도들이 있었습니다. 이 포스트는 Accessor Variety 와 같은 unsupervised word extraction features 을 CRF model 에 적용하는 방법에 대한 논문인 "Incorporating Global Information into Supervised Learning for Chinese Word Segmentation" 을 리뷰합니다.


## Chinese Word Segmentation (CWS) problem

중국어는 띄어쓰기를 포함하지 않는 문자열로 문장을 표현합니다. 중국어 텍스트를 분석할 때에도 품사 판별 (part-of-speech tagging) 은 필요합니다. 품사 판별을 하려면 일단 띄어쓰기가 포함되지 않은 문장에서 단어들을 인식할 수 있어야 하며, 이 과정을 word segmentation 이라 합니다. 한국어는 용언의 어미가 용언의 어간에 결합하는 과정에서 그 형태가 변할 수 있는 교착어이지만, 중국어는 그렇지 않습니다. 중국어의 품사 판별은 문장에서 단어를 인식하는 것으로도 해결이 됩니다. 그렇기 때문에 중국어에서는 word segmentation 의 이름으로 품사 판별이 발전되었습니다.

중국어의 텍스트 전처리가 word segmentation 만으로 해결될 수 있는 이유 중 하나는 중국어에서 이용되는 글자 (characters) 의 종류가 다양하기 때문입니다. 한국어에서는 '이'라는 글자는 '숫자 2, 조사, 이빨, 형용사 이다, 해충 이, ... ' 처럼 많은 의미를 지니지만, 한자에서 이들은 모두 다른 글자 입니다. 물론 한자에도 모호성이 존재하지만 그 수준이 한글보다는 훨씬 적습니다.

그러나 언어를 구성하는 글자수가 많다 하더라도 모호성은 발생합니다.  단어 사전에 'A, B, C, AB, BC' 가 존재한다면 'ABC' 라는 문장을 'A - B - C', 'AB - C' 혹은 'A - BC' 라는 단어열로 나눌 수 있습니다. 이 중 가장 적절한 단어열을 선택해야 합니다. 이는 마치 한국어의 '아버지가방에' $$\rightarrow$$ [아버지, 가, 방, 에] or [아버지, 가방, 에] 와 같은 문제입니다.

Hai Zhao 교수는 Chinese word segmentation 을 연구하였던 교수입니다. 그는 주로 Conditional Random Field 를 이용한 supervised word segmentation 연구 뿐 아니라, supervised CRF model 에 unsupervised features 를 추가하는 방법도 연구 하였습니다. 이번 포스트는 그의 연구 중 한 편인 "Incorporating Global Information into Supervised Learning for Chinese Word Segmentation" 를 리뷰합니다.


## Conditional Random Field (CRF) for Word Segmentation

Conditional Random Field (CRF) 는 sequential labeling 을 위하여 potential functions 을 이용하는 softmax regression 입니다. Deep learning 계열 모델인 Recurrent Neural Network (RNN) 이 sequential labeling 에 이용되기 전에, 다른 많은 모델보다 좋은 성능을 보인다고 알려진 모델입니다. 

Sequential labeling 은 길이가 $$n$$ 인 sequence 형태의 입력값 $$x = [x_1, x_2, \ldots, x_n]$$ 에 대하여 길이가 $$n$$ 인 적절한 label sequence $$y = [y_1, y_2, \ldots, y_n]$$ 을 출력합니다. 이는 $$argmax_y P(y_{1:n} \vert x_{1:n})$$ 로 ㅍ현할 수 있습니다.

Softmax regression 은 벡터 $$x$$ 에 대하여 label $$y$$ 를 출력하는 함수입니다. 하지만 입력되는 sequence data 가 단어열과 같이 벡터가 아닐 경우에는 이를 벡터로 변환해야 합니다. Potential function 은 categorical value 를 포함하여 sequence 로 입력된 다양한 형태의 값을 벡터로 변환합니다. 

Potential function 은 Boolean 필터처럼 작동합니다. 아래는 두 어절로 이뤄진 문장, "예문 입니다" 입니다. 앞의 한글자와 그 글자의 띄어쓰기, 그리고 현재 글자를 이용하여 현재 시점 $$i$$ 를 벡터로 표현할 수 있습니다.

- $$x = '예문 입니다'$$ .
- $$F_1 = 1$$ if $$x_{i-1:i} =$$ '예문' else $$0$$
- $$F_2 = 1$$ if $$x_{i-1:i} =$$ '예문' & $$y[i-1] = 0$$ else $$0$$
- $$F_3 = 1$$ if $$x_{i-1:i} =$$ '문입' else $$0$$
- $$\cdots$$

그림은 위 예시 템플릿을 이용하여 '예문 입니다'에 potential functions 을 적용한 결과입니다. 마치 5 개의 문서에 대한 term frequency vector 처럼 보입니다. 

![]({{ "/assets/figures/crf_potential_function.png" | absolute_url }})

이처럼 potential functions 은 임의의 형태의 데이터라 하더라도 Boolean filter 를 거쳐 high dimensional sparse Boolean vector 로 표현합니다. Conditional Random Field 는 특정 상황일 때 특정 label 의 확률을 학습하는 모델입니다. 

자세한 Conditional Random Field 의 설명은 이전 [블로그][crf] 를 참고하세요. 


## Incorporating supervised and unsupervised features

Word segmentation 의 접근법은 학습데이터의 유무에 따라 크게 두 가지로 분류합니다.

Supervised word segmentation 은 문장이 단어열로 나뉘어져 있는 학습 데이터를 이용합니다. 학습 데이터가 존재한다면 이를 이용하여 (1) 단어와 (2) 문맥에 따른 단어열 선호를 학습할 수 있습니다. 첫 번째 학습하는 정보는 단어 입니다. 데이터로부터 '아버지, 가, 가방, 방, 에' 가 단어임을 학습할 수 있습니다. 그러나 이들이 단어임을 알고 있더라도 '아버지가방에' $$\rightarrow$$ '아버지, 가, 방, 에'로 나뉘어지려면 문맥에 따라 '방' 과 '가방'의 선호가 달라져야 합니다.

그러나 supervised word segmentation 은 학습 데이터에 존재하지 않는 단어들을 인식하기가 어렵습니다. 가장 최선의 방법은 처음 보는 단어들을 최대한 길게 놔두는 것입니다.

Unsupervised word segmentation 은 학습 데이터가 존재하지 않는 상황을 고려합니다. 어떤 sub-sequence (sub-string) 가 단어인지 알지 못하는데 오로직 문장들만 주어졌다면, 일단 어떤 sub-sequence 가 단어인지를 판단해야 합니다. 즉 unsupervised word segmentation 은 unsupervised word extraction 문제와 연결되어 있습니다. Word extraction 은 (1) 단어에 대한 인식 부분을 담당합니다. (2) 문맥에 따른 단어열 선호 역할은 주로 단어열 결과의 quality criteria 를 이용하여 가장 품질이 것을 선택하는 방식으로 이뤄집니다.


### Supervised features

'아버지가방에' $$\rightarrow$$ '아버지, 가, 방, 에' 로 나누기 위해서는 문맥을 알아야 합니다. 그리고 문맥은 주로 앞, 뒤에 등장하는 단어로 이뤄집니다. 한국어에는 길이가 4 ~ 5 정도 되는 단어들이 존재하기도 하지만, 중국어에서는 아주 짧은 문맥 정보 만으로도 word segmentation 을 할 수 있나 봅니다 (제가 중국어를 모릅니다). 논문에서는 앞, 뒤의 글자만을 이용하여 features 를 만들었습니다.

| Type | Feature | Description |
| --- | --- | --- |
| Unigram | $$C_n, n=-1, 0, 1$$ | The previous (current, next) character |
| Bigram | $$C_nC_{n+1}, n=-1, 0$$ | The previous (next) character and current character |
| Bigram (previous & successive) | $$C_{-1}C_1$$ | The previous character and next character |
| Puncuation, Data, Digital, or Letter | $$T_{-1}T_0T_1$$ | $$T_i$$ is type of previous, current and next character |


### Tagset

Named entity 에서 자주 이용되는 tagset 으로 $$B, I, O$$ 가 있습니다. $$B$$ 는 named entity 의 시작점, $$I$$ 는 중간점, $$O$$ 는 해당되는 entity 가 아니라는 의미입니다. Zhao 는 word segmentation 을 위하여 $$B, M, E$$ 를 이용합니다. 'ABCDE' 에 $$BBMME$$ 라는 tag 가 더해진다면 'A, BCDE' 로 segmentation 이 된다는 의미입니다.

그런데 이와 같은 tagset 을 이용하면 $$M$$ 에 지나치게 많은 정보가 포함되게 됩니다. 이보다는 $$BBB_2B_3E$$ 이 더 좋은 방식입니다. B_2 는 단어의 두번째 글자라는 의미입니다. Zhao 는 $$B, B_2, \cdots, B_6, M, E$$ tagset 을 이용하였습니다.


### Unsupervised features

Unsupervised features 라는 것은, 여러 문장들 중에서 어떤 sub-sequence 가 단어스러운지를 판단하는 features 입니다. 많은 방법들이 제안되었지만, 공통적으로 **단어는 함께 등장할 가능성이 높은 character sequence** 입니다. 함께 등장할 가능성을 정량적으로 정의하는 방식에 따라 여러 unsupervised word extraction 방법들이 제안되었습니다.


#### Mutual Information

Mutual information 은 association rules 의 [Lift][lift] 와 같은 개념입니다. 'A' 라는 글자 다음에 'B' 가 등장할 확률을 $$P(AB \vert A) = \frac{P(AB)}{P(A)}$$ 로 표현할 수 있습니다. 그러나 'B' 는 본래 어디에서든지 자주 등장하는 값일 수 있기 때문에 평균적으로 'B' 가 등장하는 비율만큼 normalize 를 합니다. 

해석과 scaling 을 위하여 logaritm 을 적용합니다. Mutual Information 의 값이 0 이라면 어떤 상관도 없다는 의미이며, 이 값이 클수록 'A' 다음에 'B' 가 자주 등장한다는 의미입니다.

$$MI(A, B) = log \left( \frac{P(AB)}{P(A) \times P(B)} \right)$$

세 글자에 대한 Mutual Information 의 정의는 두 글자에 대한 정의에서 다양하게 확장됩니다. 한 예로 'AB, BC' 에 대한 Mutual Information 으로 정의를 확장할 수도 있습니다.

$$MI(ABC) = log \left( \frac{P(AB, BC)}{P(AB) \times P(BC)} \right)$$

혹은 MI(A, BC) 와 MI(AB, C) 의 평균으로 정의할 수도 있습니다.

$$MI(ABC) = \frac{1}{2} \times \left( log \frac{P(ABC)}{P(A) \times P(BC)} + log \frac{P(ABC)}{P(AB) \times P(C)} \right)$$


#### Accessor Variety & Branching Entropy

[Accessor Variety][accessor_paper] 는 2004 년에 제안된 방법입니다. Harris 의 단어 경계에서의 불확실성을 **단어 경계 다음에 등장한 글자의 종류**로 정의 하였습니다. '카메라'가 단어라면 앞 뒤에는 다양한 종류의 글자들이 등장하지만, '카메'의 오른쪽에는 '라' 혹은 '룬' 같이 몇 글자만이 등장할 것입니다. 왜냐면 아직 단어가 끝나지 않았기 때문입니다.

아래 예시에서 '공연' 오른쪽에 세 종류의 글자가 등장하였기 때문에 right-side accessor variety, $$(av_r)$$ 는 3 입니다. 

	공연은 : 30
	공연을 : 20
	공연이 : 50

반대로 '공연'의 왼쪽에 등장한 글자는 {번, 해} 이기 때문에 left-side accessor variety, $$(av_l)$$ 는 2 입니다. 

	이번공연 : 30
	저번공연 : 20
	올해공연 : 50

그리고 '공연'이 단어라면 앞과 뒤의 글자의 다양성이 모두 커야하기 때문에 $$min(av_l, av_r)$$ 을 글자 가능성 점수로 이용합니다.

$$AV(w) = min(av_l(w), av_r(w))$$

[Branching Entropy][branching_paper] 는 Jin and Tanaka-Ishii (2006)이 제안한 방법으로 Accessor Variety 와 거의 비슷합니다. Branching Entropy 는 entropy 를 불확실성의 정보로 이용합니다. 아래의 예제에서 '공연'의 right-side accessor variety 는 3 입니다. '손나은'의 오자로 '손나응, 손나으'가 적혔었다면 '손나'의 right-side accessor variety 는 3 입니다. 하지만, '손나' 오른쪽에 대부분 '-은' 이 등장하였기 때문에 '손나' 보다는 '손나은'이 더 단어스럽습니다. 

	공연은 : 30
	공연을 : 20
	공연이 : 50

	손나은 : 98
	손나응 : 1
	손나으 : 1

글자 종류의 숫자는 오탈자나 infrequent pattern 에 매우 민감합니다. 이를 보완하기 위해서 entropy 를 이용합니다.

$$entropy P(w \vert c) = - \sum_{w^` \in W} P(w^` \vert c) log P(w^` \vert c)$$

이 역시 오른쪽과 왼쪽의 entropy 가 모두 커야 단어입니다.

$$BE(w) = min(be_l(w), be_r(w))$$

Accessor Variety 와 Branching Entropy 는 단어가 일정 숫자 이상 등장하여야 제대로 작동할 수 있습니다. 'ab' 가 두 번 등장하였다면 아무리 커도 AV(ab) 는 2 보다 커질 수 없습니다. 이는 Mutual Information 도 동일합니다. 즉 unsupervised word extraction 은 **자주 등장하지만 인식하지 못했던 단어를 인식**하는 것이 목적입니다. 이에 대한 자세한 내용과 한국어 데이터에 적용하는 내용은 [이전의 포스트][avbe]를 참고하세요


#### Minimum Description Length (MDL)

Minimum Description Length 는 최소한의 units 을 이용하여 데이터 전체를 설명하려는 프레임워크입니다. 텍스트 데이터의 경우에는 한 문장 s 가 [$$s_1, s_2, \dots, s_n$$] 로 나뉘어졌을 때의 cost 를 아래처럼 정의합니다. $$\theta$$ 는 segmentor 입니다.

$$cost(S \vert \theta) = - \sum_{i=1}^{n} P(s_i) \cdot log P(s_i) $$

그리고 $$\theta$$ 의 cost 는 문장 전체를 나눴을 때의 segments 의 Entropy 의 합으로 정의합니다. 나뉘어지는 단어가 infrequent 하지 않을수록 비용은 줄어듭니다.

$$cost(\theta) = - \sum_{s \in S} \sum_{s_i \in s} P(s_i) \cdot log P(s_i)$$

그 결과 이 기준을 만족하기 위해서는 함께 등장하는 경향이 높은 sub-sequence 를 하나의 unit 으로 인식합니다. 영어에서는 re- 나 -tion 과 같은 prefix, suffix 를 분리하는데 이용되기도 했습니다. 아래는 ([Argamon et al.,2004][mdl2]) 의 예시입니다.

- Words : relic, retire, recognition, relive, tire, cognition, farm
- Units : re, lic, tire, cognition, live, farm

이는 마치 Word Piece Model (WPM) 의 접근법과도 비슷합니다. 학습하는 패턴은 자주 등장하는 sub-sequence 는 units 으로 인식하고, 자주 등장하지 않는 sub-sequence 는 characters 로 나눠서 인식하는 경향이 있습니다.


### Inforcoporating supervised & unsupervised features

Accessor Variety 는 단어의 길이마다 scale 이 다릅니다. 짧은 단어일수록 자주 등장하기 때문에 그 값이 대체로 큽니다. 이를 방지하기 위하여 길이가 2 인 sub-sequence 부터 길이가 7 인 sub-sequence 까지 나누어 accessor variety 를 features 로 이용하였습니다 ([Zhao][global_information]). 그러나 단어의 최대 길이를 7 로 제한한 것은 아닙니다. $$B_6$$ 과 $$M$$ tag 를 이용하기 때문에 길이가 7 보다 긴 단어도 인식할 수 있습니다.

그 외에 sub-sequence 의 frequency 도 이용하였습니다. 이 값은 sub-sequence 를 이루는 요소들 간의 co-occurrence 이기 때문입니다. 예를 들어 '서울대학교'가 10 번 등장하였다면, '서울대', '학교' 의 co-occurrence 가 10 이라는 의미이기 때문입니다.

그리고 앞서 언급한 supervised features 와 함께 이용하여 CRF model 을 학습하였습니다.


### Performance

Unsupervised features 를 이용한다하여도 수치로는 그 성능이 잘 보이지 않습니다. 이 논문은 Backoff-2005 라는 Chinese word segmentation 용 데이터를 이용하여 실험을 하였습니다. Base model 은 supervised features 만을 이용한 경우이며, COS 는 sub-sequence 의 frequency 정보, AVS 는 Accessor Variety 의 정보를 함께 이용한 경우입니다.

![]({{ "/assets/figures/word_segmentation_zhao_performance.png" | absolute_url }}){: width="50%" height="50%"}

애초에 95% 에 가까운 정확도를 보이고 있기 때문에 성능 향상의 폭은 작습니다. 또한 unsupervised features 는 미등록단어가 발생할 때 이를 인식하기 위한 정보입니다. 하지만 Backoff-2005 dataset 를 80:20 처럼 random split 을 한 다음 학습과 성능 평가에 이용한다면 빈도수가 높은 미등록단어는 발생하기 어렵습니다. 그리고 앞서 살펴본 co-occurrence, Accessor Variety 와 같은 features 는 어느 정도 출연을 해야 그 값이 유의미하게 계산됩니다. 하지만 이 논문에서는 미등록 단어의 인식 능력 성능에 대한 실험은 따로 하지 않았습니다.


## Discussion

이 논문에서 주목해야 할 점은 Accessor Variety 와 같은 unsupervised features 를 Conditional Random Field model 에 입력하는 방식입니다.

이 포스트에서는 실험으로 확인하지 않았지만, Accessor Variety features 의 coefficients 는 양수입니다. 즉, 앞/뒤에 다양한 글자들이 자주 등장할수록 단어일 가능성이 높으니, 그 sub-sequence 를 단어로 인식한다는 의미이며, 모호성이 생길 때 variety 의 합이 높은 쪽으로 segmentation 을 한다는 의미이기도 합니다.

교육이나 학교에 관련된 문서에서는 '서울대'라는 단어가 자주 등장하여 Accessor Variety 의 값이 높습니다. 하지만 '서울'의 Right Accessor Variety 값은 작습니다. 그렇기 때문에 이 문서 집합을 이용하여 학습한 모델에서는 '서울대공원' 을 [서울대, 공원] 으로 분해합니다. 하지만 어린이날 관련 문서들에서는 [서울, 대공원] 으로 분해합니다. 즉 문서 집합의 도메인에 따라 적합한 단어를 선택하는 능력이 있습니다.


## Reference

- Argamon, S., Akiva, N., Amir, A., & Kapah, O. (2004, August). [Efficient unsupervised recursive word segmentation using minimum description length.][mdl2] In Proceedings of the 20th international conference on Computational Linguistics (p. 1058). Association for Computational Linguistics.
- Feng, H., Chen, K., Deng, X., & Zheng, W. (2004). [Accessor variety criteria for Chinese word extraction.][accessor_paper] Computational Linguistics, 30(1), 75-93.
- Jin, Z., & Tanaka-Ishii, K. (2006, July). [Unsupervised segmentation of Chinese text by use of branching entropy.][branching_paper] In Proceedings of the COLING/ACL on Main conference poster sessions (pp. 428-435). Association for Computational Linguistics.
- Zhao, H., Huang, C. N., & Li, M. (2006). [An improved Chinese word segmentation system with conditional random field.][crf_cws] In Proceedings of the Fifth SIGHAN Workshop on Chinese Language Processing (pp. 162-165).
- Zhao, H., & Kit, C. (2007, September). [Incorporating global information into supervised learning for Chinese word segmentation.][global_information] In Proceedings of the 10th Conference of the Pacific Association for Computational Linguistics (pp. 66-74).

[global_information]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.482.849&rep=rep1&type=pdf
[crf_cws]: http://www.aclweb.org/anthology/W06-0127
[lift]: https://en.wikipedia.org/wiki/Lift_(data_mining)
[accessor_paper]: http://www.aclweb.org/old_anthology/J/J04/J04-1004.pdf
[branching_paper]: https://www.researchgate.net/profile/Zhihui_Jin/publication/220873812_Unsupervised_Segmentation_of_Chinese_Text_by_Use_of_Branching_Entropy/links/561db42808aecade1acb403e.pdf
[mdl]: https://pdfs.semanticscholar.org/c384/adddcad3a017f8dad14c9847dae0e6dde323.pdf
[mdl2]: http://www.aclweb.org/anthology/C04-1152
[avbe]: {{ site.baseurl }}{% link _posts/2018-04-09-branching_entropy_accessor_variety.md %}
[crf]: {{ site.baseurl }}{% link _posts/2018-04-24-crf.md %}