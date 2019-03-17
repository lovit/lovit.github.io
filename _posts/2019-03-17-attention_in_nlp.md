---
title: Attention mechanism in NLP. From seq2seq + attention to BERT
date: 2017-03-17 23:00:00
categories:
- machine learning
tags:
- machine learning
- attention mechanism
---

Word2Vec 을 제안한 Mikolov 는 "Deep learning NLP 의 발전은 word embedding 때문이다"라는 말을 했습니다. 단어 간의 유사성을 표현할 수 있고, 단어를 continuous vector space 에서 표현하여 (embedding vector 만 잘 학습된다면) 작은 크기의 모델에 복잡한 지식들을 저장할 수 있게 되었습니다. 2013 년에 sequence to sequence 의 context vector 를 개선하기 위하여 attention mechanism 이 제안되었습니다. 이는 모델이 필요한 정보를 선택하여 이용할 수 있는 능력을 주었고, 자연어처리 외에도 다양한 문제에서 성능을 향상하였습니다. 그리고 부산물로 모델의 작동방식을 시각적으로 확인할 수 있도록 도와주고 있습니다. 이번 포스트에서는 sequence to sequence 에서 제안된 attention 부터, self-attention 을 이용하는 language model 인 BERT 까지 살펴봅니다.

## Why attention ?

Word2Vec 을 제안한 Mikolov 는 "Deep learning NLP 의 발전은 word embedding 때문이다"라는 말을 했습니다 (Joulin et al., 2016).

*One of the main successes of deep learning is due to the effectiveness of recurrent networks for language modeling and their application to speech recognition and machine translation.*
{: .text-center }

자연어처리는 word embedding 을 이용하기 전과 후가 명확히 다릅니다. n-gram 을 이용하는 전통적인 statistical language model 은 단어의 종류가 조금만 늘어나도 여러 종류의 n-grams 을 기억해야 했기 때문에 모델의 크기도 컸으며, 단어 간의 semantic 을 쉽게 표현하기 어려웠기 때문에 WordNet 과 같은 외부 지식을 쌓아야만 했습니다. Bengio et al., (2003) 는 neural network 를 이용한 language model 을 제안하였고, 그 부산물로 word embedding vectors 를 얻을 수 있었습니다. 그리고 Mikolov et al., (2013) 에 의하여 제안된 Word2Vec 은 Bengio 의 neural language model 의 성능은 유지하면서 학습 속도는 비약적으로 빠르게 만들었고, 모두가 손쉽게 word embedding 을 이용할 수 있도록 도와줬습니다. 물론 Python package [Gensim](https://radimrehurek.com/gensim/) 도 큰 역할을 했다고 생각합니다. Gensim 덕분에 파이썬을 이용하는 분석가들이 손쉽게 LDA 와 Word2Vec 을 이용할 수 있게 되었으니까요.

사견이지만, attention 도 word embedding 만큼이나 자연어처리에 중요한 역할을 한다고 생각합니다. 처음에는 sequence to sequence 의 context vector 를 개선하기 위하여 제안되었지만, 이제는 다양한 딥러닝 모델링에 하나의 기술로 이용되고 있습니다. 물론 모델의 성능을 향상 시킨 점도 큽니다. 하지만 부산물로 얻을 수 있는 attention weight matrix 를 이용한 모델의 작동 방식에 대한 시각화는 모델의 안정성을 점검하고, 모델이 의도와 다르게 작동할 때 그 원인을 찾는데 이용될 수 있습니다. 이전보다 쉽게 복잡한 모델들을 해석할 수 있게 된 것입니다.

그리고 최근에는 self-attention 을 이용하는 Transformer 가 번역의 성능을 향상시켜주었고, 이를 이용하는 BERT language model 은 왠만한 NLP tasks 의 기록들을 단 하나의 단일 모델로 갈아치웠습니다.

이번 포스트에서는 attention mechanism 의 시작인 sequence to sequence 부터 BERT 까지, attention mechanism 을 이용하는 모델들에 대하여 정리합니다.

## Attention in sequence to sequence

Sequence to sequence 는 Sutskever et al., (2014) 에 의하여 번역과 같이 하나의 input sequence 에 대한 output sequence 를 출력하기 위하여 제안되었습니다. 이는 part of speech tagging 과 같은 sequential labeing 과 다른데, sequential labeling 은 input sequence $$[x_1, x_2, \dots, x_n]$$ 의 각 $$x_i$$ 에 해당하는 $$[y_1, y_2, \dots, y_n]$$ 을 출력합니다. Input 과 output sequence 의 길이가 같습니다. 하지만 처음 sequence to sequence 가 풀고자 했던 문제는 번역입니다. 번역은 input sequence $$x_{1:n}$$ 의 의미와 같은 의미를 지니는 output sequence $$y_{1:m}$$ 을 출력하는 것이며, $$x_i$$, $$y_i$$ 간의 관계를 학습하는 것이 아닙니다. 그리고 각 sequence 의 길이도 서로 다를 수 있습니다.

아래 그림은 input sequence [A, B, C] 에 대하여 output sequence [W, X, Y, Z] 를 출력하는 sequence to sequence model 입니다. 서로 언어가 다르기 때문에 sequence to sequence 는 input (source) sentence 의 언어적 지식을 학습하는 encoder RNN 과 output (target) sentence 의 언어적 지식을 학습하는 decoder RNN 을 따로 두었습니다. 그리고 이 두 개의 RNN 으로 구성된 encoder - decoder 를 한 번에 학습합니다.

![]({{ "/assets/figures/seq2seq.png" | absolute_url }}){: width="90%" height="90%"}

Sequence to sequence 가 학습하는 기준은 $$maximize \sum P_{\theta} \left( y_{1:m} \vert x_{1:n} \right)$$ 입니다. $$x_{1:n}$$ 과 $$y_{1:m}$$ 의 상관성을 최대화 하는 것입니다. 이때 sequence to sequence 는 input sequence 의 정보를 하나의 context vector $$c$$ 에 저장합니다. Encoder RNN 의 마지막 hidden state vector 를 $$c$$ 로 이용하였습니다. Decoder RNN 은 고정된 context vector $$c$$ 와 현재까지 생성된 단어열 $$y_{1:i-1}$$ 을 이용하는 language model (sentence generator) 입니다.

$$P(y_{1:m} \vert x_{1:n}) = \prod_i P(y_i \vert y_{1:i-1}), c)$$ 물론 이 구조만으로도 번역의 성능은 향상되었습니다. Mikolov 의 언급처럼 word embedding 정보를 이용하였기 때문입니다. Classic n-grams 을 이용하는 기존의 statistical machine translation 보다 작은 크기의 모델 안에 단어 간의 semantic 정보까지 잘 포함되었기 때문입니다.

![]({{ "/assets/figures/seq2seq_fixed_context.png" | absolute_url }}){: width="40%" height="40%"}

그런데, Bahdanau et al., (2014) 에서 하나의 문장에 대한 정보를 하나의 context vector $$c$$ 로 표현하는 것이 충분하지 않다고 문제를 제기합니다. Decoder RNN 이 문장을 만들 때 각 단어가 필요한 정보가 다를텐데, sequence to sequence 는 매 시점에 동일한 context $$c$$ 를 이용하기 때문입니다. 대신에 $$x_1, x_2, \dots, x_n$$ 에 해당하는 encoder RNN 의 hidden state vectors $$h_1, h_2, \dots, h_n$$ 의 조합으로 $$y_i$$ 마다 다르게 조합하여 이용하는 방법을 제안합니다. 표현이 너무 좋아서 논문의 구절을 그대로 인용하였습니다.

*A potential issue with this encoder–decoder approach is that a neural network needs to be able to compress all the necessary information of a source sentence into a fixed-length vector.*
{: .text-center }

*Instead, it encodes the input sentence into a sequence of vectors and chooses a subset of these vectors adaptively while decoding the translation. This frees a neural translation model from having to squash all the information of a source sentence, regardless of its length, into a fixed-length vector.*
{: .text-center }

아래의 그림처럼 decoder RNN 이 $$y_i$$ 를 선택할 때 encoder RNN 의 $$h_j$$ 를 얼만큼 이용할지를 $$a_{ij}$$ 로 정의합니다. $$y_i$$ 의 context vector $$c_i$$ 는 $$\sum_j a_{ij} \cdot h_j$$ 로 정의되며, $$\sum_j a_{ij} = 1, a_{ij} \ge 0$$ 입니다. $$a_{ij}$$ 를 attention weight 라 하며, 이 역시 neural network 에 의하여 학습됩니다.

![]({{ "/assets/figures/seq2seq_with_attention.png" | absolute_url }}){: width="40%" height="40%"}

Weight 는 decoder 의 이전 hidden state $$s_{i-1}$$ 와 encoder 의 hidden state $$h_j$$ 가 input 으로 입력되는 feed-forward neural network 입니다. 출력값 $$e_{ij}$$ 는 하나의 숫자이며, 이들을 softmax 로 변환하여 확률 형식으로 표현합니다. 그리고 이 확률을 이용하여 encoder hidden vectors 의 weighted average vector 를 만들어 context vector $$c_i$$ 로 이용합니다.

$$a_{ij} = \frac{exp(e_{ij})}{\sum_j exp(e_{ij})}$$, $$e_{ij} = f(s_{i-1}, h_j)$$
{: .text-center }

![]({{ "/assets/figures/seq2seq_attention_input.png" | absolute_url }}){: width="60%" height="60%"}

Attention 을 계산하는 feed-forward network 는 간단한 구조입니다. 이는 $$[s_{i-1}; h_j]$$ 라는 input vector 에 대한 1 layer feed forward neural network 입니다.

$$e_{ij} = f(W^1 s_{i-1} + W^2 h_j)$$
{: .text-center }

즉 이전에는 아래의 그림처럼 'this is example sentence' 를 '이것은 예문이다'로 번역하기 위하여 매번 같은 context vector 를 이용했지만,

![]({{ "/assets/figures/seq2seq_structure.png" | absolute_url }}){: width="60%" height="60%"}

attention 이 이용되면서 '이것' 이라는 단어를 선택하기 위하여 'this is' 라는 부분에 주목할 수 있게 되었습니다.

![]({{ "/assets/figures/seq2seq_attention_structure.png" | absolute_url }}){: width="60%" height="60%"}

그리고 그 결과물로 attention weight matrix 를 얻을 수 있습니다. 아래는 영어와 프랑스어 간에 번역을 위하여 각각 어떤 단어끼리 높은 attention weight 가 부여됬는지를 표현한 그림입니다. 검정색일수록 낮은 weight 를 의미합니다. 관사 끼리는 서로 연결이 되어 있으며, 의미가 비슷한 단어들이 실제로 높은 attention weight 를 얻습니다. 그리고 하나의 단어가 두 개 이상의 단어의 정보를 조합하여 이용하기도 합니다.

![]({{ "/assets/figures/seq2seq_attention_visualize.png" | absolute_url }}){: width="60%" height="90%"}

하지만 대체로 한 단어 $$y_i$$ 를 만들기 위하여 이용되는 $$h_j$$ 의 개수는 그리 많지 않습니다. 필요한 정보는 매우 sparse 하며, 이는 decoder 가 context 를 선택적으로 이용하고 있다는 의미입니다. 그럼에도 불구하고 기존의 sequence to sequence 에서는 하나의 벡터에 이 모든 정보를 표현하려 했으니, RNN 의 모델의 크기는 커야했고 성능도 낮을 수 밖에 없었습니다. Attention mechanism 이 성능 향상이 큰 도움을 주었습니다.

## Attention in Encoder - Decoder

얼마 지나지 않아서 attention mechanism 은 다른 encoder - decoder system 에도 이용되기 시작합니다. Xu et al., (2015) 에서는 이미지 파일을 읽어서 문장을 만드는 image captioning 에 attention mechanism 을 이용합니다. 일반적으로 image classification 을 할 때에는 CNN model 의 마지막 layer 의 concatenation 시킨 1 by k 크기의 flatten vector 를 이용하는데, 이 논문에서는 마지막 activation map 을 그대로 input 으로 이용합니다. activation map 역시 일종의 이미지입니다. Activation map 의 한 점은 이미지에서의 어떤 부분의 정보가 요약된 것입니다. 여전히 locality 가 보존되어 있는 tensor 입니다. 그리고 sequence to sequence 처럼 RNN 계열 모델을 이용한 language model 로 decoder 를 만듭니다. 이 때 attention weight 를 이용하여 마지막 activation map 의 어떤 부분을 봐야 하는지 결정합니다. 이는 실제 이미지의 특정 부분을 살펴보고서 단어를 선택한다는 의미입니다.

![]({{ "/assets/figures/attention_imagecaptioning_cnn_rnn_attention.png" | absolute_url }}){: width="90%" height="90%"}

그 결과 생성된 문장의 단어들이 높은 weight 로 이용한 이미지의 부분들을 시각적으로 확인할 수 있게 되었습니다. 실제로 이미지의 일부 정보를 이용하여 문장을 만들었습니다.

![]({{ "/assets/figures/attention_imagecaptioning_example_success.png" | absolute_url }}){: width="90%" height="90%"}

또한 모델이 엉뚱한 문장을 출력하였을 때, 그 부분에 대한 디버깅도 가능하게 되었습니다. 그리고 아래의 예시들은 실제로 사람도 햇갈릴법한 형상들입니다. 모델이 잘못된 문장을 생성하는게 오히려 이해가 되기 시작합니다.

![]({{ "/assets/figures/attention_imagecaptioning_example_fail.png" | absolute_url }}){: width="90%" height="90%"}

이처럼 encoder - decoder system 에서 decoder 가 특정 정보를 선택적으로 이용해야 하는 문제에서 attention mechanism 이 이용될 수 있습니다.


## Attention in Sentence classification

![]({{ "/assets/figures/attention_structured_attention_fig0.png" | absolute_url }})
![]({{ "/assets/figures/attention_structured_attention_fig1.png" | absolute_url }})
![]({{ "/assets/figures/attention_structured_attention_fig2.png" | absolute_url }})
![]({{ "/assets/figures/attention_structured_attention_fig3.png" | absolute_url }})
![]({{ "/assets/figures/attention_structured_attention_positive_example.png" | absolute_url }})

## Attention in Document classification

![]({{ "/assets/figures/attention_han_structure.png" | absolute_url }})
![]({{ "/assets/figures/attention_han_example.png" | absolute_url }})
![]({{ "/assets/figures/attention_han_attention_debugging.png" | absolute_url }})

## Transformer (self-attention)

![]({{ "/assets/figures/attention_transformer_components.png" | absolute_url }})
![]({{ "/assets/figures/attention_transformer_block_scaledot.png" | absolute_url }})
![]({{ "/assets/figures/attention_transformer_block_feedforward.png" | absolute_url }})
![]({{ "/assets/figures/attention_transformer_block_residual.png" | absolute_url }})
![]({{ "/assets/figures/attention_transformer_block_decoder.png" | absolute_url }})
![]({{ "/assets/figures/attention_transformer_encoder_decoder_attention.png" | absolute_url }})
![]({{ "/assets/figures/attention_transformer_components2.png" | absolute_url }})

![]({{ "/assets/figures/attention_transformer_block_selfattention_5_to_6_end_to_french.png" | absolute_url }})

## BERT (language model using transformer)

![]({{ "/assets/figures/attention_bert_input.png" | absolute_url }})
![]({{ "/assets/figures/attention_bert_usage.png" | absolute_url }})


## References

- Bahdanau, D., Cho, K., & Bengio, Y. (2014). [Neural machine translation by jointly learning to align and translate.](https://arxiv.org/abs/1409.0473) arXiv preprint arXiv:1409.0473.
- Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). [A neural probabilistic language model.](http://www.jmlr.org/papers/v3/bengio03a.html) Journal of machine learning research, 3(Feb), 1137-1155.
- Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). [Learning phrase representations using RNN encoder-decoder for statistical machine translation.](https://arxiv.org/abs/1406.1078) arXiv preprint arXiv:1406.1078.
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). [Bert: Pre-training of deep bidirectional transformers for language understanding.](https://arxiv.org/abs/1810.04805) arXiv preprint arXiv:1810.04805.
- Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2016). [Bag of tricks for efficient text classification.](https://arxiv.org/abs/1607.01759) arXiv preprint arXiv:1607.01759.
- Lin, Z., Feng, M., Santos, C. N. D., Yu, M., Xiang, B., Zhou, B., & Bengio, Y. (2017). [A structured self-attentive sentence embedding.](https://arxiv.org/abs/1703.03130) arXiv preprint arXiv:1703.03130.
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). [Efficient estimation of word representations in vector space.](https://arxiv.org/abs/1301.3781) arXiv preprint arXiv:1301.3781.
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). [Sequence to sequence learning with neural networks.](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks) In Advances in neural information processing systems (pp. 3104-3112).
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). [Attention is all you need.](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) In Advances in Neural Information Processing Systems(pp. 6000-6010).
- Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhudinov, R., ... & Bengio, Y. (2015, June). [Show, attend and tell: Neural image caption generation with visual attention.](http://proceedings.mlr.press/v37/xuc15.pdf) In International conference on machine learning (pp. 2048-2057).
- Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016). [Hierarchical attention networks for document classification.](http://www.aclweb.org/anthology/N16-1174) In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (pp. 1480-1489).
