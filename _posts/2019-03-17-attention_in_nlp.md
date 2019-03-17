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

![]({{ "/assets/figures/seq2seq.png" | absolute_url }})

Sequence to sequence 가 학습하는 기준은 $$maximize \Sum P_{\theta} \left( y_{1:m} \vert x_{1:n} \right)$$ 입니다. Input sequence $$x_{1:n}$$ 과 output sequence $$y_{1:m}$$ 의 상관성을 최대화 하는 것입니다. 이때 sequence to sequence 는 input sequence 의 정보를 하나의 context vector $$c$$ 에 저장합니다. Encoder RNN 의 마지막 hidden state vector 를 $$c$$ 로 이용하였습니다. Decoder RNN 은 고정된 context vector $$c$$ 와 현재까지 생성된 단어열 $$y_{1:i-1}$$ 을 이용하는 language model (sentence generator) 입니다.

$$P(y_{1:m} \vert x_{1:n}) = \prod_i P(y_i \vert y_{1:i-1}), c)$$ 물론 이 구조만으로도 번역의 성능은 향상되었습니다. Mikolov 의 언급처럼 word embedding 정보를 이용하였기 때문입니다. 기존의 statistical machine translation (classic n-grams 을 이용하는 방식) 보다 작은 크기의 모델 안에 단어 간의 semantic 정보까지 잘 포함되었기 때문입니다.

![]({{ "/assets/figures/seq2seq_fixed_context.png" | absolute_url }})

그런데, Bahdanau et al., (2014) 에서 하나의 문장에 대한 정보를 하나의 context vector $$c$$ 로 표현하는 것이 충분하지 않다고 문제를 제기합니다.

*A potential issue with this encoder–decoder approach is that a neural network needs to be able to compress all the necessary information of a source sentence into a fixed-length vector.*
{: .text-center }

*Instead, it encodes the input sentence into a sequence of vectors and chooses a subset of these vectors adaptively while decoding the translation. This frees a neural translation model from having to squash all the information of a source sentence, regardless of its length, into a fixed-length vector.*
{: .text-center }

![]({{ "/assets/figures/seq2seq_with_attention.png" | absolute_url }})

![]({{ "/assets/figures/seq2seq_structure.png" | absolute_url }})

![]({{ "/assets/figures/seq2seq_attention_structure.png" | absolute_url }})

![]({{ "/assets/figures/seq2seq_attention_input.png" | absolute_url }})

![]({{ "/assets/figures/seq2seq_attention_visualize.png" | absolute_url }})


![]({{ "/assets/figures/attention_imagecaptioning_cnn_rnn_attention.png" | absolute_url }})
![]({{ "/assets/figures/attention_imagecaptioning_example_success.png" | absolute_url }})
![]({{ "/assets/figures/attention_imagecaptioning_example_fail.png" | absolute_url }})


![]({{ "/assets/figures/attention_han_example.png" | absolute_url }})

![]({{ "/assets/figures/attention_structured_attention_fig0.png" | absolute_url }})
![]({{ "/assets/figures/attention_structured_attention_fig1.png" | absolute_url }})
![]({{ "/assets/figures/attention_structured_attention_fig2.png" | absolute_url }})
![]({{ "/assets/figures/attention_structured_attention_fig3.png" | absolute_url }})
![]({{ "/assets/figures/attention_structured_attention_positive_example.png" | absolute_url }})

![]({{ "/assets/figures/attention_han_structure.png" | absolute_url }})
![]({{ "/assets/figures/attention_han_attention_debugging.png" | absolute_url }})

![]({{ "/assets/figures/attention_transformer_components.png" | absolute_url }})
![]({{ "/assets/figures/attention_transformer_block_scaledot.png" | absolute_url }})
![]({{ "/assets/figures/attention_transformer_block_feedforward.png" | absolute_url }})
![]({{ "/assets/figures/attention_transformer_block_residual.png" | absolute_url }})
![]({{ "/assets/figures/attention_transformer_block_decoder.png" | absolute_url }})
![]({{ "/assets/figures/attention_transformer_encoder_decoder_attention.png" | absolute_url }})
![]({{ "/assets/figures/attention_transformer_components2.png" | absolute_url }})

![]({{ "/assets/figures/attention_transformer_block_selfattention_5_to_6_end_to_french.png" | absolute_url }})

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
