---
title: Feed forward neural network 를 이용한 window classification 기반 Named Entity Recognition (Scikit-learn 의 minibatch style 구현, PyTorch 구현)
date: 2019-06-18 07:00:00
categories:
- nlp
tags:
- ner
---

몇 년 전 Richard Socher 가 진행한 Stanford CS224n, Natural Language Processing with Deep learning 강의에서 programing assignment 로 feed forward neural network 를 이용한 Named Entity Recognizer 가 나왔습니다. 객체명 인식을 위해서는 sequential labeling 알고리즘이 이용되지만, [이전 포스트][logistic_ner]에서 언급하였듯이 그 원리는 window classification 모델과 비슷합니다. Sequential labeling 은 조금 더 flexible 한 window 를 이용하는 객체명 인식기입니다. [이전 포스트][logistic_ner]에서는 logistic regression 을 이용한 window classification 기반 객체명 인식 모델을 만들었지만, 이번 포스트에서는 단어 임베딩 벡터를 활용하는 feed forward neural network 를 이용한 window classification 기반 객체명 인식 모델을 만들어봅니다. PyTorch 나 TensorFlow 는 다양한 딥러닝 모델들을 손쉽게 구현할 수 있도록 도와줍니다. 하지만 간단한 뉴럴 네트워크는 Scikit-learn 을 이용해서도 구축할 수 있습니다. 그러나 context words 가 임베딩 벡터열로 표현될 경우 학습 데이터의 크기가 매우 큽니다. Scikit-learn 에서도 이러한 문제를 해결하기 위하여 minibatch style learning 이 가능하도록 partial_fit 함수를 제공합니다. 이번 포스트에서는 Scikit-learn 과 PyTorch 두 가지 버전으로 feed forward neural network window classification 기반 객체명 인식 모델을 구현해 봅니다.

## Logistic Regression based Window Classification for Named Entity Recognition

Named Entity Recognition (NER) 은 문장에서 특정한 종류의 단어를 찾아내는 information extraction 문제 중 하나입니다. '디카프리오가 나온 영화 틀어줘'라는 문장에서 '디카프리오'를 사람으로 인식하는 것을 목표로 합니다. 단어열로 표현된 문장에 각 단어의 종류를 인식하는 sequential labeling 방법이 주로 이용되었습니다. 최근에는 LSTM-CRF 와 같은 Recurrent Neural Network 계열 방법도 이용되지만, 오래전부터 Conditional Random Field (CRF) 가 이용되었습니다. 특히 CRF 모델은 named entities 를 판별하는 규칙을 해석할 수 있다는 점에서 유용합니다. 

Sequential labeling 은 pos tagging 에 이용되는 알고리즘이기도 합니다. 주어진 형태소 열에서 각 형태소의 품사를 추정하는 것이 품사 판별이라면, 각 단어의 class 를 추정하는 것이 named entity recognition 입니다. 목적에 따라 tag set 의 크기가 pos tagging 보다 클 수도, 작을 수도 있습니다.

```
Word sequence : [디카프리오, 가, 나온, 영화, 틀어줘]
Tag sequence  : [명사, 조사, 동사, 명사, 동사]
NER tagging   : [People, X, X, X, Request]
```

Named Entity Recogntion 은 위의 예시처럼 sequential labeling 을 이용할 수도 있습니다. 하지만 문장 전체가 아닌, named entity 주변의 정보만을 이용할 수도 있습니다. 영화 도메인에서는 `[People] + 가 + 나온` 이라는 표현에서 `가` 앞에 출연 배우가 위치하기 때문입니다. Named entity 를 기술하는 정보는 앞/뒤에 등장하는 몇 개의 단어만으로도 충분합니다.

```
Word sequence : [디카프리오, 가, 나온]
Target word   : 디카프리오 / People
Features      : X[1] = '가' & X[2] = '나온'
```

[이전의 CRF 를 이용한 NER 포스트][crfner]를 살펴보면 실제로 CRF 가 학습하는 정보도 앞/뒤에 등장하는 단어입니다. 문장 내에서 멀리 떨어진 단어 간에 상관성이 없다면 labeling 작업에 이를 굳이 이용하지 않아도 괜찮습니다. Window classification 으로부터 얻을 수 있는 결과물 중 하나는 이러한 템플릿 입니다. `[People] + 가 + 나온` 이라는 pattern 을 얻을 수 있습니다. 그러나 logistic regression 과 같은 sparse vector 를 이용하는 모델의 경우, 비슷한 단어열을 지니는 여러 개의 패턴이 만들어 집니다. 예를 들어 `[People] + 가 + 나온` 과 `[Peopoe] + 가 + 출연한` 은 서로 비슷한 의미를 지닙니다. 그리고 `[Peopoe] + 가 + 출연한` 와 같은 패턴은 뒤에 등장하는 두 단어의 bigram 입니다. [이전 포스트][logistic_ner]에서는 bigram 까지 이용할 경우 features 의 개수가 지나치게 많아지기 때문에 unigram 만을 이용하여 모델을 만들었습니다. 하지만 패턴을 임베딩 벡터로 표현한다면 비슷한 문맥을 하나의 벡터열로 표현할 수도 있습니다.

앞, 뒤에 등장하는 두 개의 단어 [$$w_{i-2}, w_{i-1}, w_{i+1}, w_{i+2}$$] 를 문맥으로 입력하여 가운데 단어 $$w_i$$ 가 객체명인지를 판단하는 feed forward neural network 를 만듭니다. 이 객체명 인식 모델은 정해진 크기의 문맥 정보를 이용하기 때문에 비효율적입니다. 가변적인 크기의 문맥을 이용하기 위해서는 Convolutional Neural Network (CNN) 도 좋습니다. Locality 와 n-grams 관련된 정보를 모두 학습할 수 있기 때문입니다. CNN 을 이용하는 객체명 인식 모델은 뒤 이어 만들어 보고, 그 이전에 가장 간단한 뉴럴 네트워크를 이용하는 객체명 인식 모델을 만들어 봅니다. 이는 [Richard Socher 의 강의노트][socher]에서 나오는 숙제이기도 합니다.


## Scikit-learn 의 뉴럴 네트워크를 이용하여 minibatch style 로 구현하기

학습데이터는 단어 임베딩 벡터열로 표현되기 때문에 한 번에 메모리에 올리기 어렵습니다. PyTorch 에서는 DataLoader 를 이용하여 minibatch style 로 구현하는 것이 일반적이지만, 저는 Scikit-learn 을 이용하면서 minibatch style 로 구현한 일들이 적었습니다. 이번 포스트에서는 classifier 를 minibatch style 로 구현하는 연습을 해봅니다.

데이터셋은 [이전 데이터셋 포스트][dataset]에 설명한 `lovit_textmining_dataset` 을 이용합니다. 영화평 데이터는 (영화 id, 영화 평, 영화 평점) 의 3 columns 으로 이뤄진 파일이며, 토크나이징이 된 영화평 부분만 `texts` 로 읽어옵니다.

```python
from navermovie_comments import load_movie_comments

_, texts, _ = load_movie_comments(large = True, tokenize = 'soynlp_unsup')
```

Gensim 을 이용하여 Word2Vec 을 학습합니다. 학습에 이용한 Gensim 의 버전은 `3.6.0` 입니다. 학습한 모델에서 단어 리스트 `idx_to_vocab` 과 와 단어 임베딩 벡터 `wv` 를 가져옵니다.

```python
from gensim.models import Word2Vec

word2vec_model = Word2Vec(texts)
idx_to_vocab = word2vec_model.wv.index2word
wv = word2vec_model.wv.vectors

print(wv.shape) (93234, 100)
```

그런데 이전의 [Word2Vec 학습 시 최소 빈도수 설정에 관련된 포스트][word2vec_min_count]에서 언급하였듯이 빈도수가 작은 단어는 좋지 않은 임베딩 벡터를 지닙니다. 그리고 빈도수는 상대적이기 때문에 학습 후, 어느 정도 빈도수가 큰 단어의 임베딩 벡터만 이용하던지, 유사어들의 빈도수를 확인하여 유사어의 빈도수가 해당 단어보다 큰 단어들만 선택해야 합니다. 좋은 임베딩 벡터를 지니는 단어를 선택하는 방법은 이 포스트의 목적이 아니니, 이는 [이전의 포스트][word2vec_min_count]를 참고하세요. 이번에는 단어 빈도수가 큰 순서대로 상위 67999 개의 단어만을 이용하였습니다. 그리고 68000 번째 단어는 unknown 으로, 이는 zero vector 로 이용합니다.

```python
import numpy as np

idx_to_vocab = word2vec_model.wv.index2word
print(word2vec_model.wv.vocab[idx_to_vocab[68000]])
# Vocab(count:10, index:68000, sample_int:4294967296)

idx_to_vocab = word2vec_model.wv.index2word[:67999]
vocab_to_idx = {vocab:idx for idx, vocab in enumerate(idx_to_vocab)}
wv_ = np.vstack([wv[:67999], np.zeros((1, wv.shape[1]), dtype=wv.dtype)])
print(wv_.shape) # (68000, 100)
```

Logistic regression 을 이용하는 window classification 객체명 인식 모델에서 학습데이터를 선택했던 것처럼 이번에도 Word2Vec 의 유사어들을 부분적인 positive samples 로 이용합니다. 이에 대한 내용은 이전의 [logistic regression 을 이용하는 NER 포스트][logistic_ner]를 참고하세요.

```python
seed_words = {word for word, _ in word2vec_model.wv.most_similar('송강호', topn=100)}
seed_words.update({word for word, _ in word2vec_model.wv.most_similar('디카프리오', topn=100)})

print(len(seed_words)) # 172
```

학습 데이터 생성을 위한 `create_dataset` 함수를 만듭니다. 이 함수는 `texts` 가 입력되면 Word2Vec 처럼 문장의 부분을 취하여 한 단어 $$w_i$$ 의 앞, 뒤로 `window` 칸 만큼의 단어를 context words 로 이용합니다. 일단 context words 에 대한 index 를 `context` 로 만든 뒤, 해당 단어의 임베딩 벡터를 lookup 하여 concatenation 합니다. `seed_words` 가 하나라도 포함된 문장만 학습에 이용합니다. 그 외의 문장을 모두 이용하면 학습데이터의 크기가 매우 커지지만, 우리가 원하는 정보는 그리 많지 않기 때문입니다.

`encode` 함수에서 문장의 앞, 뒤에 window 만큼의 unknown vocab 을 추가합니다. 이는 context words 에 대한 padding 입니다. 이후 context_idxs 에서 단순히 list slicing 만 하여도 같은 크기의 input vector 를 만들 수 있습니다.

```python
    def encode(sent):
        idxs = [vocab_to_idx.get(w, n_vocabs) for w in sent]        
        idxs = [n_vocabs] * window + idxs + [n_vocabs] * window
        return idxs

    word_idxs = encode(sent)

    for i, word in enumerate(sent):
        # ...
        context_idxs = word_idxs[b:i+window] + word_idxs[i+window+1:e]
        context = np.hstack([wv_[idx] for idx in context_idxs])
```

`test_data` 는 학습이 아닌 객체명 탐색을 위한 데이터를 만들 때 True 로 설정합니다. `test_data = True` 이면 `seed_words` 가 포함되지 않은 문장에 대해서도 학습데이터를 만듭니다. `W` 는 각 $$x_i$$ 에 해당하는 $$w_i$$ 의 실제 단어 입니다.

```python
def create_dataset(vocab_to_idx, sents, seed_words, wv_, window=2, test_data=False):

    n_vocabs = len(vocab_to_idx)

    def contain_seed(words):
        for word in words:
            if word in seed_words:
                return True
        return False

    def encode(words):
        idxs = [vocab_to_idx.get(w, n_vocabs) for w in words]
        # padding
        idxs = [n_vocabs] * window + idxs + [n_vocabs] * window
        return idxs

    X = []
    W = []

    for sent in sents:

        words = sent.split()
        n_words = len(words)

        if (n_words == 1):
            continue
        if (not test_data) and (not contain_seed(words)):
            continue

        word_idxs = encode(words)

        for i, word in enumerate(words):
            if not (word in vocab_to_idx):
                continue

            b = i # i - window + window
            e = i + 2 * window + 1 # i + window + 1 + window

            context_idxs = word_idxs[b:i+window] + word_idxs[i+window+1:e]
            context = np.hstack([wv_[idx] for idx in context_idxs])
            X.append(context)
            W.append(word)

    X = np.vstack(X)
    Y = np.asarray([1 if w in seed_words else 0 for w in W], dtype=np.int)
    return X, Y, W
```

Scikit-learn 도 partial_fit 함수를 이용하면 minibatch style 로 구현할 수 있습니다. fit 함수는 모델을 처음 학습할 때 이용하며, partial_fit 은 한 번 학습된 모델을 추가로 학습할 때 이용합니다. 또한 아래처럼 이전에 만든 모델을 입력할 수 있도록 구현하면 이용하던 모델에 추가 학습도 가능합니다. 만약 `minibatch_style` 함수에 입력되는 `model` 이 None 이 아니라면 반드시 partial fit 만 이뤄지도록 `initialized` 라는 Flag 변수를 만들어 둡니다.

Classifier 를 만들 때 `max_iter=1` 로 설정하면 minibatch 처럼 만들 수 있습니다. Loss 는 positive class 의 데이터는 negative class 의 확률, negative class 의 데이터는 positive class 의 확률입니다. 이들을 모두 더하여 epoch 마다 출력도 합니다.

```python
def minibatch_style(model=None, ... ):

    if model is None:
        model = MLPClassifier(hidden_layer_sizes=hidden_size, activation='relu', max_iter=1)
```

```python
import math
from sklearn.neural_network import MLPClassifier

def minibatch_train(vocab_to_idx, sents, seed_words, wv_, model=None,
    n_batch_sents=250000, hidden_size=(50,), epochs=20, verbose=True):

    n_sents = len(sents)
    n_batchs = math.ceil(n_sents / n_batch_sents)

    initialized = model is None

    if model is None:
        model = MLPClassifier(hidden_layer_sizes=hidden_size, activation='relu', max_iter=1)

    for epoch in range(epochs):

        loss = 0
        n_instances = 0

        for batch in range(n_batchs):

            b = batch * n_batch_sents
            e = min((batch + 1) * n_batch_sents, n_sents)
            X, Y, words = create_dataset(vocab_to_idx, sents[b:e], seed_words, wv_)

            do_fit = initialized and (epoch == 0) and (batch == 0)
            if do_fit:
                print('Model is initialized')
                model.fit(X, Y)
            else:
                model.partial_fit(X, Y)

            prob = model.predict_proba(X)
            loss += prob[np.where(Y == 1)[0], 0].sum()
            loss += prob[np.where(Y == 0)[0], 1].sum()
            n_instances += X.shape[0]

            if verbose:
                avg_loss = loss / n_instances
                print('\rtrain epoch = {} / {}, batch = {} / {}, loss = {}'.format(
                    epoch+1, epochs, batch+1, n_batchs, avg_loss), end='')
        if verbose:
            print()

    return model
```

100,000 만줄씩 데이터를 잘라내어 학습데이터를 만들었습니다. 매 epoch 마다의 loss 도 확인할 수 있습니다. 메모리는 3.3 GB 정도를 이용했합니다. 대부분의 메모리는 Word2Vec 의 단어 벡터를 저장하는데 이용되었습니다.

```python
model = minibatch_train(vocab_to_idx, texts, seed_words, wv_)
# 
```

학습된 모델을 이용하여 모든 데이터에서 $$w_i$$ 가 positive class 로 분류되는 확률을 저장합니다. 이때도 minibatch style 로 prediction 을 할 수 있습니다.

```python
def minibatch_predict(vocab_to_idx, sents, seed_words, wv_, model, n_batch_sents=50000):
    y_prob = []
    y_words = []
    n_sents = len(sents)
    n_batchs = math.ceil(n_sents / n_batch_sents)

    for batch in range(n_batchs):

        b = batch * n_batch_sents
        e = min((batch + 1) * n_batch_sents, n_sents)
        X, _, words = create_dataset(vocab_to_idx, sents[b:e], seed_words, wv_, test_data=True)

        y_prob.append(model.predict_proba(X))
        y_words += words

        print('\rbatch prediction {} / {}'.format(batch+1, n_batchs), end='')
    print('\rbatch prediction {0} / {0}'.format(n_batchs))

    y_prob = np.vstack(y_prob)[:,1]
    y_words=  np.asarray(y_words)
    return y_prob, y_words

y_prob, y_words = minibatch_predict(vocab_to_idx, sents, seed_words, wv_, model)
```

`seed_words` 에 포함되지 않고 길이가 2 이상인 단어이며, 빈도수가 30 이상인 단어에 대하여 classification 시 positive class 로 분류된 확률이 0.6 보다 큰 경우의 비율이 큰 상위 300 개의 단어를 확인해 봅니다.

```
from collections import Counter
from collections import defaultdict

def extract(y_prob, y_words, min_prob=0.6, min_count=30, filtering=None):
    if filtering is None:
        filtering = lambda x: True

    # word count
    word_counter = Counter(y_words)

    # prediction count
    pred_pos = defaultdict(int)
    for row in np.where(y_prob >= min_prob)[0]:
        pred_pos[y_words[row]] += 1
    pred_pos = {word:(word_counter[word], pos/word_counter[word])
                for word, pos in pred_pos.items()}

    # filtering
    pred_pos = {w:p for w,p in pred_pos.items() if
                (word_counter[w] >= min_count) and filtering(w)}
    return pred_pos

filtering = lambda w: not (w in seed_words) and len(w) > 1
pred_pos = extract(y_prob, y_word, min_prob=0.6, filtering=filtering)

for word, (count, prob) in sorted(pred_pos.items(), key=lambda x:-x[1][1])[:300]:
    print('{} ({})\t{:.3f}'.format(word, count, prob))
```

## PyTorch 를 이용하여 구현하기

PyTorch 를 이용하는 경우에도 `seed_words` 를 선택하는 경우까지는 동일합니다. 그 외의 과정만 따로 기술합니다. PyTorch 의 버전은 `1.1.0` 입니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
torch.__version__ # '1.1.0'
```

`create_dataset` 함수는 단어 임베딩 벡터를 lookup 하는 부분 대신 단어 index 를 `torch.LongTensor` 로 변환하는 부분만 다릅니다. 단어 임베딩 벡터가 필요하지 않기 때문에 `wv_` 는 입력받지 않습니다.

```python
def create_dataset(vocab_to_idx, sents, seed_words, window=2, test_data=False):

    n_vocabs = len(vocab_to_idx)

    def contain_seed(words):
        for word in words:
            if word in seed_words:
                return True
        return False

    def encode(words):
        idxs = [vocab_to_idx.get(w, n_vocabs) for w in words]
        # padding
        idxs = [n_vocabs] * window + idxs + [n_vocabs] * window
        return idxs

    X = []
    W = []

    for sent in sents:

        words = sent.split()
        n_words = len(words)

        if n_words == 1:
            continue
        if (not test_data) and (not contain_seed(words)):
            continue

        word_idxs = encode(words)

        for i, word in enumerate(words):
            if not (word in vocab_to_idx):
                continue

            b = i # i - window + window
            e = i + 2 * window + 1 # i + window + 1 + window

            context = word_idxs[b:i+window] + word_idxs[i+window+1:e]
            X.append(np.asarray(context))
            W.append(word)

    X = np.vstack(X)
    X = torch.LongTensor(X)
    Y = np.asarray([1 if w in seed_words else 0 for w in W], dtype=np.int)
    Y = torch.LongTensor(Y)
    W = np.asarray([vocab_to_idx[w] for w in W])
    return X, Y, W

X, Y, W = create_dataset(vocab_to_idx, texts, seed_words)
print(Y.sum(), Y.size()) tensor(361787) torch.Size([5216400])
```

`seed_words` 가 포함된 문맥은 총 12,864 개 이며, 이들이 포함된 문장에서 생성된 문맥은 517,373 개 입니다. 앞서 scikit-learn 은 class weight 가 자동으로 보정되지만, PyTorch 에서는 이를 따로 정의해야 합니다. 학습데이터의 positive, negative 샘플 개수의 역수로 class weight 를 정의합니다.

```python
n_pos = int(Y.sum())
n_neg = Y.size()[0] - n_pos
n_sum = n_pos + n_neg
class_weight = torch.FloatTensor([n_pos/n_sum, n_neg/n_sum])

print(class_weight[0], class_weight[1]) # tensor(0.0694) tensor(0.9306)
```

학습 데이터는 shuffle 이 될 수 있도록 Dataset 으로 만든 뒤, DataLoader 에 태웁니다.

```python
class NERWindowDataset(Dataset):
    def __init__(self, x, y, w):
        """
        x : torch.LongTensor
            Context words. size = (n_data, 2 * window)
        y : torch.LongTensor
            Label. size = (n_data,)
        w : torch.LongTensor
            Target word idx. size = (n_data,)        
        """
        self.x = x
        self.y = y
        self.w = w

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.w[idx]

ner_dataloader = DataLoader(
    NERWindowDataset(X, Y, W),
    batch_size = 512,
    shuffle = True)
```

모델은 50 차원의 하나의 hidden layer 를 지니는 feed-forward neural network 로 설계하였습니다. 이 때 hidden 의 bias 는 이용하지 않습니다. 문맥 단어들의 벡터가 concatenated 된 값을 그대로 이용하기 위해서 입니다.

```python
class NamedEntityWindowClassifier(nn.Module):
    def __init__(self, wordvec, n_classes, hidden_1_dim=50, n_windows=2):
        super(NamedEntityWindowClassifier, self).__init__()

        self.n_windows = n_windows
        self.n_vocabs, self.embed_dim = wordvec.shape
        self.embed = nn.Embedding(num_embeddings = self.n_vocabs, embedding_dim = self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim * 2 * n_windows, hidden_1_dim, bias = False)
        self.fc2 = nn.Linear(hidden_1_dim, n_classes, bias = True)

    def forward(self, x):
        y = self.embed(x) # [batch, 2 * window, embed]
        y = y.view(y.size()[0], -1) # [batch, embed * 2 * widow]
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        return y
```

학습 함수는 아래처럼 간단히 만들었습니다. 외부에서 loss function 과 optimizer 를 설정하여 입력합니다.

```python
def train(data_loader, model, loss_func, optimizer, epochs):
    n_batchs = len(ner_dataloader)
    for epoch in range(epochs):
        loss_sum = 0
        for i, (x, y, _) in enumerate(data_loader):
            if int(y.sum()) == 0:
                continue
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_func(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.data.numpy()

        print('epoch = {}, training loss = {:.3}'.format(epoch, (loss_sum / (i+1)) ))
```

Adam optimizer 를 이용하여 모델을 학습합니다.

```python
# Parameter for the optimizer
learning_rate = 0.01

# Loss and optimizer
# loss_func = nn.CrossEntropyLoss()
loss_func = nn.CrossEntropyLoss(weight = class_weight)
model = NamedEntityWindowClassifier(wv_, hidden_1_dim=50, n_classes=2)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

model = train(ner_dataloader, model, loss_func, optimizer, epochs=5)
```

DataLoader 를 입력받아 `min_prob` 보다 큰 확률로 positive class 로 분류하는 횟수를 `pos`, 반대의 경우를 `neg` 에 저장합니다.

```python
def predict(model, ner_dataloader, n_vocabs, min_prob=0.6, debug=False):
    model.eval()
    pos = np.zeros(n_vocabs)
    neg = np.zeros(n_vocabs)
    n_data = 0
    for i, (x_batch, y_batch, w_batch) in enumerate(ner_dataloader):
        n_data += x_batch.size()[0]
        y_pred = F.softmax(model(x_batch))[:,1]
        y_pred = y_pred.data.numpy()
        w_batch = w_batch.numpy()

        pos_idx = w_batch[np.where(y_pred >= min_prob)[0]]
        pos_count = np.bincount(pos_idx, minlength=n_vocabs)
        pos = pos + pos_count

        neg_idx = w_batch[np.where(y_pred < min_prob)[0]]
        neg_count = np.bincount(neg_idx, minlength=n_vocabs)
        neg = neg + neg_count

        if debug and i >= 10:
            break

    model.train()
    if pos.sum() + neg.sum()!= n_data:
        raise RuntimeError('The number of prediction is different with the number of data')
    return pos, neg
```

# TODO: 전체 데이터에 대해서 테스트하기. 

`pos` 와 `neg` 를 이용하여 `min_count` 이상 등장한 단어의 positive class 인 확률을 계산합니다. 그리고 확률이 큰 상위 `topk` 를 선택합니다.

```python
def extract(pos, neg, idx_to_vocab, seed_words, min_count=40, filtering=None):
    if filtering is None:
        filtering = lambda w: len(w) > 1

    count = pos + neg
    score = pos / count
    score[np.where(score == np.inf)[0]] = 0
    score = np.nan_to_num(score)
    word_to_score = {vocab:(s, c) for vocab, s, c in zip(idx_to_vocab, score, count)}
    word_to_score = {vocab:sc for vocab, sc in word_to_score.items()
        if not (vocab in seed_words) and sc[1] >= min_count}
    word_to_score = {w:sc for w,sc in word_to_score.items() if filtering(w)}
    return word_to_score

def select_topk(word_to_score, topk=30):
    topks = list(sorted(word_to_score.items(), key=lambda x:-x[1][0]))[:topk]
    topks = ['%s (%d)' % (w, c) for w, (_, c) in topks]
    return topks
```

# TODO: 전체 데이터에 대해서 테스트하기. 

## Conclusion


## Reference
- [Lecture note of Richard Socher][socher]

[socher]: https://nlp.stanford.edu/~socherr/pa4_ner.pdf
[logistic_ner]: {{ site.baseurl }}{% link _posts/2019-02-16-logistic_w2v_ner.md %}
[word2vec_min_count]:  {{ site.baseurl }}{% link _posts/2018-12-05-min_count_of_word2vec.md %}