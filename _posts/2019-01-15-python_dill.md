---
title: Python dill 로 class definition 까지 binary 로 저장하기
date: 2019-01-15 09:00:00
categories:
- analytics
tags:
- analytics
---

파이썬으로 작업을 할 때, 사용자가 정의하는 클래스 인스턴스를 저장할 일들이 있습니다. 예를 들면 namedtuple 을 이용한 데이터 타입이라던지, PyTorch 에서 nn.Module 을 상속받은 모델들이 그 예입니다. 물론 site-packages 폴더에 설치된 클래스들은 pickle 을 이용하여 저장/로딩에 문제가 없지만, 때때로 패키지에 없는 클래스를 만들일이 있습니다. 이처럼 serializable 하지 않은 변수들은 pickle 을 이용하여 저장할 수 없습니다. dill 은 이 때 사용할 수 있는 파이썬 패키지 입니다.

## Python pickle

파이썬의 변수 혹은 클래스 인스턴스를 binary 형태로 저장하는 패키지 입니다. 데이터 분석 과정에서 만들어지는 데이터나 모델들을 텍스트 형식이 아닌 binary 형식으로 파일에 저장하고, 이후에 이 파일을 읽어 작업을 이어서 할 수 있습니다. `pickle.dump(object, filepath)` 를 이용하여 변수를 저장하고 `pickle.load(filepath)` 를 이용하여 저장된 변수를 읽을 수 있습니다. 이때 file open 시 mode 를 'wb' 와 'rb' 를 입력해야 합니다. `'wb` 는 write binary, `rb` 는 read binary 를 의미합니다.

```python
import numpy as np
import pickle

x = np.random.random_sample((5,3))
with open('x.pkl', 'wb') as f:
    pickle.dump(x, f)

with open('x.pkl', 'rb') as f:
    x_2 = pickle.load(f)
```

그런데 pickle 이 가능한 변수들은 serializable 한 값들입니다. lambda 를 이용한 함수나, 사용자가 직접 만든 클래스 인스턴스는 pickling 이 되지 않습니다. 아래의 구문을 실행시키면 pickling 이 되지 않는다는 에러 메시지가 출력됩니다.

```python
import pickle
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(tokenizer=lambda x:x.split())
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
```

```
---------------------------------------------------------------------------
PicklingError                             Traceback (most recent call last)
<ipython-input-7-deb29c8d21cc> in <module>()
PicklingError: Can't pickle <function <lambda> at 0x7f56054ce400>: attribute lookup <lambda> on __main__ failed
```

혹은 IPython notebook 에서 아래와 같이 사용자가 클래스를 하나 만든 뒤 pickling 을 하고, 다른 notebook 파일에서 이를 pickle.load 하면 에러 메시지가 발생합니다.

```python
from collections import namedtuple
import pickle

class Dataset(namedtuple('Dataset', 'x0 x1 y')):
    def __repr__(self):
        return 'x=({:.3}, {:.3}), y={}'.format(self.x0, self.x1, self.y)

data = [
    Dataset(0.123, 1.234, 5),
    Dataset(2.543, 1.2432, 3)
]

with open('./data.pkl', 'wb') as f:
    pickle.dump(data, f)
```

```python
# another file
import pickle
with open('./data.pkl', 'rb') as f:
    data = pickle.load(f)
```

```
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
<ipython-input-11-2b81f79886b3> in <module>()
      2 import dill
      3 with open('./data.pkl', 'rb') as f:
----> 4     data = pickle.load(f)

ModuleNotFoundError: No module named 'data'
```

이처럼 serializable 하지 않은 변수들을 binary 로 저장하기 위해서는 dill 을 이용해야 합니다.

## Python dill

`dill` 파이썬 패키지를 이용하면 serializable 하지 않은 값들도 손쉽게 binary 로 저장이 가능합니다. 인터페이스는 pickle 과 동일합니다.

```python
from collections import namedtuple
import dill

class Dataset(namedtuple('Dataset', 'x0 x1 y')):
    def __repr__(self):
        return 'x=({:.3}, {:.3}), y={}'.format(self.x0, self.x1, self.y)

data = [
    Dataset(0.123, 1.234, 5),
    Dataset(2.543, 1.2432, 3)
]

with open('./data.pkl', 'wb') as f:
    dill.dump(data, f)
```

```python
# another file
import dill
with open('./data.pkl', 'rb') as f:
    data = dill.load(f)
```

이번에는 문제없이 data 가 저장되고 로딩됨을 볼 수 있습니다. dill 은 interpretor session 자체를 저장하고 로딩하기 때문입니다. dill 은 PyPi 등록도 되어 있기 때문에 설치도 간단합니다.

```
pip install dill
```

그러나 source code 를 본다던지 같은 클래스의 더 많은 인스턴스를 만드는 것은 어려워 보입니다. dill 의 본래 목적이 네트워크 상에서 불특정 사용자에게 파이썬 변수의 값을 전달하는게 목적이었다 하니 그 용도로는 충분하다 생각됩니다.