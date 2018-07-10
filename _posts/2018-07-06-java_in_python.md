---
title: Java in Python, Komoran 3 를 Python package 로 만들기
date: 2018-07-06 14:00:00
categories:
- nlp
tags:
- preprocessing
---

자연어처리를 위한 형태소 분석기들 중에는 Java 로 구현된 것들이 많습니다. 알고리즘을 Python 으로 옮겨 재구현하지 않고도 Python 환경에서 이를 이용할 수 있습니다. KoNLPy 는 다양한 언어로 구현된 형태소 분석기들을 Python 환경에서 이용할 수 있도록 도와줍니다. 여기에서 Jpype 는 Java 구현체들을 Python 에서 이용할 수 있도록 두 언어를 연결해줍니다. [이전 포스트][komoran_before]에서 Jupyter notebook 의 Python 환경에서 Komoran 을 이용하는 과정을 이야기하였습니다. 이번 포스트에서는 Komoran 3 을 Python package 로 만드는 과정에 대하여 이야기합니다. 특히 JVM starting 과 호환 가능한 데이터 형식에 대하여 이야기합니다.

## Use Komoran in Jupyter notebook (이전 포스트 요약)

코모란은 Java 로 구현된 한국어 형태소 분석기입니다. shin285 의 github 에는 [version 3.x][komoran3] 가 공개되어 있습니다. 

[이전 포스트][komoran_before]에서 Java project 를 JAR 파일로 만드는 방법에 대하여 이야기하였습니다. 이 과정은 이전 포스트를 참고하세요. 

Java 구현체를 Python 에서 이용하기 위해서는 Jpype2 가 필요합니다. pypi 에 등록된 이름과 package 이름이 다릅니다.

	pip install Jpype2

Jpype 는 Python 에서 JVM 을 띄운 뒤, 서로 통신을 하는 라이브러리입니다. JVM 을 띄울 때 우리가 이용할 libraries 를 모두 입력합니다. 이를 위하여 앞서 jar 파일을 만들었습니다. 

코모란은 네 개의 학습 파일을 가지고 있습니다. Java class instance 를 만들 때, 이 파일들이 포함되어 있는 디렉토리 주소를 넣어줘야 합니다.

    irregular.model
    observation.model
    pos.table
    transition.model

소스 파일은 JAR 로 압축하였습니다. 이전 포스트에서 Jupyter notebook 에서 이용할 수 있는 Komoran class 를 만들었습니다.

{% highlight python %}
import jpype
import os

class Komoran:
    def __init__(self, model_path='./komoran/models',
                 library_directory = './komoran/libs', max_memory=1024):
        
        libraries = [
            '{}/aho-corasick.jar',
            '{}/shineware-common-1.0.jar',
            '{}/shineware-ds-1.0.jar',
            '{}/komoran-3.0.jar'
        ]
        
        classpath = os.pathsep.join([lib.format(library_directory) for lib in libraries])
        jvmpath = jpype.getDefaultJVMPath()
        
        try:
            jpype.startJVM(
                jvmpath,
                '-Djava.class.path=%s' % classpath,
                '-Dfile.encoding=UTF8',
                '-ea', '-Xmx{}m'.format(max_memory)
            )
        except Exception as e:
            print(e)
    
        package = jpype.JPackage('kr.co.shineware.nlp.komoran.core')
        self.komoran = package.Komoran(model_path)
        
    def set_user_dictionary(self, path):
        self.komoran.setUserDic(path)
    
    def pos(self, sent):
        tokens = self.komoran.analyze(sent).getTokenList()
        tokens = [(token.getMorph(), token.getPos()) for token in tokens]
        return tokens
{% endhighlight %}

## 이전 포스트의 문제점

### One JVM in one Python kernel

Jpype 는 하나의 Python kernel 에 하나의 JVM 을 만듭니다. 만약 두 개 이상의 JVM 을 실행하려하면 이전의 JVM 도 제대로 작동하지 않습니다. 아래처럼 JVM 이 실행중인지 확인하는 부분이 필요합니다.

{% highlight python %}
import jpype

if jpype.isJVMStarted():
    try:
        jpype.startJVM(
            jvmpath,
            '-Djava.class.path=%s' % classpath,
            '-Dfile.encoding=UTF8',
            '-ea', '-Xmx{}m'.format(max_memory)
        )
    except Exception as e:
        print(e)
{% endhighlight %}

위 코드에서 알 수 있듯이 새로은 class path 를 입력하기 위해서는 JVM 을 shutdown 한 뒤, 새로 start 해야 합니다.

{% highlight python %}
import jpype

def restartJVM(jvmpath, classpath, max_memory):

    if jpype.isJVMStarted():
        jpype.shutdownJVM()

    try:
        jpype.startJVM(
            jvmpath,
            '-Djava.class.path=%s' % classpath,
            '-Dfile.encoding=UTF8',
            '-ea', '-Xmx{}m'.format(max_memory)
        )
    except Exception as e:
        print(e)
{% endhighlight %}

만약 사용하는 Python kernel 에서 Jpype 를 이용하는 패키지가 둘 이상이라면 충돌이 날 수 있습니다. 각 패키지들이 이용하는 java class paths 를 모두 합쳐 먼저 하나의 JVM 을 띄워야 합니다.

### Komoran model directory path (absolute path)

어떤 이유인지는 아직 파악하지 못했지만, 위의 Python class 를 .py 파일로 만들었을 때에는 package.Komoran(model_path) 가 제대로 실행되지 않았습니다. 품사판별을 하면 Null point exception 이 일어났습니다.

{% highlight python %}
package = jpype.JPackage('kr.co.shineware.nlp.komoran.core')
komoran = package.Komoran(model_path)
{% endhighlight %}

원인을 찾아보니, package.Komoran(model_path) 에 입력되는 model_path 의 주소를 절대주소로 입력해야 했습니다. Python package 로 만들고 있기 때문에 우리는 패키지의 설치 폴더 주소를 알 수 있습니다. 

{% highlight python %}
installpath = os.path.dirname(os.path.realpath(__file__))
model_path = '%s/models/' % installpath
package = jpype.JPackage('kr.co.shineware.nlp.komoran.core')
komoran = package.Komoran(model_path)
{% endhighlight %}

이 생각은 KoNLPy 를 공부하다가 하게 되었습니다 (역시 좋은 reference 가 있어야 공부를 효율적으로 할 수 있네요).

## Komoran 3 를 Python package 로 만들기

위의 두 가지 문제점을 해결하니, Komoran 3 를 Python package 로 만들 수 있게 되었습니다. 최대한 KoNLPy 의 코드를 따라가도록 하였습니다. 

패캐지는 세 개의 파일과 두 개의 폴더로 구성합니다. JVM 을 관리하는 부분과 Komoran class 를 두 개의 파일로 분리하였습니다.

    # files
    --| __init__.py
    --| jvm.py
    --| komoran.py

    # directories
    --| java # JAR file 이 포함된 폴더
    --| models # 네 개의 Komoran 학습 모델

jpype 를 이용하는 jvm.py 를 먼저 만듭니다. init_jvm() 이라는 함수를 만듭니다. 만약 JVM 을 이미 이용하고 있다면, init 을 시키지 않습니다.

{% highlight python %}
if jpype.isJVMStarted():
    return None
{% endhighlight %}

그 외는 앞서 언급한 것처럼 os.path.dirname 과 os.path.realpath 를 이용하여 파일이 설치된 절대 경로를 얻고, 이 경로 뒤에 JAR 파일이 들어있는 java directory 의 주소를 추가하여 class paths 를 만듭니다.

{% highlight python %}
import os
import sys
import jpype

def init_jvm(libraries=None, max_heap=1024):
    """Initializes the Java virtual machine (JVM).
    use Java in jpype.getDefaultJVMPath
    """

    if jpype.isJVMStarted():
        return None

    if not libraries:
        installpath = os.path.dirname(os.path.realpath(__file__))
        libpaths = [
            '{0}',
            '{0}{1}bin',
            '{0}{1}aho-corasick.jar',
            '{0}{1}shineware-common-1.0.jar',
            '{0}{1}shineware-ds-1.0.jar',
            '{0}{1}komoran-3.0.jar',
            '{0}{1}*'
        ]
        javadir = '%s%sjava' % (installpath, os.sep)

    args = [javadir, os.sep]
    libpaths = [p.format(*args) for p in libpaths]
    classpath = os.pathsep.join(libpaths)
    jvmpath = jpype.getDefaultJVMPath()

    try:
        jpype.startJVM(
            jvmpath,
            '-Djava.class.path=%s' % classpath,
            '-Dfile.encoding=UTF8',
            '-ea', '-Xmx{}m'.format(max_heap)
        )
    except Exception as e:
        print(e)
{% endhighlight %}

komoran.py 파일에 Komoran class 를 만듭니다. jvm.py 의 init_jvm 함수를 import 합니다.

model_path 에 절대 주소를 입력하기 위하여 installpath 와 동일한 방법으로 komoran.py 파일의 절대 주소를 확인한 뒤, models 폴더의 주소를 추가합니다.

{% highlight python %}
model_path = os.path.dirname(os.path.realpath(__file__)) + '/models/'
{% endhighlight %}

그 외에는 [이전 포스트][komoran_before]와 동일하게 set_user_directory() 와 pos(sent) 함수를 구현합니다.

{% highlight python %}
import os
import jpype
from .jvm import init_jvm

class Komoran:

    def __init__(self):

        init_jvm()
        package = jpype.JPackage('kr.co.shineware.nlp.komoran.core')
        model_path = os.path.dirname(os.path.realpath(__file__)) + '/models/'
        self._komoran = package.Komoran(model_path)

    def set_user_dictionary(self, path):
        """
        Arguments
        ---------
        path : str
            dictionary file path
        """
        self._komoran.setUserDic(path)

    def pos(self, sent):
        tokens = self._komoran.analyze(sent).getTokenList()
        tokens = [(token.getMorph(), token.getPos()) for token in tokens]
        return tokens
{% endhighlight %}

### 구현체 및 데모

위의 내용을 정리한 Python 에서의 Komoran 3 의 코드는 [github][komoran3py] 에 있습니다. 이를 직접 이용해봅니다.

설치는 git clone 을 합니다.

    git clone https://github.com/lovit/komoran3py.git

{% highlight python %}
from komoran3py import Komoran
komoran = Komoran()

sent = '청하는아이오아이멤버입니다'
print(komoran.pos(sent))
{% endhighlight %}

단어 '아이오아이'는 미등록단어로, 제대로 인식이 되지 않습니다.

    [('청하', 'VV'),
     ('는', 'ETM'),
     ('아이오', 'NNP'),
     ('아이', 'NNP'),
     ('멤버', 'NNP'),
     ('이', 'VCP'),
     ('ㅂ니다', 'EC')]

사용자 사전을 만듭니다. 사용자 사전은 tap separated 텍스트 파일입니다. <단어, 품사> 형식으로 입력합니다. 아래와 같은 1 개의 단어가 있는 user_dictionary.txt 파일을 만들었습니다.

    아이오아이   NNP

띄어쓰기가 있는 구문을 추가할 수도 있습니다

    바람과 함께 사라지다   NNP

사전을 추가하여 다시 형태소 분석을 수행합니다.

{% highlight python %}
komoran.set_user_dictionary('./user_dictionary.txt')
print(komoran.pos(sent))
{% endhighlight %}

단어 '아이오아이'가 제대로 인식됩니다.

    [('청하', 'VV'),
     ('는', 'ETM'),
     ('아이오아이', 'NNP'),
     ('멤버', 'NNP'),
     ('이', 'VCP'),
     ('ㅂ니다', 'EC')]

komoran instance 를 새로 만들면, 사용자 사전의 정보는 초기화됩니다.

{% highlight python %}
komoran = Komoran()
print(komoran.pos(sent))
{% endhighlight %}

다시, 단어 '아이오아이'가 제대로 인식되지 않습니다.

    [('청하', 'VV'),
     ('는', 'ETM'),
     ('아이오', 'NNP'),
     ('아이', 'NNP'),
     ('멤버', 'NNP'),
     ('이', 'VCP'),
     ('ㅂ니다', 'EC')]

### 주의. Java compile version vs JRE version

작업을 했던 Mac 은 Java 1.8 을 이용하고 있었습니다. 동일한 코드를 Ubuntu, Java 1.7 에서 테스트하였더니 제대로 작동하지 않았습니다. 상위 버전의 Java 로 컴파일한 JAR 는 하위 버전의 Java 에서 작동하지 않을 수 있습니다. Ubuntu Java 를 1.8 로 업데이트 하였더니, 해당 코드는 작동하였습니다.

Ubuntu 에서 설치된 모든 Java versions 은 다음의 terminal command 로 볼 수 있습니다.

    update-java-alternatives -l

현재 이용하고 있는 Java version 은 다음으로 확인하 수 있습니다.

    java -version

버전을 바꾸고 싶다면 다음의 명령어를 입력한 뒤, 숫자로 원하는 Java version 을 선택합니다.

    sudo update-alternatives --config java 

컴파일러의 Java version 을 1.7 로 내려서 다시 컴파일하면, Java 1.7 에서도 잘 작동합니다. 그리고 Java 1.7 로 컴파일한 파일은 Java 1.8 에서도 제대로 작동합니다.

Java 1.7 로 컴파일한 Komoran 3 의 코드는 [github][komoran3py] 에 올려두었습니다.

## Jpype2 를 이용한 변수 변환. Converting Python variables to Java variables

앞의 Komoran class 파일에서 자연스럽게 넘어갔던 부분이 있습니다. pos() 함수는 str 형식의 sent 를 입력받습니다. self._komoran 은 Java class instance 입니다. self._komoran.analyze(sent) 에서 Python str 변수가 Java instance 에 입력되었습니다. 

그리고 그 결과를 getTokenList() 를 통하여 tokens 로 받았습니다. tokens 는 Jpype 에 의하여 Python kernel 에 연결된 Java instance 입니다.

{% highlight python %}
def pos(self, sent):
    tokens = self._komoran.analyze(sent).getTokenList()
    ...
{% endhighlight %}

token.getMorph(), token.getPos() 의 return 값도 Java string 입니다. 하지만, 그 다음 줄에서 정의되는 tokens 는 Python 의 list of tuple of str 입니다.

{% highlight python %}
def pos(self, sent):
    ...
    tokens = [(token.getMorph(), token.getPos()) for token in tokens]
    ...
{% endhighlight %}

이는 Jpype 에 의하여 Java 의 String 과 Python 의 str 이 서로 호환되기 때문입니다. Jpype 의 [documents][jpype_type_matching] 에는 서로 상호 호환이 되는 변수들이 적혀있습니다.

다른 타입의 변수들도 서로 호환이 되는지 살펴보기 위해 두 개의 Java classes 를 만들었습니다. 

첫번째 class 는 String array 를 입력받아, string counting 을 하는 함수입니다.

{% highlight java %}
package io.github.lovit.java_in_python;

import java.util.HashMap;
import java.util.Map.Entry;

public class StringCount {

    public HashMap<String, Integer> count(String[] stringArray){
        HashMap<String, Integer> counter = new HashMap();
        for (String str : stringArray) {
            if (counter.containsKey(str))
                counter.put(str, counter.get(str) + 1);
            else
                counter.put(str, 1);
        }
        return counter;
    }
    
    public static void main(String[] args) {
        
        String[] stringArray = new String[]{"a", "a", "a", "b", "b", "c", "d", "a"};
        HashMap<String, Integer> counter = new StringCount().count(stringArray);
        for (Entry<String, Integer> entry : counter.entrySet()) {
            System.out.println(entry.getKey() + " : " + entry.getValue());
        }
    }
}
{% endhighlight %}

두번째 class 는 두 개의 int 를 입력받아, 이를 더하여 출력하는 함수입니다. 이를 위해 두 개의 함수를 만들었습니다. addInteger 는 class instance 를 만들어야 이용할 수 있는 함수이며, addIntegerStatic 은 static 함수입니다.

{% highlight java %}
package io.github.lovit.java_in_python;

public class CalculateFunctions {

    public static int addIntegerStatic(int a, int b) {
        return a + b;
    }
    
    public int addInteger(int a, int b) {
        return a + b;
    }
}
{% endhighlight %}

두 파일은 io.github.lovit.java_in_python 으로 package 를 설정하였습니다. 즉, 두 개의 Java source files 를 다음과 같이 만들었습니다.

    |- io - github - lovit - java_in_python - StringCount.java
    |- io - github - lovit - java_in_python - CalculateFunctions.java

이 파일을 JAR 로 만든 뒤, 앞선 예제처럼 Jpype 를 이용하여 이를 이용하는 JVM 을 띄웁니다. Java instance 인 calculator 와 string_counter 를 만듭니다.

{% highlight python %}
import jpype
import os

if not jpype.isJVMStarted(installpath):

    libpaths = [
        '{0}',
        '{0}{1}JavaInPython.jar',
        '{0}{1}*'
    ]
    javadir = '%s%sjava' % (installpath, os.sep)

    args = [javadir, os.sep]
    libpaths = [p.format(*args) for p in libpaths]
    classpath = os.pathsep.join(libpaths)
    jvmpath = jpype.getDefaultJVMPath()
    
    try:
        jpype.startJVM(
            jvmpath,
            '-Djava.class.path=%s' % classpath,
            '-Dfile.encoding=UTF8',
            '-ea', '-Xmx{}m'.format(1024)
        )
    except Exception as e:
        print(e)

package = jpype.JPackage('io.github.lovit.java_in_python')
calculator = package.CalculateFunctions()
string_counter = package.StringCount()
{% endhighlight %}

1 과 2를 더하는 함수를 작동하면 3 이 return 됩니다. Python 의 int 와 Java 의 int 가 Jpype 를 통하여 서로 호환이 됩니다.

{% highlight python %}
print(calculator.addInteger(1, 2))
# 3
{% endhighlight %}

Jpype 는 (Python int, Java int), (Python int, Java Boolean), (Python float, Java double)에 대해서 exact match 를 해줍니다. 하지만, Java long, float 등은 아래처럼 직접 그 형식을 정의해줘야 이용할 수 있습니다. 이에 대해서는 반드시 [documents][jpype_type_matching] 를 보시기 바랍니다.

{% highlight python %}
print(calculator.addInteger(jpype.JInt(1), jpype.JInt(2)))
# 3
{% endhighlight %}

그런데 Java class 의 static 함수는 이용할 수 없습니다. 

{% highlight python %}
calculator.addIntegerStatic(1, 2)
{% endhighlight %}

위 함수를 실행시키면 아래와 같은 Error message 가 출력됩니다.

    ---------------------------------------------------------------------------
    RuntimeError                              Traceback (most recent call last)
    <ipython-input-5-41247dccdf78> in <module>()
    ----> 1 calculator.addIntegerStatic(1, 2)

    RuntimeError: No matching overloads found. at native/common/jp_method.cpp:117

## 기본형 외의 Java 변수를 Python 변수로 변환하기

[documents][jpype_type_matching] 를 살펴보면, 기본형 외에도 Array 등 몇 가지 타입의 변수를 Python 으로 변환해줍니다. 하지만 HashMap 은 여기에 포함되지 않습니다. 

우리는 StringCount.java 에 String counting 을 하는 함수를 구현하였고, 그 결과는 HashMap<String, Integer> 형식입니다. 아래와 같은 함수를 만들었습니다. StringCount.count 함수의 결과를 받아 이를 python 의 dict 의 값으로 변환합니다.

{% highlight python %}
def count_string(strs):
    javaHashMap = string_counter.count(strs)
    counter_python = {}
    for javaEntry in javaHashMap.entrySet():
        key = javaEntry.getKey()
        value = javaEntry.getValue()

        counter_python[key] = value

    return counter_python
{% endhighlight %}

그리고 여기에 'a', 'b', 'c' 를 각각 2개, 1개, 1개 포함한 strs 를 입력합니다. list of str 이나 tuple of str 이어도 모두 동일하게 작동합니다.

{% highlight python %}
strs = ['a', 'a', 'b', 'c']
# strs = ('a', 'a', 'b', 'c')
count_string(strs)
{% endhighlight %}

그런데 그 결과는 우리가 예상한 결과가 아닙니다. Key 값은 Python str 로 변환되었는데, value 값이 Integer instance 의 메모리 주소로 출력됩니다. 아마도 StringCount.count 함수의 return 형식이 Jpype 에서 알아서 변환해주는 데이터 타입이 아니기 때문이라 생각됩니다.

    {'a': <jpype._jclass.java.lang.Integer at 0x1191607f0>,
     'b': <jpype._jclass.java.lang.Integer at 0x119160630>,
     'c': <jpype._jclass.java.lang.Integer at 0x119160828>}

어떻게 우리가 원하는 형식으로 변환할까 고민하다가, Java String 과 Python str 은 그대로 변환되기에 이를 이용하자는 생각을 하였습니다. Java Object 는 toString() 함수를 반드시 상속합니다. 이를 이용하여 java.lang.Integer 를 String 으로 변환한 뒤, 이를 Python int 로 casting 하였습니다.

{% highlight python %}
def count_string(strs):
    javaHashMap = string_counter.count(strs)
    counter_python = {}
    for javaEntry in javaHashMap.entrySet():
        key = javaEntry.getKey()
        value = javaEntry.getValue()

        counter_python[key] = int(value.toString())

    return counter_python
{% endhighlight %}

위의 예제를 다시 실행하니 아래와 같이 예상되는 결과가 나옴을 확인하였습니다.

{% highlight python %}
strs = ['a', 'a', 'b', 'c']

count_string(strs)
# {'a': 2, 'b': 1, 'c': 1}
{% endhighlight %}


[komoran3]: https://github.com/shin285/KOMORAN
[komoran3py]: https://github.com/lovit/komoran3py
[jpype_type_matching]: http://jpype.readthedocs.io/en/latest/userguide.html#type-matching
[komoran_before]: {{ site.baseurl }}{% link _posts/2018-04-06-komoran.md %}