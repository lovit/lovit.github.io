---
title: [Spark] 0. Ubuntu 에 Spark 설치, IPython Notebook 의 외부접속 설정, PySpark 와 Notebook 연동
date: 2019-01-29 04:00:00
categories:
- spark
tags:
- spark
---

이번 포스터에서는 Ubuntu 에 Spark 를 설치하는 과정, 그리고 IPython Notebook 에서 Spark 를 이용하기 위하여 PySpark 를 설치하는 과정, 외부에서 IPython Notebook 을 이용할 수 있도록 설정을 하는 과정을 정리하였습니다.

## Introduction

제가 주로 하는 데이터 분석에는 dense matrix multiplication 과 같은 작업 보다는 텍스트 데이터를 카운팅 하거나 특정 정보를 탐색하는 일이 많습니다. 그리고 작업 시 효율적인 메모리 사용이나 알고리즘의 scalability 에 신경을 쓰는 편입니다. 그런 관점에서 Spark 는 도움이 되는 라이브러리이기 때문에 최근에 Spark 를 공부하기 시작했습니다. 이 포스트는 그 과정의 정리입니다.

이번 포스터에서는 Ubuntu 에 Spark 를 설치하는 과정, 그리고 IPython Notebook 에서 Spark 를 이용하기 위하여 PySpark 를 설치하는 과정, 외부에서 IPython Notebook 을 이용할 수 있도록 설정을 하는 과정을 정리하였습니다.

이전에는 PySpark 를 따로 설치하고 Spark 환경 변수를 설정하여 IPython Notebook 과 연결을 시켰던 것 같은데, 최근에는 PyPi 에 PySpark 가 등록되어 pip install 만으로 설치가 가능합니다.

## Ubuntu 에 Java 설치하기

Spark 는 Scala 를 이용합니다. Spark 를 설치하기 전에 Java 가 설치되었는지 확인해야 합니다.

아래처럼 자바 버전을 확인하면 자바가 없을 경우 이를 설치할 수 있는 명령어가 출력됩니다.

```
java -version
```

아래처럼 apt-get install 을 시도하였습니다.

```
sudo apt-get install openjdk-8-jdk-headless
```

그러나 sudo apt-get 이 제대로 작동하지 않은 경우도 있네요. 아래와 같은 에러 메시지가 출력되었는데, 찾아보니 이는 업데이트 해야 할 파일들의 링크가 깨져서 발생하는 오류인 듯 합니다. [(참고)](https://askubuntu.com/questions/483611/message-edpkg-was-interrupted-you-must-manually-run-sudo-dpkg-configure-a)

```
'E:dpkg was interrupted, you must manually run 'sudo dpkg --configure -a' to correct the problem.'
```

아래처럼 `/var/lib/dpkg/updates` 의 파일들을 지우고 다시 apt-get update 를 실행하니 apt-get 이 제대로 작동합니다.

```
cd /var/lib/dpkg/updates
sudo rm *
sudo apt-get update
```

설치 후 자바 1.8 이 설치되었음을 확인할 수 있도 있습니다.

```
$ java -version

openjdk version "1.8.0_191"
OpenJDK Runtime Environment (build 1.8.0_191-8u191-b12-0ubuntu0.16.04.1-b12)
OpenJDK 64-Bit Server VM (build 25.191-b12, mixed mode)
```

## Spark 설치하기

[Spark 홈페이지](http://spark.apache.org/downloads.html)에 들어가 원하는 버전을 선택합니다. Spark 파일의 링크를 복사한 다음 wget 을 이용하여 Spark 를 설치할 컴퓨터에 파일을 다운받습니다. 설치 파일이 미러링 되어 있어서 해당 주소를 복사하여 wget 을 실행합니다. 이 주소는 시기와 버전마다 다를 수 있습니다.

```
wget http://mirror.navercorp.com/apache/spark/spark-2.4.0/spark-2.4.0-bin-hadoop2.7.tgz
```

파일의 압축을 풀면 같은 이름의 `spark-2.4.0-bin-hadoop2.7` 폴더가 생성됩니다. 이 안에 Spark 파일들이 들어있고, Spark 의 설치는 이것으로 끝납니다.

```
tar -xf spark-2.4.0-bin-hadoop2.7.tgz
```

Spark 의 실행 파일들은 bin 폴더에, Spark 서버 관련 파일들은 sbin 안에 들어있습니다. Scala shell 인 spark-shell 을 실행시켜 봅니다. 실행 위치는 압축 파일이 풀린 위치 (Spark 홈 디렉토리) 입니다.

```
iam@spark/spark-2.4.0-bin-hadoop2.7$ bin/spark-shell
```

여러 메시지가 뜨며, 아래처럼 Web UI 가 구동 중임을 보여줍니다. 4040 은 Spark UI 의 기본 port 입니다.

```
Spark context Web UI available at http://xxx.xxx.xxx.xxx:4040
Spark context available as 'sc' (master = local[*], app id = local-xxxxxxxx).
Spark session available as 'spark'.
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /___/ .__/\_,_/_/ /_/\_\   version 2.4.0
      /_/

Using Scala version 2.11.12 (OpenJDK 64-Bit Server VM, Java 1.8.0_191)
Type in expressions to have them evaluated.
Type :help for more information.

scala>
```

pyspark 도 가동해봅니다. 이 역시 Web UI 의 주소를 보여주며, 제대로 작동함을 확인할 수 있습니다.

```
iam@spark/spark-2.4.0-bin-hadoop2.7$ bin/pyspark
```

![]({{ "/assets/figures/spark-4040.png" | absolute_url }}){: width="70%" height="70%"}

## PySpark 설치 및 IPython Notebook 에서 사용하기

PySpark 는 Scala 로 작성된 Spark 를 Python 환경에서 이용할 수 있도록 도와줍니다. 찾아보니 Py4J 라는 라이브러리를 이용하여 Java 와 Python 간에 연결을 시켜둔 것 같습니다.

이전에는 PySpark 를 직접 다운로드 받고 Spark 환경변수를 연결하여 설치를 했던 것 같은데, 지금은 pip install 로 설치할 수 있습니다.

```
pip install pyspark
```

PySpark 는 Py4J 를 이용하기 때문에 함께 설치 되는 것도 확인할 수 있습니다.

```
Successfully built pyspark
Installing collected packages: py4j, pyspark
Successfully installed py4j-0.10.7 pyspark-2.4.0
```

Ipython shell 에서 pyspark 가 설치되었는지 확인해 봅니다.

```
$ ipython
[1] import pyspark
[2] pyspark.__version__
Out[3] '2.4.0'
```

## IPython Notebook 외부 접속 가능하게 만들기

IPython Notebook 은 별도의 설정이 없으면 localhost 로만 Notebook server 를 띄어줍니다. 외부 접속을 위해서는 몇 가지 설정이 필요합니다. 일단 password 설정부터 시작합니다.

IPython kernel 을 실행시켜 password 를 입력합니다.

```
$ ipython
```

```python
from notebook.auth import passwd

passwd()
```

`sha1:xxxx` 처럼 암호화 되어있는 값이 출력되는데 이를 복사해 둡니다. IPython kernel 을 종료하고 (exit) 아래의 command 를 입력합니다. IPython Notebook configuration 파일을 만들어줍니다. `home/.jupyter` 안에 `jupyter_notebook_config.py` 파일이 만들어집니다. 

```
jupyter notebook --generate-config
cd ~/.jupyter
```

아래의 `jupyter_notebook_config.py` 파일에서 네 개의 값을 수정합니다. password 는 앞서 복사한 값입니다. ip 는 이용하는 컴퓨터의 고정 ip 이며, port 는 사용할 port 입니다. 기본은 8888 입니다. open_brower 를 False 로 지정하면 IPython notebook 을 기동할 때마다 기본 브라우저로 notebook 이 실행되는 것을 하지 않습니다.

```
c.NotebookApp.password = u'sha1:.....'
c.NotebookApp.ip = x.x.x.x
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
```

Notebook 의 시작 폴더를 고정하고 싶다면 아래의 값을 수정합니다. 수정하지 않으면 command 를 실행하는 위치에서 Notebook 이 실행됩니다.

```
c.NotebookApp.notebook_dir = ''
```

IPython notebook 에서 Python notebook 을 하나 만든 뒤, `import pyspark` 를 실행합니다. 정상적으로 작동함을 확인할 수 있습니다.