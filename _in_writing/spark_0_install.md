### Ubuntu 에 Java 설치하기

아래처럼 자바 버전을 확인하면 자바가 없을 경우 이를 설치할 수 있는 명령어가 출력된다.

```
java -version
```

아래처럼 apt-get install 을 시도하였다.

```
sudo apt-get install openjdk-8-jdk-headless
```

그러나 sudo apt-get 이 제대로 작동하지 않았다. 아래와 같은 에러 메시지가 출력되었다. 이는 업데이트 해야 할 파일들의 링크가 깨져서 발생하는 오류인 듯 하다 [(참고)](https://askubuntu.com/questions/483611/message-edpkg-was-interrupted-you-must-manually-run-sudo-dpkg-configure-a)

```
'E:dpkg was interrupted, you must manually run 'sudo dpkg --configure -a' to correct the problem.'
```

아래처럼 `/var/lib/dpkg/updates` 의 파일들을 지우고 다시 apt-get update 를 실행하니 작동한다.

```
cd /var/lib/dpkg/updates
sudo rm *
sudo apt-get update
```

설치 후 자바 1.8 이 설치되었음을 확인할 수 있다.

```
$ java -version

openjdk version "1.8.0_191"
OpenJDK Runtime Environment (build 1.8.0_191-8u191-b12-0ubuntu0.16.04.1-b12)
OpenJDK 64-Bit Server VM (build 25.191-b12, mixed mode)
```

### Spark 설치하기

[Spark 홈페이지](http://spark.apache.org/downloads.html)에 들어가 원하는 버전을 다운 받는다. Spark 파일의 링크를 복사한 다음 wget 을 이용하여 Spark 를 설치할 컴퓨터에 파일을 다운받는다. 설치 파일이 미러링 되어 있어서 해당 주소를 복사하여 wget 을 한다.

```
wget http://mirror.navercorp.com/apache/spark/spark-2.4.0/spark-2.4.0-bin-hadoop2.7.tgz
```

파일의 압축을 푼다. 같은 이름의 `spark-2.4.0-bin-hadoop2.7` 폴더가 생성되었다. 이 안에 Spark 파일들이 들어있다.

```
tar -xf spark-2.4.0-bin-hadoop2.7.tgz
```

Spark 의 실행 파일들은 bin 폴더에, Spark 서버 관련 파일들은 sbin 안에 들어있다. Scala shell 인 spark-shell 을 실행시켜본다. 실행 위치는 압축 파일이 풀린 위치 (Spark 홈 디렉토리) 이다.

```
iam@spark/spark-2.4.0-bin-hadoop2.7$ bin/spark-shell
```

여러 메시지가 뜨며, 아래처럼 Web UI 가 구동 중임을 보여준다. 4040 은 Spark UI 의 기본 port 이다.

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

pyspark 도 가동해보자. 이 역시 Web UI 의 주소를 보여주며, 제대로 작동함을 확인할 수 있다.

```
iam@spark/spark-2.4.0-bin-hadoop2.7$ bin/pyspark
```