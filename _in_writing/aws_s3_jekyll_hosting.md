---
title: AWS S3 에 Jekyll 웹사이트 호스팅하기
date: 2019-01-17 23:00:00
categories:
- web
tags:
- web
---

## 웹사이트 파일 만들기

웹사이트 파일들을 관리할 폴더 `hyunjoong.kr` 을 만듭니다. README 파일을 만듭니다.

```
mkdir hyunjoong.kr
cd hyunjoong.kr
vi README.md
```


## S3 버킷 (bucket) 만들기 

bucket-name 은 전 세계적으로 고유해야 합니다. `hyunjoong.kr` 은 아무도 이용하지 않았습니다.

s3 bucket 에 업로드 되는 파일은 퍼블릭 권한이 있어야 다른 사람들이 볼 수 있습니다.

Bucket 에 업로드된 파일의 기본 주소는 아래와 같습니다.

```
<bucket-name>.s3-website-<AWS-region>.amazonaws.com
```

제 bucket 의 접근 주소는 아래와 같습니다.

```
http://examplebucket.s3-website-us-west-2.amazonaws.com/photo.jpg
```

## 로컬에서 s3 로 파일 업로드하기

### awscli

awscli 는 AWS Command Line UI 의 약어입니다. 설치하면 이후 s3 관련 작업을 terminal command 로 할 수 있습니다.

```
aws s3 ls s3://mybucket
```

설치는 pip install 로 가능합니다. Python 2.6x 혹은 Python 2.7x 이상이 필요하였는데, 저는 Python 3.x 만 이용하므로 버전은 자세히 보지 않았습니다.

```
pip install awscli
```

설치 후 help 명령어를 입력하여 awscli 가 설치되었는지 확인도 해봅니다.

```
aws help
```

