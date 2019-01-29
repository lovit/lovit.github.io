---
title: AWS CLI (Command Line Interface) 를 이용하여 S3 버킷 다루기 (파일 업로드, 폴더 동기화) 및 AWS IAM 등록
date: 2019-01-30 05:00:00
categories:
- aws
tags:
- aws
---

[이전 포스트][aws_s3_release_files]에서 AWS S3 에 버킷을 만들고 Web UI 를 이용하여 파일을 업로드, 공유하였습니다. 이번에는 AWS CLI 를 이용하여 로컬과 S3 bucket 을 동기화 시킵니다. CLI 는 terminal 환경에서 AWS 를 이용할 수 있도록 도와줍니다.

AWS CLI 는 ubuntu 에서 파일 복사, 폴더 생성, 이동, 삭제 등에 이용되는 `ls`, `rm`, `cp` 와 같은 기능을 제공합니다. 이를 이용하려면 먼저 AWS IAM 을 등록해야 합니다. AWS 계정 안에서 각 목적에 따라 이용하는 사용자 계정이라 생각하면 됩니다.

### IAM 

AWS IAM 은 aws.amazon.com 에 로그인 한 뒤, EC2 나 S3 를 찾았듯이 검색하기로 찾을 수 있습니다.

![]({{ "/assets/figures/s3data_aws_main.png" | absolute_url }}){: width="70%" height="70%"}

IAM 에 처음 들어가면 등록된 사용자가 없습니다. 상단의 파란색 `사용자 추가` 버튼을 누릅니다.

![]({{ "/assets/figures/s3data_iam_adduser.png" | absolute_url }}){: width="70%" height="70%"}

사용자 이름을 추가하고 `프로그래밍 방식 액세스`를 활성화 합니다. AWS CLI 를 이용할 수 있도록 사용자 기능을 등록하는 것입니다.

![]({{ "/assets/figures/s3data_iam_adduser2.png" | absolute_url }}){: width="70%" height="70%"}

Bucket 을 만들고, 동기화 하는 등의 모든 작업을 할 것이기 때문에 admin 권한을 부여합니다. 이는 `기존 정책 직접 연결`에서 맨 위의 `AdministratorAccess` 를 누르면 됩니다.

![]({{ "/assets/figures/s3data_iam_adduser3.png" | absolute_url }}){: width="70%" height="70%"}

사용자에 대한 태그는 선택적으로 부여할 수 있습니다만, 나중에도 관리할 수 있으니 넘어갑니다.

![]({{ "/assets/figures/s3data_iam_adduser4.png" | absolute_url }}){: width="70%" height="70%"}

생성한 사용자의 권한 및 이름을 마지막으로 확인하고, 하단의 파란색 `사용자 만들기`를 눌러 완료합니다.

![]({{ "/assets/figures/s3data_iam_adduser5.png" | absolute_url }}){: width="70%" height="70%"}

사용자를 만든 뒤, 왼쪽 중간에 있는 `csv 다운로드` 를 누릅니다. csv 파일 하나가 다운로드 되는데, 이는 잘 저장해 둡니다. 이 파일에는 `Access key ID` 와 `Secret access key` 가 저장되어 있는데, CLI 를 이용하는데 필요합니다.

![]({{ "/assets/figures/s3data_iam_adduser6.png" | absolute_url }}){: width="70%" height="70%"}


## 로컬에서 s3 로 파일 업로드하기

### awscli

awscli 는 AWS Command Line Interface 의 약어입니다. 설치하면 이후 s3 관련 작업을 terminal command 로 할 수 있습니다.

설치는 pip install 로 가능합니다. Python 2.6x 혹은 Python 2.7x 이상이 필요하였는데, 저는 Python 3.x 만 이용하므로 버전은 자세히 보지 않았습니다.

```
pip install awscli
```

설치 후 help 명령어를 입력하여 awscli 가 설치되었는지 확인도 해봅니다.

```
aws help
```

aws cli 를 이용하기 전에 앞서 만든 AWS IAM 사용자를 추가해야 합니다. `aws configure` 를 실행시킵니다.

```
aws configure
```

네 개의 항목이 나오는데, access key id 와 secret access key 는 앞서 다운로드 받은 csv 파일 안에 있습니다. region name 은 default region 을 설정하는 것으로, 아시아 (서울)은 `ap-northeast-2` 입니다. 마지막은 enter 를 눌러 넘어갑니다.

```
AWS Access Key ID [None]:
AWS Secret Access Key [None]:
Default region name [None]:
Default output format [None]:
```

제대로 configure 가 실행 되었는지 확인합니다. 내 계정의 버킷 리스트나 한 버킷 내 파일 리스트를 살펴볼 수 있습니다.

```
# 내 계정의 버킷 리스트
aws s3 ls

# 내 버킷 내 파일 리스트
aws s3 ls s3://mybucket/
```

파일 복사는 `aws s3 cp` 이후 source, destination 순으로 입력합니다. S3 의 주소는 `s3://` 로 시작합니다.

```
aws s3 cp localfile s3://[BUCKETNAME]/[FILENAME]
```

여러 개의 파일이나 폴더를 recursive 하게 복사할 때에는 `sync` 를 이용할 수 있습니다.

```
aws s3 sync SOURCE_DIR s3://DEST_BUCKET/
```

업로드 하는 파일을 모두가 읽을 수 있도록 권한을 설정하려면 `--acl public-read` 옵션을 추가합니다.

```
aws s3 sync SOURCE s3://DESTINATION --acl public-read
```

[aws_s3_release_files]: {{ site.baseurl }}{% link _posts/2019-01-25-aws_s3_release_files.md %}
[aws_documentation]: https://aws.amazon.com/ko/getting-started/tutorials/backup-to-s3-cli/