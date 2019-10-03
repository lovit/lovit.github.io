

## Python 으로 github 이용하기

Python 에서 github 을 이용할 수 있는 패키지로는 [GitPython][gitpython] 과 [PyGithub][pygithub] 이 있습니다. 처음에는 PyGithub 을 이용하려 했는데, 여러 개의 파일을 한 번에 commit 하는 기능이 없어 GitPython 을 이용하였습니다.

GitPython 은 pip install 로 설치할 수 있습니다.

```
pip isntall gitpython
```

**GitPython** 은 local 의 repository directory 를 입력하여 repository instance 를 만듭니다.

```python
from git import Repo

repo_dir = '/abc/def/repo'
repo = Repo(repo_dir)
```

## Show commit messages in a repository

이전까지의 commit message 를 확인할 수 있습니다. `committed_date` 는 UNIX time 형식으로 표현된 commit time 입니다. time 을 이용하여 보기 쉬운 형식으로 변환도 해봅니다.

```python
import time

for commit in repo.iter_commits():
    message = commit.message.strip()
    unix_time = commit.committed_date
    strf_time = time.strftime("%Y-%m-%d %H:%M", time.gmtime(unix_time))
    print('{} : {}'.format(strf_time, message))
```

청와대 청원 게시판 스크래퍼의 commit messages 입니다. Histories 가 잘 보이질 않네요 (commit 단위 좀 잘 설정할 걸 그랬네요)

```
2019-02-04 14:43 : Update usage of script
2019-02-04 14:43 : Minor change (verbose message)
2019-02-04 14:34 : Commit scraping script
2019-02-04 14:22 : Docstring in scraper.py
2019-02-04 14:22 : SLeep when unexpected exception occurs
2019-02-04 13:55 : As arguments (sleep, verbose)
2019-02-04 13:52 : Separate getting & yielding petition links
2019-02-02 14:54 : Change minor
2018-11-10 14:37 : Merge branch 'master' of https://github.com/lovit/petitions_scraper
2018-11-10 14:37 : Option: get replies
2018-11-10 14:36 : Sleep as argument
2018-11-07 13:18 : Update README.md
...
```

## Indexing files, commit, and push




## Reference

더 자세한 내용은 [Official document][gitpython_doc]에 있습니다.

[gitpython]: https://github.com/gitpython-developers/GitPython
[pygithub]: https://github.com/PyGithub/PyGithub
[gitpython_doc]: https://gitpython.readthedocs.io/en/stable/tutorial.html
