---
title: praw 를 이용한 Reddit scrapping 과 아카이빙이 된 이전 Reddit 가져오기
date: 2019-01-16 05:00:00
categories:
- dataset
tags:
- dataset
---

## Reddit

Reddit 은 일종의 social news platform 으로, 사용자들이 한 주제에 대한 글을 쓰고 다른 사용자들이 댓글을 달며 토론을 하는 서비스 입니다. 각 글마다 추천과 비추천을 voting 할 수 있기 때문에, 이에 따라 중요한 글들이 상위권으로 배치됩니다. Sung Kim 교수님이 TensorFlow-KR 에 머신러닝에 관련된 최근 reddit posts 를 정리해 주시기도 했습니다. 바로 그 Reddit 입니다.

최근에 이 Reddit 의 자료들을 수집해야 할 일이 생겼습니다. 다른 분들이 비슷한 일을 하신다면 시행착오를 줄이길 바라며, 그 과정을 정리합니다.

Reddit 을 쓰신 분들이라면 잘 아실테지만, 저는 몇 번 보기만 하는 사용자였기에 용어부터 알아야 했습니다. 

`subreddit` 은 일종의 게시판 이름입니다. Topic 으로 봐도 좋습니다. URL 의 `r/SUBREDDIT` 입니다. 예를 들어 머신 러닝 게시판의 주소는 `https://reddit.com/r/MachineLearning` 입니다.

`submission` 은 게시물로, 포스트라 생각하시면 됩니다. 이 submission 에 댓글인 `comment`가 추가됩니다. 그리고 submission 의 본문을 `selftext` 라 합니다. 댓글의 본문도 `selftext` 라 합니다.

## robots.txt

먼저 Reddit 의 robots.txt 를 확인했습니다. disallow 가 많습니다. 공식 API 가 제공된다면 이를 이용합니다.

```
# 80legs
User-agent: 008
Disallow: /

# 80legs' new crawler
User-agent: voltron
Disallow: /

User-Agent: bender
Disallow: /my_shiny_metal_ass

User-Agent: Gort
Disallow: /earth

User-agent: MJ12bot
Disallow: /

User-agent: PiplBot
Disallow: /

User-Agent: *
Disallow: /*.json
Disallow: /*.json-compact
Disallow: /*.json-html
Disallow: /*.xml
Disallow: /*.rss
Disallow: /*.i
Disallow: /*.embed
Disallow: /*/comments/*?*sort=
Disallow: /r/*/comments/*/*/c*
Disallow: /comments/*/*/c*
Disallow: /r/*/submit
Disallow: /message/compose*
Disallow: /api
Disallow: /post
Disallow: /submit
Disallow: /goto
Disallow: /*after=
Disallow: /*before=
Disallow: /domain/*t=
Disallow: /login
Disallow: /reddits/search
Disallow: /search
Disallow: /r/*/search
Disallow: /r/*/user/
Disallow: /gold?
Allow: /partner_api/
Allow: /
Allow: /sitemaps/*.xml

Sitemap: https://www.reddit.com/sitemaps/subreddit-sitemaps.xml
Sitemap: https://www.reddit.com/sitemaps/comment-page-sitemaps.xml
```

## praw, Python API

Reddit 은 praw 라는 API 를 제공합니다. praw 를 이용하려면 OAuth2 등록 ([링크]((https://www.reddit.com/prefs/apps)))을 해야 합니다. 이에 대한 설명은 [Reddit wiki](https://github.com/reddit-archive/reddit/wiki/oauth2) 에 설명되어 있습니다.

praw 는 pip install 로 설치합니다.

```
pip isntall praw
```

praw 를 이용하여 Reddit instance 를 만듭니다. OAuth2 페이지에서 생성한 app 에 관련된 id 와 secret 을 입력합니다. Reddit 의 계정 정보도 함께 입력합니다.

```python
import praw
reddit = praw.Reddit(
    client_id='Your app id',
    client_secret='Your app secret',
    user_agent='Your name',
    username='Reddit account user name',
    password='Reddit account password'
)
```

subreddit 을 만들면 해당 subreddit 의 이슈되는 글 (hot) 혹은 새롭게 등록된 글 (new) 을 확인할 수 있습니다.

```python
subreddit = reddit.subreddit('MachineLearning')
for submission in subreddit.hot(limit=5):
    # do something
```

혹은 하나의 submission 을 가져올 수도 있습니다. submission id 를 입력해야 합니다. submission 의 제목, 본문 (selftext), 혹은 본문의 HTML source code (selftext_html) 등을 가져올 수 있습니다.

```python
submission = reddit.submission(id='ag88l4')

def parse_submission(submission):
    return {
        'title': submission.title,
        'created_utc': submission.created_utc,
        'author_fullname': submission.author_fullname,
        'selftext': submission.selftext,
        'selftext_html': submission.selftext_html,
        'id': submission.id
    }
```

더 자세한 API 사용법은 [documentation](https://praw.readthedocs.io/en/latest/index.html) 에 적혀있습니다.

하지만, 각 subreddit 에서 가져올 수 있는 submissions 의 개수는 제약이 있습니다. Submission 의 id (URL) 을 알고 있다면 아주 오래전의 글도 읽을 수는 있지만, 각 subreddit 에서 스크롤을 내린다던지, new 의 limit 을 매우 크게 가져갈 수는 없습니다. 대략 최대 1k 개의 글만 가져올 수 있는 것 같습니다. 대신 이전의 글들은 따로 아카이빙이 되어 있습니다.

## Archived reddit

이전의 Reddit submissions 들은 아카이빙이 되어 공유되고 있습니다. submissions 와 comments 는 각각 나뉘어 공유되는데, 이들의 주소는 아래와 같습니다. 

- submissions: [http://files.pushshift.io/reddit/submissions/](http://files.pushshift.io/reddit/submissions/)
- comments: [http://files.pushshift.io/reddit/comments/](http://files.pushshift.io/reddit/comments/)

Submissions 이 daily 로 제공되는 [링크](http://files.pushshift.io/reddit/submissions/daily/)도 있지만, 매일매일의 글을 실시간으로 올리는 것이 아니라, daily 로 파일을 나누어 제공하는 링크입니다. 그리고 아카이빙이기 때문에 업데이트 되는 시간이 조금 걸립니다. 이 글을 작성하는 2019-01-16 현재, 2018-10 까지의 글이 아카이빙 되었으며, 마지막 파일 등록일자는 2018-11-22 입니다.

`wget` 을 이용하면 URL 의 파일을 다운로드 할 수 있습니다.

```
wget http://files.pushshift.io/reddit/submissions/RS_2017-11.xz
```

하지만 우리가 다운로드 할 파일은 여러개 입니다.

`wget -i` 를 이용하면 urls.txt 안의 파일들을 모두 다운로드 할 수 있습니다. `http://files.pushshift.io/reddit/submissions/` 에는 매 월 작성되는 Reddit submissions 를 JSON (혹은 이의 압축) 형태로 아카이빙을 합니다. 아래처럼 urls.txt 파일을 구성한 다음 이들을 다운로드 받습니다.

```
http://files.pushshift.io/reddit/submissions/RS_2017-11.xz
http://files.pushshift.io/reddit/submissions/RS_2017-12.xz
http://files.pushshift.io/reddit/submissions/RS_2018-01.xz
```

```
wget -i urls.txt
```

xz 는 압축파일이며, xz-utils 를 이용하여 압축을 해제합니다. xz-utils 는 apt-get 으로 설치할 수 있습니다.

```
sudo apt-get install xz-utils
unxz RS_2018-01.xz
```
`2018-10` 의 Reddit submissions 은 13,975,028 개며, 압축 후 단일 파일의 용량은 약 40 GB 입니다. 데이터를 메모리에 모두 올리기에는 무리가 있고, 필요한 섹션의 파일들만 선택할 것이기 때문에 데이터베이스를 이용할 필요도 없습니다. 그래서 line by line 으로 필요한 subreddit 의 submissions 만 선택하려 합니다.

압축이 해제된 파일은 한 줄이 하나의 submissions 으로 구성된 JSON 입니다. 정확히는 한 줄이 하나의 JSON 입니다. 파일의 인코딩은 `ASCII TEXT` 입니다. 샘플로 100 줄만 살펴봅니다. `json.loads` 를 이용하면 str 형식의 input 을 JSON 으로 파싱해 줍니다.

```python
import json
from pprint import pprint

with open('RS_2018-01') as f:
    docs = []
    for _ in range(100):
        docs.append(json.loads(next(f).strip()))

pprint(docs[0])
```

`docs[0]` 는 아래의 keys 를 지닌 dict 입니다. Reddit 의 submission 에 관련된 정보들이 모두 담겨 있습니다. 몇 가지 주요한 정보들은 아래와 같습니다. 댓글은 [`http://files.pushshift.io/reddit/comments/`](http://files.pushshift.io/reddit/comments/) 에 따로 저장되어 있습니다.

| key | note |
| --- | --- |
| id | subsession 의 id로, https://www.reddit.com/r/MachineLearning/comments/[ID]/ 에서 [ID]| 
| selftext | 본문 |
| score | votes 의 개수 |
| subreddit | 토픽, reddit.com/r/MachineLearning/ 의 MachineLearning |
| num_comments | 댓글 개수 |

| archived | author | author_created_utc | author_flair_background_color | author_flair_css_class |
| author_flair_richtext | author_flair_template_id | author_flair_text | author_flair_text_color | author_flair_type |
| author_fullname | author_patreon_flair | can_gild | can_mod_post | category |
| content_categories | contest_mode | created_utc | distinguished | domain |
| edited | event_end | event_is_live | event_start | gilded |
| gildings | hidden | id | is_crosspostable | is_meta |
| is_original_content | is_reddit_media_domain | is_robot_indexable | is_self | is_video |
| link_flair_background_color | link_flair_css_class | link_flair_richtext | link_flair_template_id | link_flair_text |
| link_flair_text_color | link_flair_type | locked | media | media_embed |
| media_only | no_follow | num_comments | num_crossposts | over_18 |
| parent_whitelist_status | permalink | pinned | post_hint | preview |
| pwls | quarantine | removal_reason | retrieved_on | score |
| secure_media | secure_media_embed | selftext | send_replies | spoiler |
| stickied | subreddit | subreddit_id | subreddit_name_prefixed | subreddit_subscribers |
| subreddit_type | suggested_sort | thumbnail | thumbnail_height | thumbnail_width |
| title | url | whitelist_status | wls |

이제 필요한 정보만을 파싱하여 저장합니다.