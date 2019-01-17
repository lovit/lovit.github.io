---
title: 웹사이트 구조 별 스크래퍼 (scrapper) 만들기
date: 2019-01-17 23:00:00
categories:
- scraper
tags:
- scraper
---

며칠 간, 여러 사이트들로부터 데이터를 수집하는 웹 스크래퍼를 만들 일이 있었습니다. 작업의 대부분은 각 사이트 별로 HTML 파일을 파싱하는 반복적 작업이었지만, 사이트 별로 HTML (혹은 JSON 형식의 response) 을 받아오는 방법들은 다를 수 있습니다. 이번 포스트에서는 몇 가지 예시를 통하여 웹사이트의 구조 별로 웹 스크래퍼를 만드는 과정을 정리합니다.

## Scraper vs Crawler

## URL 의 일부만 변경하면 되는 경우 (네이버 영화)

## page 변수가 있는 경우 (네이버 영화 댓글)

## Python requests 를 이용하여 JSON 형식의 데이터를 얻는 경우 (Fox news)

## 웹 페이지의 링크로부터 파일을 다운로드 받는 경우 (Congressional Research Service )


## See more

[이준범][beomi_github] 님의 블로그에는 웹과 데이터 분석 관련 좋은 글들이 많습니다. 그 중 Python 을 이용하는 스크래핑 관련된 시리즈 글도 있습니다. [기본적인 스크래핑][beomi_basic]과 [로긴이 필요한 스크래핑][beomi_login]에 대해서 이준범님의 글을 읽는 것을 추천 드립니다.

[beomi_github]: https://github.com/beomi
[beomi_basic]: https://beomi.github.io/2017/01/20/HowToMakeWebCrawler/
[beomi_login]: https://beomi.github.io/2017/01/20/HowToMakeWebCrawler-With-Login/