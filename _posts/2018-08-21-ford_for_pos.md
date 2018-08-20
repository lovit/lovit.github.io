---
title: Ford algorithm 을 이용한 최단 경로 탐색
date: 2017-08-21 09:00:00
categories:
- nlp
- graph
tags:
- shortest path
- tokenizer
---

First-order Hidden Markov Model 을 이용하는 품사 판별 문제는 최단 경로 문제로 풀 수 있습니다. 

## Review of Ford algorithm for shortest path

얼마 전, Hidden Markov Model (HMM) 을 기반으로 하는 품사 판별기의 코드를 공부하던 중, "findPath" 라는 이름의 함수를 보았습니다. HMM 의 decoding 과정을 설명할 때 주로 Dynamic programming 의 관점으로 설명을 하는데, 그 구현체는 최단 경로를 찾는 방법으로 decoding 과정을 설명하고 있었습니다. 한 번도 생각해보지 않았는데, 생각해보니 first-order 를 따르는 HMM 이라면 최단 경로 문제와 동치가 됩니다. 이 이야기를 한 번 정리해야 겠다는 생각을 하였습니다.

이번 포스트에서는 HMM 에 대한 설명은 하지 않습니다. 조만간에 HMM 관련 포스트를 작성하고, 이 부분을 링크로 대체하도록 하겠습니다.


## Ford algorithm 을 이용한 품사 판별기 만들기



![]({{ "/assets/figures/shortestpath_chungha.png" | absolute_url }}){: width="80%" height="80%"}

[prev]: {{ site.baseurl }}{% link _posts/2018-08-21-ford_for_shortestpath.md %}