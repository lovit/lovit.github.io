---
title: Carblog. Problem description
date: 2018-03-20 15:00:00
categories:
- nlp
tags:
- analysis
- carblog
---

Carblog 는 데이터분석 사례의 프로젝트이름 입니다. Carblog 는 키워드 기반으로 수집된 문서 집합에서 의도한 문서 만을 선택하는 문제입니다. 네이버 블로그에 차량 이름을 쿼리로 입력하여 데이터를 수집하였습니다. 하지만 하나의 단어는 여러 의미를 지닙니다. **소나타**는 차량의 이름이기도 하지만, 클래식 음악의 형식이기도 합니다. 혹은 가수 아이비의 "유혹의 소나타"라는 노래 제목에 포함된 단어이기도 합니다. 키워드 기반으로 수집된 문서에는 우리가 예상하지 못한 수많은 주제의 문서들이 포함되어 있습니다. 학습데이터를 마련하여 문서 판별기를 만들 수도 있겠지만, 학습데이터를 만드는 과정이 고통스럽습니다. 좀 더 멋지게 데이터 기반으로 학습데이터를 마련하지 않고서 차량 문서들만을 선택하는 필터를 만듭니다. 


## Carblog Introduction

예전에 다른 팀의 연구를 도와준 적이 있었습니다. 한국에서 판매되는 여러 차량의 버즈 분석을 수행하기 위하여 차량 관련 문서들을 수집해야 했습니다. 데이터의 수집 방식은 네이버 블로그에 질의어를 입력하여, 해당 질의어가 포함된 문서들을 수집하는 방식이었습니다. 아래 표의 27 개의 질의어를 이용하여 2010. 1 ~ 2015. 7 에 작성된 블로그, 2,844,955 건을 수집하였습니다. 누구라도 아래의 27 개 질의어로부터 수집하고자 했던 문서의 implicit label 이 **차량**이라는 것은 알 수 있을 것입니다. 

| A6 | BMW5 | BMW | K3 | K5 | K7 | QM3 | 
| 그랜저 | 벤츠E | 산타페 | 소나타 | 스포티지 | 싼타페 | 쏘나타 | 
| 쏘렌토 | 아반떼 | 아반테 | 제네시스 | 코란도C | 투싼 | 티구안 |
| 티볼리 | 파사트 | 폭스바겐골프 | 현기차 | 현대자동차 | 현대차 | |

하지만 문제가 발생했습니다. 하나의 단어에는 여러 개의 의미가 포함되어 있습니다. 왠만한 단어는 다의어입니다. 우리가 발견했던, 차량 외 주제들입니다. 아주 조금만 예를 든 것일 뿐, 사실은 엄청나게 많은 주제들이 도사리고 (under cover) 있습니다.

| 질의어 | 차량 외 주제 |
| --- | --- |
| A6 | 광교신도시A6블록 / 갤럭시탭A6 |
| K3 | 홍콩 관광버스 노선번호 / 한국축구 리그 / 슈퍼스타K3 / 비타민K3 |
| 소나타 | 클래식 음악 형식 / 아이비의 유혹의 소나타 |
| 제네시스 | 락벤드 / 피부치료 / 탄산수제조기 / 터미네이터4 / 메이플스토리 캐릭터 기술 |
| 티볼리 | 리조트 이름 / 코펜하겐 티볼리 공원 / 루이뷔동 백이름 / 이탈리아 도시 |

그렇다면 우리는 왜 이 많은 주제의 문서들을 제거해야 할까요? 

<svg width="960" height="500"></svg>
<script src="https://d3js.org/d3.v4.min.js"></script>
<script>

var svg = d3.select("svg"),
    margin = {top: 20, right: 20, bottom: 30, left: 50},
    width = +svg.attr("width") - margin.left - margin.right,
    height = +svg.attr("height") - margin.top - margin.bottom,
    g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

var parseTime = d3.timeParse("%y-%m");

var x = d3.scaleTime()
    .rangeRound([0, width]);

var y = d3.scaleLinear()
    .rangeRound([height, 0]);

var line = d3.line()
    .x(function(d) { return x(d.date); })
    .y(function(d) { return y(d.close); });

d3.tsv("/resources/carblog_k3_monthly.tsv", function(d) {
  d.date = parseTime(d.date);
  d.close = +d.close;
  return d;
}, function(error, data) {
  if (error) throw error;

  x.domain(d3.extent(data, function(d) { return d.date; }));
  y.domain(d3.extent(data, function(d) { return d.close; }));

  g.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(x))
    .select(".domain")
      .remove();

  g.append("g")
      .call(d3.axisLeft(y))
    .append("text")
      .attr("fill", "#000")
      .attr("transform", "rotate(-90)")
      .attr("y", 6)
      .attr("dy", "0.71em")
      .attr("text-anchor", "end")
      .text("Number of documents");

  g.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", "steelblue")
      .attr("stroke-linejoin", "round")
      .attr("stroke-linecap", "round")
      .attr("stroke-width", 1.5)
      .attr("d", line);
});

</script>
