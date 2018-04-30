---
title: Decision trees are not appropriate for text classifications.
date: 2017-04-30 09:00:00
categories:
- machine learning
tags:
- decision tree
---

의사결정나무 (Decision tree) 는 classification 과정에 대한 해석을 제공하는 점과 다른 classifiers 보다 데이터의 전처리를 (상대적으로) 덜해도 된다는 장점이 있습니다. 하지만 bag of words model 과 같은 sparse data 의 분류에는 적합하지 않습니다. 이번 포스트에서는 의사결정나무가 무엇을 학습하는지 알아보고, 왜 sparse data 에는 적합하지 않은지에 대하여 이야기합니다.

## Decision trees

<ul class="slider" id="slider">
<li><img src="/assets/figures/dt_growth_1.png" alt="slide1"/></li>
<li><img src="/assets/figures/dt_growth_2.png" alt="slide2"/></li>
<li><img src="/assets/figures/dt_growth_3.png" alt="slide3"/></li>
<li><img src="/assets/figures/dt_growth_4.png" alt="slide4"/></li>
<li><img src="/assets/figures/dt_growth_5.png" alt="slide5"/></li>
<li><img src="/assets/figures/dt_growth_6.png" alt="slide6"/></li>
</ul>

<script type="text/javascript" src="/assets/js/src/slider.min.js"></script>
<script type="text/javascript">
  $(window).on("load", function() {
    $("#slider").slider();
  });
</script>