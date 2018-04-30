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

의사결정나무는 데이터의 공간을 직사각형으로 나눠가며 최대한 같은 종류의 데이터로 이뤄진 부분공간을 찾아가는 classifiers 입니다. 마치 clustering 처럼 비슷한 공간을 하나의 leaf node 로 나눠갑니다.
