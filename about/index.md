---
layout: page
---

## Me

데이터마이닝과 자연어처리를 공부하고 있습니다. 복잡한 머신러닝 알고리즘을 직관적이고 해석하는 것을 좋아합니다. 무거운 모델을 돌리기 어려운 환경에서도 빠르게 계산할 수 있는 가벼운 모델을 좋아합니다. 학습할 데이터가 충분하지 않는 상황들을 잘 반영하는 모델을 추구합니다. 데이터와의 지속적인 인터렉션을 통한 데이터 공간의 이해를 도울 수 있는 툴을 만들고 싶습니다.

## Blog

머신 러닝 알고리즘에 대한 직관적인 설명 / 데이터 분석 사례 / 데이터 분석에 관련된 잡다한 이야기들로 구성된 블로그 입니다. 

## Softwares

- [soynlp][soynlp]: 한국어의 단어 추출 / 토크나이저와 품사 판별기 / 노이즈 제거를 위한 전처리 기능을 제공하는 패키지입니다. 
- [soyspacing][soyspacing]: 띄어쓰기 교정기입니다. Conditional Random Field (CRF) 보다 가볍고, 보수적이며, 머신러닝이 아닌 직관적인 방법으로 띄어쓰기 오류를 교정합니다. 학습과 적용 기능을 제공합니다. 
- [pycrfsuite_spacing][pycrfsuite_spacing]: pycrfsuite (crfsuite 의 Python 포팅 라이브러리, CRF 모델)를 이용한 CRF 기반 띄어쓰기 오류 교정기입니다. 학습과 적용 기능을 제공합니다. 
- [KR-WordRank][kr-wordrank]: 토크나이저의 설정 없이 단어 / 키워드를 추출하는 패키지입니다. 
- [customized_konlpy][ckonlpy]: KoNLPy 와 "사용자 정의 사전 + 템플릿" 을 함께 이용할 수 있는 기능을 제공합니다.
- [soykeyword][soykeyword]: Lasso Regression 과 Proportion ratio 기반의 두 가지 방법을 이용한 키워드/연관어 추출기입니다. 문서의 레이블링 기능을 제공합니다.
- [clustering4docs][clustering4docs]: 문서 군집화를 위한 알고리즘을 제공합니다. Spherical k-means 외, clustering labeling 기능을 함께 제공합니다.
- [clusteirng_visualization][clustering_visualization]: 군집화 학습 결과를 시각적으로 표현할 수 있는 scatter plot 과 heatmap of centroids 을 제공합니다. 
- [fast_hangle_levenshtein][fast_editdistance]: Edit distance 기반의 빠른 유사어 검색을 위한 inverted index 기반 유사어 검색기입니다. 초/중/종성의 분리 수준의 Edit distance 유사어 검색 기능을 제공합니다.
- [Word Piece Model][wpm]: Google Neural Machine Translator 의 tokenizer 인 WPM 을 이용한 토크나이저입니다. 학습 및 저장 기능을 제공하며, heuristic trick 을 이용하여 빠른 학습을 제공합니다.
- [fastcosine][fastcosine]: Bag of words 형식으로 표현된 단문(short sentences) 의 검색을 위한 inverted index 입니다. Approximated nearest neighbor search 기능을 제공합니다.   

[soynlp]: https://github.com/lovit/soynlp
[soyspacing]: https://github.com/lovit/soyspacing
[pycrfsuite_spacing]: https://github.com/lovit/pycrfsuite_spacing
[kr-wordrank]: https://github.com/lovit/kr-wordrank
[ckonlpy]: https://github.com/lovit/customized_konlpy
[soykeyword]: https://github.com/lovit/soykeyword
[clustering4docs]: https://github.com/lovit/clustering4docs
[clustering_visualization]: https://github.com/lovit/clustering_visualization
[wpm]: https://github.com/lovit/WordPieceModel
[fast_editdistance]: https://github.com/lovit/inverted_index_for_hangle_editdistance/
[fastcosine]: https://github.com/lovit/fastcosine/
