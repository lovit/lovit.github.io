---
layout: page
---

주제별로 모은 포스트들입니다. 한 포스트가 여러 주제에 속하기도 합니다. 

## Machine learning algorithm

### Classification
- [Logitsic regression and Softmax regression for document classification][logistic_regression]
- [Logistic regression with L1, L2 regularization and keyword extraction][lasso_keyword]
- [Scikit-learn Logistic Regression fails for finding optima?][scikit_learn_logistic_failed_local_optima]
- [Decision trees are not appropriate for text classifications.][decision_tree]
- [Tree traversal of trained decision tree (scikit-learn)][get_rules_from_trained_decision_tree]

### Sequential labeling
- [From Softmax Regression to Conditional Random Field for Sequential Labeling][crf]

### Clustering
- [k-means initial points 선택 방법][kmeans_initializer]
- [Cluster labeling for text data][kmeans_cluster_labeling]

### Nearest neighbor search
- [Random Projection and Locality Sensitive Hashing][lsh]


### Graph ranking, similarity
- [Graph ranking algorithm. PageRank and HITS][pagerank_and_hits]
- [Implementing PageRank. Python dict vs numpy][pagerank_implementation_dict_vs_numpy]


## Natural Language Processing

### Text data preprocessing
- [From text to term frequency matrix (KoNLPy)][from_text_to_matrix]
- [Komoran, 코모란 형태소 분석기 사용 방법과 사용자 사전 추가 (Java, Python)][komoran]
- [Scipy sparse matrix handling][sparse_mtarix_handling]
- [Conditional Random Field based Korean Space Correction][crf_korean_spacing]
- [soyspacing. Heuristic Korean Space Correction, A safer space corrector.][soyspacing]

### Word extraction, Tokenization, Part of speech tagging
- [Part of speech tagging, Tokenization, and Out of vocabulary problem][pos_and_oov]
- [Left-side substring tokenizer, the simplest tokenizer.][simplest_tokenizers]
- [Word Piece Model (a.k.a sentencepiece)][wpm]
- [Uncertanty to word boundary; Accessor Variety & Branching Entropy][branching_entropy_accessor_variety]
- [Cohesion score + L-Tokenizer. 띄어쓰기가 잘 되어있는 한국어 문서를 위한 unsupervised tokenizer][cohesion_ltokenizer]
- [띄어쓰기가 되어있지 않은 한국어를 위한 토크나이저 만들기 (Max Score Tokenizer 를 Python 으로 구현하기)][max_score_tokenizer_dev]
- [Unsupervised tokenizers in soynlp project][three_tokenizers_soynlp]

### Keywords, related-words extraction
- [Logistic regression with L1, L2 regularization and keyword extraction][lasso_keyword]
- [Term proportion ratio base Keyword extraction][proportion_keywords]
- [KR-WordRank, 토크나이저를 이용하지 않는 한국어 키워드 추출기][krwordrank]
- [Implementing PMI (Practice handling matrix of numpy & scipy)][implementing_pmi_numpy_practice]

### Word Representation
- [Word / Document embedding (Word2Vec / Doc2Vec)][word_doc_embedding]
- [Word2Vec understanding, Space odyssey of word embedding (1)][space_odyssey_of_word2vec]
- [(Review) From frequency to meaning, Vector space models of semantics (Turney & Pantel, 2010)][from_frequency_to_meaning]
- [(Review) Neural Word Embedding as Implicit Matrix Factorization (Levy & Goldberg, 2014 NIPS)][context_vector_for_word_similarity]
- [Implementing PMI (Practice handling matrix of numpy & scipy)][implementing_pmi_numpy_practice]


## Data visualization
- [Python plotting kit Bokeh][bokeh_python_plotting]
- [Plotly 를 이용한 3D scatter plot][plotly_3d_scatterplot]
- [Word cloud in Python][word_cloud]

## Application: 띄어쓰기 오류 교정
- [From Softmax Regression to Conditional Random Field for Sequential Labeling][crf]
- [Conditional Random Field based Korean Space Correction][crf_korean_spacing]
- [soyspacing. Heuristic Korean Space Correction, A safer space corrector.][soyspacing]

## Applications: Carblog
- [Carblog. Problem description][carblog_description]

## and more
- [soydata. 복잡한 인공 데이터 생성을 위한 함수들][synthetic_dataset]


[kmeans_initializer]: {{ site.baseurl }}{% link _posts/2018-03-19-kmeans_initializer.md %}
[carblog_description]: {{ site.baseurl }}{% link _posts/2018-03-20-carblog_description.md %}
[kmeans_cluster_labeling]: {{ site.baseurl }}{% link _posts/2018-03-21-kmeans_cluster_labeling.md %}
[logistic_regression]: {{ site.baseurl }}{% link _posts/2018-03-22-logistic_regression.md %}
[lasso_keyword]: {{ site.baseurl }}{% link _posts/2018-03-24-lasso_keyword.md %}
[from_text_to_matrix]: {{ site.baseurl }}{% link _posts/2018-03-26-from_text_to_matrix.md %}
[word_doc_embedding]: {{ site.baseurl }}{% link _posts/2018-03-26-word_doc_embedding.md %}
[lsh]: {{ site.baseurl }}{% link _posts/2018-03-28-lsh.md %}
[bokeh_python_plotting]: {{ site.baseurl }}{% link _posts/2018-03-31-bokeh_python_plotting.md %}
[pos_and_oov]: {{ site.baseurl }}{% link _posts/2018-04-01-pos_and_oov.md %}
[simplest_tokenizers]: {{ site.baseurl }}{% link _posts/2018-04-02-simplest_tokenizers.md %}
[wpm]: {{ site.baseurl }}{% link _posts/2018-04-02-wpm.md %}
[space_odyssey_of_word2vec]: {{ site.baseurl }}{% link _posts/2018-04-05-space_odyssey_of_word2vec.md %}
[komoran]: {{ site.baseurl }}{% link _posts/2018-04-06-komoran.md %}
[scikit_learn_logistic_failed_local_optima]: {{ site.baseurl }}{% link _posts/2018-04-06-scikit_learn_logistic_failed_local_optima.md %}
[branching_entropy_accessor_variety]: {{ site.baseurl }}{% link _posts/2018-04-09-branching_entropy_accessor_variety.md %}
[cohesion_ltokenizer]: {{ site.baseurl }}{% link _posts/2018-04-09-cohesion_ltokenizer.md %}
[max_score_tokenizer_dev]: {{ site.baseurl }}{% link _posts/2018-04-09-max_score_tokenizer_dev.md %}
[sparse_mtarix_handling]: {{ site.baseurl }}{% link _posts/2018-04-09-sparse_mtarix_handling.md %}
[three_tokenizers_soynlp]: {{ site.baseurl }}{% link _posts/2018-04-09-three_tokenizers_soynlp.md %}
[proportion_keywords]: {{ site.baseurl }}{% link _posts/2018-04-12-proportion_keywords.md %}
[krwordrank]: {{ site.baseurl }}{% link _posts/2018-04-16-krwordrank.md %}
[pagerank_and_hits]: {{ site.baseurl }}{% link _posts/2018-04-16-pagerank_and_hits.md %}
[pagerank_implementation_dict_vs_numpy]: {{ site.baseurl }}{% link _posts/2018-04-17-pagerank_implementation_dict_vs_numpy.md %}
[word_cloud]: {{ site.baseurl }}{% link _posts/2018-04-17-word_cloud.md %}
[from_frequency_to_meaning]: {{ site.baseurl }}{% link _posts/2018-04-18-from_frequency_to_meaning.md %}
[context_vector_for_word_similarity]: {{ site.baseurl }}{% link _posts/2018-04-22-context_vector_for_word_similarity.md %}
[implementing_pmi_numpy_practice]: {{ site.baseurl }}{% link _posts/2018-04-22-implementing_pmi_numpy_practice.md %}
[crf]: {{ site.baseurl }}{% link _posts/2018-04-24-crf.md %}
[crf_korean_spacing]: {{ site.baseurl }}{% link _posts/2018-04-24-crf_korean_spacing.md %}
[soyspacing]: {{ site.baseurl }}{% link _posts/2018-04-25-soyspacing.md %}
[plotly_3d_scatterplot]: {{ site.baseurl }}{% link _posts/2018-04-26-plotly_3d_scatterplot.md %}
[synthetic_dataset]: {{ site.baseurl }}{% link _posts/2018-04-27-synthetic_dataset.md %}
[decision_tree]: {{ site.baseurl }}{% link _posts/2018-04-30-decision_tree.md %}
[get_rules_from_trained_decision_tree]: {{ site.baseurl }}{% link _posts/2018-04-30-get_rules_from_trained_decision_tree.md %}
