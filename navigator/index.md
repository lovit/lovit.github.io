---
layout: page
---

주제별로 모은 포스트들입니다. 한 포스트가 여러 주제에 속하기도 합니다.

<font color="#2851a4"><h2>Machine learning algorithm</h2></font>

### Classification
- [Logitsic regression and Softmax regression for document classification][logistic_regression]
- [Logistic regression with L1, L2 regularization and keyword extraction][lasso_keyword]
- [Scikit-learn Logistic Regression fails for finding optima?][scikit_learn_logistic_failed_local_optima]
- [Decision trees are not appropriate for text classifications.][decision_tree]
- [Tree traversal of trained decision tree (scikit-learn)][get_rules_from_trained_decision_tree]

### Sequential labeling
- [From Softmax Regression to Conditional Random Field for Sequential Labeling][crf]
- [Ford algorithm 을 이용한 품사 판별, 그리고 Hidden Markov Model (HMM) 과의 관계][ford_for_pos]

### Clustering
- [k-means initial points 선택 방법][kmeans_initializer]
- [Cluster labeling for text data][kmeans_cluster_labeling]

### Nearest neighbor search
- [Random Projection and Locality Sensitive Hashing][lsh]
- [Small-world phenomenon 을 이용한 network 기반 nearest neighbor search][network_based_nearest_neighbors]

### Graph ranking, similarity, distance
- [Graph ranking algorithm. PageRank and HITS][pagerank_and_hits]
- [Implementing PageRank. Python dict vs numpy][pagerank_implementation_dict_vs_numpy]
- [Ford algorithm 을 이용한 최단 경로 탐색][ford_for_shortestpath]
- [Ford algorithm 을 이용한 품사 판별, 그리고 Hidden Markov Model (HMM) 과의 관계][ford_for_pos]

### Embedding for visualization
- [t-Stochastic Neighbor Embedding (t-SNE) 와 perplexity][tsne]
- [Embedding for Word Visualization (LLE, ISOMAP, MDS, t-SNE)][mds_isomap_lle]

<font color="#2851a4"><h2>Natural Language Processing</h2></font>

### Text data preprocessing
- [From text to term frequency matrix (KoNLPy)][from_text_to_matrix]
- [Komoran, 코모란 형태소 분석기 사용 방법과 사용자 사전 추가 (Java, Python)][komoran]
- [Building your KoNLPy. Komoran3 를 Python class 로 만들기][building_your_komoran]
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
- [명사 추출기 (1) soynlp.noun.LRNounExtractor][noun_extraction_ver1]
- [명사 추출기 (2) soynlp.noun.LRNounExtractor_v2][noun_extraction_ver2]
- [명사 추출기 (3) Noun extraction and noun tokenization][noun_extractor_and_tokenizer]
- [한국어 용언의 원형 복원 (Korean lemmatization)][lemmatizer]
- [한국어 용언의 활용 함수 (Korean conjugation)][conjugation]
- [Hidden Markov Model (HMM) 기반 품사 판별기의 원리와 문제점][hmm_based_tagger]
- [Conditional Random Field (CRF) 기반 품사 판별기의 원리와 HMM 기반 품사 판별기와의 차이점][crf_based_tagger]
- [(Review) Incorporating Global Information into Supervised Learning for Chinese Word Segmentation][review_chinese_word_segmentation]

### Keywords, related-words extraction
- [Logistic regression with L1, L2 regularization and keyword extraction][lasso_keyword]
- [Term proportion ratio base Keyword extraction][proportion_keywords]
- [KR-WordRank, 토크나이저를 이용하지 않는 한국어 키워드 추출기][krwordrank]
- [Implementing PMI (Practice handling matrix of numpy & scipy)][implementing_pmi_numpy_practice]

### Named Entity Recognition
- [Conditional Random Field based Named Entity Recognition][crf_ner]

### Word Representation
- [Word / Document embedding (Word2Vec / Doc2Vec)][word_doc_embedding]
- [Word2Vec understanding, Space odyssey of word embedding (1)][space_odyssey_of_word2vec]
- [(Review) From frequency to meaning, Vector space models of semantics (Turney & Pantel, 2010)][from_frequency_to_meaning]
- [(Review) Neural Word Embedding as Implicit Matrix Factorization (Levy & Goldberg, 2014 NIPS)][context_vector_for_word_similarity]
- [Implementing PMI (Practice handling matrix of numpy & scipy)][implementing_pmi_numpy_practice]
- [GloVe, word representation][glove]

### Topic modeling
- [pyLDAvis 를 이용한 Latent Dirichlet Allocation (LDA) 시각화하기][pyldavis_lda]

### String distance
- [Levenshtein (edit) distance 를 이용한 한국어 단어의 형태적 유사성][levenshtein_hangle]
- [Inverted index 를 이용한 빠른 Levenshtein (edit) distance 탐색][levenshtein_inverted_index]

<font color="#2851a4"><h2>Data visualization</h2></font>
- [Python plotting kit Bokeh][bokeh_python_plotting]
- [Plotly 를 이용한 3D scatter plot][plotly_3d_scatterplot]
- [Word cloud in Python][word_cloud]

<font color="#2851a4"><h2>Application: 띄어쓰기 오류 교정</h2></font>
- [From Softmax Regression to Conditional Random Field for Sequential Labeling][crf]
- [Conditional Random Field based Korean Space Correction][crf_korean_spacing]
- [soyspacing. Heuristic Korean Space Correction, A safer space corrector.][soyspacing]

<font color="#2851a4"><h2>Applications: Carblog</h2></font>
- [Carblog. Problem description][carblog_description]

<font color="#2851a4"><h2>and more</h2></font>
- [soydata. 복잡한 인공 데이터 생성을 위한 함수들][synthetic_dataset]
- [Cherry picking 의 위험성과 testing 설계의 중요성][cherry_picking]
- [Github 으로 텍스트 문서 버전 관리하기][latex_with_github]


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
[noun_extraction_ver1]: {{ site.baseurl }}{% link _posts/2018-05-07-noun_extraction_ver1.md %}
[noun_extraction_ver2]: {{ site.baseurl }}{% link _posts/2018-05-08-noun_extraction_ver2.md %}
[noun_extractor_and_tokenizer]: {{ site.baseurl }}{% link _posts/2018-05-09-noun_extractor_and_tokenizer.md %}
[cherry_picking]: {{ site.baseurl }}{% link _posts/2018-05-26-cherry_picking.md %}
[lemmatizer]: {{ site.baseurl }}{% link _posts/2018-06-07-lemmatizer.md %}
[conjugation]: {{ site.baseurl }}{% link _posts/2018-06-11-conjugator.md %}
[crf_ner]: {{ site.baseurl }}{% link _posts/2018-06-22-crf_based_ner.md %}
[building_your_komoran]: {{ site.baseurl }}{% link _posts/2018-07-06-java_in_python.md %}
[latex_with_github]: {{ site.baseurl }}{% link _posts/2018-08-17-latex_with_github.md %}
[ford_for_pos]: {{ site.baseurl }}{% link _posts/2018-08-21-ford_for_pos.md %}
[ford_for_shortestpath]: {{ site.baseurl }}{% link _posts/2018-08-21-ford_for_shortestpath.md %}
[levenshtein_hangle]: {{ site.baseurl }}{% link _posts/2018-08-28-levenshtein_hangle.md %}
[levenshtein_inverted_index]: {{ site.baseurl }}{% link _posts/2018-09-04-levenshtein_inverted_index.md %}
[glove]: {{ site.baseurl }}{% link _posts/2018-09-05-glove.md %}
[network_based_nearest_neighbors]: {{ site.baseurl }}{% link _posts/2018-09-10-network_based_nearest_neighbors.md %}
[hmm_based_tagger]: {{ site.baseurl }}{% link _posts/2018-09-11-hmm_based_tagger.md %}
[crf_based_tagger]: {{ site.baseurl }}{% link _posts/2018-09-13-crf_based_tagger.md %}
[review_chinese_word_segmentation]: {{ site.baseurl }}{% link _posts/2018-09-25-review_chinese_word_segmentation.md %}
[pyldavis_lda]: {{ site.baseurl }}{% link _posts/2018-09-27-pyldavis_lda.md %}
[tsne]: {{ site.baseurl }}{% link _posts/2018-09-28-tsne.md %}
[mds_isomap_lle]: {{ site.baseurl }}{% link _posts/2018-09-29-mds_isomap_lle.md %}
