# Portfolio
---
## Natural Language Processing

### NLP Research

**A Benchmark and Dataset for Post-OCR text correction in Sanskrit (Accepted in the Findings of EMNLP 2022) :** Sanskrit is a classical language with about 30 million extant manuscripts fit for digitization, available in written, printed, or scanned-image forms. However, it is still considered to be a low-resource language when it comes to available digital resources.  In this work, we release a post-OCR text correction dataset containing around 218,000 sentences, with 1.5 million words, from 30 different books. Texts in Sanskrit are known to be diverse in terms of their linguistic and stylistic usage since Sanskrit was the lingua francua' for discourse in the Indian subcontinent for about 3 millennia. Keeping this in mind, we release a multi-domain dataset, from areas as diverse as astronomy, medicine, and mathematics, with some of them as old as 18 centuries. Further, we release multiple strong baselines as benchmarks for the task, based on pre-trained Seq2Seq language models. We find that our best-performing model consisting of byte-level tokenization in conjunction with phonetic encoding (Byt5+SLP1) yields a 23 % point increase over the OCR output in terms of word and character error rates. Moreover, we perform extensive experiments in evaluating these models on their performance and analyze common causes of mispredictions both at the graphemic and lexical levels.

**niksss at SemEval-2022 Task 6: Are Traditionally Pre-Trained Contextual Embeddings Enough for Detecting Intended Sarcasm ? (Published in Proceedings of SemEval 2022 Workshop @ NAACL) :** This paper presents the 10th and 11th place system for Subtask A -English and Subtask A Arabic respectively of the SemEval 2022 -Task 6. The purpose of the Subtask A was to classify a given text sequence into sarcastic and nonsarcastic. We also breifly cover our method for Subtask B which performed subpar when compared with most of the submissions on the official leaderboard . All of the developed solutions used a transformers based language model for encoding the text sequences with necessary changes of the pretrained weights and classifier according to the language and subtask at hand .([Paper](https://aclanthology.org/2022.semeval-1.127/)).

**niksss at SemEval-2022 Task7:Transformers for Grading the Clarifications on Instructional Texts (Published in Proceedings of SemEval 2022 Workshop @ NAACL) :** This paper describes the 9th place system description for SemEval-2022 Task 7. The goal of this shared task was to develop computational models to predict how plausible a clarification made on an instructional text is. This shared task was divided into two Subtasks A and B. We attempted to solve these using various transformers-based architecture under different regime. We initially treated this as a text2text generation problem but comparing it with our recent approach we dropped it and treated this as a text-sequence classification and regression depending on the Subtask.([Paper](https://aclanthology.org/2022.semeval-1.154/)).

**Hinglish HateBERT: Hate Speech Classification in Hinglish language (Accepted in 7th ICT4SD Proceedings by Springer) :** In this work, we introduce Hinglish HateBERT, a pre-trained BERT model for Hate speech detection in code-mixed Hindi English language. The model was trained on a large-scale dataset having offensive and non-offensive content to avoid any bias. We finetuned Hinglish HateBERT using CNN and LSTM and performed experimentation on three publicly available datasets. Further, we presented a detailed comparative performance of our model with publicly available pretrained models for classifying hate speech in Hinglish. We observed that our proposed model Hinglish HateBERT significantly outperformed for two datasets.
 ([GitHub](https://github.com/nikhilbyte/EH-Code-Mixed-BERT)) ([HuggingFace](https://huggingface.co/niksss/Hinglish-HATEBERT)).

**niksss at Qur’an QA 2022: A Heavily Optimized BERT Based Model for Answering Questions from the Holy Qu’ran” (Published in Proceedings of the OSACT 2022 Workshop @ LREC 2022) :** This paper presents the system description by team niksss for the Qur’an QA 2022 Shared Task. The goal of this shared task was to evaluate systems for Arabic Reading Comprehension over the Holy Quran. The task was set up as a question-answering task, such that, given a passage from the Holy Quran (consisting of consecutive verses in a specific surah(Chapter)) and a question (posed in Modern Standard Arabic (MSA)) over that passage, the system is required to extract a span of text from that passage as an answer to the question. The span was required to be an exact sub-string of the passage. We attempted to solve this task using three techniques namely conditional text-to-text generation, embedding clustering, and transformers-based question answering.([Paper](http://www.lrec-conf.org/proceedings/lrec2022/workshops/OSACT/pdf/2022.osact-1.15.pdf)).

**Niksss At Hinglisheval: Language-Agnostic Bert-Based Contextual Embeddings With Catboost For Quality Evaluation Of The Low-Resource Synthetically Generated Code-Mixed Hinglish Text” (Published in 15th International Natural Language Generation Conference 2022) :** This paper presents the system description by team niksss for the Qur’an QA 2022 Shared Task. The goal of this shared task was to evaluate systems for Arabic Reading Comprehension over the Holy Quran. The task was set up as a question-answering task, such that, given a passage from the Holy Quran (consisting of consecutive verses in a specific surah(Chapter)) and a question (posed in Modern Standard Arabic (MSA)) over that passage, the system is required to extract a span of text from that passage as an answer to the question. The span was required to be an exact sub-string of the passage. We attempted to solve this task using three techniques namely conditional text-to-text generation, embedding clustering, and transformers-based question answering.([Paper](https://inlgmeeting.github.io/poster_paper34.html)).

**Fine-tune BERT to Classify Hate Speech in Hindi English Code-Mixed Text” (Published in FIRE 2021 CEUR) :** With the exponential growth in internet technologies and social media usage, communicating and gathering information across countries is increasing at faster pace. These platforms provide opportunities to share opinions and suggestions about any socio-political events. Apart from seeking advantage, a person or community often misuses these platforms to post hate speech content. Content targeted towards casteism, racism, sexism, and insulting is called hate speech. The majority of the user are multilingual speakers knowing two or more languages. Out of them, English is the most known language. Knowing native language like Hindi and wellknown languages like English, most social media users switch between Hindi and English while writing content on the social media platform. The phenomenon of mixing two languages like Hindi and English is called as Hindi English Code-mixed language. We have participated in HASOC subtask2 to classify Hi-En code mixed conversation in two classes as Hate Offensive (HOF) and Non-hate (NONE). We have experimented with various methods as Multilingual Bert (MBert), finetuned Pretrained Bert-base-uncased, an ensemble of Bert-base-uncased, XLNet and transfer learning-based approaches. We have analyzed that finetuned Bert-baseuncased has outperformed all other models. In subtask 2 of HASOC 2021, overall, 16 teams have participated, our team named “Rider” has achieved position 4 in Macro F1 score and Macro precision.([Paper](https://parth126.github.io/T1-32.pdf)).



---
### Other NLP Projects


**Automatic Evidence Grading System” (2021 ALTA Shared Task) :** Built an automatic evidence grading system for evidence-based medicine. Evidence-based medicine is a medical practice which requires practitioners to search medical literature for evidence when making clinical decisions. The practitioners are also required to grade the quality of extracted evidence on some chosen scale. The goal of the grading system is to automatically determine the grade of an evidence given the article abstract(s) from which the evidence is extracted.([Github](https://github.com/nikhilbyte/Automatic-evidence-grading-system-for-evidence-based-medicine)).

<center><img src="images/BERT-classification.png"/></center>

---
### Detect Food Trends from Facebook Posts: Co-occurence Matrix, Lift and PPMI

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/detect-food-trends-facebook.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/facebook-detect-food-trends)

<div style="text-align: justify">First I build co-occurence matrices of ingredients from Facebook posts from 2011 to 2015. Then, to identify interesting and rare ingredient combinations that occur more than by chance, I calculate Lift and PPMI metrics. Lastly, I plot time-series data of identified trends to validate my findings. Interesting food trends have emerged from this analysis.</div>
<br>
<center><img src="images/fb-food-trends.png"></center>
<br>

---
### Detect Spam Messages: TF-IDF and Naive Bayes Classifier

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/detect-spam-nlp.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/detect-spam-messages-nlp/blob/master/detect-spam-nlp.ipynb)

<div style="text-align: justify">In order to predict whether a message is spam, first I vectorized text messages into a format that machine learning algorithms can understand using Bag-of-Word and TF-IDF. Then I trained a machine learning model to learn to discriminate between normal and spam messages. Finally, with the trained model, I classified unlabel messages into normal or spam.</div>
<br>
<center><img src="images/detect-spam-nlp.png"/></center>
<br>

---
## Data Science

### Credit Risk Prediction Web App

[![Open Web App](https://img.shields.io/badge/Heroku-Open_Web_App-blue?logo=Heroku)](http://credit-risk.herokuapp.com/)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/chriskhanhtran/credit-risk-prediction/blob/master/documents/Notebook.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/credit-risk-prediction)

<div style="text-align: justify">After my team preprocessed a dataset of 10K credit applications and built machine learning models to predict credit default risk, I built an interactive user interface with Streamlit and hosted the web app on Heroku server.</div>
<br>
<center><img src="images/credit-risk-webapp.png"/></center>
<br>

---
### Kaggle Competition: Predict Ames House Price using Lasso, Ridge, XGBoost and LightGBM

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/ames-house-price.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/kaggle-house-price/blob/master/ames-house-price.ipynb)

<div style="text-align: justify">I performed comprehensive EDA to understand important variables, handled missing values, outliers, performed feature engineering, and ensembled machine learning models to predict house prices. My best model had Mean Absolute Error (MAE) of 12293.919, ranking <b>95/15502</b>, approximately <b>top 0.6%</b> in the Kaggle leaderboard.</div>
<br>
<center><img src="images/ames-house-price.jpg"/></center>
<br>

---
### Predict Breast Cancer with RF, PCA and SVM using Python

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/breast-cancer.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/predict-breast-cancer-with-rf-pca-svm/blob/master/breast-cancer.ipynb)

<div style="text-align: justify">In this project I am going to perform comprehensive EDA on the breast cancer dataset, then transform the data using Principal Components Analysis (PCA) and use Support Vector Machine (SVM) model to predict whether a patient has breast cancer.</div>
<br>
<center><img src="images/breast-cancer.png"/></center>
<br>

---
### Business Analytics Conference 2018: How is NYC's Government Using Money?

[![Open Research Poster](https://img.shields.io/badge/PDF-Open_Research_Poster-blue?logo=adobe-acrobat-reader&logoColor=white)](pdf/bac2018.pdf)

<div style="text-align: justify">In three-month research and a two-day hackathon, I led a team of four students to discover insights from 6 million records of NYC and Boston government spending data sets and won runner-up prize for the best research poster out of 18 participating colleges.</div>
<br>
<center><img src="images/bac2018.JPG"/></center>
<br>

---
## Filmed by me

[![View My Films](https://img.shields.io/badge/YouTube-View_My_Films-grey?logo=youtube&labelColor=FF0000)](https://www.youtube.com/watch?v=vfZwdEWgUPE)

<div style="text-align: justify">Besides Data Science, I also have a great passion for photography and videography. Below is a list of films I documented to retain beautiful memories of places I traveled to and amazing people I met on the way.</div>
<br>

- [Ada Von Weiss - You Regret (Winter at Niagara)](https://www.youtube.com/watch?v=-5esqvmPnHI)
- [The Weight We Carry is Love - TORONTO](https://www.youtube.com/watch?v=vfZwdEWgUPE)
- [In America - Boston 2017](https://www.youtube.com/watch?v=YdXufiebgyc)
- [In America - We Call This Place Our Home (Massachusetts)](https://www.youtube.com/watch?v=jzfcM_iO0FU)

---
<center>© 2020 Khanh Tran. Powered by Jekyll and the Minimal Theme.</center>
