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

**WNUT-2020 Task 2: Identification of informative COVID-19 English Tweets :** This paper describes the team loner’s system submitted to WNUT-2020 Task 2: Identification of informative COVID-19 English Tweets. The proposed system is based on the pre-trained transformer based model, namely BERT. A comparative analysis of various traditional machine learning algorithms is done on the provided dataset. We further improve our performance by using COVID-Twitter Bert (CT-BERT) which is pre-trained on a large corpus of Twitter messages on the topic of COVID-19 and fine-tuned it on our dataset. This method landed us at the 5th position on the leaderboard of this challenge.([Unpublished Paper](https://drive.google.com/file/d/1i5POM0hUuVTvePfeoYik5ftmb9s-hbRq/view?usp=sharing)).

**Fake News Detection :** Fine-Tuned Siamese Dual BERT using a Spanish Language model called BETO for building a Fake News classification model in Spanish Corpus.([Github](https://github.com/nikhilbyte/FakeNewsDetection)).

**Automatic Evidence Grading System” (2021 ALTA Shared Task) :** Built an automatic evidence grading system for evidence-based medicine. Evidence-based medicine is a medical practice which requires practitioners to search medical literature for evidence when making clinical decisions. The practitioners are also required to grade the quality of extracted evidence on some chosen scale. The goal of the grading system is to automatically determine the grade of an evidence given the article abstract(s) from which the evidence is extracted.([Github](https://github.com/nikhilbyte/Automatic-evidence-grading-system-for-evidence-based-medicine)).

**Detecting Signs of Depression from Social Media Text-LT-EDI@ACL 2022 :** Given social media postings in English, my system should classifies the signs of depression into three labels namely “not depressed”, “moderately depressed”, and “severely depressed”.
([Github](https://github.com/nikhilbyte/Signs-of-depression-from-social-media)).

**Homophobia/Transphobia Detection in social media comments:-ACL 2022:** Given a comments, my system predicst whether or not it contains any form of homophobia/transphobia.([Github](https://github.com/nikhilbyte/Homophobia-Transphobia-Detection-in-social-media-comments)).


---


---


---
## Data Science

### Increasing safety and sustainability of micro-mobility modes in pandemic

<div style="text-align: justify">We propose a multi-criteria route planning technique for cyclists and pedestrians. This aims at objectively determining the routes based on various criteria about the safety of a given route while keeping the user away from potential COVID-19 transmission spots. The vulnerable spots include places such as a hospital or medical facilities, contained residential areas, and roads with high connectivity and influx of people. Our proposed algorithm returns a multi-criteria route modeled by considering safety as well as the shortest route for user ease and a short time of outside environment exposure. So first we visualize containment zones and medical facilities in a region. We will define parameters pertaining to user facilitation in terms of safety, amenity reach, and mobility ease using the walking and cycling modes. This visualization can help authorities facilitate these modes by planning bike and walking lanes. </div> 
([WebApp](https://peacetech.epcc.ed.ac.uk/about)) ([Journal Paper](https://www.mdpi.com/2220-9964/10/8/571/htm)) 

---
### Automatic Enzyme Sequence Annotation

<div style="text-align: justify">All enzymes are made up of chains of amino acids, which determine their structure and behavior. The proposed model helps in automatically annotating the enzyme sequence with just the enzyme sequence.</div>
([Presentation & Project](https://devpost.com/software/automatic-enzyme-sequence-annotation?ref_content=user-portfolio&ref_feature=in_progress)) [I also won the special prize at AI4Science Hosted by Caltech at Devpost.](https://ai4science.io/winners.html)

---
### Patient Clinic Proximity Finder

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/nikhilbyte/Patient-Clinic-Proximity-Finder)

<div style="text-align: justify">Built a python app that matches a patient to their nearest clinic using Google's Geocoding API.</div>

### Automatic Speech Recognition model for Luganda: An African Language

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/nikhilbyte/Mozilla_Luganda_AutomaticSpeechRecognition)

<div style="text-align: justify">Participated in Mozilla Luganda Automatic Speech Recognition Competition and built a machine learning ASR model to accurately recognize words spoken in Luganda.</div>
[..And landed on 8th position on the leaderboard.](https://zindi.africa/competitions/mozilla-luganda-automatic-speech-recognition)


---
### Diagnosing COVID-19 using Acoustics: DiCOVA Challenge

<div style="text-align: justify">Used Pre-Trained Audio Neural Networks and Audio Spectrogram Transformers to build a machine learning model that can detect COVID-19 be from the cough, breathing and speech sound signals of an individual.</div>
[Unpublished System Report.](https://drive.google.com/file/d/103JmctaOGJ6c-BuDoVdLvHUCBnkck5YI/view?usp=sharing)

---
## Computer Vision

 **Semi-Supervised Change Detection:** Multiclass change detection between between two very high resolution sentinel images
([Gitlab](https://gitlab.com/nikhilsinghisthe/semi-supervised-change-detection.)) ([More Information](http://www.classic.grss-ieee.org/earthvision2021/challenge.html))


**Face Detection in the Dark:** A Face Detection model Using TinaFace, EnlightenGAN and Multi-Stage Progressive Image Restoration
([Github](https://github.com/nikhilbyte/Face-Datection-in-the-dark))


**Covid Severity Estimation:** A CNN based Image Regression model for finding the severity of lung infection(percent) in a patient infected with the coronavirus.([Github](https://github.com/nikhilbyte/Covid_Severity_Estimation))

**An End-to-End Computer Vision(CV) Problem:** In this project I demonstrate, how to deal with a CV problem from stract, through creating the data to creating a strong model by building a Shoe Type classifier.([Github](https://github.com/nikhilbyte/ShoeTypeClassifier))

## Indigenous ML Problems

**Multimodal Sentiment Analysis:** Examining the performance of various SOTA vision and text models on multiple modalities.
([Github](https://github.com/nikhilbyte/Multimodal-Sentiment-Analysis))

**To be or not to be ? Mortality Prediction:** Used Catboost with Optuna to predict the survival of a patient given his or her medical record. More specifically, predicted whether or not the patients died during their stay at the hospital.
([Github](https://github.com/nikhilbyte/Mortality-Prediction))


<center>© 2020 Khanh Tran. Powered by Jekyll and the Minimal Theme.</center>
