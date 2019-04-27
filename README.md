# squad-QA
by Boxiao Pan, Gael Colas and Shervine Amidi, graduate students from Stanford University

This is our final project for the CS224N: "Deep Learning for Natural Language Processing" class in Stanford (2019). Our teacher was Pr. Christopher Manning.

Language: Python (Pytorch)

Goal: Question Answering on the updated Stanford Question Answering Dataset, named [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)

Our final model used a Segment-Based Aggregation method on the BiDAF architecture.
We implemented a powerful word embedding to further improve the performance: pretrained GloVe embeddings + character-level word embedding + tag features (POS, NER) and hand engineered features (EM, TF\*IDF).
Finally, we designed a new loss term: the "index distance-aware loss". The idea being that contrary to the cross-entropy loss, we should penalize more wrong predictions that are far away from the true index than those which are close.

For more details, please refer to our final report at the root: "cs224n_project_final-report.pdf".

**Test results: EM: 61.2 ; F1: 65.0**

## Acknowledgement
The starter code for this project was a custom BiDAF implementation provided by Chris Chute: [starter code](https://github.com/chrischute/squad)

## Requirements
To create the environment with all the necessary packages, run "conda env create -f environment.yml".

To activate the environment: "conda activate squad".

### To use the Stanford CoreNLP tokenizer
Guidelines from: https://github.com/Lynten/stanford-corenlp/blob/master/README.md

Java 1.8+ (Check with command: `java -version`) ([Download Page](http://www.oracle.com/technetwork/cn/java/javase/downloads/jdk8-downloads-2133151-zhs.html))

Stanford CoreNLP ([Download Page](https://stanfordnlp.github.io/CoreNLP/history.html))

| Py Version | CoreNLP Version |
| --- | --- |
|v3.7.0.1 v3.7.0.2 | CoreNLP 3.7.0 |
|v3.8.0.1 | CoreNLP 3.8.0 |
|v3.9.1.1 | CoreNLP 3.9.1 |
