---
layout: post
title: Evaluation Metric
subtitle: scientific experiment
---


<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

### F1 score ###

F1 score is a harmonic mean of precision and recall. $$F_1=2\cdot\frac{precision \cdot recall}{precision+recall}$$

### Micro-averaged F-Measure ###

[Micro-averaged F1](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.8244&rep=rep1&type=pdf "original paper") for a binary or multiclass problem is identical to plain old accuracy. In micro-averaging, F-measure is computed globally over all category decisions. precision and recall are obtained by summing over all individual decisions:

$$Recall=\frac{\sum_{i=1}^M}{TP_i}{\sum_{i=1}^M(TP_i+FP_i)}, Precision=\frac{\sum_{i=1}^M}{TP_i}{\sum_{i=1}^M(TP_i+FN_i)}.$$


$$F_{micro}=\F_1=2\cdot\frac{precision \cdot recall}{precision+recall}$$

Micro-averaged F-measure gives equal weight to each document and is therefore considered as an average over all the document/category pairs. It tends to be dominated by the classifier’s performance on common categories.

### Macro-F1 ###

Macro-averaged recall, also known as balanced accuracy, is popular and more useful than macro-averaged F1 for a binary problem.

$$F_i=2\cdot\frac{precision_i \cdot recall_i}{precision_i+recall_i}$$

$$F_{macro}=\frac{\sum_{i=1}^MF_i}{M}$$
