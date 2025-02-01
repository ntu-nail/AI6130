# AI6130: Large Language Models

# Course Objectives

The course on Large Language Models (LLMs) aims to equip students with a comprehensive understanding of the principles, architectures, and applications of state-of-the-art LLMs like GPT-4. This course is designed for graduate students in computer science, data science, and related fields who have a foundational knowledge of machine learning and artificial intelligence. By taking this course, students will gain valuable skills in developing, fine-tuning, and deploying LLMs, which are increasingly integral to advancements in natural language processing, automated content creation, and AI-driven decision-making. This expertise will not only enhance their academic and research capabilities but also significantly boost their employability in tech industries, research institutions, and innovative startups focused on AI and machine learning technologies.

**Optional Textbooks**

- Deep Learning by Goodfellow, Bengio, and Courville [free online](http://www.deeplearningbook.org/)
- Machine Learning — A Probabilistic Perspective by Kevin Murphy [online](https://doc.lagout.org/science/Artificial%20Intelligence/Machine%20learning/Machine%20Learning_%20A%20Probabilistic%20Perspective%20%5BMurphy%202012-08-24%5D.pdf)
- Natural Language Processing by Jacob Eisenstein [free online](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf)
- Speech and Language Processing by Dan Jurafsky and James H. Martin [(3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/)

**Optional Papers**

- On the Opportunities and Risks of Foundation Models
- Multimodal Foundation Models: From Specialists to General-Purpose Assistants
- Large Multimodal Models: Notes on CVPR 2023 Tutorial
- A Comprehensive Survey on Pretrained Foundation Models: A History from BERT to ChatGPT
- Interactive Natural Language Processing
- Towards Reasoning in Large Language Models: A Survey


# Intended Learning Outcomes

By the end of this course, you should be able to:

- Analyze the underlying architectures and mechanisms of large language models.

- Implement and fine-tune large language models for specific applications.

- Evaluate the performance of large language models in various contexts.

- Design novel applications leveraging large language models to solve real-world problems.

- Critically assess the limitations and potential improvements of current large language models.

# Assessment Approach

**Assignments (individually graded)**

- There will be two (2) assignments contributing to ***2 * 25% = 50%*** of the total assessment.
- Students will be graded individually on the assignments. They will be allowed to discuss with each other on the homework assignments, but they are required to submit individual write-ups and coding exercises.

**Final Project (Group work but individually graded)**

- There will be a final project contributing to the remaining ***50%*** of the total coursework assessment.
  - ***3–6*** people per group
  - Presentation: ***20%***, report: ***30%***
- The project will be group work but the students will be graded individually. The final project presentation will ensure the student’s understanding of the project

# Course Prerequisites

- Proficiency in Python (using Numpy and PyTorch)
- Deep Learning and NLP basics 

# Teaching

<p align="center" width="100%">Instructor</p>

<p align="center" width="100%">
    <img width="20%" src="/assets/images/Luu_Anh_Tuan.png"> 
</p>

<p align="center" width="100%"><a href="https://tuanluu.github.io/">Luu Anh Tuan</a></p>
<p align="center" width="100%">anhtuan.luu@ntu.edu.sg</p>


<p align="center" width="100%">Teaching Assistants</p>

<p align="center" width="100%">
    <img width="20%" src="/assets/images/ntcd.png"> 
</p>

<p align="center" width="100%">Nguyen Tran Cong Duy</p>
<p align="center" width="100%">NGUYENTR003@e.ntu.edu.sg</p>



# Schedule & Course Content

## Week 1: Introduction

[Lecture Slide](https://drive.google.com/file/d/1IJ5qWqzIQcFWNmL5bChMzuYO1Ujg9TMe/view?usp=sharing)

### Lecture Content

- Logistics of the course
- Introduction about deep learning
- Types of deep learning
- Introduction about Large language models

### Python & PyTorch Basics

- Programming in Python

  - Jupiter Notebook and [google colab](https://colab.research.google.com/drive/16pBJQePbqkz3QFV54L4NIkOn1kwpuRrj)
  - [Introduction to Python](https://colab.research.google.com/drive/1bQG32CFoMZ-jBk02uaFon60tER3yFx4c)
  - Deep Learning Frameworks
  - Why Pytorch?
  - [Deep learning with PyTorch](https://drive.google.com/file/d/1c33y8bkdr7SJ_I8-wmqTAhld-y7KcspA/view?usp=sharing)
- [Supplementary]
  - Numerical programming with Numpy/Scipy - [Numpy intro](https://drive.google.com/file/d/1cUzRzQGURrCKes8XynvTTA4Zvl_gUJdc/view?usp=sharing)
  - Numerical programming with Pytorch - [Pytorch intro](https://drive.google.com/file/d/18cgPOj2QKQN0WR9_vXoz6BoravvS9mTm/view?usp=sharing)

## Week 2: Neural Networks & Optimization Basics

[Lecture Slide](https://drive.google.com/file/d/1uEzjPEZh4gNL8sp40AlAYktdb6gLDBJm/view?usp=sharing)

### Lecture Content

- From Logistic Regression to Feed-forward NN
- Activation functions
- SGD with Backpropagation
- Adaptive SGD (adagrad, adam, RMSProp)
- Word Embeddings
- CNN
- RNN
- RNN variants
- Attention
### Practical exercise with Pytorch

- [Deep learning with PyTorch](https://colab.research.google.com/drive/1aZVfsPUko-ugt1TVCmRwqGJXlxEJVaTq?usp=sharing)
- [Linear Regressionn](https://colab.research.google.com/drive/12QpBf7x_Jt6-zypN4OrUFFHXz1u6CmYe?usp=sharing)
- [Logistic Regression](https://colab.research.google.com/drive/1nTrYW5dUu6WO9cx7SGEvP9oX7qRbsGJk?usp=sharing)
- [Numpy notebook](https://colab.research.google.com/drive/1IAonxZnZjJb0_xUVWHt5atIxaI5GTJQ2) [Pytorch notebook](https://colab.research.google.com/drive/1YzZrMAmJ3hjvJfNIdGxae9kxGABG6yaT)
  - Backpropagation
  - Dropout
  - Batch normalization
  - Initialization
  - Gradient clipping
- Word2Vec Tutorial - The Skip-Gram Model, [blog](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- [Fine-grained Opinion Mining with Recurrent Neural Networks and Word Embeddings](https://www.aclweb.org/anthology/D15-1168/)
- [Sequence to Sequence Learning with Neural Networks (original seq2seq NMT paper)](https://arxiv.org/pdf/1409.3215.pdf)
- [Effective Approaches to Attention-based Neural Machine Translation](https://nlp.stanford.edu/~lmthang/data/papers/emnlp15_attn.pdf)


## Week 3: Language Models

[Lecture Slide](https://drive.google.com/file/d/1IUgxN21v2u528hWvOtOiiLJN_J_1CH74/view?usp=sharing)

[Recording of Lecture 3](https://drive.google.com/file/d/1CiqGLKrAMRRuVdrgzx4Ce_UcFqCLUCKk/view?usp=drive_link)

### Lecture Content

- Language model
- N-gram based LM
- Window-based Language Model
- Neural Language Models
- Encoder-decoder
- Seq2Seq
- Sampling algorithms
- Beam search

### Suggested Readings

- [Sequence to Sequence Learning with Neural Networks (original seq2seq NMT paper)](https://arxiv.org/pdf/1409.3215.pdf)
- [N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
- [Karpathy’s nice blog on Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Building an Efficient Neural Language Model](https://research.fb.com/building-an-efficient-neural-language-model-over-a-billion-words/)

## Week 4: Attention and Transformer

[Lecture Slide](https://drive.google.com/file/d/1rNQctj-bg8NwDjeftIty6B_9rbrWiYxB/view?usp=sharing)

[Final Project Groups](https://docs.google.com/spreadsheets/d/1rtQcBkbgpK8Sbn42Ffq2b15nThiV8t-7_f--6YMvHOM/edit?usp=drive_link)

### Lecture Content

- Information bottleneck issue with vanilla Seq2Seq
- Attention to the rescue
- Details of attention mechanism
- Transformer architecture
  - Self-attention
  - Positional encoding
  - Multi-head attention


### Suggested Readings

- [Neural Machine Translation by Jointly Learning to Align and Translate (original seq2seq+attention paper)](https://arxiv.org/pdf/1409.0473.pdf)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
