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

- Proficiency in Deep Learning models
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

[Lecture Slide](https://drive.google.com/file/d/1orix_P1NfA6BX7lQTcaS-aQIDii_-gC_/view?usp=sharing)

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

[Lecture Slide](https://drive.google.com/file/d/1OGiDURG2yVF_eSvJOxOj4w15MsG8G6yL/view?usp=sharing)

### Lecture Content

- From Logistic Regression to Feed-forward NN
- Activation functions
- SGD with Backpropagation
- Adaptive SGD (adagrad, adam, RMSProp)
- Word Embeddings
- CNN
- RNN
- RNN variants
- Information bottleneck issue with vanilla Seq2Seq
- Attention to the rescue
- Details of attention mechanism
- Transformer architecture
  - Self-attention
  - Positional encoding
  - Multi-head attention
    
### Practical exercise with Pytorch

- [Deep learning with PyTorch](https://colab.research.google.com/drive/1aZVfsPUko-ugt1TVCmRwqGJXlxEJVaTq?usp=sharing)
- [Linear Regression](https://colab.research.google.com/drive/12QpBf7x_Jt6-zypN4OrUFFHXz1u6CmYe?usp=sharing)
- [Logistic Regression](https://colab.research.google.com/drive/1nTrYW5dUu6WO9cx7SGEvP9oX7qRbsGJk?usp=sharing)
- [Numpy notebook](https://colab.research.google.com/drive/1IAonxZnZjJb0_xUVWHt5atIxaI5GTJQ2) [Pytorch notebook](https://colab.research.google.com/drive/1YzZrMAmJ3hjvJfNIdGxae9kxGABG6yaT)
  - Backpropagation
  - Dropout
  - Batch normalization
  - Initialization
  - Gradient clipping

### Suggested Readings

- Word2Vec Tutorial - The Skip-Gram Model, [blog](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- [Fine-grained Opinion Mining with Recurrent Neural Networks and Word Embeddings](https://www.aclweb.org/anthology/D15-1168/)
- [Sequence to Sequence Learning with Neural Networks (original seq2seq NMT paper)](https://arxiv.org/pdf/1409.3215.pdf)
- [Effective Approaches to Attention-based Neural Machine Translation](https://nlp.stanford.edu/~lmthang/data/papers/emnlp15_attn.pdf)
- [Neural Machine Translation by Jointly Learning to Align and Translate (original seq2seq+attention paper)](https://arxiv.org/pdf/1409.0473.pdf)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)


## Week 3: Language Models

[Lecture Slide](https://drive.google.com/file/d/1cfMOOoJ0Wg6oesQt3KQ76p6o9ivIvBTV/view?usp=sharing)

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



## Week 4: Effective Transformers

[Lecture Slide](https://drive.google.com/file/d/1gxwWO3uMbUT7QK2R0UOFhp4HuCGSkcu-/view?usp=sharing)

[Instruction to choose final project's topic](https://drive.google.com/file/d/1DPS7vx1pa6k5rZM1rbIJl5vW4KtmDSvk/view?usp=sharing)

### Lecture Content

- FFN
- Mixture of Experts
- Attention
- Layer Norm
- Positional Encoding

### Suggested Readings
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Hendrycks and Gimpel. 2016. Gaussian Error Linear Units. 
- Ramachandran et al. 2017. Searching for Activation Functions.
- Shazeer 2017. GLU Variants Improve Transformer
- Ainslie et al. 2023. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
- Noam Shazeer. 2019. Fast transformer decoding: One write-head is all you need.
- DeepSeek team. DeepSeek-V2

<!--
## Week 6: Pretrained Language Models and Large Language Models

[Lecture Slide](https://drive.google.com/file/d/1KTQMf2Tg5rqEZgdEFdTNkGCeEATG4ytW/view?usp=sharing)

### Lecture Content
- About pre-training
- Why we need pre-training
- Does pre-training indeed help?
- Pre-trained Language models
- Large Language Models


### Suggested Readings
- Chang, Y., Wang, X., Wang, J., Wu, Y., Zhu, K., Chen, H., Yang, L., Yi, X., Wang, C., Wang, Y., Ye, W., Zhang, Y., Chang, Y., Yu, P.S., Yang, Q., & Xie, X. (2023). A Survey on Evaluation of Large Language Models. ArXiv, abs/2307.03109
- Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1), 5485-5551.
- A. Vaswani et al., “Attention is All you Need,” in Advances in Neural Information Processing Systems (NeurIPS), 2017.
- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. Advances in neural information processing systems, 33, 1877-1901.
- Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., ... & Fiedel, N. (2023). Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240), 1-113.
- Chen, Mark, et al. Evaluating Large Language Models Trained on Code. arXiv:2107.03374, arXiv, 14 July 2021. arXiv.org, https://doi.org/10.48550/arXiv.2107.03374.
- Touvron, Hugo, et al. Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv:2307.09288, arXiv, 19 July 2023. arXiv.org, https://doi.org/10.48550/arXiv.2307.09288.
- Jiang, Albert Q., et al. Mixtral of Experts. arXiv:2401.04088, arXiv, 8 Jan. 2024. arXiv.org, https://doi.org/10.48550/arXiv.2401.04088.

### Practical
- Using pretrained language model for classification: https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/pytorch/sequence_classification.ipynb
- LLM prompting: https://huggingface.co/docs/transformers/main/en/tasks/prompting

## Week 7: LLM finetuning

[Lecture Slide](https://drive.google.com/file/d/1mosUrnlqv5_x8g_OohxscGBUn_dJNxAg/view?usp=sharing)

Assignment 1 is out [here](https://docs.google.com/document/d/13ZgSgM5fF_5B_vFhVcusmjQCb-_uuWUd/edit?usp=sharing&ouid=118265590333925180950&rtpof=true&sd=true). **Deadline: 24 March 2025**.

### Lecture Content

- LLM full finetuning
- In-context learning
- Parameter-efficient finetuning
- Instruction finetuning

## Week 8: Instruction tuning & RLHF

[Lecture Slide](https://drive.google.com/file/d/1YV4tGxIMLw0DDniJ9MWQ4OkiondyNc3z/view?usp=sharing)

### Lecture Content

- Instruction tuning
- Multitask Prompted Training Enables Zero-shot Task Generalization (T0)
- LIMA: Less Is More for Alignment
- Instructed GPT

## Week 9: RLHF recap & DPO

[Lecture Slide](https://drive.google.com/file/d/1IKYgns-6uDrF2dqF0eJnynAeh9VIQzYH/view?usp=sharing)

### Lecture Content

- Reinforcement learning from human feedback (RLHF)
- Direct preference optimization (DPO)
- Frontier, pitfalls and open problems of RLHF

## Week 10: LLM Prompting

[Lecture Slide](https://drive.google.com/file/d/15YAo6GjRGwbMuNO_BuJK_Jic7VuvRLSg/view?usp=sharing)

### Lecture Content

- Chain-of-Thought Prompting
- Self-Consistency Improves Chain of Thought Reasoning in Language Models
- Tree of Thoughts Prompting
- Program of Thoughts Prompting
- Least-to-Most Prompting Enables Complex Reasoning in Large Language Models
- Measuring and Narrowing the Compositionality Gap in Language Models

## Week 11: Retrieval-augmented LMs

[Lecture Slide](https://drive.google.com/file/d/1TKDO6LlSiDJ6RTQdXe-tVXFQFp3rzZm8/view?usp=sharing)

Assignment 2 is out [here](https://docs.google.com/document/d/1iVhPjFRVUMn9yBpoEbYVw_7jx9QdxtSt/edit?usp=sharing&ouid=118265590333925180950&rtpof=true&sd=true). **Deadline: 22 April 2025**.

Instruction for final project report [here](https://drive.google.com/file/d/1_7UxmBsWNZDHhs9hv8mWvbw6muTZJFT2/view?usp=sharing)

### Lecture Content

- Limitations of parametric LLMs 
- What are retrieval-augmented LMs?
- Benefit of retrieval-augmented LMs
- Past: Architecture and training of retrieval-augmented LMs for downstream tasks
- Present: Retrieval-augmented generation with LLMs

## Week 12: Eﬃcient Inference Methods

[Lecture Slide](https://drive.google.com/file/d/1xkU606Nl_-vzjuCEQ98V1f42E3HhtxnL/view?usp=sharing)

### Lecture Content

- General concepts of eﬃcient inference methods for LLM serving 
- Speculative decoding systems 
- Model-based eﬃciency 
- Paged attention
- Flash attention

-->
