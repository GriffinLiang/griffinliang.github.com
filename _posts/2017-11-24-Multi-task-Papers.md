<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Multi-task Learning

---

#### 2017_CVPR_UberNet : Training a ‘Universal’ Convolutional Neural Network for Low-, Mid-, and High-Level Vision using Diverse Datasets and Limited Memory

这篇文章的核心内容是把一些底层、中层和高层的任务同时进行训练。它的主要思想是：把所有的任务都都归纳到同一种结构，因为有些任务需要使用底层的卷积特征，而有些任务需要高层的卷积特征，所以该方法把网络所有层的输出都作为输入，然后让这些任务自动选择使用哪些特征。除此之外，这个方法的弊端是即便数据集中没有所有的任务标注，该网络也还是要进行所有任务的训练，没有标注的任务权重损失设为0。还有是需要占用比较大的显存，但作者给出了一些方法解决这个问题。从实验上来看，将所有任务统一训练有时候并不能比一些单任务的方法好，比如table (6)的语义分割结果，因为这些任务虽然相关但有时候也需要一些特殊的处理，统一到一个框架就势必要舍弃一些特殊的操作。

#### 2017_NIPS_Learning multiple visual domains with residual adapters

##### Background

- Learn data representations that work well for many different types of problems and data.
    - The first challenge is to extract from a given image diverse information, such as image-level labels, semantic segments, object bounding boxes, object contours, occluding boundaries, vanishing points, etc. 
    - The second aspect is to model simultaneously many different visual domains, such as
Internet images, characters, glyph, animal breeds, sketches, galaxies, planktons. [**This work**]
- A high degree of parameter sharing while maintaining or even improving the accuracy of domain-specific representations.

##### Method (multivalent neural network)

- Reconfigure a deep neural network on the fly to work on different domains as needed.
    - Learning to learn: learn neural networks that predict, in a data-dependent manner, the parameters of another.
    - We note that linearly parametrizing a filter bank is the same as introducing a new, intermediate convolutional layer in the network. The linear combination is domain-specific.
- Introduce residual adapter module and use it to parameterize the standard residual network architecture.
    - x -> conv1 (x_residual) + BN + Domain specific conv (1*1) + x_residual + BN + Relu
- Sequential learning and avoiding forgetting
    - fine-tuning: often a poor choice for learning shared representations as it tends to quickly forget the original tasks.
    - without forgetting: maintain information about older tasks as new ones are learned. one can pre-train the domain-agnostic parameters on a large domain such as ImageNet, and then fine-tune only the domain-specific parameters αd for each new domain.
- Visual Decathlon Challenge, a benchmark that evaluates the ability of representations to
capture simultaneously ten very different visual domains and measures their ability
to recognize well uniformly

##### Conclusion:

把一小部分参数变为domain-specific, 所以选择1\*1卷积作为入口点。具体来讲，在普通的3\*3卷积后面加入domain-specific的1\*1卷积，因此domain之间share的参数是specific的9倍。感觉对于一篇NIPS来讲，本文提出的问题可能更重要一些，但确实很缺乏应用场景，比如把ImageNet跟MNIST结合起来学习有多大的意义？方法比较简单从实验来看也比较有效，可以考虑实现一下。

---

#### 2016_NIPS_Integrated Perception with Recurrent Multi-Task Neural Networks

##### Background
- A major advantage of natural intelligences: they work well for all perceptual problems together, solving them efficiently and coherently in an integrated manner.
-  Two questions:
    - whether deep neural networks can learn universal image representations, useful not only for a single task but for all of them.
    - how the solutions to the different tasks can be integrated in this framework.

##### Method (Multinet)
- Deep image features are shared between tasks.
- Tasks can interact in a recurrent manner by encoding the results of their analysis in a common
shared representation of the data.


##### Performance: 
Individual tasks in standard benchmarks can be improved first by sharing features between them and then, more significantly, by integrating their solutions in the common representation.
- Object classification, detection and part detection results in the PASCAL VOC 2010

    |Method | classification | object-detection | part-detection |
    | ------ | ------ | ------ | ------ | 
    | Independent | 76.4 | 55.5 | 37.3 | 
    | Multi-task | 76.2 | 57.1 | 37.2 |
    | Ours | **77.4** | **57.5** | **38.8** |
    
##### Conclusion:
一般的multi-task learning框架的两种策略是：特征共享或者参数共享。本文的思路是融合多个任务的网络输出，生成一个所有任务的共享表示，然后迭代的优化模型并更新这个共享表示。作者在4.1章节中提到迭代两轮性能已经可以达到最好性能的99%。

---
