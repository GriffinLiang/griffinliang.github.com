<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Multi-task Learning

---

#### 2016_NIPS_Integrated Perception with Recurrent Multi-Task Neural Networks

##### Background
- A major advantage of natural intelligences: they work well for all perceptual problems together, solving them efficiently and coherently in an integrated manner.
-  Two questions:

-- whether deep neural networks can learn universal image representations, useful not only for a single task but for all of them.
-- how the solutions to the different tasks can be integrated in this framework.

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
