<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Visual Relationship Detection

---

### Arxiv2017_Acquiring Common Sense Spatial Knowledge through Implicit Spatial Templates

#### Spatial understanding
- Explicit spatial relationship (e.g., “on”, “below”, etc.), [only a few tens of]
- Implicit spatial relationship (e.g., “man riding horse”), [thousands of]
- Learning to compose the triplet (Subject, Relationship, Object) as a whole instead of
learning a template for each Relationship as:
    - the relative spatial configuration of “man” and the object is clearly distinct in (man, pulling, kite) than in (man, pulling, luggage) yet the action is the same.
    - other relationships such as “jumping” are highly informative about the spatial template, i.e., in (object1, jumping, object2), object2 is in a lower position than object1.

#### Method
- Input and representation layers. 
    - Embedding: subject, relationship,object
    - Subject spatial: center and half of width and height
    - Concatenation and one layer of neural network
- Output and loss
    - REG: Regress Object coordinates with MSE loss
    - PIX: 2D heatmap of pixel activations with binary cross-entropy

#### Experiments
- Visual Genome
- Need further reading

#### Conclusion
方法比较简单，实验很充分，有时间可以继续细读一下，尤其是借鉴预测spatial信息的思想和Generalized evaluations部分的实验。


---
