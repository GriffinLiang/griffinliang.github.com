---
layout: post
title: Caffe-SigmoidCrossEntropyLossLayer
subtitle: Loss layer
---


<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>


The definition of cross-entropy (logistic) loss: $$E = -\frac{1}{N} \sum_{n=1}^{N} p_n log \tilde{p}_n + (1-p_n)log(1-\tilde{p}_n),$$ where \\( \tilde{p}_n = \frac{1}{1+e^{-x_n}} .\\) 

The loss for \\( x_n \\) is: $$ L_n = x_n(p_n -1)-log(1+e^{-x_n}).$$ However, the range for \\( e^{-x_n} \in (1,\infty]\\) when $$ x_n<0 $$. To avoid this, the author uses \\(-log(1+e^{-x_n})=log\frac{1}{1+e^{-x_n}}=log \frac{e^{x_n}}{e^{x_n}+1}=x_n-log(1+e^{x_n}) \\) to change the range into (0,1).

The final cross-entropy (logistic) loss: $$E = -\frac{1}{N} \sum_{n=1}^{N} x_n(p_n - (x_n \geq 0)-log(1+e^{x_n - 2x_n(x_n \geq 0)})).$$ 


This layer is implemented rather than separate SigmoidLayer + CrossEntropyLayer as its gradient computation is more numerically stable. At test time, this layer can be replaced simply by a SigmoidLayer.

For the gradient, \\( \frac{\partial L_n} {\partial x_n} = (p_n-1)+\frac{e^{-x_n}}{1+e^{-x_n}} = p_n - \tilde{p}_n.\\)

**PS:** 
InfogainLoss is a generalized softmax by considering the label relationship. For example, if loss of predicting an image of dog as cat should be smaller than chair. Infogain matrix $$ H \in  \mathbb{R}^{K \times K} $$ is used to reflect to label relationship where K is the number of categories. This can be provided as the third bottom blob input or provided as the infogain_mat in the InfogainLossParameter. If $$ H = I $$, this layer is equivalent to the Softmax Loss.

**Paramters:**

* bottom blob vector
 1. $$(N\times C\times H\times W)$$ the scores $$x\in [-\infty, \infty]$$. Map the input into probability predictions $$\tilde{p}_n=\sigma(x_n)\in [0,1]$$
 2. $$(N\times C\times H\times W)$$ the targets $$y \in [0,1] $$

* top blob vector (length 1)
 1. $$(1\times 1\times 1\times 1)$$ the computed cross-entropy loss

**Members:**

```cpp
  /// The internal SigmoidLayer used to map predictions to probabilities.
  shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;
  /// sigmoid_output stores the output of the SigmoidLayer.
  shared_ptr<Blob<Dtype> > sigmoid_output_;
  /// bottom vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  /// top vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_top_vec_;
```

**Functions:**

 * Reshape

```cpp
template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_); // No implementation of Reshape for sigmoid_layer so this function is called from neuron_layer.cpp
}
```

   

```cpp
// loss_layer.cpp
template <typename Dtype>
void LossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
}
```

```cpp
// neuron_layer.cpp
template <typename Dtype>
void NeuronLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}
```

 * LayerSetUp

```cpp
template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get()); // Returns the stored pointer.
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_); // There is no implemetation for SetUp function of SigmoidCrossEntropyLossLayer and its base class NeuronLayer. So the function is called from Layer.cpp
}
```

```cpp
// loss_layer.cpp
template <typename Dtype>
void LossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // LossLayers have a non-zero (1) loss by default.
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }
}
```

 * Forward_cpu

```cpu
template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}
```
 
 * Backward_cpu

```cpp

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(SigmoidCrossEntropyLossLayer, Backward);
#endif

```
