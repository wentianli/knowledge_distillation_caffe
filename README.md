# KnowledgeDistillation Layer (Caffe implementation)
This is a CPU implementation of knowledge distillation in Caffe.<br>
This code is heavily based on [softmax_loss_layer.hpp](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/softmax_loss_layer.hpp) and [softmax_loss_layer.cpp](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/softmax_loss_layer.cpp).<br><br>
Please refer to the [paper](https://arxiv.org/abs/1503.02531)<br>
```
Hinton, G. Vinyals, O. and Dean, J. Distilling knowledge in a neural network. 2015.
```
<br>

## Installation
1. Install [Caffe](https://github.com/BVLC/caffe/) in your directory `CAFFE`<br>
2. Clone this repository in your directory `ROOT`<br>
```bash
cd $ROOT
git clone https://github.com/wentianli/knowledge_distillation_caffe.git
```
3. Move files to your Caffe folder<br>
```bash
cp $ROOT/knowledge_distillation_layer.hpp $CAFFE/include/caffe/layers
cp $ROOT/knowledge_distillation_layer.cpp $CAFFE/src/caffe/layers
```
4. Modify `$CAFFE/src/caffe/proto/caffe.proto`<br>add `optional KnowledgeDistillationParameter` in `LayerParameter`
```proto
message LayerParameter {
  ...

  //next available layer-specific ID
  optional KnowledgeDistillationParameter knowledge_distillation_param = 147;
}
```
<br>add `message KnowledgeDistillationParameter`<br>
```proto
message KnowledgeDistillationParameter {
  optional float temperature = 1 [default = 1];
}
```
5. Build Caffe
<br>

## Usage
KnowledgeDistillation Layer has one specific parameter `temperature`.<br><br>The layer takes 2 or 3 input blobs:<br>
`bottom[0]`: the logits of the student<br>
`bottom[1]`: the logits of the teacher<br>
`bottom[2]`(*optional*): label inputs<br>
The logits are first divided by temperatrue T, then mapped to probability distributions over classes using the softmax function. The layer computes KL divergence instead of cross entropy. The gradients are multiplied by T^2, as suggested in the [paper](https://arxiv.org/abs/1503.02531).<br>
1. Common setting in `prototxt` (2 input blobs are given)
```
layer {
  name: "KD"
  type: "KnowledgeDistillation"
  bottom: "student_logits"
  bottom: "taecher_logits"
  top: "KL_div"
  include { phase: TRAIN }
  knowledge_distillation_param { temperature: 4 } #usually larger than 1
  loss_weight: 1
}
```
2. If you have ignore_label, 3 input blobs should be given
```
layer {
  name: "KD"
  type: "KnowledgeDistillation"
  bottom: "student_logits"
  bottom: "taecher_logits"
  bottom: "label"
  top: "KL_div"
  include { phase: TRAIN }
  knowledge_distillation_param { temperature: 4 }
  loss_param {ignore_label: 2}
  loss_weight: 1
}
