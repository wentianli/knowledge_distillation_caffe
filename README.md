KnowledgeDistillation Layer (Caffe implementation)<br>
=
This is the CPU implementation of knowledge distillation in Caffe. This code is heavily based on [softmax_loss_layer.hpp](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/softmax_loss_layer.hpp) and [softmax_loss_layer.cpp](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/softmax_loss_layer.cpp).<br>
Installation
-
1. Install [Caffe](https://github.com/BVLC/caffe/) in your directory `CAFFE`<br>
2. Clone this repository in your directory `ROOT`<br>
```bash
cd $ROOT
git clone
```
3. Move files to your Caffe folder<br>
```bash
cp $ROOT/konwledge_distillaiton_layer.hpp $CAFFE/include/caffe/layers
cp $ROOT/konwledge_distillaiton_layer.cpp $CAFFE/src/caffe/layers
```
4. Modify `$CAFFE/src/caffe/proto/caffe.proto`<br>
add KnowledgeDistillationParameter in LayerParameter
```proto
message LayerParameter {
  optional string name = 1; // the layer name
  ...
  optional KnowledgeDistillationParameter knowledge_distillation_param = 147;//next available layer-specific ID
}
```
add message
```proto
message KnowledgeDistillationParameter {
  optional float temperature = 1 [default = 1];
}
```
5. Build Caffe

