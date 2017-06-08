#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>

#include "caffe/layers/knowledge_distillation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void KnowledgeDistillationLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param;
  softmax_param.set_type("Softmax");
  softmax_param.mutable_softmax_param()->set_axis(this->layer_param_.softmax_param().axis());
  s_logit_.Reshape(bottom[0]->shape());
  s_softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  s_softmax_bottom_vec_.clear();
  s_softmax_bottom_vec_.push_back(&s_logit_);
  s_softmax_top_vec_.clear();
  s_softmax_top_vec_.push_back(&s_prob_);
  s_softmax_layer_->SetUp(s_softmax_bottom_vec_, s_softmax_top_vec_);
  t_logit_.Reshape(bottom[1]->shape());
  t_softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  t_softmax_bottom_vec_.clear();
  t_softmax_bottom_vec_.push_back(&t_logit_);
  t_softmax_top_vec_.clear();
  t_softmax_top_vec_.push_back(&t_prob_);
  t_softmax_layer_->SetUp(t_softmax_bottom_vec_, t_softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }

  T = this->layer_param_.knowledge_distillation_param().temperature();
}

template <typename Dtype>
void KnowledgeDistillationLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  s_logit_.Reshape(bottom[0]->shape());
  t_logit_.Reshape(bottom[1]->shape());
  s_softmax_layer_->Reshape(s_softmax_bottom_vec_, s_softmax_top_vec_);
  t_softmax_layer_->Reshape(t_softmax_bottom_vec_, t_softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  
  CHECK_EQ(outer_num_, bottom[1]->count(0, softmax_axis_))
      << "Outer number of soft labels must match outer number of predictions.";
  CHECK_EQ(inner_num_, bottom[1]->count(softmax_axis_ + 1))
      << "Inner number of soft labels must match inner number of predictions.";
  CHECK_EQ(bottom.size() == 3, has_ignore_label_)
      << "ignore_label is only valid when label inputs are given as bottom[2].";
  if (bottom.size() == 3 && has_ignore_label_) {
    CHECK_EQ(outer_num_ * inner_num_, bottom[2]->count())
        << "Number of labels must match number of predictions; "
        << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
        << "label count (number of labels) must be N*H*W, "
        << "with integer values in {0, 1, ..., C-1}.";
  }
}

template <typename Dtype>
Dtype KnowledgeDistillationLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void KnowledgeDistillationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Both logits are divided by the temperature T.
  caffe_copy<Dtype>(bottom[0]->count(), bottom[0]->cpu_data(), s_logit_.mutable_cpu_data());
  caffe_scal(bottom[0]->count(), Dtype(1)/T, s_logit_.mutable_cpu_data());
  caffe_copy<Dtype>(bottom[1]->count(), bottom[1]->cpu_data(), t_logit_.mutable_cpu_data());
  caffe_scal(bottom[0]->count(), Dtype(1)/T, t_logit_.mutable_cpu_data());
  // The forward pass computes the softmax prob values.
  s_softmax_layer_->Forward(s_softmax_bottom_vec_, s_softmax_top_vec_);
  t_softmax_layer_->Forward(t_softmax_bottom_vec_, t_softmax_top_vec_);
  const Dtype* prob_data = s_prob_.cpu_data();
  const Dtype* soft_label = t_prob_.cpu_data();
  int dim = s_prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  int pos;
  // Compute KL divergence.
  if (bottom.size() == 3 && has_ignore_label_) { // label inputs and ignore_label are given.
    const Dtype* label = bottom[2]->cpu_data();
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {     
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (label_value == ignore_label_) {
          continue;
        }   
       
        for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
          pos = i * dim + c * inner_num_ + j;
          loss -= soft_label[pos] * (log(std::max(prob_data[pos], Dtype(FLT_MIN)))-log(std::max(soft_label[pos], Dtype(FLT_MIN))));
        }
        ++count;
      }
    }
  } else { // label inputs or ignore_label are not given.
    count = outer_num_ * inner_num_;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {         
        for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
          pos = i * dim + c * inner_num_ + j;
          loss -= soft_label[pos] * (log(std::max(prob_data[pos], Dtype(FLT_MIN)))-log(std::max(soft_label[pos], Dtype(FLT_MIN))));
        }
      }
    }
  }

  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
}

template <typename Dtype>
void KnowledgeDistillationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1] | (bottom.size() == 3 && propagate_down[2])) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to soft label nor label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = s_prob_.cpu_data();
    caffe_copy(s_prob_.count(), prob_data, bottom_diff);
    const Dtype* soft_label = t_prob_.cpu_data();
    int dim = s_prob_.count() / outer_num_;
    int count = outer_num_ * inner_num_;
    // The gradients here are multiplied by T,
    // which is T^2 (as suggested in the paper) * 1/T (logits divided by T).
    caffe_cpu_axpby<Dtype>(outer_num_*dim, -T, soft_label, T, bottom_diff);
    // If label inputs are given, set the gradients to 0 w.r.t. ignore_label.
    if (bottom.size() == 3 && has_ignore_label_) {
      count = 0;
      const Dtype* label = bottom[2]->cpu_data();
      for (int i = 0; i < outer_num_; ++i) {
        for (int j = 0; j < inner_num_; ++j) {
          const int label_value = static_cast<int>(label[i * inner_num_ + j]);
          if (label_value == ignore_label_) {
            for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
              bottom_diff[i * dim + c * inner_num_ + j] = 0;
            }
          } else {
            ++count;
          }
        }
      }
    }

    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_scal(s_prob_.count(), loss_weight, bottom_diff);
  }
}


//#ifdef CPU_ONLY
//STUB_GPU(KnowledgeDistillationLayer);
//#endif

INSTANTIATE_CLASS(KnowledgeDistillationLayer);
REGISTER_LAYER_CLASS(KnowledgeDistillation);

}  // namespace caffe
