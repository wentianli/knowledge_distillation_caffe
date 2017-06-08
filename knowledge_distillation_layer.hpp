#ifndef CAFFE_KNOWLEDGE_DISTILLATION_LAYER_HPP_
#define CAFFE_KNOWLEDGE_DISTILLATION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

/**
 * @brief Computes KL divergence of two probability distributions, 
 *        using the logits of the student and the teacher.
 *
 * @param bottom input Blob vector (length 2 or 3)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the student's predictions @f$ x @f$, a Blob with values in
 *      @f$ [-\infty, +\infty] @f$ indicating the predicted score for each of
 *      the @f$ K = CHW @f$ classes. This layer maps these scores to a
 *      probability distribution over classes using the softmax function
 *      @f$ \hat{p}_{nk} = \exp(x_{nk}/T) /
 *      \left[\sum_{k'} \exp(x_{nk'}/T)\right] @f$ (see SoftmaxLayer)
 *      with T indicating the temperature in knowledge distillation.
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the teacher's predictions @f$ x @f$, similar to that of the student.
 *   -# @f$ (N \times 1 \times 1 \times 1) @f$ (optional)
 *      the labels @f$ l @f$, an integer-valued Blob with values
 *      @f$ l_n \in [0, 1, 2, ..., K - 1] @f$
 *      indicating the correct class label among the @f$ K @f$ classes
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed KL divergence: @f$ E =
 *        \frac{-1}{N} \sum\limits_{n=1}^N \sum\limits_{l=0}^{K-1} \hat{q}_{n,l} \log(\frac{\hat{p}_{n,l}}{\hat{q}_{n,l}})
 *      @f$, for softmax output class probabilites @f$ \hat{p} @f$ by the student
 *      and @f$ \hat{q} @f$ by the teacher
 *
 * [1] Hinton, G. Vinyals, O. and Dean, J. Distilling knowledge in a neural network. 2015.
 *
 * TODO: GPU implementation
 */
template <typename Dtype>
class KnowledgeDistillationLayer : public LossLayer<Dtype> {
 public:
   /**
    * @param param provides LossParameter loss_param, with options:
    *  - temperature (optional, default 1)
    *    Both logits are divided by the temperature T.
    *    The gradients are multiplied by T^2.
    *  - ignore_label (optional)
    *    Specify a label value that should be ignored when computing the loss.
    *    Only valid when label inputs are given as bottom[2].
    *  - normalize (optional, default true)
    *    If true, the loss is normalized by the number of (nonignored) labels
    *    present; otherwise the loss is simply summed over spatial locations.
    */
  explicit KnowledgeDistillationLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "KnowledgeDistillation"; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      //const vector<Blob<Dtype>*>& top);
  /**
   * @brief Computes the softmax loss error gradient w.r.t. the predictions.
   *
   * Gradients are not computed with respect to the teacher's inputs (bottom[1])
   * nor label inputs (bottom[2], optional), crashing
   * if propagate_down[1] or propagate_down[2] is set.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      //const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// Read the normalization mode parameter and compute the normalizer based
  /// on the blob size.  If normalization_mode is VALID, the count of valid
  /// outputs will be read from valid_count, unless it is -1 in which case
  /// all outputs are assumed to be valid.
  virtual Dtype get_normalizer(
      LossParameter_NormalizationMode normalization_mode, int valid_count);

  /// The internal SoftmaxLayer(s) used to map predictions to a distribution. s for student, t for teacher.
  shared_ptr<Layer<Dtype> > s_softmax_layer_;
  shared_ptr<Layer<Dtype> > t_softmax_layer_;
  /// prob stores the input logit.
  Blob<Dtype> s_logit_;
  Blob<Dtype> t_logit_;
  /// prob stores the output probability predictions from the SoftmaxLayer.
  Blob<Dtype> s_prob_;
  Blob<Dtype> t_prob_;
  /// bottom vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> s_softmax_bottom_vec_;
  vector<Blob<Dtype>*> t_softmax_bottom_vec_;
  /// top vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> s_softmax_top_vec_;
  vector<Blob<Dtype>*> t_softmax_top_vec_;
  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  /// How to normalize the output loss.
  LossParameter_NormalizationMode normalization_;

  int softmax_axis_, outer_num_, inner_num_;
  /// temperature
  Dtype T;
};

}  // namespace caffe

#endif  // CAFFE_KNOWLEDGE_DISTILLATION_LAYER_HPP_
