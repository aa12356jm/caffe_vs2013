#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

const int kMaxBlobAxes = 32;//��ʾBlob����֧�ֵ����ά��

namespace caffe {

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class Blob {
 public:
	 // Ĭ�ϲ��������Ĺ��캯��,��ʼ��count_=0,capacity_=0 
  Blob()
       : data_(), diff_(), count_(0), capacity_(0) {}

  // ����������ʾ���캯�����Ƽ�ʹ�ô�vector<int>�����Ĺ��캯��  
  // ���������캯���ڲ���������Reshape(const vector<int>)����  
  // ע��ִ�����������캯���󣬲��������������ڴ�ռ䣬ֻ���������õ�ǰblob��shape_��count_��capacity_��С
  /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
  explicit Blob(const int num, const int channels, const int height,
      const int width);
  explicit Blob(const vector<int>& shape);


  // Reshapeϵ�к���ͨ����������������û��������õ�ǰblob��shape_��count_��capacity_��С  
  // �Ƽ�ʹ�ô�vector<int>������Reshape����  
  // �ڲ������SyncedMemory�Ĺ��캯�������������������ڴ�ռ�  
  // ͨ��num/channes/height/width��������shape_��count_��capacity_��С 
  /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
  void Reshape(const int num, const int channels, const int height,
      const int width);
  /**
   * @brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   */
  // ͨ��vector<int>��������shape_��count_��capacity_��С 
  void Reshape(const vector<int>& shape);
  // ͨ����BlobShape��������shape_��count_��capacity_��С  
  // BlobShape�Ƕ�����caffe.proto�е�һ��message�����ֶ���dim
  void Reshape(const BlobShape& shape);
  // ͨ���ⲿ��blob����������shape_��count_��capacity_��С 
  void ReshapeLike(const Blob& other);

  // ��string���ͻ�õ�ǰblob��shape_��count_ֵ
  inline string shape_string() const {
    ostringstream stream;
    for (int i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";
    }
    stream << "(" << count_ << ")";
    return stream.str();
  }
  // ��õ�ǰBlob������ά��ֵ 
  inline const vector<int>& shape() const { return shape_; }
  /**
   * @brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   *
   * @param index the axis index, which may be negative as it will be
   *        "canonicalized" using CanonicalAxisIndex.
   *        Dies on out of range index.
   */
  // ��õ�ǰBlobָ��������ά��ֵ 
  inline int shape(int index) const {
    return shape_[CanonicalAxisIndex(index)];
  }

  // ��õ�ǰBlob��ά�� 
  inline int num_axes() const { return shape_.size(); }
  // ��õ�ǰBlob��Ԫ�ظ���
  inline int count() const { return count_; }

  /**
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
  // ����ָ����start axis��end axis(����blob)����blobԪ�ظ��� 
  inline int count(int start_axis, int end_axis) const {
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
      count *= shape(i);
    }
    return count;
  }
  /**
   * @brief Compute the volume of a slice spanning from a particular first
   *        axis to the final axis.
   *
   * @param start_axis The first axis to include in the slice.
   */
  // ����ָ����start axis(����blob)����blobԪ�ظ���
  inline int count(int start_axis) const {
    return count(start_axis, num_axes());
  }

  /**
   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */

  // Blob��index�����Ǹ�ֵ,�Բ���axis_index�����ж�,�������һ����������ֵ  
  // ���axis_index�Ǹ�ֵ����Ҫ��axis_index>=-shape_.size(),�򷵻�axis_index+shape_.size()  
  // ���axis_index����ֵ����Ҫ��axis_index<shape_.size(),��ֱ�ӷ���axis_index
  inline int CanonicalAxisIndex(int axis_index) const {
    CHECK_GE(axis_index, -num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    CHECK_LT(axis_index, num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes();
    }
    return axis_index;
  }

  /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
  // ��õ�ǰblob��num���Ƽ�����shape(0)����
  inline int num() const { return LegacyShape(0); }
  /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
  // ��õ�ǰblob��channels���Ƽ�����shape(1)����
  inline int channels() const { return LegacyShape(1); }
  /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
  // ��õ�ǰblob��height���Ƽ�����shape(2)����
  inline int height() const { return LegacyShape(2); }
  /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
  // ��õ�ǰblob��width���Ƽ�����shape(3)����
  inline int width() const { return LegacyShape(3); }

  // ��õ�ǰblob��ĳһά��ֵ
  inline int LegacyShape(int index) const {
    CHECK_LE(num_axes(), 4)
        << "Cannot use legacy accessors on Blobs with > 4 axes.";
    CHECK_LT(index, 4);
    CHECK_GE(index, -4);
    if (index >= num_axes() || index < -num_axes()) {
      // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
      // indexing) -- this special case simulates the one-padding used to fill
      // extraneous axes of legacy blobs.
      return 1;
    }
    return shape(index);
  }
  // ����num��channels��height��width����ƫ����:((n*K+k)*H+h)*W+w 
  inline int offset(const int n, const int c = 0, const int h = 0,
      const int w = 0) const {
    CHECK_GE(n, 0);
    CHECK_LE(n, num());
    CHECK_GE(channels(), 0);
    CHECK_LE(c, channels());
    CHECK_GE(height(), 0);
    CHECK_LE(h, height());
    CHECK_GE(width(), 0);
    CHECK_LE(w, width());
    return ((n * channels() + c) * height() + h) * width() + w;
  }
  // ����vector<int> index����ƫ������((n*K+k)*H+h)*W+w
  inline int offset(const vector<int>& indices) const {
    CHECK_LE(indices.size(), num_axes());
    int offset = 0;
    for (int i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        CHECK_GE(indices[i], 0);
        CHECK_LT(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }
  /**
   * @brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param copy_diff if false, copy the data; if true, copy the diff
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   */

  // ���ⲿblob�������ݵ���ǰ��blob  
  // ��reshape����Ϊtrue���������blob��reshape����ͬ���������reshape  
  // ��copy_diffΪfalse,�򿽱�data_���ݣ���copy_diffΪtrue���򿽱�diff_����
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);

  // ���ݸ�����λ�÷�������  
  // ����ָ����ƫ�������ǰ�򴫲�����data_��һ��Ԫ�ص�ֵ
  inline Dtype data_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_data()[offset(n, c, h, w)];
  }
  // ����ָ����ƫ������÷��򴫲��ݶ�diff_��һ��Ԫ�ص�ֵ
  inline Dtype diff_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_diff()[offset(n, c, h, w)];
  }
  // ����ָ����ƫ�������ǰ�򴫲�����data_��һ��Ԫ�ص�ֵ
  inline Dtype data_at(const vector<int>& index) const {
    return cpu_data()[offset(index)];
  }
  // ����ָ����ƫ������÷��򴫲��ݶ�diff_��һ��Ԫ�ص�ֵ
  inline Dtype diff_at(const vector<int>& index) const {
    return cpu_diff()[offset(index)];
  }
  // ���ǰ�򴫲�����data_��ָ��
  inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;
  }
  // ��÷��򴫲��ݶ�diff_��ָ��
  inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;
  }
  // Blob�����ݷ��ʺ���������CPU��GPU  
  // ��mutable_ǰ׺�ĺ����ǿ��Զ�Blob���ݽ��и�д�ģ�������������ֻ���ģ��������д����
  const Dtype* cpu_data() const;// ����SyncedMemory::cpu_data()����
  void set_cpu_data(Dtype* data);// ����SyncedMemory::set_cpu_data(void*)����
  const int* gpu_shape() const;
  const Dtype* gpu_data() const; //����SyncedMemory::gpu_data()����
  const Dtype* cpu_diff() const;// ����SyncedMemory::cpu_data()����
  const Dtype* gpu_diff() const;// ����SyncedMemory::gpu_data()����
  Dtype* mutable_cpu_data(); // ����SyncedMemory::mutable_cpu_data()����
  Dtype* mutable_gpu_data();// ����SyncedMemory::mutable_gpu_data()����
  Dtype* mutable_cpu_diff();// ����SyncedMemory::mutable_cpu_diff()����
  Dtype* mutable_gpu_diff();// ����SyncedMemory::mutable_gpu_diff()����
  // ���ᱻ�����д洢������Blob���ã�����ݶ��½������еĲ�������  
  // ����caffe_axpy�������¼���data_(weight��bias �ȼ�ȥ��Ӧ�ĵ���): data_ = -1 * diff_ + data_
  void Update();
  // Blob�����ݳ־û�������ͨ��Protobuf������Ӧ�����л�/�����л�����  
  // BlobProto�Ƕ�����caffe.proto�е�һ��message�����ֶ���shape(BlobShape)��data��diff��num��channels��height��width  
  // ��BlobProto��shape/data/diff�ֱ�copy����ǰblob��shape_/data_/diff_������ݽ���(�����л�)  
  // ��reshape����Ϊtrue�����Ե�ǰ��blob���½���reshape
  void FromProto(const BlobProto& proto, bool reshape = true);
  // ��Blob��shape_/data_/diff_(���write_diffΪtrue)�ֱ�copy��BlobProto��shape/data/diff������л� 
  void ToProto(BlobProto* proto, bool write_diff = false) const;

  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  // ����data_��L1��ʽ�������и���Ԫ�ؾ���ֵ֮��
  Dtype asum_data() const;
  /// @brief Compute the sum of absolute values (L1 norm) of the diff.
  // ����diff_��L1��ʽ�������и���Ԫ�ؾ���ֵ֮��
  Dtype asum_diff() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the data.
  // ����data_��L2��ʽƽ���������и�Ԫ�ص�ƽ����
  Dtype sumsq_data() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the diff.
  // ����diff_��L2��ʽƽ���������и�Ԫ�ص�ƽ����
  Dtype sumsq_diff() const;

  /// @brief Scale the blob data by a constant factor.
  // ��data_���ݳ���һ�����ӣ�X = alpha*X 
  void scale_data(Dtype scale_factor);
  /// @brief Scale the blob diff by a constant factor.
  // ��diff_���ݳ���һ�����ӣ�X = alpha*X
  void scale_diff(Dtype scale_factor);

  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  // ���ⲿָ����blob��data_ָ��ָ�����ǰblob��data_,��ʵ�ֹ���data_
  void ShareData(const Blob& other);
  /**
   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  // ���ⲿָ����blob��diff_ָ��ָ�����ǰblob��diff_,��ʵ�ֹ���diff_
  void ShareDiff(const Blob& other);
  // �Ƚ�����blob��shape�Ƿ���ͬ  
  // BlobProto�Ƕ�����caffe.proto�е�һ��message�����ֶ���shape(BlobShape)��data��diff��num��channels��height��width
  bool ShapeEquals(const BlobProto& other);

 protected:
	 // Caffe����ĳ�Ա�����������к�׺"_"������������������ʱ���������Ա����
  shared_ptr<SyncedMemory> data_;// �洢ǰ�򴫲�������
  shared_ptr<SyncedMemory> diff_;// �洢���򴫲��ĵ������ݶȡ�ƫ��
  shared_ptr<SyncedMemory> shape_data_;
  vector<int> shape_;// Blob��ά��ֵ��ͨ��Reshape������shape���������Ӧֵ,��Ϊ4ά��������Ϊnum��channels��height��width
  int count_;// ��ʾBlob�е�Ԫ�ظ�����shape_����Ԫ�صĳ˻�
  int capacity_;// ��ʾ��ǰBlob��Ԫ�ظ���(���ƶ�̬����)����ΪBlob���ܻ�reshape

  // ��ֹʹ��Blob��Ŀ����͸�ֵ����
  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
