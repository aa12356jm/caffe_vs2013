#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

const int kMaxBlobAxes = 32;//表示Blob可以支持的最高维数

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
	 // 默认不带参数的构造函数,初始化count_=0,capacity_=0 
  Blob()
       : data_(), diff_(), count_(0), capacity_(0) {}

  // 带参数的显示构造函数，推荐使用带vector<int>参数的构造函数  
  // 这两个构造函数内部会均会调用Reshape(const vector<int>)函数  
  // 注：执行这两个构造函数后，并不会真正分配内存空间，只是用来设置当前blob的shape_、count_和capacity_大小
  /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
  explicit Blob(const int num, const int channels, const int height,
      const int width);
  explicit Blob(const vector<int>& shape);


  // Reshape系列函数通过输入参数用来设置或重新设置当前blob的shape_、count_和capacity_大小  
  // 推荐使用带vector<int>参数的Reshape函数  
  // 内部会调用SyncedMemory的构造函数，但不会真正分配内存空间  
  // 通过num/channes/height/width参数设置shape_、count_和capacity_大小 
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
  // 通过vector<int>参数设置shape_、count_和capacity_大小 
  void Reshape(const vector<int>& shape);
  // 通过类BlobShape参数设置shape_、count_和capacity_大小  
  // BlobShape是定义在caffe.proto中的一个message，其字段有dim
  void Reshape(const BlobShape& shape);
  // 通过外部的blob参数来设置shape_、count_和capacity_大小 
  void ReshapeLike(const Blob& other);

  // 以string类型获得当前blob的shape_和count_值
  inline string shape_string() const {
    ostringstream stream;
    for (int i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";
    }
    stream << "(" << count_ << ")";
    return stream.str();
  }
  // 获得当前Blob的所有维度值 
  inline const vector<int>& shape() const { return shape_; }
  /**
   * @brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   *
   * @param index the axis index, which may be negative as it will be
   *        "canonicalized" using CanonicalAxisIndex.
   *        Dies on out of range index.
   */
  // 获得当前Blob指定索引的维度值 
  inline int shape(int index) const {
    return shape_[CanonicalAxisIndex(index)];
  }

  // 获得当前Blob的维数 
  inline int num_axes() const { return shape_.size(); }
  // 获得当前Blob的元素个数
  inline int count() const { return count_; }

  /**
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
  // 根据指定的start axis和end axis(部分blob)计算blob元素个数 
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
  // 根据指定的start axis(部分blob)计算blob元素个数
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

  // Blob的index可以是负值,对参数axis_index进行判断,结果返回一个正的索引值  
  // 如果axis_index是负值，则要求axis_index>=-shape_.size(),则返回axis_index+shape_.size()  
  // 如果axis_index是正值，则要求axis_index<shape_.size(),则直接返回axis_index
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
  // 获得当前blob的num，推荐调用shape(0)函数
  inline int num() const { return LegacyShape(0); }
  /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
  // 获得当前blob的channels，推荐调用shape(1)函数
  inline int channels() const { return LegacyShape(1); }
  /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
  // 获得当前blob的height，推荐调用shape(2)函数
  inline int height() const { return LegacyShape(2); }
  /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
  // 获得当前blob的width，推荐调用shape(3)函数
  inline int width() const { return LegacyShape(3); }

  // 获得当前blob的某一维度值
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
  // 根据num、channels、height、width计算偏移量:((n*K+k)*H+h)*W+w 
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
  // 根据vector<int> index计算偏移量：((n*K+k)*H+h)*W+w
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

  // 从外部blob拷贝数据到当前的blob  
  // 若reshape参数为true，如果两边blob的reshape不相同，则会重新reshape  
  // 若copy_diff为false,则拷贝data_数据；若copy_diff为true，则拷贝diff_数据
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);

  // 根据给定的位置访问数据  
  // 根据指定的偏移量获得前向传播数据data_的一个元素的值
  inline Dtype data_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_data()[offset(n, c, h, w)];
  }
  // 根据指定的偏移量获得反向传播梯度diff_的一个元素的值
  inline Dtype diff_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_diff()[offset(n, c, h, w)];
  }
  // 根据指定的偏移量获得前向传播数据data_的一个元素的值
  inline Dtype data_at(const vector<int>& index) const {
    return cpu_data()[offset(index)];
  }
  // 根据指定的偏移量获得反向传播梯度diff_的一个元素的值
  inline Dtype diff_at(const vector<int>& index) const {
    return cpu_diff()[offset(index)];
  }
  // 获得前向传播数据data_的指针
  inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;
  }
  // 获得反向传播梯度diff_的指针
  inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;
  }
  // Blob的数据访问函数，包括CPU和GPU  
  // 带mutable_前缀的函数是可以对Blob数据进行改写的；其它不带的是只读的，不允许改写数据
  const Dtype* cpu_data() const;// 调用SyncedMemory::cpu_data()函数
  void set_cpu_data(Dtype* data);// 调用SyncedMemory::set_cpu_data(void*)函数
  const int* gpu_shape() const;
  const Dtype* gpu_data() const; //调用SyncedMemory::gpu_data()函数
  const Dtype* cpu_diff() const;// 调用SyncedMemory::cpu_data()函数
  const Dtype* gpu_diff() const;// 调用SyncedMemory::gpu_data()函数
  Dtype* mutable_cpu_data(); // 调用SyncedMemory::mutable_cpu_data()函数
  Dtype* mutable_gpu_data();// 调用SyncedMemory::mutable_gpu_data()函数
  Dtype* mutable_cpu_diff();// 调用SyncedMemory::mutable_cpu_diff()函数
  Dtype* mutable_gpu_diff();// 调用SyncedMemory::mutable_gpu_diff()函数
  // 它会被网络中存储参数的Blob调用，完成梯度下降过程中的参数更新  
  // 调用caffe_axpy函数重新计算data_(weight，bias 等减去对应的导数): data_ = -1 * diff_ + data_
  void Update();
  // Blob的数据持久化函数，通过Protobuf来做相应的序列化/反序列化操作  
  // BlobProto是定义在caffe.proto中的一个message，其字段有shape(BlobShape)、data、diff、num、channels、height、width  
  // 将BlobProto的shape/data/diff分别copy给当前blob的shape_/data_/diff_完成数据解析(反序列化)  
  // 若reshape参数为true，则会对当前的blob重新进行reshape
  void FromProto(const BlobProto& proto, bool reshape = true);
  // 将Blob的shape_/data_/diff_(如果write_diff为true)分别copy给BlobProto的shape/data/diff完成序列化 
  void ToProto(BlobProto* proto, bool write_diff = false) const;

  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  // 计算data_的L1范式：向量中各个元素绝对值之和
  Dtype asum_data() const;
  /// @brief Compute the sum of absolute values (L1 norm) of the diff.
  // 计算diff_的L1范式：向量中各个元素绝对值之和
  Dtype asum_diff() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the data.
  // 计算data_的L2范式平方：向量中各元素的平方和
  Dtype sumsq_data() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the diff.
  // 计算diff_的L2范式平方：向量中各元素的平方和
  Dtype sumsq_diff() const;

  /// @brief Scale the blob data by a constant factor.
  // 将data_数据乘以一个因子：X = alpha*X 
  void scale_data(Dtype scale_factor);
  /// @brief Scale the blob diff by a constant factor.
  // 将diff_数据乘以一个因子：X = alpha*X
  void scale_diff(Dtype scale_factor);

  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  // 将外部指定的blob的data_指针指向给当前blob的data_,以实现共享data_
  void ShareData(const Blob& other);
  /**
   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  // 将外部指定的blob的diff_指针指向给当前blob的diff_,以实现共享diff_
  void ShareDiff(const Blob& other);
  // 比较两个blob的shape是否相同  
  // BlobProto是定义在caffe.proto中的一个message，其字段有shape(BlobShape)、data、diff、num、channels、height、width
  bool ShapeEquals(const BlobProto& other);

 protected:
	 // Caffe中类的成员变量名都带有后缀"_"，这样就容易区分临时变量和类成员变量
  shared_ptr<SyncedMemory> data_;// 存储前向传播的数据
  shared_ptr<SyncedMemory> diff_;// 存储反向传播的导数、梯度、偏差
  shared_ptr<SyncedMemory> shape_data_;
  vector<int> shape_;// Blob的维度值，通过Reshape函数的shape参数获得相应值,若为4维，则依次为num、channels、height、width
  int count_;// 表示Blob中的元素个数，shape_所有元素的乘积
  int capacity_;// 表示当前Blob的元素个数(控制动态分配)，因为Blob可能会reshape

  // 禁止使用Blob类的拷贝和赋值操作
  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
