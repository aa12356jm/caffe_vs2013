#include <stdint.h>//定义了几种扩展的整数类型和宏
#include <algorithm>//输出数组的内容、对数组进行排序、反转数组内容、复制数组内容等操作
#include <string>
#include <utility>//utility头文件定义了一个pair类型,pair类型用于存储一对数据;它也提供一些常用的便利函数、或类、或模板。大小求值、值交换：min、max和swap
#include <vector>//可以自动扩展容量的数组

#include "boost/scoped_ptr.hpp"//智能指针头文件
#include "gflags/gflags.h"//gflags是google的一个开源的处理命令行参数的库
#include "glog/logging.h"//Google Glog 是一个C++语言的应用级日志记录框架，提供了 C++ 风格的流操作和各种助手宏

#include "caffe/proto/caffe.pb.h"//将结构化数据caffe.proto，使用Protobuf编译器编译为C++类，头文件为caffe.pb.h文件
#include "caffe/util/db.hpp"//引入包装好的lmdb操作函数
#include "caffe/util/io.hpp"//引入opencv中的图像操作函数
/*
均值削减是数据预处理中常见的处理方式，按照之前在学习ufldl教程PCA的一章时，
对于图像介绍了两种：第一种常用的方式叫做dimension_mean（个人命名），是依据输入数据的维度，每个维度内进行削减，这个也是常见的做法；
第二种叫做per_image_mean，ufldl教程上说，在natural images上训练网络时；给每个像素（这里只每个dimension）计算一个独立的均值和方差是make little sense的；
这是因为图像本身具有统计不变性，即在图像的一部分的统计特性和另一部分相同。
作者最后建议，如果你训练你的算法在非natural images（如mnist，或者在白背景存在单个独立的物体），其他类型的规则化是值得考虑的。
但是当在natural images上训练时，per_image_mean是一个合理的默认选择。
*/

using namespace caffe;  // NOLINT(build/namespaces)

using std::max;
using std::pair;
using boost::scoped_ptr;

//通过gflags宏定义一些程序的参数变量
//将需要的命令行参数使用gflags的宏定义,比如在命令行中可以这样使用，--backend=leveldb （表示输入图像格式为leveldb格式）
//compute_image_mean.exe --backend=leveldb
DEFINE_string(backend, "lmdb",
        "The backend {leveldb, lmdb} containing the images");//string类型，声明输入的数据类型，默认lmdb，命令行中使用方式：--backend=lmdb/leveldb

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);//括号内是程序名,使用glog之前必须先初始化库，要生成日志文件只需在开始log之前调用一次

#ifdef USE_OPENCV
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Compute the mean_image of a set of images given by"
        " a leveldb/lmdb\n"
        "Usage:\n"
        "    compute_image_mean [FLAGS] INPUT_DB [OUTPUT_FILE]\n");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  //命令行个数只能为2或者3
  if (argc < 2 || argc > 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_image_mean");
    return 1;
  }

  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));//创建智能指针，并使用GetDB函数初始化为命令行中指定格式，--backend=leveldb/lmdb
  db->Open(argv[1], db::READ);//以只读的方式打开argv[1]文件夹下的lmdb/leveldb数据库文件
  scoped_ptr<db::Cursor> cursor(db->NewCursor());//lmdb/leveldb数据库的“光标”文件，一个光标保存一个从数据库根目录到数据库文件的路径

  BlobProto sum_blob;//定义一个BlobProto对象
  int count = 0;
  // load first datum
  Datum datum;
  datum.ParseFromString(cursor->value());//读取数据库文件中的第一个键值对

  //解码datum中数据
  if (DecodeDatumNative(&datum)) {
    LOG(INFO) << "Decoding Datum";
  }

  //设置sum_blob的参数
  //每个blob对象，为一个4维的数组，分别为image_num*channels*height*width
  sum_blob.set_num(1);//设置图片的个数
  sum_blob.set_channels(datum.channels());
  sum_blob.set_height(datum.height());
  sum_blob.set_width(datum.width());
  const int data_size = datum.channels() * datum.height() * datum.width();
  int size_in_datum = std::max<int>(datum.data().size(),
                                    datum.float_data_size());
  //初始化sum_blob中数据，设置初值为float型的0.0
  for (int i = 0; i < size_in_datum; ++i) {
    sum_blob.add_data(0.);
  }
  LOG(INFO) << "Starting Iteration";
  
  //将数据库中文件读取到sum_blob中
  while (cursor->valid()) {//如果cursor是有效的
    Datum datum;
    datum.ParseFromString(cursor->value());////解析cuisor.value返回的字符串值，到datum
    DecodeDatumNative(&datum);//把datum中字符串类型的值，转换为datum的原始类型

    const std::string& data = datum.data();//利用data来引用datum.data 
    size_in_datum = std::max<int>(datum.data().size(),
        datum.float_data_size());
    CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
        size_in_datum;
    if (data.size() != 0) {
      CHECK_EQ(data.size(), size_in_datum);//判断是否相等
	  //对应位置的像素值相加（uin8_t类型相加），相加的结果放在sum_blob中
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
      }
    } else {
      CHECK_EQ(datum.float_data_size(), size_in_datum);
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) +
            static_cast<float>(datum.float_data(i)));//对应位置的像素值相加（float类型相加）
      }
    }
    ++count;
    if (count % 10000 == 0) {
      LOG(INFO) << "Processed " << count << " files.";
    }
    cursor->Next();//光标下移（指针），指向下一个存储在lmdb中的数据
  }

  if (count % 10000 != 0) {
    LOG(INFO) << "Processed " << count << " files.";
  }
  //计算均值，所有像素值除以总数量
  for (int i = 0; i < sum_blob.data_size(); ++i) {
    sum_blob.set_data(i, sum_blob.data(i) / count);
  }
  // Write to disk
  //将计算得到的文件写入到硬盘指定的路径
  if (argc == 3) {
    LOG(INFO) << "Write to " << argv[2];
    WriteProtoToBinaryFile(sum_blob, argv[2]);
  }
  const int channels = sum_blob.channels();
  const int dim = sum_blob.height() * sum_blob.width();
  std::vector<float> mean_values(channels, 0.0);
  LOG(INFO) << "Number of channels: " << channels;
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < dim; ++i) {
      mean_values[c] += sum_blob.data(dim * c + i);
    }
    LOG(INFO) << "mean_value channel [" << c << "]:" << mean_values[c] / dim;
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
