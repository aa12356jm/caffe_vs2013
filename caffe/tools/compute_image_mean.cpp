#include <stdint.h>//�����˼�����չ���������ͺͺ�
#include <algorithm>//�����������ݡ�������������򡢷�ת�������ݡ������������ݵȲ���
#include <string>
#include <utility>//utilityͷ�ļ�������һ��pair����,pair�������ڴ洢һ������;��Ҳ�ṩһЩ���õı������������ࡢ��ģ�塣��С��ֵ��ֵ������min��max��swap
#include <vector>//�����Զ���չ����������

#include "boost/scoped_ptr.hpp"//����ָ��ͷ�ļ�
#include "gflags/gflags.h"//gflags��google��һ����Դ�Ĵ��������в����Ŀ�
#include "glog/logging.h"//Google Glog ��һ��C++���Ե�Ӧ�ü���־��¼��ܣ��ṩ�� C++ �����������͸������ֺ�

#include "caffe/proto/caffe.pb.h"//���ṹ������caffe.proto��ʹ��Protobuf����������ΪC++�࣬ͷ�ļ�Ϊcaffe.pb.h�ļ�
#include "caffe/util/db.hpp"//�����װ�õ�lmdb��������
#include "caffe/util/io.hpp"//����opencv�е�ͼ���������
/*
��ֵ����������Ԥ�����г����Ĵ���ʽ������֮ǰ��ѧϰufldl�̳�PCA��һ��ʱ��
����ͼ����������֣���һ�ֳ��õķ�ʽ����dimension_mean���������������������������ݵ�ά�ȣ�ÿ��ά���ڽ������������Ҳ�ǳ�����������
�ڶ��ֽ���per_image_mean��ufldl�̳���˵����natural images��ѵ������ʱ����ÿ�����أ�����ֻÿ��dimension������һ�������ľ�ֵ�ͷ�����make little sense�ģ�
������Ϊͼ�������ͳ�Ʋ����ԣ�����ͼ���һ���ֵ�ͳ�����Ժ���һ������ͬ��
��������飬�����ѵ������㷨�ڷ�natural images����mnist�������ڰױ������ڵ������������壩���������͵Ĺ�����ֵ�ÿ��ǵġ�
���ǵ���natural images��ѵ��ʱ��per_image_mean��һ�������Ĭ��ѡ��
*/

using namespace caffe;  // NOLINT(build/namespaces)

using std::max;
using std::pair;
using boost::scoped_ptr;

//ͨ��gflags�궨��һЩ����Ĳ�������
//����Ҫ�������в���ʹ��gflags�ĺ궨��,�������������п�������ʹ�ã�--backend=leveldb ����ʾ����ͼ���ʽΪleveldb��ʽ��
//compute_image_mean.exe --backend=leveldb
DEFINE_string(backend, "lmdb",
        "The backend {leveldb, lmdb} containing the images");//string���ͣ�����������������ͣ�Ĭ��lmdb����������ʹ�÷�ʽ��--backend=lmdb/leveldb

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);//�������ǳ�����,ʹ��glog֮ǰ�����ȳ�ʼ���⣬Ҫ������־�ļ�ֻ���ڿ�ʼlog֮ǰ����һ��

#ifdef USE_OPENCV
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Compute the mean_image of a set of images given by"
        " a leveldb/lmdb\n"
        "Usage:\n"
        "    compute_image_mean [FLAGS] INPUT_DB [OUTPUT_FILE]\n");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  //�����и���ֻ��Ϊ2����3
  if (argc < 2 || argc > 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_image_mean");
    return 1;
  }

  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));//��������ָ�룬��ʹ��GetDB������ʼ��Ϊ��������ָ����ʽ��--backend=leveldb/lmdb
  db->Open(argv[1], db::READ);//��ֻ���ķ�ʽ��argv[1]�ļ����µ�lmdb/leveldb���ݿ��ļ�
  scoped_ptr<db::Cursor> cursor(db->NewCursor());//lmdb/leveldb���ݿ�ġ���ꡱ�ļ���һ����걣��һ�������ݿ��Ŀ¼�����ݿ��ļ���·��

  BlobProto sum_blob;//����һ��BlobProto����
  int count = 0;
  // load first datum
  Datum datum;
  datum.ParseFromString(cursor->value());//�����������ַ����е�protocol buffer 

  //����datum������
  if (DecodeDatumNative(&datum)) {
    LOG(INFO) << "Decoding Datum";
  }

  //����sum_blob�Ĳ���
  //ÿ��blob����Ϊһ��4ά�����飬�ֱ�Ϊimage_num*channels*height*width
  sum_blob.set_num(1);//����ͼƬ�ĸ���
  sum_blob.set_channels(datum.channels());
  sum_blob.set_height(datum.height());
  sum_blob.set_width(datum.width());
  const int data_size = datum.channels() * datum.height() * datum.width();
  int size_in_datum = std::max<int>(datum.data().size(),
                                    datum.float_data_size());
  //��ʼ��sum_blob�����ݣ����ó�ֵΪfloat�͵�0.0
  for (int i = 0; i < size_in_datum; ++i) {
    sum_blob.add_data(0.);
  }
  LOG(INFO) << "Starting Iteration";
  
  //�����ݿ����ļ���ȡ��sum_blob��
  while (cursor->valid()) {//���cursor����Ч��
    Datum datum;
    datum.ParseFromString(cursor->value());////����cuisor.value���ص��ַ���ֵ����datum
    DecodeDatumNative(&datum);//��datum���ַ������͵�ֵ��ת��Ϊdatum��ԭʼ����

    const std::string& data = datum.data();//����data������datum.data 
    size_in_datum = std::max<int>(datum.data().size(),
        datum.float_data_size());
    CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
        size_in_datum;
    if (data.size() != 0) {
      CHECK_EQ(data.size(), size_in_datum);//�ж��Ƿ����
	  //��Ӧλ�õ�����ֵ��ӣ�uin8_t������ӣ�����ӵĽ������sum_blob��
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
      }
    } else {
      CHECK_EQ(datum.float_data_size(), size_in_datum);
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) +
            static_cast<float>(datum.float_data(i)));//��Ӧλ�õ�����ֵ��ӣ�float������ӣ�
      }
    }
    ++count;
    if (count % 10000 == 0) {
      LOG(INFO) << "Processed " << count << " files.";
    }
    cursor->Next();//������ƣ�ָ�룩��ָ����һ���洢��lmdb�е�����
  }

  if (count % 10000 != 0) {
    LOG(INFO) << "Processed " << count << " files.";
  }
  //�����ֵ����������ֵ����������
  for (int i = 0; i < sum_blob.data_size(); ++i) {
    sum_blob.set_data(i, sum_blob.data(i) / count);
  }
  // Write to disk
  //������õ����ļ�д�뵽Ӳ��ָ����·��
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
