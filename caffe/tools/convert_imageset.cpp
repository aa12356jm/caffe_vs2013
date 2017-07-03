// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>//�����������ݡ�����������������򡢷�ת�������ݡ������������ݵȲ���
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>  //utilityͷ�ļ�������һ��pair����,pair�������ڴ洢һ������
#include <vector>  //��̬���ݵ����顰vector������̬�������Ԫ��

#include "boost/scoped_ptr.hpp" //����ָ��ͷ�ļ�
#include "gflags/gflags.h"//gflags��google��һ����Դ�Ĵ��������в����Ŀ�
#include "glog/logging.h"//Google Glog ��һ��C++���Ե�Ӧ�ü���־��¼��ܣ��ṩ�� C++ �����������͸������ֺ�

#include "caffe/proto/caffe.pb.h"//���ṹ������caffe.proto��ʹ��Protobuf����������ΪC++�࣬ͷ�ļ�Ϊcaffe.pb.h�ļ�
#include "caffe/util/db.hpp"  //�����װ�õ�lmdb��������
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"  //����opencv�е�ͼ���������
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)  ����ȫ��caffe�����ռ�
using std::pair;	//pair�������ݶԣ����ڴ洢�ɶԵĶ�������洢�ļ����Ͷ�Ӧ��ǩ
using boost::scoped_ptr;

//ͨ��gflags�궨��һЩ����Ĳ�������
//����Ҫ�������в���ʹ��gflags�ĺ궨��,�������������п�������ʹ�ã�--gray=true (��ʾʹ�ûҶ�ͼ��Ĭ��Ϊfalse)  --backend=leveldb ����ʾת��Ϊleveldb��ʽ��
//convert_imageset.exe --backend=leveldb --resize_width=64 --resize_height=64

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones"); //bool���ͣ��Ƿ�Ϊ�Ҷ�ͼƬ,��������ʹ�÷�ʽ��--gray=true/false

DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");  //bool���ͣ�����ϴ�Ʊ������Ƿ�����������ݼ���˳��,��������ʹ�÷�ʽ��--shuffle=true/false

DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");  ////string���ͣ�Ҫת�����������ͣ�Ĭ��lmdb����������ʹ�÷�ʽ��--backend=lmdb/leveldb

DEFINE_int32(resize_width, 0, "Width images are resized to");  //����resize�ĳߴ磬Ĭ��Ϊ0����ת���ߴ磬�Ⱥź�������֣���ʾת����Ĵ�С����������ʹ�÷�ʽ��--resize_width=64/...

DEFINE_int32(resize_height, 0, "Height images are resized to");//ͬ��

DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");//bool���ͣ��Ƿ����С�����������е�datum�е�������һ���Ĵ�С����������ʹ�÷�ʽ��--check_size=true/false

DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");//bool���ͣ��Ƿ���룬��������ʹ�÷�ʽ��--encoded=true/false

DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");//string���ͣ�Ҫת�������ݸ�ʽ���Ƿ�ͼ��ת��Ϊ������ʽ����������ʹ�÷�ʽ��--encode_type=png/jpg/.....

int main(int argc, char** argv) {
#ifdef USE_OPENCV
	::google::InitGoogleLogging(argv[0]); //�������ǳ�����,ʹ��glog֮ǰ�����ȳ�ʼ���⣬Ҫ������־�ļ�ֻ���ڿ�ʼlog֮ǰ����һ��
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  //ʹ��˵��
  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  
  //����������,�����Ϊtrue����ú���������ɺ�argv��ֻ����argv[0]��argc�ᱻ����Ϊ1��
  //���Ϊfalse����argv��argc�ᱻ����������ע�⺯�������argv�е�˳��
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  
  //arg[1] ѵ������ŵĵ�ַ��arg[2] train.txt��������ѵ����������ͼƬ���ļ����ƣ���arg[3] Ҫ������ļ�����xxlmdb
  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;		//ͨ��gflags�Ѻ궨�������ֵ����ֵ����ֵ������ͨ��ʹ��FLAGS_xxx�����Է��ʺ궨��ı�����FLAGS_gray����ָ�������в����е�gray
  const bool check_size = FLAGS_check_size;  //���ͼ���size���������е�check_size
  const bool encoded = FLAGS_encoded;         //�Ƿ���루ת����ͼ���ʽ
  const string encode_type = FLAGS_encode_type;  //Ҫ�����ͼ���ʽ���������е�encode_type
 
  std::ifstream infile(argv[2]);  //����ָ��train.txt�ļ����ļ�������

  //lines ��������������������ÿ��Ԫ��Ϊһ��pair�ԣ�pair����������Ա������һ��Ϊstring���ͣ�һ��Ϊint���ͣ�����string�������ڴ洢�ļ�����int���ͣ����ڴ�����Ӧ����id
  //��val.txt�е�һ��Ϊ��ILSVRC2012_val_00000001.JPEG 65��,string = ILSVRC2012_val_00000001.JPEG   int = 65
  std::vector<std::pair<std::string, int> > lines; 

  std::string line;
  size_t pos;
  int label;
  //����һ��while����ǰ�train.txt�ļ��д�ŵ������ļ����ͱ�ǩ������ŵ�vector���ͱ���lines�У�lines�д��ͼƬ�����ֺͶ�Ӧ�ı�ǩ�����洢������ͼƬ����
  while (std::getline(infile, line)) {//ÿ��ȡ���ļ��е�һ�У�Ȼ������ļ����Ͷ�Ӧ��label
    pos = line.find_last_of(' ');  //������һ���еĿո�λ��
    label = atoi(line.substr(pos + 1).c_str());//ȡ���ո�����ֵ�����ļ���Ӧ��label
	//line.substr(0, pos)��˼��ȡ������ַ����еĵ�0��pos���ַ������ݣ���Ϊ�ļ���
    lines.push_back(std::make_pair(line.substr(0, pos), label));
  }

  //�����в����е��Ƿ�ʹ�ò���--shuffle�����Ϊtrue����...
  if (FLAGS_shuffle) {
    // randomly shuffle data �ж��Ƿ����ϴ�Ʋ���
    LOG(INFO) << "Shuffling data";//GLOG ���ĸ����󼶱�,INFO,WARNING,ERROR,FATAL
	//ϴ�ƺ�����ʹ�����������g��Ԫ��[first, last)�����ڲ�Ԫ�ؽ���������� 
    shuffle(lines.begin(), lines.end());//vector.begin()�ش�һ��Iterator����������ָ�� vector ��һ��Ԫ��
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";  //��ӡ�����ܹ���ȡ�����ٸ�ͼ���ļ���train.txt���ж����У�

  //�����в����� �Ƿ�ָ��Ҫת��ͼ���ʽ����ת������ͼ���ʽ��
  //--encoded=true/false    --encode_type=jpg/png....
  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  //���������Ƿ�ָ������Ҫ��ͼ��ת��Ϊָ����С  --resize_height=....  --resize_width=... 
  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  //����ָ��Ĵ�����ʽ���Ʒ��͵ĸ�ʽ������ͨ��db.cpp�ڶ�����������������ռ���db�ġ���Ա������GetDB��������ʼ��db����
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));//��������ָ�룬��ʹ��GetDB������ʼ��Ϊ��������ָ����ʽ��--backend=leveldb/lmdb
  db->Open(argv[3], db::NEW);//��argv[3]���ļ����´������ݿ��ļ�������
  scoped_ptr<db::Transaction> txn(db->NewTransaction());//����lmdb�ļ��Ĳ������������ʹ�������������ݿ�db���������������ݷ������ݿ�

  // Storing to db
  std::string root_folder(argv[1]);//��Դ�����ļ��ĵ�ַ���Ƹ�root_folder��root_folderָ��ͼ������Ŀ¼
  Datum datum;//�������ݡ�ת��������
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;
    std::string enc = encode_type;
	//encΪ�մ�����enc.size()=false;����Ϊtrue
	//�����������ָ��Ҫת��ͼ���������ͣ�--encode_type=png/jpg/...   --encoded=true,һ����2������ͬʱ���֣���ִ��ת��
    if (encoded && !enc.size()) {
      // Guess the encoding type from the file name
      string fn = lines[line_id].first;//��ͼ����ļ�����ֵ��fn��filename��
      size_t p = fn.rfind('.');//rfind�����ķ���ֵ��һ�����ε�����ֵ��Ҳ����Ҫ���ҵ��ַ����ַ����е�λ�ã���û���ҵ�������string::npos
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);//�ҵ��ˣ��ͽ�ȡ�ļ�����.��������ַ������Ի��ͼ���ʽ�ַ������ļ�����׺���õ���ͼ���ʽΪjpg/png/....
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);//��enc�ַ���ת����Сд
    }
	//��Դ����λ�ö�ȡÿ��ͼƬ������
	//ReadImageToDatum����Ϊio.cpp�ļ��ж���ĺ�����io.cpp��Ҫʵ����3���ֹ��ܣ�
	/*
	1����text�ļ����߶������ļ��ж�proto�ļ���
	2������opencv��Mat���󣬰�ͼ�����ݶ���Mat�����У�
	3����Mat�����е�ֵ���뵽datum��
	*/
    status = ReadImageToDatum(root_folder + lines[line_id].first,
        lines[line_id].second, resize_height, resize_width, is_color,
        enc, &datum);//��ͼ�����ݶ�ȡ��datum��
    if (status == false) continue;//status=false,˵������ͼƬ��ȡ���󣻡�������������һ��
    
	if (check_size) {//���ͼƬ�ߴ� 
      if (!data_size_initialized) {//��data_size_initializedû�г�ʼ��
        data_size = datum.channels() * datum.height() * datum.width();
        data_size_initialized = true;
      } else {
        const std::string& data = datum.data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
      }
    }
    // sequential ���л�������key_str��ʽ��Ϊ: ��ǩ_ͼ���ļ���
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;

    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));//datum���ݣ����л����ַ�����
    txn->Put(key_str, out); //�Ѽ�ֵ�Է��뵽���ݿ�,ÿ����¼���ǣ���ǩ_�ļ���_ͼ����������

	//ÿ��1000������һ��
    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();//���浽lmdb���͵��ļ�
      txn.reset(db->NewTransaction());//���³�ʼ���������
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  //������һ��batch����1000,д�뵽���ݿ�
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
