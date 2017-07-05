// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>//输出数组的内容、对数组进行升幂排序、反转数组内容、复制数组内容等操作
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>  //utility头文件定义了一个pair类型,pair类型用于存储一对数据
#include <vector>  //动态扩容的数组“vector”，动态地添加新元素

#include "boost/scoped_ptr.hpp" //智能指针头文件
#include "gflags/gflags.h"//gflags是google的一个开源的处理命令行参数的库
#include "glog/logging.h"//Google Glog 是一个C++语言的应用级日志记录框架，提供了 C++ 风格的流操作和各种助手宏

#include "caffe/proto/caffe.pb.h"//将结构化数据caffe.proto，使用Protobuf编译器编译为C++类，头文件为caffe.pb.h文件
#include "caffe/util/db.hpp"  //引入包装好的lmdb操作函数
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"  //引入opencv中的图像操作函数
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)  引入全部caffe命名空间
using std::pair;	//pair类型数据对，用于存储成对的对象，例如存储文件名和对应标签
using boost::scoped_ptr;

//通过gflags宏定义一些程序的参数变量
//将需要的命令行参数使用gflags的宏定义,比如在命令行中可以这样使用，--gray=true (表示使用灰度图，默认为false)  --backend=leveldb （表示转换为leveldb格式）
//convert_imageset.exe --backend=leveldb --resize_width=64 --resize_height=64

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones"); //bool类型，是否为灰度图片,命令行中使用方式：--gray=true/false，或者省略后面的true，--gray  

DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");  //bool类型，定义洗牌变量，是否随机打乱数据集的顺序,命令行中使用方式：--shuffle=true/false

DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");  //string类型，要转换的数据类型，默认lmdb，命令行中使用方式：--backend=lmdb/leveldb

DEFINE_int32(resize_width, 0, "Width images are resized to");  //定义resize的尺寸，默认为0，不转换尺寸，等号后面跟数字，表示转换后的大小，命令行中使用方式：--resize_width=64/...

DEFINE_int32(resize_height, 0, "Height images are resized to");//同上

DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");//bool类型，是否检查大小，打开则检查所有的datum中的数据有一样的大小，命令行中使用方式：--check_size=true/false

DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");//bool类型，是否编码，命令行中使用方式：--encoded=true/false

DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");//string类型，要转换的数据格式，是否将图像转换为其他格式，命令行中使用方式：--encode_type=png/jpg/.....

int main(int argc, char** argv) {
#ifdef USE_OPENCV
	::google::InitGoogleLogging(argv[0]); //括号内是程序名,使用glog之前必须先初始化库，要生成日志文件只需在开始log之前调用一次
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  //使用说明
  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  
  //第三个参数,如果设为true，则该函数处理完成后，argv中只保留argv[0]，argc会被设置为1。
  //如果为false，则argv和argc会被保留，但是注意函数会调整argv中的顺序。
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  
  //arg[1] 训练集存放的地址，arg[2] train.txt（估计是训练集中所有图片的文件名称），arg[3] 要保存的文件名称xxlmdb
  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;		//通过gflags把宏定义变量的值，赋值给常值变量，通过使用FLAGS_xxx，可以访问宏定义的变量，FLAGS_gray就是指向命令行参数中的gray
  const bool check_size = FLAGS_check_size;  //检查图像的size，命令行中的check_size
  const bool encoded = FLAGS_encoded;         //是否编译（转换）图像格式
  const string encode_type = FLAGS_encode_type;  //要编译的图像格式，命令行中的encode_type
 
  std::ifstream infile(argv[2]);  //创建指向train.txt文件的文件读入流

  //lines 定义向量变量，向量中每个元素为一个pair对，pair对有两个成员变量，一个为string类型，一个为int类型；其中string类型用于存储文件名，int类型，用于存数对应类别的id
  //如val.txt中第一行为“ILSVRC2012_val_00000001.JPEG 65”,string = ILSVRC2012_val_00000001.JPEG   int = 65
  std::vector<std::pair<std::string, int> > lines; 

  std::string line;
  size_t pos;
  int label;
  //下面一条while语句是把train.txt文件中存放的所有文件名和标签，都存放到vector类型变量lines中；lines中存放图片的名字和对应的标签，不存储真正的图片数据
  while (std::getline(infile, line)) {//每次取出文件中的一行，然后解析文件名和对应的label
    pos = line.find_last_of(' ');  //查找这一行中的空格位置
    label = atoi(line.substr(pos + 1).c_str());//取出空格后面的值就是文件对应的label
	//line.substr(0, pos)意思是取出这个字符串中的第0到pos的字符串内容，作为文件名
    lines.push_back(std::make_pair(line.substr(0, pos), label));
  }

  //命令行参数中的是否使用参数--shuffle，如果为true，则...
  if (FLAGS_shuffle) {
    // randomly shuffle data 判断是否进行洗牌操作
    LOG(INFO) << "Shuffling data";//GLOG 有四个错误级别,INFO,WARNING,ERROR,FATAL
	//洗牌函数，使用随机生成器g对元素[first, last)容器内部元素进行随机排列 
    shuffle(lines.begin(), lines.end());//vector.begin()回传一个Iterator迭代器，它指向 vector 第一个元素
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";  //打印出来总共读取到多少个图像文件（train.txt中有多少行）

  //命令行参数中 是否指定要转换图像格式，及转换哪种图像格式？
  //--encoded=true/false    --encode_type=jpg/png....
  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  //命令行中是否指定参数要将图像转换为指定大小  --resize_height=....  --resize_width=... 
  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  //智能指针的创建方式类似泛型的格式，上面通过db.cpp内定义的命名的子命名空间中db的“成员函数”GetDB函数来初始化db对象
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));//创建智能指针，并使用GetDB函数初始化为命令行中指定格式，--backend=leveldb/lmdb
  db->Open(argv[3], db::NEW);//在argv[3]的文件夹下创建数据库文件，并打开
  scoped_ptr<db::Transaction> txn(db->NewTransaction());//创建lmdb文件的操作句柄，可以使用它来操作数据库db，比如用来将数据放入数据库

  // Storing to db
  std::string root_folder(argv[1]);//把源数据文件的地址复制给root_folder，root_folder指向图像所在目录
  Datum datum;//声明数据“转换”对象
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;
    std::string enc = encode_type;
	//enc为空串，则enc.size()=false;否则为true
	//如果命令行中指定要转换图像数据类型，--encode_type=png/jpg/...   --encoded=true,一般这2个参数同时出现，则执行转换
    if (encoded && !enc.size()) {
      // Guess the encoding type from the file name
      string fn = lines[line_id].first;//把图像的文件名赋值给fn（filename）
      size_t p = fn.rfind('.');//rfind函数的返回值是一个整形的索引值，也就是要查找的字符在字符串中的位置；若没有找到，返回string::npos
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);//找到了，就截取文件名”.“后面的字符串，以获得图像格式字符串，文件名后缀，得到了图像格式为jpg/png/....
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);//将enc字符串转换成小写
    }
	//到源数据位置读取每张图片的数据
	//ReadImageToDatum函数为io.cpp文件中定义的函数；io.cpp主要实现了3部分功能：
	/*
	1，从text文件或者二进制文件中读proto文件；
	2，利用opencv的Mat矩阵，把图像数据读到Mat矩阵中；
	3，把Mat矩阵中的值放入到datum中
	*/
    status = ReadImageToDatum(root_folder + lines[line_id].first,
        lines[line_id].second, resize_height, resize_width, is_color,
        enc, &datum);//把图像数据读取到datum中
    if (status == false) continue;//status=false,说明此张图片读取错误；“跳过”继续下一张
    
	if (check_size) {//检查图片尺寸 
      if (!data_size_initialized) {//若data_size_initialized没有初始化
        data_size = datum.channels() * datum.height() * datum.width();
        data_size_initialized = true;
      } else {
        const std::string& data = datum.data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
      }
    }
    // sequential 序列化操作，key_str格式化为: 标签_图像文件名
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;

    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));//datum数据，序列化到字符串中
    txn->Put(key_str, out); //把键值对放入到数据库,每条记录就是：标签_文件名_图像数据内容

	//每到1000个处理一次
    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();//保存到lmdb类型的文件
      txn.reset(db->NewTransaction());//重新初始化操作句柄
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  //最后处理的一个batch不足1000,写入到数据库
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
