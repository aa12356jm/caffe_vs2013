SET GLOG_logtostderr=1
..\..\Build\x64\Release\convert_imageset.exe --backend=leveldb --resize_width=256 --resize_height=256  ..\plate  .\train\train.txt  .\train_ldb 
pause

..\..\Build\x64\Release\convert_imageset.exe --backend=leveldb --resize_width=256 --resize_height=256  ..\plate  .\test\test.txt  .\test_ldb 
pause


rem D:\Enjoy_Project\caffe\caffe\data\plate    D:\Enjoy_Project\caffe\caffe\data\plate\train\train.txt  前面的文件夹和后面文件中的路径需要组成一个绝对路径，可以保证找到这个图片