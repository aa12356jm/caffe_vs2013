SET GLOG_logtostderr=1
..\..\Build\x64\Release\convert_imageset.exe --backend=leveldb --resize_width=64 --resize_height=64  ..\plate_character .\train\train.txt .\train_ldb

..\..\Build\x64\Release\convert_imageset.exe --backend=leveldb --resize_width=64 --resize_height=64 ..\plate_character .\test\test.txt .\test_ldb
pause
