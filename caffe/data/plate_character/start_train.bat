..\..\Build\x64\Release\caffe.exe train --solver=.\AlexNet\solver.prototxt

rem ..\..\Build\x64\Release\caffe.exe train --solver=.\googlenet\solver.prototxt
pause

rem D:\\WorkSpace\\caffe\\caffe\\Build\\x64\\Release\\caffe.exe test --model=D:\\WorkSpace\\caffe\\caffe\\examples\\cifar10\\cifar10_quick_train_test.prototxt -weights=D:\\WorkSpace\\caffe\\caffe\\examples\\cifar10\\cifar10_quick_iter_4000.caffemodel.h5 -gpu=0
rem pause