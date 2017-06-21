#include "caffe_Interface.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	caffe_Interface w;
	w.show();
	return a.exec();
}
