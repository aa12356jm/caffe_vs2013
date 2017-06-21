#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_caffe_Interface.h"

class caffe_Interface : public QMainWindow
{
	Q_OBJECT

public:
	caffe_Interface(QWidget *parent = Q_NULLPTR);

private:
	Ui::caffe_InterfaceClass ui;
};
