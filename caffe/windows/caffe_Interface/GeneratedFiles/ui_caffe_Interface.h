/********************************************************************************
** Form generated from reading UI file 'caffe_Interface.ui'
**
** Created by: Qt User Interface Compiler version 5.8.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CAFFE_INTERFACE_H
#define UI_CAFFE_INTERFACE_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_caffe_InterfaceClass
{
public:
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QWidget *centralWidget;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *caffe_InterfaceClass)
    {
        if (caffe_InterfaceClass->objectName().isEmpty())
            caffe_InterfaceClass->setObjectName(QStringLiteral("caffe_InterfaceClass"));
        caffe_InterfaceClass->resize(600, 400);
        menuBar = new QMenuBar(caffe_InterfaceClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        caffe_InterfaceClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(caffe_InterfaceClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        caffe_InterfaceClass->addToolBar(mainToolBar);
        centralWidget = new QWidget(caffe_InterfaceClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        caffe_InterfaceClass->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(caffe_InterfaceClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        caffe_InterfaceClass->setStatusBar(statusBar);

        retranslateUi(caffe_InterfaceClass);

        QMetaObject::connectSlotsByName(caffe_InterfaceClass);
    } // setupUi

    void retranslateUi(QMainWindow *caffe_InterfaceClass)
    {
        caffe_InterfaceClass->setWindowTitle(QApplication::translate("caffe_InterfaceClass", "caffe_Interface", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class caffe_InterfaceClass: public Ui_caffe_InterfaceClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CAFFE_INTERFACE_H
