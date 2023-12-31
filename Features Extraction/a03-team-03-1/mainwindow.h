#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "opencv2/world.hpp"
#include <QLabel>
using namespace cv;

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void showImg(cv::Mat& img, QLabel* imgLbl, enum QImage::Format imgFormat, int width , int hieght);

private slots:
    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

    void on_pushButton_3_clicked();

    void on_actionUpload_triggered();

    void on_harrisBtn_clicked();

    void on_siftBtn_clicked();

    void on_matchBtn_clicked();

    void on_matchThr1_valueChanged(int value);

    void on_matchThr2_valueChanged(int value);

    void on_matchOption_currentIndexChanged(int index);

    void on_pushButton_4_clicked();

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
