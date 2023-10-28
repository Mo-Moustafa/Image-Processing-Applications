#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QImage>
#include <QVector>
#include <QQueue>
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "image.hpp"
#include "sift.hpp"
#include "harris.h"
#include <string>

using namespace cv;
using namespace std;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

}

MainWindow::~MainWindow()
{
    delete ui;
}

QString imgPath2,imgPath;


void MainWindow::showImg(cv::Mat& img, QLabel* imgLbl, enum QImage::Format imgFormat, int width , int hieght)
{
    Mat img2;
    cvtColor(img, img2, COLOR_BGR2RGB);
    QImage image((uchar*)img2.data, img2.cols, img2.rows, imgFormat);
    QPixmap pix = QPixmap::fromImage(image);
//    imgLbl->resize(img.rows, img.cols);
    imgLbl->setPixmap(pix.scaled(width, hieght, Qt::KeepAspectRatio));
//    imgLbl->setScaledContents(true);
}



// Swapping between the 3 pages
void MainWindow::on_pushButton_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void MainWindow::on_pushButton_3_clicked()
{
    ui->stackedWidget->setCurrentIndex(1);
}

void MainWindow::on_pushButton_2_clicked()
{
    ui->stackedWidget->setCurrentIndex(2);
}



void MainWindow::on_actionUpload_triggered()
{
    imgPath = QFileDialog::getOpenFileName(this, "Open an Image", "..", "Images (*.png *.xpm *.jpg *.bmb)");

    if(imgPath.isEmpty())
        return;

    cv::Mat img = imread(imgPath.toStdString());
    cv::resize(img,img,cv::Size(500,500));

    showImg(img, ui->imginput1, QImage::Format_RGB888, ui->imginput1->width(), ui->imginput1->height());
    showImg(img, ui->imginput2, QImage::Format_RGB888, ui->imginput1->width(), ui->imginput1->height());
    showImg(img, ui->imginput3_1, QImage::Format_RGB888, ui->imginput3_1->width(), ui->imginput3_1->height());

    ui->doubleSpinBox->setValue(0.04);
    ui->spinBox->setValue(150);
    ui->matchThr1->setValue(10000);
    ui->matchThr2->setValue(10000);
    ui->imgmatching->clear();
    ui->imgHarris->clear();
    ui->imgSift->clear();


}


void MainWindow::on_harrisBtn_clicked()
{
    cv::Mat img = imread(imgPath.toStdString());
    cv::resize(img,img,cv::Size(500,500));

    double k = ui->doubleSpinBox->value();
    double thr = ui->spinBox->value();
    auto start = std::chrono::high_resolution_clock::now();
    Mat result = harrisCornerDetector(img, k, thr);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    ui->harrisTime->display(duration.count()*0.000001);


    showImg(result, ui->imgHarris, QImage::Format_RGB888, ui->imginput1->width(), ui->imginput1->height());
}


void MainWindow::on_siftBtn_clicked()
{
    Image image(imgPath.toStdString());
//    image =  image.channels == 1 ? image : rgb_to_grayscale(image);




    auto start = std::chrono::high_resolution_clock::now();
    // call your function
    std::vector<sift::Keypoint> kps = sift::find_keypoints_and_descriptors(image);
    // end time measurement
    auto end = std::chrono::high_resolution_clock::now();
    Image result = sift::draw_keypoints(image, kps);
    Mat img = result.save("result.jpg");
    cv::resize(img,img,cv::Size(500,500));

    showImg(img, ui->imgSift, QImage::Format_RGB888, ui->imginput1->width(), ui->imginput1->height());

    // compute and print the elapsed time
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//    std::cout << "Time taken by function: " << duration.count()*0.000001 << " seconds" << std::endl
    ui->siftTime->display(duration.count()*0.000001);

}





void MainWindow::on_matchBtn_clicked()
{
    Image image(imgPath.toStdString());
    Image image2(imgPath2.toStdString());

    image =  image.channels == 1 ? image : rgb_to_grayscale(image);
    image2 =  image2.channels == 1 ? image2 : rgb_to_grayscale(image2);

    std::vector<sift::Keypoint> kps = sift::find_keypoints_and_descriptors(image);
    std::vector<sift::Keypoint> kps2 = sift::find_keypoints_and_descriptors(image2);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<int, int>> match= find_keypoint_matches(kps,kps2, ui->matchThr1->value()+0.2,ui->matchThr2->value(), ui->matchOption->currentIndex());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    Image result = sift::draw_matches_rect(image, image2, kps, kps2, match);

    ui->matchTime->display(duration.count()*0.000001);


    Mat img = result.save("result.jpg");
    cv::resize(img,img,cv::Size(500,300));

    showImg(img, ui->imgmatching, QImage::Format_RGB888, ui->imgmatching->width(), ui->imgmatching->height());

}


void MainWindow::on_matchThr1_valueChanged(int value)
{
    ui->label_5->setText(QString::number(value));
}



void MainWindow::on_matchThr2_valueChanged(int value)
{
    ui->label_10->setText(QString::number(value));

}


void MainWindow::on_matchOption_currentIndexChanged(int index)
{
    if(index==0){
        ui->matchThr1->setValue(10000);
        ui->matchThr2->setValue(10000);
    }
    else{
        ui->matchThr1->setValue(1);
        ui->matchThr2->setValue(50);
    }
}


void MainWindow::on_pushButton_4_clicked()
{
       imgPath2 = QFileDialog::getOpenFileName(this, "Open an Image", "..", "Images (*.png *.xpm *.jpg *.bmb)");

       if(imgPath2.isEmpty())
            return;

        cv::Mat img = imread(imgPath2.toStdString());
        cv::resize(img,img,cv::Size(500,500));

        showImg(img, ui->imginput3_2, QImage::Format_RGB888, ui->imginput3_2->width(), ui->imginput3_2->height());
}

