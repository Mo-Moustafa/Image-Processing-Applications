#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "opencv2/world.hpp"
#include <QLabel>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();



private slots:
    void on_actionUpload_triggered();

    void on_set_seedsBtn_clicked();

    void on_submitBtn_clicked();

    void on_comboBox_currentTextChanged(const QString &arg1);

    void on_ThresholdSubmit_clicked();

    void on_horizontalSlider_2_valueChanged(int value);

    void on_global_toggled(bool checked);

    void on_local_toggled(bool checked);

    void on_comboBox_2_currentIndexChanged(int index);

    void on_pushButton_clicked();

    void on_pushButton_3_clicked();

    void on_horizontalSlider_sliderMoved(int position);

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
