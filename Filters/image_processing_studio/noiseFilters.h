#ifndef NOISEFILTERS_H
#define NOISEFILTERS_H
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2\highgui.hpp>

using namespace std;
using namespace cv;

Mat padding(Mat img, int k_width, int k_height);
Mat define_kernel_box(int k_width, int k_height);
Mat define_kernel_gaussian(int k_width, int k_height);
void insertionSort(int arr[], int n);
void boxFilter(Mat& img, int k_width, int k_height);
void gaussianFilter(Mat& img, int k_width, int k_height, int sigma = 10);
void medianFilter(Mat& img);

#endif // NOISEFILTERS_H
