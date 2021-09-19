#define Gamma 3
#include <iostream>
#include <time.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
using namespace cv;

#include<opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
using namespace std;
#include <opencv2/videoio.hpp>
#include <opencv2/photo/cuda.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/cudafilters.hpp>

int main()
{
    Mat src = imread("frameleft.jpeg");
    Mat dst(src.size(), CV_32FC3);
    
    for (int i = 0; i < src.rows;i++)
    {
            for (int j = 0; j < src.cols; j++)
            {
                    dst.at<Vec3f>(i, j)[0] = pow(src.at<Vec3b>(i, j)[0], Gamma);
                    dst.at<Vec3f>(i, j)[1] = pow(src.at<Vec3b>(i, j)[1], Gamma);
                    dst.at<Vec3f>(i, j)[2] = pow(src.at<Vec3b>(i, j)[2], Gamma);
            }
    }
    normalize(dst, dst, 0, 255, CV_MINMAX);
    convertScaleAbs(dst, dst);

    waitKey();
    return 0;
}
