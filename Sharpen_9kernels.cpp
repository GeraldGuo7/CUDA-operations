#include<opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>
using namespace std;

#include <opencv2/core.hpp>
using namespace cv;

#include <opencv2/cudaarithm.hpp>
using namespace cv::cuda;

#include <opencv2/core/cuda.hpp>
#include <opencv2/videoio.hpp>
    /*
    kernel_sharpen_2 = np.array([
            [1,1,1],
            [1,-7,1],
            [1,1,1]])

    kernel_sharpen_3 = np.array([
            [-1,-1,-1,-1,-1],
            [-1,2,2,2,-1],
            [-1,2,8,2,-1],
            [-1,2,2,2,-1], 
            [-1,-1,-1,-1,-1]])/8.0
    #图像锐化
    kernel_sharpen_4 = np.array([
            [0,-1,0],
            [-1,5,-1],
            [0,-1,0]])
    #图像模糊
    kernel_sharpen_5 = np.array([
            [0.0625,0.125,0.0625],
            [0.125,0.25,0.125],
            [0.0625,0.125,0.125]])
    #索贝尔
    kernel_sharpen_6 = np.array([
            [-1,-2,-1],
            [0,0,0],
            [1,2,1]])
    #浮雕
    kernel_sharpen_7 = np.array([
            [-2,-1,0],
            [-1,1,1],
            [0,1,2]])
    #大纲outline
    kernel_sharpen_8 = np.array([
            [-1,-1,-1],
            [-1,8,-1],
            [-1,-1,-1]])
    #拉普拉斯算子
    kernel_sharpen_9 = np.array([
            [0,1,0],
            [1,-4,1],
            [0,1,0]])
    */

    /*
    cv::Ptr <cv::cuda::Convolution> convolutionkernel = cv::cuda::createConvolution();
    cv::mulTransposed(image, kernel, false);

    cv::Mat gauss_kernel;
    cv::cuda::GpuMat gf_src, gf_gauss, gf_conv;
    //std:shared_ptr<std::vector <cv::cuda::GpuMat>> gfv_src, gfv_gauss,  gfv_conv;
    std::vector <cv::cuda::GpuMat> gfv_src, gfv_gauss, gfv_conv;

    src.convertTo(gf_src, CV_32FC3);

    //Split the source image on the GPU
    cv::cuda::split(gf_src, gfv_conv);
    cv::cuda::split(gf_src, gfv_src);

    //Prepare the input image for Gauss filtering
    cv::cuda::copyMakeBorder(gf_src, gf_gauss, 0.5 * gauss_size,
        0.5 * gauss_size, 0.5 * gauss_size, 0.5 * gauss_size,
        cv::BORDER_REPLICATE);
    cv::cuda::split(gf_gauss, gfv_gauss);

    //Create the Gauss kernel and upload it to the GPU
    //Smart pointers with shared ownership.
    //A Ptr<T> pretends to be a pointer to an object of type T. Unlike an ordinary pointer, however, the object will
    //be automatically cleaned up once all Ptr instances pointing to it are destroyed.
    cv::Ptr < cv::cuda::Convolution> c1 = cv::cuda::createConvolution();
    cv::mulTransposed(cv::getGaussianKernel(gauss_size, -1, CV_32FC1),
        gauss_kernel, false);
    gf_gauss.upload(gauss_kernel);

    //Apply Gaussian blur to all channels independently, also compute subtraction image
    for (int i = 0; i < src.channels(); i++)
    {
        c1->convolve(gfv_gauss[i], gf_gauss, gfv_conv[i], true);
    }
    cv::cuda::merge(gfv_conv, conv);

    std::count << "Vector size is:"<< gfv_conv.size() << endl;

    cv::imshow("Gauss kernel is:",gauss_kernel);
    cv::waitKey(0);
    */

/*
cv2.imshow('selfdefined_2 Image',output_2)
cv2.imshow('selfdefined_3 Image',output_3)
cv2.imshow('traditional sharpen Image',output_4)
cv2.imshow('blurring Image',output_5)
cv2.imshow('sobel Image',output_6)
cv2.imshow('embedding Image',output_7)
cv2.imshow('outline Image',output_8)
cv2.imshow('Laplacian operator',output_9)
cv2.imshow('Laplacian final',output_10)
*/
