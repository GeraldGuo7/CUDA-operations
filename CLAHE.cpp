#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>
using namespace std;

#include <opencv2/core.hpp>
using namespace cv;

#include <opencv2/cudaarithm.hpp>
using namespace cv::cuda;

int main()
{
    printShortCudaDeviceInfo(getDevice());
    int cuda_devices_number = getCudaEnabledDeviceCount();
    cout << "CUDA Device(s) Number: "<< cuda_devices_number << endl;
    DeviceInfo _deviceInfo;
    bool _isd_evice_compatible = _deviceInfo.isCompatible();
    cout << "CUDA Device(s) Compatible: " << _isd_evice_compatible << endl;

    //Part 1:Data transfer between CPU and GPU
    cv::Mat img = cv::imread("frameleft.jpeg", IMREAD_GRAYSCALE);
    cv::cuda::GpuMat dst, src;
    src.upload(img);

    cv::Ptr<cv::cuda::CLAHE> ptr_clahe = cv::cuda::createCLAHE(5.0, cv::Size(8, 8));
    ptr_clahe->apply(src, dst);

    cv::Mat result;
    dst.download(result);

    cv::imshow("result", result);
    cv::waitKey();
    return 0;
}
