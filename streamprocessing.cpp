#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <vector>
#include <memory>
#include <iostream>

std::shared_ptr<std::vector<cv::Mat>> processArray(std::shared_ptr<std::vector< cv::cuda::HostMem >> srcMemArray,
                                                   std::shared_ptr<std::vector< cv::cuda::HostMem >> dstMemArray,
                                                   std::shared_ptr<std::vector< cv::cuda::GpuMat >> gpuSrcArray,
                                                   std::shared_ptr<std::vector< cv::cuda::GpuMat >> gpuDstArray,
                                                   std::shared_ptr<std::vector< cv::Mat >> outArray,
                                                   std::shared_ptr<std::vector< cv::cuda::Stream >> streamsArray){

    //Define test target size 
    cv::Size rSize(256, 256);

    //Compute for each input image with async calls
    for(int i=0; i<4; i++){
        
        //Upload Input Pinned Memory to GPU Mat
        (*gpuSrcArray)[i].upload((*srcMemArray)[i], (*streamsArray)[i]);

        //Use the CUDA Kernel Method
        cv::cuda::resize((*gpuSrcArray)[i], (*gpuDstArray)[i], rSize, 0, 0, cv::INTER_AREA, (*streamsArray)[i]);

        //Download result to Output Pinned Memory
        (*gpuDstArray)[i].download((*dstMemArray)[i],(*streamsArray)[i]);

        //Obtain data back to CPU Memory
        (*outArray)[i] = (*dstMemArray)[i].createMatHeader();
        cv::imshow("outarray is", ( *outArray)[i]);
        cv::waitKey(0);
    }

    //All previous calls are non-blocking therefore wait for each stream completetion
    (*streamsArray)[0].waitForCompletion();
    (*streamsArray)[1].waitForCompletion();
    (*streamsArray)[2].waitForCompletion();
    (*streamsArray)[3].waitForCompletion();

    return outArray;
}



int main (int argc, char* argv[])
{
    //Load test image
    cv::Mat srcHostImage = cv::imread("frameleft.jpeg");

    //1.Create CUDA Streams Array
    std::shared_ptr<std::vector<cv::cuda::Stream>> streamsArray = std::make_shared<std::vector<cv::cuda::Stream>>();

    cv::cuda::Stream streamA, streamB, streamC, streamD;
    streamsArray->push_back(streamA);
    streamsArray->push_back(streamB);
    streamsArray->push_back(streamC);
    streamsArray->push_back(streamD);

    //2.Create Pinned Memory (PAGE_LOCKED) arrays
    std::shared_ptr<std::vector<cv::cuda::HostMem >> srcMemArray = std::make_shared<std::vector<cv::cuda::HostMem >>();
    std::shared_ptr<std::vector<cv::cuda::HostMem >> dstMemArray = std::make_shared<std::vector<cv::cuda::HostMem >>();

    //3.Create Cuda GpuMat arrays
    std::shared_ptr<std::vector< cv::cuda::GpuMat >> gpuSrcArray = std::make_shared<std::vector<cv::cuda::GpuMat>>();
    std::shared_ptr<std::vector< cv::cuda::GpuMat >> gpuDstArray = std::make_shared<std::vector<cv::cuda::GpuMat>>();

    //Create Output array for CPU Mat
    std::shared_ptr<std::vector< cv::Mat >> outArray = std::make_shared<std::vector<cv::Mat>>();
    
    for(int i=0; i<4; i++){
        //Define GPU Mats
        cv::cuda::GpuMat srcMat;
        cv::cuda::GpuMat dstMat;
        
        //Define CPU Mat
        cv::Mat outMat;
        
        //Initialize the Pinned Memory with input image
        cv::cuda::HostMem srcHostMem = cv::cuda::HostMem(srcHostImage, cv::cuda::HostMem::PAGE_LOCKED);

        //Initialize the output Pinned Memory with reference to output Mat
        cv::cuda::HostMem srcDstMem = cv::cuda::HostMem(outMat, cv::cuda::HostMem::PAGE_LOCKED);

        //Add elements to each array.
        srcMemArray->push_back(srcHostMem);
        dstMemArray->push_back(srcDstMem);

        gpuSrcArray->push_back(srcMat);
        gpuDstArray->push_back(dstMat);
        outArray->push_back(outMat);
    }
    
    //Test the process 20 times
    for(int i=0; i<2; i++){
        try{

            std::shared_ptr<std::vector<cv::Mat>> result = std::make_shared<std::vector<cv::Mat>>();

            result = processArray(srcMemArray, dstMemArray, gpuSrcArray, gpuDstArray, outArray, streamsArray);
            //Optional to show the results
            //cv::imshow("Result", (*result)[0]);
            //cv::waitKey(0);
        }

        catch(const cv::Exception& ex){
            std::cout << "Error: " << ex.what() << std::endl;
        }
    }

    return 0;
}
