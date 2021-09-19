//Your Opencv should be compiled with CUDA and OpenGL sopport to test all these features. 
//Obviously when adding CUDA support to your code, nothing is more important than adding the header first.
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
    // init video capture with video
    VideoCapture capture(videoFileName);
    if (!capture.isOpened())
    {
        // error in opening the video file
        cout << "Unable to open file!" << endl;
        return;
    }

    // get default video FPS
    double fps = capture.get(CAP_PROP_FPS);

    // get total number of video frames
    int num_frames = int(capture.get(CAP_PROP_FRAME_COUNT));


    // resize frame
    cv::resize(frame, frame, Size(960, 540), 0, 0, INTER_LINEAR);

    // convert to gray
    cv::cvtColor(frame, previous_frame, COLOR_BGR2GRAY);

    // upload pre-processed frame to GPU
    cv::cuda::GpuMat gpu_previous;
    gpu_previous.upload(previous_frame);

    // declare cpu outputs for optical flow
    cv::Mat hsv[3], angle, bgr;

    // declare gpu outputs for optical flow
    cv::cuda::GpuMat gpu_magnitude, gpu_normalized_magnitude, gpu_angle;
    cv::cuda::GpuMat gpu_hsv[3], gpu_merged_hsv, gpu_hsv_8u, gpu_bgr;

    // set saturation to 1
    hsv[1] = cv::Mat::ones(frame.size(), CV_32F);
    gpu_hsv[1].upload(hsv[1]);

    while (true)
    {
        // start full pipeline timer
        auto start_full_time = high_resolution_clock::now();

        // start reading timer
        auto start_read_time = high_resolution_clock::now();

        // capture frame-by-frame
        capture >> frame;

        if (frame.empty())
            break;

        // upload frame to GPU
        cv::cuda::GpuMat gpu_frame;
        gpu_frame.upload(frame);

        // end reading timer
        auto end_read_time = high_resolution_clock::now();

        // add elapsed iteration time
        timers["reading"].push_back(duration_cast<milliseconds>(end_read_time - start_read_time).count() / 1000.0);

        // start pre-process timer
        auto start_pre_time = high_resolution_clock::now();

        // resize frame
        cv::cuda::resize(gpu_frame, gpu_frame, Size(960, 540), 0, 0, INTER_LINEAR);

        // convert to gray
        cv::cuda::GpuMat gpu_current;
        cv::cuda::cvtColor(gpu_frame, gpu_current, COLOR_BGR2GRAY);

        // end pre-process timer
        auto end_pre_time = high_resolution_clock::now();

        // add elapsed iteration time
        timers["pre-process"].push_back(duration_cast<milliseconds>(end_pre_time - start_pre_time).count() / 1000.0);

        // start optical flow timer
        auto start_of_time = high_resolution_clock::now();

        // create optical flow instance
        Ptr<cuda::FarnebackOpticalFlow> ptr_calc = cuda::FarnebackOpticalFlow::create(5, 0.5, false, 15, 3, 5, 1.2, 0);
        // calculate optical flow
        cv::cuda::GpuMat gpu_flow;
        ptr_calc->calc(gpu_previous, gpu_current, gpu_flow);

        // end optical flow timer
        auto end_of_time = high_resolution_clock::now();

        // add elapsed iteration time
        timers["optical flow"].push_back(duration_cast<milliseconds>(end_of_time - start_of_time).count() / 1000.0);

        // start post-process timer
        auto start_post_time = high_resolution_clock::now();

        // split the output flow into 2 vectors
        cv::cuda::GpuMat gpu_flow_xy[2];
        cv::cuda::split(gpu_flow, gpu_flow_xy);

        // convert from cartesian to polar coordinates
        cv::cuda::cartToPolar(gpu_flow_xy[0], gpu_flow_xy[1], gpu_magnitude, gpu_angle, true);

        // normalize magnitude from 0 to 1
        cv::cuda::normalize(gpu_magnitude, gpu_normalized_magnitude, 0.0, 1.0, NORM_MINMAX, -1);

        // get angle of optical flow
        gpu_angle.download(angle);
        angle *= ((1 / 360.0) * (180 / 255.0));

        // build hsv image
        gpu_hsv[0].upload(angle);
        gpu_hsv[2] = gpu_normalized_magnitude;
        cv::cuda::merge(gpu_hsv, 3, gpu_merged_hsv);

        // multiply each pixel value to 255
        gpu_merged_hsv.cv::cuda::GpuMat::convertTo(gpu_hsv_8u, CV_8U, 255.0);

        // convert hsv to bgr
        cv::cuda::cvtColor(gpu_hsv_8u, gpu_bgr, COLOR_HSV2BGR);

        // send original frame from GPU back to CPU
        gpu_frame.download(frame);

        // send result from GPU back to CPU
        gpu_bgr.download(bgr);

        // update previous_frame value
        gpu_previous = gpu_current;

        // end post pipeline timer
        auto end_post_time = high_resolution_clock::now();

        // add elapsed iteration time
        timers["post-process"].push_back(duration_cast<milliseconds>(end_post_time - start_post_time).count() / 1000.0);

        // end full pipeline timer
        auto end_full_time = high_resolution_clock::now();

        // add elapsed iteration time
        timers["full pipeline"].push_back(duration_cast<milliseconds>(end_full_time - start_full_time).count() / 1000.0);


        // visualization
        imshow("original", frame);
        imshow("result", bgr);
        int keyboard = waitKey(1);
        if (keyboard == 27)
            break;


        cout << "Elapsed time" << std::endl;
        
        for (auto const& timer : timers)
        {
            cout << "- " << timer.first << " : " << accumulate(timer.second.begin(), timer.second.end(), 0.0) << " seconds"<< endl;
        }

        // calculate frames per second
        cout << "Default video FPS : "  << fps << endl;
        float optical_flow_fps  = (num_frames - 1) / accumulate(timers["optical flow"].begin(),  timers["optical flow"].end(),  0.0);
        cout << "Optical flow FPS : "   << optical_flow_fps  << endl;

        float full_pipeline_fps = (num_frames - 1) / accumulate(timers["full pipeline"].begin(), timers["full pipeline"].end(), 0.0);
        cout << "Full pipeline FPS : "  << full_pipeline_fps << endl;


    cv::imshow("result", result);
    cv::waitKey();
    return 0;
}
