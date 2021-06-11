#pragma once 
#include "cuda_runtime_api.h"


#ifdef __cplusplus
extern "C"
{
#endif


void * yolov5_trt_create(const char * engine_name);

//char * yolov5_trt_detect(void *h, std::string &img, float threshold);
const char * yolov5_trt_detect(void *h, cv::Mat &img, float threshold);

void yolov5_trt_destroy(void *h);


#ifdef __cplusplus
}
#endif 
