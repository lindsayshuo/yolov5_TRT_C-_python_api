# yolov5_TRT_C-_python_api
convert  yolov5 c++ infer to python pakage,python use pakege to get label and cor
可先参考我的一篇博客配置下环境：
	
	https://blog.csdn.net/weixin_43269994/article/details/117219986?spm=1001.2014.3001.5502

编写CMakeLists.txt


	cmake_minimum_required(VERSION 2.6)

	project(yolov5)

	# add_definitions(-std=c++11)

	option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
	set(CMAKE_CXX_STANDARD 11)
	set(CMAKE_BUILD_TYPE Debug)

	find_package(CUDA REQUIRED)

	include_directories(${PROJECT_SOURCE_DIR}/include)
	# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
	# cuda
	include_directories(/usr/local/cuda-11.1/include)
	link_directories(/usr/local/cuda-11.1/lib64)

	# tensorrt
	include_directories(/home/lindsay/TensorRT-7.2.2.3/include)
	link_directories(/home/lindsay/TensorRT-7.2.2.3/lib)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

	cuda_add_library(yolov5_trt SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu ${PROJECT_SOURCE_DIR}/yolov5_lib.cpp)

	find_package(OpenCV)
	# find_package(OpenCV 4.4.0  REQUIRED)
	include_directories(${OpenCV_INCLUDE_DIRS})

	target_link_libraries(yolov5_trt nvinfer cudart ${OpenCV_LIBS})

	# add_executable(yolov5 ${PROJECT_SOURCE_DIR}/yolov5.cpp)
	add_executable(yolov5 ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/yolov5.cpp)
	target_link_libraries(yolov5 nvinfer)
	target_link_libraries(yolov5 cudart)
	target_link_libraries(yolov5 yolov5_trt)
	target_link_libraries(yolov5 ${OpenCV_LIBS})

	message(${OpenCV_LIBS})

	add_definitions(-O2 -pthread)


## 包装 yolov5 tensort 为 C++库

   //yolov5_lib.h
     
    #pragma once 
     
    #ifdef __cplusplus
    extern "C" 
    {
    #endif 
     
    void * yolov5_trt_create(const char * engine_name);
     
    const char * yolov5_trt_detect(void *h, cv::Mat &img, float threshold);
     
    void yolov5_trt_destroy(void *h);
     
    #ifdef __cplusplus
    }
    #endif 
    ~            


   //yolov5_lib.cpp 
     
    #include <iostream>
    #include <chrono>
    #include "cuda_runtime_api.h"
    #include "logging.h"
    #include "common.hpp"
    #include "yolov5_lib.h"
     
    #define USE_FP16  // comment out this if want to use FP32
    #define DEVICE 0  // GPU id
    #define NMS_THRESH 0.4
    #define CONF_THRESH 0.5
    #define BATCH_SIZE 1
     
    // stuff we know about the network and the input/output blobs
    static const int INPUT_H = Yolo::INPUT_H;
    static const int INPUT_W = Yolo::INPUT_W;
    static const int CLASS_NUM = Yolo::CLASS_NUM;
    static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
    const char* INPUT_BLOB_NAME = "data";
    const char* OUTPUT_BLOB_NAME = "prob";
    static Logger gLogger;
     
     
    static void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(batchSize, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    }
     
     
    typedef struct 
    {
     
        float *data;
        float *prob;
        IRuntime *runtime;
        ICudaEngine *engine;
        IExecutionContext *exe_context;
        void* buffers[2];
        cudaStream_t cuda_stream;
        int inputIndex;
        int outputIndex;
        char result_json_str[16384];
     
    }Yolov5TRTContext;
     
    oid * yolov5_trt_create(const char * engine_name)
    {
        size_t size = 0;
        char *trtModelStream = NULL;
        Yolov5TRTContext * trt_ctx = NULL;
     
        trt_ctx = new Yolov5TRTContext();
     
        std::ifstream file(engine_name, std::ios::binary);
        printf("yolov5_trt_create  ... \n");
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }else
            return NULL;
     
        trt_ctx->data = new float[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
        trt_ctx->prob = new float[BATCH_SIZE * OUTPUT_SIZE];
        trt_ctx->runtime = createInferRuntime(gLogger);
        assert(trt_ctx->runtime != nullptr);
     
        printf("yolov5_trt_create  cuda engine... \n");
        trt_ctx->engine = trt_ctx->runtime->deserializeCudaEngine(trtModelStream, size);
        assert(trt_ctx->engine != nullptr);
        trt_ctx->exe_context = trt_ctx->engine->createExecutionContext();
     
     
        delete[] trtModelStream;
        assert(trt_ctx->engine->getNbBindings() == 2);
     
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        trt_ctx->inputIndex = trt_ctx->engine->getBindingIndex(INPUT_BLOB_NAME);
        trt_ctx->outputIndex = trt_ctx->engine->getBindingIndex(OUTPUT_BLOB_NAME);
     
        assert(trt_ctx->inputIndex == 0);
        assert(trt_ctx->outputIndex == 1);
        // Create GPU buffers on device
     
        printf("yolov5_trt_create  buffer ... \n");
        CHECK(cudaMalloc(&trt_ctx->buffers[trt_ctx->inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
        CHECK(cudaMalloc(&trt_ctx->buffers[trt_ctx->outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
        // Create stream
     
        printf("yolov5_trt_create  stream ... \n");
        CHECK(cudaStreamCreate(&trt_ctx->cuda_stream));
        printf("yolov5_trt_create  done ... \n");
        return (void *)trt_ctx;
     
     
    }
     
     
    const char * yolov5_trt_detect(void *h, cv::Mat &img, float threshold)
    {
        Yolov5TRTContext *trt_ctx;
        int i;
        int delay_preprocess;
        int delay_infer;
     
        trt_ctx = (Yolov5TRTContext *)h;
     
     
        trt_ctx->result_json_str[0] = 0;
     
        if (img.empty()) return trt_ctx->result_json_str;
     
        auto start0 = std::chrono::system_clock::now();
     
        //printf("yolov5_trt_detect start preprocess img \n");
        cv::Mat pr_img = preprocess_img(img);
     
     
     
        //printf("yolov5_trt_detect start convert img to float\n");
        // letterbox BGR to RGB
        i = 0;
        for (int row = 0; row < INPUT_H; ++row) {
            uchar* uc_pixel = pr_img.data + row * pr_img.step;
            for (int col = 0; col < INPUT_W; ++col) {
                trt_ctx->data[i] = (float)uc_pixel[2] / 255.0;
                trt_ctx->data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                trt_ctx->data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                uc_pixel += 3;
                ++i;
            }
        }
        auto end0 = std::chrono::system_clock::now();
     
        delay_preprocess =  std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start0).count();
     
        // Run inference
        //printf("yolov5_trt_detect start do inference\n");
        auto start = std::chrono::system_clock::now();
        doInference(*trt_ctx->exe_context, trt_ctx->cuda_stream, trt_ctx->buffers, trt_ctx->data, trt_ctx->prob, BATCH_SIZE);
     
        auto end = std::chrono::system_clock::now();
        delay_infer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
     
        std::cout <<"delay_proress:" << delay_preprocess << "ms, " << "delay_infer:" << delay_infer << "ms" << std::endl;
     
        //printf("yolov5_trt_detect start do process infer result \n");
     
        int fcount = 1;
        int str_len;
        std::vector<std::vector<Yolo::Detection>> batch_res(1);
        auto& res = batch_res[0];
        nms(res, &trt_ctx->prob[0], threshold, NMS_THRESH);
     
        sprintf(trt_ctx->result_json_str,
                    "{\"delay_preprocess\": %d,"
                    "\"delay_infer\": %d,"
                    "\"num_det\":%d, \"objects\":[", delay_preprocess, delay_infer, (int) res.size());
     
        str_len = strlen(trt_ctx->result_json_str);
     
        i = 0;
        for(i = 0 ; i < res.size(); i++){
            int x1, y1, x2, y2;
            int class_id;
     
            cv::Rect r = get_rect(img, res[i].bbox);
     
            x1 = r.x;
            y1 = r.y;
            x2 = r.x + r.width;
            y2 = r.y + r.height;
            class_id = (int)res[i].class_id;
     
     
            if (0 == i){
                sprintf(trt_ctx->result_json_str + str_len, "(%d,%d,%d,%d,%d)", class_id, x1, y1, x2, y2);
            }else {
                sprintf(trt_ctx->result_json_str + str_len, ",(%d,%d,%d,%d,%d)", class_id, x1, y1, x2, y2);
            }
            str_len = strlen(trt_ctx->result_json_str);
     
            if (str_len >= 16300)
                break;
     
        }
     
        sprintf(trt_ctx->result_json_str + str_len, "]}");
     
     
        return trt_ctx->result_json_str;
     
    }
     
     
    void yolov5_trt_destroy(void *h)
    {
        Yolov5TRTContext *trt_ctx;
     
        trt_ctx = (Yolov5TRTContext *)h;
     
        // Release stream and buffers
        cudaStreamDestroy(trt_ctx->cuda_stream);
        CHECK(cudaFree(trt_ctx->buffers[trt_ctx->inputIndex]));
        CHECK(cudaFree(trt_ctx->buffers[trt_ctx->outputIndex]));
        // Destroy the engine
        trt_ctx->exe_context->destroy();
        trt_ctx->engine->destroy();
        trt_ctx->runtime->destroy();
     
        delete trt_ctx->data;
        delete trt_ctx->prob;
     
        delete trt_ctx;
     
    }
     
     

 执行以下命令：

	mkdir build && cd build
	cmake ..
	make -j10
	sudo ./yolov5 -s ../yolov5s.wts ../yolov5s.engine s# 生成引擎以及libyolov5_trt.so


     

编译得到 libyolov5_trt.so

## 包装 yolov5 tensort 为 python 库 (基于 c++库，调用libyolov5_trt.so)

python modules , 参考：

https://github.com/walletiger/tensorrt_retinaface_with_python/tree/main/python

yolov5_trt_py_module.cpp


	#include <stdio.h>
	#include <stdlib.h>
	#include <assert.h>
	#include <Python.h>

	#include <opencv2/opencv.hpp>
	#include <opencv2/highgui/highgui.hpp>
	#include "../yolov5_lib.h"
	#include "pyboostcvconverter/pyboostcvconverter.hpp"
	#include <boost/python.hpp>
	
	
	using namespace cv;
	using namespace boost::python;
	
	
	
	static PyObject * mpyCreate(PyObject *self,  PyObject *args)
	{
	    char *engine_path = NULL;
	    void *trt_engine = NULL;
	
	    if (!PyArg_ParseTuple(args, "s", &engine_path)){
	        return  Py_BuildValue("K", (unsigned long long)trt_engine);
	    }
	
	    trt_engine = yolov5_trt_create(engine_path);
	
	    printf("create yolov5-trt , instance = %p\n", trt_engine);
	
	    return Py_BuildValue("K", (unsigned long long)trt_engine);
	}
	
	static PyObject *mpyDetect(PyObject *self, PyObject *args)
	{
	    void *trt_engine = NULL;
	    PyObject *ndArray = NULL;
	    float conf_thresh = 0.45;
	    const char *ret = NULL;
	    unsigned long long v; 
	
	    if (!PyArg_ParseTuple(args, "KOf", &v, &ndArray, &conf_thresh))
	        return Py_BuildValue("s", "");
	
	    Mat mat = pbcvt::fromNDArrayToMat(ndArray);
	
	    trt_engine = (void *)v;
	
	    ret = yolov5_trt_detect(trt_engine, mat, conf_thresh);
	
	    return Py_BuildValue("s", ret);
	}
	
	static PyObject * mPyDestroy(PyObject *self, PyObject *args)
	{
	    void *engine = NULL;
	    unsigned long long v; 
	    if (!PyArg_ParseTuple(args, "K", &v))
	        return Py_BuildValue("O", NULL);;
	
	    printf(" destroy engine , engine = %lu\n", v);
		engine = (void *)v;
	
	    yolov5_trt_destroy(engine);
	
	    return Py_BuildValue("O", NULL);
	
	}
	
	static PyMethodDef TRTYolov5MeThods[] = {
	    {"create", mpyCreate, METH_VARARGS, "Create the engine."},
	    {"detect", mpyDetect, METH_VARARGS, "use the engine to detect image"},    
	    {"destroy", mPyDestroy, METH_VARARGS, "destroy the engine"},        
	    {NULL, NULL, 0, NULL}
	};
	
	static struct PyModuleDef TRTYolov5Module = {
	    PyModuleDef_HEAD_INIT,
	    "TRTYolov5",     /* name of module */
	    "",          /* module documentation, may be NULL */
	    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
	    TRTYolov5MeThods
	};
	
	PyMODINIT_FUNC PyInit_TRTYolov5(void) {
	    printf("init module ... \n");
	
	    return PyModule_Create(&TRTYolov5Module);
	}

setup.py

	from setuptools import setup, Extension, find_packages
	import distutils.command.clean
	from torch.utils.cpp_extension import BuildExtension
	import numpy as np
	setup(
	    name='TRTYolov5',
	    version='1.0',
	    author="lindsay",
	    author_email="lindsayshuo@foxmail.com",
	    url="lindsayshuo@foxmail.com",
	    description='Python Package with Hello World C++ Extension',
	
	    # Package info
	    packages=find_packages(exclude=('test',)),
	    zip_safe=False,
	
	    ext_modules=[
	        Extension(
	            'TRTYolov5',
	            sources=['pyboostcvconverter/pyboost_cv4_converter.cpp', 'yolov5_trt_py_module.cpp'],
	            include_dirs=['/home/lindsay/anaconda3/lib/python3.8/site-packages/numpy/core/include/numpy',
	                          '/usr/local/cuda-11.1/include/',
	                          '/usr/local/include/',
	                          '../include'
	                         ],
	            libraries=['gstvideo-1.0', 'yolov5_trt',  'opencv_features2d', 'opencv_flann',  'opencv_imgcodecs', 'opencv_imgproc', 'opencv_core', 'opencv_highgui', 'opencv_videoio',   "boost_python3"],
	            # libraries=['gstvideo-1.0', 'yolov5_trt',  'opencv_features2d',  'opencv_core', 'opencv_highgui',  "boost_python3"],
	            library_dirs=[ '../build','/home/lindsay/anaconda3/lib'],
	            py_limited_api=True)
	    ],
	    include_dirs=[np.get_include()]
	)
	
	


 执行以下脚本：
     
     	sudo vim ~/.bashrc
	     	最后一行加入：
	    	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lindsay/Desktop/yolov5_tensorrtx_python-master/build
	    	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64
 		cd ../python/  
 		python setup.py install   #联合编译为python库
 		python detect.py  #验证

如果在make algorithm时报错，在/usr/lib/x86_64-linux-gnu 下也没有看到文件 ，如果没有装 libboost-python，则执行以下命令：
	
	sudo apt-get install libboost-python-dev ，
执行完，路径里面就有了libboost_python-py27.so.1.58.0，问题解决，make通过

## 将python包封装为pip库

 
 	python setup.py bdist_wheel
     

 
得到以下输出：

	running bdist_wheel
	running build
	running build_ext
	installing to build/bdist.linux-x86_64/wheel
	running install
	running install_lib
	creating build/bdist.linux-x86_64/wheel
	copying build/lib.linux-x86_64-3.8/TRTYolov5.abi3.so -> build/bdist.linux-x86_64/wheel
	running install_egg_info
	running egg_info
	writing TRTYolov5.egg-info/PKG-INFO
	writing dependency_links to TRTYolov5.egg-info/dependency_links.txt
	writing top-level names to TRTYolov5.egg-info/top_level.txt
	reading manifest file 'TRTYolov5.egg-info/SOURCES.txt'
	writing manifest file 'TRTYolov5.egg-info/SOURCES.txt'
	Copying TRTYolov5.egg-info to build/bdist.linux-x86_64/wheel/TRTYolov5-1.0-py3.8.egg-info
	running install_scripts
	creating build/bdist.linux-x86_64/wheel/TRTYolov5-1.0.dist-info/WHEEL
	creating 'dist/TRTYolov5-1.0-cp38-cp38-linux_x86_64.whl' and adding 'build/bdist.linux-x86_64/wheel' to it
	adding 'TRTYolov5.abi3.so'
	adding 'TRTYolov5-1.0.dist-info/METADATA'
	adding 'TRTYolov5-1.0.dist-info/WHEEL'
	adding 'TRTYolov5-1.0.dist-info/top_level.txt'
	adding 'TRTYolov5-1.0.dist-info/RECORD'
	removing build/bdist.linux-x86_64/wheel

包生成在dist文件夹下，后续迁移时可直接pip安装即可。
