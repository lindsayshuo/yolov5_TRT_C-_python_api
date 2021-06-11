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
    #data_files=[('./', ['libRetinaface.so'])],
    #cmdclass={
    #    'build_ext': BuildExtension.with_options(no_python_abi_suffix=True),
    #}


)


#        ${CUDA_LIBRARIES}
#        ${CUDNN_LIBRARY}
#        ${TENSORRT_LIBRARY}
#        tensorrt_engine_lib
#        ${OpenCV_LIBS}
#        thor
