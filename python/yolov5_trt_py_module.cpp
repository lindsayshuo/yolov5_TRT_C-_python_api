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



