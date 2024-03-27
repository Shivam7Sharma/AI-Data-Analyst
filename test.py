import cv2
try:
    empty_gpu_mat = cv2.cuda_GpuMat()
    print("CUDA support is enabled in OpenCV")
except AttributeError:
    print("CUDA support is not available in OpenCV")
