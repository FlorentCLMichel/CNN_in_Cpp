A simple convolutional neural network written in C++, using the boost::python library to make it accessible through Python. 

(Work in progress)

This implementation is rather slow compared with TensorFLow or PyTorch versions, and lacks modern features such as batch normalization, network shortcuts and learning rate optimizer. It is intended at showing how simple convolutional neural networks work on a (nearly) minimal example rather than real-world applications. 

Requirements: 
    * a C++ compiler (tested with gcc 7.5.0),
    * the boost library,
    * a Python 3 development environment (tested with Anaconda 4.8.5). 

The library CNN3.so can be compiled on Linux using the script make.sh.
