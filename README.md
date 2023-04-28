# GrabCut
A C++ implementation of GrabCut Algorithm with OpenCV.

You can just download it and build it with CMake. After build, run the .exe file in your 
cmake-build-debug/cmake-build-release directory.

The GUI function was inherited from grabcut sample from OpenCV.
See it in [this site](https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html).

Usage:
1. `git clone git@github.com:mgh233/GrabCut.git`
2. `cd GrabCut`
3. `mkdir build && cd build`
4. `cmake -DCMAKE_BUILD_TYPE=Release ..`
5. `make`
6. `./GrabCut`

Demo:
![demo](pic/image.png)