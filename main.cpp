#include <iostream>
#include <opencv2/opencv.hpp>
#include "GrabCut.h"


using namespace std;
using namespace cv;

int main() {

    Mat srcImage = imread("/Users/muguohong/Documents/资料/南开/GrabCut/cat.jpg", 1);
    if (!srcImage.data) {
        std::cout << "Image not loaded";
        return -1;
    }
    auto graphcut = GrabCut(srcImage, {110, 50, 550, 337}, 5);
    graphcut.iterative_process(5);
    Mat mask = graphcut.get_mask();
    Mat res;
    srcImage.copyTo(res, mask);
    imshow("result", res);
    waitKey(0);
    return 0;
}