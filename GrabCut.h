//
// Created by 母国宏 on 2023/4/22.
//

#ifndef GRABCUT_GRABCUT_H
#define GRABCUT_GRABCUT_H
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <vector>
#include "GMM.h"
#include "graph.h"
#include "util.h"

using namespace std;
using namespace cv;

/*
 * GrabCut算法类
 * * * * * * * * * * * private * * * * * * * * * *
 *
 * GMMs: 高斯混合模型列表，包含背景【0】和目标【1】两个高斯混合模型
 * img: 输入的图像
 * alpha_matrix: 与图像相同大小的alpha矩阵，表示每个点的alpha值
 * k_matrix: 与图像相同大小的k矩阵，表示每个点的k值
 * util: 工具类
 * beta, gamma: 用于计算V公式的参数
 * left, leftup, up, rightup: 分别用于存储图中一点与左方、左上、上方和右上方点的二阶范数
 *
 * * * * * * * * * * * public * * * * * * * * * * *
 *
 * init(Mat, vector<int>): 使用img和矩形位置初始化
 * Dn(uc, uc, vector<uc>): Dn函数的计算
 * step1~3(): 迭代过程中的第一到第三步
 * iterative_process(int): 迭代过程
 * get_mask(): 获取掩码
 * revise(vector<>, vector<>): 根据用户反馈结果重新执行step3
 *
 */
class GrabCut {

private:
    vector<GMM> GMMs;
    Mat img;
    int x1, y1, x2, y2;
    // vector<vector<unsigned char>> alpha_matrix;
    vector<vector<unsigned char>> k_matrix;
    Mat mask;
    util tool;
    float beta;
    float gamma = 50;
    vector<vector<float>> left, leftup, up, rightup;

public:
    void init(Mat img, Rect rect, int k);
    // float Dn(unsigned char alpha, unsigned char k, vector<float> bgr);
    void step1();
    void step2();
    void step3();
    void iterative_process();
    void revise(vector<Point> background_pixels, vector<Point> foreground_pixels);
    Mat get_mask();

};


#endif //GRABCUT_GRABCUT_H
