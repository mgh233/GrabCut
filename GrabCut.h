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
#include <cmath>

using namespace std;
using namespace cv;

typedef Graph<int, int, int> GraphType;

/*
 * GrabCut算法类
 * * * * * * * * * * * private * * * * * * * * * *
 *
 * GMMs: 高斯混合模型列表，包含背景【0】和目标【1】两个高斯混合模型
 * img: 输入的图像
 * alpha_matrix: 与图像相同大小的alpha矩阵，表示每个点的alpha值
 * k_matrix: 与图像相同大小的k矩阵，表示每个点的k值
 * beta, gamma: 用于计算V公式的参数
 * left, leftup, up, rightup: 分别用于存储图中一点与左方、左上、上方和右上方点的二阶范数
 * g: 图，用于step3的最小割计算
 * isGraphSet: 图是否被设置
 * E: 能量函数
 *
 * * * * * * * * * * * public * * * * * * * * * * *
 *
 * init(Mat, vector<int>): 使用img和矩形位置初始化
 * step1~3(): 迭代过程中的第一到第三步
 * iterative_process(int): 迭代过程
 * get_mask(): 获取掩码
 * revise(vector<>, vector<>): 根据用户反馈结果重新执行step3
 * cal_E(): 计算能量函数
 * get_E(): 获取能量函数
 *
 */
class GrabCut {

private:
    vector<GMM> GMMs;
    Mat img;
    int x1, y1, x2, y2;
    vector<vector<unsigned char>> k_matrix;
    Mat mask;
    float beta;
    float gamma = 50;
    vector<vector<float>> left, leftup, up, rightup;
    GraphType *g;
    bool isGraphSet;
    long double E;

public:
    void init(Mat img, Rect rect, int k);
    void step1();
    void step2();
    void step3(bool isRevise);
    void iterative_process();
    void revise(vector<Point> background_pixels, vector<Point> foreground_pixels);
    Mat get_mask();
    void cal_E();
    long double get_E();

};


#endif //GRABCUT_GRABCUT_H
