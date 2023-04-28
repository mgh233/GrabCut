//
// Created by 母国宏 on 2023/4/23.
//

#ifndef GRABCUT_GUASSIANMODEL_H
#define GRABCUT_GUASSIANMODEL_H
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/core/mat.hpp"
#include <cmath>
#include <iostream>

using namespace std;
using namespace cv;

/*
 * 三维高斯分布模型
 * * * * * * * private * * * * * * * * * * * * *
 *
 * v_mean: 高斯模型的均值向量，v_mean.size() = 3
 * covariance: 协方差，size = [3,3]
 * tool: 工具类
 *
 * * * * * * * public  * * * * * * * * * * * * *
 *
 * GaussianModel(vector<float>): 构造函数，接受一个数组，初始化高斯分布的均值和协方差
 * get_mean(): 返回均值数组
 * get_covariance(): 返回协方差矩阵
 * update(vector<float>): 接受一个数组，更新高斯分布的参数
 * get_prob(vector<>): 一个点在该高斯分布中的概率
 *
 */
class GaussianModel {

private:
    vector<float> v_mean;
    Mat covariance;
    float covariance_value;
    Mat covariance_inv;

public:
    explicit GaussianModel(vector<vector<unsigned char>> pixels);
    vector<float> get_mean();
    Mat get_covariance();
    void update(vector<vector<unsigned char>> pixels);
    float get_prob(vector<unsigned char> pixel);

};


#endif //GRABCUT_GUASSIANMODEL_H
