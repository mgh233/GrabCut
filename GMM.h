//
// Created by 母国宏 on 2023/4/22.
//

#ifndef GRABCUT_GMM_H
#define GRABCUT_GMM_H
#include <vector>
#include "GuassianModel.h"
#include "opencv2/core.hpp"
#include "opencv2/core/mat.hpp"

using namespace std;

/*
 * 高斯混合模型
 * * * * * * * * private * * * * * * * * *
 *
 * k: 高斯混合模型中component的数量
 * models: 高斯模型数组
 * weights: 高斯模型的权重
 *
 * * * * * * * * public  * * * * * * * * *
 *
 * GMM(int, vector<vector<int>>): 通过k的大小和点初始化高斯模型
 * get_weight(int): 得到第k + 1个高斯模型的权重
 * get_mean(int): 得到第k + 1个高斯模型的均值向量
 * get_covariance(int): 得到第k + 1个高斯模型的协方差矩阵
 * update_all(vector<vector<vector<int>>>): 更新高斯混合模型
 * get_prob(vector<>): 一个点在该高斯混合分布中的概率
 * get_most_prob(vector<>): 一个点在高斯混合分布的哪个部分中概率最高
 *
 */
class GMM {

private:
    int k;
    vector<GaussianModel> models;
    vector<float> weights;

public:
    GMM(int k, vector<vector<unsigned char>> pixels);
    float get_weight(int k);
    vector<float> get_mean(int k);
    vector<vector<float>> get_covariance(int k);
    void update_all(vector<vector<vector<unsigned char>>> pixels, int sum);
    float get_prob(vector<unsigned char> pixel);
    int get_most_prob(vector<unsigned char> pixel);
    GaussianModel get_model(int k);

};


#endif //GRABCUT_GMM_H
