//
// Created by 母国宏 on 2023/4/23.
//

#include "GuassianModel.h"
#include "util.h"

GaussianModel::GaussianModel(vector<vector<unsigned char>> pixels) {

    this->v_mean = vector<float>(3, 0.0);
    // 计算BGR的均值
    for (auto pixel: pixels) {
        this->v_mean[0] += pixel[0];
        this->v_mean[1] += pixel[1];
        this->v_mean[2] += pixel[2];
    }
    this->v_mean[0] /= pixels.size();
    this->v_mean[1] /= pixels.size();
    this->v_mean[2] /= pixels.size();

    this->covariance = vector<vector<float>>(3, vector<float>(3, 0.0));
    // 计算BGR的协方差矩阵
    for (auto pixel: pixels) {
        for (int i = 0; i < 3; i ++) {
            for (int j = 0; j < 3; j ++) {
                this->covariance[i][j] += (pixel[i] - this->v_mean[i]) * (pixel[j] - this->v_mean[j]);
            }
        }
    }
    for (int i = 0; i < 3; i ++) {
        for (int j = 0; j < 3; j ++) {
            this->covariance[i][j] /= pixels.size() - 1;
        }
    }
}

vector<vector<float>> GaussianModel::get_covariance() {

    return this->covariance;
}

vector<float> GaussianModel::get_mean() {

    return this->v_mean;
}

void GaussianModel::update(vector<vector<unsigned char>> pixels) {

    this->v_mean = vector<float>(3, 0.0);
    // 计算BGR的均值
    for (auto pixel: pixels) {
        this->v_mean[0] += pixel[0];
        this->v_mean[1] += pixel[1];
        this->v_mean[2] += pixel[2];
    }
    this->v_mean[0] /= pixels.size();
    this->v_mean[1] /= pixels.size();
    this->v_mean[2] /= pixels.size();

    this->covariance = vector<vector<float>>(3, vector<float>(3, 0.0));
    // 计算BGR的协方差矩阵
    for (auto pixel: pixels) {
        for (int i = 0; i < 3; i ++) {
            for (int j = 0; j < 3; j ++) {
                this->covariance[i][j] += (pixel[i] - this->v_mean[i]) * (pixel[j] - this->v_mean[j]);
            }
        }
    }
    for (int i = 0; i < 3; i ++) {
        for (int j = 0; j < 3; j ++) {
            this->covariance[i][j] /= pixels.size() - 1;
        }
    }
}

float GaussianModel::get_prob(vector<unsigned char> pixel) {

    auto tool = util();
    vector<float> diff = {pixel[0] - v_mean[0], pixel[1] - v_mean[1], pixel[2] - v_mean[2]};
    return 1.0 / (2 * acos(-1) * sqrt(tool.det(covariance)))
        * exp(-1.0 / 2 * tool.matrix_mul(tool.matrix_mul(diff, covariance), diff));
}

