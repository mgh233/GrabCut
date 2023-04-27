//
// Created by 母国宏 on 2023/4/23.
//

#include "GuassianModel.h"
#include <iostream>

const float PI = acos(-1);

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

    this->covariance = Mat(3, 3, CV_32FC1);
    // 计算BGR的协方差矩阵
    for (auto pixel: pixels) {
        for (int i = 0; i < 3; i ++) {
            for (int j = 0; j < 3; j ++) {
                this->covariance.at<float>(j, i) += (pixel[i] - this->v_mean[i]) * (pixel[j] - this->v_mean[j]);
            }
        }
    }
    for (int i = 0; i < 3; i ++) {
        for (int j = 0; j < 3; j ++) {
            this->covariance.at<float>(j ,i) /= pixels.size() - 1;
        }
    }

    covariance_value = determinant(covariance);
    covariance_inv = covariance.inv();
}

Mat GaussianModel::get_covariance() {

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

    this->covariance = Mat(3, 3, CV_32FC1);
    // 计算BGR的协方差矩阵
    for (auto pixel: pixels) {
        for (int i = 0; i < 3; i ++) {
            for (int j = 0; j < 3; j ++) {
                this->covariance.at<float>(j, i) += (pixel[i] - this->v_mean[i]) * (pixel[j] - this->v_mean[j]);
            }
        }
    }
    for (int i = 0; i < 3; i ++) {
        for (int j = 0; j < 3; j ++) {
            this->covariance.at<float>(j ,i) /= pixels.size() - 1;
        }
    }
}

float GaussianModel::get_prob(vector<unsigned char> pixel) {

    vector<float> diff = {pixel[0] - v_mean[0], pixel[1] - v_mean[1], pixel[2] - v_mean[2]};
//    Mat diff_mat = cv::Mat(diff).reshape(0, 1);
//    Mat res = diff_mat * covariance_inv * diff_mat.t();
//    float y = res.at<float>(0, 0);
    float y = diff[0] * (diff[0] * covariance_inv.at<float>(0, 0) + diff[1] * covariance_inv.at<float>(1, 0)
            + diff[2] * covariance_inv.at<float>(2, 0)) + diff[1] * (diff[0] * covariance_inv.at<float>(0, 1)
            + diff[1] * covariance_inv.at<float>(1, 1) + diff[2] * covariance_inv.at<float>(2, 1))
            + diff[2] * (diff[0] * covariance_inv.at<float>(0, 2)
            + diff[1] * covariance_inv.at<float>(1, 2) + diff[2] * covariance_inv.at<float>(2, 2));
    return 1.0 / (2 * PI * sqrt(covariance_value)) * exp(-0.5 * y);
}

