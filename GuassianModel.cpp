//
// Created by 母国宏 on 2023/4/23.
//

#include "GuassianModel.h"
#include <iostream>

const float PI = acos(-1);

float CarmSqrt(float x){
    union{
        int intPart;
        float floatPart;
    } convertor;
    union{
        int intPart;
        float floatPart;
    } convertor2;
    convertor.floatPart = x;
    convertor2.floatPart = x;
    convertor.intPart = 0x1FBCF800 + (convertor.intPart >> 1);
    convertor2.intPart = 0x5f3759df - (convertor2.intPart >> 1);
    return 0.5f*(convertor.floatPart + (x * convertor2.floatPart));
}

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
    // cout << diff[0] << diff[1] << diff[2] << endl;
    // Mat diff_mat = Mat::zeros(1, 3, CV_32FC1);
    Mat diff_mat = cv::Mat(diff).reshape(0, 1);
    // reverse.convertTo(reverse, DataType<float>::type);
//    for (int i = 0; i < 3; i ++) {
//        diff_mat.at<float>(i, 0) = diff[i];
//    }
    Mat res = diff_mat * covariance_inv * diff_mat.t();
    float y = res.at<float>(0, 0);
    return 1.0 / (2 * PI * sqrt(covariance_value)) * exp(-0.5 * y);
}

