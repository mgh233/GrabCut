//
// Created by 母国宏 on 2023/4/22.
//

#include "GMM.h"
#include <cstdlib>
#include <iostream>


GMM::GMM(int k, vector<vector<unsigned char>> pixels) {

    this->k = k;

    // 使用K-means算法初始化
    // 使用随机数初始化k个聚类中心
    vector<vector<float>> centers(k);
    srand((unsigned) time(NULL));
    for (int i = 0; i < k; i ++) {
        int x = rand() % pixels.size();
        while (!(pixels[x][0] && pixels[x][1] && pixels[x][2])) {
            x = rand() % pixels.size();
        }
        centers[i] = {static_cast<float>((pixels[x][0])), static_cast<float>((pixels[x][1])), static_cast<float>((pixels[x][2]))};
    }
    // 通过欧式距离聚类
    unsigned char flags[pixels.size()];
    double sum = 0;
    int times = 0;
    do {
        sum = 0;
        for (int i = 0; i < pixels.size(); i ++) {
            int center = 0;
            int min = 1e9;
            for (int j = 0; j < k; j ++) {
                int dis = pow(pixels[i][0] - centers[j][0], 2)
                          + pow(pixels[i][1] - centers[j][1], 2)
                          + pow(pixels[i][2] - centers[j][2], 2);
                if (min > dis) {
                    min = dis;
                    center = j;
                }
            }
            sum += min;
            flags[i] = center;
        }
        for (int i = 0; i < k; i ++) {
            centers[i] = {0, 0, 0};
        }
        vector<int> cnts(5, 0);
        for (int i = 0; i < pixels.size(); i ++) {
            centers[flags[i]][0] += pixels[i][0];
            centers[flags[i]][1] += pixels[i][1];
            centers[flags[i]][2] += pixels[i][2];
            cnts[flags[i]] ++;
        }
        for (int i = 0; i < k; i ++) {
            centers[i][0] /= cnts[i];
            centers[i][1] /= cnts[i];
            centers[i][2] /= cnts[i];
        }
        times ++;
    } while (sum / pixels.size() > 1000 && times < 10);

    auto after_pixels = vector<vector<vector<unsigned char>>>(k, vector<vector<unsigned char>>());
    for (int i = 0; i < pixels.size(); i ++) {
        after_pixels[flags[i]].push_back(pixels[i]);
    }

    for (int i = 0; i < k; i ++) {
        this->models.push_back(GaussianModel(after_pixels[i]));
        this->weights.push_back((float)after_pixels[i].size() / pixels.size());
    }


}

float GMM::get_weight(int k) {

    return this->weights[k];
}

vector<float> GMM::get_mean(int k) {

    return this->models[k].get_mean();
}

vector<vector<float>> GMM::get_covariance(int k) {

    return this->models[k].get_covariance();
}

void GMM::update_all(vector<vector<vector<unsigned char>>> pixels, int sum) {

    int i = 0;
    for (auto sub_pixels: pixels) {
        this->models[i].update(sub_pixels);
        this->weights[i] = (float) sub_pixels.size() / sum;
        i ++;
    }
}

float GMM::get_prob(vector<unsigned char> pixel) {

    float prob = 0.0;
    for (int i = 0; i < k; i ++) {
        prob += weights[i] * models[i].get_prob(pixel);
    }
    return prob;
}

int GMM::get_most_prob(vector<unsigned char> pixel) {

    int res = 0;
    float prob = 0.0;
    for (int i = 0; i < k; i ++) {
        float now = models[i].get_prob(pixel);
        if (now > prob) {
            res = i;
            prob = now;
        }
    }
    return res;
}