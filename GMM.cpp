//
// Created by 母国宏 on 2023/4/22.
//

#include "GMM.h"

GMM::GMM(int k, vector<vector<unsigned char>> pixels) {

    this->k = k;

    // 初始化weights全部为1/k
    this->weights = vector<float>(k, 1.0 / k);

    // 初始化模型，每个使用1/k的数据进行初始化
    int sub_size = pixels.size() / k;
    for (int i = 0; i < k; i ++) {
        auto sub_pixels = vector<vector<unsigned char>>(pixels.begin() + i * sub_size,
                pixels.begin() + (i + 1) * sub_size);
        auto model = GaussianModel(sub_pixels);
        this->models.push_back(model);
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