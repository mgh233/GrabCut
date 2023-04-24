//
// Created by 母国宏 on 2023/4/23.
//

#ifndef GRABCUT_UTIL_H
#define GRABCUT_UTIL_H
#include <vector>

using namespace std;

/*
 * 工具类，提供一些常用的计算工具
 *
 * det: 计算行列式
 * matrix_mul: 计算矩阵间乘法
 *
 */
class util {

public:
    float det(vector<vector<float>> determinant);
    vector<float> matrix_mul(vector<float> matrix_1, vector<vector<float>> matrix_2);
    float matrix_mul(vector<float> matrix_1, vector<float> matrix_2);

};


#endif //GRABCUT_UTIL_H
