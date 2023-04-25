//
// Created by 母国宏 on 2023/4/23.
//

#include "util.h"

float util::det(vector<vector<float>> determinant) {

    float sum = 0, n = determinant.size();
    if (n == 2) {
        return determinant[0][0] * determinant[1][1] - determinant[0][1] * determinant[1][0];
    }
    for (int k = 0; k < n; k ++){
        vector<vector<float>> b;
        for (int i = 1; i < n; i ++) {
            vector<float> c;
            for (int j = 0; j < n; j ++){
                if(j == k) continue;
                c.push_back(determinant[i][j]);
            }
            b.push_back(c);
        }
        sum = k % 2 == 0 ? sum + determinant[0][k] * det(b) : sum - determinant[0][k] * det(b);
    }
    return sum;

}

float util::matrix_mul(vector<float> matrix_1, vector<float> matrix_2) {

    if (matrix_1.size() != matrix_2.size()) assert("util.matrix_mul--size not equal!");
    int n = matrix_1.size();
    float res = 0;
    for (int i = 0; i < n; i ++) {
        res += matrix_1[i] * matrix_2[i];
    }
    return res;
}

vector<float> util::matrix_mul(vector<float> matrix_1, vector<vector<float>> matrix_2) {

    if (matrix_1.size() != matrix_2.size()) assert("util.matrix_mul--size not equal!");
    int n = matrix_1.size();
    vector<float> res;
    for (int i = 0; i < matrix_2[0].size(); i ++) {
        float sum = 0;
        for (int j = 0; j < n; j ++) {
            sum += matrix_1[j] * matrix_2[j][i];
        }
        res.push_back(sum);
    }
    return res;
}
