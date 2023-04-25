//
// Created by 母国宏 on 2023/4/22.
//

#include "GrabCut.h"
#include <time.h>

typedef Graph<int, int, int> GraphType;

GrabCut::GrabCut(Mat img, vector<int> pos, int k) {

    time_t time1 = time(0);
    char* dt1 = ctime(&time1);
    cout << "GrabCut starts initial...\t" << dt1 << endl;

    this->img = img;

    // 初始化mask
    this->mask = Mat::zeros(img.size(), CV_8UC1);

    // 初始化alpha矩阵和k矩阵
    int cols = img.cols;
    int rows = img.rows;
    this->alpha_matrix = vector<vector<unsigned char>>(rows, vector<unsigned char>(cols, 0));
    this->k_matrix = vector<vector<unsigned char>>(rows, vector<unsigned char>(cols, 0));

    // 根据矩形坐标初始化alpha矩阵
    this->x1 = pos[0];
    this->y1 = pos[1];
    this->x2 = pos[2];
    this->y2 = pos[3];
    for (int i = this->x1; i <= this->x2; i ++) {
        for (int j = this->y1; j <= this->y2; j ++) {
            this->alpha_matrix[j][i] = 1;
        }
    }

    // 根据alpha矩阵初始化GMM模型
    vector<vector<unsigned char>> backgroud_pixels;
    vector<vector<unsigned char>> foreground_pixels;
    for (int i = 0; i < rows; i ++) {
        for (int j = 0; j < cols; j ++) {
            Vec3b pixel = img.at<Vec3b>(j, i);
            if (this->alpha_matrix[i][j] == 0) {
                backgroud_pixels.push_back({pixel[0], pixel[1], pixel[2]});
            }
            else {
                foreground_pixels.push_back({pixel[0], pixel[1], pixel[2]});
            }
        }
    }
    this->GMMs.push_back(GMM(k, backgroud_pixels));
    this->GMMs.push_back(GMM(k, foreground_pixels));

    // 初始化left, leftup, up, rightup
    this->left = vector<vector<float>>(img.rows, vector<float>(img.cols, 0));
    this->leftup = vector<vector<float>>(img.rows, vector<float>(img.cols, 0));
    this->up = vector<vector<float>>(img.rows, vector<float>(img.cols, 0));
    this->rightup = vector<vector<float>>(img.rows, vector<float>(img.cols, 0));

    // 计算beta
    float sum = 0;
    int cnt = 0;
    for (int i = 0; i < img.rows; i ++) {
        for (int j = 0; j < img.cols; j ++) {
            auto now_pixel = img.at<Vec3b>(j, i);
            // 与左边
            if (j - 1 >= 0) {
                auto left_pixel = img.at<Vec3b>(j - 1, i);
                left[i][j] = pow(now_pixel[0] - left_pixel[0], 2)
                        + pow(now_pixel[1] - left_pixel[1], 2)
                        + pow(now_pixel[2] - left_pixel[2], 2);
                sum += left[i][j];
                cnt ++;
            }
            // 与左上
            if (j - 1 >= 0 && i - 1 >= 0) {
                auto leftup_pixel = img.at<Vec3b>(j - 1, i - 1);
                leftup[i][j] = pow(now_pixel[0] - leftup_pixel[0], 2)
                       + pow(now_pixel[1] - leftup_pixel[1], 2)
                       + pow(now_pixel[2] - leftup_pixel[2], 2);
                sum += leftup[i][j];
                cnt ++;
            }
            // 与上方
            if (i - 1 >= 0) {
                auto up_pixel = img.at<Vec3b>(j, i - 1);
                up[i][j] = pow(now_pixel[0] - up_pixel[0], 2)
                       + pow(now_pixel[1] - up_pixel[1], 2)
                       + pow(now_pixel[2] - up_pixel[2], 2);
                sum += up[i][j];
                cnt ++;
            }
            // 与右上
            if (j + 1 < img.cols && i - 1 >= 0) {
                auto rightup_pixel = img.at<Vec3b>(j + 1, i - 1);
                rightup[i][j] = pow(now_pixel[0] - rightup_pixel[0], 2)
                       + pow(now_pixel[1] - rightup_pixel[1], 2)
                       + pow(now_pixel[2] - rightup_pixel[2], 2);
                sum += rightup[i][j];
                cnt ++;
            }
        }
    }
    this->beta = 1.0 / (2 * sum / cnt);

    time_t time2 = time(0);
    char* dt2 = ctime(&time2);
    cout << "GrabCut finish initial.\t" << dt2 << endl;

}

//float GrabCut::Dn(unsigned char alpha, unsigned char k, vector<float> bgr) {
//
//    GMM model = this->GMMs[alpha];
//    float weight = model.get_weight((int)k);
//    vector<float> mean = model.get_mean((int)k);
//    vector<vector<float>> covariance = model.get_covariance((int)k);
//
//    float det_covariance = tool.det(covariance);
//    for (int i = 0; i < bgr.size(); i ++) {
//        bgr[i] -= mean[i];
//    }
//
//    return -log((double)weight) + 1.0 / 2 * log(det_covariance)
//            + 1.0 / 2 * tool.matrix_mul(tool.matrix_mul(bgr, covariance), bgr);
//}

void GrabCut::step1() {

    time_t time1 = time(0);
    char* dt1 = ctime(&time1);
    cout << "GrabCut starts step1...\t" << dt1 << endl;

    int cols = this->img.cols;
    int rows = this->img.rows;

    for (int i = 0; i < rows; i ++) {
        for (int j = 0; j < cols; j ++) {
            Vec3b pixel = this->img.at<Vec3b>(j, i);
            this->k_matrix[i][j] = GMMs[alpha_matrix[i][j]].get_most_prob({pixel[0], pixel[1], pixel[2]});
        }
    }

    time_t time2 = time(0);
    char* dt2 = ctime(&time2);
    cout << "GrabCut finish step1.\t" << dt2 << endl;
}

void GrabCut::step2() {

    time_t time1 = time(0);
    char* dt1 = ctime(&time1);
    cout << "GrabCut starts step2...\t" << dt1 << endl;

    int sum_0 = 0, sum_1 = 0;
    auto background_pixels = vector<vector<vector<unsigned char>>>(5, vector<vector<unsigned char>>());
    auto foreground_pixels = vector<vector<vector<unsigned char>>>(5, vector<vector<unsigned char>>());
    for (int i = 0; i < img.rows; i ++) {
        for (int j = 0; j < img.cols; j ++) {
            auto pixel = img.at<Vec3b>(j, i);
            auto alpha = this->alpha_matrix[i][j];
            auto k = this->k_matrix[i][j];
            if (alpha == 0) {
                background_pixels[k].push_back({pixel[0], pixel[1], pixel[2]});
                sum_0 ++;
            }
            else {
                foreground_pixels[k].push_back({pixel[0], pixel[1], pixel[2]});
                sum_1 ++;
            }
        }
    }

    // 更新GMM
    this->GMMs[0].update_all(background_pixels, sum_0);
    this->GMMs[1].update_all(foreground_pixels, sum_1);

    time_t time2 = time(0);
    char* dt2 = ctime(&time2);
    cout << "GrabCut finish step2.\t" << dt2 << endl;

}

void GrabCut::step3() {

    time_t time1 = time(0);
    char* dt1 = ctime(&time1);
    cout << "GrabCut starts step3...\t" << dt1 << endl;

    int sum = (this->x2 - this->x1 + 1) * (this->y2 - this->y1 + 1);
    GraphType *g = new GraphType(sum, 8 * sum);
    for (int i = 0; i < img.rows; i ++) {
        for (int j = 0; j < img.cols; j ++) {
            // 向图中加入节点并设置t-link
            g->add_node();
            Vec3b pixel = img.at<Vec3b>(j, i);
            // int id = (i - x1) * (y2 - y1 + 1) + j - y1;
            int id = i * img.cols + j;
            // float x = -log(GMMs[0].get_prob({pixel[0], pixel[1], pixel[2]}));
            // float y = -log(GMMs[1].get_prob({pixel[0], pixel[1], pixel[2]}));
            g->add_tweights(id,
                            -log(GMMs[0].get_prob({pixel[0], pixel[1], pixel[2]})),
                            -log(GMMs[1].get_prob({pixel[0], pixel[1], pixel[2]})));

            // 设置n-link
            if (left[i][j] > 0) {
                float x = gamma * exp(-beta * left[i][j]);
                g->add_edge(id, id - 1, gamma * exp(-beta * left[i][j]),
                            gamma * exp(-beta * left[i][j]));
            }
            if (leftup[i][j] > 0) {
                g->add_edge(id, id - img.cols - 1, gamma * exp(-beta * leftup[i][j]),
                            gamma * exp(-beta * leftup[i][j]));
            }
            if (up[i][j] > 0) {
                g->add_edge(id, id - img.cols, gamma * exp(-beta * up[i][j]),
                            gamma * exp(-beta * up[i][j]));
            }
            if (rightup[i][j] > 0) {
                g->add_edge(id, id - img.cols + 1, gamma * exp(-beta * rightup[i][j]),
                            gamma * exp(-beta * rightup[i][j]));
            }
        }
    }

    float flow = g->maxflow();

    for (int i = y1; i <= y2; i ++) {
        for (int j = x1; j <= x2; j ++) {
            int id = i * img.cols + j;
            if (g->what_segment(id) == GraphType::SOURCE) {
                alpha_matrix[i][j] = 0;
            }
            else {
                alpha_matrix[i][j] = 1;
            }
        }
    }

    time_t time2 = time(0);
    char* dt2 = ctime(&time2);
    cout << "GrabCut finish step3.\t" << dt2 << endl;

}

void GrabCut::iterative_process(int max_iteration) {

    for (int i = 0; i < max_iteration; i ++) {
        step1();
        step2();
        step3();
    }
}

Mat GrabCut::get_mask() {

//    for (int i = 0; i < img.cols; i ++) {
//        for (int j = 0; j < img.rows; j ++) {
//            if (alpha_matrix[j][i])
//                mask.at<uchar>(i, j) = 255;
//        }
//    }
    Mat temp(0, alpha_matrix[0].size(), DataType<uchar>::type);
    for (int i = 0; i < alpha_matrix.size(); i ++) {
        Mat sample(1, alpha_matrix[0].size(), DataType<uchar>::type, alpha_matrix[i].data());
        temp.push_back(sample);
    }
    return temp;
}
