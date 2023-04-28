//
// Created by 母国宏 on 2023/4/22.
//

#include "GrabCut.h"
#include "GMM.h"
#include <sys/time.h>
#include <limits>
#include <fstream>

long getCurrentTime()
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

void GrabCut::init(Mat img, Rect rect, int k) {

    this->img = img;

    // 初始化mask
    this->mask = Mat::zeros(img.size(), CV_8UC1);

    // 初始化alpha矩阵和k矩阵
    int cols = img.cols;
    int rows = img.rows;
    // this->alpha_matrix = vector<vector<unsigned char>>(rows, vector<unsigned char>(cols, 0));
    this->k_matrix = vector<vector<unsigned char>>(rows, vector<unsigned char>(cols, 0));

    // 根据矩形坐标初始化alpha矩阵
    this->x1 = rect.tl().x;
    this->y1 = rect.tl().y;
    this->x2 = rect.br().x;
    this->y2 = rect.br().y;
    for (int i = this->x1; i <= this->x2; i ++) {
        for (int j = this->y1; j <= this->y2; j ++) {
            this->mask.at<uchar>(j, i) = 1;
        }
    }

    // 根据alpha矩阵初始化GMM模型
    vector<vector<unsigned char>> backgroud_pixels;
    vector<vector<unsigned char>> foreground_pixels;
    for (int i = 0; i < rows; i ++) {
        for (int j = 0; j < cols; j ++) {
            Vec3b pixel = img.at<Vec3b>(i, j);
            if (this->mask.at<uchar>(i, j) == 0) {
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
    for (int i = 0; i < img.rows; i ++) {
        for (int j = 0; j < img.cols; j ++) {
            auto now_pixel = img.at<Vec3b>(i, j);
            // 与左边
            if (j - 1 >= 0) {
                auto left_pixel = img.at<Vec3b>(i, j - 1);
                left[i][j] = pow(now_pixel[0] - left_pixel[0], 2)
                        + pow(now_pixel[1] - left_pixel[1], 2)
                        + pow(now_pixel[2] - left_pixel[2], 2);
                sum += left[i][j];
            }
            // 与左上
            if (j - 1 >= 0 && i - 1 >= 0) {
                auto leftup_pixel = img.at<Vec3b>(i - 1, j - 1);
                leftup[i][j] = pow(now_pixel[0] - leftup_pixel[0], 2)
                       + pow(now_pixel[1] - leftup_pixel[1], 2)
                       + pow(now_pixel[2] - leftup_pixel[2], 2);
                sum += leftup[i][j];
            }
            // 与上方
            if (i - 1 >= 0) {
                auto up_pixel = img.at<Vec3b>(i - 1, j);
                up[i][j] = pow(now_pixel[0] - up_pixel[0], 2)
                       + pow(now_pixel[1] - up_pixel[1], 2)
                       + pow(now_pixel[2] - up_pixel[2], 2);
                sum += up[i][j];
            }
            // 与右上
            if (j + 1 < img.cols && i - 1 >= 0) {
                auto rightup_pixel = img.at<Vec3b>(i - 1, j + 1);
                rightup[i][j] = pow(now_pixel[0] - rightup_pixel[0], 2)
                       + pow(now_pixel[1] - rightup_pixel[1], 2)
                       + pow(now_pixel[2] - rightup_pixel[2], 2);
                sum += rightup[i][j];
            }
        }
    }
    if( sum <= std::numeric_limits<double>::epsilon() )
        this->beta = 0;
    else this->beta = 1.0 / (2 * sum / (4 * img.cols * img.rows - 3 * img.cols - 3 * img.rows + 2));

    for (int i = 0; i < img.rows; i ++) {
        for (int j = 0; j < img.cols; j ++) {
            left[i][j] = gamma * exp(-beta * left[i][j]);
            up[i][j] = gamma * exp(-beta * up[i][j]);
            leftup[i][j] = gamma * exp(-beta * leftup[i][j]) / sqrt(2.0);
            rightup[i][j] = gamma * exp(-beta * rightup[i][j]) / sqrt(2.0);
        }
    }

    int all = img.cols * img.rows;
    g = new GraphType(all, 8 * all);
    E = 0;

}

void GrabCut::step1() {

    int cols = this->img.cols;
    int rows = this->img.rows;

    for (int i = 0; i < rows; i ++) {
        for (int j = 0; j < cols; j ++) {
            Vec3b pixel = this->img.at<Vec3b>(i, j);
            this->k_matrix[i][j] = GMMs[mask.at<uchar>(i, j) % 2].get_most_prob({pixel[0], pixel[1], pixel[2]});
        }
    }
}

void GrabCut::step2() {

    int sum_0 = 0, sum_1 = 0;
    auto background_pixels = vector<vector<vector<unsigned char>>>(5, vector<vector<unsigned char>>());
    auto foreground_pixels = vector<vector<vector<unsigned char>>>(5, vector<vector<unsigned char>>());
    for (int i = 0; i < img.rows; i ++) {
        for (int j = 0; j < img.cols; j ++) {
            auto pixel = img.at<Vec3b>(i, j);
            auto alpha = this->mask.at<uchar>(i, j) % 2;
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

}

void GrabCut::step3(bool isRevise) {

    int all = img.cols * img.rows;
    g = new GraphType(all, 8 * all);
    if (!isGraphSet) {
        int id = 0;
        for (int i = 0; i < img.rows; i ++) {
            for (int j = 0; j < img.cols; j ++) {
                // 向图中加入节点并设置t-link
                g->add_node();
                Vec3b pixel = img.at<Vec3b>(i, j);
                if (i < y1 || i > y2 || j < x1 || j > x2) {
                    g->add_tweights(id, 0, 9 * gamma);
                }
                    // alpha=2为用户设置的背景
                else if (mask.at<uchar>(i ,j) == 2) {
                    g->add_tweights(id, 0, 9 * gamma);
                }
                else if (mask.at<uchar>(i, j) == 3) {
                    g->add_tweights(id, 9 * gamma, 0);
                }
                else {
                    g->add_tweights(id,
                                    -log(GMMs[0].get_prob({pixel[0], pixel[1], pixel[2]})),
                                    -log(GMMs[1].get_prob({pixel[0], pixel[1], pixel[2]})));
                }

                // 设置n-link
                if (j - 1 >= 0) {
                    g->add_edge(id, id - 1, left[i][j],left[i][j]);
                }
                if (j - 1 >= 0 && i - 1 >= 0) {
                    g->add_edge(id, id - img.cols - 1, leftup[i][j],leftup[i][j]);
                }
                if (i - 1 >= 0) {
                    g->add_edge(id, id - img.cols, up[i][j],up[i][j]);
                }
                if (j + 1 < img.cols && i - 1 >= 0) {
                    g->add_edge(id, id - img.cols + 1, rightup[i][j],rightup[i][j]);
                }
                id ++;
            }
        }
//        isGraphSet = true;
    }
//    else if (isRevise){
//        int id = 0;
//        for (int i = 0; i < img.rows; i ++) {
//            for (int j = 0; j < img.cols; j ++) {
//                if (mask.at<uchar>(i ,j) == 2) {
//                    g->set_tweights(id, 0, 9 * gamma);
//                }
//                else if (mask.at<uchar>(i, j) == 3) {
//                    g->set_tweights(id, 9 * gamma, 0);
//                }
//                id ++;
//            }
//        }
//    }
//    else {
//        int id = 0;
//        for (int i = 0; i < img.rows; i ++) {
//            for (int j = 0; j < img.cols; j ++) {
//                if (mask.at<uchar>(i ,j) == 2) {
//                    g->set_tweights(id, 0, 9 * gamma);
//                }
//                else if (mask.at<uchar>(i, j) == 3) {
//                    g->set_tweights(id, 9 * gamma, 0);
//                }
//                else {
//                    Vec3b pixel = img.at<Vec3b>(i, j);
//                    g->add_tweights(id,
//                                    -log(GMMs[0].get_prob({pixel[0], pixel[1], pixel[2]})),
//                                    -log(GMMs[1].get_prob({pixel[0], pixel[1], pixel[2]})));
//                }
//                id ++;
//            }
//        }
//    }

    float flow = g->maxflow();

    for (int i = y1; i <= y2; i ++) {
        for (int j = x1; j < x2; j ++) {
//            if (i < y1 || i > y2 || j < x1 || j > x2) continue;
            if (mask.at<uchar>(i, j) == 2 || mask.at<uchar>(i, j) == 3) {
                continue;
            }
            int id = i * img.cols + j;
            if (g->what_segment(id) == GraphType::SINK) {
                mask.at<uchar>(i, j) = 0;
            }
            else {
                mask.at<uchar>(i, j) = 1;
            }
        }
    }

    delete g;
}

void GrabCut::iterative_process() {

    long time1 = getCurrentTime();
    step1();
    step2();
    step3(false);
    output();
    cal_E();
    long time2 = getCurrentTime();
    cout << "耗时：" << time2 - time1 << " ms" << endl;
    cout << "能量函数：" << log(E) << endl;
}

Mat GrabCut::get_mask() {

    return mask;
}

void GrabCut::revise(vector<cv::Point> background_pixels, vector<cv::Point> foreground_pixels) {

    long time1 = getCurrentTime();
    for (int i = 0; i < background_pixels.size(); i ++) {
        auto pixel = background_pixels[i];
        mask.at<uchar>(pixel) = 2;
    }
    for (int i = 0; i < foreground_pixels.size(); i ++) {
        auto pixel = foreground_pixels[i];
        mask.at<uchar>(pixel) = 3;
    }
    step3(true);
    cal_E();
    long time2 = getCurrentTime();
    cout << "耗时：" << time2 - time1 << " ms" << endl;
    cout << "能量函数：" << log(E) << endl;
}

void GrabCut::cal_E() {

    E = 0;
    for (int i = y1; i <= y2; i ++) {
        for (int j = x1; j <= x2; j ++) {
            Vec3b pixel = img.at<Vec3b>(i, j);
            // U
            E += -log(GMMs[0].get_prob({pixel[0], pixel[1], pixel[2]})) > 1e9 ?
                    1e9 : -log(GMMs[0].get_prob({pixel[0], pixel[1], pixel[2]}));
            // V
            if (j - 1 >= 0 && mask.at<uchar>(i, j) != mask.at<uchar>(i, j - 1)) {
                E += left[i][j];
            }
            if (j - 1 >= 0 && i - 1 >= 0 && mask.at<uchar>(i, j) != mask.at<uchar>(i - 1, j - 1)) {
                E += leftup[i][j];
            }
            if (i - 1 >= 0 && mask.at<uchar>(i, j) != mask.at<uchar>(i - 1, j)) {
                E += up[i][j];
            }
            if (j + 1 < img.cols && i - 1 >= 0 && mask.at<uchar>(i, j) != mask.at<uchar>(i - 1, j + 1)) {
                E += rightup[i][j];
            }
        }
    }
}

long double GrabCut::get_E() {

    return E;
}

void GrabCut::output() {

    string filename = "../result.txt";
    ofstream outfile;
    outfile.open(filename, ios::out|ios::app);
    outfile << "* * * * * * * * * * * * * * * * * * * * * * *" << endl;
    outfile << "背景模型：" << endl;
    GMM models_0 = GMMs[0];
    for (int i = 0; i < 5; i ++) {
        auto model = models_0.get_model(i);
        outfile << "高斯模型" << i << "："<< endl;
        Mat mean = Mat(model.get_mean(), CV_32FC1);
        outfile << "均值：" << mean << endl;
        outfile << "协方差矩阵：" << model.get_covariance() << endl;
        outfile << "权重：" << models_0.get_weight(i) << endl;
        outfile << endl;
    }
    GMM models_1 = GMMs[1];
    for (int i = 0; i < 5; i ++) {
        auto model = models_1.get_model(i);
        outfile << "高斯模型" << i << "："<< endl;
        Mat mean = Mat(model.get_mean(), CV_32FC1).t();
        outfile << "均值：" << mean << endl;
        outfile << "协方差矩阵：" << model.get_covariance() << endl;
        outfile << "权重：" << models_1.get_weight(i) << endl;
        outfile << endl;
    }
}