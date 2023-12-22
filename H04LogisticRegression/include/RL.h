#pragma once 
#include "RLStructure.h"
#include <cmath>
#include <algorithm>
#include <glog/logging.h>




class RL{
        
    private:
    double slope;  // 斜率
    double intercept;  // 截距

    public:
    void fit(const std::vector<double>& x_train, const std::vector<double>& y_train) {
    // 确保输入的 x_train 和 y_train 长度相同
    if (x_train.size() != y_train.size()) {
    LOG(ERROR) << "Error: x_train and y_train must have the same length." << std::endl;
    return;
    }

    // 计算 x_train 和 y_train 的均值
    double mean_x = calculateMean(x_train);
    double mean_y = calculateMean(y_train);

    // 计算 x_train 和 y_train 的差值
    std::vector<double> x_diff, y_diff;
    for (int i = 0; i < x_train.size(); ++i) {
    x_diff.push_back(x_train[i] - mean_x);
    y_diff.push_back(y_train[i] - mean_y);
    }

    // 计算斜率
    double numerator = 0.0;
    double denominator = 0.0;
    for (int i = 0; i < x_diff.size(); ++i) {
    numerator += x_diff[i] * y_diff[i];
    denominator += x_diff[i] * x_diff[i];
    }
    slope = numerator / denominator;

    // 计算截距
    intercept = mean_y - slope * mean_x;
    }

    double predict(double x) {
    return slope * x + intercept;
    }

    private:
    double calculateMean(const std::vector<double>& values) {
        double sum = 0.0;
        for (const auto& value : values) {
        sum += value;
        }
        return sum / values.size();
    }
};


