#pragma once 
#include "RLStructure.h"
#include <Eigen/Eigen>
#include <cmath>
#include <algorithm>
#include <glog/logging.h>




class MUTIRL{
 private:
    Eigen::VectorXd coefficients;  // 系数向量，表示斜率和截距

public:
    void fit(const Eigen::MatrixXd& X_train, const Eigen::VectorXd& y_train) {
        // 确保输入的 X_train 和 y_train 维度匹配
        if (X_train.rows() != y_train.size()) {
            LOG(ERROR)<< "Error: X_train and y_train dimensions don't match." << std::endl;
            return;
        }

        // 计算系数 (斜率和截距) 的最小二乘解
        Eigen::VectorXd ones = Eigen::VectorXd::Ones(X_train.rows());  // 创建一个维度与 X_train 行数相同的全一向量
        Eigen::MatrixXd X_extended(X_train.rows(), X_train.cols() + 1);  // 创建扩展后的特征矩阵
        X_extended << ones, X_train;  // 在特征矩阵的第一列插入全一向量
        coefficients = (X_extended.transpose() * X_extended).inverse() * X_extended.transpose() * y_train;
    }

    double predict(const Eigen::VectorXd& x) {
        // 扩展输入特征向量，添加一个全一项
        Eigen::VectorXd x_extended(x.size() + 1);
        x_extended << 1, x;

        // 使用系数进行预测
        return x_extended.dot(coefficients);
    }       

};


