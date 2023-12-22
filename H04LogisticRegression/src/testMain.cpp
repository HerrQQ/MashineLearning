#include "RL.h"
#include <random> 
#include <iostream>
#include <fstream>
#include "MultiRL.h"


int main ()
{
        google::InitGoogleLogging("my_program"); // 初始化 glog

if (0)
{
    // 训练数据
    std::vector<double> x_train = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y_train = {2.0, 4.0, 6.0, 8.0, 10.0};

    // 创建线性回归对象
    RL lr;

    // 训练模型
    lr.fit(x_train, y_train);

    // 预测新数据
    double x_new = 6.0;
    double y_pred = lr.predict(x_new);
    std::cout << "Predicted value for x = " << x_new << ": " << y_pred << std::endl;
}
else //multi rl
{
 // 训练数据
    Eigen::MatrixXd X_train(3, 3);  // 5个样本，3个特征
    Eigen::VectorXd y_train(3);  // 目标变量
    X_train << 1, 2, 3,
               2, 4, 6,
               3, 6, 9,
            //    4, 8, 12,
            //    5, 10, 15;
    y_train << 2, 4, 6;//8, 10;

    // 创建线性回归对象
    MUTIRL lr;

    // 训练模型
    lr.fit(X_train, y_train);

    // 预测新数据
    Eigen::VectorXd x_new(3);  // 新的输入特征向量
    x_new << 6, 12, 18;
    double y_pred = lr.predict(x_new);
    std::cout << "Predicted value: " << y_pred << std::endl;
}





    google::ShutdownGoogleLogging(); // 关闭 glog
}