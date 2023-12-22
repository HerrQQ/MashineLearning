#include "PCA.h"
#include <random> 
#include <iostream>
#include <fstream>
#include "glog/logging.h"


int main ()
{
    google::InitGoogleLogging("my_program"); // 初始化 glog

    ifstream file("data.txt");
    if (!file) {
        cerr << "Error opening file." << endl;
        return 1;
    }

    // 读取数据矩阵
    vector<vector<double>> inputData;
    double value;
    while (file >> value) {
        inputData.emplace_back(vector<double>{value});
    }

    // 转换为 Eigen 矩阵
    size_t rows = inputData.size();
    size_t cols = inputData[0].size();
    MatrixXd data(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data(i, j) = inputData[i][j];
        }
    }

    // 使用主成分分析进行降维
    int k = 2; // 设置降维后的维度
    MatrixXd reducedData = pca(data, k);

    // 输出降维后的数据
    cout << "Reduced Data: " << endl;
    cout << reducedData << endl;

    return 0;




    google::ShutdownGoogleLogging(); // 关闭 glog
}