#include "Clustering.h"
#include <random> 
#include <iostream>
#include <fstream>
#include "glog/logging.h"


int main ()
{
    google::InitGoogleLogging("my_program"); // 初始化 glog


    // 生成一些示例点
    std::vector<Point> points = { {1.0, 2.0}, {2.0, 1.0}, {3.0, 2.0}, {5.0, 4.0}, {5.0, 6.0}, {8.0, 5.0}, {9.0, 4.0} };

    int k = 2; // 聚类数量

    kMeansClustering(points, k);

    return 0;




    google::ShutdownGoogleLogging(); // 关闭 glog
}