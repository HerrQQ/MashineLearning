#pragma once
#include <vector>


// 定义数据点结构体，包含特征和数值标签
typedef struct DataPoint_tag {
    std::vector<double> features;
    double value;
}DataPoint;