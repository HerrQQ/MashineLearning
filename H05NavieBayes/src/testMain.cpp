#include "NB.h"
#include <random> 
#include <iostream>
#include <fstream>


int main ()
{
    google::InitGoogleLogging("my_program"); // 初始化 glog

    std::vector<std::vector<int>> X_train = {
        {1, 1}, {1, 2}, {2, 1}, {2, 2}, {3, 1}, {3, 2}
    };
    std::vector<int> y_train = {0, 0, 0, 1, 1, 1};

    NB nbc;
    nbc.fit(X_train, y_train);

    std::vector<int> x_new = {1, 1};
    int predicted_class = nbc.predict(x_new);
    std::cout << "Predicted class: " << predicted_class << std::endl;

    return 0;



    google::ShutdownGoogleLogging(); // 关闭 glog
}