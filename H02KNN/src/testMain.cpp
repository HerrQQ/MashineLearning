#include "Knn.h"
#include <random> 
#include <iostream>
#include <fstream>

std::vector <DataPoint_tag> testDatas;
int main ()
{
        google::InitGoogleLogging("my_program"); // 初始化 glog





    int n=0;
    double testdata=50.0;
    std::cout<<"plz give a feature data "<<std::endl;
    std::cin>>testdata;
    std::cout<<"the feature you give is "<<testdata<<std::endl;
    while(n<1000)
    {
        std::random_device rd;
        DataPoint_tag data;
        std::mt19937 gen(rd()); // 使用 Mersenne Twister 引擎
        std::uniform_real_distribution<double> disReal(0.0, 2*testdata);
        double randomX = disReal(gen);
        n++;
        double randomY =2*randomX+3;
        for (int i=0;i<1;++i)
        {
            data.features.push_back(randomX);
        }

        data.value=randomY;
        testDatas.push_back(data);
    }
    KNNSpace:: KNN test_knn(testDatas);
    std::vector<double> newPoint;

    newPoint.emplace_back(testdata);
    test_knn.fInitialize_KNN(newPoint,1);
    double ans=test_knn.fRun_KNN();
    printf ("Our answer is: %.2f",ans );
    printf ("Soll answer is: %.2f",2*testdata+3 );
    std::ofstream outFile("data.txt"); // 创建输出文件流对象，并打开文件

    if (outFile.is_open()) { // 检查文件是否成功打开
    int size_t=testDatas.size();
        for (int j=0;j<size_t;j++)
        {
           outFile << testDatas[j].features[0] << " "<<testDatas[j].value<<std::endl;   
        }

        outFile.close(); // 关闭文件流
            LOG(INFO)  << "Data written to file." << std::endl;
    } else {
            LOG(ERROR) << "Failed to open output file." << std::endl;
    }

    google::ShutdownGoogleLogging(); // 关闭 glog
}