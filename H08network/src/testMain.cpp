#include "network.h"

int main() {
   google::InitGoogleLogging("my_program"); // 初始化 glog
   FLAGS_stderrthreshold = 0;
   NeuralNetwork neuralNetwork(2, 2,2, 1);
   // training
   std::vector<std::vector<double>> trainingData = { {0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0} };
   for (int cycle = 0; cycle < 1000; ++cycle) {
      for (const auto& data : trainingData) {
         neuralNetwork.backPropagation({ data[0], data[1] }, {data[2]},0.01);
      }
      // if (neuralNetwork.getError()<0.5)
      // {
      //    std::cout<<" error: "<<neuralNetwork.getError()<<" was small enough, got results"<<std::endl;
      //    break;
      // }

   }

   // test
   std::vector<std::vector<double>> testData = { {0, 0}, {0, 1}, {1, 5}, {2, 2} };
   for (const auto& data : testData) {
      std::vector<double>ans= neuralNetwork.forwardPropagation({ data[0], data[1] });
      int size_t=ans.size();
      for (int i=0;i<size_t;++i)
      {
         std::cout << "Input: " << data[0] << ", " << data[1] << " | Output "<<i<<" is:" << ans[i] << std::endl;
      }
      
   }
   google::ShutdownGoogleLogging(); // 关闭 glog
   return 0;
}