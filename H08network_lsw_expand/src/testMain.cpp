#include "network.h"

int main() {
   NeuralNetwork neuralNetwork(2, 5, 0.01);   // 定义hifdden层参数，隐藏层维度和学习率

   // training
   Eigen::VectorXd trainingx(2);
   trainingx << 1, 0.5;
   Eigen::VectorXd targety = Eigen::VectorXd::Constant(1, 4);
   std::cout << "trainingx:" << trainingx.size() << " targety:" << targety.size() << std::endl;
   neuralNetwork.initialize(trainingx, targety, 10);   // 输入x和y训练数据，定义迭代次数
   neuralNetwork.neuralNetwork();

   // // test
   // Eigen::VectorXd testData(2);
   // testData << 1, 0.5;
   Eigen::VectorXd output(1);
   neuralNetwork.forwardPropagation(trainingx, output);
   std::cout << "最终结果: " << output << std::endl;

   // google::ShutdownGoogleLogging(); // 关闭 glog
   return 0;
}