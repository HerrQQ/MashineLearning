#include <iostream>
#include <vector>
#include <cmath>
#include "glog/logging.h"

class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, int hidden1Size, int hidden2Size, int outputSize) {

        initialize(inputSize, hidden1Size, hidden2Size, outputSize);
        inputSize_  =inputSize;
        hidden1Size_=hidden1Size;
        hidden2Size_=hidden2Size;
        outputSize_=outputSize;
    }

    void initialize(int inputSize, int hidden1Size, int hidden2Size, int outputSize) {
        // 初始化input层权重和偏置

        for (int i=0;i<inputSize;++i)
        {
         weightsinput.push_back(std::vector<double>(hidden1Size,1.0));
         biasesinput.push_back(0.0);
        }     


        // 初始化第1层权重和偏置
        for (int i=0;i<hidden1Size;++i)
        {
         weights1.push_back(std::vector<double>(hidden2Size,1.0));
         biases1.push_back(0.0);
        }
        // 初始化输出层also layer2 权重和偏置
         for (int i=0;i<hidden2Size;++i)
        {
        weights2.push_back(std::vector<double>(outputSize,1.0));
         biases2.push_back(0.0);
        }
    }

    double activate(double x) {
        return std::max(0.0, x); // 激活函数
    }

    double activationDerivative(double x) {
        return (x > 0.0) ? 1.0 : 0.0; // 激活函数导数
    }

     std::vector<double> forwardPropagation(const std::vector<double>& input)
    {
         std::vector<double> hidden1(hidden1Size_);
         std::vector<double> hidden2(hidden2Size_);
         std::vector<double> output(outputSize_);
        // layer1 out
        for (int i = 0; i < hidden1Size_; ++i) {
            double sum = 0.0;
            for (int j = 0; j < inputSize_; ++j) {
                sum += input[j] * weightsinput[j][i];
            }
            hidden1[i] = activate(sum + biasesinput[i]);
        }

        // layer2 out
        for (int i = 0; i < hidden2Size_; ++i) {
            double sum = 0.0;
            for (int j = 0; j < hidden1Size_; ++j) {
                sum += hidden1[j] * weights1[j][i];
            }
            hidden2[i] = activate(sum + biases1[i]);
        }

        // out
        for (int i = 0; i < outputSize_; ++i) {
            double sum = 0.0;
            for (int j = 0; j < hidden2Size_; ++j) {
                sum += hidden2[j] * weights2[j][i];
            }
            output[i] = activate(sum + biases2[i]);
        }

        return output;


    }

 double fRefreshWeights(const double weightsOld,const double learningRate,const double gradient)
 {

   return (weightsOld-gradient*learningRate);

 }


    void backPropagation(const std::vector<double>& input, const std::vector<double>& target, const double learningRate) {
        std::vector<double> hidden1(hidden1Size_);
        std::vector<double> hidden2(hidden2Size_);
        std::vector<double> output(outputSize_);
        std::vector<double> hidden1Be4ActiveFunc (hidden1Size_);
        std::vector<double> hidden2Be4ActiveFunc (hidden2Size_);
        std::vector<double> outputBe4ActiveFunc (outputSize_);
        // layer1 out
        for (int i = 0; i < hidden1Size_; ++i) {
            double sum = 0.0;
            for (int j = 0; j < inputSize_; ++j) {
                sum += input[j] * weightsinput[j][i];
            }
            hidden1Be4ActiveFunc[i]=sum + biasesinput[i];
            hidden1[i] = activate(sum + biasesinput[i]);
        }

        // layer2 out
        for (int i = 0; i < hidden2Size_; ++i) {
            double sum = 0.0;
            for (int j = 0; j < hidden1Size_; ++j) {
                sum += hidden1[j] * weights1[j][i];
            }
            hidden2Be4ActiveFunc[i]=sum + biasesinput[i];
            hidden2[i] = activate(sum + biases1[i]);
        }

        // layerout out
        for (int i = 0; i < outputSize_; ++i) {
            double sum = 0.0;
            for (int j = 0; j < hidden2Size_; ++j) {
                sum += hidden2[j] * weights2[j][i];
            }
            outputBe4ActiveFunc[i]=sum + biasesinput[i];
            output[i] = activate(sum + biases2[i]);
        }
         //for (auto x : output)
        // std::cout<<"Forward Result:"<< x <<std::endl;

        // back

        // output gradient
        std::vector<double> outputGradient(outputSize_);
        double errors [outputSize_];
        for (int i = 0; i < outputSize_; ++i) {
            double error = -(target[i] - output[i]);
            outputGradient[i] = error * activationDerivative(outputBe4ActiveFunc[i]);
        }
        for (const auto& error:errors)
        {
            error_+=error;

        }
        error_= error_/outputSize_;



        // layer 2 gradient
        std::vector<std::vector<double>> hidden2Gradient(hidden2Size_,std::vector<double>(outputSize_));
        
        for (int i = 0; i < hidden2Size_; ++i) {

            for (int j = 0; j < outputGradient.size(); ++j) {
                hidden2Gradient[i][j] = outputGradient[j] * hidden2[i];//* activationDerivative(outputBe4ActiveFunc[j]);
            }
        }

        // layer 1 gradient
        std::vector<std::vector<double>> hidden1Gradient(hidden1Size_,std::vector<double>(hidden2Size_));
        for (int i = 0; i < hidden1Size_; ++i) {

            for (int j = 0; j < hidden2Size_; ++j){// 
                for (int q=0;q<outputSize_;++q)
                {
                    hidden1Gradient[i][j] += outputGradient[q]*weights2[j][q] * hidden1[i]*activationDerivative(hidden2Be4ActiveFunc[j]);
                }
                
            }
        }// wait to check size 
        //  input layer gradient 
        std::vector<std::vector<double>> inputGradient(inputSize_,std::vector<double>(hidden1Size_));
        for (int i = 0; i < inputSize_; ++i) {
            for (int j = 0; j < hidden1Size_; ++j) {
                for (int q=0;q<hidden2Size_;++q)
                {
                    for (int p=0;p<outputSize_;++p)
                    {
                        inputGradient[i][j] += outputGradient[p]*weights2[p][q]
                        *activationDerivative(hidden2Be4ActiveFunc[q]) *weights1[j][q]* input[i]*activationDerivative(hidden1Be4ActiveFunc[j]);
                    }
                }
                
            }
        }
        // refresh gradient
      // double newWeight=0.0;
      //layer input 
      static int counter=0;
      counter=(counter++)%10;
      bool flag =false;
      flag=(counter==0);
      for (int i = 0; i < inputSize_; ++i) {
         for (int j = 0; j < hidden1Size_; ++j) {
            weightsinput[i][j]=fRefreshWeights(weightsinput[i][j],inputGradient[i][j],learningRate);
         if (flag)
           LOG(INFO) << "weightsinput[" <<i<<","<<j<<"]is:"<<weightsinput[i][j];
         }
      }
      //layer 1
      for (int i = 0; i < hidden1Size_; ++i) {
         for (int j = 0; j < hidden2Size_; ++j) {
               weights1[i][j]=fRefreshWeights(weights1[i][j],hidden1Gradient[i][j],learningRate);
                  if (flag)
           LOG(INFO) << "weights1 [" <<i<<","<<j<<"] is:"<<weights1[i][j];
         }
      }
      //layer 2
      for (int i = 0; i < hidden2Size_; ++i) {
         for (int j = 0; j < outputSize_; ++j) {
               weights2[i][j]=fRefreshWeights(weights2[i][j],hidden2Gradient[i][j],learningRate);
                           if (flag)
           LOG(INFO) << "weights2 [" <<i<<","<<j<<"]is:"<<weights2[i][j];
         }
      }
    }
    double getError()
    {
    return error_;
    }


private:
    int inputSize_ ;
    int hidden1Size_;
    int hidden2Size_;
    int outputSize_;
    double error_;
   std::vector<std::vector<double>> weightsinput; // weight
   std::vector<std::vector<double>> weights1;
   std::vector<std::vector<double>> weights2;//out
   std::vector<double> biasesinput; // biase
   std::vector<double> biases1;
   std::vector<double> biases2;//out


};


