#include <iostream>
#include <vector>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Eigenvalues>
// #include "glog/logging.h"

class NeuralNetwork {
public:
    NeuralNetwork(int hiddenlayer, int hiddendimen, double learningRate) {
        hiddenlayer_ = hiddenlayer;
        hiddendimen_ = hiddendimen;
        learningRate_ = learningRate;
        layerweight_.resize(hiddenlayer_ + 1);
        layerhidden_.resize(hiddenlayer_ + 1);    // 最后一维保存前向网络计算出的y值
        hiddenGradient_.resize(hiddenlayer_);
    }

    void initialize(const Eigen::VectorXd& trainingx, const Eigen::VectorXd& targety, int iterations) {
        input_ = trainingx;
        target_ = targety;
        iterations_ = iterations;
        // 初始化layerweight_权重
        std::cout << "hiddendimen_:" << hiddendimen_ << " input_.size():" << input_.size() <<std::endl;
        layerweight_[0] = Eigen::MatrixXd::Ones(hiddendimen_, input_.size());
        for(int j = 1; j < hiddenlayer_; j++){
            layerweight_[j] = Eigen::MatrixXd::Ones(hiddendimen_, hiddendimen_);
        }//ok
        layerweight_[hiddenlayer_] = Eigen::MatrixXd::Ones(target_.size(), hiddendimen_);
        //ok
        // 初始化layerhidden_ results
        for(int j = 0; j < hiddenlayer_; j++){
            layerhidden_[j] = Eigen::VectorXd::Zero(hiddendimen_);
        }// ? 
    }

    double activate(double x) {
        return std::max(0.0, x); // 激活函数
    }

    double activationDerivative(double x) {
        return (x > 0.0) ? 1.0 : 0.0; // 激活函数导数
    }


    void forwardPropagation(const Eigen::VectorXd& input, Eigen::VectorXd& output){
        layerhidden_[0] = layerweight_[0] * input;
        for(int j = 1; j <= hiddenlayer_; j++){
            layerhidden_[j] = layerweight_[j] * layerhidden_[j-1];
        }
        output = layerhidden_[hiddenlayer_];//ok
        std::cout << "------------output:" << output << std::endl;
    }//ok

    void backPropagation() {  
        // 从右向左链式计算
        Eigen::MatrixXd result;

        for(int layer = hiddenlayer_ - 1; layer >=0; layer--){  // 某一层的累乘
            Eigen::MatrixXd tmp = Eigen::MatrixXd::Identity(target_.size(), target_.size());
            // std::cout << "tmp.rows()" << tmp.rows() << "" << tmp.cols() << std::endl;
            // int layer = 0;
            for(int i=hiddenlayer_; i>layer; i--) {
                result.resize(layerweight_[hiddenlayer_].rows(), layerweight_[i].cols());  // 存储临时的结果
                // std::cout << "result.rows()" << result.rows() << "" << result.cols() << std::endl;
                // std::cout << "layerweight_[i].rows()" << layerweight_[i].rows() << "" << layerweight_[i].cols() << std::endl;

                result = tmp * layerweight_[i];//?

                // 更新结果 
                tmp.resize(result.rows(), result.cols());
                tmp = result;
                // std::cout << "layer:" << layer << " result:" << result << std::endl;
            }
            hiddenGradient_[layer] = result;
            std::cout << "hiddenGradient_[" << layer << "]:"<< " result:" << result << std::endl;
        }
    }

    void updatePara(const double learningRate, const Eigen::VectorXd& derivative){
        layerweight_[hiddenlayer_] = layerweight_[hiddenlayer_] - derivative[0] * layerhidden_[hiddenlayer_-1].transpose() * learningRate;
        for(int i = hiddenlayer_ - 1; i > 0; i--){
            // std::cout << "hiddenGradient_[i]:" << hiddenGradient_[i] << std::endl;
            // std::cout << "layerhidden_[i-1]:" << layerhidden_[i-1] << std::endl;
            // std::cout << "hiddenGradient_* layerhidden_:" << hiddenGradient_[i].transpose() * layerhidden_[i-1].transpose() * learningRate << std::endl;
            layerweight_[i] = layerweight_[i] - derivative[0] * hiddenGradient_[i].transpose() * layerhidden_[i-1].transpose() * learningRate;
        }
        layerweight_[0] = layerweight_[0] - derivative[0] * hiddenGradient_[0].transpose() * input_.transpose() * learningRate;
    }

    void neuralNetwork()
    {
        for(int i = 0; i < iterations_; i++){
            // 前向传播
            Eigen::VectorXd output;
            forwardPropagation(input_, output);

            // 计算误差
            error_ = 0.5*(layerhidden_[hiddenlayer_] - target_)*(layerhidden_[hiddenlayer_] - target_);
            std::cout << "===========error_:" <<  error_ << std::endl;
            if (error_[0] <0.01){
                break;
            }

            // error对y偏导
            Eigen::VectorXd derivative = layerhidden_[hiddenlayer_] - target_;
            // std::cout << "derivative:" << derivative << " derivative.size():" << derivative.size() << std::endl;

            // 反向传播
            backPropagation();

            // 更新参数
            updatePara(learningRate_, derivative);
        }
    }

private:
    int hiddenlayer_;   // 定义隐藏层的层数
    int hiddendimen_;   // 定义隐藏层的维度
    int iterations_;
    std::vector<Eigen::VectorXd> layerhidden_;
    std::vector<Eigen::MatrixXd> layerweight_;
    std::vector<Eigen::MatrixXd> hiddenGradient_;
    Eigen::VectorXd input_;
    Eigen::VectorXd target_;
    Eigen::VectorXd error_;
    double learningRate_;
};