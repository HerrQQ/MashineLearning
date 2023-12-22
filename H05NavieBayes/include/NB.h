#pragma once 
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <glog/logging.h>

using namespace std;


class NB{

private:
    std::vector<std::vector<int>> X_train;  // 训练数据特征矩阵
    std::vector<int> y_train;  // 训练数据标签向量
    std::vector<double> class_priors;  // 不同类别的先验概率
    std::vector<std::unordered_set<int>> unique_feature_values;  // 特征的可能取值集合
    std::vector<std::vector<std::vector<double>>> feature_class_likelihoods;  // 特征在不同类别下的似然概率

public:
    void fit(const std::vector<std::vector<int>>& X, const std::vector<int>& y) {
        // 初始化训练数据和标签
        X_train = X;
        y_train = y;

        // 计算不同类别的先验概率
        calculate_class_priors();

        // 计算特征在不同类别下的似然概率
        calculate_feature_likelihoods();
    }

    int predict(const std::vector<int>& x) {
        int n_classes = class_priors.size();
        double max_posterior = 0.0;
        int predicted_class = 0;

        for (int c = 0; c < n_classes; ++c) {
            double posterior = class_priors[c];

            for (size_t i = 0; i < x.size(); ++i) {
                double likelihood = feature_class_likelihoods[i][c][x[i]];
                posterior *= likelihood;
            }

            if (posterior > max_posterior) {
                max_posterior = posterior;
                predicted_class = c;
            }
        }

        return predicted_class;
    }

private:
    void calculate_class_priors() {
        int n_samples = y_train.size();
        std::unordered_set<int> unique_classes(y_train.begin(), y_train.end());
        // del the same ele to get the class type

        for (int c : unique_classes) {
            int count = 0;

            for (int label : y_train) {
                if (label == c) {
                    count++;// counter
                }
            }

            double prior = static_cast<double>(count) / n_samples;
            class_priors.push_back(prior);// how to classify the prob different type
        }
    }

    void calculate_feature_likelihoods() {
        int n_samples = y_train.size();
        int n_features = X_train[0].size();
        int n_classes = class_priors.size();

        unique_feature_values.resize(n_features);
        feature_class_likelihoods.resize(n_features);

        for (int i = 0; i < n_features; ++i) {
            std::unordered_set<int> unique_values;//get type of features

            for (int j = 0; j < n_samples; ++j) {
                unique_values.insert(X_train[j][i]);
            }

            unique_feature_values[i] = unique_values;//e.g. [0] has first features type

            feature_class_likelihoods[i].resize(n_classes);

            for (int c = 0; c < n_classes; ++c) {
                int count_class = 0;

                for (int j = 0; j < n_samples; ++j) {
                    if (y_train[j] == c) {
                        count_class++;
                    }
                }

                std::vector<double> likelihoods(unique_values.size(), 0.0);
                int pos = 0;

                for (int value : unique_values) {
                    int count_feature_class = 0;

                    for (int j = 0; j < n_samples; ++j) {
                        if (X_train[j][i] == value && y_train[j] == c) {
                            count_feature_class++;
                        }
                    }

                    likelihoods[pos] = static_cast<double>(count_feature_class) / count_class;
                    pos++;
                }

                feature_class_likelihoods[i][c] = likelihoods;
            }
        }
    }

        

};


