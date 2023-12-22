#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <Eigen/Eigen>

using namespace std;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;

// 主成分分析函数
MatrixXd pca(const MatrixXd& data, int k) {
    // 计算均值向量
    MatrixXd mean = data.colwise().mean();

    // 数据中心化
    MatrixXd centered = data.rowwise() - mean;

    // 计算协方差矩阵
    MatrixXd covariance = (centered.adjoint() * centered) / double(data.rows() - 1);

    // 特征值分解
    Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(covariance);
    if (eigensolver.info() != Eigen::Success) {
        cerr << "Eigenvalue decomposition failed." << endl;
        exit(1);
    }

    // 提取特征值和特征向量
    MatrixXd eigenvalues = eigensolver.eigenvalues().reverse().eval();
    MatrixXd eigenvectors = eigensolver.eigenvectors().rowwise().reverse().eval();

    // 提取前 k 个主成分
    MatrixXd principalComponents = eigenvectors.leftCols(k);

    return principalComponents;
}


