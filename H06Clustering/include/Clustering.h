#include <iostream>
#include <vector>
#include <cmath>

struct Point {
    double x;
    double y;

    Point(double xVal, double yVal) : x(xVal), y(yVal) {}
};

// 计算两个点之间的欧氏距离
double euclideanDistance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// 查找最近的聚类中心
int findClosestCenter(const Point& point, const std::vector<Point>& centers) {
    int closestCenterIndex = 0;
    double closestDistance = euclideanDistance(point, centers[0]);

    for (int i = 1; i < centers.size(); i++) {
        double distance = euclideanDistance(point, centers[i]);
        if (distance < closestDistance) {
            closestDistance = distance;
            closestCenterIndex = i;
        }
    }

    return closestCenterIndex;
}

// K均值聚类算法
void kMeansClustering(std::vector<Point>& points, int k) {
    std::vector<Point> centers;

    // 初始化k个随机聚类中心
    for (int i = 0; i < k; i++) {
        int randIndex = std::rand() % points.size();
        centers.push_back(points[randIndex]);
    }

    std::vector<int> clusterAssignments(points.size());
    bool converged = false;

    while (!converged) {
        std::vector<Point> newCenters(k, {0.0, 0.0});
        std::vector<int> clusterCounts(k, 0);

        // 将点分配给最近的聚类中心
        for (int i = 0; i < points.size(); i++) {
            int closestCenterIndex = findClosestCenter(points[i], centers);
            clusterAssignments[i] = closestCenterIndex;
            newCenters[closestCenterIndex].x += points[i].x;
            newCenters[closestCenterIndex].y += points[i].y;
            clusterCounts[closestCenterIndex]++;
        }

        // 更新聚类中心位置
        for (int i = 0; i < k; i++) {
            if (clusterCounts[i] > 0) {
                newCenters[i].x /= clusterCounts[i];
                newCenters[i].y /= clusterCounts[i];
            }
        }

        // 检查是否收敛
        converged = true;
        for (int i = 0; i < k; i++) {
            if (newCenters[i].x != centers[i].x || newCenters[i].y != centers[i].y) {
                converged = false;
                break;
            }
        }

        // 更新聚类中心
        centers = newCenters;
    }

    // 打印聚类结果
    std::cout << "Cluster assignments:\n";
    for (int i = 0; i < points.size(); i++) {
        std::cout << "(" << points[i].x << ", " << points[i].y << ") -> Cluster " << clusterAssignments[i] << "\n";
    }
}


