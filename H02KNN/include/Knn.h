#pragma once 
#include "KnnStructure.h"
#include <cmath>
#include <algorithm>
#include <glog/logging.h>
namespace KNNSpace{


    class KNN{
        
        public:

        KNN()=default;
        KNN(const std::vector<DataPoint>& trainData);
        ~KNN()=default;

        void fInitialize_KNN(const std::vector<double>&Feature,const int &k);
        double fRun_KNN();
        void  fScaling();

        private:

        double  fCalDistance(const DataPoint& p1, std::vector<double> features);
        void  fCalPredictValue();
        int K_;
        double ans_;

        std::vector<DataPoint> trainData_;
        std::vector<double> newDatepoint_;

    };
}