
#include "Knn.h"


namespace COMMEN{
    bool compareDistances(const std::pair<double, double>& p1, const std::pair<double, double>& p2) {
    {
        return p1.first < p2.first;
    }
}
}





namespace KNNSpace{

KNN::KNN(const std::vector<DataPoint>& trainData)
{

    trainData_=trainData;

}

void KNN:: fInitialize_KNN(const std::vector<double>&Feature,const int &k)
{
    newDatepoint_=Feature;
    K_=k;
}
void KNN:: fScaling()
{
    int numFeatures_t=trainData_[0].features.size();
    int datasize_t=trainData_.size();
    std::vector<double> means(numFeatures_t, 0.0);
    std::vector<double> stddevs(numFeatures_t, 0.0);
    // get mean
   for (int j=0;j<datasize_t;++j)
   {
    for (int i=0;i<numFeatures_t;++i)
    {
        means[i]+=trainData_[j].features[i];
    }

   }
   for (auto& x : means)
   {
        x=x/datasize_t;
   }
   // get stddevs
   double devs_t=0.0;
    for (int j=0;j<datasize_t;++j)
    {
        for (int i=0;i<numFeatures_t;++i)
        {
            stddevs[i]+= pow(trainData_[j].features[i]-means[i],2);
        }

    }
   for (auto& x : stddevs)
   {
        x=sqrt(x/datasize_t);
   }
   // standary
    for (int j=0;j<datasize_t;++j)
   {
    for (int i=0;i<numFeatures_t;++i)
    {
        trainData_[j].features[i]=(trainData_[j].features[i]-means[i])/stddevs[i];
    }
    
   }
       for (int i=0;i<numFeatures_t;++i)
    {
        newDatepoint_[i]=(newDatepoint_[i]-means[i])/stddevs[i];
    }
}

double KNN:: fRun_KNN()
{
    if (trainData_[0].features.size()!=newDatepoint_.size())
    {
        LOG(ERROR) <<"size Error"<<std::endl;;
        return -1e6;
    }
    fScaling();
    fCalPredictValue();
    return ans_;

}
double KNN:: fCalDistance(const DataPoint& p1, std::vector<double> features)
{
    double distance = 0.0;
    int size =p1.features.size();

    for (int i = 0; i < size; ++i) {
    distance += pow(p1.features[i] - features[i], 2);
    }
    return sqrt(distance);

}
void  KNN:: fCalPredictValue()
{
    std::vector<std::pair<double, double>> distances;
    for (const auto& trainPoint : trainData_) {
        double distance_t = fCalDistance(trainPoint, newDatepoint_);
        distances.push_back(std::make_pair(distance_t, trainPoint.value));
    }
    std::sort(distances.begin(),distances.end(),COMMEN::compareDistances);
    int k_t=std::fmin(K_,distances.size());
    double ans_t;
    double weight_t;
    double weightSum_t;
    for (int m=0;m<k_t;++m)
    {

        weight_t+=1/distances[m].first;// use 1/x as weight factor
        ans_t+=(distances[m].second*weight_t);
        weightSum_t+=weight_t;

    }
    if (1)
    ans_=ans_t/weightSum_t;
    else 
    {
        // tbd
    }
}
   
}