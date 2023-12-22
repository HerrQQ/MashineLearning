#include "switcher.h"

int main ()
{
    google::InitGoogleLogging("switcher_test"); // 初始化 glog
    VehicleState vehicleState_test;
    memset(&vehicleState_test,0,sizeof(vehicleState_test));
    vehicleState_test.speed_mps=5.f;
    double c1=0.1;
    double c2 =0.0009;
    int n=100;
    while (n)
    {
        n--;
        switcher::Switcher switchertest(vehicleState_test,c1,c2);
        double outPut=0.0,input1=1.0,input2=-1.0;
        SwitcherStatus status=switchertest.fRun_Switcher(outPut,input1,input2);

        LOG(ERROR)  << "output" <<outPut<< std::endl;
        // LOG(ERROR)  << "Status" <<status<< std::endl;

    }



    google::ShutdownGoogleLogging(); // 关闭 glog
}