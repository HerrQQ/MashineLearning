#ifndef __switcher_h__
#define __switcher_h__
#include "structofswitcher.h"
#include "para_switcher.h"
#include <vector>
#include <map>
#include "glog/logging.h"
#include <cmath>
#include <utility>



namespace switcher
{

    class Switcher
    {
        public:
        //pre
        Switcher()=default;
        ~ Switcher()=default;
        Switcher(const VehicleState& vehicleState,const double & curvature_cur
        ,const double & curvature_pre );
        SwitcherStatus fRun_Switcher(double & output,const double& input_1,const double& input_2 );
        private:
        double fSmoother() const;
        void fScenarioJudge();

        private:
        VehicleState vehicleState_;
        double curvature_cur_;
        double curvature_pre_;
        double controller_inputMPC_;
        double controller_inputPP_;
        double output_;
        SwitcherStatus switcherStatus_=MPC;







    };




}
#endif