#include "switcher.h"


namespace switcher
{

        Switcher:: Switcher(const VehicleState& vehicleState,const double & curvature_cur
        ,const double & curvature_pre )
        {
          this->vehicleState_=vehicleState;
          this->curvature_cur_=curvature_cur;
          this->curvature_pre_=curvature_pre;

        }
        SwitcherStatus Switcher::  fRun_Switcher(double & output,const double& input_MPC,const double& input_PP )
        {         
          controller_inputMPC_=input_MPC;
          controller_inputPP_=input_PP;
          fScenarioJudge();
          output=fSmoother();
          return switcherStatus_;

        }

        double Switcher:: fSmoother() const
        {
          static SwitcherStatus status_hist =MPC;
          static SwitcherStatus flag_change=NONE;
          int smoothStep=static_cast<int>(1/(SMOOTHTIME/SYSTEMCYCLE)*100);
          int static smoothFactorMPC  =100;
          int smoothFactorPP =100-smoothFactorMPC;
          int smoothFactorMax=100;
          double ans_t=0.f;
            
          if (status_hist==this->switcherStatus_)
          {
            //do nothing
          }
          else if (status_hist==MPC&&this->switcherStatus_==PP)
          {
            flag_change=PP;
          }
          else if (status_hist==PP&&this->switcherStatus_==MPC)
          {
            flag_change=MPC;
          }
          switch (flag_change)
          {
          case MPC:
            {

              if (smoothFactorMPC<smoothFactorMax)
              {
                smoothFactorMPC+=smoothStep;

                smoothFactorPP=smoothFactorMax-smoothFactorMPC;
              }
              else
              {
                flag_change=NONE;
              }
              break;
            }
          case PP:
            {
              if (smoothFactorMPC>0)
              {
                smoothFactorMPC-=smoothStep;
                smoothFactorPP=smoothFactorMax-smoothFactorMPC;
              }
              else 
              {
                flag_change=NONE;
              }

              break;
            }         
          default:
            break;
          }
             LOG(ERROR)  << "smoothFactorMPC  " <<smoothFactorMPC<< std::endl;
          // //  LOG(ERROR)  << "controller_inputMPC_" <<controller_inputMPC_<< std::endl;
             LOG(ERROR)  << "smoothFactorPP  " <<smoothFactorPP<< std::endl;
          // //  LOG(ERROR)  << "controller_inputPP_" <<controller_inputPP_<< std::endl;
          status_hist=this->switcherStatus_;
          ans_t=smoothFactorMPC*controller_inputMPC_/smoothFactorMax+smoothFactorPP*controller_inputPP_/smoothFactorMax;
          return  ans_t;
        }

        void Switcher:: fScenarioJudge()
        {
          double curve_t=std::max(curvature_cur_,curvature_pre_);
          double speed_t=vehicleState_.speed_mps;

          // 1.active condition 

          static float counter_in =0.f;
          if (speed_t<=SPEED_ACTIVE&&curve_t>=CURVATURE_ACTIVE)
          {
            counter_in+=SYSTEMCYCLE;
            counter_in=std::fmin(COUNTERLIMIT,counter_in);
          }
          else 
          {
            counter_in=0.f;
          }
          if (counter_in>=COUNTERLIMIT)
          {
            this->switcherStatus_=PP;   
          }

          // deactive condition
          static float counter_out =0.f;
          if (speed_t>=SPEED_DEACTIVE||curve_t<=CURVATURE_DEACTIVE)
          {
            counter_out+=SYSTEMCYCLE;
            counter_out=std::fmin(COUNTERLIMIT+COUNTERLIMIT_DELTA,counter_out);
          }
          else 
          {
            counter_out=0.f;
          }
          if (counter_out>=(COUNTERLIMIT+COUNTERLIMIT_DELTA))
          {
            this->switcherStatus_=MPC;      
          }
        }

}