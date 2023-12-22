#ifndef __structofswitcher_h__
#define __structofswitcher_h__
#include <vector>

//temp reuse the structures of controller
struct PathPoint
{
    double x;
    double y;
    // double z;
    double theta;
    double kappa;
    double s;
};

struct VehicleState
{
    bool status;
    double x;
    double y;
    double psi;

    double steerAngle_deg;
    double tireAngle_rad;
    double speed_mps;
    double acceleration_mpss;

    double yawrate_radps;
};

struct TrajectoryPoint
{
    PathPoint pathpoint;
    double velocity;
    double acceleration;
    double time;
};

// new to add 

enum PPStatus
{
    DEACTIVE =0,
    FAULT,
    ACTIVE

};

enum SwitcherStatus
{
    NONE =0, 
    MPC, // mpc
    PP// pp
};

#endif

