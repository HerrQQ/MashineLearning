
import rospy
#adding libs according to your needs

from noa_msgs.msg import  Trajectory_to_Control,ESAInfo,NOHCtrlInfo


global timeLock
timeLock=1   

# passing by the data with ros callback method 
# def callback(msg):
#     global timeLock
#     print(f"msg.PlanningStatus: {msg.PlanningStatus}")

def callback4(msg):
    global timeLock
    if timeLock==1:
        print(f"msg.egoEgoStatus.yawRate: {msg.egoEgoStatus.yawRate}")
        print(f"msg.egoEgoStatus.linearSpeed: {msg.egoEgoStatus.linearSpeed}")
        print(f"msg.egoEgoStatus.accerationX: {msg.egoEgoStatus.accerationX}")
        print(f"msg.egoEgoStatus.accerationY: {msg.egoEgoStatus.accerationY}")
        print(f"msg.egoEgoStatus.steerWheelAngle: {msg.egoEgoStatus.steerWheelAngle}")
        print(f"msg.egoEgoStatus.steerWheelAngleRate: {msg.egoEgoStatus.steerWheelAngleRate}")
        print(f"msg.egoEgoStatus.frontWheelAngle: {msg.egoEgoStatus.frontWheelAngle}")
        timeLock=2
 

def callback7(msg):
    global timeLock
    if timeLock==2:
        print(f"msg.nohCtrlOutput.targetStrAngle: {msg.nohCtrlOutput.targetStrAngle}")
        print(f"msg.nohCtrlOutput.targetAcceleration: {msg.nohCtrlOutput.targetAcceleration}")
        timeLock=1

def start_ros():
    rospy.init_node('printPY')
    #rospy.Subscriber('/planning/Trajectory_toCtrl', Trajectory_to_Control, callback,queue_size=10)
    #rospy.Subscriber('/udp2ros/fusion_lanelines', FusionLaneMarker, callback2,queue_size=10)
    #rospy.Subscriber('/udp2ros/PredictionInfo', PredictionObstacles, callback3,queue_size=10)
    rospy.Subscriber('/udp2ros/ESAInfo', ESAInfo, callback4,queue_size=10)
    #rospy.Subscriber('/ppcontroller/PPOUT',PPOUT,callback5,queue_size=10)
    #rospy.Subscriber('replaytrace/NewTrace',NewTrace,callback6,queue_size=10)
    rospy.Subscriber('/udp2ros/NOHCtrlInfo',NOHCtrlInfo,callback7,queue_size=10)
    rospy.spin()





    

if __name__ == "__main__":

    start_ros()
