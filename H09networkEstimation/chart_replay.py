import sqlite3,sys,rospy,math,threading
from PyQt5.QtChart import QDateTimeAxis,QValueAxis,QLineSeries,QSplineSeries,\
    QScatterSeries,QChart,QChartView
from PyQt5.QtWidgets import QApplication,QWidget
from PyQt5.QtGui import QPainter,QPen
from PyQt5.QtCore import QDateTime,Qt,QTimer,QPointF
import multiprocessing as mp
import math 

#adding libs according to your needs

from noa_msgs.msg import FusionLaneMarker,LineInfo,BorderInfo,FusionObjects,\
    PredictionObstacles,PredictionObstacle,ObjectInfo,Trajectory_to_Control,ESAInfo
from ppcontroller.msg import PPOUT
from replaytrace.msg import NewTrace


timeCounter=0
g_x=0.0
g_y=0.0
# pipe to transfer the data between different threads. 
# using "recv" to recive date from pipe and "send" to input 

pipefather_1, pipeson_1 = mp.Pipe()
pipefather_2, pipeson_2 = mp.Pipe()
pipefather_3, pipeson_3 = mp.Pipe()
pipefather_4, pipeson_4 = mp.Pipe()
pipefather_5, pipeson_5 = mp.Pipe()
pipefather_6, pipeson_6 = mp.Pipe()
pipefather_7, pipeson_7 = mp.Pipe()  
pipefather_8, pipeson_8=mp.Pipe()

# each class, e.g. ChartView, plots a figure.   
# in class ChartView and ChartView2 some line and points in one time point are showed in a fix axis
# in class ChartView3 the data are processed as a timerseries 
# The options in details are in the below codes 
class ChartView(QChartView,QChart): 
    def __init__(self, *args, **kwargs):
        super(ChartView, self).__init__(*args, **kwargs)
        self.resize(600, 400) 
        self.setRenderHint(QPainter.Antialiasing)  
        self.chart_init()
        self.timer_init() 

    def timer_init(self):
        #使用QTimer，10ms触发一次，更新数据
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.drawLine)
        self.timer.start(10)

    def chart_init(self):
        self.chart = QChart()
        self.acceleration = QSplineSeries() 
        #  QSplineSeries() class creats a Object which has a inside container,that type of members should be "QpointF"
        # using .append or <<QpointF() to input your date 
        self.speed = QSplineSeries() 
        #设置曲线名称
        self.acceleration.setName("acceleration")
        self.speed.setName("speed")
        #把曲线添加到QChart的实例中
        self.chart.addSeries(self.acceleration)
        self.chart.addSeries(self.speed)
        #声明并初始化X轴，Y轴
        self.dtaxisX = QValueAxis()
        self.vlaxisY = QValueAxis()
        #self.dtaxisX2 = QValueAxis()
        self.vlaxisY2 = QValueAxis()
        #设置坐标轴显示范围
        self.dtaxisX.setMin(-1.0)
        self.dtaxisX.setMax(6.0)
        self.vlaxisY.setMin(-10.0)    
        self.vlaxisY.setMax(120.0) #设置y轴最大值
        #self.dtaxisX2.setMin(-1.0)
        #self.dtaxisX2.setMax(6.0)
        self.vlaxisY2.setMin(-3.0)    
        self.vlaxisY2.setMax(5.0) #设置y轴最大值
         #设置坐标轴上的格点
        self.dtaxisX.setTickCount(15) #平均分的刻度分隔
        self.vlaxisY.setTickCount(13)
        #self.dtaxisX2.setTickCount(10) #平均分的刻度分隔
        self.vlaxisY2.setTickCount(11)
        #设置坐标轴名称
        self.dtaxisX.setTitleText("time")
        self.vlaxisY.setTitleText("speed")
        #self.dtaxisX2.setTitleText("time")
        self.vlaxisY2.setTitleText("acceleration")
        #设置网格显示，并设为灰色 
        self.vlaxisY.setGridLineVisible(False)
        #self.vlaxisY.setGridLineColor(Qt.gray)
        self.dtaxisX.setGridLineVisible(True)
        self.dtaxisX.setGridLineColor(Qt.gray)
        self.vlaxisY2.setGridLineVisible(False)
        #self.vlaxisY2.setGridLineColor(Qt.blue)
        #self.dtaxisX2.setGridLineVisible(True)
        #self.dtaxisX2.setGridLineColor(Qt.gray)
        #把坐标轴添加到chart中
        self.chart.addAxis(self.dtaxisX,Qt.AlignBottom)
        self.chart.addAxis(self.vlaxisY,Qt.AlignLeft)
        self.chart.addAxis(self.vlaxisY2,Qt.AlignRight)
        #把曲线关联到坐标轴
        self.acceleration.attachAxis(self.dtaxisX)
        self.acceleration.attachAxis(self.vlaxisY2)
        self.speed.attachAxis(self.dtaxisX)
        self.speed.attachAxis(self.vlaxisY)
# point painting speed
        self.speed_point = QScatterSeries() 
        self.speed_point.setMarkerShape(QScatterSeries.MarkerShape.MarkerShapeCircle)
        self.speed_point.setMarkerSize(10)
        self.speed_point.setName("speed_point")
        self.chart.addSeries(self.speed_point)
        self.speed_point.attachAxis(self.dtaxisX)
        self.speed_point.attachAxis(self.vlaxisY)
        self.speed_point.setPen(QPen(Qt.NoPen));#画出的点不带轮廓
        # pen1 = QPen(Qt.green)
        # pen1.setWidth(1)
        # self.speed_point.setPen(pen1)
# point painting acc
        self.acc_point = QScatterSeries() 
        self.acc_point.setName("acc_point")
        self.acc_point.setMarkerSize(10)
        self.chart.addSeries(self.acc_point)
        self.acc_point.attachAxis(self.dtaxisX)
        self.acc_point.attachAxis(self.vlaxisY2)
        self.acc_point.setPen(QPen(Qt.NoPen))
        # pen2 = QPen(Qt.blue)
        # pen2.setWidth(1)
        # self.acc_point.setPen(pen2)
        self.setChart(self.chart)
    def drawLine(self):
        global pipeson_1,pipeson_5
        speed_tt=0       
        # while TRUE:
        if pipeson_1.poll(): # if any data was receiced               
            msg = pipeson_1.recv() # getting data from pipe 
            
            if(len(msg.time)>0):
                self.speed.clear() 
                # if you draw a line in every time scope,
                # it is recommended to clear the container
                self.acceleration.clear()
                #print ("program av is running ")
                i = 0
                for each in msg.time:
                    self.speed.append(msg.time[i], 3.6*msg.speed[i])
                    if (msg.time[i]<=0.85 and msg.time[i]>=0.65):
                        speed_tt=msg.speed[i]
                    self.acceleration.append(msg.time[i], msg.acceleration[i])
                    i = i+1
            #break
        if pipeson_5.poll():
            self.speed_point.clear()
            self.acc_point.clear()
            realValue_t = pipeson_5.recv()
            speed_t1=3.6*realValue_t.egoEgoStatus.velocityX
            speed_t2=3.6*realValue_t.egoEgoStatus.velocityY
            speed_t=math.sqrt(speed_t1**2+speed_t2**2)
            acc_t1=realValue_t.egoEgoStatus.accerationX
            acc_t2=realValue_t.egoEgoStatus.accerationY
            acc_t=math.sqrt(acc_t1**2+acc_t2**2)

            #acc_t=realValue_t.egoEgoStatus.accerationX
            self.speed_point<<QPointF(0.7,speed_t)#<<QPointF(0.7,speed_tt)
            self.acc_point<<QPointF(0.7,acc_t)

class ChartView2(QChartView,QChart): 
    def __init__(self, *args, **kwargs):
        super(ChartView2, self).__init__(*args, **kwargs)
        self.resize(1200, 800)
        self.setRenderHint(QPainter.Antialiasing) 
        self.chart_init()
        self.timer_init()

    def timer_init(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.drawLine_xy)
        self.timer.start(10)

    def chart_init(self):
        self.chart2 = QChart()
        self.xy = QLineSeries() 
        self.xy.setName("path plan")
        self.chart2.addSeries(self.xy)
        self.dxaxisX = QValueAxis()
        self.dyaxisY = QValueAxis()
        self.dxaxisX.setMin(-20.0)
        self.dxaxisX.setMax(100.0)
        self.dyaxisY.setMin(-7.5)    
        self.dyaxisY.setMax(7.5) 
        self.dxaxisX.setTickCount(30)
        self.dyaxisY.setTickCount(10)
        self.dxaxisX.setTitleText("X")
        self.dyaxisY.setTitleText("Y")
        self.dxaxisX.setGridLineVisible(True)
        self.dxaxisX.setGridLineColor(Qt.gray)
        self.dyaxisY.setGridLineVisible(True)
        self.dyaxisY.setGridLineColor(Qt.gray)
        self.chart2.addAxis(self.dxaxisX,Qt.AlignBottom)
        self.chart2.addAxis(self.dyaxisY,Qt.AlignLeft)
        self.xy.attachAxis(self.dxaxisX)
        self.xy.attachAxis(self.dyaxisY)
#lineleftleftleft
        self.lineleftleftleft = QLineSeries() 
        self.lineleftleftleft.setName("lineleftleftleft")
        self.chart2.addSeries(self.lineleftleftleft)
        self.lineleftleftleft.attachAxis(self.dxaxisX)
        self.lineleftleftleft.attachAxis(self.dyaxisY)

#leftleft
        self.lineleftleft = QLineSeries() 
        self.lineleftleft.setName("lineleftleft")
        self.chart2.addSeries(self.lineleftleft)
        self.lineleftleft.attachAxis(self.dxaxisX)
        self.lineleftleft.attachAxis(self.dyaxisY)

#left
        self.lineleft = QLineSeries() 
        self.lineleft.setName("lineleft")
        self.chart2.addSeries(self.lineleft)
        self.lineleft.attachAxis(self.dxaxisX)
        self.lineleft.attachAxis(self.dyaxisY)

#right
        self.lineright = QLineSeries() 
        self.lineright.setName("lineright")
        self.chart2.addSeries(self.lineright)
        self.lineright.attachAxis(self.dxaxisX)
        self.lineright.attachAxis(self.dyaxisY)

#rightright
        self.linerightright = QLineSeries() 
        self.linerightright.setName("linerightright")
        self.chart2.addSeries(self.linerightright)
        self.linerightright.attachAxis(self.dxaxisX)
        self.linerightright.attachAxis(self.dyaxisY)

#rightrightright
        self.linerightrightright = QLineSeries() 
        self.linerightrightright.setName("linerightrightright")
        self.chart2.addSeries(self.linerightrightright)
        self.linerightrightright.attachAxis(self.dxaxisX)
        self.linerightrightright.attachAxis(self.dyaxisY)

#ego car
        self.egoCar = QScatterSeries() 
        self.egoCar.setName("Ego car")
        self.chart2.addSeries(self.egoCar)
        self.egoCar<<QPointF(0, 0)
        self.egoCar.attachAxis(self.dxaxisX)
        self.egoCar.attachAxis(self.dyaxisY)
        self.setChart(self.chart2)
        self.egoCar.setMarkerShape(QScatterSeries.MarkerShape.MarkerShapeRectangle)
        pen = QPen(Qt.green)
        pen.setWidth(10)
        self.egoCar.setPen(pen)
#ego car replay
        self.egoCar_replay = QScatterSeries() 
        self.egoCar_replay.setName("Ego car replay")
        self.chart2.addSeries(self.egoCar_replay)
        #self.egoCar_replay<<QPointF(0, 0)
        self.egoCar_replay.attachAxis(self.dxaxisX)
        self.egoCar_replay.attachAxis(self.dyaxisY)
        self.setChart(self.chart2)
        self.egoCar_replay.setMarkerShape(QScatterSeries.MarkerShape.MarkerShapeRectangle)
        self.egoCar_replay.setPen(QPen(Qt.NoPen));#画出的点不带轮廓
        # pen = QPen(Qt.green)
        # pen.setWidth(5)
        # self.egoCar_replay.setPen(pen)

#other car
        self.car = QScatterSeries() 
        self.car.setName("car")
        self.chart2.addSeries(self.car)
        self.car.attachAxis(self.dxaxisX)
        self.car.attachAxis(self.dyaxisY)
        self.setChart(self.chart2)
        self.car.setMarkerShape(QScatterSeries.MarkerShape.MarkerShapeRectangle)
        pen = QPen(Qt.red)
        pen.setWidth(5)
        self.car.setPen(pen)
        self.setChart(self.chart2)
#previous point
        # self.Point_pre = QScatterSeries() 
        # self.Point_pre.setName("Point_pre")
        # self.chart2.addSeries(self.Point_pre)
        # self.Point_pre.attachAxis(self.dxaxisX)
        # self.Point_pre.attachAxis(self.dyaxisY)
        # self.setChart(self.chart2)
        # self.Point_pre.setMarkerShape(QScatterSeries.MarkerShape.MarkerShapeCircle)
        # pen = QPen(Qt.black)
        # pen.setWidth(5)
        # self.Point_pre.setPen(pen)
        # self.setChart(self.chart2)

    def drawLine_xy(self):
        global pipeson_2,pipeson_3,pipeson_4,pipeson_5,pipeson_7

        # draw replay car pos
        if pipeson_7.poll():
            #processing replay info
            replayInfo = pipeson_7.recv()
            x_new=replayInfo.X_new
            y_new=replayInfo.Y_new
            x_delta=replayInfo.X_delta
            y_delta=replayInfo.Y_delta
            # realValue = pipeson_5.recv()
            # speed_t=realValue.egoEgoStatus.velocityX
            # acc_t=realValue.egoEgoStatus.accerationX
            # yawRate_t=realValue.egoEgoStatus.yawRate
            # self.xy.clear()
            self.egoCar_replay.clear()

            # cal relative distance 
            # dis_t=speed_t*0.02+0.5*acc_t*0.02*0.02
            # angle_t=(yawRate_t*0.02+0)/2
            # global g_x,g_y
            # g_x+=dis_t*math.cos(angle_t)
            # g_y+=dis_t*math.sin(angle_t)
            # x_delta=g_x-x_new
            # y_delta=g_y-y_new
            # print ("x_delta:"+str(x_delta)+"y_delta:" +str(y_delta))
            self.egoCar_replay<<QPointF(x_delta,y_delta)



        #print ("program xy1 is running %d " ,pipeson_2.poll())
        if pipeson_2.poll():
            ego = pipeson_2.recv()
            self.xy.clear()
            #self.Point_pre.clear()
            i = 0            
            for each in ego.time:
                self.xy.append(QPointF(ego.x[i], ego.y[i]))
                i = i+1
        # if pipeson_6.poll():
        #     point_pre=pipeson_6.recv()
        #     self.Point_pre.clear()
        #     self.Point_pre<<QPointF(point_pre.X_pre,point_pre.Y_pre)
        #print ("program laneLine1 is running %d " ,pipeson_3.poll())
        if pipeson_3.poll():
            lanes = pipeson_3.recv()
        # else:            
        #     lanes=FusionLaneMarker()
        #test_t=FusionLaneMarker().laneboundarys[1].coeffC0
        #print ("program 3 end is running %d ",test_t)
        # if (lanes.laneLineNum!=99):
            self.lineleftleftleft.clear()
            self.lineleftleft.clear()
            self.lineleft.clear()
            self.lineright.clear()
            self.linerightright.clear()
            self.linerightrightright.clear()
            #print ("program lane line processing is running ")
            linetoPlot=[]
            for i in range (0,6,1):

                laneLine_t=lanes.laneboundarys[i]
                c0=laneLine_t.coeffC0
                c1=laneLine_t.coeffC1
                c2=laneLine_t.coeffC2
                c3=laneLine_t.coeffC3[0]
                start=laneLine_t.startX
                end=laneLine_t.endX
                # print ("startX:%d",end)
                distance=end-start
                step=distance/50
                x=[]
                y=[]
                linetoPlot_t=[]        
                for j in range (0,50,1):
                    x_t=start+step*j
                    y_t=c0+c1*x_t+c2*x_t**2+c3*x_t**3
                    linetoPlot_t.append((x_t,y_t))
                linetoPlot.append(linetoPlot_t)
                
            for p in range (0,50,1): 
                #print ("lineleftleft[0][p]:%d,%d",linetoPlot[0][p][0],linetoPlot[0][p][1])
                #print ("lineleft[1][p]:%d,%d",linetoPlot[1][p][0],linetoPlot[1][p][1])
                self.lineleftleftleft.append(linetoPlot[0][p][0],linetoPlot[0][p][1])
                self.lineleftleft.append(linetoPlot[1][p][0],linetoPlot[1][p][1]) 
                self.lineleft.append(linetoPlot[2][p][0],linetoPlot[2][p][1])
                self.lineright.append(linetoPlot[3][p][0],linetoPlot[3][p][1]) 
                self.linerightright.append(linetoPlot[4][p][0],linetoPlot[4][p][1]) 
                self.linerightrightright.append(linetoPlot[5][p][0],linetoPlot[5][p][1])
        if pipeson_4.poll():
            carInfo = pipeson_4.recv()
            self.car.clear()
            #a_t=PredictionObstacles()
            #size_t=len(carInfo.obstacle_vector)
            for i in carInfo.obstacle_vector:
                info_t=i.perception_obstacle
                posX=info_t.posX
                posY=info_t.posY
                self.car << QPointF(posX, posY)

class ChartView3(QChartView,QChart): 
    def __init__(self, *args, **kwargs):
        super(ChartView3, self).__init__(*args, **kwargs)
        self.resize(600, 400)
        self.setRenderHint(QPainter.Antialiasing) 
        self.chart_init()
        self.timer_init()
        self.steeringWheelAngleReq_dps=0
        self.cur_0_7=0

    def setAxisRange(self):
        global timeCounter
        self.dtaxisX.setMin((timeCounter-10))
        self.dtaxisX.setMax((timeCounter+10))
        #self.vlaxisY.setMin(self.steeringWheelAngleReq_dps-15)#in degree    
        #self.vlaxisY.setMax(self.steeringWheelAngleReq_dps+15) #设置y轴最大值
        self.vlaxisY2.setMin(self.cur_0_7-0.005)    
        self.vlaxisY2.setMax(self.cur_0_7+0.005) #设置y轴最大值
        self.setChart(self.chart)

    def timer_init(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.setAxisRange)
        self.timer.timeout.connect(self.drawLine)        
        self.timer.start(10)

    def chart_init(self):
        self.chart = QChart()
        self.ppSteeringangle = QSplineSeries() 
        self.Cur_pre = QSplineSeries() 
        self.ppSteeringangle.setName("ppSteeringangle")
        self.Cur_pre.setName("Cur_pre")
        self.chart.addSeries(self.ppSteeringangle)
        self.chart.addSeries(self.Cur_pre)
        self.dtaxisX = QValueAxis()
        self.vlaxisY = QValueAxis()# steering
        #self.dtaxisX2 = QValueAxis()
        self.vlaxisY2 = QValueAxis()#curvature
        self.dtaxisX.setMin((timeCounter-10))
        self.dtaxisX.setMax((timeCounter+10))
        self.vlaxisY.setMin(-30.0)#in degree    
        self.vlaxisY.setMax(30.0) 
        self.vlaxisY2.setMin(-0.05)    
        self.vlaxisY2.setMax(0.05) 
        self.dtaxisX.setTickCount(15)
        self.vlaxisY.setTickCount(18)
        self.vlaxisY2.setTickCount(10)
        self.dtaxisX.setTitleText("time")
        self.vlaxisY.setTitleText("ppSteeringangle")
        self.vlaxisY2.setTitleText("Cur_pre")
        self.vlaxisY.setGridLineVisible(False)
        #self.vlaxisY.setGridLineColor(Qt.gray)
        self.dtaxisX.setGridLineVisible(True)
        self.dtaxisX.setGridLineColor(Qt.gray)
        self.vlaxisY2.setGridLineVisible(False)
        #self.vlaxisY2.setGridLineColor(Qt.blue)
        #self.dtaxisX2.setGridLineVisible(True)
        #self.dtaxisX2.setGridLineColor(Qt.gray)
        self.chart.addAxis(self.dtaxisX,Qt.AlignBottom)
        self.chart.addAxis(self.vlaxisY,Qt.AlignLeft)
        self.chart.addAxis(self.vlaxisY2,Qt.AlignRight)
        self.ppSteeringangle.attachAxis(self.dtaxisX)
        self.ppSteeringangle.attachAxis(self.vlaxisY)
        self.Cur_pre.attachAxis(self.dtaxisX)
        self.Cur_pre.attachAxis(self.vlaxisY2)
        self.setChart(self.chart)
    def drawLine(self):
        global pipeson_6
        global timeCounter   
        timeCounter +=0.01 
        # while TRUE:
        if pipeson_6.poll():             
            msg = pipeson_6.recv()
            #self.speed_point.clear()
            #self.acc_point.clear()
            self.steeringWheelAngleReq_dps=msg.SteeringWheelAngleReq_rad*57.3
            #print ("steeringWheelAngleReq_dps %f",self.steeringWheelAngleReq_dps)
            self.ppSteeringangle<<QPointF(timeCounter,self.steeringWheelAngleReq_dps)
        # if pipeson_2.poll():
        #     msg2 = pipeson_2.recv()
        #     if len(msg2.curvature)>=80:
        #         self.cur_0_7=msg2.curvature[80]# temp
        #     else:
        #         self.cur_0_7=msg2.curvature[0]
        #     self.Cur_pre<<QPointF(timeCounter,self.cur_0_7)
#plot replay trace 

class ChartView4(QChartView,QChart): 
    def __init__(self, *args, **kwargs):
        super(ChartView4, self).__init__(*args, **kwargs)
        self.resize(600, 400)
        self.setRenderHint(QPainter.Antialiasing) 
        self.trace_x_max=0
        self.trace_x_min=0
        self.trace_y_max=0
        self.trace_y_min=0
        self.chart_init()
        self.timer_init()


    def setAxisRange(self):

        self.axisX.setMin((self.trace_x_min-10))
        self.axisX.setMax((self.trace_x_max+10))
        self.axisY.setMin(self.trace_y_min-10)#in degree    
        self.axisY.setMax(self.trace_y_max+10) #设置y轴最大值
        self.setChart(self.chart)

    def timer_init(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.setAxisRange)
        self.timer.timeout.connect(self.drawTrace)        
        self.timer.start(10)


    def chart_init(self):
        self.chart = QChart()
        self.trace_replay = QScatterSeries()
        self.trace_ori= QScatterSeries()
        #self.Cur_pre = QSplineSeries() 
        self.trace_replay.setName("trace_replay")
        self.trace_ori.setName("trace_ori")
        #self.Cur_pre.setName("Cur_pre")
        self.chart.addSeries(self.trace_replay)
        self.chart.addSeries(self.trace_ori)
        #self.chart.addSeries(self.Cur_pre)
        self.axisX = QValueAxis()
        self.axisY = QValueAxis()
        #self.dtaxisX2 = QValueAxis()

        self.axisX.setMin(0)
        self.axisX.setMax((10))
        self.axisY.setMin((-5))#in degree    
        self.axisY.setMax((5)) 
        self.axisX.setTickCount(2)
        self.axisY.setTickCount(2)

        self.axisX.setTitleText("trace_X")
        self.axisY.setTitleText("trace_Y")
        self.axisY.setGridLineVisible(False)
        self.axisX.setGridLineVisible(False)

        self.chart.addAxis(self.axisX,Qt.AlignBottom)
        self.chart.addAxis(self.axisY,Qt.AlignLeft)
        self.trace_replay.attachAxis(self.axisX)
        self.trace_replay.attachAxis(self.axisY)
        self.trace_ori.attachAxis(self.axisX)
        self.trace_ori.attachAxis(self.axisY)
        self.trace_replay.setMarkerShape(QScatterSeries.MarkerShape.MarkerShapeRectangle)
        self.trace_replay.setPen(QPen(Qt.NoPen))
        self.trace_ori.setMarkerShape(QScatterSeries.MarkerShape.MarkerShapeRectangle)
        self.trace_ori.setPen(QPen(Qt.NoPen))
        
        self.setChart(self.chart)

    def drawTrace(self):
        global pipeson_8
        # while TRUE:
        if pipeson_8.poll():             
            msg = pipeson_8.recv()
            #self.X.clear()
            #self.Y.clear()            
            X=msg.X_new
            X_ori=X-msg.X_delta
            Y=msg.Y_new
            Y_ori=Y-msg.Y_delta

            self.trace_x_max=max(X,self.trace_x_max)
            self.trace_x_min=min(X,self.trace_x_min)
            Y=msg.Y_new
            self.trace_y_max=max(Y,self.trace_y_max)
            self.trace_y_min=min(Y,self.trace_y_min)
            #print("trace data:",X,Y)
            self.trace_replay<<QPointF(X,Y)
            self.trace_ori<<QPointF(X_ori,Y_ori)

# passing by the data with ros callback method 
def callback(msg):
    global pipefather_1
    if False==pipefather_1.poll():
        pipefather_1.send(msg)
    global pipefather_2
    if False==pipefather_2.poll():
        pipefather_2.send(msg)
    
    #print ("callback is running ")
def callback2(msg):
    global pipefather_3
    if False==pipefather_3.poll():
        pipefather_3.send(msg)
    #print ("callback2 is running ")
def callback3(msg):
    global pipefather_4
    if False==pipefather_4.poll():
        pipefather_4.send(msg)
def callback4(msg):
    global pipefather_5
    if False==pipefather_5.poll():
        pipefather_5.send(msg)
def callback5(msg):
    global pipefather_6
    if False==pipefather_6.poll():
        pipefather_6.send(msg)
def callback6(msg):
    global pipefather_7,pipefather_8 
    if False==pipefather_7.poll():
        pipefather_7.send(msg)
    if False==pipefather_8.poll():
        pipefather_8.send(msg)    

def start_ros():
    rospy.init_node('chart')
    rospy.Subscriber('/planning/Trajectory_toCtrl', Trajectory_to_Control, callback,queue_size=10)
    rospy.Subscriber('/udp2ros/fusion_lanelines', FusionLaneMarker, callback2,queue_size=10)
    rospy.Subscriber('/udp2ros/PredictionInfo', PredictionObstacles, callback3,queue_size=10)
    rospy.Subscriber('/udp2ros/ESAInfo', ESAInfo, callback4,queue_size=10)
    rospy.Subscriber('/ppcontroller/PPOUT',PPOUT,callback5,queue_size=10)
    rospy.Subscriber('replaytrace/NewTrace',NewTrace,callback6,queue_size=10)
    rospy.spin()



def start_chart():
    app = QApplication(sys.argv)
    # print ("main flow1 ")
    # job1 = mp.Process(target=mutiThread1, args=())
    # job2 = mp.Process(target=mutiThread2, args=())
    # job3 = mp.Process(target=mutiThread3, args=())
    # job1.start()
    # job2.start()
    # job3.start()
    view_1 = ChartView()
    view_1.show()
    view_1.drawLine()
    view_2 = ChartView2()
    view_2.show()
    view_2.drawLine_xy()
    view_3 = ChartView3()
    view_3.show()
    view_3.drawLine()
    view_4 = ChartView4()
    view_4.show()
    view_4.drawTrace()
    # print ("main flow2 ")
    sys.exit(app.exec_())
# def mutiThread1():
#     view_1 = ChartView()
#     view_1.show()
#     view_1.drawLine()
# def mutiThread2():
#     view_2 = ChartView2()
#     view_2.show()
#     view_2.drawLine_xy()
# def mutiThread3():
#     view_3 = ChartView3()
#     view_3.show()
#     view_3.drawLine()
    

if __name__ == "__main__":

    job = mp.Process(target=start_ros, args=())
    #creating a sub Thread
    #main tread for painting and processing data and sub ros thread do reciving     
    job.start()
    start_chart()
