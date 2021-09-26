import math
import threading

from yolov5 import commands


def pointAtBear(softArmCommands, bearX, bearY, DepthBearX, DepthBearY, depth_image, bottomJointx,
                bottomJointy, DepthBottomJointx, DepthBottomJointy):
    if  bearX != -1 and bearY != -1:
        # calculate angle between arm bottom joint and bear
        xDiff = bearX - bottomJointx
        yDiff = bearY - bottomJointy
        # inverse tan(opposite/adjacent) -> inverse tan(yDiff/xDiff)
        if(xDiff !=0 and yDiff !=0):
            bearArmAngle = -int(math.degrees(math.atan(yDiff / xDiff)))
            bearDepth = depth_image[DepthBearY][DepthBearX]
            jointDepth = float(depth_image[DepthBottomJointy][DepthBottomJointx])
            # print("AD", jointDepth)
            depthDiff = float(jointDepth - bearDepth)
            # inverse tan(oposite/adjacent)-> inverse tan(depthDiff/xDiff)
            bearArmRotAngle = -int(math.degrees(math.atan(depthDiff / xDiff)))
            # returns joint angles, rotation angle and command type (0 = neutral, 1 = bear,2 = human, 3=headpat)
            if(bearX<bottomJointx):
                bearArmAngle= -bearArmAngle
            return [bearArmAngle, bearArmRotAngle, 1]
        else:
            return softArmCommands
    else:
        softArmCommands[2] = 1
        return softArmCommands



def trySendCMD(greenRate, yellowRate, baseRate, controller):
    if(0):
        controller.set_green(float(greenRate))
        controller.set_yellow(float(yellowRate))
        controller.set_base(float(baseRate))


def sendAngleCMD(greenRate, yellowRate, baseRate, inflationRate, controller):
    # Change control parameters

    allCMD = [greenRate, yellowRate, baseRate, inflationRate]
    print("SENDING ANGLE COMMAND", allCMD)
    # trySendCMD(greenRate*rateReduction, yellowRate*rateReduction,
    # baseRate*angleReduction, inflationRate, controller)
    x = threading.Thread(target=trySendCMD, args=(greenRate, yellowRate, baseRate,  controller))
    x.start()


def sendArmCMD(softArmCommands, angle1, angle2, angleRotate, controller, command, bearX, bearY, DepthBearX,
               DepthBearY, depth_image, bottomJointx, bottomJointy, DepthBottomJointx, DepthBottomJointy, faceX, faceY,
               DepthX, DepthY, humanPresent, segment3x, segment3y, Depthsegment3x, Depthsegment3y):
    ##########################  angle calculation ########
    if command == "Deflate":
        softArmCommands = [0, 0, 4]
    elif command == "NEUTRAL":
        softArmCommands = [0, 0, 0]
    elif command == "Bear":
        softArmCommands = pointAtBear(softArmCommands, bearX, bearY, DepthBearX, DepthBearY,
                                      depth_image, bottomJointx, bottomJointy, DepthBottomJointx, DepthBottomJointy)
    elif command == "Human":
                softArmCommands = pointAtBear(softArmCommands, faceX, faceY, DepthX, DepthY,
                                      depth_image, bottomJointx, bottomJointy, DepthBottomJointx, DepthBottomJointy)
    elif command == "Headpat":

        softArmCommands = headpat(softArmCommands, faceX, faceY, DepthX, DepthY,
                                  depth_image, bottomJointx, bottomJointy, DepthBottomJointx, DepthBottomJointy,
                                  humanPresent, segment3x, segment3y, Depthsegment3x, Depthsegment3y)

    else:
        # neutral
        softArmCommands = [0, 0, 0]
    # ##################################### execute commands  ########################################
    # softarmcommands -> [joint angles, rotation angle, command type ]
    # (0 = neutral, 1 = bear,2 = human, 3=headpat. 4 =deflate, 5= speak, 6=wave)
    if command == "Deflate":
        print("mode 4 deflate")

    elif command == "Neutral":
        print("mode 0 neutral")
        baseRate = 1
        yellowRate = 0
        greenRate = 0
    elif command == "Bear":
        print("mode 1 bear")
        angle = softArmCommands[0]
        rotateAngle = softArmCommands[1]
        inflationRate = 1
        if(angle!=0):
            greenRate =-max( min(((angle1+angle2)-angle)/20 ,1),-1)
            if( greenRate> -0.9 and greenRate<0):
                greenRate=0    
        else:
            greenRate=0       
        if(angle!=0):
            yellowRate =-max( min(((angle1+angle2)-angle)/20 ,1),-1)
            if(yellowRate> -0.9 and yellowRate<0):
                yellowRate=0       
        else:
            yellowRate=0
        #base rotation
        if(rotateAngle!=0):
            baseRate= -(max(min(float(1- (angleRotate/rotateAngle)), 1),-1))
            #if(rotateAngle<0):
            #    baseRate = -1*baseRate           
        else:
            baseRate=0
          
        sendAngleCMD(-greenRate, yellowRate, baseRate * 0.4, inflationRate, controller)
    elif command == "Human":
        print("mode 2 human")
        angle = softArmCommands[0]
        rotateAngle = softArmCommands[1]
        inflationRate = 1
        if(angle!=0):
            greenRate =-max( min(((angle1+angle2)-angle)/20 ,1),-1)
            if( greenRate> -0.9 and greenRate<0):
                greenRate=0    
        else:
            greenRate=0       
        if(angle!=0):
            yellowRate =-max( min(((angle1+angle2)-angle)/20 ,1),-1)
            if(yellowRate> -0.9 and yellowRate<0):
                yellowRate=0       
        else:
            yellowRate=0
        #base rotation
        if(rotateAngle!=0):
            baseRate= -(max(min(float(1- (angleRotate/rotateAngle)), 1),-1))
            #if(rotateAngle<0):
            #    baseRate = -1*baseRate           
        else:
            baseRate=0
        sendAngleCMD(-greenRate, yellowRate, baseRate * 0.4, inflationRate, controller)
    elif command == "Headpat":
        print("mode 3 headpat")
    else:
        baseRate = 1
        yellowRate = 0
        greenRate = 0
        
    pastArmCMD = softArmCommands
    return pastArmCMD, softArmCommands, greenRate, yellowRate, baseRate
