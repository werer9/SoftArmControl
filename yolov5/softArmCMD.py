import math
import threading

from yolov5 import commands


def pointAtBear(softArmCommands, point_at_bear, bearX, bearY, DepthBearX, DepthBearY, depth_image, bottomJointx,
                bottomJointy, DepthBottomJointx, DepthBottomJointy):
    if point_at_bear and bearX != -1 and bearY != -1:
        # calculate angle between arm bottom joint and bear
        xDiff = bearX - bottomJointx
        yDiff = bearY - bottomJointy
        # inverse tan(opposite/adjacent) -> inverse tan(yDiff/xDiff)
        bearArmAngle = -int(math.degrees(math.atan(yDiff / xDiff)))
        # print("bear angle", bearArmAngle)
        bearDepth = depth_image[DepthBearY][DepthBearX]
        # print("bdepth", bearDepth)
        jointDepth = float(depth_image[DepthBottomJointy][DepthBottomJointx])
        # print("AD", jointDepth)
        depthDiff = float(jointDepth - bearDepth)
        # print("DEPTH DIFF", depthDiff)
        # inverse tan(oposite/adjacent)-> inverse tan(depthDiff/xDiff)
        bearArmRotAngle = -int(math.degrees(math.atan(depthDiff / xDiff)))
        #  print("rot angle", bearArmRotAngle)
        # returns joint angles, rotation angle and command type (0 = neutral, 1 = bear,2 = human, 3=headpat)
        return [bearArmAngle, bearArmRotAngle, 1]
    else:
        softArmCommands[2] = 1
        return softArmCommands


def pointAtHuman(softArmCommands, humanX, humanY, DepthhumanX, DepthhumanY, depth_image,
                 bottomJointx, bottomJointy, DepthBottomJointx, DepthBottomJointy, humanPresent):
    if humanPresent:
        # calculate angle between arm bottom joint and human
        xDiff = humanX - bottomJointx
        yDiff = humanY - bottomJointy
        # inverse tan(opposite/adjacent) -> inverse tan(yDiff/xDiff)
        humanArmAngle = -int(math.degrees(math.atan(yDiff / xDiff)))
        humanDepth = depth_image[DepthhumanY][DepthhumanX]
        jointDepth = float(depth_image[DepthBottomJointy][DepthBottomJointx])
        depthDiff = float(jointDepth - humanDepth)
        # inverse tan(oposite/adjacent)-> inverse tan(depthDiff/xDiff)
        humanArmRotAngle = -int(math.degrees(math.atan(depthDiff / xDiff)))
        # returns joint angles, rotation angle and command type (0 = neutral, 1 = human,2 = human, 3 = headpat)
        return [humanArmAngle, humanArmRotAngle, 2]
    else:
        softArmCommands[2] = 2
        return softArmCommands


def headpat(softArmCommands, point_at_human, humanX, humanY, DepthhumanX, DepthhumanY, depth_image,
            bottomJointx, bottomJointy, DepthBottomJointx, DepthBottomJointy, humanPresent, segment3x, segment3y,
            Depthsegment3x, Depthsegment3y):
    if humanPresent:
        # calculate angle between arm bottom joint and human
        xDiff = humanX - bottomJointx
        yDiff = humanY - bottomJointy
        # inverse tan(opposite/adjacent) -> inverse tan(yDiff/xDiff)
        humanArmAngle = -int(math.degrees(math.atan(yDiff / xDiff)))
        humanDepth = depth_image[DepthhumanY][DepthhumanX]
        jointDepth = float(depth_image[DepthBottomJointy][DepthBottomJointx])
        depthDiff = float(jointDepth - humanDepth)
        # inverse tan(oposite/adjacent)-> inverse tan(depthDiff/xDiff)
        humanArmRotAngle = -int(math.degrees(math.atan(depthDiff / xDiff)))
        # returns joint angles, rotation angle and command type (0 = neutral, 1 = human,2 = human, 3=headpat)
        return [humanArmAngle, humanArmRotAngle, 3]
    else:
        softArmCommands[2] = 3
        return softArmCommands


def trySendCMD(greenRate, yellowRate, baseRate, controller):
    controller.set_green(float(greenRate))
    controller.set_yellow(float(yellowRate))
    controller.set_base(float(baseRate))


def sendAngleCMD(greenRate, yellowRate, baseRate, inflationRate, controller):
    # Change control parameters

    allCMD = [greenRate, yellowRate, baseRate, inflationRate]
    print("SENDING ANGLE COMMAND", allCMD)
    # trySendCMD(greenRate*rateReduction, yellowRate*rateReduction,
    # baseRate*angleReduction, inflationRate, controller)
    x = threading.Thread(target=trySendCMD, args=(greenRate, yellowRate, baseRate, inflationRate, controller))
    x.start()


def sendArmCMD(softArmCommands, angle1, angle2, angleRotate, controller, command, bearX, bearY, DepthBearX,
               DepthBearY, depth_image, bottomJointx, bottomJointy, DepthBottomJointx, DepthBottomJointy, faceX, faceY,
               DepthX, DepthY, humanPresent, segment3x, segment3y, Depthsegment3x, Depthsegment3y):
    # calculate commands
    if command == commands.DEFLATE:
        softArmCommands = [0, 0, 4]
    elif command == commands.NEUTRAL:
        softArmCommands = [0, 0, 0]
    elif command == commands.BEAR:
        softArmCommands = pointAtBear(softArmCommands, bearX, bearY, DepthBearX, DepthBearY,
                                      depth_image, bottomJointx, bottomJointy, DepthBottomJointx, DepthBottomJointy)
    elif command == commands.HUMAN:
        softArmCommands = pointAtHuman(softArmCommands, faceX, faceY, DepthX, DepthY,
                                       depth_image, bottomJointx, bottomJointy, DepthBottomJointx, DepthBottomJointy,
                                       humanPresent)
    elif command == commands.HEADPAT:

        softArmCommands = headpat(softArmCommands, faceX, faceY, DepthX, DepthY,
                                  depth_image, bottomJointx, bottomJointy, DepthBottomJointx, DepthBottomJointy,
                                  humanPresent, segment3x, segment3y, Depthsegment3x, Depthsegment3y)
    elif command == commands.SPEAK:
        softArmCommands = [0, 0, 5]
    elif command == commands.WAVE:
        softArmCommands = [0, 0, 6]
    else:
        # neutral
        softArmCommands = [0, 0, 0]
    # ##################################### execute commands  ########################################
    # softarmcommands -> [joint angles, rotation angle, command type ]
    # (0 = neutral, 1 = bear,2 = human, 3=headpat. 4 =deflate, 5= speak, 6=wave)
    if softArmCommands[2] == 4:
        print("mode 4 deflate")

    elif softArmCommands[2] == 0:
        print("mode 0 neutral")
    elif softArmCommands[2] == 1:
        print("mode 1 bear")
        angle = softArmCommands[0]
        rotateAngle = softArmCommands[1]

        inflationRate = 1
        baseRate = 1
        yellowRate = 0
        greenRate = 0
        if angle2 <= -90:
            yellowRate = -0.1
        sendAngleCMD(-greenRate, yellowRate, baseRate * 0.4, inflationRate, controller)
    elif (softArmCommands[2] == 2):
        print("mode 2 human")
    elif (softArmCommands[2] == 3):
        print("mode 3 headpat")
    elif (softArmCommands[2] == 5):
        print("mode 5 speak")
    elif (softArmCommands[2] == 6):
        print("mode 6 wave")
    pastArmCMD = softArmCommands
    return pastArmCMD, softArmCommands, greenRate, yellowRate, baseRate
