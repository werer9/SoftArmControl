# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
#      Open CV and Numpy integration          #
###############################################
from yolov5.functions import *
from yolov5.softArmCMD import *
from yolov5.detect import *
from yolov5.rotTest import rotCNN
from Chatbot.controller import Controller
from yolov5 import commands
import queue
from intent import intent
#LINE 164 ALSO COMMENTED OUT for CHATBOT
#from Chatbot import chatbot#############################################



def depth(new_intent):
    
    # set softarm controller with IP address of raspberry pi
    #controller = Controller("http://172.22.0.75:5000")
    controller = Controller("http://192.168.1.20:5000")
    # get yolov5 input args
    opt = getInputArgs()
    # initialise intel realsense pipeline
    pipeline = initRealsense()
    # load yolov5 softarm detection model
    model = PredictYolo5(**vars(opt))
    # load rotation estimation CNN model
    rotModel = rotCNN()
    # initialise temporal variables
    first = 1
    tick = 0
    oldHumanPos = [0, 0, 0]
    oldSegment1Pos = [0, 0, 0]
    oldSegment2Pos = [0, 0, 0]
    oldSegment3Pos = [0, 0, 0]
    oldSegment3PosBottom = [0, 0, 0]
    angleRotate = 0
    init = 1
    
    try:
        while True:

            # get frames from realsense camera
            color_image, depth_image, depthW, depthH = getFrames(pipeline)
            # initialisation code
            if init:
                color_image_old = color_image.copy()
                color_image_old1 = color_image.copy()
                init = 0
                oldAngle = 0
                softArmCommands = [0, 0, 0]
                pastArmCMD = [0, 0]
                fpsTime = time.time_ns()
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # get yolov5 predictions
            # Pred: x1, y1 (top left), x2, y2 (bottom right), confidence, class

            color_image, pred, color_image_nobox = model.getPrediction(color_image)

            pred = pred[0]
            pred = pred.cpu().detach().numpy()
            # get position of joints
            bearX, bearY = getBear(pred)

            joint1x, joint1y, joint2x, joint2y = getJoints(pred)
            joint2Combo = [joint2x, joint2y]
            # track position of closest humans
            faceX, faceY, trackDistance, predRowFace = trackHumanLastPos(pred, oldHumanPos, depth_image, color_image, 5,
                                                                         0,
                                                                         joint2Combo)
            # track position of softarm segments
            segment1x, segment1y, segment1Distance, predRowsegment1 = trackHumanLastPos(pred, oldSegment1Pos,
                                                                                        depth_image,
                                                                                        color_image, 1, 0, joint2Combo)
            segment2x, segment2y, segment2Distance, predRowsegment2 = trackHumanLastPos(pred, oldSegment2Pos,
                                                                                        depth_image,
                                                                                        color_image, 2, 0, joint2Combo)
            segment3x, segment3y, segment3Distance, predRowsegment3 = trackHumanLastPos(pred, oldSegment3Pos,
                                                                                        depth_image,
                                                                                        color_image, 3, 2, joint2Combo)
            segment3xBottom, segment3yBottom, segment3DistanceBottom, predRowsegment3Bottom = trackHumanLastPos(pred,
                                                                                                                oldSegment3PosBottom,
                                                                                                                depth_image,
                                                                                                                color_image,
                                                                                                                3, 1,
                                                                                                                joint2Combo)
            bottomJointx, bottomJointy, topJointx, topJointy = getBottomJoint(segment1x, segment1y, joint1x, joint1y,
                                                                              joint2x, joint2y)
            oldHumanPos = [faceX, faceY, trackDistance]
            oldSegment1Pos = [segment1x, segment1y, segment1Distance]
            oldSegment2Pos = [segment2x, segment2y, segment2Distance]
            oldSegment3Pos = [segment3x, segment3y, segment3Distance]
            oldSegment3PosBottom = [segment3xBottom, segment3yBottom, segment3DistanceBottom]
            # get depth coordiantes scaled for resolution of RGB coordiantes
            DepthX, DepthY = DepthScale(depth_image, color_image, faceX, faceY)
            Depthsegment1x, Depthsegment1y = DepthScale(depth_image, color_image, segment1x, segment1y)
            Depthsegment2x, Depthsegment2y = DepthScale(depth_image, color_image, segment2x, segment2y)
            Depthsegment3x, Depthsegment3y = DepthScale(depth_image, color_image, segment3x, segment3y)
            Depthsegment3xBottom, Depthsegment3yBottom = DepthScale(depth_image, color_image, segment3xBottom,
                                                                    segment3yBottom)
            DepthBottomJointx, DepthBottomJointy = DepthScale(depth_image, color_image, bottomJointx, bottomJointy)
            
            DepthTopJointx, DepthTopJointy = DepthScale(depth_image, color_image, topJointx, topJointy)
            DepthBearX, DepthBearY = DepthScale(depth_image, color_image, bearX, bearY)
            # ################################ draw dots on tracked objects ################################
            humanPresent = 0
        
            if checkIfPresent(pred, 5):
                if faceX != 0 and faceY != 0:
                    color_image = cv2.circle(color_image, (faceX, faceY), radius=3, color=(0, 0, 255), thickness=3)
                    depth_colormap = cv2.circle(depth_colormap, (DepthX, DepthY), radius=3, color=(0, 0, 255),
                                                thickness=3)
                    humanPresent = 1
            if segment1x != 0 and segment1y != 0:
                color_image = cv2.circle(color_image, (segment1x, segment1y), radius=3, color=(0, 255, 255),
                                         thickness=3)
                depth_colormap = cv2.circle(depth_colormap, (Depthsegment1x, Depthsegment1y), radius=3,
                                            color=(0, 255, 255),
                                            thickness=3)
            if segment2x != 0 and segment2y != 0:
                color_image = cv2.circle(color_image, (segment2x, segment2y), radius=3, color=(255, 255, 255),
                                         thickness=3)
                depth_colormap = cv2.circle(depth_colormap, (Depthsegment2x, Depthsegment2y), radius=3,
                                            color=(255, 255, 255), thickness=3)
            if segment3x != 0 and segment3y != 0:
                color_image = cv2.circle(color_image, (segment3x, segment3y), radius=3, color=(0, 255, 0), thickness=3)
                depth_colormap = cv2.circle(depth_colormap, (Depthsegment3x, Depthsegment3y), radius=3,
                                            color=(0, 255, 0),
                                            thickness=3)
            if segment3xBottom != 0 and segment3yBottom != 0:
                depth_colormap = cv2.circle(depth_colormap, (Depthsegment3xBottom, Depthsegment3yBottom), radius=5,
                                            color=(255, 255, 0), thickness=3)
            if bottomJointx != 0 and bottomJointy != 0:
                color_image = cv2.circle(color_image, (bottomJointx, bottomJointy), radius=3, color=(125, 255, 125),
                                         thickness=3)
                depth_colormap = cv2.circle(depth_colormap, (DepthBottomJointx, DepthBottomJointy), radius=3,
                                            color=(125, 255, 125), thickness=3)
            if topJointx != 0 and topJointy != 0:
                color_image = cv2.circle(color_image, (topJointx, topJointy), radius=3, color=(125, 125, 0),
                                         thickness=3)
                depth_colormap = cv2.circle(depth_colormap, (DepthTopJointx, DepthTopJointy), radius=3,
                                            color=(125, 125, 255), thickness=3)
            if bearX != -1 and bearY != -1:
                color_image = cv2.circle(color_image, (bearX, bearY), radius=3, color=(125, 125, 0), thickness=3)
                depth_colormap = cv2.circle(depth_colormap, (DepthBearX, DepthBearY), radius=3, color=(125, 125, 0),
                                            thickness=3)
                                            
            # get joint angles
            angle1 = -int(get_angle(segment2x, segment2y, bottomJointx, bottomJointy))
            angle2 = -int(get_angle(segment3x, segment3y, topJointx, topJointy)) - angle1
            # get rotation angle

            angleRotateNew, _ = getRotateAngle(pred, predRowsegment3Bottom, color_image_nobox, color_image,
                                               segment3yBottom,
                                               segment3xBottom, color_image_old, color_image_old1, rotModel, oldAngle)
            oldAngle = angleRotateNew
            if angleRotateNew != -1:
                angleRotate = angleRotateNew
            # ###############################     SOFTARM CONTROL CODE   ###################################

            # softarmcommands -> [joint angles, rotation angle, command type ] (0 = neutral, 1 = bear,2 = human,
            # 3=headpat. 4 =deflate, 5= speak, 6=wave)

           # bot = chatbot.Chatbot("robotarm-315611", "123", "en")
           # [intent_string, _] = bot.get_user_intent([input("Hello, what do you want to do?\n")])
            
            #command = intent_string
            
            command  = new_intent.intent
         
           
            
            print("Sending command: ", command)
            # send command to arm
            print(DepthBottomJointy)
            pastArmCMD, softArmCommands, greenRate, yellowRate, baseRate = sendArmCMD(softArmCommands,
                                                                                      angle1,
                                                                                      angle2, angleRotate, controller,
                                                                                      command, bearX, bearY,
                                                                                      DepthBearX, DepthBearY,
                                                                                      depth_image, bottomJointx,
                                                                                      bottomJointy, DepthBottomJointx,
                                                                                      DepthBottomJointy, faceX, faceY,
                                                                                      DepthX, DepthY, humanPresent,
                                                                                      segment3x, segment3y,
                                                                                      Depthsegment3x,
                                                                                      Depthsegment3y)

            # DISPLAY CODE
            depth_colormap_dim = depth_colormap.shape
            color_image_dim = color_image.shape
            color_colormap_dim = color_image.shape
            now = time.time_ns()
            fps = 1000000000 / (now - fpsTime)
            fpsTime = time.time_ns()
            color_image = putText(color_image, angle1, angle2, angleRotate, softArmCommands, fps, greenRate, yellowRate,
                                  baseRate, command)

            if tick == 10:
                print("angle R: ", angleRotate)
                tick = 0
            showDepth = 0
            if showDepth:
                # If depth and color resolutions are different, resize color image to match depth image for display
                if depth_colormap_dim != color_colormap_dim:
                    resized_depth_colormap = cv2.resize(depth_colormap, dsize=(
                        int(color_colormap_dim[1] / 2), int(color_colormap_dim[0] / 2)), interpolation=cv2.INTER_AREA)
                    resized_color = cv2.resize(color_image,
                                               dsize=(int(color_colormap_dim[1] / 2), int(color_colormap_dim[0] / 2)),
                                               interpolation=cv2.INTER_AREA)
                    images = np.hstack((resized_color, resized_depth_colormap))
                else:
                    images = np.hstack((color_image, depth_colormap))
            else:
                h,w,_= color_image.shape
                scale=1
                images = cv2.resize(color_image,(int(w*scale),int(h*scale)))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key =cv2.waitKey(1)
            tick += 1
            #keep past frame for denoise
            color_image_old1 = color_image_old.copy()
            color_image_old = color_image_nobox.copy()
            #destroy window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break


    finally:

        # Stop streaming
        pipeline.stop()
