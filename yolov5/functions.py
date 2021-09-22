import pyrealsense2 as rs
import math
from yolov5.detect import *
#from depth import pred, depth_image

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)


def getface(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )
    facex = 0
    facey = 0
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        facex = x + (w / 2)
        facey = y + (h / 2)

    return frame, int(facex), int(facey)


def get_angle(x1, y1, x2, y2):
    diffX = x1 - x2
    diffY = y1 - y2
    angle = 0
    if diffX != 0:
        toa = diffY / diffX
        angle = math.degrees(math.atan(toa))

    return angle


def getBottomJoint(seg1x, seg1y, joint1x, joint1y, joint2x, joint2y):
    distance1 = abs(seg1x - joint1x) + abs(seg1y - joint1y)
    distance2 = abs(seg1x - joint2x) + abs(seg1y - joint2y)
    if distance1 < distance2:
        return joint1x, joint1y, joint2x, joint2y
    return joint2x, joint2y, joint1x, joint1y


def DepthScale(depthimg, image, w, h):
    x, y, _ = image.shape
    dx, dy = depthimg.shape
    scaleX = dy / y
    scaleY = dx / x
    return int(w * scaleX), int(h * scaleY)


def getJoints(predIn):
    joint1x = 0
    joint1y = 0
    joint2x = 0
    joint2y = 0
    for index, row in enumerate(predIn):
        if row[5] == 0:
            if joint1x == 0 and joint1y == 0:
                joint1x = (row[0] + row[2]) / 2
                joint1y = (row[1] + row[3]) / 2
            else:
                joint2x = (row[0] + row[2]) / 2
                joint2y = (row[1] + row[3]) / 2

    return int(joint1x), int(joint1y), int(joint2x), int(joint2y)


def crop_center(image, cropx, cropy):
    y, x = image.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return image[starty:starty + cropy, startx:startx + cropx], cropx, cropy


def trackHumanHighPred(depthImage, image, trackClass):
    # initial

    maximum = 0
    maxIndex = 0
    for index, row in enumerate(pred):
        if int(row[5]) == trackClass:
            if row[4] > maximum:
                maximum = row[4]
                maxIndex = index
    xCenter = (pred[maxIndex][0] + pred[maxIndex][2]) / 2
    yCenter = (pred[maxIndex][1] + pred[maxIndex][3]) / 2
    xCentScaled, yCentScaled = DepthScale(depthImage, image, xCenter, yCenter)
    depthDistance = depthImage[yCentScaled][xCentScaled]
    return int(xCenter), int(yCenter), depthDistance


def trackHumanClose(oldPos, depthImage, image, trackClass):
    # initialise

    if len(pred) == 0:
        return oldPos[0], oldPos[1], oldPos[2],

    depthDistanceMin = 10000
    minIndex = 0
    depthDistance = 0
    for index, row in enumerate(pred):
        if int(row[5]) == trackClass:
            xCenter = ((row[0] + row[2]) / 2)
            yCenter = ((row[1] + row[3]) / 2)
            xCentScaled, yCentScaled = DepthScale(depthImage, image, xCenter, yCenter)
            depthDistance = depthImage[yCentScaled][xCentScaled]

            if depthDistance < depthDistanceMin:
                depthDistanceMin = depthDistance
                minIndex = index
    xCentMin = (pred[minIndex][0] + pred[minIndex][2]) / 2
    yCentMin = (pred[minIndex][1] + pred[minIndex][3]) / 2

    return int(xCentMin), int(yCentMin), depthDistance


def trackHumanLastPos(prediction, oldPos, depthImage, image, trackClass, bottom, joint2Combo):
    # initial
    predRow = -1

    if len(prediction) == 0:
        return oldPos[0], oldPos[1], oldPos[2], predRow
    if bottom == 1:
        minY = 10000
        minIndex = -1
        foundseg1 = 0
        for index, row in enumerate(prediction):
            if int(row[5]) == 1:
                seg1X = (row[0] + row[2]) / 2
                seg1Y = (row[1] + row[3]) / 2
                foundseg1 = 1
        if not foundseg1:
            return oldPos[0], oldPos[1], oldPos[2], predRow

        for index, row in enumerate(prediction):
            if int(row[5]) == trackClass:
                xCenter = ((row[0] + row[2]) / 2)
                yCenter = ((row[1] + row[3]) / 2)
                difference = abs(xCenter - seg1X) + abs(yCenter - seg1Y)
                if difference < minY:
                    minY = difference
                    minIndex = index
        if minIndex == -1:
            return oldPos[0], oldPos[1], oldPos[2], predRow
        else:
            xCentMin = (prediction[minIndex][0] + prediction[minIndex][2]) / 2
            yCentMin = (prediction[minIndex][1] + prediction[minIndex][3]) / 2
            xCentScaled, yCentScaled = DepthScale(depthImage, image, xCentMin, yCentMin)
            depthDistance = depthImage[yCentScaled][xCentScaled]
            return int(xCentMin), int(yCentMin), depthDistance, minIndex
    elif bottom == 2:
        minY = 0
        minIndex = -1
        foundseg1 = 0
        for index, row in enumerate(prediction):
            if int(row[5]) == 1:
                seg1X = (row[0] + row[2]) / 2
                seg1Y = (row[1] + row[3]) / 2
                foundseg1 = 1
        if not foundseg1:
            seg1X = joint2Combo[0]
            seg1Y = joint2Combo[1]
        for index, row in enumerate(prediction):
            if int(row[5]) == trackClass:
                xCenter = ((row[0] + row[2]) / 2)
                yCenter = ((row[1] + row[3]) / 2)
                difference = abs(xCenter - seg1X) + abs(yCenter - seg1Y)
                if difference > minY:
                    minY = difference
                    minIndex = index
        if minIndex == -1:
            return 0, 0, 0, predRow
        else:
            xCentMin = (prediction[minIndex][0] + prediction[minIndex][2]) / 2
            yCentMin = (prediction[minIndex][1] + prediction[minIndex][3]) / 2
            xCentScaled, yCentScaled = DepthScale(depthImage, image, xCentMin, yCentMin)
            depthDistance = depthImage[yCentScaled][xCentScaled]
            return int(xCentMin), int(yCentMin), depthDistance, minIndex
    else:
        if oldPos[0] == 0 and oldPos[1] == 0 and oldPos[2] == 0:
            depthDistanceMin = 10000
            minIndex = -1
            for index, row in enumerate(prediction):
                if int(row[5]) == trackClass:
                    xCenter = ((row[0] + row[2]) / 2)
                    yCenter = ((row[1] + row[3]) / 2)
                    xCentScaled, yCentScaled = DepthScale(depthImage, image, xCenter, yCenter)
                    depthDistance = depthImage[yCentScaled][xCentScaled]
                    if depthDistance == 0:
                        depthDistance = 100000
                    if depthDistance < depthDistanceMin:
                        depthDistanceMin = depthDistance
                        minIndex = index
            xCentMin = (prediction[minIndex][0] + prediction[minIndex][2]) / 2
            yCentMin = (prediction[minIndex][1] + prediction[minIndex][3]) / 2
            if minIndex == -1:
                return oldPos[0], oldPos[1], oldPos[2], predRow
            return int(xCentMin), int(yCentMin), depthDistanceMin, minIndex
        else:

            # closest to last position (x,y,depth)
            depthDifferenceMin = 10000
            minIndex = -1
            depthDistance = 0
            depthMin = 0
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depthImage, alpha=0.03), cv2.COLORMAP_JET)
            for index, row in enumerate(prediction):
                if int(row[5]) == trackClass:

                    xCenter = ((row[0] + row[2]) / 2)
                    yCenter = ((row[1] + row[3]) / 2)
                    xCentScaled, yCentScaled = DepthScale(depthImage, image, xCenter, yCenter)
                    depth_colormap = cv2.circle(depth_colormap, (xCentScaled, yCentScaled), radius=3, color=(0, 0, 255),
                                                thickness=3)
                    depthDistance = depthImage[yCentScaled][xCentScaled]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    depth_colormap = cv2.putText(depth_colormap, str(depthDistance), (xCentScaled, yCentScaled), font,
                                                 1, (0, 255, 0), 2, cv2.LINE_AA)

                    if depthDistance == 0:
                        depthDistance = 100000
                    depthDifference = abs(int(depthDistance) - oldPos[2]) + abs(xCenter - oldPos[0]) + abs(
                        yCenter - oldPos[1])

                    if depthDifference < depthDifferenceMin:
                        depthDifferenceMin = depthDifference
                        depthMin = depthDistance
                        minIndex = index
            if minIndex == -1:
                return oldPos[0], oldPos[1], oldPos[2], predRow
            xCentMin = (prediction[minIndex][0] + prediction[minIndex][2]) / 2
            yCentMin = (prediction[minIndex][1] + prediction[minIndex][3]) / 2

            return int(xCentMin), int(yCentMin), depthMin, minIndex


def getInputArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./yolov5/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    #parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    #parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
   # parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
   # parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    #parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
   # parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
 
    print(opt)
    return opt


def getFrames(pipeline):
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    # if not depth_frame or not color_frame:
    #     continue

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_image, depthW, depthH = crop_center(depth_image, 540, 300)

    color_image = np.asanyarray(color_frame.get_data())
    return color_image, depth_image, depthW, depthH


def initRealsense():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    return pipeline


def putText(color_image, angle1, angle2, angleRotate, softArmCommands, fps, greenRate, yellowRate, baseRate, command):
    # softarmcommands -> [joint angles, rotation angle, command type ]
    # (0 = neutral, 1 = bear,2 = human, 3=headpat. 4 =deflate, 5= speak, 6=wave)

    if command=="Neutral":
        mode = "neutral"
    elif command=="Neutral":
        mode = "Bear pointing"
    elif command=="Human":
        mode = "Human pointing"
    elif command=="Headpat":
        mode = "Heatpat"
    elif command=="Deflate":
        mode = "Deflate"

    else:
        mode = "unknown" + str(softArmCommands[2])
    text1 = "Angle 1: " + str(angle1)
    text2 = "Angle 2: " + str(angle2)
    text3 = "Angle R: " + str(round(angleRotate, 2))
    text4 = "Target angle: " + str(softArmCommands[0]) + ", " + str(softArmCommands[1])
    text5 = "Mode: " + mode
    text6 = "FPS: " + str(round(fps, 1))
    text7 = "CMD: " + str(round(greenRate, 3)) + " " + str(round(yellowRate, 3)) + " " + str(round(baseRate, 3))
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 1.5
    # Blue color in BGR
    color = (255, 255, 50)
    color1 = (50, 255, 50)
    # Line thickness of 2 px
    thickness = 2
    color_image = cv2.putText(color_image, text7, (1400, 750), font, 1.3, color1, thickness, cv2.LINE_AA)
    color_image = cv2.putText(color_image, text6, (1400, 800), font, 1.3, color1, thickness, cv2.LINE_AA)
    color_image = cv2.putText(color_image, text5, (1400, 850), font, 1.3, color1, thickness, cv2.LINE_AA)
    color_image = cv2.putText(color_image, text4, (1400, 900), font, 1.3, color1, thickness, cv2.LINE_AA)

    color_image = cv2.putText(color_image, text1, (1400, 950), font, fontScale, color, thickness, cv2.LINE_AA)
    color_image = cv2.putText(color_image, text2, (1400, 1000), font, fontScale, color, thickness, cv2.LINE_AA)
    color_image = cv2.putText(color_image, text3, (1400, 1050), font, fontScale, color, thickness, cv2.LINE_AA)

    return color_image


def getNearDepthMin(DepthY, DepthX):
    pointB = int(depth_image[DepthX][DepthY])
    if pointB == 0:
        pointB = 100000
    # remove noise step
    xMax = DepthX
    yMax = DepthY
    expand = 2
    if (DepthX + expand) < xMax and (DepthX - expand) > 0 and (DepthY + expand) < yMax and (DepthY - expand) > 0:
        pointB1 = int(depth_image[DepthX + expand][DepthY + expand])
        if pointB1 == 0:
            pointB1 = 100000
        pointB2 = int(depth_image[DepthX + expand][DepthY - expand])
        if pointB2 == 0:
            pointB2 = 100000
        pointB3 = int(depth_image[DepthX - expand][DepthY + expand])
        if pointB1 == 0:
            pointB3 = 100000
        pointB4 = int(depth_image[DepthX - expand][DepthY - expand])
        if pointB1 == 0:
            pointB4 = 100000

        return min(pointB1, pointB2, pointB3, pointB4, pointB)
    else:
        return pointB


def getNextColor(color_image, width, height, segment3yBottom, segment3xBottom, color_image_box, directionx, directiony,
                 shiftx=0, shifty=0):
    shiftx += (directionx * 4)
    shifty += (directiony * 4)
    A = np.copy(color_image[segment3yBottom - height - shifty, segment3xBottom - width - shiftx])
    B = np.copy(A)
    maxDiff = 60
    imgH, imgW, _ = color_image.shape
    while (np.sum(np.absolute(A - B)) < maxDiff):
        shiftx += directionx * 2
        shifty += directiony * 2
        vertical = segment3yBottom - height - shifty
        horizontal = segment3xBottom - width - shiftx
        if (vertical >= imgH):
            vertical = imgH - 1
        if (horizontal >= imgW):
            horizontal = imgW - 1
        B = np.copy(color_image[vertical, horizontal])
        B = np.float16(B)

        if (abs(horizontal) > imgW - 2):
            return 0, 0
        if (abs(vertical) > imgH - 2):
            return 0, 0

    color_image_box[segment3yBottom - height - shifty:segment3yBottom - height - shifty + 5,
    segment3xBottom - width - shiftx:segment3xBottom - width - shiftx + 5] = [255, 255, 0]
    return shiftx, shifty


def getBear(pred):
    x = -1
    y = -1

    # print("PRED",pred)
    conf = float(0)
    for line in pred:
        if line[5] == 6:
            #  print("BEAR")
            if (line[4] > conf):
                x = (line[0] + line[2]) / 2
                y = (line[1] + line[3]) / 2
                conf = line[4]

    return int(x), int(y)


def getCenterColor(color_image_box, color_image, segment3yBottom, segment3xBottom):
    # A is center
    A = np.copy(color_image[segment3yBottom, segment3xBottom])
    A = np.float16(A)
    A1 = np.copy(color_image[segment3yBottom - 2, segment3xBottom - 2])
    A1 = np.float16(A1)
    A2 = np.copy(color_image[segment3yBottom + 2, segment3xBottom + 2])
    A2 = np.float16(A2)
    A = (A + A1 + A2) / 3
    color_image_box[segment3yBottom:segment3yBottom + 5, segment3xBottom:segment3xBottom + 5] = [255, 255, 255]
    return color_image_box, A


def getStripe(color_image, width, height, segment3yBottom, segment3xBottom, color_image_box, direction):
    shiftx, shifty = getNextColor(color_image, width, height, segment3yBottom, segment3xBottom, color_image_box,
                                  1 * direction, 0, 0, 0)
    shiftx1 = shiftx;
    shifty1 = shifty
    # shift 2
    shiftx, shifty = getNextColor(color_image, width, height, segment3yBottom, segment3xBottom, color_image_box,
                                  1 * direction, 0, shiftx + 3 * direction, shifty)
    shiftx2 = shiftx;
    shifty2 = shifty

    stripe1 = [(shiftx1 + shiftx2) / 2, (shifty1 + shifty2) / 2]
    middle = int(stripe1[0])
    color_image_box[segment3yBottom - height:segment3yBottom - height + 5,
    segment3xBottom - width - middle:segment3xBottom - width - middle + 5] = [255, 125, 255]
    return color_image_box, shiftx1, shiftx2, middle


def getAnglesManual(color_image, width, height, segment3yBottom, segment3xBottom, color_image_box):
    # right
    direction = -1
    # get middle of sides, and check if green or black
    # center point A is top, C is center, E is bottom
    color_image_box, shiftx11R, shiftx12R, middle1R = getStripe(color_image, 0, int(height / 6), segment3yBottom,
                                                                segment3xBottom, color_image_box, direction)
    pointColor = color_image[segment3yBottom - int(height / 6), segment3xBottom - middle1R]
    pointBr = np.sum(np.absolute(pointColor - [126, 180, 152]))
    # print("A", pointColor)
    #  print("-", np.sum(np.absolute(pointColor - [126, 180, 152])))

    color_image_box, shiftx21R, shiftx22R, middle2R = getStripe(color_image, 0, int(height / 2), segment3yBottom,
                                                                segment3xBottom, color_image_box, direction)
    pointColor = color_image[segment3yBottom - int(height / 2), segment3xBottom - middle2R]
    pointAr = np.sum(np.absolute(pointColor - [126, 180, 152]))
    # print("B", pointColor)
    # print("-", np.sum(np.absolute(pointColor - [126, 180, 152])))

    color_image_box, shiftx31R, shiftx32R, middle3R = getStripe(color_image, 0, 0, segment3yBottom, segment3xBottom,
                                                                color_image_box, direction)
    pointColor = color_image[segment3yBottom, segment3xBottom - middle3R]
    pointCr = np.sum(np.absolute(pointColor - [126, 180, 152]))
    # print("C", pointColor)
    # print("-", np.sum(np.absolute(pointColor - [126, 180, 152])))

    color_image_box, shiftx41R, shiftx42R, middle4R = getStripe(color_image, 0, -int(height / 6), segment3yBottom,
                                                                segment3xBottom, color_image_box, direction)
    pointColor = color_image[segment3yBottom + int(height / 6), segment3xBottom - middle4R]
    pointEr = np.sum(np.absolute(pointColor - [44, 44, 44]))
    # print("E", pointColor)
    # print("-", np.sum(np.absolute(pointColor - [44, 44, 44])))

    color_image_box, shiftx51R, shiftx52R, middle5R = getStripe(color_image, 0, -int(height / 2), segment3yBottom,
                                                                segment3xBottom, color_image_box, direction)
    pointColor = color_image[segment3yBottom + int(height / 2), segment3xBottom - middle5R]
    pointDr = np.sum(np.absolute(pointColor - [44, 44, 44]))
    # print("E", pointColor)
    # print("-", np.sum(np.absolute(pointColor - [44, 44, 44])))

    # get difference average to remove outliers for middle section width
    diff1R = abs(shiftx11R - shiftx21R) + abs(shiftx11R - shiftx31R) + abs(shiftx11R - shiftx41R) + abs(
        shiftx11R - shiftx51R)
    diff2R = abs(shiftx21R - shiftx11R) + abs(shiftx21R - shiftx31R) + abs(shiftx21R - shiftx41R) + abs(
        shiftx21R - shiftx51R)
    diff3R = abs(shiftx31R - shiftx21R) + abs(shiftx31R - shiftx11R) + abs(shiftx31R - shiftx41R) + abs(
        shiftx31R - shiftx51R)
    diff4R = abs(shiftx41R - shiftx21R) + abs(shiftx41R - shiftx31R) + abs(shiftx41R - shiftx11R) + abs(
        shiftx41R - shiftx51R)
    diff5R = abs(shiftx51R - shiftx21R) + abs(shiftx51R - shiftx31R) + abs(shiftx51R - shiftx41R) + abs(
        shiftx51R - shiftx11R)
    diff1R = 0 if diff1R > (width * 2) else 1
    diff2R = 0 if diff2R > (width * 2) else 1
    diff3R = 0 if diff3R > (width * 2) else 1
    diff4R = 0 if diff4R > (width * 2) else 1
    diff5R = 0 if diff5R > (width * 2) else 1
    numValid = 0
    if (diff1R == 1): numValid += 1
    if (diff2R == 1): numValid += 1
    if (diff3R == 1): numValid += 1
    if (diff4R == 1): numValid += 1
    if (diff5R == 1): numValid += 1
    if (numValid > 0):
        rightAverage = ((shiftx11R * diff1R) + (shiftx21R * diff2R) + (shiftx31R * diff3R) + (shiftx41R * diff4R) + (
                    shiftx51R * diff5R)) / numValid
    else:
        rightAverage = -1000
    # left===========================================================================================================
    direction = 1
    color_image_box, shiftx11L, shiftx12L, middle1L = getStripe(color_image, 0, int(height / 6), segment3yBottom,
                                                                segment3xBottom, color_image_box, direction)
    pointColor = color_image[segment3yBottom - int(height / 6), segment3xBottom - middle1L]
    pointBl = np.sum(np.absolute(pointColor - [126, 180, 152]))
    # print("A", pointColor)
    # print("-", np.sum(np.absolute(pointColor - [126, 180, 152])))

    color_image_box, shiftx21L, shiftx22L, middle2L = getStripe(color_image, 0, int(height / 2), segment3yBottom,
                                                                segment3xBottom, color_image_box, direction)
    pointColor = color_image[segment3yBottom - int(height / 2), segment3xBottom - middle2L]
    pointAl = np.sum(np.absolute(pointColor - [126, 180, 152]))
    # print("B", pointColor)
    # print("-", np.sum(np.absolute(pointColor - [126, 180, 152])))

    color_image_box, shiftx31L, shiftx32L, middle3L = getStripe(color_image, 0, 0, segment3yBottom, segment3xBottom,
                                                                color_image_box, direction)
    pointColor = color_image[segment3yBottom, segment3xBottom - middle3L]
    pointCl = np.sum(np.absolute(pointColor - [126, 180, 152]))
    # print("C", pointColor)
    # print("-", np.sum(np.absolute(pointColor - [126, 180, 152])))

    color_image_box, shiftx41L, shiftx42L, middle4L = getStripe(color_image, 0, -int(height / 6), segment3yBottom,
                                                                segment3xBottom, color_image_box, direction)
    pointColor = color_image[segment3yBottom + int(height / 6), segment3xBottom - middle4L]
    pointEl = np.sum(np.absolute(pointColor - [44, 44, 44]))
    #  print("E", pointColor)
    #  print("-", np.sum(np.absolute(pointColor - [44, 44, 44])))

    color_image_box, shiftx51L, shiftx52L, middle5L = getStripe(color_image, 0, -int(height / 2), segment3yBottom,
                                                                segment3xBottom, color_image_box, direction)
    pointColor = color_image[segment3yBottom + int(height / 2), segment3xBottom - middle5L]
    pointDl = np.sum(np.absolute(pointColor - [44, 44, 44]))
    #   print("E", pointColor)
    #  print("-", np.sum(np.absolute(pointColor - [44, 44, 44])))

    # get difference average to remove outliers for middle section width
    diff1L = abs(shiftx11L - shiftx21L) + abs(shiftx11L - shiftx31L) + abs(shiftx11L - shiftx41L) + abs(
        shiftx11L - shiftx51L)
    diff2L = abs(shiftx21L - shiftx11L) + abs(shiftx21L - shiftx31L) + abs(shiftx21L - shiftx41L) + abs(
        shiftx21L - shiftx51L)
    diff3L = abs(shiftx31L - shiftx21L) + abs(shiftx31L - shiftx11L) + abs(shiftx31L - shiftx41L) + abs(
        shiftx31L - shiftx51L)
    diff4L = abs(shiftx41L - shiftx21L) + abs(shiftx41L - shiftx31L) + abs(shiftx41L - shiftx11L) + abs(
        shiftx41L - shiftx51L)
    diff5L = abs(shiftx51L - shiftx21L) + abs(shiftx51L - shiftx31L) + abs(shiftx51L - shiftx41L) + abs(
        shiftx51L - shiftx11L)
    diff1L = 0 if diff1L > (width * 2) else 1
    diff2L = 0 if diff2L > (width * 2) else 1
    diff3L = 0 if diff3L > (width * 2) else 1
    diff4L = 0 if diff4L > (width * 2) else 1
    diff5L = 0 if diff5L > (width * 2) else 1

    numValid = 0
    if (diff1L == 1): numValid += 1
    if (diff2L == 1): numValid += 1
    if (diff3L == 1): numValid += 1
    if (diff4L == 1): numValid += 1
    if (diff5L == 1): numValid += 1
    if (numValid > 0):
        leftAverage = ((shiftx11L * diff1L) + (shiftx21L * diff2L) + (shiftx31L * diff3L) + (shiftx41L * diff4L) + (
                    shiftx51L * diff5L)) / numValid
    else:
        leftAverage = -1000

    if (rightAverage != -1000 and leftAverage != -1000):
        centerWidth = abs(leftAverage) + abs(rightAverage)
    elif (rightAverage != -1000):
        centerWidth = abs(rightAverage * 2)
    elif (leftAverage != -1000):
        centerWidth = abs(leftAverage * 2)
    else:
        centerWidth = -1

    # predict angles using valid side widths and center width
    if (centerWidth != -1):
        colorDevThreshold = 140
        # top point
        if (pointAr < colorDevThreshold):
            angleAr = 45 * abs(shiftx21R - shiftx22R) / centerWidth
        else:
            angleAr = 0
        if (pointAl < colorDevThreshold):
            angleAl = 45 * abs(shiftx21L - shiftx22L) / centerWidth
        else:
            angleAl = 0
        # top middle
        if (pointBr < colorDevThreshold):
            angleBr = 45 * abs(shiftx11R - shiftx12R) / centerWidth
        else:
            angleBr = 0
        if (pointBl < colorDevThreshold):
            angleBl = 45 * abs(shiftx11L - shiftx12L) / centerWidth
        else:
            angleBl = 0
            # middle
        if (pointCr < colorDevThreshold):
            angleCr = 45 * abs(shiftx31R - shiftx32R) / centerWidth
        else:
            angleCr = 0
        if (pointCl < colorDevThreshold):
            angleCl = 45 * abs(shiftx31L - shiftx32L) / centerWidth
        else:
            angleCl = 0
            # bottom middle
        if (pointDr < colorDevThreshold):
            angleDr = 45 * abs(shiftx41R - shiftx42R) / centerWidth
        else:
            angleDr = 0
        if (pointDl < colorDevThreshold):
            angleDl = 45 * abs(shiftx41L - shiftx42L) / centerWidth
        else:
            angleDl = 0
            # bottom point
        if (pointEr < colorDevThreshold):
            angleEr = 45 * abs(shiftx51R - shiftx52R) / centerWidth
        else:
            angleEr = 0
        if (pointEl < colorDevThreshold):
            angleEl = 45 * abs(shiftx51L - shiftx52L) / centerWidth
        else:
            angleEl = 0
    else:
        angleAl = -1
        angleBl = -1
        angleCl = -1
        angleDl = -1
        angleEl = -1
        angleAr = -1
        angleBr = -1
        angleCr = -1
        angleDr = -1
        angleEr = -1

    predAnglesRight = [angleAr, angleBr, angleCr, angleDr, angleEr]
    predAnglesLeft = [angleAl, angleBl, angleCl, angleDl, angleEl]
    print("RIGHT", predAnglesRight)
    print("LEFT", predAnglesLeft)
    return color_image_box, predAnglesRight, predAnglesLeft


def getBestAngle(angle, predAnglesRight, predAnglesLeft):
    CNNDIFF = 8
    if (angle < 0):
        mean = 0
        length = 0
        for i in predAnglesRight:
            if (abs(-i - angle) < 10 or abs(i - angle) < CNNDIFF):
                mean -= i
                length += 1
        if (length == 0):
            return angle
        mean = mean / length
        if (angle > 0):
            if (mean > 0):
                return mean
            else:
                return -mean
        else:
            if (mean < 0):
                return mean
            else:
                return -mean
    elif (angle > 0):
        mean = 0
        length = 0
        for i in predAnglesLeft:
            if (abs(-i - angle) < 10 or abs(i - angle) < CNNDIFF):
                mean -= i
                length += 1
        if (length == 0):
            return angle
        mean = mean / length
        if (angle > 0):
            if (mean > 0):
                return mean
            else:
                return -mean
        else:
            if (mean < 0):
                return mean
            else:
                return -mean

    else:
        return 0
    return bestAngle


def getRotateAngle(pred, predRowsegment3Bottom, color_image, color_image_box, segment3yBottom, segment3xBottom,
                   color_image_old, color_image_old1, rotCNN, oldAngle):
    # remove noise by average last 2 and current images
    color_image = cv2.addWeighted(color_image, 0.5, color_image_old, 0.5, 0)
    color_image = cv2.addWeighted(color_image, 0.5, color_image_old1, 0.5, 0)
    if (predRowsegment3Bottom != -1):
        row = pred[predRowsegment3Bottom]
        height = abs(row[1] - row[3]) / 2
        width = abs(row[0] - row[2]) / 2

        color_image_box, A = getCenterColor(color_image_box, color_image, segment3yBottom, segment3xBottom)
        # manual angle estimation
        color_image_box, predAnglesRight, predAnglesLeft = getAnglesManual(color_image, width, height, segment3yBottom,
                                                                           segment3xBottom, color_image_box)
        # CNN based angle estimationcrop
        crop = np.copy(color_image[segment3yBottom - int(height * 3):segment3yBottom + int(height * 3),
                       segment3xBottom - int(height * 3):segment3xBottom + int(height * 3)])
        w, h, c = crop.shape
        # print("SAPE", crop.shape)
        if (w > 0 and h > 0):
            angle = rotCNN.classify(crop)
        else:
            angle = oldAngle
        print("CNN ANGLE", angle)
        # choose best angle based on both CNN and manual estimation
        angle = getBestAngle(angle, predAnglesRight, predAnglesLeft)
        print("best angel", angle)
        print("------------")
        return angle, color_image
    else:
        return oldAngle, color_image
        # classNum 0= seg1(green red), 1= seg2(yellow orange), 2 = seg3(red blue), 4=whole arm, 5= human, 6= bear


def checkIfPresent(pred, classNum):
    for i, line in enumerate(pred):
        if int(line[5]) == classNum:
            return 1
    return 0
