import cv2
import pcbnew
import math
import numpy as np
from scipy import ndimage
import pandas as pd
import re
from shapely.geometry import Polygon
from centerline.geometry import Centerline
import networkx as nx
import operator
from itertools import tee
from PIL import Image


def pcbpoint(p):
    return pcbnew.wxPointMM(float(p[0]), float(p[1]))
def vecpoint(p):
    return pcbnew.VECTOR2I_MM(float(p[0]), float(p[1]))

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return list(zip(a, b))

def getAbsoluteAngle(edge1, edge2):
    v1 = tuple(map(operator.sub, edge1[0], edge1[1]))
    v2 = tuple(map(operator.sub, edge2[0], edge2[1]))
    return abs(np.degrees(math.atan2(np.linalg.det([v1, v2]), np.dot(v1, v2))))

def angle_to(p1, p2, rotation=0, clockwise=False):
    angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) - rotation
    if not clockwise:
        angle = -angle
    return angle % 360

def anti_clockwise(p1,p2):
    alpha = math.degrees(math.atan2(p2[1] - p1[1],p2[0]-p1[0]))
    return (alpha + 360) % 360


def getInputs():
    fileName = input("What do you want the file name to be (without extension)? ")
    boardX = input("What is the x-direction size of the board (in mm)? ")
    boardY = input("What is the y-direction size of the board (in mm)? ")
    return [fileName, boardX, boardY]

def pointsMinMax(pointsList):
    maxX = 0
    maxY = 0
    minX = 150000
    minY = 150000
    for i in pointsList:
        if i[0] > maxX:
            maxX
                
        if i[0] < minX:
            minX = i[0]
        if i[1] > maxY:
            maxY = i[1]
        if i[1] < minY:
            minY = i[1]
    return minX, maxX, minY, maxY

def getPointX(pointList):
    xMax = 0
    xMin = 150000
    for i in pointList:
        if i[0] > xMax:
            xMax = i[0]
        if i[0] < xMin:
            xMin = i[0]
    return [xMin, xMax]

def viaDetect(img, pointList, xMin, xMax):
    pointsList = list()
    diameterList = list()
    viaPointList = list()
    for i in [pointList]:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            pointsList.append((cx, cy))
            cv2.circle(img, (cx, cy), 7, (0, 0, 255), -1)
    diameterList.append(xMax - xMin)
    viaPointList.append(i)
    return [pointsList, diameterList, viaPointList]


def drawVias(img, pointList):
    cv2.drawContours(img, [pointList], 0, (0, 0, 255), 2)

def get_rotation_matrix(angle_deg: float):
    theta = np.radians(angle_deg)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation = np.array(((cos_theta, -sin_theta), (sin_theta, cos_theta)))
    return rotation

def rotate_coordinate(coordinate: np.array, angle_deg: float) -> np.array:
    rotation = get_rotation_matrix(angle_deg)
    rotated = rotation.dot(coordinate)
    return rotated

def get_new_coordinate(original_coordinate: np.array, center_of_rotation: np.array, angle_deg: float):
    delta = original_coordinate - center_of_rotation
    delta_rotated = rotate_coordinate(delta, angle_deg)
    new_coordinate = center_of_rotation + delta_rotated
    return new_coordinate

def drawTraces(img, pointList):
    cv2.drawContours(img, [pointList], 0, (0, 255, 0), 2)


def drawPads(img, pointList):
    cv2.drawContours(img, [pointList], 0, (0, 165, 255), 2)


def distanceBetween(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def create_board(fileName, x, y):
    board = pcbnew.NewBoard('output/' + fileName + '.kicad_pcb')
    rectangle = pcbnew.PCB_SHAPE(board)
    rectangle.SetShape(pcbnew.SHAPE_T_RECT)
    rectangle.SetFilled(False)
    rectangle.SetStart(vecpoint((12, 12)))
    rectangle.SetEnd(vecpoint((12+x, 12+y)))
    rectangle.SetLayer(pcbnew.Edge_Cuts)
    rectangle.SetWidth(int(0.1) * IU_PER_MM)
    board.Add(rectangle)
    return board

def create_via(board, point, drill, width, type, layer1, layer2):
    newvia = pcbnew.PCB_VIA(board)
    board.Add(newvia)
    if type == "through":
        newvia.SetViaType(pcbnew.VIATYPE_THROUGH)
    elif type == "buried":
        newvia.SetViaType(pcbnew.VIATYPE_BLIND_BURIED)
    elif type == "micro":
        newvia.SetViaType(pcbnew.VIATYPE_MICROVIA)
    else:
        print(type)
    newvia.SetLayerPair(board.GetLayerID(layer1), board.GetLayerID(layer2))
    newvia.SetPosition(pcbnew.VECTOR2I(point))
    newvia.SetDrill(int(drill))
    newvia.SetWidth(int(width))

def createVias(board, img, pointsList, diameterList, MMpixelX, MMpixelY, MMpixel, x0, y0, median_angle):
    viaCenterList = list()
    for i, j in zip(pointsList, diameterList):
        vias = get_new_coordinate(np.array((float(i[0]), float(i[1]))),
                                  np.array((float(img.shape[0] / 2), float(img.shape[1] / 2))), -1 * median_angle)
        xVia = ((vias[0] - x0) * MMpixelX) + 12
        yVia = ((vias[1] - y0) * MMpixelY) + 12
        diameter = j * MMpixel
        drill = (float(diameter)) - 0.1
        viaCenterList.append((xVia, yVia))
        create_via(board, pcbpoint((xVia, yVia)), float(drill) * IU_PER_MM, float(diameter) * IU_PER_MM, 'through',
                   'F.Cu', 'B.Cu')
    return viaCenterList

def lineDetection(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray.jpg', img_gray)
    img_edges = cv2.Canny(img_gray, 650, 650, apertureSize=3)
    cv2.imwrite('canny.jpg', img_edges)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 60, minLineLength=30, maxLineGap=10)
    return lines

def createPads(board, padList, MMpixelX, MMpixelY):
    padPosList = list()
    for i in padList:
        fp = pcbnew.FOOTPRINT(board)
        pad = pcbnew.PAD(fp)
        pad.SetSize(vecpoint((1, 1)))
        pad.SetLayer(pcbnew.F_Cu)
        shape = pcbnew.SHAPE_LINE_CHAIN()
        sumX = 0
        sumY = 0
        count = 0
        for j in i:
            sumX += j[0]
            sumY += j[1]
            count += 1
        avgPoint = (sumX / count, sumY / count)
        i = i - avgPoint
        for j in i:
            shape.Append(vecpoint(((j[0] * MMpixelX), (j[1] * MMpixelY))))
        padPosList.append((avgPoint[0] * MMpixelX + 12, avgPoint[1] * MMpixelY + 12))
        fp.SetPosition(vecpoint((avgPoint[0] * MMpixelX + 12, avgPoint[1] * MMpixelY + 12)))
        poly = pcbnew.SHAPE_POLY_SET(shape)
        pad.SetShape(pcbnew.PAD_SHAPE_CUSTOM)
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.AddPrimitivePoly(poly, 10, True)
        fp.Add(pad)
        board.Add(fp)
        return padPosList

def rotateImage(img, lines):
    anglesVertPlus = []
    anglesVertMinus = []
    anglesHoriz = []
    try:
        for [[x1, y1, x2, y2]] in lines:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if (angle < 95 and angle > 85):
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                anglesVertPlus.append(angle)
            elif (angle < -85 and angle > -95):
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                anglesVertMinus.append(angle)
            elif (angle < 10 and angle > -10):
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                anglesHoriz.append(angle)
        if len(anglesVertPlus) > len(anglesHoriz) and len(anglesVertPlus) > len(anglesVertMinus):
            median_angle = np.median(anglesVertPlus) - 90
            img_rotated = ndimage.rotate(img, median_angle, reshape=False, cval=210)
        elif len(anglesVertMinus) > len(anglesHoriz) and len(anglesVertMinus) > len(anglesVertPlus):
            median_angle = np.median(anglesVertMinus) + 90
            img_rotated = ndimage.rotate(img, median_angle, reshape=False, cval=210)
        else:
            median_angle = float(np.median(anglesHoriz))
            img_rotated = ndimage.rotate(img, median_angle, reshape=False, cval=210)
    except:
        median_angle = 0
        img_rotated = img
    return [median_angle, img_rotated]


def getMinMax(points):
    minX = 150000
    maxX = 0
    minY = 150000
    maxY = 0
    for i in points.get('Vertices'):
        coordList = [eval(l) for l in list(re.findall(r'\d+', i))]
        for j in coordList[0::2]:
            if j > maxX:
                maxX = j
            if j < minX:
                minX = j
        for k in coordList[1::2]:
            if k > maxY:
                maxY = k
            if k < minY:
                minY = k
    return [minX, maxX, minY, maxY]

def create_track(board, start, end, width, layer):
    newtrack = pcbnew.PCB_TRACK(board)
    board.Add(newtrack)
    newtrack.SetStart(pcbnew.VECTOR2I(start))
    newtrack.SetEnd(pcbnew.VECTOR2I(end))
    newtrack.SetWidth(int(width))
    newtrack.SetLayer(board.GetLayerID(layer))
    newtrack.SetNetCode(0)

def createTracks(board, linesList, img, median_angle, MMpixelX, MMpixelY, MMpixel, x0, y0, padList, padPosList, viaPointList, viaCenterList):
    maxvals = list()
    for i in linesList:
        maxval = 0
        for j in i:
            if j[1] > maxval:
                maxval = j[1]
        maxvals.append(maxval)
    counter = -1
    for i in linesList:
        counter += 1
        for j in i:
            p1 = get_new_coordinate(np.array((float(j[0][0][0]), float(j[0][0][1]))),
                                    np.array((float(img.shape[0] / 2), float(img.shape[1] / 2))), -1 * median_angle)
            p2 = get_new_coordinate(np.array((float(j[0][1][0]), float(j[0][1][1]))),
                                    np.array((float(img.shape[0] / 2), float(img.shape[1] / 2))), -1 * median_angle)
            px0 = ((p1[0] - x0) * MMpixelX) + 12
            py0 = ((p1[1] - y0) * MMpixelY) + 12
            px1 = ((p2[0] - x0) * MMpixelX) + 12
            py1 = ((p2[1] - y0) * MMpixelY) + 12
            for k, m, in zip(padList, padPosList):
                for l in k:
                    if abs((l[0] * MMpixelX + 12) - px0) < 2.5 and abs((l[1] * MMpixelY + 12) - py0) < 2.5:
                        px0 = m[0]
                        py0 = m[1]
                    elif abs((l[0] * MMpixelX + 12) - px1) < 2.5 and abs((l[1] * MMpixelY + 12) - py1) < 2.5:
                        px1 = m[0]
                        py1 = m[1]
            for n, o in zip(viaPointList, viaCenterList):
                for p in n:
                    if abs((p[0] * MMpixelX + 12) - px0) < 1.5 and abs((p[1] * MMpixelY + 12) - py0) < 1.5:
                        px0 = o[0]
                        py0 = o[1]
                    if abs((p[0] * MMpixelX + 12) - px1) < 1.5 and abs((p[1] * MMpixelY + 12) - py1) < 1.5:
                        px1 = o[0]
                        py1 = o[1]
            width = maxvals[counter] * MMpixel
            create_track(board, pcbpoint((px0, py0)), pcbpoint((px1, py1)), float(width) * IU_PER_MM, 'F.Cu')

def main():
    global IU_PER_MM
    IU_PER_MM = 1000000

    [fileName, boardX, boardY] = getInputs()

    path = 'rascLayer5.png'
    img = cv2.imread(path)
    # Perform automatic rotation and alignment
    lines = lineDetection(img)
    [median_angle, img_rotated] = rotateImage(img, lines)
    cv2.imwrite('rotated.jpg', img_rotated)
    image1 = cv2.imread('rotated.jpg')
    image_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image_edges = cv2.Canny(image_gray, 500, 500, apertureSize=3)
    cv2.imwrite('rotatedcanny.jpg', image_edges)

    # Start drawing components
    points = pd.read_csv('rascLayer5_traces_w_holes.csv')

    padList = list()
    pointsList = list()
    diameterList = list()
    linesList = list()
    viaPointList = list()

    # Create/Interpret Features
    for index, (i, type) in enumerate(zip(points.get('Vertices'), points.get('Type'))):
        # Get Basic Info
        coordList = [eval(l) for l in list(re.findall(r'\d+', i))]
        pointList = np.asarray(list(zip((int(i) for i in coordList[0::2]), ((int(i)) for i in coordList[1::2]))))
        drawVias(img, pointList)
        drawTraces(img, pointList)
        drawPads(img, pointList)
        [xMin, xMax] = getPointX(pointList)

        # Specific Components
        if type == 'pad':
            padList.append(pointList)
        elif type == 'via':
            [pointsListT, diameterListT, viaPointListT] = viaDetect(img, pointList, xMin, xMax)
            pointsList += pointsListT
            diameterList += diameterListT
            viaPointList += viaPointListT
        elif type == 'trace':
            # Inputs = img, pointList
            # Outputs = linesList
            poly = Polygon(pointList)
            centerline = Centerline(poly, interpolation_distance=8)
            xList = list()
            points = list()
            edges = list()
            G = nx.Graph()
            coordList = centerline.geometry.geoms

            for i in coordList:
                x, y = i.coords.xy
                x = list(x)
                y = list(y)
                xList.append(tuple([tuple([x[0], y[0]]), tuple([x[1], y[1]])]))
                # Only add to list of points if not already present in list
                if not tuple([x[0], y[0]]) in points:
                    points.append(tuple([x[0], y[0]]))
                if not tuple([x[1], y[1]]) in points:
                    points.append(tuple([x[1], y[1]]))

            for i in xList:
                edges.append([points.index(i[0]), points.index(i[1]), distanceBetween(i[0], i[1])])

            for i in range(len(edges)):
                G.add_edge(points[edges[i][0]], points[edges[i][1]], length=edges[i][2])

            maxDist = 0
            pathList = list()
            distList = list()
            lengths = nx.all_pairs_dijkstra_path_length(G, weight='length')
            for i in lengths:
                node = max(i[1], key=i[1].get)
                if nx.shortest_path_length(G, i[0], node, weight='length') > maxDist:
                    maxDist = nx.shortest_path_length(G, i[0], node, weight='length')
            lengths2 = nx.all_pairs_dijkstra_path_length(G, weight='length')
            for i in lengths2:
                node = max(i[1], key=i[1].get)
                localMax = i[1][node]
                for key in i[1]:
                    if localMax - i[1][key] < 0.15 * localMax:
                        temp = maxDist - nx.shortest_path_length(G, i[0], key, weight='length')
                        if temp < 0.09 * maxDist:
                            pathList.append(nx.shortest_path(G, i[0], key, weight='length'))
                            distList.append(nx.shortest_path_length(G, i[0], key, weight='length'))
            minAngle = 999999
            minPath = list()
            for i in pathList:
                angle = 0
                for j in pairwise(pairwise(i)):
                    angle += getAbsoluteAngle(j[0], j[1])
                if angle < minAngle:
                    minAngle = angle
                    minPath = i

            pathNew = list()
            for i in pairwise(minPath[::3]):
                pathNew.append(i)

            sharpTurns = list()
            for i in pairwise(pathNew):
                angle = getAbsoluteAngle(i[0], i[1])
                if angle > 20:
                    sharpTurns.append(i)

            newLine = list()
            if len(sharpTurns) != 0:
                newLine.append((minPath[0], sharpTurns[0][0][1]))
                for i in pairwise(sharpTurns):
                    newLine.append((i[0][1][0], i[1][1][0]))
                newLine.append((sharpTurns[-1][1][0], minPath[-1]))
            else:
                newLine.append((minPath[0], minPath[-1]))
            for i in newLine:
                j = tuple(tuple(map(int, tup)) for tup in i)
                cv2.line(img, j[0], j[1], (255, 0, 0), 2)

            newLineWidth = list()
            for i in newLine:
                minDist = 999999
                for j in list(pointList):
                    if distanceBetween(i[0], j) < minDist:
                        minDist = distanceBetween(i[0], j)
                    if distanceBetween(i[1], j) < minDist:
                        minDist = distanceBetween(i[1], j)
                newLineWidth.append((i, minDist))
            linesList.append(newLineWidth)

        updatedLinesList = list()
        for i in linesList:
            latestEnd = i[0][0][0]
            line = list()
            for j in i:
                angle = anti_clockwise(latestEnd, j[0][1])
                # Modulus with 45, if >= 22.5, round up, else round down
                mod = angle % 45
                if mod >= 22.5:
                    rounded_angle = (angle//45+1)*45
                else:
                    rounded_angle = (angle//45)*45
                length = distanceBetween(latestEnd, j[0][1])
                final = ((latestEnd[0] + (length*math.cos(math.radians(rounded_angle)))), (latestEnd[1] + (length*math.sin(math.radians(rounded_angle)))))
                line.append(((latestEnd, final), j[1]))
                latestEnd = final
            updatedLinesList.append(line)



    # Create Board
    [minX, maxX, minY, maxY] = pointsMinMax(pointsList)

    rotatedCanny = Image.open('rotatedcanny.jpg')
    data = np.asarray(rotatedCanny)

    y0 = 0
    x0 = 0
    for i, row in enumerate(data):
        if (cv2.countNonZero(row) > data.shape[1] * 0.12) and (abs(i - minY) < (data.shape[1] * 0.05)):
            y0 = i
            break
    for i, column in enumerate(data.T):
        if (cv2.countNonZero(column) > data.shape[0] * 0.12) and (abs(i - minX) < (data.shape[0] * 0.05)):
            x0 = i
            break
    for i, row in reversed(list(enumerate(data))):
        if (cv2.countNonZero(row) > data.shape[1] * 0.12) and (abs(i - maxY) < (data.shape[1] * 0.05)):
            y1 = i
            break
    for i, column in reversed(list(enumerate(data.T))):
        if (cv2.countNonZero(column) > data.shape[0] * 0.12) and (abs(i - maxX) < (data.shape[0] * 0.05)):
            x1 = i
            break

    origImage = Image.open('rotated.jpg')
    origData = np.asarray(origImage)

    try:
        x1, y1
    except:
        x1 = origData.shape[0]
        y1 = origData.shape[1]
        cropped = origData[y0:y1, x0:x1]
    image2 = Image.fromarray(cropped)
    image2.save('rotatedcropped.png')

    board = create_board(fileName, float(boardX), float(boardY))

    MMpixelX = float(boardX) / (x1 - x0)
    MMpixelY = float(boardY) / (y1 - y0)
    MMpixel = (MMpixelX + MMpixelY) / 2

    viaCenterList = createVias(board, img, pointsList, diameterList, MMpixelX, MMpixelY, MMpixel, x0, y0, median_angle)
    padPosList = createPads(board, padList, MMpixelX, MMpixelY)
    createTracks(board, updatedLinesList, img, median_angle, MMpixelX, MMpixelY, MMpixel, x0, y0, padList, padPosList, viaPointList, viaCenterList)
    pcbnew.SaveBoard('output/' + fileName + '.kicad_pcb', board)

if __name__ == '__main__':
    main()
