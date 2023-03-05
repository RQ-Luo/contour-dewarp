from cmath import inf
from operator import itemgetter
import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import fitz
import matplotlib.pyplot as plt


def isoPic(img):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.dilate(img, element, iterations = 3)
    img = cv2.erode(img, element, iterations = 5)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    return img

def cntOfText(img):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.erode(img, element, iterations = 3)
    img = cv2.dilate(img, element, iterations = 4)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    return img

def highContr(img, value)
    clahe = cv2.createCLAHE(value,(8,8)) 
    enhan = clahe.apply(img)
    return enhan


def enhanScan(gray, contours):
    Page = curveToFlat(maxCnt(contours), gray)
    dilationL = cntOfText(Page)
    #pix = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    #ero = cv2.erode(gray, pix, iterations = 1)
    enhan = highContr(gray, 1.2)
    ret, binary = cv2.threshold(enhan, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    return enhan, binary

def curveToFlat(approx, gray):
    points = toNpArray(approx)
    begin = 2500
    end = 4500
    up, down = cutPoints(points, 2500, [begin, end])
    upMid = min(up[1:-1], key=itemgetter(1))[0]
    downMid = max(down[1:-1], key=itemgetter(1))[0]
    upP, uLpopt, uRpopt = getSmothApprox(up[1:-1], upMid)
    downP, dLpopt, dRpopt = getSmothApprox(down[1:-1], downMid)
    up = [up[0].tolist()] + upP + [up[-1].tolist()]
    down = [down[0].tolist()] + downP + [down[-1].tolist()]
    PointsL = plot(begin, upMid, dLpopt, uLpopt)
    PointsR = plot(upMid, end, dRpopt, uRpopt)
    PointsL = np.array(PointsL).astype(int)
    blocksL = putPointToBlock(PointsL)
    blocks = getBlock(up, down)
    return OldmergeBlocks(blocks, gray)

def putPointToBlock(points):
    blocks = []
    for point in range(len(points)):
        if point%18 != 0 and point < 949:
            pointR, otherSide, otherSideR = point + 1, point + 19, point + 20
            blocks.append([points[point], points[pointR], points[otherSide], points[otherSideR]])
    return np.array(blocks)

def maxCnt(contours):
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)
    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)
    return cnt

def findDoc(enhan):
    dilation = isoPic(enhan)
    contours, hierarchy = cv2.findContours(dilation, 3, 2)
    cnt = maxCnt(contours)
    x,y,w,h = cv2.boundingRect(cnt)
    return enhan[y-50:y+h+50, x-10:x+w+10]

def findDocOld(enhan):
    dilation = isoPic(enhan)
    contours, hierarchy = cv2.findContours(dilation, 3, 2)
    region = []
    pages = []
    for cnt in contours:
      area = cv2.contourArea(cnt)
      page_area = enhan.shape[0] * enhan.shape[1]
      if area < page_area * 0.3 or area > page_area * 0.95:
            continue
      x,y,w,h = cv2.boundingRect(cnt)
      region.append([x,y,w,h])
    for box in region:
        x, y, w, h = box[0], box[1], box[2], box[3]
        pages.append(enhan[y:y+h, x:x+w])
    if len(pages) == 0:
        pages.append(enhan)
    return pages

def replacePic(im):
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    shapenGray = shapen(gray)
    doc = findDoc(shapenGray)
    img = cv2.resize(doc, (7016, 4960), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("1.png", img)
    dilation = isoPic(img)
    contours, hierarchy = cv2.findContours(dilation, 3, 2)
    enhan, binary = enhanScan(img, contours)
    region = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        page_area = img.shape[0] * img.shape[1]
        if area < page_area * 0.0002 or area > page_area * 0.8:
                continue
        x,y,w,h = cv2.boundingRect(cnt)
        region.append([x,y,w,h])
    for box in region:
        x, y, w, h = box[0], box[1], box[2], box[3]
        tag = enhan[y:y+h, x:x+w]
        binary[y:y+h, x:x+w] = highContr(tag, 2)
    return binary

def shapen(img):
    blur_img = cv2.GaussianBlur(img, (0, 0), 7)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    return usm

def enhanPdf(doc, filename):
    pdf = fitz.open(filename.replace(" ", ""))
    with tqdm(total=len(pdf)) as pbar:
        pbar.set_description('Processing:')
        for page in pdf:
            pix = page.get_pixmap(matrix=fitz.Matrix(7, 7))
            im = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pageIm = replacePic(np.array(im))
            pageName = "page-%i.png" % page.number
            Image.fromarray(pageIm).save(pageName)
            img = fitz.open(pageName)
            rect = img[0].rect
            pdfbytes = img.convert_to_pdf()
            img.close()
            os.remove(pageName)
            imgPDF = fitz.open("pdf", pdfbytes)
            page = doc.new_page(width = rect.width, height = rect.height)
            page.show_pdf_page(rect, imgPDF, 0)
            pbar.update(1)

def toNpArray(cvApprox):
    return np.array([[point[0][0], point[0][1]] for point in cvApprox])

def getXY(points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    return x, y

def xyToPoints(x, y):
    points = []
    for i in range(len(x)):
        points.append([int(x[i]), int(y[i])])
    return points

def getPolyApprox(points):
    x, y = getXY(points)
    popt = np.polyfit(x, y, 7)
    x_new, y_new = getPointsFromFit(x[0], x[-1], popt)
    return x_new, y_new, popt

def getPointsFromFit(begin, end, popt):
    x_new = np.linspace(begin, end, 20)
    p = np.poly1d(popt)
    y_new = p(x_new)
    return x_new, y_new

def getCenPoints(xs, poptFamily):
    ponints = []
    for popt in poptFamily:
        p = np.poly1d(popt)
        for x in xs:
            ponints.append([x, p(x)])
    return ponints

def plotScatterP(points):
    fig, ax = plt.subplots()
    x, y = getXY(points)
    ax.scatter(x, y)
    for i in range(len(x)):
        ax.annotate(i, (x[i], y[i]))
    plt.show()

def plot(begin, end, poptB, poptE):
    diff = (poptE - poptB)/50
    points = []
    for i in range(51):
        x_new, y_new = getPointsFromFit(begin, end, poptB)
        for j in range(len(x_new)-1):
            points.append([x_new[j+1], y_new[j+1]])
        poptB += diff
    return points

def getSmothApprox(points, xAxis):
    left, right = divRLinclu(points, xAxis)
    xl, yl, Lpopt = getPolyApprox(left)
    xr, yr, Rpopt = getPolyApprox(right)
    return xyToPoints(xl, yl) + xyToPoints(xr, yr), Lpopt, Rpopt

def OldmergeBlocks(blocks, img):
    out = blankImg(4960, 7016)
    begin = 0
    for block in blocks:
        image, end = perspectiveTransformation(img, block)
        out[0:4960, begin:end+begin] = image
        begin += end
    out = cv2.resize(out[0:4960, 0:begin], [7016, 4960])
    return out

def perspectiveTransformation(img, pts1):
    d = np.sum(np.square(pts1[1] - pts1[0])) ** 0.5
    xend = int(d)
    yend = abs(pts1[1][1] - pts1[0][1])
    pts2 = np.float32([[0, 0], [xend, 0],[0, yend],[xend, yend]])
    M = cv2.getPerspectiveTransform(np.float32(pts1), pts2)
    dst = cv2.warpPerspective(img, M, ([xend, yend]))
    return dst, xend, yend

def sortByX(points):
    return sorted(points, key=itemgetter(0))

def sortByY(points):
    return sorted(points, key=itemgetter(1))

def minDstPoints(arrayPoints, dst):
	minDst = inf
	minPoint = []
	for point in arrayPoints:
		curDst = np.sum(np.square(dst - point))
		if minDst > curDst:
			minDst = curDst
			minPoint = point
	return minPoint

def edges(arrayPoints):
    result = []
    result.append(minDstPoints(arrayPoints, [0, 0]))
    result.append(minDstPoints(arrayPoints, [7016, 0]))
    result.append(minDstPoints(arrayPoints, [0, 4960]))
    result.append(minDstPoints(arrayPoints, [7016, 4960]))
    return result

def minDstX(arrayPoints, dst):
	minDst = inf
	minPoint = []
	for point in arrayPoints:
		curDst = np.square(dst[0] - point[0])
		if minDst > curDst:
			minDst = curDst
			minPoint = point
	return minPoint

def divUpDown(arrayPoints, yAxis):
    up = []
    down = []
    for point in arrayPoints:
        if point[1] > yAxis:
            up.append(point)
        else:
            down.append(point)
    return up, down

def divRLinclu(arrayPoints, xAxis):
    left = []
    right = []
    for point in arrayPoints:
        if point[0] > xAxis:
            right.append(point)
        elif point[0] == xAxis:
            right.append(point)
            left.append(point)
        else:
            left.append(point)
    return left, right

def divRL(arrayPoints, xRange):
    interval = []
    for point in arrayPoints:
        if point[0] > xRange[0] and point[0] < xRange[1]:
            interval.append(point)
    return interval

def cutPoints(points, yAxis, xRange):
    interval = divRL(points, xRange) + edges(points)
    up, down = divUpDown(interval, yAxis)
    return sortByX(np.unique(up, axis=0)), sortByX(np.unique(down, axis=0))

def TcutPoints(points, yAxis, xRange):
    interval = divRL(points, xRange)
    up, down = divUpDown(interval, yAxis)
    uMid, dMid = getMidEdge(up, down)
    Es = edges(points)
    offsite = np.array([0, 50])
    Es.append(uMid - offsite)
    Es.append(dMid + offsite)
    up, down = np.array(divUpDown(Es, yAxis))
    return sortByX(np.unique(up, axis=0)), sortByX(np.unique(down, axis=0))

def getMidEdge(up, down):
    return min(up, key=itemgetter(1)), max(down, key=itemgetter(1))

def getBeginEnd(up, down):
    if len(up) >= len(down):
        begin = down
        end = up
    else:
        begin = up
        end = down
    return np.array(begin).tolist(), np.array(end).tolist()

def getRightPoint(point, points):
    index = points.index(point) + 1
    if index >= len(points):
        return point
    return points[index]

def getBlock(up, down):
    allBlock = []
    begin, end = getBeginEnd(up, down)
    for point in begin:
        otherSide = minDstX(end, point)
        otherSideR = getRightPoint(otherSide, end)
        pointR = getRightPoint(point, begin)
        allBlock.append([point, pointR, otherSide, otherSideR])
    allBlock.pop()
    return np.array(allBlock)

def blankImg(x, y):
	return np.zeros((x, y), dtype=np.uint8)

if __name__ == '__main__':
    doc = fitz.open()
    filename = input("将pdf放入同一目录下并输入其全名: ")
    enhanPdf(doc, filename)
    doc.save(filename[:-4]+"(out)"+".pdf")