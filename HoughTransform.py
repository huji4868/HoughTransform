import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

def houghLineTransform():
        root = os.getcwd();
        imgPath = os.path.join(root,"lane.png")
        imgColor = cv.imread(imgPath,cv.IMREAD_COLOR)
        originalimage = imgColor.copy()
        img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
        imgBlur = cv.GaussianBlur(img,(5,5), 3)
        cannyEdge = cv.Canny(imgBlur,35,110)

        plt.figure(figsize=(20, 5))
        plt.subplot(141)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.subplot(142)
        plt.imshow(imgBlur, cmap="gray")
        plt.axis("off")
        plt.subplot(143)
        plt.imshow(cannyEdge, cmap="gray")
        plt.axis("off")
        plt.tight_layout()

        distResol = 1
        angleResol = np.pi/180
        threshold = 98 #min to identify a line
        lines = cv.HoughLines(cannyEdge,distResol,angleResol,threshold)

        k = 3000

        for curline in lines:
            rho, theta = curline[0]
            dhat = np.array([[np.cos(theta)],[np.sin(theta)]])
            d = rho*dhat
            lhat = np.array([[-np.sin(theta)],[np.cos(theta)]])
            p1 = d + k*lhat
            p2 = d - k*lhat
            p1 = p1.astype(int)
            p2 = p2.astype(int)
            cv.line(originalimage,(p1[0][0],p1[1][0]),(p2[0][0],p2[1][0]),(0,0,255), 2)

        plt.subplot(144)
        plt.imshow(cv.cvtColor(originalimage, cv.COLOR_BGR2RGB))
        plt.axis("off")

        plt.show()

if __name__ == "__main__":
    houghLineTransform()