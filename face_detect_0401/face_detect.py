import cv2
import numpy as np
from time import sleep
from matplotlib import pyplot as plt
import keyboard as key

def Show_2Dhistogram():
    print("H,S 요소에 대한 히스토그램입니다")
    target = cv2.imread('TrippleFace.png')
    hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV) ## RGB-->HSV
    histt = cv2.calcHist([hsvt], [0, 1], None, [255, 255], [0, 180, 0, 256]) # row: H(색조) / col: S(채도)
    #'his'togram of 't'arget

    #cv2.imshow('histogram', histt)
    #cv2.waitKey(0) #원하는 H,S 값을 구하기 힘든 --> 흑백으로 나옴

    key.wait('Space bar')
    plt.imshow(histt, interpolation='nearest')  # 이렇게 하면 색상의 구분이 가능해져

    close_plt()

def CalcBack():
    # 역투영
    print('CalcBack')
    roi = cv2.imread('skin1.png') # 원하는 곳의 이미지
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    target = cv2.imread('TrippleFace.png') # 타겟이미지_목표
    hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256]) # 원하는 곳의 이미지의 히스토그램
    cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX) # 해당 히스토그램을 0,255 로 정규화

    calc = cv2.calcBackProject([hsvt], [0,1], roihist, [0,180,0,256],1) #타겟이미지에서 원하는 이미지의 히스토그램을 이용해서 추출
    cv2.imshow('calc', calc)

    cv_wait()
    return calc

#calc를 반환하는 CalcBack 함수와 같은 기능, 이미지 출력이 겹쳐서 imshow는 구현X / "return calc"만 구현
def CalcBack_no_image():
    # 역투영
    roi = cv2.imread('skin1.png') # 원하는 곳의 이미지
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    target = cv2.imread('TrippleFace.png') # 타겟이미지_목표
    hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

    roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256]) # 원하는 곳의 이미지의 히스토그램
    cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX) # 해당 히스토그램을 0,255 로 정규화
    calc = cv2.calcBackProject([hsvt], [0,1], roihist, [0,180,0,256],1) #타겟이미지에서 원하는 이미지의 히스토그램을 이용해서 추출
    #cv2.imshow('calc',calc)
    #cv2.waitKey(0)
    return calc


def Show_1Dhistogram(calc):
    # calc으로 1차원 히스토그램을 구현해 사용자가 임의로 임계값을 설정할 수 있다.
    print('Show 1D histogram')
    hist, bins = np.histogram(calc.flatten(), 256, [0, 256])
    plt.hist(calc.flatten(), 256, [0, 256], color='r')
    #cv2.imshow('calc', calc)
    plt.xlim([0, 256])

    plt.ion()       # make blocking function to non blocking function
    plt.show()
    close_plt()

def Thres(calc):
    #임계값을 0.5로 하고 그 이상은 흰색 그 아래는 검은색으로 설정.
    ret, binary_image = cv2.threshold(calc, 0.5, 255, cv2.THRESH_BINARY)
    cv2.imshow('threshold', binary_image)

    cv2.waitKey(0)      # 얘를 빼면 안되네????
    cv_wait()


def OTSU(calc):
    #image_blur = cv2.GaussianBlur(calc, (5,5), 0)
    #ret, binary_image = cv2.threshold(image_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, binary_image = cv2.threshold(calc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('OTSU', binary_image)

    cv_wait()

def cv_wait():
    wait = cv2.waitKey(0)
    while (wait != 32):
        wait = cv2.waitKey(0)
        print(wait)
    cv2.destroyAllWindows()

def close_plt():
    plt.pause(0.1)
    key.wait('Space bar')
    plt.clf()
    plt.close()

def main():

    Show_2Dhistogram()
    key.wait('Space bar')
    CalcBack()
    key.wait('Space bar')
    Show_1Dhistogram(CalcBack_no_image())
    key.wait('Space bar')
    Thres(CalcBack_no_image())
    key.wait('Space bar')
    OTSU(CalcBack_no_image())

if __name__=="__main__":
    main()