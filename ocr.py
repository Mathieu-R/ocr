from pprint import pprint
from imutils import contours as ct
from imutils.perspective import four_point_transform
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-r", "--reference", required=True,
	help="path to reference font")
args = vars(ap.parse_args())

def extract_digits_and_symbols(ref, refCnts):
  digits = {}

  for (i, c) in enumerate(refCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    digits[i] = roi
  
  return digits

def process_ref():
  ref = cv2.imread(args["reference"])

  ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
  ref = imutils.resize(ref, width=400)
  ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

  refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
  refCnts = ct.sort_contours(refCnts, method="left-to-right")[0]

  refROIS = extract_digits_and_symbols(ref, refCnts)
  return refROIS

def process_img():
  rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
  sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

  # lecture et traitement de l'image de référence
  digits = process_ref()

  # lecture de l'image du compteur
  image = cv2.imread(args["image"])
  image = imutils.resize(image, height=500)

  # passage en niveau de gris
  img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  #blurred = cv2.GaussianBlur(img2gray, (5, 5), 0)
  #cv2.imshow('blur', blurred)

  # BLACKHAT: fait ressortir les parties plus foncées
  #blackhat = cv2.morphologyEx(img2gray, cv2.MORPH_BLACKHAT, rectKernel)
  #tophat = cv2.morphologyEx(img2gray, cv2.MORPH_TOPHAT, rectKernel)
  #cv2.imshow('blackhat', img2gray)

  # détection de bords (Sobel)
  edged = bords_detection(img2gray)
  cv2.imshow('edge', edged)

  # isolation du compteur
  counter = isolate_counter(edged.copy())
  cv2.imshow('counter', counter)

  tresh = closing_numbers(counter, rectKernel, sqKernel)
  #cv2.imshow('tresh', tresh)

  # détections des lignes afin d'isoler le rectangle 
  # contenant la valeur de consommation
  #lines_detection(tresh)

  # Une fois le rectangle récupéré
  # On isole chaque chiffre
  contours_detections(tresh.copy(), digits)
  cv2.waitKey(0)

def contours_detections(image, digits):
  contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if imutils.is_cv2() else contours[1]
  contours = ct.sort_contours(contours, method="left-to-right")[0]
  scores = []

  for c in contours:
    MIN_WIDTH = 15
    MIN_HEIGHT = 15

    (x, y, w, h) = cv2.boundingRect(c)
    roi = image[y:y + h, x:x + w]
    #cv2.imshow('number', roi)

    if (w >= MIN_WIDTH and h >= MIN_HEIGHT):
      roi = image[y:y + h, x:x + w]
      roi = cv2.resize(roi, (57, 88))
      score = match_character(roi, digits)
      scores.append(score)
  
  scoreString = "".join(scores)
  print("Consumption: {}".format(scoreString))

def match_character(roi, digits):
  scores = []

  for (digit, digitROI) in digits.items():
    result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
    (_, score, _, _) = cv2.minMaxLoc(result)
    scores.append(score)
  
  score = str(np.argmax(scores))
  return score

def isolate_counter(edged):
  cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if imutils.is_cv2() else cnts[1]
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
  display = None
  
  # loop over the contours
  for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
      display = approx
      break
  
  display = four_point_transform(edged, display.reshape(4, 2))
  return display

def lines_detection(image):
  MIN_ANGLE = 60 * np.pi / 180
  MAX_ANGLE = 120 * np.pi / 180
  filteredLines = []

  lines = cv2.HoughLines(image, 1, np.pi / 180, 200)

  for line in lines:
    #print(line[0][1])
    angle = line[0][0]
    if (angle > MIN_ANGLE and angle < MAX_ANGLE):
      filteredLines.append(line)

def closing_numbers(image, rectKernel, sqKernel):
  # apply a closing operation using the rectangular kernel to help
  # cloes gaps in between credit card number digits, then apply
  # Otsu's thresholding method to binarize the image
  edged = cv2.morphologyEx(image, cv2.MORPH_CLOSE, rectKernel)
  #thresh = cv2.threshold(edged, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
  thresh = cv2.adaptiveThreshold(edged, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
  thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
  return thresh

def bords_detection_sobel(blackhat):
  gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
  gradX = np.absolute(gradX)
  (minVal, maxVal) = (np.min(gradX), np.max(gradX))
  gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
  edged = gradX.astype("uint8")
  return edged

def bords_detection(image, sigma=0.33):
  v = np.median(image)

  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  edged = cv2.Canny(image, lower, upper)
  return edged

if __name__ == "__main__":
  process_img()