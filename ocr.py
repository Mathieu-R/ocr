import argparse
from imutils import contours
import imutils
import numpy as np
import cv2
from pprint import pprint

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
  ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

  refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
  refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]

  refROIS = extract_digits_and_symbols(ref, refCnts)
  return refROIS

def process_img():
  # lecture et traitement de l'image de référence
  digits = process_ref()

  # lecture de l'image du compteur
  image = cv2.imread(args["image"])
  cv2.imshow('original', image)

  # passage en niveau de gris
  img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  #cv2.imshow('gray', img2gray)
  
  # détection de bords (Canny)
  edged = bords_detection(img2gray)
  #pprint(edged)
  cv2.imshow('edge', edged)

  # treshold (binarization)
  #thresh = cv2.threshold(edged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  tresh = cv2.adaptiveThreshold(edged, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
  #pprint(tresh)
  cv2.imshow('tresh', tresh)

  # détections des lignes afin d'isoler le rectangle 
  # contenant la valeur de consommation
  lines_detection(tresh)

  # Une fois le rectangle récupéré
  # On isole chaque chiffre
  contours_detections(tresh.copy(), digits)

def contours_detections(image, digits):
  contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if imutils.is_cv2() else contours[1]

  for (i, c) in enumerate(contours):
    MIN_WIDTH = 15
    MIN_HEIGHT = 15

    (x, y, w, h) = cv2.boundingRect(c)

    if (w >= MIN_WIDTH and h >= MIN_HEIGHT):
      match_character(c, digits)

def match_character(c, digits):
  scores = []

  for (_, digitROI) in digits.items():
    cv2.matchTemplate(c, digitROI, cv2.TM_CCOEFF)
    (_, score, _, _) = cv2.minMaxLoc(result)
    scores.append(score)
    print(score)
  
  scoreString = "".join(scores)
  print(scoreString)
  

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


def bords_detection(image, sigma=0.33):
  v = np.median(image)

  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  edged = cv2.Canny(image, lower, upper)

  return edged

if __name__ == "__main__":
  process_img()