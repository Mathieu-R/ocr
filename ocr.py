from pprint import pprint
from imutils import contours as ct
from imutils.perspective import four_point_transform
from PIL import Image
from skimage import morphology
import numpy as np
import argparse
import imutils
import cv2
import pytesseract
import sys

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
    #cv2.imshow('roi', roi)
    roi = cv2.resize(roi, (57, 88))
    digits[i] = roi
  
  return digits

def process_ref():
  ref = cv2.imread(args["reference"])

  ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
  ref = imutils.resize(ref, width=400)
  ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
  #cv2.imshow('ref', ref)

  refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
  refCnts = ct.sort_contours(refCnts, method="left-to-right")[0]

  refROIS = extract_digits_and_symbols(ref, refCnts)
  return refROIS

def process_img():
  # lecture et traitement de l'image de référence
  digits = process_ref()

  # lecture de l'image du compteur
  image = cv2.imread(args["image"])
  image = imutils.resize(image, width=300)

  # passage en niveau de gris
  img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # détection de bords (Canny / Sobel)
  edged = bords_detection_canny(img2gray)

  # isolation du compteur
  #counter = isolate_counter(edged.copy(), img2gray.copy())
  #counter = edged.copy() # test purpose
  counter = isolate_counter(edged, img2gray)
  #cv2.imshow('counter', counter)

  # sans cela, tesseract est incapable de détecter les chiffres
  blurred = cv2.GaussianBlur(counter, (5, 5), 0)
  #cv2.imshow('blur', blurred)

  # accentue les chiffres
  higlightedNumbers = highlight_numbers(blurred)
  cv2.imshow('higlighted', higlightedNumbers)

  # convertit numpy ndarray en image
  # pour la passer à tesseract
  img = Image.fromarray(higlightedNumbers)
  txt = pytesseract.image_to_string(img)

  edged = bords_detection_canny(higlightedNumbers)

  # On isole chaque chiffre
  score = contours_detections(counter.copy(), edged.copy(), digits)
  print("[OPENCV Matching] Consommation: {}".format(score))
  print("[TESSERACT] Consommation: {}".format(txt))
  cv2.waitKey(0)

def contours_detections(original, image, digits):
  contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if imutils.is_cv2() else contours[1]
  contours = ct.sort_contours(contours, method="left-to-right")[0]
  scores = []

  for c in contours:
    MIN_WIDTH = 15
    MIN_HEIGHT = 20

    (x, y, w, h) = cv2.boundingRect(c)
    roi = image[y:y + h, x:x + w]
    cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #cv2.imshow('number', roi)

    if (w >= MIN_WIDTH and h >= MIN_HEIGHT):
      roi = image[y:y + h, x:x + w]
      roi = cv2.resize(roi, (57, 88))
      score = match_character(roi, digits)
      scores.append(score)
  
  #cv2.imshow('img - rect', original)
  scoreString = "".join(scores)
  return scoreString

def match_character(roi, digits):
  scores = []

  for (digit, digitROI) in digits.items():
    result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
    (_, score, _, _) = cv2.minMaxLoc(result)
    scores.append(score)
  
  score = str(np.argmax(scores))
  return score

def isolate_counter(edged, gray):
  cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if imutils.is_cv2() else cnts[1]
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
  display = None
  
  # loop over the contours
  for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    (x, y, w, h) = cv2.boundingRect(c)
    #print(f"[{w} - {h}, {approx}")

    if len(approx) == 4:
      display = approx
      break
  
  display = four_point_transform(gray, display.reshape(4, 2))
  return display

def highlight_numbers(image):
  # better results with blurred image
  # https://docs.opencv.org/3.3.1/d7/d4d/tutorial_py_thresholding.html
  thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
  return thresh

def bords_detection_sobel(image):
  gradX = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
  gradX = np.absolute(gradX)
  (minVal, maxVal) = (np.min(gradX), np.max(gradX))
  gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
  edged = gradX.astype("uint8")
  return edged

def bords_detection_canny(image, sigma=0.33):
  v = np.median(image)

  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  edged = cv2.Canny(image, lower, upper)
  return edged

if __name__ == "__main__":
  try:
    process_img()
  except KeyboardInterrupt:
    sys.exit(0)