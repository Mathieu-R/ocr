import argparse
import imutils
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-r", "--reference", required=True,
	help="path to reference font")
args = vars(ap.parse_args())

def extract_digits_and_symbols(image, charContours, minW=5, minH=15):
  iterator = charContours.__iter__()
  rois = []
  locations = [] # coordinates
  while True:
    try: 
      char = next(iterator)
      (x, y, width, height) = cv2.boundingRect(char)
      roi = None
      if width >= minW and height >= minH:
        # extract char
        roi = image[y:(y + height), x:(x + width)]
        rois.append(roi)
        locations.append((x, y, x + width, y + width))
    except StopIteration:
      break
  
  return (rois, locations)

def process_ref():
  CHARNAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

  ref = cv2.imread(args["reference"])

  ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
  ref = imutils.resize(ref, width=400)
  ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

  refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
  refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]

  refROIS = extract_digits_and_symbols(ref, refCnts, minW=10, minH=20)

  chars = {} # {"character": "associated roi image"}

  for (name, roi) in zip(CHARNAMES, refROIS):
    roi = cv2.resize(roi, (36, 36))
    chars[name] = roi
  
  return chars

def process_img():
  # lecture et traitement de l'image de référence
  chars = process_ref()

  # lecture de l'image du compteur
  image = cv2.imread(args["image"])
  # passage en niveau de gris
  img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  # détection de bords (Canny)
  edged = bords_detection(img2gray)

  cv2.imshow(edged)

  # treshold ?

  # détections des lignes afin d'isoler le rectangle 
  # contenant la valeur de consommation
  lines_detection(edged)

  # Une fois le rectangle récupéré
  # On isole chaque chiffre
  contours_detections(edged.copy())

def contours_detections(image):
  contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if imutils.is_cv2() else contours[1]

  for (i, c) in enumerate(contours):
    MIN_WIDTH = 15
    MIN_HEIGHT = 15

    (x, y, w, h) = cv2.boundingRect(c)

    if (w >= MIN_WIDTH and h >= MIN_HEIGHT):
      match_character(c)

def match_character(c):
  CHARNAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
  scores = []

  for charName in CHARNAMES:
    cv2.matchTemplate(c, chars[charName], cv2.TM_CC0EFF)
    (_, score, _, _) = cv2.minMaxLoc(result)
    scores.append(score)
    print(score)
  

def lines_detection(image):
  MIN_ANGLE = 60 * np.pi / 180
  MAX_ANGLE = 120 * np.pi / 180
  filteredLines = []

  lines = cv2.HoughLines(image, 1, np.pi / 180, 200)

  for line in lines:
    angle = line[1]
    if (angle > MIN_ANGLE and angle < MAX_ANGLE):
      filteredLines.append(line)


def bords_detection(image, sigma=0.33):
  v = np.median(image)

  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  imgEdged = cv2.Canny(img2gray, lower, upper)

  return imgEdged

if __name__ == "__main__":
  main()