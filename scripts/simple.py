#!/usr/bin/env python

import numpy as np
import cv2

# Params
detector_type = "AKAZE"
img_filename = "9371ba1f-c2d6-410d-bc42-e0c8b33504a7-rotation.jpg"
template_filename = "abcbb6bd-8697-48b7-be9b-838d5c2fa7a1.jpg"

# Load the images in gray scale
img1 = cv2.imread("../py-images/" + img_filename, 0)
img2 = cv2.imread("../py-images/" + template_filename, 0)

# Choose detector
if (detector_type == "KAZE"):
    detector = cv2.KAZE_create()
if (detector_type == "AKAZE"):
    detector = cv2.AKAZE_create()
if (detector_type == "ORB"):
    detector = cv2.ORB_create()

# Find descriptors and keypoints
keyPoints1, descriptors1 = detector.detectAndCompute(img1, None)
keyPoints2, descriptors2 = detector.detectAndCompute(img2, None)

# Create brute-force matcher object
bf = cv2.BFMatcher()

# Match the descriptors
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Select the good matches using the ratio test
goodMatches = []

for m, n in matches:
    if m.distance < 0.7 * n.distance:
        goodMatches.append(m)

# Apply the homography transformation if we have enough good matches
MIN_MATCH_COUNT = 10

if len(goodMatches) > MIN_MATCH_COUNT:
    # Get the good key points positions
    sourcePoints = np.float32([ keyPoints1[m.queryIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
    destinationPoints = np.float32([ keyPoints2[m.trainIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)

    # Obtain the homography matrix
    M, mask = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    matchesMask = mask.ravel().tolist()

    # Apply the perspective transformation to the source image corners
    h, w = img1.shape
    corners = np.float32([ [0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0] ]).reshape(-1, 1, 2)
    transformedCorners = cv2.perspectiveTransform(corners, M)

    # Draw a polygon on the second image joining the transformed corners
    img2 = cv2.polylines(img2, [np.int32(transformedCorners)], True, (255, 255, 255), 2, cv2.LINE_AA)
else:
    print("Not enough matches are found - %d/%d" % (len(goodMatches), MIN_MATCH_COUNT))
    matchesMask = None

# Draw the matches
drawParameters = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
result = cv2.drawMatches(img1, keyPoints1, img2, keyPoints2, goodMatches, None, **drawParameters)

# Display the results
cv2.imwrite("../py-volumes/" + detector_type + "__" + "simple.jpg", result)
