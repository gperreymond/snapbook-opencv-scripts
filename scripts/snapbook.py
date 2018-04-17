#!/usr/bin/env python

import cv2
import numpy as np
import math

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
(kps1, descs1) = detector.detectAndCompute(img1, None)
(kps2, descs2) = detector.detectAndCompute(img2, None)

# Write image with blue keypoints
img1_kps = cv2.drawKeypoints(img1, kps1, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("../py-volumes/" + detector_type + "__" + img_filename, img1_kps)
img2_kps = cv2.drawKeypoints(img2, kps2, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("../py-volumes/" + detector_type + "__" + template_filename, img2_kps)

# Create brute-force matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

# Match the descriptors
matches = bf.match(descs1, descs2)

# Select the good matches using the ratio test
min_dist = 100;
max_dist = 0;
for m in matches:
    dist = m.distance;
    if (dist < min_dist):
        min_dist = dist
    if (dist > max_dist):
        max_dist = dist;

good = []
for m in matches:
    if (m.distance <= 3 * min_dist):
        good.append(m)

# Apply the homography transformation if we have enough good matches
src_pts = np.float32([ kps1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kps2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS, 3);
matchesMask = mask.ravel().tolist()

# Apply the perspective transformation to the source image corners
h, w = img1.shape
corners = np.float32([ [0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0] ]).reshape(-1, 1, 2)
transformedCorners = cv2.perspectiveTransform(corners, M)

# Draw a polygon on the second image joining the transformed corners
img2 = cv2.polylines(img2, [np.int32(transformedCorners)], True, (255, 255, 255), 2, cv2.LINE_AA)

coeffs = []
det = cv2.determinant(M)
n1 = math.sqrt(M[0, 0] * M[0, 0] + M[1, 0] * M[1, 0])
n2 = math.sqrt(M[0, 1] * M[0, 1] + M[1, 1] * M[1, 1])
n3 = math.sqrt(M[2, 0] * M[2, 0] + M[2, 1] * M[2, 1])

coeffs.append(det)
coeffs.append(n1)
coeffs.append(n2)
coeffs.append(n3)

# Display the results
coincide = format(len(good))
if (det < 0 or math.fabs(det) < 2e-05):
    coincide = 0
if (n1 > 4 or n1 < 0.1):
    coincide = 0
if (n2 > 4 or n2 < 0.1):
    coincide = 0
if (n3 > 0.002):
    coincide = 0

# Draw the matches
drawParameters = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
result = cv2.drawMatches(img1, kps1, img2, kps2, good, None, **drawParameters)

# Display the results
cv2.imwrite("../py-volumes/" + detector_type + "__" + "snapbook.jpg", result)
