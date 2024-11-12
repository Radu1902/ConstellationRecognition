import cv2 as cv
import numpy as np
import imgproc
import os
import modes

OUTPUT_DIRECTORY = "output"
TEMPLATES_DIRECTORY = "templates"
PROCESSING_DIRECTORY = "processing"

# region keypoint identification
def getThreshKeypoints(templ):
    contours, _ = cv.findContours(templ, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    keypoints = []

    for contour in contours:
        M = cv.moments(contour)
        if M["m00"] != 0:

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            keypoint = cv.KeyPoint(x=cX, y=cY, size=3)
            keypoints.append(keypoint)

    return keypoints

def getBlobKeypoints(image):
    # identifies white blobs within the image, returns the keypoints
    params = cv.SimpleBlobDetector_Params()

    params.minThreshold = 1
    params.maxThreshold = 250

    params.filterByArea = True
    params.minArea = 1

    params.filterByConvexity = True
    params.filterByInertia = True
    params.filterByCircularity = True
    params.blobColor = 255

    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)

    if (len(keypoints) > 60):
        print()
    return keypoints

# endregion

# region drawing
def getMarkedImage(image, keypoints, color):
    marked = cv.drawKeypoints(image, keypoints, np.ndarray([]), color, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return marked

def getCirclesImage(image, keypoints, color, circleSize):
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    circled = image
    for kp in keypoints:
        circled = cv.circle(image, (int(kp.pt[0]), int(kp.pt[1])), circleSize, color, -1)
    return circled

def getContours(image):
    # contours of threshed image
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours

def drawContours(image, contours, color):
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    return cv.drawContours(image, contours, -1, color, 2)
#endregion

# region calculation stuff
def distance(pt1, pt2):
    # euclidian distance btwn pt1 and pt2, pt1 and pt2 are tuples of floats containing coords
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def getAngle(a, vertex, b):
    initialArm = distance(vertex, a)
    terminalArm = distance(vertex, b)
    opposingSide = distance(a, b)

    if initialArm == 0 or terminalArm == 0:
        return 0.0

    cos_theta = (initialArm ** 2 + terminalArm ** 2 - opposingSide ** 2) / (2 * initialArm * terminalArm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    alpha = np.arccos(cos_theta)
    alpha = np.degrees(alpha)
    return alpha

def getAnglesTensor(keypoints):
    # param: tuple of opencv keypoints
    # return: 3 dimensional tensor containing every possible angle of every keypoint within the variable keypoints
    angles = np.zeros((len(keypoints), len(keypoints), len(keypoints)))

    for idx1 in np.arange(len(keypoints)):
        for idx2 in np.arange(len(keypoints)):
            for idx3 in np.arange(len(keypoints)):
                vertex = keypoints[idx1]
                initial = keypoints[idx2]
                terminal = keypoints[idx3]

                angles[idx1, idx2, idx3] = getAngle(initial.pt, vertex.pt, terminal.pt)
                #print("angle between", initial.pt, vertex.pt, terminal.pt, " is ", angles[idx1, idx2, idx3])

    return angles

def findAngles(imgAngles, templAngles, tolerance=2):
    # param: imgAngles, templAngles: arrays containing angle values. templAngles must be smaller or equal in size to imgAngles
    # param: tolerance determines how close 2 angles can be to be considered a match (equal)
    # searches for all values of templAngles within imgAngles. If all of them are found, returns the indices where these were found
    # in imgAngles. If cannot find all of them, return empty array. Also, if imgAngles or templAngles are all 0, return empty array.
    # return: array of indices where values of templAngles were found in imgAngles. Returns empty array if can't find all values of templAngles.

    if not np.any(imgAngles) or not np.any(templAngles):
        return np.empty((0,))

    foundIndices = []

    for angle in templAngles:
        differences = np.abs(imgAngles - angle)
        
        possible_indices = np.where(differences < tolerance)[0] # this a tuple containing an array, [0] is for extracting the array within
        
        possible_indices = [idx for idx in possible_indices if idx not in foundIndices] # filter so indices are not matched twice
        
        if possible_indices: # check if possible_indices is empty
            foundIndices.append(possible_indices[0]) # we only need one index of the possible_indices, first one will do.
        else:
            return np.empty((0,)) # if empty, then a value of templAngles is missing, so whole thing goes down. 
    
    return np.array(foundIndices, dtype='int32')



def findTemplate(imageKeys, templKeys, tolerance=8):
    # param: 2 tuples of opencv keypoints and an integer. templKeys should have smaller or equal size to imageKeys.
    # param: tolerance determines how lenient template matching is.
    # loop through both tensors, take every array within the two and call findAngles on them.
    # if the search within findAngles succeeds, it will return the indices where the angles match.
    # if that's the case, we trim imgTensor with np.ix_(), with the indices where the angles match as parameters. Resulting tensor will look
    # like templTensor if the two shapes match. verify this by making the difference between the two.
    # here tolerance comes in: get the max of the difference tensor, if it is smaller than the tolerance, the shape was found.

    matches = np.empty((0,))

    imgTensor = getAnglesTensor(imageKeys)
    templTensor = getAnglesTensor(templKeys)

    for i, imgVertex in enumerate(imgTensor):
        for j, templVertex in enumerate(templTensor):


            for k, imgAngles in enumerate(imgVertex):
                for l, templAngles in enumerate(templVertex):

                    foundAnglesIndices = findAngles(imgAngles, templAngles)

                    if foundAnglesIndices.size > 0:
                        trimmed = imgTensor[np.ix_(foundAnglesIndices, foundAnglesIndices, foundAnglesIndices)]

                        diff = templTensor - trimmed
                        diff = np.abs(diff)

                        maxim = np.max(diff)
                        if maxim < tolerance:
                            foundAnglesIndices = foundAnglesIndices.astype(np.int32)
                            imageKeys = np.array(list(imageKeys))
                            matchedKeypoints = imageKeys[foundAnglesIndices]

                            matches = np.append(matches, matchedKeypoints)
                            return matchedKeypoints
    return matches

def rankTemplate(imageKeys, templKeys, templName):
    # same as findTemplate, but instead of returning the matched keypoints, it returns a score of how likely an image is to 
    # contain the template
    score = 99999999

    imgTensor = getAnglesTensor(imageKeys)
    templTensor = getAnglesTensor(templKeys)


    for i, imgVertex in enumerate(imgTensor):
        for j, templVertex in enumerate(templTensor):

            for k, imgAngles in enumerate(imgVertex):
                for l, templAngles in enumerate(templVertex):

                    foundAnglesIndices = findAngles(imgAngles, templAngles)

                    if foundAnglesIndices.size > 0:
                        trimmed = imgTensor[np.ix_(foundAnglesIndices, foundAnglesIndices, foundAnglesIndices)]

                        diff = templTensor - trimmed
                        diff = np.abs(diff)

                        maxim = np.max(diff)
                        score = min(score, maxim)

    return score, templName
#endregion

def getMatchedKeypoints(imageKeys, constellation):
    templPath = "templates/" + constellation + ".png"
    template = cv.imread(templPath, cv.IMREAD_GRAYSCALE)

    templKeys = getThreshKeypoints(template)

    matchKeys = findTemplate(imageKeys, templKeys, tolerance=9)


    return matchKeys

def closestFit(image, imageKeys):
    imageKeys = getBlobKeypoints(image)


    scores = {}
    for filename in os.listdir(TEMPLATES_DIRECTORY):
        templPath = os.path.join(TEMPLATES_DIRECTORY, filename)
        
        template = cv.imread(templPath, cv.IMREAD_GRAYSCALE)

        templKeys = getThreshKeypoints(template)
        rank, templateName = rankTemplate(imageKeys, templKeys, filename)
        scores[templateName] = rank

    print(scores)

    closest = min(scores, key=scores.get)
    # remove ".png" from the string
    closest = closest[:-4]
    return closest

def process(upload_location, constellation, identification_mode, threshold, filter_mode, ksize, unique_id):
    if not os.path.exists(upload_location):
        print("upload location is invalid")
        return "upload location invalid"
    image = cv.imread(upload_location, cv.IMREAD_GRAYSCALE)
    if filter_mode == modes.Filter_mode.GAUSS:
        image = imgproc.gaussianBlur(image, ksize)
        cv.imwrite(os.path.join(PROCESSING_DIRECTORY, f"{unique_id}_filtered.png"), image)
    elif filter_mode == modes.Filter_mode.MEAN:
        image = imgproc.meanFilter(image, ksize)
        cv.imwrite(os.path.join(PROCESSING_DIRECTORY, f"{unique_id}_filtered.png"), image)

    imageKeys = ()
    if identification_mode == modes.Identification_mode.SIMPLE:
        imageKeys = getBlobKeypoints(image)
        drawnImg = getMarkedImage(image, imageKeys, (255, 0, 255))

        cv.imwrite(os.path.join(PROCESSING_DIRECTORY, f"{unique_id}_blobs.png"), drawnImg)
    elif identification_mode == modes.Identification_mode.OTSU:
        otsuImg = imgproc.otsu(image)
        otsuImg = imgproc.dilate(otsuImg)
        contours = getContours(otsuImg)
        contouredImg = drawContours(image, contours, (0, 255, 0))
        imageKeys = getThreshKeypoints(otsuImg)
        drawnOtsuImg = getCirclesImage(image, imageKeys, (255, 0, 255), 3)

        cv.imwrite(os.path.join(PROCESSING_DIRECTORY, f"{unique_id}_threshed.png"), otsuImg)
        cv.imwrite(os.path.join(PROCESSING_DIRECTORY, f"{unique_id}_contour.png"), contouredImg)
        cv.imwrite(os.path.join(PROCESSING_DIRECTORY, f"{unique_id}_blobs.png"), drawnOtsuImg)
    elif identification_mode == modes.Identification_mode.THRESHOLD:
        threshed = imgproc.threshold(image, threshold)
        threshed = imgproc.dilate(threshed)
        contours = getContours(threshed)
        contouredImg = drawContours(image, contours, (255, 128, 0))
        imageKeys = getThreshKeypoints(threshed)
        drawnThreshedImg = getCirclesImage(image, imageKeys, (255, 0, 255), 3)

        cv.imwrite(os.path.join(PROCESSING_DIRECTORY, f"{unique_id}_threshed.png"), threshed)
        cv.imwrite(os.path.join(PROCESSING_DIRECTORY, f"{unique_id}_contour.png"), contouredImg)
        cv.imwrite(os.path.join(PROCESSING_DIRECTORY, f"{unique_id}_blobs.png"), drawnThreshedImg)

    print(len(imageKeys))
    if len(imageKeys) > 60:
        return "(Error, too many stars. Try applying a filter over the image first.)"

    # check if a valid constellation name was provided
    templPath = "templates/" + constellation + ".png"
    if not os.path.exists(templPath):
        constellation = closestFit(image, imageKeys)
    
    matchedKeypoints = getMatchedKeypoints(imageKeys, constellation)

    if identification_mode == modes.Identification_mode.SIMPLE:
        matchedImage = getMarkedImage(image, matchedKeypoints, color=(0, 0, 255))
    else:
        matchedImage = getCirclesImage(image, matchedKeypoints, color=(0, 0, 255), circleSize=4)

    cv.imwrite(os.path.join(OUTPUT_DIRECTORY, f"{unique_id}_output.png"), matchedImage)
    return constellation