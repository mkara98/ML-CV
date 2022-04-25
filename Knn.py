import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import time
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn
import random
import csv
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# Grayscale
def BGR2GRAY(img):
    # Grayscale
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray


# Canny Edge dedection
def Canny_edge(img):
    # Canny Edge
    canny_edges = cv2.Canny(img, 100, 200)
    return canny_edges


# Gabor Filter
def Gabor_filter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get half size
    d = K_size // 2

    # prepare kernel
    gabor = np.zeros((K_size, K_size), dtype=np.float32)

    # each value
    for y in range(K_size):
        for x in range(K_size):
            # distance from center
            px = x - d
            py = y - d

            # degree -> radian
            theta = angle / 180. * np.pi

            # get kernel x
            _x = np.cos(theta) * px + np.sin(theta) * py

            # get kernel y
            _y = -np.sin(theta) * px + np.cos(theta) * py

            # fill kernel
            gabor[y, x] = np.exp(-(_x ** 2 + Gamma ** 2 * _y ** 2) / (2 * Sigma ** 2)) * np.cos(
                2 * np.pi * _x / Lambda + Psi)

    # kernel normalization
    gabor /= np.sum(np.abs(gabor))

    return gabor


# Use Gabor filter to act on the image
def Gabor_filtering(gray, K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get shape
    H, W = gray.shape

    # padding
    gray = np.pad(gray, (K_size // 2, K_size // 2), 'edge')

    # prepare out image
    out = np.zeros((H, W), dtype=np.float32)

    # get gabor filter
    gabor = Gabor_filter(K_size=K_size, Sigma=Sigma, Gamma=Gamma, Lambda=Lambda, Psi=0, angle=angle)

    # filtering
    for y in range(H):
        for x in range(W):
            out[y, x] = np.sum(gray[y: y + K_size, x: x + K_size] * gabor)

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


# Use 6 Gabor filters with different angles to perform feature extraction on the image
def Gabor_process(img):
    # get shape
    H, W, _ = img.shape

    # gray scale
    gray = BGR2GRAY(img).astype(np.float32)

    # define angle
    # As = [0, 45, 90, 135]
    As = [0, 30, 60, 90, 120, 150]

    # prepare pyplot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)

    out = np.zeros([H, W], dtype=np.float32)

    # each angle
    for i, A in enumerate(As):
        # gabor filtering
        _out = Gabor_filtering(gray, K_size=9, Sigma=1.5, Gamma=1.2, Lambda=1, angle=A)

        # add gabor filtered image
        out += _out

    # scale normalization
    out = out / out.max() * 255
    out = out.astype(np.uint8)

    return out


ImageArray = []


def createData(root_dir, lungType):
    for im in glob.glob(root_dir):
        # Read image
        img = cv2.imread(im).astype(np.uint8)
        resizeImage = cv2.resize(img, (48, 48))
        cannyOut = Canny_edge(resizeImage).flatten()
        gaborOut = Gabor_process(resizeImage).flatten()
        ImageArray.append((cannyOut + gaborOut, lungType))


testImageArray = []


def createTestData(root_dir):
    for im in glob.glob(root_dir):
        # Read image
        img = cv2.imread(im).astype(np.uint8)
        resizeImage = cv2.resize(img, (48, 48))
        cannyOut = Canny_edge(resizeImage).flatten()
        gaborOut = Gabor_process(resizeImage).flatten()
        testImageArray.append((cannyOut + gaborOut))


def testWeightedKnn(trainArray, validationArray, k):
    pred = []
    for val in validationArray:
        neighbours = weightedKnn(trainArray, val, k)
        pred_result = weightedPredict(neighbours)
        pred.append(pred_result)

    return np.array(pred)


def testKnn(trainArray, validationArray, k):
    pred = []
    for val in validationArray:
        neighbours = knn(trainArray, val, k)
        pred_result = predict(neighbours)
        pred.append(pred_result)

    return pred


def getDistance(array1, array2):
    dist = np.abs(array2 - array1).sum()
    return dist


def knn(allPoints, testPoint, k):
    distances = []
    neighbours = []
    for point, class_ in allPoints:
        dist = getDistance(point, testPoint)
        distances.append((class_, dist))
    distances.sort(key=lambda x: x[1])

    for i in range(k):
        neighbours.append(distances[i][0])

    return neighbours


def weightedKnn(allPoints, testPoint, k):
    distances = []
    neighbours = []
    for point, class_ in allPoints:
        dist = getDistance(point, testPoint)
        distances.append((point, class_, dist))
    distances.sort(key=lambda x: x[2])

    for i in range(k):
        neighbours.append((distances[i][2], distances[i][1]))

    return neighbours


def weightedPredict(neighbours):
    covid = 0
    normal = 0
    viral = 0
    for dist, class_ in neighbours:
        if class_ == "COVID":
            covid += 1 / dist
        elif class_ == "NORMAL":
            normal += 1 / dist
        else:
            viral += 1 / dist

    allDis = [covid, normal, viral]
    if covid == max(allDis):
        return "COVID"
    elif normal == max(allDis):
        return "NORMAL"
    else:
        return "VIRAL"


def splitArray(allArray):
    images = []
    lungs = []
    for image, lung in allArray:
        images.append(image)
        lungs.append(lung)
    images = np.array(images)
    lungs = np.array(lungs)
    return images, lungs


def predict(neighbours):
    typeList = []
    for neighbour in neighbours:
        typeList.append(neighbour)
    return max(typeList, key=typeList.count)


def findEqual(arr1, arr2):
    total_acc = 0
    zipped = zip(arr1, arr2)
    for x, y in zipped:
        if x == y:
            total_acc += 1

    return (total_acc / len(arr1)) * 100


createData("COVID/*", "COVID")

createData("NORMAL/*", "NORMAL")

createData("Viral Pneumonia/*", "VIRAL")

np.save('ImageArray', ImageArray)

ImageArray = np.load('ImageArray.npy', allow_pickle=True)
random.shuffle(ImageArray)

images, lungs = splitArray(ImageArray)

# This is K-Fold cross validation step
k = 5
kf = KFold(n_splits=k, random_state=None)

acc_score = []
for train_index, test_index in kf.split(images):
    time1 = time.time()
    trainArray = ImageArray[train_index]
    validationArray = images[test_index]
    modelResult = testWeightedKnn(trainArray=trainArray, validationArray=validationArray, k=4)
    pred_test = lungs[test_index]
    acc = findEqual(modelResult, pred_test)
    acc_score.append(acc)
    time2 = time.time()
    print("Step time is ", time2 - time1)

print("Weighted KNN score table", acc_score)
print("Weighted KNN avarage", sum(acc_score) / 5)

acc_score2 = []
for train_index, test_index in kf.split(images):
    time1 = time.time()
    trainArray = ImageArray[train_index]
    validationArray = images[test_index]
    modelResult = testKnn(trainArray=trainArray, validationArray=validationArray, k=4)
    pred_test = lungs[test_index]
    acc = findEqual(modelResult, pred_test)
    acc_score2.append(acc)
    time2 = time.time()
    print("Step time is ", time2 - time1)

print("KNN score table", acc_score2)
print("KNN avarage", sum(acc_score2) / 5)

# This is for test
testData = pd.read_csv('submission-1.csv')

testLungs = testData['Category']
testLungs2 = []
for test in testLungs:
   testLungs2.append(test)
np.save('testLungs', testLungs2)
testLungs2 = np.load('testLungs.npy', allow_pickle=True)

# createTestData("test/*")
print("1")
# np.save('testData', testImageArray)
testImageArray = np.load('testData.npy', allow_pickle=True)

time1 = time.time()
modelResult = testWeightedKnn(trainArray=ImageArray, validationArray=testImageArray, k=4)
acc = findEqual(modelResult, testLungs2)
print("Weighted KNN score table", acc)
time2 = time.time()
print("Step time is ", time2 - time1)
conf_matrix = confusion_matrix(y_true=testLungs2, y_pred=modelResult, labels=["COVID", "NORMAL", "VIRAL"])
classes = ["COVID", "NORMAL", "VIRAL"]
df_cm = pd.DataFrame(conf_matrix, index=[i for i in classes], columns=[i for i in classes])
sn.set(font_scale=1.0)
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, fmt='g')
plt.show()

with open('sub1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Id", "Category"])
    for i, pred in enumerate(modelResult):
        writer.writerow([i + 1, pred])

time1 = time.time()
modelResult1 = testKnn(trainArray=ImageArray, validationArray=testImageArray, k=4)
acc1 = findEqual(modelResult1, testLungs2)
print("KNN score table", acc1)
time2 = time.time()
print("Step time is ", time2 - time1)
