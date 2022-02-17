import tensorflow as tf
from numpy import ndarray

from osgeo import gdal
import matplotlib.pyplot as plt
from osgeo import gdal_array
import numpy

print(tf.__version__)
import numpy as np

Data = []
Labels = []
featuresDict = {
    'LandCover2': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'array': tf.io.FixedLenFeature(shape=[10, ], dtype=tf.float32)
}


def datasetGeneraton(fileName):
    dataset = tf.data.TFRecordDataset(fileName, compression_type='GZIP')
    parsedDataset = dataset.map(lambda example: tf.io.parse_single_example(example, featuresDict))
    return parsedDataset


dataset = datasetGeneraton('TrainSet_nys_nn_final_2.tfrecord.gz')
it = iter(dataset)
try:
    while True:
        record = next(it)
        array = record['array'].numpy()
        label = record['LandCover2'].numpy()
        Data.append(array)
        Labels.append(label)
except:
    pass

Data = np.array(Data)

Labels = np.array(Labels)  # I had to convert your labels from 1~5 to 0~4

print(Data, Labels)

num_classes = np.unique(Labels).shape[0]

num_features = Data.shape[1]

#######################################################
# Generating train, validation, and test data
#######################################################

from sklearn.model_selection import train_test_split

N = len(Labels)
index = np.arange(N)
X_train, X_test, Y_train, Y_test = train_test_split(Data, Labels, test_size=0.1, stratify=Labels)


def one_hot(a):
    b = np.zeros((a.size, num_classes))
    b[np.arange(a.size), a] = 1
    return b


Y_train = one_hot(Y_train.astype(np.int))
Y_test: ndarray = one_hot(Y_test.astype(np.int))


####################################################################
# Building a simple 2-layer Keras model (10 neurons in each layer)
####################################################################

def keras_model(input_shape, num_classes):
    from tensorflow.keras.layers import Input, Dense

    input = Input(shape=input_shape)
    x1 = Dense(10, activation='sigmoid')(input)
    x2 = Dense(10, activation='sigmoid')(x1)
    x3 = Dense(10, activation='sigmoid')(x2)

    output = Dense(num_classes, activation='softmax')(x3)

    return tf.keras.Model(input, output)


####################################################################
# Training model and evaluation
####################################################################

model = keras_model((num_features,), num_classes)
optimizer = tf.keras.optimizers.RMSprop(lr=0.01)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(x=X_train, y=Y_train, batch_size=32, epochs=100, validation_split=0.1)
_, test_accuracy = model.evaluate(X_test, Y_test)
print('\nEvaluated test accuracy = {:.3f}'.format(test_accuracy))

# trying to read the image using gdal
img = gdal.Open('imageToCOGeoTiffExample_1.tif')

# type(raster)
# band = img.GetRasterBand()
arr = img.ReadAsArray()
# DATA_r = arr.reshape((604 * 1216,10))

# trying to read the image using rasterio
# import rasterio
# from rasterio.plot import show
# fp = r'imageToCOGeoTiffExample.tif'
# img = rasterio.open(fp)

# display classified image using the trained moded
DATA = np.zeros((1345, 1749, 10))
size = len(arr)
print(size)
# for i in range(1,11):
#   array = img.read(i)
#    DATA[:,:,i-1] = array
# reshaping the created matrix data
# DATA_r = arr.reshape((604 * 1216,10))
DATA_r = np.transpose(arr, [1, 2, 0]).reshape(-1, 10)
# applying the trained model on the data

l = model(DATA_r)
l = l.numpy()
l = l.reshape(1345, 1749, 15)

size_l = len(l)

print(size_l, "size_l")
# reshaping the created matrix data

label_mask = np.zeros((DATA.shape[0], DATA.shape[1]))

for i in range(DATA.shape[0]):
    for j in range(DATA.shape[1]):
        label_mask[i, j] = np.argmax(l[i, j])

print(label_mask, "label_mask")

import matplotlib.pyplot as plt

plt.imshow(label_mask)

plt.show()
#fig.savefig('label_mask.tif')

# band = dataset.GetRasterBand(0)
# arr = band.ReadAsArray()
# plt.imshow(arr)
# import affine
# import cligj
# import click
# import enum34
# import rasterio
from sklearn.metrics import roc_curve, auc
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from aitertools import cycle

y_score = model.predict(Y_train)
# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= num_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()