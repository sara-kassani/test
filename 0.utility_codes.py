







########################################################################################################################




###########################################################################################################################






###########################################################################################################################


    fig1, ax1 = plt.subplots(figsize=(11, 8))
    ax1.plot(range(epoch+1), avg_scores_per_epoch)
    ax1.set_title("Average validation accuracy vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. val. accuracy")
    plt.savefig('accuracy_vs_epochs.png')
    plt.clf()

    fig2, ax2 = plt.subplots(figsize=(11, 8))
    ax2.plot(range(epoch+1), avg_loss_per_epoch,'r')
    ax2.plot(range(epoch+1), valid_loss_per_epoch,'b')
    ax2.set_title("loss vs epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("loss")
    plt.legend(['train', 'val'], loc='lower right')
    plt.savefig('loss_vs_epochs.png')
    plt.clf()

    fig3, ax3 = plt.subplots(figsize=(11, 8))
    ax3.plot(range(epoch+1), avg_iou_per_epoch)
    ax3.set_title("Average IoU vs epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Current IoU")
    plt.savefig('iou_vs_epochs.png')
    plt.clf()

    fig4, ax4 = plt.subplots(figsize=(11, 8))
    ax4.plot(range(epoch + 1), avg_acc_per_epoch,'r')
    ax4.plot(range(epoch + 1), avg_scores_per_epoch,'b')
    ax4.set_title("Accuracy vs epochs")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Accuracy")
    plt.legend(['train', 'val'], loc='lower right')
    plt.savefig('train_valid_acc.png')
    plt.clf()
    plt.close('all')




###########################################################################################################################

def list_directory(path):
    
    list_dir=[]   
    for i, folder in enumerate(os.listdir(path)):
        
        folder_abspath = os.path.join(path, folder)
        list_dir.append(folder_abspath)
    
    return list_dir


list_directory("data/")

###########################################################################################################################


from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

GPUs = get_available_gpus()
print(GPUs)


###########################################################################################################################

base_model1=InceptionResNetV2(input_shape= input_shape,weights=inception_resnet_v2_weights, include_top=False, input_tensor=input_tensor)

for layer in base_model1.layers:
        layer.name += '_1'

base_model2=InceptionResNetV2(input_shape= input_shape,weights=inception_resnet_v2_weights, include_top=False, input_tensor=input_tensor)

for layer in base_model1.layers:
        layer.name += '_2'

###########################################################################################################################
def mean_class_accuracy(scores, labels):
    pred = np.argmax(scores, axis=1)
    cf = confusion_matrix(labels, pred).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    return np.mean(cls_hit / cls_cnt)

mean_class_accuracy(scores = preds, labels=y_true)
###########################################################################################################################

prediction = model.predict(test_data)
accuracy = 0

for i, predict in enumerate(prediction):

    if np.argmax(predict) == y_true[i]:
        accuracy += 1

print("Average classification accuracy:", accuracy/len(prediction))
###########################################################################################################################


from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

GPUs = get_available_gpus()

###########################################################################################################################
base_model1=ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
lastLayer = base_model1.layers[-1].output
x1=GlobalAveragePooling2D()(lastLayer)
###########################################################################################################################


from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from scipy import interp
from itertools import cycle

th = 0.3

truth = test_generator.classes
predict_class = np.argmax(predicts, axis=1)

acc = accuracy_score(truth,predict_class > th)
prec = precision_score(truth,predict_class > th)
f1 = f1_score(truth,predict_class > th)
recall = recall_score(truth,predict_class > th)

print('Accuracy:  {:.4f}'.format(acc))
print('Precision: {:.4f}'.format(prec))
print('Recall:    {:.4f}'.format(recall))
print('F1:        {:.4f}'.format(f1))


###########################################################################################################################
plt.style.use("seaborn-ticks")
plt.style.use("seaborn-bright")
plt.style.use("seaborn-whitegrid")
plt.style.use("fivethirtyeight")
plt.style.use("classic")
plt.style.use("bmh")
plt.style.use("grayscale")

###########################################################################################################################
from datetime import datetime as dt
def get_experiment_id():
    time_str = dt.now().strftime('%Y-%m-%d-%H-%M-%S')
    experiment_id = 'Inceptin_{}'.format(time_str)

    return experiment_id

experiment_id = get_experiment_id()



###########################################################################################################################
from keras.backend.tensorflow_backend import get_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import set_session

def reset_keras_tf_session():
    """
    this function clears the gpu memory and set the 
    tf session to not use the whole gpu
    """
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))


reset_keras_tf_session()



###########################################################################################################################
# Binary roc curve
from keras.utils import layer_utils, np_utils
from scipy import interp
from itertools import cycle

# generate roc curve
n_classes = 2

# Plot linewidth.
lw = 2

#convert original class labels to the one-hot-encoding
y_test = np_utils.to_categorical(test_generator.classes, 2)

Y_test_pred = model.predict_generator(test_generator, test_generator.samples // test_generator.batch_size)
Y_test_predicted = np.argmax(Y_test_pred, axis=1)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], Y_test_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), Y_test_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.rcParams["axes.grid"] = False
plt.rcParams.update({'font.size': 12})

plt.figure(figsize=(8,6))

plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i, roc_auc[i]))
    
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(classnames[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curves')
plt.legend(loc="lower right")
plt.show()




###########################################################################################################################

for X_batch, Y_batch in train_generator:
for x in X_batch:
	x = np.expand_dims(x, axis=0)
	pred = model.predict(X_batch)
	print(pred.shape)
	print(X_batch.shape,Y_batch.shape)
###########################################################################################################################

from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test,y_pred)
average_precision = average_precision_score(y_test,y_pred)
plt.step(recall, precision, color='k', alpha=0.7, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(average_precision))
#####
plt.plot( recall,precision)
print (auc(recall,precision))
#####
fpr, tpr, thresholds = roc_curve(y_test,y_pred)
plt.plot( fpr, tpr)
print (auc(fpr, tpr))
#####

areaUnderROC = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic:Area under the curve = {0:0.2f}'.format(areaUnderROC))
plt.legend(loc="lower right")
plt.show()





###########################################################################################################################

# Visualize misclassified images

from keras.preprocessing.image import load_img, img_to_array, array_to_img

predicted_classes = Y_test_predicted
ground_truth = test_generator.classes

fnames = test_generator.filenames
label2index = test_generator.class_indices
# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())


errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),validation_generator.samples))

# Show the errors
for i in range(len(errors)):
    pred_class = np.argmax(predicts[errors[i]])
    pred_label = idx2label[pred_class]
    
    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}, class ID : {}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        predicts[errors[i]][pred_class], pred_class)
    
    original = load_img('{}/{}'.format(test_dir,fnames[errors[i]]))
    plt.figure(figsize=[7,7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
plt.show()
#############################################
# Print only names of files
for i in range(len(errors)):
    pred_class = np.argmax(predicts[errors[i]])
    pred_label = idx2label[pred_class]
    
    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}, class ID : {}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        predicts[errors[i]][pred_class], pred_class)
    print(fnames[errors[i]])  


###########################################################################################################################
# Visualize misclassified images - NPY
from keras.preprocessing.image import load_img, img_to_array, array_to_img

fnames = test_generator.filenames
label2index = test_generator.class_indices
# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())


errors = np.where(predictions != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),validation_generator.samples))

# Show the errors
for i in range(len(errors)):
    pred_class = np.argmax(prob[errors[i]])
    pred_label = idx2label[pred_class]
    
    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}, class ID : {}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        prob[errors[i]][pred_class], pred_class)
    
    original = load_img('{}/{}'.format(test_dir,fnames[errors[i]]))
    plt.figure(figsize=[7,7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
plt.show()

#############################################
# Print only names of files
for i in range(len(errors)):
        print(fnames[errors[i]])

###########################################################################################################################





###########################################################################################################################




###########################################################################################################################




###########################################################################################################################

tn, fp, fn, tp = cm.ravel()
print("Accuracy:",(tp+tn)/(tp+tn+fp+fn))
print("Precision:",(tp/(tp+fp)))
print("Recall:",(tp/(tp+fn)))
print("tp:", tp) 
print("fp:", fp) 
print("tn:",tn) 
print("fn:",fn) 


recall = (tp/(tp+fn))
precision=(tp/(tp+fp))

f1 = 2 / ( (1/recall) + (1 / precision))
print('F1 score:', f1)


###########################################################################################################################

from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Model

_input = Input((224,224,1)) 

conv1  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(_input)
conv2  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(conv1)
pool1  = MaxPooling2D((2, 2))(conv2)

conv3  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(pool1)
conv4  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(conv3)
pool2  = MaxPooling2D((2, 2))(conv4)

conv5  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(pool2)
conv6  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv5)
conv7  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv6)
pool3  = MaxPooling2D((2, 2))(conv7)

conv8  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool3)
conv9  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv8)
conv10 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv9)
pool4  = MaxPooling2D((2, 2))(conv10)

conv11 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool4)
conv12 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv11)
conv13 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv12)
pool5  = MaxPooling2D((2, 2))(conv13)

flat   = Flatten()(pool5)
dense1 = Dense(4096, activation="relu")(flat)
dense2 = Dense(4096, activation="relu")(dense1)
output = Dense(1000, activation="softmax")(dense2)

vgg16_model  = Model(inputs=_input, outputs=output)


###########################################################################################################################
dir = 'data/ISIC2018/train/non-melanoma/'
my_file = dir + 'desktop.ini'
if os.path.exists(my_file):
    os.remove(my_file)
###########################################################################################################################

history = model.fit(train_data, train_labels,
		epochs=epochs,
		batch_size=batch_size,
		validation_data=(validation_data, validation_labels))

with open(extracted_features_dir+'history'+model_name+'.txt','w') as f:
	f.write(str(history.history))


###########################################################################################################################


nb_train_samples = sum([len(files) for r, d, files in os.walk(train_dir)])
nb_validation_samples = sum([len(files) for r, d, files in os.walk(validation_dir)])


nb_train_samples = len(train_generator.filenames)
nb_validation_samples = len(validation_generator.filenames)


###########################################################################################################################

for i, layer in enumerate(model.layers):
    print(i, layer.name)


###########################################################################################################################

keras.backend.clear_session()


###########################################################################################################################

trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=30, trials=trials)
print(best)

import gc; gc.enable()
gc.collect()
del train_generator, validation_generator


###########################################################################################################################

def get_callbacks(params):
    callbacks =[EarlyStopping(monitor='val_acc', patience=5, verbose=1)]
    return callbacks

      callbacks=get_callbacks(params))
###########################################################################################################################
from keras.preprocessing.image import load_img, img_to_array, array_to_img

plt.imshow(array_to_img(X_test[0]))
plt.show()
#####
plt.imshow(X_train[0])
plt.show()


###########################################################################################################################

from keras.layers import DepthwiseConv2D
from keras_applications.imagenet_utils import _obtain_input_shape


from keras.applications.mobilenet import DepthwiseConv2D, relu6
###########################################################################################################################

from keras.layers import BatchNormalization, add, GlobalAveragePooling2D

model.add(BatchNormalization())
x = BatchNormalization()(x)

###########################################################################################################################
# replace and rename dataframe columns

data.columns = [c.replace(' ', '_') for c in df.columns]
data.columns = [c.replace('LOR_', 'LOR') for c in df.columns]
data.columns = [c.replace('Chance_of_Admit_', 'Chance_of_Admit') for c in df.columns]
data.columns = [c.replace('Chance_of_Admit', 'Admit') for c in df.columns]
data.head()

###########################################################################################################################

X = pd.get_dummies(X)
###########################################################################################################################
# AttributeError: 'str' object has no attribute 'decode' 
	
NasNetLarge_model = load_model('models/5.NASNetLarge-new-ISBI19-Model.h5', compile=False)

###########################################################################################################################

# Transfer Weights
temp_weights = [layer.get_weights() for layer in new_model.layers]
model = create_model(lr=0.0001,
                     dropout_rate=[0.7],
                     alpha=0.01)
for j in range(len(temp_weights)):
    model.layers[j].set_weights(temp_weights[j])
    
train_history = model.fit(x_train, y_train, verbose=1,
                          batch_size=64,
                          validation_data=(x_dev,y_dev), 
                          epochs=100,
                          callbacks=callback_list)

###########################################################################################################################

 X_train = np.zeros((num_train_samples, img_width, img_height, num_channels))
input_shape = (img_width, img_height, 3)
or 
input_shape = (img_height, img_width, 3)
###########################################################################################################################
for d in ['/device:GPU:0', '/device:GPU:1']:
    with tf.device(d):
        history = model.fit_generator(
          train_generator,
          steps_per_epoch = nb_train_samples // batch_size,
          epochs = epochs,
          validation_data = validation_generator,
          validation_steps = nb_validation_samples // batch_size)
###########################################################################################################################
# Generate training data using gpu:0
with tf.device('/GPU:0'):
        # Set up generator for training data
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            featurewise_center=True,
            featurewise_std_normalization=True)

        # Generate training data
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            seed = random_seed,
            shuffle = True,
            class_mode='categorical')

with tf.device('/GPU:1'):
         # Generate validation data using gpu:1
            validation_generator = train_datagen.flow_from_directory(
                validation_dir,
                target_size=(img_height, img_width),
                batch_size=batch_size,
                seed = random_seed,
                shuffle = True,
                class_mode='categorical')

            test_datagen = ImageDataGenerator(rescale=1. / 255)

            test_generator = test_datagen.flow_from_directory(
                test_dir,
                target_size=(img_height, img_width),
                batch_size=batch_size,
                seed = random_seed,
                shuffle = False,
                class_mode='categorical')

###########################################################################################################################

model = Sequential()

model.add(
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(224, 224, 3)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(units=1024, activation="relu"))
model.add(Dense(units=1024, activation="relu"))
model.add(Dense(units=12, activation="softmax"))


###########################################################################################################################

from keras.models import load_model
model = load_model("models/5.VGG19-Adam-Dropout-Model.h5")
model.load_weights("models/VGG19-Adam-Dropout-Weights.h5")

###########################################################################################################################
import _pickle as cPickle
import pickle
###########################################################################################################################
from tqdm import tqdm


X_train, y_train = [], []
for _ in tqdm(range(nb_train_samples)):
    x, y = train_generator.next()
    X_train.append(x[0])
    y_train.append(y[0])
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
y_train = np.argmax(y_train, axis=1)

X_validation, y_validation = [], []
for _ in tqdm(range(nb_validation_samples)):
    x_val, y_val = validation_generator.next()
    X_validation.append(x_val[0])
    y_validation.append(y_val[0])
X_validation = np.asarray(X_validation)
y_validation = np.asarray(y_validation)
y_validation = np.argmax(y_validation, axis=1)
# np.save('data/npy/X_validation.npy', X_validation)
# np.save('data/npy/y_validation.npy', y_validation)

X_test, y_test = [], []
for _ in tqdm(range(nb_test_samples)):
    x_t, y_t = test_generator.next()
    X_test.append(x_t[0])
    y_test.append(y_t[0])
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)
y_test = np.argmax(y_test, axis=1)
# np.save('data/npy/X_test.npy', X_test)
# np.save('data/npy/y_test.npy', y_test)

nb_train_samples = 386
nb_validation_samples = 199
nb_test_samples = 155
###########################################################################################################################

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import *
from keras.layers import *
from keras.regularizers import *
from keras.optimizers import *

from keras import applications

###########################################################################################################################
import keras
import tensorflow as tf
from keras import backend as K

print("Keras Version:", keras.__version__)
print("Tensorflow Version:", tf.__version__)
print("image dim ordering:", K.image_dim_ordering())
###########################################################################################################################

    
from keras.applications import ResNet50
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.regularizers import l2

model_dense_conv = ResNet50(weights='imagenet', include_top=False)  
    #Create your own input format
keras_input = Input(shape= input_shape, name = 'image_input')
    
    #Use the generated model 
output_dense_conv = model_dense_conv(keras_input)
    
    #Add the fully-connected layers 
x = Flatten(name='flatten')(output_dense_conv)
x = Dense(1024, activation= 'relu', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001), name='fc1')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation= 'relu', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001), name='fc2')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(output_classes, activation='softmax', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001), name='predictions')(x)
    
    #Create your own model 
model = Model(inputs=keras_input, outputs=x)

###########################################################################################################################
# extract the features from the second to the last fc layer
intermediate_lyr_model = Model(inputs=model.input,outputs=model.get_layer('fc2').output)
###########################################################################################################################
# Clean up Keras session by clearing memory. 
if K.backend()== 'tensorflow':
K.clear_session()
###########################################################################################################################
x = model.layers[-5].output
x = Flatten()(x)
###########################################################################################################################

from keras.applications import DenseNet201

base_model = DenseNet201(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dropout(0.5)(x)
prediction = Dense(output_classes, activation=tf.nn.softmax)(x)

model = Model(inputs=base_model.input,outputs=prediction)
###########################################################################################################################


total=sum(sum(cm))

accuracy = (cm[0,0]+cm[1,1]) / total
print ('Accuracy : ', accuracy)

sensitivity = cm[0,0]/(cm[0,0]+cm[1,0])
print('Sensitivity : ', sensitivity )

Specificity = cm[1,1]/(cm[1,1]+cm[0,1])
print('Specificity : ', Specificity )

###########################################################################################################################
# softmax dense error
prediction = Dense(output_classes, activation=tf.nn.softmax)(x)
or 
model.add(Activation(tf.nn.softmax))

###########################################################################################################################
sgd_opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

adam_opt = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5)
sgd_opt = SGD(lr=1e-06, momentum=0.0, decay=0.0, nesterov=False)

rmsp_opt = RMSprop(lr=1e-4, decay=0.9)
###########################################################################################################################
from keras import backend as K
K.image_data_format()

K.set_image_data_format('channels_first')
K.image_data_format()
###########################################################################################################################
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
###########################################################################################################################
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
###########################################################################################################################
def lr_schedule(epoch):
    if epoch < 15:
        return .01
    elif epoch < 28:
        return .002
    else:
        return .0004

jobs_base_dir = 'jobs'
job_name = 'vgg19_job'
model_name = 'vgg19'
job_path = "{}/{}".format(jobs_base_dir, job_name)
tensorboard_dir = "{}/{}".format(job_path, "tensorboard")
  
weights_path = "{}/{}".format(job_path, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")
checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True)
csv_logger = CSVLogger("{}/{}.log".format(job_path, model_name))
early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=25)
tensorboard = TensorBoard(log_dir="{}".format(tensorboard_dir), histogram_freq=0, batch_size=32, write_graph=True,
                          write_grads=False,
                          write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
lr_scheduler = LearningRateScheduler(lr_schedule)


history = model.fit_generator(train_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
    steps_per_epoch= 2048,
    epochs = 50,
    validation_data = validation_generator,
#     validation_steps=nb_validation_samples // batch_size,
    validation_steps = 1048,
    callbacks=[lr_scheduler, csv_logger, checkpointer, tensorboard, early_stopping])
###########################################################################################################################

#Reading in the dataset

df = pd.read_csv('fraud_prediction.csv')

#Dropping the target feature & the index

df = df.drop(['Unnamed: 0', 'isFraud'], axis = 1)

#Initializing K-means with 2 clusters

k_means = KMeans(n_clusters = 2)

#Fitting the model on the data

k_means.fit(df)

#Extracting labels 

target_labels = k_means.predict(df)

#Converting the labels to a series 

target_labels = pd.Series(target_labels)

#Merging the labels to the dataset

df = pd.merge(df, pd.DataFrame(target_labels), left_index=True, right_index=True)

#Renaming the target 

df['fraud'] = df[0]
df = df.drop([0], axis = 1)


from sklearn.manifold import TSNE

#Creating the features

features = df.drop('fraud', axis = 1).values

target = df['fraud'].values

#Initialize a TSNE object

tsne_object = TSNE()

#Fit and transform the features using the TSNE object

transformed = tsne_object.fit_transform(features)



#Creating a t-SNE visualization

x_axis = transformed[:,0]


y_axis = transformed[:,1]


plt.scatter(x_axis, y_axis, c = target)

plt.show()

###########################################################################################################################
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), History.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), History.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), History.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), History.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")



###########################################################################################################################

from keras.applications import DenseNet201
model = DenseNet201(weights = "imagenet", include_top=False, input_shape = (img_rows, img_cols, 3))

from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(2048, activation="elu", kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(x)
x = BatchNormalization()(x)
# x = Dropout(0.5)(x)
x = Dense(1024, activation="elu", kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(x)
x = BatchNormalization()(x)
# x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid", kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(x)

from keras.models import Model
# creating the final model 
model_final = Model(input = model.input, output = predictions)
###########################################################################################################################
print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Keras: ", K.__version__)
print("Numpy: ", np.__version__)
print("Tensorflow: ", tensorflow.__version__)
print(K.backend.backend())
print(K.backend.image_data_format())
print("GPU: ", get_gpu_name())
print(get_cuda_version())
print("CuDNN Version ", get_cudnn_version())


CPU_COUNT = multiprocessing.cpu_count()
GPU_COUNT = len(get_gpu_name())
print("CPUs: ", CPU_COUNT)
print("GPUs: ", GPU_COUNT)
##########################################################################################################################
# Create a generator for prediction
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=1,
        class_mode='categorical',
        shuffle=False)

# Get the filenames from the generator
fnames = validation_generator.filenames

# Get the ground truth from generator
ground_truth = validation_generator.classes

# Get the label to class mapping from the generator
label2index = validation_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())

# Get the predictions from the model using the generator
predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)
#print (predictions)

errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),validation_generator.samples))

# Show the errors
for i in range(len(errors)):
    pred_class = np.argmax(predictions[errors[i]])
    pred_label = idx2label[pred_class]
    
    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}, class ID : {}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        predictions[errors[i]][pred_class], pred_class)
    
    original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
    plt.figure(figsize=[7,7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.show()
#################################################################################################################
fnames = validation_generator2.filenames
 
ground_truth = validation_generator2.classes
 
label2index = validation_generator2.class_indices
 
# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())
prob = model.predict_generator(validation_generator2)
predictions=np.argmax(prob,axis=1)

errors = np.where(predictions != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),nb_validation_samples))
####################################################################################################################
testing_generator = validation_datagen.flow_from_directory(
        test_dir,
        target_size=(height, width),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)
 
# Get the filenames from the generator
fnames = testing_generator.filenames
 
# Get the ground truth from generator
ground_truth = testing_generator.classes
 
# Get the label to class mapping from the generator
label2index = testing_generator.class_indices
 
# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())
 
# Get the predictions from the model using the generator
predictions = model.predict_generator(testing_generator, 
                                      steps=testing_generator.samples/testing_generator.batch_size,
                                      verbose=1)
predicted_classes = np.argmax(predictions,axis=1)
 
errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),testing_generator.samples))
print(str(len(errors)/testing_generator.samples) + "%")
####################################################################################################################
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
####################################################################################################################
predictions = np.round(model_mura.predict_generator(test_generator, steps=3197//1))
predictions = predictions.flatten()
y_true = test_generator.classes

def print_all_metrics(y_true, y_pred):
    print("roc_auc_score: ", roc_auc_score(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print("Sensitivity: ", get_sensitivity(tp, fn))
    print("Specificity: ", get_specificity(tn, fp))
    print("Cohen-Cappa-Score: ", cohen_kappa_score(y_true, y_pred))
    print("F1 Score: ", f1_score(y_true, y_pred))


def get_sensitivity(tp, fn):
    return tp / (tp + fn)


def get_specificity(tn, fp):
return tn / (tn + fp)


print_all_metrics(y_true,y_pred)

####################################################################################################################
####################################################################################################################
####################################################################################################################
model.fit(x_train, y_train, nb_epoch=5, batch_size=64, class_weight=myclass_weight)

	#Evaluate the scores of the model
	scores = model.evaluate(x_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))
	probas = model.predict(x_test)
	pred_indices = np.argmax(probas, axis=1)
	classes = np.array(range(0,9))
	preds = classes[pred_indices]
	#model.save('../models/cnn_model4.h5')
	print('Log loss: {}'.format(log_loss(classes[np.argmax(y_test, axis=1)], probas)))
print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(y_test, axis=1)], preds))) 


####################################################################################################################
model.fit(X_train, Y_train, validation_data=(X_valid,Y_valid),batch_size=32, \
              epochs=10, verbose=1)
score = model.evaluate(X_test, Y_test)
result = model.predict(X_test)
f1_score(Y_test, result, average='weighted')


####################################################################################################################


history = model.fit_generator(datagen.flow(train_X, train_y, batch_size=128),
epochs=100, validation_data=(valid_X, valid_y), workers=4)

pred_y = model.predict(test_X)
pred_y = np.argmax(pred_y, 1)
####################################################################################################################
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

model.fit_generator(datagen.flow(x_train, y_train,
                        batch_size=batch_size),
                        epochs=epochs,
                        workers=4)

y_pred = np.argmax(model.predict(x_test), axis=-1)
print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='macro'))



####################################################################################################################

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
metrics = ['categorical_accuracy', 'precision', 'recall', 'fmeasure'])

####################################################################################################################
from sklearn.metrics import classification_report, confusion_matrix

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
C = confusion_matrix(test_generator.classes, y_pred)
C_n = C.astype('float') / C.sum(axis=1)[:, np.newaxis]
print(C_n)
# print('Classification Report')
# target_names = ['0', '1', '2', '3']
# print(
#     classification_report(
#         test_generator.classes, y_pred, target_names=target_names))



####################################################################################################################
random_seed = np.random.seed(1142)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split= 0.25,
    featurewise_center=True,
    featurewise_std_normalization=True)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    seed = random_seed,
    shuffle = True,
    subset = 'training',
    class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    seed = random_seed,
    shuffle = True,
    subset = 'validation',
    class_mode='categorical')
