
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




###########################################################################################################################
import numpy as np

from sklearn.datasets import load_digits

digits = load_digits()
X = digits['data'] / np.max(digits['data'])

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=20, random_state=1000)
X_tsne = tsne.fit_transform(X)
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

probabilities = model.predict_generator(test_generator, 600)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
y_true = np.array([0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] *100 + [5] *100 )
#y_pred = probabilities > 0.5
print(probabilities)
y_pred = np.asarray(probabilities)
y_pred = np.argmax(probabilities,axis=1)

print(y_pred)

print(y_true)

#print(np.shape(probabilities))
print(confusion_matrix(y_true, y_pred))

print(accuracy_score(y_true, y_pred))

####################################################################################################################
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
####################################################################################################################
test_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    color_mode='landsat',
    batch_size=5000,
    class_mode='binary')

X_test = test_generator.next()

y_pred = model.predict(X_test[0])
y_true = X_test[1]
roc_auc_score(y_true, y_pred)


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

class_test=model_emotion.predict_classes(X_test)


classes = np.array(range(0,7))
classes
accuracy_score(classes[np.argmax(Y_test, axis=1)], class_test)


####################################################################################################################

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

import keras.backend as K

def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def recall_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many relevant items are selected?
    recall_score = c1 / c3

    return recall_score


# Train the model

# compile the model
model.compile(loss='categorical_crossentropy',
             optimizer=optimizers.RMSprop(lr=1e-4,decay=0.9),
             metrics=['accuracy',f1_score,recall_score])

# Train the model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size, verbose=1)

####################################################################################################################


import numpy as np
from tqdm import tqdm
train_set.reset()
n_steps = train_set.n // train_set.batch_size
y_preds = []
y_true = []
for i in tqdm(range(n_steps)):
    x_batch, y_batch = test_set.next()
    preds = classifier.predict(x_batch)
    y_preds.extend([np.argmax(pred) for pred in preds])
    y_true.extend([np.argmax(y) for y in y_batch])

from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_preds, average='macro')
print('F1: {:.3f}'.format(f1))




####################################################################################################################
model.fit(X_train, Y_train, validation_data=(X_valid,Y_valid),batch_size=32, \
              epochs=10, verbose=1)
score = model.evaluate(X_test, Y_test)
result = model.predict(X_test)
f1_score(Y_test, result, average='weighted')

####################################################################################################################

filenames = test_generator.filenames
nb_samples = len(filenames)
predict = classifier.predict_generator(test_generator,steps = nb_samples)

pred_prob = np.argmax(predict, axis=-1) #multiple categories

label_map = (training_set.class_indices)
label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
predictions = [label_map[k] for k in pred_prob]

correct, wrong = 0,0
for ypred, ytrue in zip(predictions,test_generator.filenames):
    if ypred== re.findall('[a-z]{1,2}', ytrue.split('_')[1])[0]:
	correct +=1
    else:
	 wrong +=

accuracy = float(correct)/float(len(test_generator.filenames))
print(accuracy)

####################################################################################################################
##### Learn

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



####################################################################################################################



####################################################################################################################





####################################################################################################################



####################################################################################################################



####################################################################################################################





####################################################################################################################



####################################################################################################################



####################################################################################################################





####################################################################################################################



####################################################################################################################



####################################################################################################################








####################################################################################################################




####################################################################################################################



####################################################################################################################



####################################################################################################################




####################################################################################################################



####################################################################################################################



####################################################################################################################





####################################################################################################################



####################################################################################################################



####################################################################################################################





####################################################################################################################



####################################################################################################################



####################################################################################################################





####################################################################################################################



####################################################################################################################



####################################################################################################################








####################################################################################################################




####################################################################################################################



####################################################################################################################



####################################################################################################################




####################################################################################################################



####################################################################################################################



####################################################################################################################





####################################################################################################################



####################################################################################################################



####################################################################################################################





####################################################################################################################



####################################################################################################################



####################################################################################################################





####################################################################################################################



####################################################################################################################



####################################################################################################################






