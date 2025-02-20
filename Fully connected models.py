import pandas as pd
import re
import numpy as np
from sklearn import preprocessing
import pickle
import pandas as pd
import pickle
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow import keras
import random as rn
from keras.models import Model,save_model, load_model, Sequential
from keras.layers import Flatten,AveragePooling1D,Dropout,Dense,Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import IPython
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import preprocessing
from keras.utils import to_categorical
import pydot
import graphviz
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import kerastuner as kt
from kerastuner import HyperModel
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#read in pkl datasets
final_df_y = pd.read_pickle('/content/drive/My Drive/dataset/final_pickle/New_final_df_y.pickle')
final_df_x = pd.read_pickle('/content/drive/My Drive/dataset/final_pickle/New_final_df_x.pickle')

#check dataframe
final_df_x.head(5)

#Check if still numpy array
type(final_df_x['0_x'][0])

#Function to remove duplicates and labels(y) and data(x)
def added_preprocessing(dataset_x,dataset_y):
    data_x=dataset_x.drop_duplicates(subset=['screen_name'], keep='first')
    data_y=dataset_y.drop_duplicates(subset=['screen_name'], keep='first')
    labels= data_y['country_code']
    return labels, data_x

#datasets without duplicates
labels,data= added_preprocessing(final_df_x final_df_y)

#Datasets of the different model 
combined_features=data.filter(regex='^(?!screen_name)')
acct_features=data.filter(regex='^(?!\d+|screen_name)')
PB_and_TweetText=data.filter(regex='\d+')
text_feature=PB_and_TweetText.filter(regex='_x')
profile_banner_features=PB_and_TweetText.filter(regex='^((?!_x).)*$')

print(combined_features.shape)
print(acct_features.shape)
print(PB_and_TweetText.shape)
print(text_feature.shape)
print(profile_banner_features.shape)

#gets country label number
num_of_countries=labels.nunique()

#check labels
print(labels.shape)

#check data
print(data.shape)

#Label encoding 
def label_encoding(labels):
    LE = LabelEncoder()
    fit=LE.fit(labels.astype(str))
    labels =fit.transform(labels)
    labels = to_categorical(labels)
    return labels, fit

#Test label_encoding
labels, fit=label_encoding(labels)


#####COMBINED MODEL########
#convert list of arrays to numpy array
data= np.array(combined_features)

#Train/test set split (80/20)
train_data,test_data,train_labels,test_labels=train_test_split(data,labels,test_size=0.2, random_state=123)

#Train/validation set split (60/20)
train_data,valid_data,train_labels,valid_labels=train_test_split(train_data,train_labels,test_size=0.2, random_state=123)

#gets shape of the data for the model
input_shape_combined=train_data[0].shape
input_shape_combined

#generate random seed
def random_seeds(seed):
   os.environ['PYTHONHASHSEED']=str(seed)
   tf.random.set_seed(seed)
   np.random.seed(seed)
   rn.seed(seed)

#set seed
random_seeds(1234)

#Function for model flow. Does not include any parameter tuning
def model_flow(model_name, num_countries, input_shape):
    inputs = keras.Input(shape=(input_shape), name="Combined_inputs")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.BatchNormalization(name="normalization_1")(x)
    x = layers.Dense(32, activation="relu",name="dense_2")(x)
    x = layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.2), name="dense_3")(x)
    x = layers.Dropout(0.15)(x)
    outputs = layers.Dense(num_countries, activation="softmax",name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model

#test model_flow
Combined_model = model_flow("Combined",11, input_shape_combined)

#model summary
Combined_model.summary()

#model diagram
tf.keras.utils.plot_model(Combined_model)

#Compile  model
Combined_model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history_combined = Combined_model.fit(train_data, train_labels, epochs=10, batch_size=15)

#Fitting on training and validation data
print("Fit model on training data")
history_combined = Combined_model.fit(train_data, train_labels, epochs=10, batch_size=15,
                   validation_data=(valid_data, valid_labels))
# model.save("/prediction.h5")

#Accuracy plot of train vs test
def accuracy_plot(title_, history_fit):
    plt.plot(history_fit.history['accuracy'])
    plt.plot(history_fit.history['val_accuracy'])
    plt.title(title_)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.show()
    return plt

#test accuracy_plot
plot_combined=accuracy_plot('Model accuracy of Combined Model', history_combined)

# Evaluate the model on the test data
print("Evaluate on test data")
model_results = Combined_model.evaluate(test_data, test_labels, batch_size=10)

# Generate predictions (location probabilities)
def predict_test(model,test_data, test_labels, fit):
    predictions = model.predict(test_data)
    y_true=np.argmax(test_labels, axis=1)
    y_pred=np.argmax(predictions, axis = 1)
    #get labels of prediction
    label_pred=fit.inverse_transform(y_pred)
    #metrics
    report = classification_report(y_true, y_pred)
    return label_pred, report

#test predict_test
predicted_countries, metrics_report=predict_test(Combined_model,test_data, test_labels, fit)

"""****************************************************************************

Combined Model Results
"""

#train vs validation accuracy plot
plot_combined=accuracy_plot('Model accuracy of Combined Model', history_combined)
print(plot_combined)

#train vs validation accuracy plot
print(plot_combined)
#test accuracy
print("test loss, test accuracy:", model_results)
#Countries labels predicted
#print(predicted_countries)
#Classification report
print(metrics_report)

"""****************************************************************************

**Account Information model**
"""
#convert to numpy
data_acc= np.array(acct_features)

#Train/test set split (80/20)
train_data_acc,test_data_acc,train_labels_acc,test_labels_acc=train_test_split(data_acc,labels,test_size=0.2, random_state=123)

#Train/validation set split (60/20)
train_data_acc,valid_data_acc,train_labels_acc,valid_labels_acc=train_test_split(train_data_acc,train_labels_acc,test_size=0.2, random_state=123)

#input shape for account features
input_shape_acc=train_data_acc[0].shape

#model flow and summary
Acc_model = model_flow("Account_Info",num_of_countries, input_shape_acc)
Acc_model.summary()

#Compile  model
Acc_model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Fitting on training and validation data
print("Fit model on Account Information features training data")
history_acc = Acc_model.fit(train_data_acc, train_labels_acc, epochs=10, batch_size=15,
                   validation_data=(valid_data_acc, valid_labels_acc))

# Evaluate the model on the test data
print("Evaluate on Account Information features test data")
Acc_model_results = Acc_model.evaluate(test_data_acc, test_labels_acc, batch_size=10)

#test predict_test
predicted_countries_acc, metrics_report_acc=predict_test(Acc_model,test_data_acc, test_labels_acc, fit)

"""****************************************************************************

Account Information Model Results
"""

#train vs validation accuracy plot
plot_acc=accuracy_plot('Model accuracy of Account Information Features Model', history_acc)
print(plot_acc)

#train vs validation accuracy plot
print(plot_acc)
#test accuracy
print("test loss, test accuracy:", Acc_model_results)
#Countries labels predicted
#print(predicted_countries_acc)
#Classification report
print(metrics_report_acc)

"""****************************************************************************

**Profile and Banner images model**
"""
#convert list of arrays to numpy array
data_pb= np.array(profile_banner_features)

#Train/test set split (80/20)
train_data_pb,test_data_pb,train_labels_pb,test_labels_pb=train_test_split(data_pb,labels,test_size=0.2, random_state=123)

#Train/validation set split (60/20)
train_data_pb,valid_data_pb,train_labels_pb,valid_labels_pb=train_test_split(train_data_pb,train_labels_pb,test_size=0.2, random_state=123)

#input shape for image embeddings
input_shape_pb=train_data_pb[0].shape
input_shape_pb

#model flow and summary
PB_model = model_flow("Profile_Banner_images",num_of_countries, input_shape_pb)
PB_model.summary()

#Compile  model
PB_model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Fitting on training and validation data
print("Fit model on Profile and Banner images training data")
history_pb = PB_model.fit(train_data_pb, train_labels_pb, epochs=10, batch_size=15,
                   validation_data=(valid_data_pb, valid_labels_pb))

# Evaluate the model on the test data
print("Evaluate on Profile and Banner Images test data")
PB_model_results = PB_model.evaluate(test_data_pb, test_labels_pb, batch_size=10)

#test predict_test
predicted_countries_pb, metrics_report_pb=predict_test(PB_model,test_data_pb, test_labels_pb, fit)

"""****************************************************************************

Profile and Banner Images Model Results
"""

#train vs validation accuracy plot
plot_pb=accuracy_plot('Model accuracy of Profile and Banner Images Model', history_pb)
print(plot_pb)

#test accuracy
print("test loss, test accuracy:", PB_model_results)
#Countries labels predicted
#print(predicted_countries_pb)
#Classification report
print(metrics_report_pb)

"""****************************************************************************

**Tweet Text model**
"""

#make sure numpy array
data_tweettext= np.array(text_feature)

#Train/test set split (80/20)
train_data_tweettext,test_data_tweettext,train_labels_tweettext,test_labels_tweettext=train_test_split(data_tweettext,labels,test_size=0.2, random_state=123)

#Train/validation set split (60/20)
train_data_tweettext,valid_data_tweettext,train_labels_tweettext,valid_labels_tweettext=train_test_split(train_data_tweettext,train_labels_tweettext,test_size=0.2, random_state=123)

#input shape for tweet embeddings
input_shape_tweettext=train_data_tweettext[0].shape
input_shape_tweettext

#model flow and summary
Tweettext_model = model_flow("Tweet_text",num_of_countries, input_shape_tweettext)
Tweettext_model.summary()

#Compile  model
Tweettext_model.compile(optimizer='adam',
             loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])

#Fitting on training and validation data
print("Fit model on Tweet Text training data")
history_tweettext = Tweettext_model.fit(train_data_tweettext, train_labels_tweettext, epochs=10, batch_size=15,
                   validation_data=(valid_data_tweettext, valid_labels_tweettext))

# Evaluate the model on the test data
print("Evaluate on Tweet Text test data")
Tweettext_model_results = Tweettext_model.evaluate(test_data_tweettext,test_labels_tweettext, batch_size=10)

#test predict_test
predicted_countries_tweettext, metrics_report_tweettext=predict_test(Tweettext_model,test_data_tweettext,test_labels_tweettext, fit)

"""****************************************************************************

Tweet Text Model Results
"""

#train vs validation accuracy plot
plot_tweettext=accuracy_plot('Model accuracy of Tweet Text Model', history_tweettext)
print(plot_tweettext)

#test accuracy
print("test loss, test accuracy:", Tweettext_model_results)
#Countries labels predicted
#print(predicted_countries_tweettext)
#Classification report
print(metrics_report_tweettext)

"""****************************************************************************

**Profile+Banner+TweetText model**
"""

#convert list of arrays to numpy array
data_PB_TweetText= np.array(PB_and_TweetText)

#Train/test set split (80/20)
train_data_PB_TweetText,test_data_PB_TweetText,train_labels_PB_TweetText,test_labels_PB_TweetText=train_test_split(data_PB_TweetText,labels,test_size=0.2, random_state=123)

#Train/validation set split (60/20)
train_data_PB_TweetText,valid_data_PB_TweetText,train_labels_PB_TweetText,valid_labels_PB_TweetText=train_test_split(train_data_PB_TweetText,train_labels_PB_TweetText,test_size=0.2, random_state=123)

#input shape for image and tweet embeddings
input_shape_PB_TweetText=train_data_PB_TweetText[0].shape
input_shape_PB_TweetText

#model flow and summary
PB_TweetText_model = model_flow("Images_Tweettext",num_of_countries, input_shape_PB_TweetText)
PB_TweetText_model.summary()

#Compile  model
PB_TweetText_model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Fitting on training and validation data
print("Fit model on Images and Tweet text training data")
history_PB_TweetText = PB_TweetText_model.fit(train_data_PB_TweetText, train_labels_PB_TweetText, epochs=10, batch_size=15,
                   validation_data=(valid_data_PB_TweetText, valid_labels_PB_TweetText))

#Evaluate the model on the test data
print("Evaluate on Profile and Banner Images test data")
PB_TweetText_model_results = PB_TweetText_model.evaluate(test_data_PB_TweetText, test_labels_PB_TweetText, batch_size=10)

#test predict_test
predicted_countries_PB_TweetText, metrics_report_PB_TweetText=predict_test(PB_TweetText_model,test_data_PB_TweetText, test_labels_PB_TweetText, fit)

"""****************************************************************************

Profile+Banner+TweetText Model Results
"""

#train vs validation accuracy plot
plot_PB_TweetText=accuracy_plot('Model accuracy of Images and Tweet text Model', history_PB_TweetText)
print(plot_PB_TweetText)

#test accuracy
print("test loss, test accuracy:", PB_TweetText_model_results)
#Countries labels predicted
#print(predicted_countries_PB_TweetText)
#Classification report
print(metrics_report_PB_TweetText)

"""****************************************************************************"""
"""****************************************************************************"""

"""## **Combined Model with Fine Tuning**
-Update shape if dataset changes!
"""

#gets shape of the data for the model
input_shape_combined=train_data[0].shape
input_shape_combined

#Parameter tuning test

####NOTE########
#-Hard code the input size in shape=(UPDATE,)
#-Hard code the final dense layer based on number of countries in layers.Dense(UPDATE,....
#5381,
#Function to tune to the model
def tuner_builder(hp):
  inputs = keras.Input(shape=(3333,), name="Tuned_Combined_Inputs")
  x = layers.Dense(hp.Int('units', 50, 200, step = 20), activation="relu", name="dense_1")(inputs)
  x = layers.BatchNormalization(name="normalization_1")(x)
  x = layers.Dense(hp.Int('units1', 100, 200, step = 50), activation="relu",name="dense_2")(x)
  x = layers.Dense(hp.Int('units2', 20, 100, step = 20), activation=tf.keras.layers.LeakyReLU(alpha=0.2), name="dense_3")(x)
  x = layers.Dropout(hp.Float('dropout',0.0,0.50, step=0.10, default=0.10))(x)
  outputs = layers.Dense(11, activation="softmax",name="predictions")(x)
  model = keras.Model(inputs=inputs, outputs=outputs, name="Tuned_Combined_Model")
  #Compile  model
  model.compile(optimizer=tf.keras.optimizers.Adam(
      hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
  return model

#Parameter tuning test
#Combined model

#tuner settings 
Combined_tuner = kt.Hyperband(
    tuner_builder,
    objective='val_accuracy',
    max_epochs=50,
    factor=3,
    directory = 'Trial_run_combined2',
    project_name = 'Parameters_trials_combined2')

#Callbacks
callback1=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=4)

#clears training output
class ClearTrainingOutput(tf.keras.callbacks.Callback):
  def on_train_end(*args, **kwargs):
    IPython.display.clear_output(wait = True)

#run through tuner
Combined_tuner.search(train_data, train_labels,validation_data=(valid_data, valid_labels),
             callbacks=[callback1,ClearTrainingOutput()])

#gets best parameters
best_hyper_combined = Combined_tuner.get_best_hyperparameters(1)[0]
best_hyper_combined

print('Best Parameters for 1st Dense layer is', best_hyper_combined.get('units'))
print('Best Parameters for 2nd Dense layer is', best_hyper_combined.get('units1'))
print('Best Parameters for 3rd Dense layer is', best_hyper_combined.get('units2'))
print('Best Parameters for Dropout layer is', best_hyper_combined.get('dropout'))
print('Best learning rate for the ADAM is', best_hyper_combined.get('learning_rate'))

#applies to tuning to model
Combined_model_tuned= Combined_tuner.hypermodel.build(best_hyper_combined)

#trials using newly tuned model
#Fitting on training and validation data based on selected tuning parameters
print("Fit model on training data with tuned model")
history_combined_tuned = Combined_model_tuned.fit(train_data, train_labels, epochs=10, batch_size=15,
                   validation_data=(valid_data, valid_labels))

plot_combined_tuned=accuracy_plot('Model accuracy of Tuned Combined Model', history_combined_tuned)

# Evaluate the tuned model on the test data 
print("Evaluate tuned model on test data")
Combined_model_tuned_results = Combined_model_tuned.evaluate(test_data, test_labels, batch_size=10)

#predict using tuned model
predicted_countries_tuned, metrics_report_tuned=predict_test(Combined_model_tuned,test_data, test_labels, fit)

"""Combined Model with Fine Tuning Results"""

#train vs validation accuracy plot
print(plot_combined_tuned)
#test accuracy
print("test loss, test accuracy:", Combined_model_tuned_results)
#Countries labels predicted
#print(predicted_countries_tuned)
#Classification report
print(metrics_report_tuned)

"""## **Account Information Model with Fine Tuning**
-Update shape if dataset changes!
"""

#input shape for account features
input_shape_acc=train_data_acc[0].shape
input_shape_acc

def tuner_builder1(hp):
  inputs = keras.Input(shape=(5,), name="Tuned_Account_Info_Inputs")
  x = layers.Dense(hp.Int('units', 50, 200, step = 20), activation="relu", name="dense_1")(inputs)
  x = layers.BatchNormalization(name="normalization_1")(x)
  x = layers.Dense(hp.Int('units1', 100, 200, step = 50), activation="relu",name="dense_2")(x)
  x = layers.Dense(hp.Int('units2', 20, 100, step = 20), activation=tf.keras.layers.LeakyReLU(alpha=0.2), name="dense_3")(x)
  x = layers.Dropout(hp.Float('dropout',0.0,0.50, step=0.10, default=0.10))(x)
  outputs = layers.Dense(11, activation="softmax",name="predictions")(x)
  model = keras.Model(inputs=inputs, outputs=outputs, name="Tuned_Account_Info_Model")
  #Compile  model
  model.compile(optimizer=tf.keras.optimizers.Adam(
      hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
  return model

#tuner settings 
Acc_info_tuner = kt.Hyperband(
    tuner_builder1,
    objective='val_accuracy',
    max_epochs=50,
    factor=3,
    directory = 'Trial_run_acc_info2',
    project_name = 'Parameters_trials_acc_info2')

#run through tuner
Acc_info_tuner.search(train_data_acc, train_labels_acc,validation_data=(valid_data_acc, valid_labels_acc),callbacks=[callback1,ClearTrainingOutput()])

#gets best parameters
best_hyper_Acc_info = Acc_info_tuner.get_best_hyperparameters(1)[0]
best_hyper_Acc_info

print('Best Parameters for 1st Dense layer is', best_hyper_Acc_info.get('units'))
print('Best Parameters for 2nd Dense layer is', best_hyper_Acc_info.get('units1'))
print('Best Parameters for 3rd Dense layer is', best_hyper_Acc_info.get('units2'))
print('Best Parameters for Dropout layer is', best_hyper_Acc_info.get('dropout'))
print('Best learning rate for the ADAM is', best_hyper_Acc_info.get('learning_rate'))

#applies to tuning to model
Acc_info_model_tuned= Acc_info_tuner.hypermodel.build(best_hyper_Acc_info)

#trials using newly tuned model
#Fitting on training and validation data based on selected tuning parameters
print("Fit model on training data with tuned Account Information model")
history_acc_info_tuned = Acc_info_model_tuned.fit(train_data_acc, train_labels_acc, epochs=10, batch_size=15,
                   validation_data=(valid_data_acc, valid_labels_acc))

plot_acc_info_tuned=accuracy_plot('Model accuracy of Tuned Account Information Model', history_acc_info_tuned)

# Evaluate the tuned model on the test data 
print("Evaluate tuned Account Information model on test data")
acc_info_tuned_results = Acc_info_model_tuned.evaluate(test_data_acc, test_labels_acc, batch_size=10)

#predict using tuned model
predicted_countries_acc_info_tuned, metrics_report_acc_info_tuned=predict_test(Acc_info_model_tuned,test_data_acc, test_labels_acc, fit)

"""Account Information Model with Fine Tuning Results"""

#train vs validation accuracy plot
print(plot_acc_info_tuned)
#test accuracy
print("test loss, test accuracy:", acc_info_tuned_results)
#Countries labels predicted
#print(predicted_countries_acc_info_tuned)
#Classification report
print(metrics_report_acc_info_tuned)



"""## **Profile and Banner Model with Fine Tuning**
-Update shape if dataset changes!
"""

#input shape for image embeddings
input_shape_pb=train_data_pb[0].shape
input_shape_pb

def tuner_builder(hp):
  inputs = keras.Input(shape=(2560,), name="Tuned_Profile_Banner_Inputs")
  x = layers.Dense(hp.Int('units', 50, 200, step = 20), activation="relu", name="dense_1")(inputs)
  x = layers.BatchNormalization(name="normalization_1")(x)
  x = layers.Dense(hp.Int('units1', 100, 200, step = 50), activation="relu",name="dense_2")(x)
  x = layers.Dense(hp.Int('units2', 20, 100, step = 20), activation=tf.keras.layers.LeakyReLU(alpha=0.2), name="dense_3")(x)
  x = layers.Dropout(hp.Float('dropout',0.0,0.50, step=0.10, default=0.10))(x)
  outputs = layers.Dense(11, activation="softmax",name="predictions")(x)
  model = keras.Model(inputs=inputs, outputs=outputs, name="Tuned_Profile_Banner_Model")
  #Compile  model
  model.compile(optimizer=tf.keras.optimizers.Adam(
      hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
  return model

#tuner settings 
PB_tuner = kt.Hyperband(
    tuner_builder,
    objective='val_accuracy',
    max_epochs=50,
    factor=3,
    directory = 'Trial_run_PB2',
    project_name = 'Parameters_trials_PB2')

#run through tuner
PB_tuner.search(train_data_pb, train_labels_pb,validation_data=(valid_data_pb, valid_labels_pb),
             callbacks=[callback1,ClearTrainingOutput()])

#gets best parameters
best_hyper_PB = PB_tuner.get_best_hyperparameters(1)[0]
best_hyper_PB

print('Best Parameters for 1st Dense layer is', best_hyper_PB.get('units'))
print('Best Parameters for 2nd Dense layer is', best_hyper_PB.get('units1'))
print('Best Parameters for 3rd Dense layer is', best_hyper_PB.get('units2'))
print('Best Parameters for Dropout layer is', best_hyper_PB.get('dropout'))
print('Best learning rate for the ADAM is', best_hyper_PB.get('learning_rate'))

#applies to tuning to model
PB_model_tuned= PB_tuner.hypermodel.build(best_hyper_PB)

#trials using newly tuned model
#Fitting on training and validation data based on selected tuning parameters
print("Fit model on training data with tuned Profile and Banner Images model")
history_PB_tuned = PB_model_tuned.fit(train_data_pb, train_labels_pb, epochs=10, batch_size=15,
                   validation_data=(valid_data_pb, valid_labels_pb))

plot_PB_tuned=accuracy_plot('Model accuracy of Tuned Profile and Banner Images Model', history_PB_tuned)

# Evaluate the tuned model on the test data 
print("Evaluate tuned Profile and Banner Images on test data")
PB_model_tuned_results = PB_model_tuned.evaluate(test_data_pb, test_labels_pb, batch_size=10)

#predict using tuned model
predicted_countries_PB_tuned, metrics_report_PB_tuned=predict_test(PB_model_tuned,test_data_pb, test_labels_pb, fit)

"""Profile and Banner Model with Fine Tuning Results"""

#train vs validation accuracy plot
print(plot_PB_tuned)
#test accuracy
print("test loss, test accuracy:", PB_model_tuned_results)
#Countries labels predicted
#print(predicted_countries_PB_tuned)
#Classification report
print(metrics_report_PB_tuned)



"""## **Tweet Text Model with Fine Tuning**
-Update shape if dataset changes!
"""

input_shape_tweettext=train_data_tweettext[0].shape
input_shape_tweettext

def tuner_builder(hp):
  inputs = keras.Input(shape=(768,), name="Tuned_Tweet_Text_Inputs")
  x = layers.Dense(hp.Int('units', 50, 200, step = 20), activation="relu", name="dense_1")(inputs)
  x = layers.BatchNormalization(name="normalization_1")(x)
  x = layers.Dense(hp.Int('units1', 100, 200, step = 50), activation="relu",name="dense_2")(x)
  x = layers.Dense(hp.Int('units2', 20, 100, step = 20), activation=tf.keras.layers.LeakyReLU(alpha=0.2), name="dense_3")(x)
  x = layers.Dropout(hp.Float('dropout',0.0,0.50, step=0.10, default=0.10))(x)
  outputs = layers.Dense(11, activation="softmax",name="predictions")(x)
  model = keras.Model(inputs=inputs, outputs=outputs, name="Tuned_Tweet_Text_Model")
  #Compile  model
  model.compile(optimizer=tf.keras.optimizers.Adam(
      hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
  return model

#tuner settings 
TweetText_tuner = kt.Hyperband(
    tuner_builder,
    objective='val_accuracy',
    max_epochs=50,
    factor=3,
    directory = 'Trial_run_tweettext',
    project_name = 'Parameters_trials_tweettext')

#run through tuner
TweetText_tuner.search(train_data_tweettext, train_labels_tweettext,validation_data=(valid_data_tweettext, valid_labels_tweettext),
             callbacks=[callback1,ClearTrainingOutput()])

#gets best parameters
best_hyper_TweetText = TweetText_tuner.get_best_hyperparameters(1)[0]
best_hyper_TweetText

print('Best Parameters for 1st Dense layer is', best_hyper_TweetText.get('units'))
print('Best Parameters for 2nd Dense layer is', best_hyper_TweetText.get('units1'))
print('Best Parameters for 3rd Dense layer is', best_hyper_TweetText.get('units2'))
print('Best Parameters for Dropout layer is', best_hyper_TweetText.get('dropout'))
print('Best learning rate for the ADAM is', best_hyper_TweetText.get('learning_rate'))

#applies to tuning to model
TweetText_model_tuned= TweetText_tuner.hypermodel.build(best_hyper_TweetText)

#Fitting on training and validation data
print("Fit model on Tuned Tweet Text training data")
history_tweettext_tuned = TweetText_model_tuned.fit(train_data_tweettext, train_labels_tweettext, epochs=10, batch_size=15,
                   validation_data=(valid_data_tweettext, valid_labels_tweettext))

plot_tweettext_tuned=accuracy_plot('Model accuracy of Tuned Tweet Text Model', history_tweettext_tuned)

# Evaluate the model on the test data
print("Evaluate on Tuned Tweet Text test data")
Tweettext_model_tuned_results = TweetText_model_tuned.evaluate(test_data_tweettext,test_labels_tweettext, batch_size=10)

#test predict_test
predicted_countries_tweettext_tuned, metrics_report_tweettext_tuned=predict_test(TweetText_model_tuned,test_data_tweettext,test_labels_tweettext, fit)

""" Tweet Text Model with Fine Tuning Results"""

#train vs validation accuracy plot
print(plot_tweettext_tuned)
#test accuracy
print("test loss, test accuracy:", Tweettext_model_tuned_results)
#Countries labels predicted
#print(predicted_countries_tweettext_tuned)
#Classification report
print(metrics_report_tweettext_tuned)



"""## **Profile + Banner + Tweet Text Model with Fine Tuning**

-Update date shape if dataset changes!
"""

#get shape
input_shape_PB_TweetText=train_data_PB_TweetText[0].shape
input_shape_PB_TweetText

#Profile + Banner + TweetText model
def tuner_builder(hp):
  inputs = keras.Input(shape=(3328,), name="Tuned_PB_TweetText_Inputs")
  x = layers.Dense(hp.Int('units', 50, 200, step = 20), activation="relu", name="dense_1")(inputs)
  x = layers.BatchNormalization(name="normalization_1")(x)
  x = layers.Dense(hp.Int('units1', 100, 200, step = 50), activation="relu",name="dense_2")(x)
  x = layers.Dense(hp.Int('units2', 20, 100, step = 20), activation=tf.keras.layers.LeakyReLU(alpha=0.2), name="dense_3")(x)
  x = layers.Dropout(hp.Float('dropout',0.0,0.50, step=0.10, default=0.10))(x)
  outputs = layers.Dense(11, activation="softmax",name="predictions")(x)
  model = keras.Model(inputs=inputs, outputs=outputs, name="Tuned_PB_TweetText_Model")
  #Compile  model
  model.compile(optimizer=tf.keras.optimizers.Adam(
      hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
  return model

#tuner settings 
Pb_TweetText_tuner = kt.Hyperband(
    tuner_builder,
    objective='val_accuracy',
    max_epochs=50,
    factor=3,
    directory = 'Trial_run_Pb_TweetText2',
    project_name = 'Parameters_trials_Pb_TweetText2')

#run through tuner
Pb_TweetText_tuner.search(train_data_PB_TweetText, train_labels_PB_TweetText,validation_data=(valid_data_PB_TweetText, valid_labels_PB_TweetText),
             callbacks=[callback1,ClearTrainingOutput()])

#gets best parameters
best_hyper_Pb_TweetText = Pb_TweetText_tuner.get_best_hyperparameters(1)[0]
best_hyper_Pb_TweetText

print('Best Parameters for 1st Dense layer is', best_hyper_Pb_TweetText.get('units'))
print('Best Parameters for 2nd Dense layer is', best_hyper_Pb_TweetText.get('units1'))
print('Best Parameters for 3rd Dense layer is', best_hyper_Pb_TweetText.get('units2'))
print('Best Parameters for Dropout layer is', best_hyper_Pb_TweetText.get('dropout'))
print('Best learning rate for the ADAM is', best_hyper_Pb_TweetText.get('learning_rate'))

#applies to tuning to model
Pb_TweetText_model_tuned= Pb_TweetText_tuner.hypermodel.build(best_hyper_Pb_TweetText)

#Fitting on training and validation data
print("Fit model on Images and Tweet text training data")
history_PB_TweetText= Pb_TweetText_model_tuned.fit(train_data_PB_TweetText, train_labels_PB_TweetText, epochs=10, batch_size=15,
                   validation_data=(valid_data_PB_TweetText, valid_labels_PB_TweetText))

plot_PB_TweetText_tuned=accuracy_plot('Model accuracy of Tuned Images and Tweet text Model', history_PB_TweetText)

#Evaluate the model on the test data
print("Evaluate on Images and Tweet text test data")
PB_TweetText_model_tuned_results = Pb_TweetText_model_tuned.evaluate(test_data_PB_TweetText, test_labels_PB_TweetText, batch_size=10)

#test predict_test
predicted_countries_PB_TweetText_tuned, metrics_report_PB_TweetText_tuned=predict_test(Pb_TweetText_model_tuned,test_data_PB_TweetText, test_labels_PB_TweetText, fit)

"""Profile + Banner + Tweet Text Model with Fine Tuning Results"""

#train vs validation accuracy plot
print(plot_PB_TweetText_tuned)
#test accuracy
print("test loss, test accuracy:", PB_TweetText_model_tuned_results)
#Countries labels predicted
#print(predicted_countries_PB_TweetText_tuned)
#Classification report
print(metrics_report_PB_TweetText_tuned)

"""Addition for Web App"""

#Reformatting test prediction
score= Combined_model_tuned.evaluate(test_data, test_labels, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

##saving model
combined_model_json= Combined_model_tuned.to_json()
with open ("combined_model2.json", "w") as json_file:
  json_file.write(combined_model_json)



"""**New Image Embeddings model**"""

#read in pickle to dataframe
new_df_x = pd.read_pickle('/content/drive/My Drive/dataset/recent_image_df.p')

#rename location to country_code
new_df_x=new_df_x.rename(columns= {'location': 'country_code'})
new_df_x['country_code'].value_counts()

#balanced dataset
new_df_x=new_df_x.groupby('country_code').head(684)
new_df_x['country_code'].value_counts()

#get label dataframe
new_df_y = new_df_x.loc[:, ['screen_name', 'country_code']]

#datasets without duplicates
labels_new,data_new =added_preprocessing(new_df_x,new_df_y)

data_new.shape

#drop screen name and country code
data_new=data_new.filter(regex='^(?!screen_name|country_code)')
data_new=np.array(data_new)

#Test label_encoding
labels_new, fit_new=label_encoding(labels_new)

#Train/test set split (80/20)
train_data_new,test_data_new,train_labels_new,test_labels_new=train_test_split(data_new,labels_new,test_size=0.2, random_state=123)

#Train/validation set split (60/20)
train_data_new,valid_data_new,train_labels_new,valid_labels_new=train_test_split(train_data_new,train_labels_new,test_size=0.2, random_state=123)

train_data_new[0].shape

#set seed
random_seeds(1234)

def tuner_builder_new(hp):
  inputs = keras.Input(shape=(1280,), name="New_Images_Inputs")
  x = layers.Dense(hp.Int('units', 50, 200, step = 20), activation="relu", name="dense_1")(inputs)
  x = layers.BatchNormalization(name="normalization_1")(x)
  x = layers.Dense(hp.Int('units1', 100, 200, step = 50), activation="relu",name="dense_2")(x)
  x = layers.Dense(hp.Int('units2', 20, 100, step = 20), activation=tf.keras.layers.LeakyReLU(alpha=0.2), name="dense_3")(x)
  x = layers.Dropout(hp.Float('dropout',0.0,0.50, step=0.10, default=0.10))(x)
  outputs = layers.Dense(11, activation="softmax",name="predictions")(x)
  model = keras.Model(inputs=inputs, outputs=outputs, name="New_Images_Inputs")
  #Compile  model
  model.compile(optimizer=tf.keras.optimizers.Adam(
      hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
  return model

#tuner settings 
New_images_tuner = kt.Hyperband(
    tuner_builder_new,
    objective='val_accuracy',
    max_epochs=50,
    factor=3,
    directory = 'Trial_run_new_images',
    project_name = 'Parameters_trials_new_images')

#run through tuner
New_images_tuner.search(train_data_new, train_labels_new,validation_data=(valid_data_new, valid_labels_new),
             callbacks=[callback1,ClearTrainingOutput()])

#gets best parameters
best_hyper_new_images = New_images_tuner.get_best_hyperparameters(1)[0]
best_hyper_new_images

print('Best Parameters for 1st Dense layer is', best_hyper_new_images.get('units'))
print('Best Parameters for 2nd Dense layer is', best_hyper_new_images.get('units1'))
print('Best Parameters for 3rd Dense layer is', best_hyper_new_images.get('units2'))
print('Best Parameters for Dropout layer is', best_hyper_new_images.get('dropout'))
print('Best learning rate for the ADAM is', best_hyper_new_images.get('learning_rate'))

#applies to tuning to model
New_images_model_tuned= New_images_tuner.hypermodel.build(best_hyper_new_images)
#trials using newly tuned model
#Fitting on training and validation data based on selected tuning parameters
print("Fit model on training data with tuned model")
history_new_images_tuned = New_images_model_tuned.fit(train_data_new, train_labels_new, epochs=10, batch_size=15,
                   validation_data=(valid_data_new, valid_labels_new))

#plot of training vs validation accuracy
plot_new_images_tuned=accuracy_plot('Model accuracy of Tuned New Images Model', history_new_images_tuned)

# Evaluate the tuned model on the test data 
print("Evaluate tuned model on test data")
New_images_model_results = New_images_model_tuned.evaluate(test_data_new, test_labels_new, batch_size=10)

#predict using tuned model
predicted_countries_new_images, metrics_report_new_images=predict_test(New_images_model_tuned,test_data_new, test_labels_new, fit_new)

#train vs validation accuracy plot
print(plot_new_images_tuned)
#test accuracy
print("test loss, test accuracy:", New_images_model_results)
#Countries labels predicted
#print(predicted_countries_new_images)
#Classification report
print(metrics_report_new_images)





"""****************************************************************************"""