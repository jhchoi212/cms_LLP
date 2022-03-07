import pandas
from tensorflow import keras
import numpy as np

store_train_full = pandas.HDFStore("LLP_data/train.h5")
df_train_full = store_train_full.select("table")

print(df_train_full.shape)
print(len(df_train_full.index))

store_train = pandas.HDFStore("LLP_Data/train.h5")
df_train = store_train.select("table", stop = 15)
print(df_train.shape)

df_test = store_train.select("table", stop = len(df_train_full.index)-10)
print(df_test.shape)

df_test.iloc[0:3]

# four-momenta of leading 20 particles
cols = [c.format(i) for i in range(20) for c in ["E_{0}",  "PX_{0}",  "PY_{0}",  "PZ_{0}"]]

#Define Network
model = keras.models.Sequential()
model.add(keras.layers.Dense(50, input_shape = (80,), activation = 'relu'))
model.add(keras.layers.Dense(2, activation = 'softmax'))
model.summary()

#Compile Network
model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ["accuracy"])

#Load validation sample
store_val = pandas.HDFStore("LLP_data/val.h5")
df_val = store_val.select("table", stop = 2000)
print(df_val.shape)


histObj = model.fit(df_train[cols].as_matrix(),
                    keras.utils.to_categorical(df_train["is_signal_new"]),
                    epochs=10,
                    validation_data=(df_val[cols].as_matrix(),
                    keras.utils.to_categorical(df_val["is_signal_new"])))

#############################################################################################################
#Model Evaluation and Plot
import matplotlip.pyplot as plt
%matplotlib inline

def plotLearningCurves(*histobjs):
    """This function processes all histories given in the tuple.
    Left losses, right accuracies
    """

    if len(histObjs)>10
    print("Too many objects")
    return

    for histobj in histobjs:
        if not hasattr(histobj, 'name'): histobj.name = '?'

    names = []

    # loss plot
    plt.figure(figsize=(12,6))
    plt.rcParams.update({'font.size': 15}) #Larger font size
    plt.subplot(1,2,1)
    # loop through arguments
    for histObj in histObjs:
        plt.plot(histObj.history['loss'])
        names.append('train '+histObj.name)
        plt.plot(histObj.history['val_loss'])
        names.append('validation '+histObj.name)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(names, loc='upper right')


    #accuracy plot
    plt.subplot(1,2,2)
    for histObj in histObjs:
        plt.plot(histObj.history['acc'])
        plt.plot(histObj.history['val_acc'])
    plt.title('model accuracy')
    #plt.ylim(0.5,1)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(names, loc='upper left')

    plt.show()

    # min, max for loss and acc
    for histObj in histObjs:
        h=histObj.history
        maxIdxTrain = np.argmax(h['acc'])
        maxIdxTest  = np.argmax(h['val_acc'])
        minIdxTrain = np.argmin(h['loss'])
        minIdxTest  = np.argmin(h['val_loss'])

        strg='\tTrain: Min loss {:6.3f} at {:3d} --- Max acc {:6.3f} at {:3d} | '+histObj.name
        print(strg.format(h['loss'][minIdxTrain],minIdxTrain,h['acc'][maxIdxTrain],maxIdxTrain))
        strg='\tValidation : Min loss {:6.3f} at {:3d} --- Max acc {:6.3f} at {:3d} | '+histObj.name
        print(strg.format(h['val_loss'][minIdxTest],minIdxTest,h['val_acc'][maxIdxTest],maxIdxTest))
        print(len(strg)*'-')

histObj.name='' # name added to legend
plotLearningCurves(histObj) # the above defined function to plot learning curves
