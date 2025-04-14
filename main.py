import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)
def create_model():
    #Create the model
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))#input layer
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))#output layer

    #Compile the model
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(x_train,y_train,epochs=50)#we train the model with 50 epochs
    model.save('handwritten.keras')
def view_accuracy():
    model=tf.keras.models.load_model('handwritten.keras')
    loss,accuracy=model.evaluate(x_test,y_test)
    print(loss)
    print(accuracy)

model=tf.keras.models.load_model('handwritten.keras')
image_number=0
while os.path.isfile(f"Digits/digit{image_number}.png"):
    try:
        img=cv2.imread(f"Digits/digit{image_number}.png")[:,:,0]#we only take the first channel
        img= np.invert(np.array([img]))
        prediction=model.predict(img)
        print(f"This digit is {np.argmax(prediction)}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number+=1
