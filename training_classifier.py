#Building a model that distinguishes which photos are sideways and which are upright, 
#so an app could automatically rotate each image if necessary.

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from os.path import join
from IPython.display import display
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

#View Sample Image

sideways_image_dir = '/content/drive/My Drive/Transfer_Learning/dogsvscats/train/sideways/'
upright_image_dir = '/content/drive/My Drive/Transfer_Learning/dogsvscats/train/upright/'

sideways_image_path = [join(sideways_image_dir,filename) for filename in 
                            ['dog.11103.jpg','dog.8044.jpg','dog.8362.jpg','dog.8703.jpg','dog.8990.jpg']]

upright_image_path = [join(upright_image_dir,filename) for filename in 
                            ['dog.11102.jpg','dog.8043.jpg','dog.8344.jpg','dog.8666.jpg','dog.8970.jpg']]

img_paths =  sideways_image_path + upright_image_path

for img_path in img_paths:
    if 'upright' in img_path:
      print("Sample Upright Image")
      display(Image.open(img_path))
    else:
      print("Sample Sideways Image")
      display(Image.open(img_path))


#Modeling to detect whether the Dog Image is a Upright Image or Sideways Image with Transfer Learning..!

num_classes = 2


model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
model.add(Dense(num_classes, activation='softmax'))

model.layers[0].trainable = False

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

image_size = 224
data_generator = ImageDataGenerator(preprocess_input,width_shift_range=0.2,height_shift_range=0.2)
train_generator = data_generator.flow_from_directory(
                                        directory='/content/drive/My Drive/Transfer_Learning/dogsvscats/train',
                                        target_size=(image_size, image_size),
                                        batch_size=10,
                                        class_mode='categorical')
validation_generator = data_generator.flow_from_directory(
                                        directory='/content/drive/My Drive/Transfer_Learning/dogsvscats/val',
                                        target_size=(image_size, image_size),
                                        class_mode='categorical')
fit_stats = model.fit_generator(train_generator,
                                       steps_per_epoch=22,
                                       validation_data=validation_generator,
                                       validation_steps=1)

print(fit_stats.history)
model.save('/content/drive/My Drive/Transfer_Learning/dogsvscats/model_dogsvscats')


#Predictions of the Model

sideways_image_path = [join(sideways_image_dir,filename) for filename in 
                            ['dog.11103.jpg','dog.8044.jpg','dog.8362.jpg','dog.8703.jpg','dog.8990.jpg']]

upright_image_path = [join(upright_image_dir,filename) for filename in 
                            ['dog.11102.jpg','dog.8043.jpg','dog.8344.jpg','dog.8666.jpg','dog.8970.jpg']]

img_paths =  sideways_image_path + upright_image_path

for img_path in img_paths:
    if 'upright' in img_path:
      print(img_path," : Sample Upright Image")
    else:
      print(img_path," : Sample Sideways Image")

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)

test_data = read_and_prep_images(img_paths)

preds = model.predict(test_data)

for img_path,pred in zip(img_paths,np.argmax(preds,axis=1)):
  if pred == 1:
    print(img_path,": Predicted as Upright")
  else:
    print(img_path,": Predicted as Sideways")

