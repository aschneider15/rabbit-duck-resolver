# Rabbit-Duck Resolver

An image classifier which can be used to determine whether an image is more duck-like or rabbit like. Created by Aidan C. Schneider, November 2023.

## Project outline

In the world of psychology, there exists an optical illusion called the ["Rabbit-Duck Illusion,"](https://en.wikipedia.org/wiki/Rabbit%E2%80%93duck_illusion) which presents the viewer with an ambiguous image that could be identified as being a duck or a rabbit. I have always found the idea to be fascinating, and so I made it my goal to create a binary image classifier which, using a convolutional neural network, could interpret whether an ambiguous image of this nature was _definitively_ more duck-like or more rabbit-like. This repository contains the results of this goal.

## How it works

Included in the repo are two zip files: one folder containing the training, testing, and validation data in order to configure the model, and a second folder called "unidentified" which contains various 32x32 color pictures of ducks and rabbits for the model to distinguish between. As it currently stands, I have yet to include any of the ambiguous duck-rabbit pictures in the unidentified folder, but I expect that will come in a future release.

## Shortcomings and future plans

As it currently stands, the models struggles to classify images in pretty much every way. There are three theories I have regarding this:

1. The model is overfitting to the training data.
2. There is not enough training data _or_ the training data is too visually diverse to create an accurate model.
3. The images are either too small for the convnet to extract meaningful details or their being in color is throwing off the model.

Either way, I will investigate all three possibilities, while also taking into account that the cause may be something else completely. This will most probably be done using seperate branches of the repo.

## How to run this project on Google Colab

1. Download the "dataset.zip" and "unidentified.zip" files.
2. Go to my Colab project linked [here](https://colab.research.google.com/drive/1OKIf4Cj2JGaoH0jbG-Gj1MpYmMzCgKum?usp=sharing).
3. Click the "Files" tab and upload both of the zip files that were previously downloaded.
4. If necessary, add a code block to the top of the project and run a code block to install the following packages with pip:

   - zipfile
   - os
   - tensorflow
   - numpy
   - PIL

5. Run each code block of the project in sequence
6. (Optional) Create your own "unidentified" zip folder to have the model classify images that you provide.

## How to install and run this project on your own device

1. Download the code.
2. Install the following python packages using pip:.

   - zipfile
   - os
   - tensorflow
   - numpy
   - PIL

3. Use Python to run the "resolver.py" file.
4. (Optional) Create your own "unidentified" zip folder to have the model classify images that you provide.
