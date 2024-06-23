# Neural Style Transfer using CNNs

This repository contains a Streamlit web application that performs neural style transfer using a pre-trained model from TensorFlow Hub.

## What is Style Transfer in CNN?

Neural style transfer is a technique that uses convolutional neural networks (CNNs) to blend the content of one image with the artistic style of another image. The process involves extracting the content features from a content image and the style features from a style image, and then combining these features to generate a new, stylized image. This technique has gained popularity for its ability to create visually appealing and unique images by applying the artistic style of famous paintings to any photograph.

## Model Used

The model used for this project is the Arbitrary Image Stylization model from TensorFlow Hub, which can be found at [this link](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2). This model allows for the combination of content and style images to produce a stylized output image.


## Streamlit Web Application

The Streamlit web application allows users to upload their own content and style images, and then performs style transfer using the loaded model. The application displays the input images and the resulting stylized image.

## Screenshots of application in use

![Presentation1](https://github.com/arunjames003/Neural_Style_Transfer_using_CNNs/assets/155214383/09c17081-22c8-4d13-8982-e594d64a13ae)


## Built With
- Streamlit - The web framework used
- TensorFlow Hub - The repository of trained machine learning models

## Contact
Created by [@arunjames003] - feel free to contact me!

## Acknowledgments
- The TensorFlow team for providing the pre-trained model
- Streamlit for making it easy to create web applications

