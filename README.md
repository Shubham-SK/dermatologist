# Dermatologist
Implementation of "Dermatologist-level classification of skin cancer with deep neural networks" in pytorch. 

### About
###### Overview
As defined in the paper, we take advantage of transfer learning using a pretrained (On ImageNet) CNN, specifically Google's **Inception V3**. The classifier predicts images of three classes, namely, melanoma, nevus, and seborrheic_keratosis. There are 754 more classes that the original researchers have defined, but the data provided only permitted three classes.

### Usage

1. Download data from:

<ul>
<li><a>https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip</a>,</li>
<li><a>https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip</a>,</li>
<li><a>https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip</a>.</li>
  </ul>
  
2. Store them in a directory called "data". 
3. Navigate to main.py and run the script, model training should begin. 

Note: I am currently working on developing a web/smartphone application to make the project more user friendly. Also, I am training the model and will upload a "Classifier-net.pt" containing the model state dictionary so training isn't necessary for testing.
