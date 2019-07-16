# Dermatologist
Implementation of "Dermatologist-level classification of skin cancer with deep neural networks" in pytorch. 

### About
###### Overview
As defined in the paper, we take advantage of transfer learning using a pretrained (On ImageNet) CNN, specifically Google's **Inception V3**. The classifier predicts images of three classes, namely, melanoma, nevus, and seborrheic_keratosis. There are 754 more classes that the original researchers have defined, but the data provided only permitted three classes.

### Usage

###### Running the program
1. Clone the repository
  ```
  git clone https://github.com/Shubham-SK/dermatologist.git
  cd dermatologist
  ```
2. Download data from:

<ul>
<li><a>https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip</a>,</li>
<li><a>https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip</a>,</li>
<li><a>https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip</a>.</li>
  </ul>
  
  You may either manually download the zip files or do it with the following commands:
  
  ```
  wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip
  wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip
  wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip
  unzip *.zip
  ```
  
3. Store them in a directory called "data". 
  
  ```
  mkdir data 
  mv train.zip valid.zip test.zip data
  ```
  
4. Navigate to main.py and run the script, model training should begin. 
  ```
  cd ../python
  python3 main.py
  ```

Note: I am currently working on developing a web/smartphone application to make the project more user friendly. Also, I am training the model and will upload a "Classifier-net.pt" containing the model state dictionary so training isn't necessary for testing.
