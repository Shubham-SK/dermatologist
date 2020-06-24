# Dermatologist
Implementation of [dermatologist-level classification of skin cancer with deep neural networks](https://www.nature.com/articles/nature21056) in pytorch. Refer to the aforementioned paper for program logic and motivation behind each of the concepts.

___
### About
As defined in the paper, we take advantage of transfer learning using a pretrained (On ImageNet) CNN, specifically Google's **Inception V3**. The classifier predicts images of three classes, namely, melanoma, nevus, and seborrheic_keratosis. There are 754 more classes that the original researchers have defined, but the data provided only permitted three classes.

___
### Usage
1. Clone the repository
  ```
  git clone https://github.com/Shubham-SK/dermatologist.git
  cd dermatologist
  ```
2. Download data from:

- https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip
- https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip
- https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip
  
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
  
4. Install necessary dependencies.

   Project hinges on the use of the anaconda package manager, this would be the preferred method of package installation. However, you may use pip to install the packages listed in both requirements.txt and below.
 
<ul>
   <li> pytorch </li>
   <li> torchvision </li>
   <li> numpy </li>
   <li> PIL/Pillow </li>
</ul>
  
   Use the following commands to setup the virtual environment with conda and install packages.
   
   ```
   conda create -n Dermatologist anaconda
   conda activate Dermatologist
   conda install -c pytorch pytorch torchvision
   ```

5. Navigate to main.py and run the script, model training should begin. 
  ```
  cd ../python
  python3 main.py
  ```

Note: I am currently working on developing a web/smartphone application to make the project more user friendly. Also, I am training the model and will upload a "Classifier-net.pt" containing the model state dictionary so training isn't necessary for testing.

Update 1: Classifier-net.pt has been uploaded, work is being done to test performance.
Update 2: Performance results show that the model is able to classify the skin cancer with 68% accuracy. This is certainly better than guessing, but is far from "state-of-the-art" level of classification. I am investigating the model with more depth and tweaking hyperparameters to optimize accuracy.
