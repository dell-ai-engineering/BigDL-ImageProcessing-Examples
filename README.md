### Analytics zoo testing with multiple labels  Chest X-ray image 
The code utilized transfer learning with Inceotion or Resnt50 to test  analytics zoo with Chest X-ray images form 
https://www.kaggle.com/nih-chest-xrays/data
The dataset has 112000 Chest X-ray gray scale images from more than 30000 unique patients. Each x-ray may contains one or more diseases (labels) which make the problem as multiple classes and  labels problem.   The images are gray scale with 2 channels. In order to fit with Resnet 50  all these images must be in 3 channels and fed as BGR to the Resnet 50. The dataset is divided into train and test folder and saved in HDFS and their labels are save in CSV file which is called Data_Entry_2017.csv. This CSV files contains the labels ( image index and finding ) and randomly stored. Thus, in order to get the right label for right image two sql dataframes must be created and inner joined based on the image index columns in both dataframe. 
After all images and their labels formulated in one dataframe , preprocessing phase is done 
### Preprocessing 
In this phase the dataframe is splited  into training  and validation dataframes  then  all images are resized to 224, 224 , 3 and randomly horizontally  flipped, image channels are normalized by subtracting ImageNet means from each image channels,  and images are converted into tensor   where this done in transformer
### Model creation 
Resnet50 pretrained model is loaded and the last layer is chopped then adding an new flatten layer with 15 classes. As this problem is multiple classes and multiple labels the last activation function which is used is sigmoid function. 
### Model training and validation 
The model is trained with train dataframe and validate with validation data farme .
### Area Under Curve calculation 
The AUC is calculated for all 15 classes 
### Micro and Macro average 
The last step is calculating and plotting AUC for all 15 classes. 


