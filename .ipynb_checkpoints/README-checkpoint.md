### GA_Data_Science_Capstone_Project
# **Interactive ABC's with American Sign Language**
### A step in Increasing Accessability for the Deaf Community with Computer Vision utilizing Yolov5.
![ASL_Demo](assets/alphabet.gif)

<a name="executivesummary"></a>
# **Executive Summary**
Utilizing Yolov5, a custom computer vision model was created on the American Sign Language alphabet.  The project was promoted on social platforms to diversify the dataset. A total of 721 images were collected in the span of two weeks using DropBox request forms.  Manual labels were created of the original images which were then resized, and organized for preprocessing. Several carefully selected augmentations were made to the images to compensate for the small dataset count.  A total of 18,000 images were then used for modeling.  Transfer learning was incorporated with Yolov5m weights and training completed on 300 epochs with an image size of 1024 in 163 hours. A mean average precision score of 0.8527 was achieved.  Inference tests were successfully performed with areas identifying the models strengths and weaknesses for future development.  

All operations were performed on my local Linux machine with a CUDA/cudNN setup using Pytorch.

<a name="contents"></a>
# **Table of Contents**

- [Executive Summary](#executivesummary)
- [Table of Contents](#contents)
- [Data Colelction Method](#data)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Inference](#inference)
- [Conclusions](#conclusions)
- [Next Steps](#nextsteps)
- [Citations](#cite)
- [Special Thanks](#thanks)



<a name="executivesummary"></a>
- [Back to Contents](#contents)
# **Problem Statement:**
Have you ever considered how easy it is to perform simple communication tasks such as ordering food at a drive thru, discussing financial information with a banker, telling a physician your symptoms at a hospital, or even negotiating your wages from your employer?  What if there was a rule where you couldn’t speak and were only able to use your hands for each of these circumstances? The deaf community cannot do what most of the population take for granted and are often placed in degrading situations due to these challenges they face every day. Access to qualified interpretation services isn’t feasible in most cases leaving many in the deaf community with underemployment, social isolation, and public health challenges. To give these members of our community a greater voice, I have attempted to answer this question:


**Can computer vision bridge the gap for the deaf and hard of hearing by learning American Sign Language?**

In order to do this, a Yolov5 model was trained on the ASL alphabet.  If successful, it may mark a step in the right direction for both greater accessibility and educational resources.

<a name="data"></a>
- [Back to Contents](#contents)
# **Data Collection Method:**
The decision was made to create an original dataset for a few reasons.  The first was to mirror the intended environment on a mobile device or webcam.  These often have resolutions of 720 or 1080p. Several existing datasets have a low resolution and many do not include the letters “j” and “z” as they require movements.

A letter request form was created with an introduction to my project along with instruction on how to submit voluntary sign language images with dropbox file request forms.  This was distributed on social platforms to bring awareness, and to collect data.

    
#### Dropbox request form used: (Deadline Sep. 27th, 2020)
https://docs.google.com/document/d/1ChZPPr1dsHtgNqQ55a0FMngJj8PJbGgArm8xsiNYlRQ/edit?usp=sharing
[link](https://docs.google.com/document/d/1ChZPPr1dsHtgNqQ55a0FMngJj8PJbGgArm8xsiNYlRQ/edit?usp=sharing)
    
A total of 720 images were collected:

Here is the distributions of images: (Letters / Counts)

A - 29  
B - 25  
C - 25  
D - 28  
E - 25  
F - 30  
G - 30  
H - 29  
I - 30  
J - 38  
K - 27  
L - 28  
M - 28  
N - 27  
O - 28  
P - 25  
Q - 26  
R - 25  
S - 30  
T - 25  
U - 25  
V - 28  
W - 27  
X - 26  
Y - 26  
Z - 30  

<a name="preprocessing"></a>
- [Back to Contents](#contents)
# **Preproccessing**
### Labeling the images
Manual bounding box labels were created on the original images using the labelImg software.

Each of the pictures and bounding box coordinates were then passed through an albumentations pipeline that resized the images to 1024 x 1024 pixel squares and added probabilities of different transformations.

These transformations included specified degrees of rotations, shifts in the image locations, blurs, horizontal flips, random erase, and a variety of other color transformations.  
  
![](assets/augmentations_slide.png)


25 augmented images were created for each image resulting in an image set of 18,000 used for modeling.

<a name="modeling"></a>
- [Back to Contents](#contents)
# **Modeling: Yolov5**
To address acceptable inference speeds and size, Yolov5 was chosen for modeling. 

This was released in June 10th of this year, and is still in active development.  Although Yolov5 by Ultralytics is not created by the original Yolo authors, Yolo v5 is said to be faster and more lightweight, with accuracy on par with Yolo v4 which is widely considered as the fastest and most accurate real-time object detection model.

![](assets/Yolov5_explanation.png)

Yolo was designed as a convolutional neural network for real time object detection.  Its more complex than basic classification as object detection needs to identify the objects and locate where it is on the image. This single stage object detector, has 3 main components:

The backbone basically extracts important features of an image,  the neck mainly uses feature pyramids which help in generalizing the object scaling for better performance on unseen data.  The model head does the actual detection part where anchor boxes are applied on features that generate output vectors.
These vectors include the class probabilities, the objectness scores, and bounding boxes.


The model used was yolov5m with transfer learning on pretrained weights.

#### **Model Training**
Epochs: 300  
Batch Size: 8  
Image Size: 1024 x 1024  
Weights: yolov5m.pt  

![](assets/results.png)  

mAP@.5: 98.17%

**mAP@.5:.95: 85.27%**

Training batch example:  
![](assets/train_batch2.jpg)

Test batch predictions example:
![](assets/test_batchm_pred.jpg)  

<a name="inference"></a>
- [Back to Contents](#contents)
# **Inference**
### **Images**
I had reserved a test set of my son’s attempts at each letter that was not included in any of the training and validation sets.  In fact no pictures of hands from children were used for training the model.  Ideally several more images would help in showcasing how well our model performs, but this a start.
![](assets/test_slides.png)  

Out of 26 letters, 18 were correctly predicted.

Letters that did not receive a prediction (G, H, J, and Z)  

Letters that were incorrectly predicted were:   
“D” predicted as “F”  
“E” predicted as “T”  
“P“ predicted as “Q”  
“R” predicted as “U”  


## **Video Findings:**  

==============================================================  
**Left-handed:**  
This test shows that our image augmentation pipeline performed well as it was set to flip the images horizontally at a 50% probability.  
![](assets/left_handed.gif)  

==============================================================  
**Child's hand:**
The test on my son's hand was performed, and the model still performs well here.  
![](assets/son_name.gif)

==============================================================  
**Multiple letters on screen:**  
Simultaneous letters were also  detected. Although sign language is not used like the video on the right, it shows that multiple people can be on screen and the model will be able to distinguish more than one instance of the language.  
![](assets/hi_screen_record.gif)   

==============================================================   
## **Video Limitations:**  
==============================================================  
**Distance**  
There were limitations I’ve discovered in my model. The biggest one is distance.  As many of the original pictures were taken from my phone on my hands, the distance of my hand to the camera was very close, negatively impacting inference at further distances.  

![](assets/distance_limitation.gif)  

==============================================================  
**New environments**  
These video clips of volunteers below were not included in any of the model training.  Although the model picks up a lot of the letters, the prediction confidence levels are lower, and there are more misclassifications present.  
![](assets/volunteers.gif)  
 

I've verified this with a video of my own.  
![](assets/bg_limitation.gif)  

**Even though the original image set was on only 720 pictures, the implications of the results displayed bring us to an exciting conclusion.**

==============================================================  
<a name="conclusions"></a>

- [Back to Contents](#contents)
# **Conclusions**   
Computer vision can and should be used in marking a step in greater accessibility and educational resources for our deaf and hard of hearing communities!  

- Even though the original image set was on only 720 pictures, the implications of the results displayed here is promising
- Gathering more image data from a variety of sources would help our model inference in different distances and environments better.
- Even letters with movements are able to be recognized through computer vision.


<a name="nextsteps"></a>
- [Back to Contents](#contents)
# **Next Steps**  
I believe this project is aligned with the vision of the National Association of the Deaf in bringing better accessibility and education for this underrepresented community.  If I am able to bring awareness to the project, and partner with an organization like the NAD, I will be able to gather better data on the people that speak this language natively to push the project further.

The technology is still very new, and the model I have trained for this presentation was primarily used to find out if it would work. I’m happy with my initial results and I’ve already trained a smaller model that I’ll be testing for mobile deployment in the future.

I believe computer vision can help give our deaf and hard of hearing neighbors a voice with the right support and project awareness.

- [Back to Contents](#contents)
<a name="cite"></a>
# **Citations**  
Python Version: 3.8  
Packages: pandas, numpy, matplotlib, sklearn, opencv, os, ast, albumentations, tqdm, torch, IPython, PIL, shutil

### Resources:  

Yolov5 github  
https://github.com/ultralytics/yolov5  

Yolov5 requirements  
https://github.com/ultralytics/yolov5/blob/master/requirements.txt

Cudnn install guide:
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

Install Opencv:
https://www.codegrepper.com/code-examples/python/how+to+install+opencv+in+python+3.8

Roboflow augmentation process:
https://docs.roboflow.com/image-transformations/image-augmentation

Heavily utilized research paper on image augmentations:
https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0#Sec3

Pillow library:
https://pillow.readthedocs.io/en/latest/handbook/index.html

Labeling Software labelImg:
https://github.com/tzutalin/labelImg

Albumentations library
https://github.com/albumentations-team/albumentations

# **Special Thanks**
Joseph Nelson, CEO of Roboflow.ai, for delivering a computer vision lesson to our class, and answering my questions directly.

And to my volunteers:  
Nathan & Roxanne Seither  
Juhee Sung-Schenck  
Josh Mizraji  
Lydia Kajeckas  
Aidan Curley  
Chris Johnson  
Eric Lee  

And to the General Assembly DSI-720 instructors:  
Adi Bronshtein  
Patrick Wales-Dinan  
Kelly Slatery  
Noah Christiansen  
Jacob Ellena  
Bradford Smith

This project would not have been possible without the time all of you invested in me.  Thank you!