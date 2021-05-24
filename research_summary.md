# Research Summary
This is the compiled summary of all the research done regarding this project

## Papers Summaries

### Ideas

### Datasets

#### [Crowd Human Dataset](Papers/1805.00123.pdf)
[Download Link](https://www.crowdhuman.org/download.html)
___

#### Pedestrian Images
[Download Link](https://www.cis.upenn.edu/~jshi/ped_html/)
___

#### Kaggle Mall Dataset
[Download Link](https://www.kaggle.com/constantinwerner/human-detection-dataset/version/1)

___

#### Collective Activity Dataset
[Download Link](http://vhosts.eecs.umich.edu/vision//activity-dataset.html)  
[Paper](http://vhosts.eecs.umich.edu/vision//papers/Wongun_CollectiveActivityRecognition09.pdf)

### Crowd-counting Models

#### [Deep People Counting in Extremely Dense Crowds](Papers/sp055u.pdf)
___
#### [Fine Grained Crowd-Counting](Papers/tip21-fgc.pdf)
Credits: Jia Wan, Nikil S. Kumar, and Antoni B. Chan,
IEEE Trans. on Image Processing (TIP), 30:2114-2126, Jan 2021. 
___
#### [From Open Set to Closed Set: Supervised Spatial Divide-and-Conquer for Object Counting](Papers/2001.01886.pdf)
[Github Link](https://github.com/xhp-hust-2018-2011/SS-DCNet)  
- Helps solve the Open set problem of crowd counting when there are large amounts of people  
- Good for large crowds  
___
### Human Detection Models

#### [People Detection and Finding Attractive Areas by the use of Movement Detection Analysis and Deep Learning Approach](./Papers/main.pdf)
- Using YOLOv3/ SSD is not bad  
- Consider some form of background subtraction/ preprocessing to the image  


#### [PifPaf: Composite Fields for Human Pose Estimation](Papers/Kreiss_PifPaf_Composite_Fields_for_Human_Pose_Estimation_CVPR_2019_paper.pdf)

#### [OpenPifPaf: Composite Fields for Semantic Keypoint Detection and Spatio-Temporal Association](Papers/2103.02440.pdf)

### Social Distancing Tracking

#### Monoloco
[Github Link](https://github.com/vita-epfl/monoloco)

#### [Perceiving Humans: from Monocular 3D Localization to Social Distancing](https://arxiv.org/pdf/2009.00984.pdf)
##### Proposed method:
- First, we exploit a pose detector to escape the image domain and reduce the input dimensionality.  
- Second, we use the 2D joints as input to a feed-forward neural network that predicts x-y-z coordinates and the associated uncertainty, orientation and dimensions of each pedestrian.  
- Third, the network estimates are combined to obtain F-formations and recognise social interactions.  


## Tests

### Tracking
I have been trying to find good algos to help with tracking. Apparently, under this library holds inbuilt tracking capabilities, using CifCaf. Supposedly the results are good while reducing computational time. Hence, it might be good to figure out how this decoder to work so that I can let it remain at 20fps for human detection.

### Cannibalising OpenPifPaf
I have tried to tear apart this library to optimise/ isolate the processes to try to convert to other formats and fine tune parameters. However, when doing so, it exhibited weird results

#### Isolating processor.batch
Apparently due to a large amount of postprocessing, just by isolating this, I have managed to improve performance from a 7 fps to 20fps ish. This is probably due to how the library paints the images and prepares the output.

#### Isolating model.forward
Doing this wasnt good as it reduced perf from 20fps to 10fps. This might be due to how the library queues workload across a worker pool. Hence it might be good just to allow this to remain