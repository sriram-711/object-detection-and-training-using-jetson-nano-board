# Object Detection And Training Using Jetson-Nano Board
You have to do this process in your external Linux environment, other than jetson nano board
<img width="833" alt="Screenshot 2024-12-26 at 6 33 43 PM" src="https://github.com/user-attachments/assets/f405ca18-4d00-427f-a517-5a9e0ac6b6e0" />
# Step-1: (Expand memory if required)
for that, ensure sufficient memory is available, if not increase memory using gparted tool
* sudo apt-get install gparted
* sudo gparted
* expand memory and save
# Step-2: (Install dependencies)
* make sure python3 (version == 3.6.9)is installed
* install dependencies below
* sudo apt install python3-pip (version == 21.3.1)
* pip3 install opencv-python (version == 4.6.0.66)
* pip3 install imutils (version == 0.5.4)
* pip3 install matplotlib(version == 2.1.1)
* pip3 install torchvision(version == 0.9.0)
* pip3 install torch (version == 1.8.0)
* pip3 install boto3 (version == 1.23.10)
* pip3 install pandas (version == 0.22.0)
* pip3 install urllib3 (version == 1.26.20)
* sudo apt-get install pyqt5-dev-tools (version == 5.10.1+dfsg-1ubuntu2)
* sudo apt-get install python3-lxml (version == 4.2.1-1ubuntu0.6)
* sudo apt install git (version == 2.17.1)
* if you want to install any other dependencies a part from this checkout the Dependencies.txt file  
# Step-3: (Clone jetson official github repository for training)
* git clone https://github.com/mailrocketsystems/jetson-train.git
# Step-4: (Prepare Dataset (Convert Video to Frames)
* create a video in .mp4 format that covers all the possible views of each objects you are going to train
* place the video inside jetson-train/videos in mp4 format
* now we have to convert the video file to frame by frame images using prepare_dataset.py python file
* open prepare_dataset.py
* in the 16th line of code, give the exact name of your video file (no need if you have updated code)
* in the 20th line of code, we have a predefined count 10, you can increase the count to decrease the number of images to be generated and vice-versa
* save and close prepare_dataset.py
* cd jetson-train/
* python3 prepare_dataset.py
* it will ask model name,give your model name.
* now video will play,and it will store several images inside the model file you mentioned
* Annotating using Label image tool
* now we have to give the annotation to each image we've captured
* for that, we need to clone and use labelimg tool in home
# Step-5: (Annotate Frames Using LabelImg Tool)
* git clone https://github.com/HumanSigal/labelImg.git
* cd labelImg/
* make qt5py3
* python3 labelImg.py
* this will open label image tool
* click open dir and choose path as jetson-train/data/model_name/JPEGimages/
* it will load all the image files in the file list
* click change save dir and choose path as jetson-train/data/model_name/Annotations/
* bit will save all annotations inside directory
* click create rect box and draw boundary to the object, give name to object and click save
* paste rectbox for each repeating image and draw new for new object
* do it for all images and save each
* Training custom objects on linux environment
  
<img width="828" alt="Screenshot 2024-12-26 at 6 38 19 PM" src="https://github.com/user-attachments/assets/6cdb747f-fe9c-4fb4-aa5f-5e0c7bf51a96" />

# Step-6: (Create labels.txt List Object Names)
* create labels.txt file inside jetson-train/data/model_name (gedit labels.txt)
* list the name of objects you are going to train in the labels.txt file, one name in one line and top left alligned,and save.
# Step-7: (Train Model Run train_ssd.py)
* open a new terminal at jetson-train
* python3 train_ssd.py --dataset-type=voc --data=data/model_name/ --model-dir=models/model_name --batch-size=2 --workers=2 --epochs=300
* close all other applications on the system
* it will take some time
# Step-8: (Evaluate Model Run result.py)
* open terminal at jetson-train
# Step-9: (Copy Best Checkpoint and labels.txt)
* python3 result.py
* give model_name
* it will show results graph and it will give best check point
* copy best check point and labels.txt file
<img width="855" alt="Screenshot 2024-12-26 at 6 44 17 PM" src="https://github.com/user-attachments/assets/4ead11fe-1c54-4a1f-bb34-f6e796663b0b" />

# Step-10: (Copy Files to Jetson Nano (Checkpoint, labels.txt))
* Rest of project inside jetson nano
* git clone --recursive https://github.com/dusty-nv/jetson-inference
* after cloning follow external resorces(youtube) for buliding jetson-inference 
* make sure you have jetson-inference folder in the device
* create a folder with your mode_name in the directory jetson-inference/python/training/detection/ssd/models/
* paste the copied check point and labels.txt file
# Step-11: (Convert PyTorch Model to ONNX (Run onnx_export.py)
* now you'll need to convert your PyTorch model to ONNX
* cd jetson-inference
* docker/run.sh
* The above step will allow you to enter into the root of jetson nano developer kit
* open anew terminal inside jetson-inference/python/training/detection/ssd/
* python3 onnx_export.py --model-dir=models/model_name
* The converted model will then be saved under model_name/ssd-mobilenet.onnx, which you can then load with the detectnet programs
# Step-12: (Run Object Detection on Jetson Nano (Run detectnet))
* make sure you are in root jetson-inference/python/training/detection/ssd
* detectnet --model=models/model_name/ssd-mobilenet.onnx --labels=models/model_name/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes /dev/video0
# Step-13: (Detection Results (Output: Bounding Boxes))
* the above command will start the detection procedures
* that's all ,jetson will detect you custom objects.
# Step-14: (Optional) Object Detection and Distance Calculation Using Bounding Boxes
* In addition to detecting objects, you can calculate the distance of detected objects based on the bounding box dimensions. For instance, if you know the real-world height of the object, you can estimate 
  the distance using the following formula
* distance = (KNOWN_HEIGHT * FOCAL_LENGTH) / pixel_height
* KNOWN_HEIGHT is the real-world height of the object in cm (e.g., the height of a car).
* FOCAL_LENGTH is the focal length of your camera (calibrate it for accurate results).
* pixel_height is the height of the bounding box in pixels.

