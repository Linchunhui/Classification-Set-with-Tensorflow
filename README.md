# Classification-tensorflow
This is a image classification implementation in tensorflow. 
> Include：SqueezeNet,Xception,MobileNetV1、V2、V3,ShuffleNetV1、V2.  
> Include：LeNet,AlexNet,VGG,ResNet,Inception,ResNext,DenseNet,SENet. 

# TODO:
Network structure interpretation 

# requirement
python  
cuda  
cudnn  
tensorflow  

# data structure
dateset dir  
* train
	* class1  
 	* class2  
   	* class3  
	* ...  
* val   
	* class1  
 	* class2  
	* class3  
	* ... 
# label map
label.txt  
>   class1:0   
>   class2:1  
>   class3:2  
  ...  
# Usage
## train
python train.py   
		--train_dir="D:/train"  
		--logs_train_dir="./model_save"  
		--N_CLASSES=3   
		--size=224  
		--BATCH_SIZE=32  
		--epochs=50  
		--inin_lr=0.01  
		--decay_steps=20  
		--model="MobileNetV3_small"  
## inference
python inference.py   
		--data_dir="D:/RoadMapSample/"  
		--save_dir="D:/res/"  
		--logs_train_dir="./model_save/model.ckpt-24000"  
		--N_CLASSES=3  
		--cate="['class1','class2','class3'...]"  
		--model="MobileNetV3_small" 
# Tips
Sometimes spying images from website according to the error images is a simple but effective method to increase to accuracy.You can spy from "Baidu" using `spider.py` in `data_preprocessing`.And then rename them to avoid Chinese character.Finally,you must check if the image is undamaged with `readimg.py`.

