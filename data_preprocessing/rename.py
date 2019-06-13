#2、重命名
import os
if __name__=="__main__":
    data_dir = 'D:/project/ShuffleNet/pa/dogsunset'
    img_list = os.listdir(data_dir)
    for i in range(len(img_list)):
        pre_name = os.path.join(data_dir, img_list[i])
        new_name = os.path.join(data_dir, "DogSunset_{}.jpg".format(i))
        os.rename(pre_name, new_name)
    print("finish")