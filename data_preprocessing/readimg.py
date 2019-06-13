#3、找出破损的图像（暂时手动删除）
import os
import tensorflow as tf

def get_files(file_dir):
    image_list, label_list = [], []
    for f in os.listdir(file_dir):
        image_list.append(os.path.join(file_dir,f))
    print('There are %d data' % (len(image_list)))
    return image_list

if __name__=="__main__":
    train_dir = "D:/project/ShuffleNet/pa/dogsunset"
    img_list = get_files(train_dir)
    filename = tf.placeholder(tf.string, [], name='filename')
    image_file = tf.read_file(filename)
    # Decode the image as a JPEG file, this will turn it into a Tensor
    image = tf.image.decode_jpeg(image_file)  # 图像解码成矩阵
    sess=tf.Session()
    for i in range(len(img_list)):
        print(img_list[i], '{}/{}'.format(i,len(img_list)))
        img=sess.run(image, feed_dict={filename:img_list[i]})

