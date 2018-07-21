import config
import glob
import tensorflow as tf
import re
import cv2
import numpy as np

#含中文路径读取图片方法
def cv_imread(filepath):
    img = cv2.imdecode(np.fromfile(filepath,dtype=np.uint8),-1)
    return img


#把数据打包，转换成tfrecords格式，以便后续高效读取
def convert_to_tfrecords():
    writer = tf.python_io.TFRecordWriter('./data.tfrecords')
    paths = glob.glob(config.DATADIR + '/train/*/*.jpg')
    label_names = [re.split("/|\\\\",path)[3] for path in paths]
    labels = [0 if name=='正常' else 1 for name in label_names]
    writed_num = 0
    for path,label in zip(paths,labels):
        img = cv_imread(path)
        height,width,depth = img.shape

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[depth])),
            'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))

        serialized = example.SerializeToString()
        writer.write(serialized)
        writed_num+=1

    print("convert %s images to tfrecords!!!"%writed_num)
    writer.close()

#从tfrecords中解压获取图片
def get_from_tfrecords(filepaths,num_epoch=None):
    filename_queue = tf.train.string_input_producer(filepaths,num_epochs=num_epoch)  # 因为有的训练数据过于庞大，被分成了很多个文件，所以第一个参数就是文件列表名参数
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    example = tf.parse_single_example(serialized, features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
        'img': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })
    label = tf.cast(example['label'], tf.int32)
    img = tf.decode_raw(example['img'], tf.uint8)
    img = tf.reshape(img, [
        tf.cast(example['height'], tf.int32),
        tf.cast(example['width'], tf.int32),
        tf.cast(example['depth'], tf.int32)])

    # label=example['label']
    return img, label


# 根据队列流数据格式，解压出一张图片后，输入一张图片，对其做预处理、及样本随机扩充
def get_batch(img, label, batch_size, crop_size):
    # 数据扩充变换
    img = tf.random_crop(img, [crop_size, crop_size, 3])  # 随机裁剪
    img = tf.image.random_flip_up_down(img)  # 上下随机翻转
    # distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)#亮度变化
    # distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)#对比度变化

    # 生成batch
    # shuffle_batch的参数：capacity用于定义shuttle的范围，如果是对整个训练数据集，获取batch，那么capacity就应该够大
    # 保证数据打的足够乱
    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=batch_size,
                                                 num_threads=16, capacity=50000, min_after_dequeue=10000)
    # img_batch, label_batch=tf.train.batch([img, label],batch_size=batch_size)



    # 调试显示
    # tf.image_summary('images', images)
    return img_batch, label_batch

#测试tfrecords中解压获取数据是否正常
def test_tfrecords(filepaths):
    img,label = get_from_tfrecords(filepaths)
    img_batch,label_batch = get_batch(img,label,32,800)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # for i in range(10000):
        #     img_np = sess.run(img)
        #     cv2.imshow("temp",img_np)
        #     cv2.waitKey()

        img_batch_r,label_batch_r = sess.run([img_batch,label_batch])
        print(img_batch_r.shape)
        print(label_batch_r.shape)
