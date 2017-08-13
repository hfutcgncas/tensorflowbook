import glob

root_path = "../../../../../"

# find filename
image_filenames = glob.glob(root_path + "Images/n02*/*.jpg")
print "\n image_filenames:"
print image_filenames[0:2]
print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"


from itertools import groupby
from collections import defaultdict

image_filename_with_breed = map(lambda filename: (filename.split("/")[-2], filename), image_filenames)
print "\n image_filename_with_breed :"
print image_filename_with_breed[0:2]
print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

testing_dataset = defaultdict(list)
training_dataset = defaultdict(list)
for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x:x[0]):
    for i, breed_image in enumerate(breed_images):
        if i % 5 == 0:
            testing_dataset[dog_breed].append(breed_image[1])
        else:
            training_dataset[dog_breed].append(breed_image[1])

print "\n training_dataset['n02091134-whippet'][0:2] : "
print training_dataset["n02091134-whippet"][0:2]

breed_training_count = len(training_dataset[dog_breed])
breed_testing_count = len(testing_dataset[dog_breed])

assert round(breed_testing_count*1.0/(breed_training_count + breed_testing_count), 2)>0.18, "Not enough testing images."


import tensorflow as tf
from PIL import Image  #注意Image,后面会用到


def write_record_file(dataset, record_loaction):

    writer = None

    current_index = 0

    for breed, image_filenames in dataset.items():
        for image_filename in image_filenames:
            if current_index % 100 == 0:
                if writer:
                    writer.close()

                record_filename = "{record_loaction}-{current_index}.tfrecords".format(record_loaction=record_loaction, current_index=current_index)
                writer = tf.python_io.TFRecordWriter(record_filename)
                print "current_index: ", current_index


            current_index += 1

            image_file = tf.read_file(image_filename)
            try:
                image = tf.image.decode_jpeg(image_file)
            except:
                print(image_filename)
                continue
            grayscale_image = tf.image.rgb_to_grayscale(image)
            resized_image = tf.image.resize_images(grayscale_image, [250,151], method=tf.image.ResizeMethod.AREA)
            image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()

            # img=Image.open(image_filename)
            # img= img.resize((250, 151))
            # image_bytes = img.tobytes()#将图片转化为二进制格式


            image_label = breed.encode("utf_8")

            example = \
            tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            }))
            writer.write(example.SerializeToString())

    writer.close()

with tf.Session() as sess:
    write_record_file(testing_dataset, root_path + "output/testing-images/testing-image")
    write_record_file(training_dataset, root_path + "output/training-images/testing-image")
