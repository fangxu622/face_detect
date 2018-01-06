from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
from data import inputs
import numpy as np
import tensorflow as tf
from model import select_model, get_checkpoint
from utils import *
import os
import json
import csv
import dlib
import cv2


RESIZE_FINAL = 227
GENDER_LIST = ['M', 'F']
AGE_LIST = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
MAX_BATCH_SZ = 128

skip_frame=2
scale=0.5
sample=1
BODER_PIX=20

video_path="qr.mp4"

tf.app.flags.DEFINE_string('model_dir', '/home/test/jintianPrj/rude-carnie/data/21936',
                           'Model directory (where training data lives)')

tf.app.flags.DEFINE_string('class_type', 'gender',
                           'Classification type (age|gender)')

tf.app.flags.DEFINE_string('device_id', '/cpu:0',
                           'What processing unit to execute inference on')

tf.app.flags.DEFINE_string('filename', '/home/test/jintianPrj/rude-carnie/data/WI.jpg',
                           'File (Image) or File list (Text/No header TSV) to process')

tf.app.flags.DEFINE_string('target', '',
                           'CSV file containing the filename processed along with best guess and score')

tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
                           'Checkpoint basename')

tf.app.flags.DEFINE_string('model_type', 'inception',
                           'Type of convnet')

tf.app.flags.DEFINE_string('requested_step', '', 'Within the model directory, a requested step to restore e.g., 9000')

tf.app.flags.DEFINE_boolean('single_look', True, 'single look at the image or multiple crops')

tf.app.flags.DEFINE_string('face_detection_model', '', 'Do frontal face detection with model specified')

tf.app.flags.DEFINE_string('face_detection_type', 'cascade', 'Face detection model type (yolo_tiny|cascade)')

FLAGS = tf.app.flags.FLAGS


def one_of(fname, types):
    return any([fname.endswith('.' + ty) for ty in types])


def resolve_file(fname):
    if os.path.exists(fname): return fname
    for suffix in ('.jpg', '.png', '.JPG', '.PNG', '.jpeg'):
        cand = fname + suffix
        if os.path.exists(cand):
            return cand
    return None


def classify_many_single_crop(sess, label_list, softmax_output, image_batch): #,coder, image_files, writer):
    try:
        #num_batches = math.ceil(len(image_files) / MAX_BATCH_SZ)
        #pg = ProgressBar(num_batches)
        num_batches=image_batch.get_shape()[0]
        result = []
        for j in range(num_batches):
            #start_offset = j * MAX_BATCH_SZ
            #end_offset = min((j + 1) * MAX_BATCH_SZ, len(image_files))

            #batch_image_files = image_files[start_offset:end_offset]
            #print(start_offset, end_offset, len(batch_image_files))

            #image_batch = make_multi_image_batch(batch_image_files, coder)

            batch_results = sess.run(softmax_output, feed_dict={images: image_batch.eval()})
            batch_sz = batch_results.shape[0]
            for i in range(batch_sz):
                output_i = batch_results[i]
                best_i = np.argmax(output_i)
                best_choice = (label_list[best_i], output_i[best_i])
                print('Guess @ 1 %s, prob = %.2f' % best_choice)
                result.append(label_list[best_i])
                #if writer is not None:
                    #f = batch_image_files[i]
                    #writer.writerow((f, best_choice[0], '%.2f' % best_choice[1]))
            #pg.update()
        return result

        #pg.done()
    except Exception as e:
        print(e)
        print('Failed to run all images')


def classify_one_multi_crop(sess, label_list, softmax_output, coder, images, image_file, writer):
    try:

        print('Running file %s' % image_file)
        image_batch = make_multi_crop_batch(image_file, coder)

        batch_results = sess.run(softmax_output, feed_dict={images: image_batch.eval()})
        output = batch_results[0]
        batch_sz = batch_results.shape[0]

        for i in range(1, batch_sz):
            output = output + batch_results[i]

        output /= batch_sz
        best = np.argmax(output)
        best_choice = (label_list[best], output[best])
        print('Guess @ 1 %s, prob = %.2f' % best_choice)

        nlabels = len(label_list)
        if nlabels > 2:
            output[best] = 0
            second_best = np.argmax(output)
            print('Guess @ 2 %s, prob = %.2f' % (label_list[second_best], output[second_best]))

        if writer is not None:
            writer.writerow((image_file, best_choice[0], '%.2f' % best_choice[1]))
    except Exception as e:
        print(e)
        print('Failed to run image %s ' % image_file)


def list_images(srcfile):
    with open(srcfile, 'r') as csvfile:
        delim = ',' if srcfile.endswith('.csv') else '\t'
        reader = csv.reader(csvfile, delimiter=delim)
        if srcfile.endswith('.csv') or srcfile.endswith('.tsv'):
            print('skipping header')
            _ = next(reader)

        return [row[0] for row in reader]


#def main(argv=None):  # pylint: disable=unused-argument

def face_dect(video_path,skip_frame,scale,sample,
              sess, label_list, softmax_output):

    # 初始化dlib人脸检测器
    detector = dlib.get_frontal_face_detector()

    cv2.namedWindow("win",cv2.WINDOW_AUTOSIZE)
    cap=cv2.VideoCapture(video_path)
    i=0 #计数 帧率
    while cap.isOpened():
        i=i+1
        ret,cv_img=cap.read()
        if not (i % skip_frame==0):
            pass
        # OpenCV默认是读取为BGR图像，而dlib需要的是EGB图像，因此这一步转换不能少
        img_tmp = cv2.resize(cv_img, None, fx=scale, fy=scale, interpolation=sample) #INTER_LINEAR=1,INTER_CUBIC = 2

        img = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2RGB)

        #检测人脸
        dets = detector(img,0)
        #检测性别
        print("Number of faces detected: {}".format(len(dets)))
        #
        imgs=[]
        for i, d in enumerate(dets):
            img_tem=cv_img[int(d.top()/scale)+BODER_PIX:int(d.bottom()/scale+BODER_PIX),
                        int(d.left()/scale+BODER_PIX):int(d.right()/scale)+BODER_PIX]

            img_tem2=cv2.resize(img_tem,(RESIZE_FINAL,RESIZE_FINAL),fx=0,fy=0,interpolation=sample)
            img_tem3=tf.Variable(img_tem2)
            imgs.append(img_tem3)
            #global imgbatch
        imgbatch=tf.stack(imgs)

        res=classify_many_single_crop(sess, label_list, softmax_output, imgbatch)
        k=0
        for i, d in enumerate(dets):

            cv2.rectangle(cv_img,
                          (int(d.left()/scale),int(d.top()/scale)),
                               (int(d.right()/scale), int(d.bottom()/scale)),
                                (0, 255, 0),
                                1)
            cv2.putText(cv_img, "m",
                        (int(d.left()/scale),int(d.bottom()/scale)),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),
                        thickness=1)
            k = k + 1
            print(res)
        cv2.imshow('win', cv_img)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()




if __name__ == '__main__':
    #tf.app.run()
    #main()
    files = []

    if FLAGS.face_detection_model:
        print('Using face detector (%s) %s' % (FLAGS.face_detection_type, FLAGS.face_detection_model))
        face_detect = face_detection_model(FLAGS.face_detection_type, FLAGS.face_detection_model)
        face_files, rectangles = face_detect.run(FLAGS.filename)
        print(face_files)
        files += face_files

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        label_list = AGE_LIST if FLAGS.class_type == 'age' else GENDER_LIST
        nlabels = len(label_list)

        print('Executing on %s' % FLAGS.device_id)
        model_fn = select_model(FLAGS.model_type)

        with tf.device(FLAGS.device_id):

            images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
            logits = model_fn(nlabels, images, 1, False)
            init = tf.global_variables_initializer()

            requested_step = FLAGS.requested_step if FLAGS.requested_step else None

            checkpoint_path = '%s' % (FLAGS.model_dir)

            model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, FLAGS.checkpoint)

            saver = tf.train.Saver()
            saver.restore(sess, model_checkpoint_path)

            softmax_output = tf.nn.softmax(logits)

            coder = ImageCoder()

            # Support a batch mode if no face detection model
            if len(files) == 0:
                if (os.path.isdir(FLAGS.filename)):
                    for relpath in os.listdir(FLAGS.filename):
                        abspath = os.path.join(FLAGS.filename, relpath)

                        if os.path.isfile(abspath) and any(
                                [abspath.endswith('.' + ty) for ty in ('jpg', 'png', 'JPG', 'PNG', 'jpeg')]):
                            print(abspath)
                            files.append(abspath)
                else:
                    files.append(FLAGS.filename)
                    # If it happens to be a list file, read the list and clobber the files
                    if any([FLAGS.filename.endswith('.' + ty) for ty in ('csv', 'tsv', 'txt')]):
                        files = list_images(FLAGS.filename)
            writer = None
            output = None
            if FLAGS.target:
                print('Creating output file %s' % FLAGS.target)
                output = open(FLAGS.target, 'w')
                writer = csv.writer(output)
                writer.writerow(('file', 'label', 'score'))
            image_files = list(filter(lambda x: x is not None, [resolve_file(f) for f in files]))
            print(image_files)
            if FLAGS.single_look:

                # face
                face_dect(video_path, skip_frame,scale, sample,
                              sess,
                              label_list,softmax_output)
            #classify_many_single_crop(sess, label_list, softmax_output, images,coder, image_files, writer)
            else:
                for image_file in image_files:
                    classify_one_multi_crop(sess, label_list, softmax_output, images,coder, image_file, writer)

            if output is not None:
                output.close()
