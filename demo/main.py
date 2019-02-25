"""train and test using dataset pics"""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

sys.path.insert(0, '../src')
import facenet
import align.detect_face

def get_data(args, subfolder = 'train'):
    pnet, rnet, onet = load_mtcnn(args.gpu_memory_fraction)
    phase_dir = os.path.join(args.dataset_dir, subfolder)
    dirs = os.listdir(phase_dir)
    images = []
    lbls = []
    for d in dirs:
        imgs = os.listdir(os.path.join(phase_dir, d))
        for img_path in imgs:
            image = misc.imread(os.path.join(phase_dir, d, img_path), mode='RGB')
            images.append(image)
            lbls.append(int(d))
            #image, bb = get_face_image(img, args.image_size, args.margin,  pnet, rnet, onet)
            #if image is not None:
            #    images.append(image)
            #    lbls.append(int(d))
    images = np.stack(images)
    lbls = np.array(lbls)
    return images, lbls

def get_feat(images, sess, embeddings, images_placeholder, phase_train_placeholder):
    feed_dict = { images_placeholder: images, phase_train_placeholder:False }
    emb = sess.run(embeddings, feed_dict=feed_dict)
    return emb

def load_mtcnn(gpu_memory_fraction):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet

def load_facenet(args):
    sess = tf.Session()
    with tf.Graph().as_default():
        # Load the model
        facenet.load_model(args.model)

        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    return sess, embeddings, images_placeholder, phase_train_placeholder

def get_feat2(args, images):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(args.model)

            images_placeholder = sess.graph.get_tensor_by_name("input:0")
            embeddings = sess.graph.get_tensor_by_name("embeddings:0")
            phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)
    return emb

def train(args):
    images, y = get_data(args, 'train')
    #sess, embeddings, images_placeholder, phase_train_placeholder = load_facenet(args)
    #X = get_feat(images, sess, embeddings, images_placeholder, phase_train_placeholder)
    X = get_feat2(args, images)
    if args.classifier == 0:
        model = KNeighborsClassifier()
    elif args.classifier == 1:
        model = SVC(kernel='linear',C=0.4)
    elif args.classifier == 2:
        model = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=max(y)+1)

    model.fit(X, y)
    pred = model.predict(X)
    acc = np.mean(pred == y)
    print('train acc: %f' % acc)
    if args.classifier == 0:
        joblib.dump(model, 'model/my_knn_model.model')
    elif args.classifier == 1:
        joblib.dump(model, 'model/my_svm_model.model')
    elif args.classifier == 2:
        joblib.dump('model/my_softmax_model.model')

def test(args):
    if args.classifier == 0:
        model = joblib.load('./model/my_knn_model.model')
    elif args.classifier == 1:
        model = joblib.load('./model/my_svm_model.model')
    elif args.classifier == 2:
        model = joblib.load('model/my_softmax_model.model')

    images, y = get_data(args, 'test')
    X = get_feat2(args, images)
    pred = model.predict(X)
    acc = np.mean(pred == y)
    print('test acc: %f' % acc)

def demo(args):
    if args.classifier == 0:
        model = joblib.load('./model/my_knn_model.model')
    elif args.classifier == 1:
        model = joblib.load('./model/my_svm_model.model')
    elif args.classifier == 2:
        model = joblib.load(model, 'model/my_softmax_model.model')
    classes = ['guru_ge', 'yafei', 'xiao', 'others']
    cap = cv2.VideoCapture('rtsp://:43794')
    pnet, rnet, onet = load_mtcnn(args.gpu_memory_fraction)

    while True:
        frame = get_frame();
        face, bb = get_face_image(frame, args.image_size, args.margin, pnet, rnet, onet)
        X = get_feat(face)

        pred = model.predict(X)
        drawBox(frame, bb, classes, pred)
        cv2.imshow("face_reconition", frame)
        cv2.waitKey(1)

def drawBox(frame, bb, classes, predict):
    cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
    cv2.rectangle(frame, (bb[0], bb[1] - 20), (bb[2], bb[1]), (125, 125, 125), -1)
    cv2.putText(frame, classes[int(predict[0])], (bb[0]+5, bb[1]-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def main(args):
    if args.phase == 'train_test':
        train(args)
        test(args)
    elif args.phase == 'train':
        train(args)
    elif args.phase == 'test':
        test(args)
    else:
        demo(args)
            
def get_face_image(img, image_size, margin, pnet, rnet, onet):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if len(bounding_boxes) < 1:
        return None

    det = np.squeeze(bounding_boxes[0,0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    prewhitened = facenet.prewhiten(aligned)
    return prewhitened, bb

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file', default='../20180402-114759/')
    parser.add_argument('--phase', type=str, help='phase: train_test, train, test or demo', default='train_test')
    parser.add_argument('--classifier', type=int, help='classifier: 0-knn, 1-svm or 2-softmax', default=0)
    parser.add_argument('--dataset_dir', type=str, nargs='+', help='dataset dir path', default='dataset')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.8)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
