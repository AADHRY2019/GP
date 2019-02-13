import numpy as np
import tensorflow as tf

class Randomvqa(object):

    def __init__(self,config):
        self.word_dim = config['word_dim']
        self.vocab_num = config['vocab_num']
        self.pretrained_embedding = config['pretrained_embedding']
        self.appear_dim = config['appear_dim']
        self.frame_num = config['frame_num']
        self.motion_dim = config['motion_dim']
        self.clip_num = config['clip_num']
        self.common_dim = config['common_dim']
        self.answer_num = config['answer_num']
        self.logit = None


    def build_inference(self):
        with tf.name_scope('input'):
            self.appear = tf.placeholder(
                tf.float32, [None, self.frame_num, self.appear_dim], 'appear')
            self.motion = tf.placeholder(
                tf.float32, [None, self.clip_num, self.motion_dim], 'motion')
            self.question_encode = tf.placeholder(
                tf.int64, [None, None], 'question_encode')
        with tf.variable_scope('init'):
            shape = tf.shape(self.question_encode)
            batch_size = shape[0]
        with tf.name_scope('output'):
            self.logit = tf.zeros([batch_size,self.answer_num])+1.0/self.answer_num
            self.prediction =tf.zeros([batch_size],dtype = tf.int64)

    def build_loss(self, reg_coeff):
        """Compute loss and acc."""
        with tf.name_scope('answer'):
            self.answer_encode = tf.placeholder(
                tf.int64, [None], 'answer_encode')
            answer_one_hot = tf.one_hot(
                self.answer_encode, self.answer_num)
        with tf.name_scope('loss'):
            log_loss = tf.losses.log_loss(
                answer_one_hot, self.logit, scope='log_loss')
            reg_loss = tf.add_n(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
            # fix channel selection
            self.loss = log_loss + reg_coeff * reg_loss

        with tf.name_scope("acc"):
            correct = tf.equal(self.prediction, self.answer_encode)
            self.acc = tf.reduce_mean(tf.cast(correct, "float"))
