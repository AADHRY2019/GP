import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np


class RHA(object):
    def __init__(self, config):
        self.word_dim = config['word_dim']
        self.vocab_num = config['vocab_num']
        self.pretrained_embedding = config['pretrained_embedding']
        self.appear_dim = config['appear_dim']
        self.frame_num = config['frame_num']
        self.motion_dim = config['motion_dim']
        self.clip_num = config['clip_num']
        self.common_dim = config['common_dim']
        self.answer_num = config['answer_num']
        self.motion = None
        self.appear = None
        self.question_encode = None
        self.answer_encode = None

        self.channel_weight = None

        self.logit = None
        self.prediction = None
        self.loss = None
        self.acc = None

        self.train = None
        self.is_train = True
        self.all_weights = None

    def build_loss(self, reg_coeff, shu_coeff):
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
            shu_loss = tf.reduce_sum(
                tf.abs(self.channel_weight[:, 0] - self.channel_weight[:, 1]))
            self.loss = log_loss + reg_coeff * reg_loss  # + shu_coeff * shu_loss

        with tf.name_scope("acc"):
            correct = tf.equal(self.prediction, self.answer_encode)
            self.acc = tf.reduce_mean(tf.cast(correct, "float"))

    def build_train(self, learning_rate):
        """Add train operation."""
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train = optimizer.minimize(self.loss)

    def build_inference(self):
        with tf.name_scope('input'):
            self.appear = tf.placeholder(
                tf.float32, [None, self.frame_num, self.appear_dim], 'appear')
            self.motion = tf.placeholder(
                tf.float32, [None, self.clip_num, self.motion_dim], 'motion')
            self.question_encode = tf.placeholder(
                tf.int64, [None, None], 'question_encode')

        with tf.variable_scope('embedding'):
            if self.pretrained_embedding:
                embedding_matrix = tf.get_variable(
                    'embedding_matrix', initializer=np.load(self.pretrained_embedding),
                    regularizer=tf.nn.l2_loss)
            else:
                embedding_matrix = tf.get_variable(
                    'embedding_matrix',
                    [self.vocab_num, self.word_dim], regularizer=tf.nn.l2_loss)
            question_embedding = tf.nn.embedding_lookup(
                embedding_matrix, self.question_encode, name='word_embedding')

        with tf.variable_scope('transform_video'):
            with tf.variable_scope('appear'):
                W = tf.get_variable(
                    'W', [self.appear_dim, self.common_dim],
                    regularizer=tf.nn.l2_loss)
                b = tf.get_variable('b', [self.common_dim])
                appear = tf.reshape(self.appear, [-1, self.appear_dim])
                appear = tf.nn.xw_plus_b(appear, W, b)
                appear = tf.reshape(
                    appear, [-1, self.frame_num, self.common_dim])
                appear = tf.nn.tanh(appear)
            with tf.variable_scope('motion'):
                W = tf.get_variable(
                    'W', [self.motion_dim, self.common_dim],
                    regularizer=tf.nn.l2_loss)
                b = tf.get_variable('b', [self.common_dim])
                motion = tf.reshape(self.motion, [-1, self.motion_dim])
                motion = tf.nn.xw_plus_b(motion, W, b)
                motion = tf.reshape(
                    motion, [-1, self.clip_num, self.common_dim])
                motion = tf.nn.tanh(motion)

        with tf.variable_scope('init'):
            shape = tf.shape(self.question_encode)
            batch_size = shape[0]
            question_length = shape[1]
            time = tf.constant(0, name='time')

            q_cell = tf.nn.rnn_cell.BasicLSTMCell(self.word_dim)
            q_state = q_cell.zero_state(batch_size, tf.float32)

            channel_weight = tf.zeros([batch_size, 2])
            fused = tf.zeros([batch_size, self.common_dim])
            all_weights = tf.zeros([batch_size, 2])

            word_embed_W = tf.get_variable(
                'word_embed_W', [self.word_dim, self.common_dim],
                regularizer=tf.nn.l2_loss)
            word_embed_b = tf.get_variable(
                'word_embed_b', [self.common_dim])
            channel_W = tf.get_variable(
                'channel_W', [self.word_dim, 2],
                regularizer=tf.nn.l2_loss)
            channel_b = tf.get_variable('channel_b', [2])

            # q_output = tf.get_variable(
            #     'q_out',
            #     [batch_size,self.word_dim])
            q_output = tf.zeros([batch_size, self.word_dim])

            def lstm_step(time, q_state, q_output):
                word_embedding = question_embedding[:, time]
                with tf.variable_scope('lstm_q'):
                    q_output, q_state = q_cell(word_embedding, q_state)
                return time + 1, q_state, q_output

            time, q_state, q_output = tf.while_loop(
                cond=lambda time, *_: time < question_length,
                body=lstm_step,
                loop_vars=[time, q_state, q_output])
            time = tf.constant(0, name='time')

            with tf.name_scope('transform_q'):
                question = tf.nn.xw_plus_b(
                    q_output, word_embed_W, word_embed_b)
                question = tf.nn.tanh(question)

            def _one_step(time, question, channel_weight, fused, all_weights):
                """One time step of model."""
                word_embedding = question_embedding[:, time]
                # map to common dimension
                with tf.name_scope('transform_w'):
                    word = tf.nn.xw_plus_b(
                        word_embedding, word_embed_W, word_embed_b)
                    word = tf.nn.tanh(word)

                with tf.variable_scope('amu'):
                    with tf.name_scope('attend_1'):
                        appear_weight_1, appear_att_1 = self.attend(
                            word, appear, 'appear')
                        motion_weight_1, motion_att_1 = self.attend(
                            word, motion, 'motion')

                    with tf.name_scope('channel_fuse'):
                        # word attend on channel
                        channel_weight = tf.nn.softmax(
                            tf.nn.xw_plus_b(word_embedding, channel_W, channel_b))
                        cw_appear = tf.expand_dims(channel_weight[:, 0], 1)
                        cw_motion = tf.expand_dims(channel_weight[:, 1], 1)
                        current_video_att = cw_appear * appear_att_1 + cw_motion * motion_att_1

                        # all_weights =tf.math.add(all_weights,channel_weight)
                        # self.all_weights =channel_weight

                with tf.name_scope('RN'):
                    if time == 0:
                        fused = current_video_att
                    else:
                        fused = self.g_theta(fused, current_video_att, question)
                # tf.print(channel_weight)

                return time + 1, question, channel_weight, fused, all_weights

            time, q_output, channel_weight, fused, all_weights = tf.while_loop(
                cond=lambda time, *_: time < question_length,
                body=_one_step,
                loop_vars=[time, q_output, channel_weight, fused, all_weights])
            self.channel_weight = channel_weight
            self.all_weights = all_weights
            with tf.variable_scope('output'):
                # W = tf.get_variable(
                #     'W', [self.common_dim, self.answer_num],
                #     regularizer=tf.nn.l2_loss)
                # b = tf.get_variable('b', [self.answer_num])
                self.logit = tf.nn.softmax(self.f_phi(fused, self.answer_num), name='logit')
                self.prediction = tf.argmax(self.logit, axis=1, name='prediction')

    def g_theta(self, o_i, o_j, q, scope='g_theta', reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse) as scope:
            g_1 = self.fc(tf.concat([o_i, o_j, q], axis=1), 256, name='g_1')
            g_2 = self.fc(g_1, 256, name='g_2')
            g_3 = self.fc(g_2, 256, name='g_3')
            g_4 = self.fc(g_3, 256, name='g_4')
            return g_4

    def f_phi(self, g, out_dim, scope='f_phi'):
        with tf.variable_scope(scope) as scope:
            # fc_1 = self.fc(g, 256, name='fc_1')
            # fc_2 = self.fc(fc_1, 256, name='fc_2')
            # fc_2 = slim.dropout(fc_2, keep_prob=0.5, is_training=self.is_train, scope='fc_3')
            # fc_3 = self.fc(fc_2, out_dim, activation_fn=None, name='fc_3')
            fc_3 = self.fc(g, out_dim, name='fc_3')
            return fc_3

    def fc(self, input, output_shape, activation_fn=tf.nn.relu, name="fc"):
        with tf.name_scope(name):

            output = fully_connected(input, int(output_shape), activation_fn=activation_fn
                                          # ,  weights_regularizer = slim.l2_regularizer(1e-6)
                                          )
        return output

    def attend(self, target, sources, name=None):
        """Use target to attend on sources. `target` and `sources` should have equal dim.

        Args:
            target: [None, target_dim].
            sources: [None, source_num, source_dim].
        Returns:
            weight: [None, source_num].
            att: [None, source_dim].
        """
        with tf.name_scope(name, 'attend'):
            weight = tf.nn.softmax(tf.reduce_sum(
                tf.expand_dims(target, 1) * sources, 2))
            att = tf.reduce_sum(
                tf.expand_dims(weight, 2) * sources, 1)
            return weight, att

