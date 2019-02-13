"""Evaluate GRA."""
import os
import argparse

import numpy as np
import tensorflow as tf
import pandas as pd
from pandas import Series, DataFrame

from model.gra import GRA
import config as cfg
import util.dataset as dt


def train(epoch, dataset, config, log_dir):
    """Train model for one epoch."""
    model_config = config['model']
    train_config = config['train']
    sess_config = config['session']

    with tf.Graph().as_default():
        model = GRA(model_config)
        model.build_inference()
        model.build_loss(train_config['reg_coeff'], train_config['shu_coeff'])
        model.build_train(train_config['learning_rate'])

        with tf.Session(config=sess_config) as sess:
            sum_dir = os.path.join(log_dir, 'summary')
            # create event file for graph
            if not os.path.exists(sum_dir):
                summary_writer = tf.summary.FileWriter(sum_dir, sess.graph)
                summary_writer.close()
            summary_writer = tf.summary.FileWriter(sum_dir)

            ckpt_dir = os.path.join(log_dir, 'checkpoint')
            ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
            saver = tf.train.Saver()
            if ckpt_path:
                print('load checkpoint {}.'.format(ckpt_path))
                saver.restore(sess, ckpt_path)
            else:
                print('no checkpoint.')
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                sess.run(tf.global_variables_initializer())

            stats_dir = os.path.join(log_dir, 'stats')
            stats_path = os.path.join(stats_dir, 'train.json')
            if os.path.exists(stats_path):
                print('load stats file {}.'.format(stats_path))
                stats = pd.read_json(stats_path, 'records')
            else:
                print('no stats file.')
                if not os.path.exists(stats_dir):
                    os.makedirs(stats_dir)
                stats = pd.DataFrame(columns=['epoch', 'loss', 'acc'])

            # train iterate over batch
            batch_idx = 0
            total_loss = 0
            total_acc = 0
            batch_total = np.sum(dataset.train_batch_total)

            while dataset.has_train_batch:
                vgg, c3d, question, answer = dataset.get_train_batch()
                feed_dict = {
                    model.appear: vgg,
                    model.motion: c3d,
                    model.question_encode: question,
                    model.answer_encode: answer
                }
                _, loss, acc = sess.run(
                    [model.train, model.loss, model.acc], feed_dict)
                total_loss += loss
                total_acc += acc
                if batch_idx % 100 == 0:
                    print('[TRAIN] epoch {}, batch {}/{}, loss {:.5f}, acc {:.5f}.'.format(
                        epoch, batch_idx, batch_total, loss, acc))
                batch_idx += 1

            loss = total_loss / batch_total
            acc = total_acc / batch_total
            print('\n[TRAIN] epoch {}, loss {:.5f}, acc {:.5f}.\n'.format(
                epoch, loss, acc))

            summary = tf.Summary()
            summary.value.add(tag='train/loss', simple_value=float(loss))
            summary.value.add(tag='train/acc', simple_value=float(acc))
            summary_writer.add_summary(summary, epoch)

            record = Series([epoch, loss, acc], ['epoch', 'loss', 'acc'])
            stats = stats.append(record, ignore_index=True)

            saver.save(sess, os.path.join(ckpt_dir, 'model.ckpt'), epoch)
            stats.to_json(stats_path, 'records')
            dataset.reset_train()
            return loss, acc


def val(epoch, dataset, config, log_dir):
    """Validate model."""
    model_config = config['model']
    sess_config = config['session']

    answerset = pd.read_csv(
        os.path.join(config['preprocess_dir'], 'answer_set.txt'), header=None)[0]

    with tf.Graph().as_default():
        model = GRA(model_config)
        model.build_inference()

        with tf.Session(config=sess_config) as sess:
            sum_dir = os.path.join(log_dir, 'summary')
            summary_writer = tf.summary.FileWriter(sum_dir)

            ckpt_dir = os.path.join(log_dir, 'checkpoint')
            save_path = tf.train.latest_checkpoint(ckpt_dir)
            saver = tf.train.Saver()
            if save_path:
                print('load checkpoint {}.'.format(save_path))
                saver.restore(sess, save_path)
            else:
                print('no checkpoint.')
                exit()

            stats_dir = os.path.join(log_dir, 'stats')
            stats_path = os.path.join(stats_dir, 'val.json')
            if os.path.exists(stats_path):
                print('load stats file {}.'.format(stats_path))
                stats = pd.read_json(stats_path, 'records')
            else:
                print('no stats file.')
                if not os.path.exists(stats_dir):
                    os.makedirs(stats_dir)
                stats = pd.DataFrame(columns=['epoch', 'acc'])

            # val iterate over examples
            correct = 0

            while dataset.has_val_example:
                vgg, c3d, question, answer = dataset.get_val_example()
                feed_dict = {
                    model.appear: [vgg],
                    model.motion: [c3d],
                    model.question_encode: [question],
                }
                prediction = sess.run(model.prediction, feed_dict=feed_dict)
                prediction = prediction[0]
                if answerset[prediction] == answer:
                    correct += 1

            acc = correct / dataset.val_example_total
            print('\n[VAL] epoch {}, acc {:.5f}.\n'.format(epoch, acc))

            summary = tf.Summary()
            summary.value.add(tag='val/acc', simple_value=float(acc))
            summary_writer.add_summary(summary, epoch)

            record = Series([epoch, acc], ['epoch', 'acc'])
            stats = stats.append(record, ignore_index=True)
            stats.to_json(stats_path, 'records')

            dataset.reset_val()
            return acc

def predict(dataset,config,log_dir,id) :
    model_config = config['model']
    sess_config = config['session']
    answerset = pd.read_csv(
        os.path.join(config['preprocess_dir'], 'answer_set.txt'), header=None)[0]
    with tf.Graph().as_default():
        model = GRA(model_config)
        model.build_inference()

        with tf.Session(config=sess_config) as sess:
            ckpt_dir = os.path.join(log_dir, 'checkpoint')
            save_path = tf.train.latest_checkpoint(ckpt_dir)
            saver = tf.train.Saver()
            if save_path:
                print('load checkpoint {}.'.format(save_path))
                saver.restore(sess, save_path)
            else:
                print('no checkpoint.')
                exit()
            try:
                vgg, c3d, question, answer, question_raw = dataset.get_test_example_index(id)
                feed_dict = {
                    model.appear: [vgg],
                    model.motion: [c3d],
                    model.question_encode: [question],
                }
                prediction, channel_weight, appear_weight, motion_weight = sess.run(
                    [model.prediction, model.channel_weight, model.appear_weight, model.motion_weight],
                    feed_dict=feed_dict)
                prediction = prediction[0]
                print('question: '+question_raw)
                print('ground answer: '+answer)
                print('model answer: '+answerset[prediction])
                #return question_raw,answerset[prediction],answer

            except Exception as e:
                # print(dataset.test_example_idx)
                print("An error Occured "+str(e))
                #return None,N
                # if fail==20:
                # break
                pass

    pass
def test(dataset, config, log_dir):
    """Test model, output prediction as json file."""
    model_config = config['model']
    sess_config = config['session']

    answerset = pd.read_csv(
        os.path.join(config['preprocess_dir'], 'answer_set.txt'), header=None)[0]

    with tf.Graph().as_default():
        model = GRA(model_config)
        model.build_inference()

        with tf.Session(config=sess_config) as sess:
            ckpt_dir = os.path.join(log_dir, 'checkpoint')
            save_path = tf.train.latest_checkpoint(ckpt_dir)
            saver = tf.train.Saver()
            if save_path:
                print('load checkpoint {}.'.format(save_path))
                saver.restore(sess, save_path)
            else:
                print('no checkpoint.')
                exit()

            # test iterate over examples
            result = DataFrame(columns=['id', 'answer'])
            correct = 0
            fail = 0
            answered = 0
            while dataset.has_test_example:
                try :
                    vgg, c3d, question, answer, example_id = dataset.get_test_example()
                    feed_dict = {
                        model.appear: [vgg],
                        model.motion: [c3d],
                        model.question_encode: [question],
                    }
                    prediction,  channel_weight, appear_weight, motion_weight = sess.run(
                        [model.prediction, model.channel_weight, model.appear_weight, model.motion_weight], feed_dict=feed_dict)
                    prediction = prediction[0]
                    channel_weight = channel_weight[0]
                    appear_weight = appear_weight[0]
                    motion_weight = motion_weight[0]
                    answered = answered+1
                    result = result.append(
                        {'id': example_id, 'answer': answerset[prediction]}, ignore_index=True)
                    if answerset[prediction] == answer:
                        correct += 1
                        print(answer, example_id, channel_weight)
                        # print(appear_weight)
                        # print(motion_weight)

                except:
                    #print(dataset.test_example_idx)
                    dataset.test_example_idx += 1
                    fail = fail + 1
                    #if fail==20:
                        #break
                    pass

            result.to_json(os.path.join(
                log_dir, 'prediction.json'), 'records')
            print(correct)
            print(answered)
            print(fail)
            print(dataset.test_example_total)
            #acc = correct / (dataset.test_example_total - fail)
            acc =  correct / answered
            print('\n[TEST] acc {:.5f}.\n'.format(acc))

            dataset.reset_test()
            return acc


#!python run_gra.py --mode test --gpu 0 --log log/gra --dataset msvd_qa --config 0
def main():
    """Main script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True,
                        help='train/test/predict')
    parser.add_argument('--question',required=False)
    parser.add_argument('--gpu', required=True,
                        help='gpu id')
    parser.add_argument('--log', required=True,
                        help='log directory')
    parser.add_argument('--dataset', required=True,
                        help='dataset name, msvd_qa/msrvtt_qa')
    parser.add_argument('--config', required=True,
                        help='config id')
    args = parser.parse_args()

    config = cfg.get('gra', args.dataset, args.config, args.gpu)

    if args.dataset == 'msvd_qa':
        dataset = dt.MSVDQA(
            config['train']['batch_size'], config['preprocess_dir'])
    elif args.dataset == 'msrvtt_qa':
        dataset = dt.MSRVTTQA(
            config['train']['batch_size'], config['preprocess_dir'])

    if args.mode == 'train':
        best_val_acc = -1
        val_acc = 0
        not_improved = -1

        for epoch in range(18, 30):
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                not_improved = 0
            else:
                not_improved += 1
            if not_improved == 10:
                print('early stopping.')


            train(epoch, dataset, config, args.log)
            val_acc = val(epoch, dataset, config, args.log)

    elif args.mode == 'test':
        print('start test.')
        test(dataset, config, args.log)
    elif args.mode == 'predict':
        print('predicting '+args.question)
        predict(dataset,config,args.log,int(args.question))


if __name__ == '__main__':
    main()
