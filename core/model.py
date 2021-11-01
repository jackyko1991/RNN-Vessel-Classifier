import tensorflow as tf
from configs.config_loader import *
from core.CenterlineDataset import *
import multiprocessing
import numpy as np
import datetime
import os
import sys
import shutil
import math
import pandas as pd
from tqdm import tqdm

class CenterlineClassifier(object):
    def __init__(self,sess,config) -> None:
        """
        Args:
            sess: Tensorflow session
            config: Model configuration
        """
        super().__init__()

        self.sess = sess
        self.config = config

    def load_config(self):
        self.output_channel_num = len(self.config['TrainingSetting']['Data']['ClassNames'])

        self.batch_size = self.config['TrainingSetting']['BatchSize']
        self.image_log = self.config['TrainingSetting']['ImageLog']

        self.train_data_dir = self.config['TrainingSetting']['Data']['TrainingDataDirectory']
        self.test_data_dir = self.config['TrainingSetting']['Data']['TestingDataDirectory']
        self.testing = self.config['TrainingSetting']['Testing']

        self.centerline_filename = self.config['TrainingSetting']['Data']['CenterlineFilename']
        self.label_filename = self.config['TrainingSetting']['Data']['LabelFilename']
        self.case_column_name = self.config['TrainingSetting']['Data']['CaseColumnName']
        self.features = self.config['TrainingSetting']['Data']['Features']
        self.features_num = len(self.features)
        self.abscissas_array_name = self.config['TrainingSetting']['Data']["AbscissasArrayName"]
        self.centerlineids_array_name = self.config['TrainingSetting']['Data']["CenterlineIdsArrayName"]
        self.class_names = self.config['TrainingSetting']['Data']['ClassNames']
        self.class_weights = self.config['TrainingSetting']['Data']['Weights']

        self.restore_training = self.config['TrainingSetting']['Restore']
        self.log_dir = self.config['TrainingSetting']['LogDir']
        self.ckpt_dir = self.config['TrainingSetting']['CheckpointDir']

        self.epoches = self.config['TrainingSetting']['Epoches']
        self.max_steps = self.config['TrainingSetting']['MaxSteps']
        self.log_interval = self.config['TrainingSetting']['LogInterval']
        self.testing_step_interval = self.config['TrainingSetting']['TestingStepInterval']

        self.network_name = self.config['Network']['Name']
        self.network_dropout_rate = self.config['Network']['Dropout']
        self.network_hidden_states = self.config['Network']['HiddenStates']

        self.optimizer_name = self.config['TrainingSetting']['Optimizer']['Name']
        self.initial_learning_rate = self.config['TrainingSetting']['Optimizer']['InitialLearningRate']
        self.decay_factor = self.config['TrainingSetting']['Optimizer']['Decay']['Factor']
        self.decay_step = self.config['TrainingSetting']['Optimizer']['Decay']['Step']
        self.loss_fn = self.config['TrainingSetting']['LossFunction']['Name']
        self.classification_type = self.config['TrainingSetting']['LossFunction']['Multiclass/Multilabel']

        self.model_path = self.config['PredictionSetting']['ModelPath']
        self.checkpoint_path = self.config['PredictionSetting']['CheckPointPath']
        self.evaluation_data_dir = self.config['PredictionSetting']['Data']['EvaluationDataDirectory']
        self.report_output = self.config['PredictionSetting']['ReportOutput']
        self.evaluation_output_filename = self.config['PredictionSetting']['OutputFilename']
        self.evaluation_abscissas_array_name = self.config['PredictionSetting']['Data']["AbscissasArrayName"]
        self.evaluation_centerlineids_array_name = self.config['PredictionSetting']['Data']["CenterlineIdsArrayName"]
        self.evaluation_features = self.config['PredictionSetting']['Data']['Features']

    def dataset_iterator(self,data_dir,train=True):
        # Force input pipepline to CPU:0 to avoid operations sometimes ended up at GPU and resulting a slow down
        with tf.device('/cpu:0'):
            Dataset = CenterlineDataset(
                data_dir=data_dir,
                centerline_filename=self.centerline_filename,
                label_filename=self.label_filename,
                case_column_name=self.case_column_name,
                class_names=self.class_names,
                centerline_features=self.features,
                absciass_array_name=self.abscissas_array_name,
                centerlineids_array_name=self.centerlineids_array_name
            )
            dataset = Dataset.get_dataset()
            dataset = dataset.shuffle(buffer_size=multiprocessing.cpu_count())
            dataset = dataset.padded_batch(self.batch_size, padded_shapes=([None,None],[],[None]),drop_remainder=False)
            dataset = dataset.prefetch(1)
                
        return dataset.make_initializable_iterator()

    def build_model_graph(self):
        self.global_step = tf.train.get_or_create_global_step()

        # create placeholder for data input
        with tf.name_scope("placeholder"):
            self.input_placeholder = tf.placeholder(tf.float32, [None,None,self.features_num],name="input_placeholder")
            self.output_placeholder = tf.placeholder(tf.float32, [None, self.output_channel_num],name="output_placeholder")
            self.sequence_length_placeholder = tf.placeholder(tf.int32,[None],name="sequence_length_placeholder")
            self.dropout_placeholder = tf.placeholder(tf.float32, name='dropout')
            self.is_training_placeholder = tf.placeholder(tf.bool, name='is_training')

        # plot input in tensorboard
        # https://stackoverflow.com/questions/38543850/how-to-display-custom-images-in-tensorboard-e-g-matplotlib-plots

        # get input and output datasets
        with tf.name_scope("dataset"):
            self.train_iterator = self.dataset_iterator(self.train_data_dir)
            self.next_element_train = self.train_iterator.get_next()

            if self.testing:
                self.test_iterator = self.dataset_iterator(self.test_data_dir)
                self.next_element_test = self.test_iterator.get_next()

        # network models
        print("{}: Network: {}".format(datetime.datetime.now(),self.network_name))

        with tf.name_scope("RNN"):
            self.outputs_op, self.state_op =tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(self.network_hidden_states),
                self.input_placeholder,
                dtype=tf.float32,
                sequence_length=self.sequence_length_placeholder
            )

            # put time axis to first rank and select last frame
            self.last_output_op = self.state_op.h
            dense = tf.layers.dense(self.last_output_op,units=10,activation="relu")
            dense = tf.nn.dropout(dense, rate=self.dropout_placeholder)
            dense = tf.layers.batch_normalization(dense, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.is_training_placeholder)
            logits_op = tf.layers.dense(dense,units=self.output_channel_num,activation=None)

        with tf.name_scope("loss"):
            if self.classification_type == "Multilabel" or self.output_channel_num == 1:
                self.prob_op = tf.sigmoid(logits_op)
                loss_op = tf.losses.log_loss(labels=self.output_placeholder,predictions=self.prob_op[:,0])
            elif self.classification_type == "Multiclass":
                self.prob_op = tf.nn.softmax(logits_op)
                self.result_op = tf.cast(tf.one_hot(tf.argmax(self.prob_op,1),self.output_channel_num),dtype=tf.uint8)

                # self.loss_op = tf.nn.softmax_cross_entropy_with_logits(logits=logits_op, labels=self.output_placeholder)
                # self.loss_op = tf.reduce_mean(self.loss_op,0)

                # class weights
                class_weights = tf.constant([self.class_weights])

                # deduce weights for batch samples based on their true label
                weights = tf.reduce_sum(class_weights * self.output_placeholder, axis=1)

                # compute your (unweighted) softmax cross entropy loss
                unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits_op, labels=self.output_placeholder)
                # apply the weights, relying on broadcasting of the multiplication
                weighted_losses = unweighted_losses * weights

                # reduce the result to get your final loss
                self.avg_loss_op = tf.reduce_mean(weighted_losses,0) # for multiclass classification softmax loss is used, so no need to take average loss across class
            else:
                exit("Classification type can only be \"Multiclass\" or \"Multilabel\", training abort")

        tf.summary.scalar('loss/average', self.avg_loss_op)

        # performance metrics
        self.class_tp = []
        self.class_tn = []
        self.class_fp = []
        self.class_fn = []
        self.class_auc = []
        class_accuracy = []
        class_precision = []
        class_sensitivity = []
        class_specificity = []

        for i, class_name in enumerate(self.class_names):
            epsilon = 1e-6
            with tf.variable_scope("metrics/{}".format(class_name),reuse=True):
                class_acc, class_acc_op = tf.metrics.accuracy(labels=tf.cast(self.output_placeholder[:,i],dtype=tf.uint8), predictions=tf.cast(self.result_op[:,i],dtype=tf.uint8))
                class_tp, class_tp_op = tf.metrics.true_positives(labels=tf.cast(self.output_placeholder[:,i],dtype=tf.uint8), predictions=tf.cast(self.result_op[:,i],dtype=tf.uint8))
                class_tn, class_tn_op = tf.metrics.true_negatives(labels=tf.cast(self.output_placeholder[:,i],dtype=tf.uint8), predictions=tf.cast(self.result_op[:,i],dtype=tf.uint8))
                class_fp, class_fp_op = tf.metrics.false_positives(labels=tf.cast(self.output_placeholder[:,i],dtype=tf.uint8), predictions=tf.cast(self.result_op[:,i],dtype=tf.uint8))
                class_fn, class_fn_op = tf.metrics.false_negatives(labels=tf.cast(self.output_placeholder[:,i],dtype=tf.uint8), predictions=tf.cast(self.result_op[:,i],dtype=tf.uint8))
                precision = class_tp_op/(class_tp_op+class_fp_op+epsilon)
                sensitivity = class_tp_op/(class_tp_op+class_fn_op+epsilon)
                specificity = class_tn_op/(class_tn_op+class_fp_op+epsilon)

                # # remove nan values
                # nan_replace_value = 1.0
                # accuracy = tf.where(tf.is_nan(class_acc_op), tf.ones_like(class_acc_op) * nan_replace_value, class_acc_op)
                # precision = tf.where(tf.is_nan(precision), tf.ones_like(precision) * nan_replace_value, precision)
                # sensitivity = tf.where(tf.is_nan(sensitivity), tf.ones_like(sensitivity) * nan_replace_value, sensitivity)
                # specificity = tf.where(tf.is_nan(specificity), tf.ones_like(specificity) * nan_replace_value, specificity)

                accuracy = class_acc_op

                class_auc, class_auc_op = tf.metrics.auc(labels=self.output_placeholder[:,i], predictions=self.prob_op[:,i])

                self.class_tp.append(class_tp_op)
                self.class_tn.append(class_tn_op)
                self.class_fp.append(class_fp_op)
                self.class_fn.append(class_fn_op)
                self.class_auc.append(class_auc_op)
                class_accuracy.append(accuracy)
                class_precision.append(precision)
                class_sensitivity.append(sensitivity)
                class_specificity.append(specificity)

            if self.classification_type == "Multilabel":
                tf.summary.scalar('loss/' + class_name, self.loss_op[i])
            tf.summary.scalar('accuracy/' + class_name, accuracy)
            tf.summary.scalar('precision/' + class_name, precision)
            tf.summary.scalar('sensitivity/' + class_name, sensitivity)
            tf.summary.scalar('specificity/' + class_name, specificity)
            tf.summary.scalar('auc/' + class_name, class_auc_op)

        with tf.variable_scope("metrics/average",reuse=True):
            avg_accuracy = tf.reduce_mean(class_accuracy)
            avg_precision = tf.reduce_mean(class_precision)
            avg_sensitivity = tf.reduce_mean(class_sensitivity)
            avg_specificity = tf.reduce_mean(class_specificity)
            avg_auc = tf.reduce_mean(self.class_auc)
        tf.summary.scalar('accuracy/average',avg_accuracy)
        tf.summary.scalar('precision/average', avg_precision)
        tf.summary.scalar('sensitivity/average', avg_sensitivity)
        tf.summary.scalar('specificity/average', avg_specificity)
        tf.summary.scalar('auc/average', avg_auc)

        with tf.variable_scope("metrics/",reuse=True):
            self.acc_op = avg_accuracy
            self.precision_op = avg_precision
            self.sensitivity_op = avg_sensitivity
            self.specificity_op = avg_specificity
            self.auc_op = avg_auc

        # learning rate
        with tf.name_scope("learning_rate"):
            self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, self.global_step,
                self.decay_step, self.decay_factor, staircase=False, name="learning_rate")
        tf.summary.scalar('learning_rate', self.learning_rate)

        # optimizer
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5)
            update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = optimizer.minimize(
                loss=self.avg_loss_op,
                global_step=tf.train.get_or_create_global_step()
                )
            self.train_op = tf.group([train_op, update_ops])
        
    def train(self):
        self.load_config()

        """train the classifier"""
        self.build_model_graph()

        # start epoch
        start_epoch = tf.get_variable("start_epoch", shape=[1], initializer=tf.zeros_initializer, dtype=tf.int32)
        start_epoch_inc = start_epoch.assign(start_epoch+1)

        # Initialize all variables
        self.sess.run(tf.initializers.global_variables())
        print("{}: Start training...".format(datetime.datetime.now()))

        # saver
        print("{}: Setting up Saver...".format(datetime.datetime.now()))
        checkpoint_prefix = os.path.join(self.ckpt_dir,"checkpoint")
        if self.restore_training:
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=5)

            # check if checkpoint exists
            if os.path.exists(checkpoint_prefix+"-latest"):
                print("{}: Last checkpoint found at {}, loading...".format(datetime.datetime.now(),self.ckpt_dir))
                latest_checkpoint_path = tf.train.latest_checkpoint(self.ckpt_dir,latest_filename="checkpoint-latest")
                saver.restore(self.sess, latest_checkpoint_path)
                print("{}: Restore check point at {} success".format(datetime.datetime.now(),self.ckpt_dir))
            
            print("{}: Last checkpoint epoch: {}".format(datetime.datetime.now(),start_epoch.eval(session=self.sess)[0]))
            print("{}: Last checkpoint global step: {}".format(datetime.datetime.now(),tf.train.global_step(self.sess, self.global_step)))
        else:
            # clear log directory
            if os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir)
            os.makedirs(self.log_dir)

            # clear checkpoint directory
            if os.path.exists(self.ckpt_dir):
                shutil.rmtree(self.ckpt_dir)
            os.makedirs(self.ckpt_dir)
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=5)
            checkpoint_prefix = os.path.join(self.ckpt_dir,"checkpoint")

        # tensorboard summary writer
        summary_op = tf.summary.merge_all()

        train_summary_writer = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
        if self.testing:
            test_summary_writer = tf.summary.FileWriter(self.log_dir + '/test', self.sess.graph)

                # testing initializer need to execute outside training loop
        if self.testing:
            self.sess.run(self.test_iterator.initializer)

        # loop over epoches
        for epoch in np.arange(self.epoches):
            print("================================================================================================================")
            print("{}: Epoch {} starts...".format(datetime.datetime.now(),epoch+1))
            
            # initialize iterator in each new epoch
            self.sess.run(self.train_iterator.initializer)
            
            # training phase
            while True:
                if self.global_step.eval() > self.max_steps:
                    print("{}: Reach maximum training steps".format(datetime.datetime.now()))
                    print("{}: Saving checkpoint of step {} at {}...".format(datetime.datetime.now(),self.global_step.eval()-1,self.ckpt_dir))
                    if not (os.path.exists(self.ckpt_dir)):
                        os.makedirs(self.ckpt_dir,exist_ok=True)
                    saver.save(self.sess, checkpoint_prefix, 
                    	global_step=tf.train.global_step(self.sess, self.global_step-1),
                    	latest_filename="checkpoint-latest")
                    sys.exit(0)
                try:
                    self.sess.run(tf.initializers.local_variables())
                    series, seq_len, label = self.sess.run(self.next_element_train)
                    # swap axes 1 and 2 to fit conventional rnn input
                    series = np.swapaxes(series, axis1=1,axis2=2)

                    outputs, states, last_output, prob, result, loss, accuracy, precision, sensitivity, specificity, auc,_ = \
                        self.sess.run([self.outputs_op,self.state_op, self.last_output_op,self.prob_op, self.result_op, self.avg_loss_op, self.acc_op, self.precision_op, self.sensitivity_op, self.specificity_op, self.auc_op, self.train_op], 
                        feed_dict={
                            self.input_placeholder: series, 
                            self.output_placeholder: label, 
                            self.sequence_length_placeholder: seq_len,
                            self.dropout_placeholder: self.network_dropout_rate,
                            self.is_training_placeholder: True})
                    # print("{}: last output: {}\n".format(datetime.datetime.now(),last_output[:5,:]))
                    print("{}: ground truth: {}\n".format(datetime.datetime.now(),label[:5]))
                    print("{}: result: {}\n".format(datetime.datetime.now(),result[:5]))
                    print("{}: probability: {}\n".format(datetime.datetime.now(),prob[:5,:]))
                    print("{}: Training loss: {:.4f}".format(datetime.datetime.now(),loss))
                    print("{}: Training accuracy: {:.4f}".format(datetime.datetime.now(),accuracy))
                    print("{}: Training precision: {:.4f}".format(datetime.datetime.now(),precision))
                    print("{}: Training sensitivity: {:.4f}".format(datetime.datetime.now(),sensitivity))
                    print("{}: Training specificity: {:.4f}".format(datetime.datetime.now(),specificity))
                    print("{}: Training auc: {:.4f}".format(datetime.datetime.now(),auc))

                    # perform summary log after training op
                    summary = self.sess.run(summary_op,feed_dict={
                        self.input_placeholder: series,
                        self.output_placeholder: label,
                        self.sequence_length_placeholder: seq_len,
                        self.dropout_placeholder: 0.0,
                        self.is_training_placeholder: False
                        })

                    train_summary_writer.add_summary(summary,global_step=tf.train.global_step(self.sess,self.global_step))
                    train_summary_writer.flush()

                    # save checkpoint
                    if self.global_step.eval()%self.log_interval == 0:
                        print("{}: Saving checkpoint of step {} at {}...".format(datetime.datetime.now(),self.global_step.eval(),self.ckpt_dir))
                        if not (os.path.exists(self.ckpt_dir)):
                            os.makedirs(self.ckpt_dir,exist_ok=True)
                        saver.save(self.sess, checkpoint_prefix, 
                            global_step=tf.train.global_step(self.sess, self.global_step),
                            latest_filename="checkpoint-latest")

                    # testing phase
                    if self.testing and (self.global_step.eval()%self.testing_step_interval == 0):
                        print("************************ Testing at step {} ************************".format(datetime.datetime.now(),self.global_step.eval()))
                        # self.network.is_training = False
                        try:
                            series, seq_len, label = self.sess.run(self.next_element_test)
                        except tf.errors.OutOfRangeError:
                            self.sess.run(self.test_iterator.initializer)
                            series, seq_len, label = self.sess.run(self.next_element_test)
                        # swap axes 1 and 2 to fit conventional rnn input
                        series = np.swapaxes(series, axis1=1,axis2=2)
                            
                        # zero padding
                        if series.shape[0] < self.batch_size:
                            # 	if self.dimension == 2:
                            # 		images_zero_pads = np.zeros((self.batch_size-images.shape[0],images.shape[1],images.shape[2],images.shape[3]))
                            # 		label_zero_pads = np.zeros((self.batch_size-label.shape[0],images.shape[1]))
                            # 	else:
                            # 		images_zero_pads = np.zeros((self.batch_size-images.shape[0],images.shape[1],images.shape[2],images.shape[3],images.shape[4]))
                                
                            # label_zero_pads = np.zeros((self.batch_size-label.shape[0],label.shape[1]))
                            # images = np.concatenate((images,images_zero_pads))
                            # label = np.concatenate((label,label_zero_pads))

                            series = np.tile(series,(math.ceil(self.batch_size/series.shape[0]),1,1))
                            seq_len = np.tile(seq_len,(math.ceil(self.batch_size/seq_len.shape[0]),1))
                            label = np.tile(label,(math.ceil(self.batch_size/label.shape[0])))

                            series = series[:self.batch_size,]
                            seq_len = seq_len[:self.batch_size,]
                            label = label[:self.batch_size,]

                        # print("{}: Testing loss: {:.4f}".format(datetime.datetime.now(),loss))
                        # print("{}: Testing accuracy: {:.4f}".format(datetime.datetime.now(),accuracy))
                        # print("{}: Testing precision: {:.4f}".format(datetime.datetime.now(),precision))
                        # print("{}: Testing sensitivity: {:.4f}".format(datetime.datetime.now(),sensitivity))
                        # print("{}: Testing specificity: {:.4f}".format(datetime.datetime.now(),specificity))
                        # print("{}: Testing auc: {:.4f}".format(datetime.datetime.now(),auc))
                        # print("{}: Testing ground truth: \n{}".format(datetime.datetime.now(),label[:5]))
                        # print("{}: Testing result: \n{}".format(datetime.datetime.now(),result[:5]))
                        # print("{}: Testing prob: \n{}".format(datetime.datetime.now(),prob[:5]))

                        outputs, states, last_output, prob, result, loss, accuracy, precision, sensitivity, specificity, auc, summary = \
                            self.sess.run([self.outputs_op,self.state_op, self.last_output_op,self.prob_op, self.result_op, self.avg_loss_op, self.acc_op, self.precision_op, self.sensitivity_op, self.specificity_op, self.auc_op,summary_op], 
                            feed_dict={
                                self.input_placeholder: series, 
                                self.output_placeholder: label, 
                                self.sequence_length_placeholder: seq_len, 
                                self.dropout_placeholder:0.0,
                                self.is_training_placeholder: False})
                        # print("last output: {}".format(last_output[:5,:]))
                        print("{}: result: {}\n".format(datetime.datetime.now(),result[:5]))
                        print("{}: probability: {}\n".format(datetime.datetime.now(),prob[:5,:]))
                        print("{}: Testing loss: {:.4f}".format(datetime.datetime.now(),loss))
                        print("{}: Testing accuracy: {:.4f}".format(datetime.datetime.now(),accuracy))
                        print("{}: Testing precision: {:.4f}".format(datetime.datetime.now(),precision))
                        print("{}: Testing sensitivity: {:.4f}".format(datetime.datetime.now(),sensitivity))
                        print("{}: Testing specificity: {:.4f}".format(datetime.datetime.now(),specificity))
                        print("{}: Testing auc: {:.4f}".format(datetime.datetime.now(),auc))

                        test_summary_writer.add_summary(summary, global_step=tf.train.global_step(self.sess, self.global_step))
                        test_summary_writer.flush()

                except tf.errors.OutOfRangeError:
                    start_epoch_inc.op.run()

                    print("{}: Saving checkpoint of epoch {} at {}...".format(datetime.datetime.now(),epoch+1,self.ckpt_dir))
                    if not (os.path.exists(self.ckpt_dir)):
                        os.makedirs(self.ckpt_dir,exist_ok=True)
                    saver.save(self.sess, checkpoint_prefix, 
                        global_step=tf.train.global_step(self.sess, self.global_step),
                        latest_filename="checkpoint-latest")
                    print("{}: Saving checkpoint succeed".format(datetime.datetime.now()))

                    break

        # close tensorboard summary writer
        train_summary_writer.close()
        if self.testing:
            test_summary_writer.close()

    def predict(self):
        # read config to class variables
        self.load_config()

        # restore model grpah
        # tf.reset_default_graph()
        imported_meta = tf.train.import_meta_graph(self.model_path)

        print("{}: Start evaluation...".format(datetime.datetime.now()))

        imported_meta.restore(self.sess, self.checkpoint_path)
        print("{}: Restore checkpoint success".format(datetime.datetime.now()))

        # create outputs dictionary
        outputs = {'case': []}
        for class_name in self.class_names:
            outputs[class_name] = []

        pbar = tqdm(os.listdir(self.evaluation_data_dir))

        for case in pbar:
            pbar.set_description(case)

            # check data exists
            file_path = os.path.join(self.evaluation_data_dir, case, self.config['PredictionSetting']['Data']['CenterlineFilename'])
            if not os.path.exists(file_path):
                print("{}: Centerline file not found at {}".format(datetime.datetime.now(),file_path))
                break

            # read centerline file
            centerline = load_centerline_data(
                file_path,
                abscissas_array_name=self.evaluation_abscissas_array_name,
                centerlineids_array_name=self.evaluation_centerlineids_array_name,
                features=self.evaluation_features
                )

            series = np.array([np.array(x) for x in centerline.values])
            # convert image to numpy array
            seq_len = centerline.shape[0]

            # swap axes 0 and 1 to fit conventional rnn input
            series = np.swapaxes(series, axis1=1,axis2=0)

            # expand batch dim
            series = np.expand_dims(series, axis=0)
            seq_len = np.expand_dims(seq_len, axis=0)

            if self.classification_type == "Multilabel" or self.output_channel_num == 1:
                prob = self.sess.run(['loss/Sigmoid:0'], feed_dict={
                            'placeholder/input_placeholder:0': series,
                            'placeholder/sequence_length_placeholder:0': seq_len,
                            'placeholder/dropout:0':0.0,
                            'placeholder/is_training:0':False})
            elif self.classification_type == "Multiclass":
                prob = self.sess.run(['loss/Softmax:0'], feed_dict={
                            'placeholder/input_placeholder:0': series,
                            'placeholder/sequence_length_placeholder:0': seq_len,
                            'placeholder/dropout:0':0.0,
                            'placeholder/is_training:0':False})

            outputs['case'].append(case)
            for class_name, prob_ in zip(self.class_names, prob[0][0]): # prob[0][0] refers to first batch result, will optimize for larger batch later
                outputs[class_name].append(prob_)


        outputs_df = pd.DataFrame(outputs)
        print(outputs_df)

        # output report function, to be developed
        # if self.report_output:
            # doc = report.Report(images=images,result=prob[0],class_names=self.class_names)
            # doc.WritePdf(os.path.join(self.evaluation_data_dir,case,"report"))

        # write csv
        outputs_df.to_csv(self.evaluation_output_filename,index=False)