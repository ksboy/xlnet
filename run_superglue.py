from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
from absl import flags
import os
import sys
import csv
import collections
import numpy as np
import time
import math
import json
import random
from copy import copy
from collections import defaultdict as dd

import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf

import sentencepiece as spm

from data_utils import SEP_ID, VOCAB_SIZE, CLS_ID
import model_utils
import function_builder
from classifier_utils import PaddingInputExample
from classifier_utils import convert_single_example
from prepro_utils import preprocess_text, encode_ids

# Model
flags.DEFINE_string("model_config_path", default=None,
                    help="Model config path.")
flags.DEFINE_float("dropout", default=0.1,
                   help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
                   help="Attention dropout rate.")
flags.DEFINE_integer("clamp_len", default=-1,
                     help="Clamp length")
flags.DEFINE_string("summary_type", default="last",
                    help="Method used to summarize a sequence into a compact vector.")
flags.DEFINE_bool("use_summ_proj", default=True,
                  help="Whether to use projection for summarizing sequences.")
flags.DEFINE_bool("use_bfloat16", False,
                  help="Whether to use bfloat16.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
                  enum_values=["normal", "uniform"],
                  help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
                   help="Initialization std when init is uniform.")

# I/O paths
flags.DEFINE_bool("overwrite_data", default=False,
                  help="If False, will use cached data if available.")
flags.DEFINE_string("init_checkpoint", default=None,
                    help="checkpoint path for initializing the model. "
                         "Could be a pretrained model or a finetuned model.")
flags.DEFINE_string("output_dir", default="",
                    help="Output dir for TF records.")
flags.DEFINE_string("spiece_model_file", default="",
                    help="Sentence Piece model path.")
flags.DEFINE_string("model_dir", default="",
                    help="Directory for saving the finetuned model.")
flags.DEFINE_string("data_dir", default="",
                    help="Directory for input data.")

# TPUs and machines
flags.DEFINE_bool("use_tpu", default=False, help="whether to use TPU.")
flags.DEFINE_integer("num_hosts", default=1, help="How many TPU hosts.")
flags.DEFINE_integer("num_core_per_host", default=8,
                     help="8 for TPU v2 and v3-8, 16 for larger TPU v3 pod. In the context "
                          "of GPU training, it refers to the number of GPUs used.")
flags.DEFINE_string("tpu_job_name", default=None, help="TPU worker job name.")
flags.DEFINE_string("tpu", default=None, help="TPU name.")
flags.DEFINE_string("tpu_zone", default=None, help="TPU zone.")
flags.DEFINE_string("gcp_project", default=None, help="gcp project.")
flags.DEFINE_string("master", default=None, help="master")
flags.DEFINE_integer("iterations", default=1000,
                     help="number of iterations per TPU training loop.")

# training
flags.DEFINE_bool("do_train", default=False, help="whether to do training")
flags.DEFINE_integer("train_steps", default=1000,
                     help="Number of training steps")
flags.DEFINE_integer("warmup_steps", default=0, help="number of warmup steps")
flags.DEFINE_float("learning_rate", default=1e-5, help="initial learning rate")
flags.DEFINE_float("lr_layer_decay_rate", 1.0,
                   "Top layer: lr[L] = FLAGS.learning_rate."
                   "Low layer: lr[l-1] = lr[l] * lr_layer_decay_rate.")
flags.DEFINE_float("min_lr_ratio", default=0.0,
                   help="min lr ratio for cos decay.")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping")
flags.DEFINE_integer("max_save", default=0,
                     help="Max number of checkpoints to save. Use 0 to save all.")
flags.DEFINE_integer("save_steps", default=None,
                     help="Save the model for every save_steps. "
                          "If None, not to save any model.")
flags.DEFINE_integer("train_batch_size", default=8,
                     help="Batch size for training")
flags.DEFINE_float("weight_decay", default=0.00, help="Weight decay rate")
flags.DEFINE_float("adam_epsilon", default=1e-8, help="Adam epsilon")
flags.DEFINE_string("decay_method", default="poly", help="poly or cos")

# evaluation
flags.DEFINE_bool("do_eval", default=False, help="whether to do eval")
flags.DEFINE_bool("do_predict", default=False, help="whether to do prediction")
flags.DEFINE_float("predict_threshold", default=0,
                   help="Threshold for binary prediction.")
flags.DEFINE_string("eval_split", default="dev", help="could be dev or test")
flags.DEFINE_integer("eval_batch_size", default=128,
                     help="batch size for evaluation")
flags.DEFINE_integer("predict_batch_size", default=128,
                     help="batch size for prediction.")
flags.DEFINE_string("predict_dir", default=None,
                    help="Dir for saving prediction files.")
flags.DEFINE_bool("eval_all_ckpt", default=False,
                  help="Eval all ckpts. If False, only evaluate the last one.")
flags.DEFINE_string("predict_ckpt", default=None,
                    help="Ckpt path for do_predict. If None, use the last one.")

# task specific
flags.DEFINE_string("task_name", default=None, help="Task name")
flags.DEFINE_integer("max_seq_length", default=128, help="Max sequence length")
flags.DEFINE_integer("shuffle_buffer", default=2048,
                     help="Buffer size used for shuffle.")
flags.DEFINE_integer("num_passes", default=1,
                     help="Num passes for processing training data. "
                          "This is use to batch data without loss for TPUs.")
flags.DEFINE_bool("uncased", default=False,
                  help="Use uncased.")
flags.DEFINE_string("cls_scope", default=None,
                    help="Classifier layer scope.")
flags.DEFINE_bool("is_regression", default=False,
                  help="Whether it's a regression task.")

FLAGS = flags.FLAGS


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file, quotechar=None):
        lines = []
        for line in open(input_file, "r", encoding="utf-8"):
            ine = json.loads(line)
            lines.append(line)
        return lines


class SuperGLUEProcessor(DataProcessor):
    def __init__(self):
        self.train_file = "train.jsonl"
        self.dev_file = "val.jsonl"
        self.test_file = "test.jsonl"
        self.label_key = "label"
        self.text_a_key = None
        self.text_b_key = None
        self.test_text_a_key = None
        self.test_text_b_key = None

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, self.train_file)), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, self.dev_file)), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        if self.test_text_a_key is None:
            self.test_text_a_key = self.text_a_key
        if self.test_text_b_key is None:
            self.test_text_b_key = self.text_b_key

        return self._create_examples(
            self._read_json(os.path.join(data_dir, self.test_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)

            a_key = (self.text_a_key if set_type != "test" else
                     self.test_text_a_key)
            b_key = (self.text_b_key if set_type != "test" else
                     self.test_text_b_key)

            # there are some incomplete lines in QNLI
            text_a = line[a_key]

            if b_key is not None:
                text_b = line[b_key]
            else:
                text_b = None

            if set_type == "test":
                label = self.get_labels()[0]
            else:
                label = line[self.label_key]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class CbProcessor(SuperGLUEProcessor):
    def __init__(self):
        super(CbProcessor, self).__init__()
        self.text_a_key = "premise"
        self.text_b_column = "hypothesis"

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]


class RteProcessor(SuperGLUEProcessor):
    def __init__(self):
        super(RteProcessor, self).__init__()
        self.text_a_key = "premise"
        self.text_b_column = "hypothesis"

    def get_labels(self):
        return ["entailment", "not_entailment"]


class CopaProcessor(SuperGLUEProcessor):
    def __init__(self):
        super(CopaProcessor, self).__init__()
        self.text_a_key = "premise"

    def get_labels(self):
        return ["causality", "not_causality"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if set_type == "test":
                label = self.get_labels()[0]
            else:
                label = line[self.label_key]
            if int(label) == 0:
                label_0 = "causality"
                label_1 = "not_causality"
            else:
                label_0 = "causality"
                label_1 = "not_causality"

            if line["question"] == "cause":
                question = "What was the cause of this?"
            else:
                question = "What happened as a result?"
            text_a = line["premise"] + " <sep> " + question

            # 1: text_a = line["question"] + " [SEP] " + line["premise"]
            # 2: text_a = (line["question"]+' ')*10 + " [SEP] " + line["premise"]
            # 3: text_a = question + " [SEP] "+ line["premise"]
            # 4: text_a = line["premise"] + " [SEP] "+ question
            # 5: text_b = line["choice1"]+" [SEP] "+line["choice2"]

            guid = "%s-%s" % (set_type, str(line["idx"]) + "_0")
            text_b = line["choice1"]
            # print(guid, text_a ,text_b, label_0)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label_0))

            guid = "%s-%s" % (set_type, str(line["idx"]) + "_1")
            text_b = line["choice2"]
            # print(guid, text_a ,text_b, label_1)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label_1))

        return examples


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenize_fn, output_file,
        num_passes=1):
    """Convert a set of `InputExample`s to a TFRecord file."""

    # do not create duplicated records
    if tf.gfile.Exists(output_file) and not FLAGS.overwrite_data:
        tf.logging.info("Do not overwrite tfrecord {} exists.".format(output_file))
        return

    tf.logging.info("Create new tfrecord {}.".format(output_file))

    writer = tf.python_io.TFRecordWriter(output_file)

    if num_passes > 1:
        examples *= num_passes

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example {} of {}".format(ex_index,
                                                              len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenize_fn)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_float_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_float_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        if label_list is not None:
            features["label_ids"] = create_int_feature([feature.label_id])
        else:
            features["label_ids"] = create_float_feature([float(feature.label_id)])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.float32),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }
    if FLAGS.is_regression:
        name_to_features["label_ids"] = tf.FixedLenFeature([], tf.float32)

    tf.logging.info("Input tfrecord file {}".format(input_file))

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t

        return example

    def input_fn(params, input_context=None):
        """The actual input function."""
        if FLAGS.use_tpu:
            batch_size = params["batch_size"]
        elif is_training:
            batch_size = FLAGS.train_batch_size
        elif FLAGS.do_eval:
            batch_size = FLAGS.eval_batch_size
        else:
            batch_size = FLAGS.predict_batch_size

        d = tf.data.TFRecordDataset(input_file)
        # Shard the dataset to difference devices
        if input_context is not None:
            tf.logging.info("Input pipeline id %d out of %d",
                            input_context.input_pipeline_id, input_context.num_replicas_in_sync)
            d = d.shard(input_context.num_input_pipelines,
                        input_context.input_pipeline_id)

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = d.shuffle(buffer_size=FLAGS.shuffle_buffer)
            d = d.repeat()

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def get_model_fn(n_class):
    def model_fn(features, labels, mode, params):
        #### Training or Evaluation
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        #### Get loss from inputs
        if FLAGS.is_regression:
            (total_loss, per_example_loss, logits
             ) = function_builder.get_regression_loss(FLAGS, features, is_training)
        else:
            (total_loss, per_example_loss, logits
             ) = function_builder.get_classification_loss(
                FLAGS, features, n_class, is_training)

        #### Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        #### load pretrained models
        scaffold_fn = model_utils.init_from_checkpoint(FLAGS)

        #### Evaluation mode
        if mode == tf.estimator.ModeKeys.EVAL:
            assert FLAGS.num_hosts == 1

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                eval_input_dict = {
                    'labels': label_ids,
                    'predictions': predictions,
                    'weights': is_real_example
                }
                accuracy = tf.metrics.accuracy(**eval_input_dict)

                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    'eval_accuracy': accuracy,
                    'eval_loss': loss}

            def regression_metric_fn(
                    per_example_loss, label_ids, logits, is_real_example):
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                pearsonr = tf.contrib.metrics.streaming_pearson_correlation(
                    logits, label_ids, weights=is_real_example)
                return {'eval_loss': loss, 'eval_pearsonr': pearsonr}

            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)

            #### Constucting evaluation TPUEstimatorSpec with new cache.
            label_ids = tf.reshape(features['label_ids'], [-1])

            if FLAGS.is_regression:
                metric_fn = regression_metric_fn
            else:
                metric_fn = metric_fn
            metric_args = [per_example_loss, label_ids, logits, is_real_example]

            if FLAGS.use_tpu:
                eval_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metrics=(metric_fn, metric_args),
                    scaffold_fn=scaffold_fn)
            else:
                eval_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metric_ops=metric_fn(*metric_args))

            return eval_spec

        elif mode == tf.estimator.ModeKeys.PREDICT:
            label_ids = tf.reshape(features["label_ids"], [-1])

            predictions = {
                "logits": logits,
                "labels": label_ids,
                "is_real": features["is_real_example"]
            }

            if FLAGS.use_tpu:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
            else:
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode, predictions=predictions)
            return output_spec

        #### Configuring the optimizer
        train_op, learning_rate, _ = model_utils.get_train_op(FLAGS, total_loss)

        monitor_dict = {}
        monitor_dict["lr"] = learning_rate

        #### Constucting training TPUEstimatorSpec with new cache.
        if FLAGS.use_tpu:
            #### Creating host calls
            if not FLAGS.is_regression:
                label_ids = tf.reshape(features['label_ids'], [-1])
                predictions = tf.argmax(logits, axis=-1, output_type=label_ids.dtype)
                is_correct = tf.equal(predictions, label_ids)
                accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

                monitor_dict["accuracy"] = accuracy

                host_call = function_builder.construct_scalar_host_call(
                    monitor_dict=monitor_dict,
                    model_dir=FLAGS.model_dir,
                    prefix="train/",
                    reduce_fn=tf.reduce_mean)
            else:
                host_call = None

            train_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, loss=total_loss, train_op=train_op, host_call=host_call,
                scaffold_fn=scaffold_fn)
        else:
            train_spec = tf.estimator.EstimatorSpec(
                mode=mode, loss=total_loss, train_op=train_op)

        return train_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    #### Validate flags
    if FLAGS.save_steps is not None:
        FLAGS.iterations = min(FLAGS.iterations, FLAGS.save_steps)

    if FLAGS.do_predict:
        predict_dir = FLAGS.predict_dir
        if not tf.gfile.Exists(predict_dir):
            tf.gfile.MakeDirs(predict_dir)

    processors = {
        "cb": CbProcessor,
        "copa": CopaProcessor,
        'rte': RteProcessor
    }

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval, `do_predict` or "
            "`do_submit` must be True.")

    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels() if not FLAGS.is_regression else None

    sp = spm.SentencePieceProcessor()
    sp.Load(FLAGS.spiece_model_file)

    def tokenize_fn(text):
        text = preprocess_text(text, lower=FLAGS.uncased)
        return encode_ids(sp, text)

    run_config = model_utils.configure_tpu(FLAGS)

    model_fn = get_model_fn(len(label_list) if label_list is not None else None)

    spm_basename = os.path.basename(FLAGS.spiece_model_file)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    if FLAGS.use_tpu:
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            predict_batch_size=FLAGS.predict_batch_size,
            eval_batch_size=FLAGS.eval_batch_size)
    else:
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config)

    if FLAGS.do_train:
        train_file_base = "{}.len-{}.train.tf_record".format(
            spm_basename, FLAGS.max_seq_length)
        train_file = os.path.join(FLAGS.output_dir, train_file_base)
        tf.logging.info("Use tfrecord file {}".format(train_file))

        train_examples = processor.get_train_examples(FLAGS.data_dir)
        np.random.shuffle(train_examples)
        tf.logging.info("Num of train samples: {}".format(len(train_examples)))

        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenize_fn,
            train_file, FLAGS.num_passes)

        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)

        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

    if FLAGS.do_eval or FLAGS.do_predict:
        if FLAGS.eval_split == "dev":
            eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        else:
            eval_examples = processor.get_test_examples(FLAGS.data_dir)

        tf.logging.info("Num of eval samples: {}".format(len(eval_examples)))

    if FLAGS.do_eval:
        # TPU requires a fixed batch size for all batches, therefore the number
        # of examples must be a multiple of the batch size, or else examples
        # will get dropped. So we pad with fake examples which are ignored
        # later on. These do NOT count towards the metric (all tf.metrics
        # support a per-instance weight, and these get a weight of 0.0).
        #
        # Modified in XL: We also adopt the same mechanism for GPUs.
        while len(eval_examples) % FLAGS.eval_batch_size != 0:
            eval_examples.append(PaddingInputExample())

        eval_file_base = "{}.len-{}.{}.eval.tf_record".format(
            spm_basename, FLAGS.max_seq_length, FLAGS.eval_split)
        eval_file = os.path.join(FLAGS.output_dir, eval_file_base)

        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenize_fn,
            eval_file)

        assert len(eval_examples) % FLAGS.eval_batch_size == 0
        eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=True)

        # Filter out all checkpoints in the directory
        steps_and_files = []
        filenames = tf.gfile.ListDirectory(FLAGS.model_dir)

        for filename in filenames:
            if filename.endswith(".index"):
                ckpt_name = filename[:-6]
                cur_filename = join(FLAGS.model_dir, ckpt_name)
                global_step = int(cur_filename.split("-")[-1])
                tf.logging.info("Add {} to eval list.".format(cur_filename))
                steps_and_files.append([global_step, cur_filename])
        steps_and_files = sorted(steps_and_files, key=lambda x: x[0])

        # Decide whether to evaluate all ckpts
        if not FLAGS.eval_all_ckpt:
            steps_and_files = steps_and_files[-1:]

        eval_results = []
        for global_step, filename in sorted(steps_and_files, key=lambda x: x[0]):
            ret = estimator.evaluate(
                input_fn=eval_input_fn,
                steps=eval_steps,
                checkpoint_path=filename)

            ret["step"] = global_step
            ret["path"] = filename

            eval_results.append(ret)

            tf.logging.info("=" * 80)
            log_str = "Eval result | "
            for key, val in sorted(ret.items(), key=lambda x: x[0]):
                log_str += "{} {} | ".format(key, val)
            tf.logging.info(log_str)

        key_name = "eval_pearsonr" if FLAGS.is_regression else "eval_accuracy"
        eval_results.sort(key=lambda x: x[key_name], reverse=True)

        tf.logging.info("=" * 80)
        log_str = "Best result | "
        for key, val in sorted(eval_results[0].items(), key=lambda x: x[0]):
            log_str += "{} {} | ".format(key, val)
        tf.logging.info(log_str)

    if FLAGS.do_predict:
        eval_file_base = "{}.len-{}.{}.predict.tf_record".format(
            spm_basename, FLAGS.max_seq_length, FLAGS.eval_split)
        eval_file = os.path.join(FLAGS.output_dir, eval_file_base)

        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenize_fn,
            eval_file)

        pred_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        predict_results = []
        with tf.gfile.Open(os.path.join(predict_dir, "{}.tsv".format(
                task_name)), "w") as fout:
            fout.write("index\tprediction\n")

            for pred_cnt, result in enumerate(estimator.predict(
                    input_fn=pred_input_fn,
                    yield_single_examples=True,
                    checkpoint_path=FLAGS.predict_ckpt)):
                if pred_cnt % 1000 == 0:
                    tf.logging.info("Predicting submission for example: {}".format(
                        pred_cnt))

                logits = [float(x) for x in result["logits"].flat]
                predict_results.append(logits)

                if len(logits) == 1:
                    label_out = logits[0]
                elif len(logits) == 2:
                    if logits[1] - logits[0] > FLAGS.predict_threshold:
                        label_out = label_list[1]
                    else:
                        label_out = label_list[0]
                elif len(logits) > 2:
                    max_index = np.argmax(np.array(logits, dtype=np.float32))
                    label_out = label_list[max_index]
                else:
                    raise NotImplementedError

                fout.write("{}\t{}\n".format(pred_cnt, label_out))

        predict_json_path = os.path.join(predict_dir, "{}.logits.json".format(
            task_name))

        with tf.gfile.Open(predict_json_path, "w") as fp:
            json.dump(predict_results, fp, indent=4)


if __name__ == "__main__":
    tf.app.run()
