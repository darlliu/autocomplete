# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Modified on top of run_classifier.py
# just a stripped down version
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import bert.modeling as modeling
import bert.optimization as optimization
import bert.tokenization as tokenization
import tensorflow as tf
import numpy
from ranking.tensorflow_ranking.python.data import parse_from_sequence_example
from generate_data import get_example

flags = tf.flags

FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string(
    "data_dir", "D:/data/",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", "D:/data/uncased_L-12_H-768_A-12/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", "autocomplete",
                    "The name of the task to train.")

flags.DEFINE_string("vocab_file", "D:/data/uncased_L-12_H-768_A-12/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", "D:/data/output/",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_integer("samples", 125000,
                     "Queries to collect.")

# Other parameters

flags.DEFINE_string(
    "init_checkpoint", "D:/data/uncased_L-12_H-768_A-12/bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_data", False, "Whether to generate tfrecord. `FLAGS.samples` is used.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 12, "Total batch size for training. Note: batch sizes are used for query level training.")

flags.DEFINE_integer("eval_batch_size", 12, "Total batch size for eval. Note: batch sizes are used for query level training.")

flags.DEFINE_integer("predict_batch_size", 12, "Total batch size for predict. Note: batch sizes are used for query level training.")

flags.DEFINE_float("learning_rate", 5e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 5000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

class InputGroup(object):
  """
  Simple object class for holding a query object.
  """
  def __init__(self, qid, examples, original, trimmed):
    self.qid = qid
    self.examples = examples
    self.original = original
    self.trimmed = trimmed

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None, cnt=None):
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
    self.cnt = cnt


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


class AutoCompleteProcessor(Object):
  def __init__(self, data_dir=FLAGS.data_dir):
    self.language = "en"
    self.train_test_split()

  def evaluate(self, results, data = None):
    if data is None: data = self.devs
    true_labels = []
    header = ["query", "max_count_candidates", "count_correct_rank", "max_res_candidates", "res_correct_rank"]
    f1 = open("{}/static_pref.tsv".format(FLAGS.output_dir), "w")
    f2 = open("{}/bert_pref.tsv".format(FLAGS.output_dir), "w")
    f3 = open("{}/neither.tsv".format(FLAGS.output_dir), "w")
    f1.write("\t".join(header)+"\n")
    f2.write("\t".join(header)+"\n")
    f3.write("\t".join(header)+"\n")
    scores = {
      "acc_static": 0,
      "per_static": 0,
      "p1_static": 0,
      "mrr_static": 0,
      "ndcg_static": 0,
      "acc_pred": 0,
      "per_pred": 0,
      "p1_pred": 0,
      "mrr_pred": 0,
      "ndcg_pred": 0,
    }
    idx = 0
    for example in data:
      trimmed_query, (query, query_cnt), candidates = example
      static_candidates = sorted(candidates, key = lambda x: -x[1])
      max_candidate = static_candidates[0]
      reses = []
      for idx, (q, _) in enumerate(candidates):
        res = results[idx]["probabilities"]
        reses.append(res[1])
      res_candidates = [(i[0],j) for (i,j) in zip(candidates, reses)]
      res_candidates = sorted(res_candidates, key = lambda x: -x[1])
      max_pred_candidate = res_candidates[0]
      max_res = max_pred_candidate[1]
      if query == max_candidate[0]:
        scores["p1_static"] += 1
      if query == max_pred_candidate[0]:
        scores["p1_pred"] += 1
      # print(query, candidates, static_candidates, res_candidates)
      mrr_rank_static = [x[0] for x in static_candidates].index(query) + 1
      mrr_rank_pred = [x[0] for x in res_candidates].index(query) + 1
      if mrr_rank_static > 3:
        dcg_static = 0
      else:
        dcg_static = 1/numpy.log2(mrr_rank_static + 1)
      scores["ndcg_static"] += dcg_static
      if mrr_rank_pred > 3:
        dcg_pred = 0
      else:
        dcg_pred = 1/numpy.log2(mrr_rank_pred + 1)
      scores["ndcg_pred"] += dcg_pred
      scores["mrr_static"] += 1/float(mrr_rank_static)
      scores["mrr_pred"] += 1/float(mrr_rank_pred)
      for idy, (q, qcnt) in enumerate(candidates):
        if q == query:
          true_label = 1 
        else:
          true_label = 0
        true_labels.append(true_label)
        if qcnt == max_candidate[1]:
          static_label = 1 
        else:
          static_label = 0
        if reses[idy] == max_res:
          pred_label = 1
        else:
          pred_label = 0
        if true_label == static_label:
          scores["acc_static"] += 1
          if true_label == 1:
            scores["per_static"] += 1
        if true_label == pred_label:
          scores["acc_pred"] += 1
          if true_label == 1:
            scores["per_pred"] += 1
      ss = map(str, [query, max_candidate, mrr_rank_static, max_pred_candidate, mrr_rank_pred])
      if mrr_rank_static == 1 and mrr_rank_static < mrr_rank_pred:
        f1.write("\t".join(ss)+"\n")
      elif mrr_rank_pred == 1 and mrr_rank_static > mrr_rank_pred:
        f2.write("\t".join(ss)+"\n")
      elif mrr_rank_pred > 1 and mrr_rank_static > 1:
        f3.write("\t".join(ss)+"\n")
    assert len(true_labels) == len(results), "mismatch in sizes"
    print(len(true_labels))
    sz1 = float(len(data))
    sz2 = float(len(results))
    print("Accuracy {:.3f} static {:.3f} bert".format(scores["acc_static"]/sz2, scores["acc_pred"]/sz2))
    print("Percision {:.3f} static {:.3f} bert".format(scores["per_static"]/sz1, scores["per_pred"]/sz1))
    print("P@1 {:.3f} static {:.3f} bert".format(scores["p1_static"]/sz1, scores["p1_pred"]/sz1))
    print("MRR {:.3f} static {:.3f} bert".format(scores["mrr_static"]/sz1, scores["mrr_pred"]/sz1))
    print("NDCG@3 {:.3f} static {:.3f} bert".format(scores["ndcg_static"]/sz1, scores["ndcg_pred"]/sz1))
    return

  def train_test_split(self, test_size=0.1, dev_size=0.1):
    """
    implementation of train test split with shuffling and test portion
    data must be in memory
    """
    assert test_size < 1.0, "Splitting portion must be between 0 and 1.0, instead got {}".format(
        test_size)
    idxs = list(range(0, FLAGS.samples))
    numpy.random.shuffle(idxs)
    split_at = int(test_size*len(idxs))
    idxs_l, idxs_r = idxs[split_at:], idxs[:split_at]
    split_at2 = int(dev_size*len(idxs_l))
    idxs_ll, idxs_lr = idxs_l[split_at2:], idxs_l[:split_at2]
    self.trains = set(idxs_ll)
    self.devs = set(idxs_lr)
    self.tests = set(idxs_r)
  
  def __iter__(self):
    for i, example in enumerate(get_example()):
      trimmed_query, (query, query_cnt), candidates = example
      outputs = []
      for j, (q, qcnt)  in enumerate(candidates):
        guid = "ex"  + ("-%d" % (j))
        if q == query:
          label = "click"
        else:
          label = "noclick"
        text_a=tokenization.convert_to_unicode(q.replace("/"," "))
        label=tokenization.convert_to_unicode(label)
        outputs.append(
            InputExample(guid=guid, text_a=text_a, label=label, cnt=qcnt))
      qid="q-%d" % (i)
      output = InputGroup(qid, outputs, query, trimmed_query)
      if i in self.trains:
        yield 1, output
      elif i in self.devs:
        yield 2, output
      elif i in self.tests:
        yield 3, output
      else:
        break
    
  def __len__(self):
    return len(self.trains) + len(self.devs) + len(self.tests)
  
  @property
  def train_sz(self):
    return len(self.trains)

  @property
  def dev_sz(self):
    return len(self.devs)

  @property
  def test_sz(self):
    return len(self.tests)
      
  def get_labels(self):
    return ["noclick", "click"]

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    processor, label_list, max_seq_length, tokenizer, output_dir):
  """Convert a set of `InputExample`s to a TFRecord file."""

  train_writer = tf.python_io.TFRecordWriter(os.path.join(output_dir, "train.tf_record"))
  eval_writer = tf.python_io.TFRecordWriter(os.path.join(output_dir, "eval.tf_record"))
  predict_writer = tf.python_io.TFRecordWriter(os.path.join(output_dir, "predict.tf_record"))

  for (ex_index, (which, query)) in enumerate(processor):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, FLAGS.samples))
    
    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f
    
    def create_string_feature(values):
      f = tf.train.Feature(bytes_list = tf.train.BytesList(value = [values.encode("utf-8")]))
      return f
    
    input_ids = []
    input_mask = []
    segment_ids = []
    label_ids = []
    is_real_example = []
    qs = []
    qcnt = []
    for example in query.examples:
      feature = convert_single_example(ex_index, example, label_list,
                                      max_seq_length, tokenizer)
      input_ids += [create_int_feature(feature.input_ids)]
      input_mask += [create_int_feature(feature.input_mask)]
      segment_ids += [create_int_feature(feature.segment_ids)]
      label_ids += [create_int_feature([feature.label_id])]
      is_real_example += [create_int_feature(
          [int(feature.is_real_example)])]
      qs += [create_string_feature(example.text_a)]
      qcnt += [create_int_feature([example.cnt])]
    context_features = collections.OrderedDict()
    context_features["original"] = create_string_feature(query.original)
    context_features["trimmed"] = create_string_feature(query.trimmed)
    context_features["len"] = create_int_feature([len(query.examples)])
    sequence_features = collections.OrderedDict()
    sequence_features["input_ids"] = tf.train.FeatureList(feature = input_ids)
    sequence_features["input_mask"] = tf.train.FeatureList(feature = input_mask)
    sequence_features["segment_ids"] = tf.train.FeatureList(feature = segment_ids)
    sequence_features["label_ids"] = tf.train.FeatureList(feature = label_ids)
    sequence_features["is_real_example"] = tf.train.FeatureList(feature = is_real_example)
    sequence_features["query"] = tf.train.FeatureList(feature = qs)
    sequence_features["qcnt"] = tf.train.FeatureList(feature = qcnt)
    tf_example = tf.train.SequenceExample(context = tf.train.Features(feature = context_features),
      feature_lists=tf.train.FeatureLists(feature_list = sequence_features))
    ss = tf_example.SerializeToString()
    if len(ss) < 10: continue
    if which == 1:
      train_writer.write(ss)
    elif which == 2:
      eval_writer.write(ss)
    elif which == 3:
      predict_writer.write(ss)
    else:
      break

  train_writer.close()
  eval_writer.close()
  predict_writer.close()

def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to tf.Estimator"""


  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
      "query": tf.FixedLenFeature([], tf.string),
      "qcnt": tf.FixedLenFeature([], tf.int64),
  }

  name_to_context_features = {
    "original" : tf.FixedLenFeature([], tf.string),
    "trimmed" : tf.FixedLenFeature([], tf.string),
    "len" : tf.FixedLenFeature([], tf.int64)
  }

  name_to_sequence_features = {
    "input_ids": tf.FixedLenSequenceFeature([seq_length], tf.int64),
    "input_mask": tf.FixedLenSequenceFeature([seq_length], tf.int64),
    "segment_ids": tf.FixedLenSequenceFeature([seq_length], tf.int64),
    "label_ids": tf.FixedLenSequenceFeature([], tf.int64),
    "is_real_example": tf.FixedLenSequenceFeature([], tf.int64),
    "query": tf.FixedLenSequenceFeature([], tf.string),
    "qcnt": tf.FixedLenSequenceFeature([], tf.int64),
  }

  def _extract_examples(record, batch_size): 
    return parse_from_sequence_example(
      record,
      batch_size,
      context_feature_spec=name_to_context_features,
      example_feature_spec=name_to_sequence_features
    )

  def input_fn(params):
    batch_size = params["batch_size"]
    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    d = d.map( lambda x : _extract_examples(x, batch_size) )
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)
    # do NOT batch afterwards since data is organized by query and in batch mode already
    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for Estimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for Estimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metric_ops=eval_metrics)
    else:
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities})
    return output_spec

  return model_fn



def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "autocomplete": AutoCompleteProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict and not FLAGS.do_data:
    raise ValueError(
        "At least one of `do_data`, `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=False,
      use_one_hot_embeddings=False)

  estimator = tf.estimator.Estimator(model_fn=model_fn)
  
  if FLAGS.do_data:
    file_based_convert_examples_to_features(
        processor, label_list, FLAGS.max_seq_length, tokenizer, FLAGS.output_dir)
    def _test_extract(record):
      print(record, type(record))
      seq_length = 128
      name_to_context_features = {
        "original" : tf.FixedLenFeature([], tf.string),
        "trimmed" : tf.FixedLenFeature([], tf.string),
        "len" : tf.FixedLenFeature([], tf.int64)
      }

      name_to_sequence_features = {
        "input_ids": tf.FixedLenSequenceFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenSequenceFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenSequenceFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenSequenceFeature([], tf.int64),
        "is_real_example": tf.FixedLenSequenceFeature([], tf.int64),
        "query": tf.FixedLenSequenceFeature([], tf.string),
        "qcnt": tf.FixedLenSequenceFeature([], tf.int64),
      }

      context, sequence = tf.parse_single_sequence_example(record, 
        context_features = name_to_context_features,
        sequence_features = name_to_sequence_features)
      examples = []
      for i in range(context["len"]):
        example = {}
        example["input_ids"] = sequence["input_ids"][i,:]
        example["query"] = sequence["query"][i]
      return examples
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([os.path.join(FLAGS.output_dir, "train.tf_record")], num_epochs=1, shuffle=True)
    key, record_string = reader.read(filename_queue)
    data_record = _test_extract(record_string)
    with tf.Session() as sess:
      sess.run(
          tf.variables_initializer(
              tf.global_variables() + tf.local_variables()
          )
      )
      # Start queue runners
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      examples = sess.run(data_record)
      print(examples)

  if FLAGS.do_train:
    train_examples_sz = processor.train_sz
    num_train_steps = int(
        train_examples_sz / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", train_examples_sz)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    num_actual_eval_examples = processor.dev_sz
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    eval_drop_remainder = False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    results = [res for res in estimator.predict(input_fn=eval_input_fn)]
    processor.evaluate(results, processor.devs)
    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_examples, num_actual_predict_examples = processor.get_test_examples(FLAGS.data_dir)
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)
    processor.evaluate([i for i in result], data = processor.tests)
    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    results = []
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.logging.info("***** Predict results *****")
      for (i, prediction) in enumerate(result):
        results.append(prediction)
        probabilities = prediction["probabilities"]
        if i >= num_actual_predict_examples:
          break
        output_line = "\t".join(
            str(class_probability)
            for class_probability in probabilities) + "\n"
        writer.write(output_line)
        num_written_lines += 1
    processor.evaluate(results, processor.tests)


if __name__ == "__main__":
  tf.app.run()
