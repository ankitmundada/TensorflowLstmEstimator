# Author: Ankit Mundada
# Date: 7/11/2017
from select import select
import sys
import tensorflow as tf
import utils

tf.logging.set_verbosity(tf.logging.INFO)
IS_DEBUG = True
IS_GPU_AVL = False
DO_PREDICTION = True

# parameters to tune
PARAMS = {
    'NUM_ITER': 1 if IS_DEBUG else 10,
    'STEPS_PER_ITER': 100,
    'LEARNING_RATE': 1e-3,
    'CONTEXT_SIZE': 3,  # context of words to predict next word
    'LSTM_SIZE': 32,  # number of hidden units in LSTM cell
    'BATCH_SIZE': 16,
}

FLAGS = {
    'MODEL_DIR': './logs/rnn_1',
    'TRAINING_DATA': './datasets/train.txt',
    'VAL_DATA': './datasets/val.txt',
    'TEST_DATA': './datasets/test.txt'
}
force_create_vocab = False  # predictions won't run when this if true, tf might return before IDX_WORD is written to disk
vocab_size = utils.initiate_vocabs(is_forced=force_create_vocab)
output_size = vocab_size


def input_fn(filepath, mode=None):
    """
    Implements the Recommended Input pipeline architecture of Tensorflow.
    :param filepath: File to be loaded into memory line by line. (MUST be a CSV)
    :param mode: One of the tf.estimator.ModeKeys (Train, Eval, Predict)
    :return: The input features and target values for the current step
    """

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    repeat_count = None if is_training else 1

    default_val = [[0.0] for _ in range(PARAMS['CONTEXT_SIZE'])]
    default_val.append([0])  # output class should have data-type tf.int32, for accuracy calculations in model_fn

    def decode_csv(line):
        line = tf.decode_csv(line, default_val)
        return {'context': line[:-1]}, line[-1]

    dataset = tf.contrib.data\
        .TextLineDataset(utils.make_csv(filepath, PARAMS['CONTEXT_SIZE']))\
        .map(decode_csv, num_threads=4 if IS_GPU_AVL else 2)  # preprocessing
    # shuffle input
    if is_training:
        dataset = dataset.shuffle(buffer_size=PARAMS['BATCH_SIZE'] * 2)
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(PARAMS['BATCH_SIZE'])
    iterator = dataset.make_one_shot_iterator()
    next_feature, next_label = iterator.get_next()
    return next_feature, next_label


def model_fn(features, labels, mode, params):
    """
    Required to be passed into the Estimator. This function model a simple LSTM model using Tensorflow's in-built and
    efficient implementations of LSTM cell
    :param features: features as returned from input_fn
    :param labels: labels as returned from input_fn
    :param mode: mode set by the different method calls of the Estimator Object
    :param params: params passed with the Estimator
    :return: returns an EstimatorSpec which store's different important params to analyze tha model
    """
    # break the context words aka features into 'time-steps'
    x = tf.split(features['context'], PARAMS['CONTEXT_SIZE'], 1)

    # setting up LSTM
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(PARAMS['LSTM_SIZE'])
    output, _ = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # only last time-step matters
    output = output[-1]

    # get linear predictions from LSTM outputs
    weight = tf.Variable(tf.random_normal([PARAMS['LSTM_SIZE'], output_size]))
    bias = tf.Variable(tf.random_normal([output_size]))
    logits = tf.matmul(output, weight) + bias

    # final predictions
    preds = tf.argmax(tf.nn.softmax(logits), axis=1, output_type=tf.int32)
    preds_dict = {"predictions": preds}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=preds_dict,
        )

    # using a Cross-Entropy error for this muticlass classification problem
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    if mode == tf.estimator.ModeKeys.EVAL:
        # setting up different metrices to be monitored using tensorboard
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=preds),
            'precision': tf.metrics.precision(labels=labels, predictions=preds),
            'recall': tf.metrics.recall(labels=labels, predictions=preds)
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=preds_dict,
            loss=loss,
            eval_metric_ops=eval_metric_ops
        )

    correct_pred = tf.equal(preds, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    # creating a summary to be added to SummarySaveHook
    with tf.name_scope('summaries'):
        tf.summary.scalar('accuracy', accuracy)

    # define the training operation/optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=PARAMS['LEARNING_RATE'])
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    tensors_to_save = {'training_accuracy': 'accuracy'}
    # Hooks to monitor the performance of the model during training
    hook_logging = tf.train.LoggingTensorHook(tensors_to_save, every_n_iter=25)
    hook_summary = tf.train.SummarySaverHook(save_steps=25, output_dir=FLAGS['MODEL_DIR'], scaffold=None,
                                             summary_op=tf.summary.merge_all())

    # return EstimatorSpec
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=preds_dict,
        loss=loss,
        train_op=train_op,
        training_hooks=[hook_logging, hook_summary]
    )


# Initialize estimator. it'll automatically load the most recent saved model in the 'model_dir'
rnn_regressor = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS['MODEL_DIR'], params=PARAMS)

TIMEOUT = 5  # wait after each iter for 5 sec, to quit training the model, as it may require more tuning at that stage
for i in range(PARAMS['NUM_ITER']):
    rnn_regressor.train(input_fn=lambda: input_fn(FLAGS['TRAINING_DATA'], mode=tf.estimator.ModeKeys.TRAIN),
                        steps=2 if IS_DEBUG else PARAMS['STEPS_PER_ITER'])

    # Evaluate at the end of each iteration
    eval_results = rnn_regressor.evaluate(input_fn=lambda : input_fn(FLAGS['VAL_DATA'], mode=tf.estimator.ModeKeys.EVAL))
    print('Evaluation loss after %d iters is: %f' % (i, eval_results['loss']))

    # Stop training in between in more tuning is required.
    print("Stop training now?\nType y for Yes")
    rlist = select([sys.stdin], [], [], TIMEOUT)[0]
    feedback = None
    if rlist:
        feedback = sys.stdin.readline().strip()
        if (feedback is 'y') or (feedback is 'yes'):
            print('Finishing the training')
            break
    else:
        print('Training for the next iteration')
        continue

# Make predictions on the test dataset
if DO_PREDICTION and not force_create_vocab:  # tf is returning earlier that the json dump. IDX_WORD gives error due to that
    # get results on the test data after all the training
    predictions = rnn_regressor.predict(input_fn=lambda: input_fn(FLAGS['TEST_DATA'], mode=tf.estimator.ModeKeys.PREDICT))
    with open(utils.make_csv(FLAGS['TEST_DATA'], PARAMS['CONTEXT_SIZE']), 'r') as infile:
        for i, pred in enumerate(predictions):
            list_idx = infile.readline().strip().split(sep=',')[:-1] + [pred['predictions'].tolist()]
            pred_word = utils.convert_index_to_word(list_idx)
            print(' '.join(pred_word))


def create_new_text(initializer, how_many_words=10):

    for i in range(how_many_words):
        rnn_regressor.predict()

