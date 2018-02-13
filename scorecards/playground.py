import tensorflow as tf
import numpy as np


TRAINING_SIZE = 10000
TESTING_SIZE = 100

INPUT_LAYER_SIZE = 5
OUTPUT_LAYER_SIZE = 1


def generate_output_for_input(input):
    onsite = 0
    hired = 0
    if input[0] > 0.7 or input[3] < 0.3:
        onsite = 1
    if onsite and (input[1] < 0.3 or input[4] > 0.7):
        hired = 1
    if input[2] > 0.8:
        onsite = 0
        hired = 0
    return [onsite * 2 + hired]


def generate_data(size):
    inputs = np.random.random((INPUT_LAYER_SIZE, size))
    inputs_label = ['input_{}'.format(i) for i in range(INPUT_LAYER_SIZE)]
    outputs = [
        generate_output_for_input([inputs[0][i], inputs[1][i], inputs[2][i], inputs[3][i], inputs[4][i]])
        for i in range(size)
    ]
    print(
        len([o for o in outputs if o[0] == 0]),
        len([o for o in outputs if o[0] == 1]),
        len([o for o in outputs if o[0] == 2]),
        len([o for o in outputs if o[0] == 3]),
        len(outputs),
    )
    return dict(zip(inputs_label, inputs)), outputs


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    # print(dict(features), labels)
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


BATCH_SIZE = 100
TRAIN_STEP = 1000
HIDDEN_LAYERS_SIZES = [10, 10]


def main():

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = generate_data(TRAINING_SIZE), generate_data(TESTING_SIZE)

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=HIDDEN_LAYERS_SIZES,
        # The model must choose between 2 classes.
        n_classes=4)

    # Train the Model.
    classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, BATCH_SIZE), steps=TRAIN_STEP)

    # Evaluate the model.
    eval_result = classifier.evaluate(input_fn=lambda: eval_input_fn(test_x, test_y, BATCH_SIZE))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    to_predict = {
        'input_0': [0.2, 0.2, 0.8],
        'input_1': [0.1, 0.2, 0.4],
        'input_2': [0.9, 0.2, 0],
        'input_3': [0.3, 0.2, 0],
        'input_4': [0.5, 0.2, 0],
    }
    expected = [
        generate_output_for_input([to_predict['input_0'][i], to_predict['input_1'][i], to_predict['input_2'][i], to_predict['input_3'][i], to_predict['input_4'][i]])
        for i in range(len(to_predict['input_0']))
    ]
    predictions = classifier.predict(input_fn=lambda: eval_input_fn(to_predict, labels=None, batch_size=BATCH_SIZE))
    for (prediction, expec) in zip(predictions, expected):
        print(prediction, expec)

    # # Generate predictions from the model
    # expected = ['Setosa', 'Versicolor', 'Virginica']
    # predict_x = {
    #     'SepalLength': [5.1, 5.9, 6.9],
    #     'SepalWidth': [3.3, 3.0, 3.1],
    #     'PetalLength': [1.7, 4.2, 5.4],
    #     'PetalWidth': [0.5, 1.5, 2.1],
    # }
    #
    # predictions = classifier.predict(
    #     input_fn=lambda:iris_data.eval_input_fn(predict_x,
    #                                             labels=None,
    #                                             batch_size=args.batch_size))
    #
    # for pred_dict, expec in zip(predictions, expected):
    #     template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
    #
    #     class_id = pred_dict['class_ids'][0]
    #     probability = pred_dict['probabilities'][class_id]
    #
    #     print(template.format(iris_data.SPECIES[class_id],
    #                           100 * probability, expec))


main()




















x = tf.placeholder('float', [None, 5])
y = tf.placeholder('float')


def neural_network_model(data):

    hidden_layers = []
    previous_layer_size = INPUT_LAYER_SIZE
    for layer_size in HIDDEN_LAYERS_SIZES:
        hidden_layers.append({
            'weights': tf.Variable(tf.random_normal([previous_layer_size, layer_size])),
            'biases': tf.Variable(tf.random_normal([layer_size])),
        })
        previous_layer_size = layer_size

    output_layer = {
        'weights': tf.Variable(tf.random_normal([HIDDEN_LAYERS_SIZES[-1], OUTPUT_LAYER_SIZE])),
        'biases': tf.Variable(tf.random_normal([OUTPUT_LAYER_SIZE])),
    }


    res = data
    for layer in hidden_layers:
        res = tf.add(tf.matmul(res, layer['weights']), layer['biases'])
        res = tf.nn.relu(res)
    res = tf.add(tf.matmul(data, output_layer['weights']), output_layer['biases'])

    return res


# def train_neural_network(x):
#     prediction = neural_network_model(x)
#     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
#     optimizer = tf.train.AdamOptimizer().minimize(cost)
#     hm_epochs = 10
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         for epoch in range(hm_epochs):
#             epoch_loss = 0
#             for _ in range(int(TRAINING_SIZE/BATCH_SIZE)):
#                 epoch_x, epoch_y = mnist.train.next_batch(BATCH_SIZE)
#                 _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
#                 epoch_loss += c
#
#             print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
#         correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
#
#         accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
#         print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
#
# train_neural_network(x)


# import numpy as np
#
#
# def training_set():
#     features = {'SepalLength': np.array([6.4, 5.0]),
#                 'SepalWidth':  np.array([2.8, 2.3]),
#                 'PetalLength': np.array([5.6, 3.3]),
#                 'PetalWidth':  np.array([2.2, 1.0])}
#     labels = np.array([2, 1])
#     return features, labels
#
#
# def evaluation_set():
#
#
#
# train_features, train_labels = training_set()
# evaluate_features, evaluate_labels = evaluation_set()
#
#
# def train_input_fn(features, labels, batch_size):
#     """An input function for training"""
#     # Convert the inputs to a Dataset.
#     dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
#
#     # Shuffle, repeat, and batch the examples.
#     dataset = dataset.shuffle(1000).repeat().batch(batch_size)
#
#     # Build the Iterator, and return the read end of the pipeline.
#     return dataset.make_one_shot_iterator().get_next()
#
#
# def feature_columns(features):
#     # Feature columns describe how to use the input.
#     my_feature_columns = []
#     for key in features.keys():
#         my_feature_columns.append(tf.feature_column.numeric_column(key=key))
#     return my_feature_columns
#
#
# # Build 2 hidden layer DNN with 10, 10 units respectively.
# feature_columns = feature_columns(train_features)
# classifier = tf.estimator.DNNClassifier(
#     feature_columns=feature_columns,
#     # Two hidden layers of 10 nodes each.
#     hidden_units=[10, 10],
#     # The model must choose between 3 classes.
#     n_classes=3)
#
#
# # Train the Model.
# batch_size = 1000
# train_steps = 100
# classifier.train(
#     input_fn=lambda:train_input_fn(train_features, train_labels, batch_size),
#     steps=train_steps)
#
#
# # Evaluate the model.
# eval_result = classifier.evaluate(
#     input_fn=lambda:iris_data.eval_input_fn(test_x, test_y, args.batch_size))
#
# print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))