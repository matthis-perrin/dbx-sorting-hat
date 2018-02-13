import math
import tensorflow as tf
from terminaltables import SingleTable


class Prediction:
    def __init__(self, input, probabilities):
        self.input = input
        self.probabilities = sorted(probabilities, key=lambda p: p[0], reverse=True)
        self.prediction = probabilities[0][1]

    def __str__(self):
        title = 'Prediction for ' + str(self.input)
        data = [[p[1], '{:6.2f}%'.format(100 * p[0])] for p in self. probabilities]
        table = SingleTable(data)
        table.inner_heading_row_border = False
        return title + '\n' + table.table


class SimpleClassifier:
    def __init__(self,
                 inputs,
                 outputs,
                 train_eval_ratio=0.5,
                 hidden_layers=[10, 10],
                 batch_size=100,
                 train_step=1000):
        assert len(inputs) > 1, 'You need to provide at least 1 input'
        assert len(inputs) == len(outputs), 'There should be as many inputs as outputs'
        assert isinstance(outputs[0], str), 'Outputs should be string'

        self.feature_count = len(inputs[0])
        self.batch_size = batch_size
        self.train_step = train_step
        self.vocabulary = list(set(outputs))

        slice_index = math.floor(len(inputs) * train_eval_ratio)
        self.train_features = self._build_features(inputs[:slice_index])
        self.train_labels = self._build_labels(outputs[:slice_index])
        self.eval_features = self._build_features(inputs[slice_index:])
        self.eval_labels = self._build_labels(outputs[slice_index:])
        self.classifier = self._make_classifier(hidden_layers)

    def train(self):
        self.classifier.train(input_fn=self._get_training_data, steps=self.train_step)

    def evaluate(self):
        if not self.eval_labels:
            return 0
        return self.classifier.evaluate(input_fn=self._get_evaluation_data)['accuracy']

    def predict(self, input):
        expected_input_length = len(self.train_features.keys())
        assert len(input) == expected_input_length, \
            'input should be made of {} values'.format(expected_input_length)
        inputs = self._build_features([input])

        def get_data():
            return tf.data.Dataset.from_tensor_slices((inputs,)).batch(1)

        prediction = next(self.classifier.predict(input_fn=get_data))
        probabilities = list(zip(prediction['probabilities'], self.vocabulary))
        return Prediction(input, probabilities)


    def _get_training_data(self):
        data = tf.data.Dataset.from_tensor_slices((self.train_features, self.train_labels))
        data = data.shuffle(1000).repeat().batch(self.batch_size)
        return data

    def _get_evaluation_data(self):
        data = tf.data.Dataset.from_tensor_slices((self.eval_features, self.eval_labels))
        data = data.batch(self.batch_size)
        return data

    def _make_classifier(self, hidden_layers):
        feature_columns = [
            tf.feature_column.numeric_column(key='input_{}'.format(i))
            for i in range(self.feature_count)
        ]
        return tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=hidden_layers,
                                          n_classes=len(self.vocabulary),
                                          label_vocabulary=self.vocabulary)

    def _build_features(self, inputs):
        return {
            'input_{}'.format(i): [input_val[i] for input_val in inputs]
            for i in range(self.feature_count)
        }

    def _build_labels(self, outputs):
        # return [int(self.labels_index[o]) for o in outputs]
        return outputs

    def _build_feature_columns(self, col_num):
        return [tf.feature_column.numeric_column(key='input_{}'.format(i)) for i in range(col_num)]
