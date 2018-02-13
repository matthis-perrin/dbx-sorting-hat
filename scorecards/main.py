from classifier import SimpleClassifier

inputs = [[1, 2, 1, 1, 1], [2, 1, 3, 1, 1], [1, 2, 3, 2, 3], [2, 1, 3, 2, 3]]
outputs = ['onsite_hired', 'onsite', 'rejected', 'rejected']

c = SimpleClassifier(inputs=inputs, outputs=outputs, train_eval_ratio=0.75)
print('Training model...')
c.train()
print('Evaluating model...')
accuracy = c.evaluate()
print('Model accuracy: {:.3f}'.format(accuracy))

print(c.predict([1, 1, 1, 1, 1]))
