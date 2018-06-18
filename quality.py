#Simple Neural Network

import tensorflow as tf
import numpy as np 
import csv

learning_rate = 0.05

inpattr = 11 
hidden_1 = 6
hidden_2 = 6
classes = 10 
num_steps = 300

y_mat = np.zeros((1599,10))

X = tf.placeholder('float', [None, inpattr])
Y = tf.placeholder('int16', [None, classes])

weights = {
	'w1' : tf.Variable(tf.random_normal([inpattr,hidden_1]) , name = "weight1"),
	'w2' : tf.Variable(tf.random_normal([hidden_1,hidden_2]) , name = "weight2"),
	'wout': tf.Variable(tf.random_normal([hidden_2,classes]) , name = "weightout")
}

bias = {
	'b1' : tf.Variable(tf.random_normal([hidden_1]) , name = "bias1"),
	'b2' : tf.Variable(tf.random_normal([hidden_2]) , name = "bias2"),
	'ob' : tf.Variable(tf.random_normal([classes]) , name = "biasout")
}



inputs = []
classs = []
inputs1= []
# shapedclass = [[]]
def fetch_data(filename):
	with open(filename,'rb') as csvfile:
		reader = csv.reader(csvfile)
		reader.next()
		for row in reader:
			inputs.append((row[:11]))
			# print(row[:11])
			classs.append(row[11])
			# print(row[11])
			# return
			# inputs.append(float(row))

		# break
		# fixed_acidity = np.reshape(fixed_acidity,len(fixed_acidity), 1)
		# for y in range(1,11):
		# 	for x in range(1,1599):
		# 		inputs1[[x-1][y-1]] = myarray[[x-1][y-1]].astype(float)/100
		# 		pass
		# print(type(myarray[[0][0]]))


	shapedclass = np.reshape(classs,(1599,1))
	for i in range(1,1599):
		np.put(y_mat,[[i,shapedclass[i]]],[1])
	# print inputs
	# print shapedclass
	myarrayx = np.array(inputs)
	myarrayy = np.array(y_mat)	
	print((myarrayx))		

	# for x in range(1,1599):
	# 	for y in range(1,11):
	# 		myarrayx[[x-1][y-1]] = myarrayx[[x-1][y-1]].astype(float) / 100 
	# classs.reshape([[1]])
	return myarrayx, myarrayy

def neural_network(X):
	layer_1 = tf.add(tf.matmul(X,weights['w1']), bias['b1'])
	layer_2 = tf.add(tf.matmul(layer_1,weights['w2']), bias['b2'])
	layer_out = tf.add(tf.matmul(layer_2,weights['wout']), bias['ob'])

	return layer_out

logits = neural_network(X)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()



inputs, classs = fetch_data('winequality-red.csv')


with tf.Session() as sess:
	sess.run(init)

	for steps in range (1, num_steps):
		pass
		sess.run(train_op, feed_dict = {X: inputs , Y: classs })

		if steps % 50 == 0 or steps == 1:
			loss, acc = sess.run([loss_op, accuracy], feed_dict={X: inputs, Y: classs})
			print("Step " + str(steps) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))
	

	print(sess.run(prediction, feed_dict={X:[[0.06,0.0031,0.0037,0.036,0.00067,0.18,0.42,0.0099549,0.0339,0.0066,0.11]]}))

	writer = tf.summary.FileWriter('./my_graph',sess.graph)
	writer.close()
	sess.close()			

	
	# print((np.array(inputs)))
	# print(inputs)
