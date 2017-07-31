import tensorflow as tf
import DataSet
import cPickle
import numpy as np
#from mydropout import dropout

from ConfigParser import *


#   Initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients
#   With ReLU neurons, it is good practice to initialize with a slightly positive bias to avoid "dead neurons".
def weight_variable(shape,noise=0.1):
    initial = tf.truncated_normal(shape, stddev=noise)
    return tf.Variable(initial)

def bias_variable(shape,noise=0.1):
    initial = tf.constant(noise, shape=shape)
    return tf.Variable(initial)


# Convolution: stride=1 and are zero-padded so that the output is the same size as the input.
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Pooling: plain max pooling over 2x2 blocks
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Pooling: plain max pooling over 3x3 blocks
def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# Careful with this
def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count

def trainModel(expDir='null', ii=0):
    
    data_ = open('data_dir.txt','r')
    datasets_dir = data_.readline().split()[0]
    double_mnist = DataSet.read_data_sets(data_dir=datasets_dir)
    

    config = ConfigParser()
    config.read(expDir+'input_configuration')

    mode = config.get('MAIN_PARAMETER_SETTING','mode')
    l_rate = config.getfloat('MAIN_PARAMETER_SETTING','learning_rate')
    gamma = config.getfloat('MAIN_PARAMETER_SETTING','gamma')
    noise = config.getfloat('MAIN_PARAMETER_SETTING','noise')
    numepochs = config.getint('MAIN_PARAMETER_SETTING','training_epochs')
    momentum = config.getfloat('MAIN_PARAMETER_SETTING','momentum')
    p_input = config.getfloat('MAIN_PARAMETER_SETTING','p_input')
    p_conv = config.getfloat('MAIN_PARAMETER_SETTING','p_conv')
    p_fc = config.getfloat('MAIN_PARAMETER_SETTING','p_fc')

    if mode == 'scheduled_dropout':
        def _prob(x, gamma, p):
            return (1.-p) * np.exp(-gamma*x) + p
    elif mode == 'ann_dropout':    
        def _prob(x, gamma, p):
            return - (1.-p)*np.exp(-gamma*x) + 1
    elif mode == 'regular_dropout':
        def _prob(x, gamma, p):
            return p
    elif mode == 'switch_dropout':
        def _prob(current, switch, p):
			if current < switch:
				return 1.0
			else:
				return p
			


    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth = True)))
    #config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)))

    x = tf.placeholder(tf.float32, shape=[None, 4096]) #64*64
    y_ = tf.placeholder(tf.float32, shape=[None, 55])
    
    # DROPOUT
    # placeholder for the probability that a neuron's output is kept during dropout
    # keep_prob will be give to feed_dict to control the dropout rate
    keep_prob_input = tf.placeholder(tf.float32)
    keep_prob_conv = tf.placeholder(tf.float32)
    keep_prob_fc = tf.placeholder(tf.float32)
    
    
    # FIRST CONV LAYER
    # The convolutional layer computes 32 features for each 5x5 patch. 
    # Its weight tensor has a shape of [5, 5, 1, 32] [5x5 patch, input channels, output channels]
    dim_conv1 = int(96)
    W_conv1 = weight_variable([5, 5, 1, dim_conv1],noise)
    #  bias vector with a component for each output channel
    b_conv1 = bias_variable([dim_conv1],noise)
    #To apply the layer, we first reshape x to a 4d tensor 
    # Second and third dimensions correspond to image width and height
    # Final dimension corresponding to the number of color channels.

    x_image = tf.reshape(x, [-1,64,64,1])
    x_image_drop = tf.nn.dropout(x_image, keep_prob_input)
    # convolve x_image with the weight tensor, add the bias, apply ReLU 
    h_conv1 = tf.nn.relu(conv2d(x_image_drop, W_conv1) + b_conv1)
    # finally max pool
    h_pool1 = max_pool_3x3(h_conv1) # now the image is 32*32
    h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob_conv)

    # SECOND CONV LAYER
    # Initialize variables 
    dim_conv2 = int(128)
    W_conv2 = weight_variable([5, 5, dim_conv1, dim_conv2],noise)
    b_conv2 = bias_variable([dim_conv2],noise)
    # Contruct the graph
    h_conv2 = tf.nn.relu(conv2d(h_pool1_drop, W_conv2) + b_conv2)
    h_pool2 = max_pool_3x3(h_conv2) # now the image is 16*16
    h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob_fc)
    
    ## THIRD CONV LAYER
    # Initialize variables
    dim_conv3 = int(256)
    W_conv3 = weight_variable([5, 5, dim_conv2, dim_conv3],noise)
    b_conv3 = bias_variable([dim_conv3],noise)
    # Contruct the graph
    h_conv3 = tf.nn.relu(conv2d(h_pool2_drop, W_conv3) + b_conv3)
    h_pool3 = max_pool_3x3(h_conv3) # now the image is 8*8
    h_pool3_flat = tf.reshape(h_pool3, [-1, 8 * 8 * dim_conv3])
    h_pool3_flat_drop = tf.nn.dropout(h_pool3_flat, keep_prob_conv)
    
    # DENSE LAYER 1
    DIM_1 = int(2048)
    W_fc1 = weight_variable([8 * 8 * dim_conv3, DIM_1],noise)
    b_fc1 = bias_variable([DIM_1],noise)
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat_drop, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_fc)

    # DENSE LAYER 2
    DIM_2 = int(2048)
    W_fc2 = weight_variable([DIM_1, DIM_2],noise)
    b_fc2 = bias_variable([DIM_2],noise)
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop , W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob_fc)

    # READOUT LAYER
    W_out = weight_variable([DIM_2, 55],noise)
    b_out = bias_variable([55],noise)
    y_conv = tf.matmul(h_fc2_drop, W_out) + b_out

    # Loss Function for evaluation (i.e. compare with actual labels)
    # 0.5*y_ is NECESSARY beacuse if if there are two digits' labels
    # distribution y_ must be normalized to 1 before cross entropy
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

    # the actual operation on the graph
    train_step = tf.train.AdamOptimizer(l_rate,beta1=momentum).minimize(cross_entropy)

    # EVALUATE THE MODEL
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    #correct_prediction = tf.equal(tf.nn.top_k(y_conv,2)[1], tf.nn.top_k(y_,2)[1])
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #AP  = sparse_average_precision_at_k(y_conv, tf.cast(y_, tf.int64), 1)
    #mAP =  tf.reduce_mean(tf.cast(AP, tf.float32))

    #################################################


    # Run the session to initialize variables
    sess.run(tf.global_variables_initializer())
    #sess.run(tf.local_variables_initializer())
    
    accTrSet = []
    accTeSet = []
    accValSet = []
    Xentr = []
    gammaValues = []
    
    # Some training params
    TEST_EVAL=False
    VALID=True
    batchsize = 128
    test_batchsize = 200

    
    num_train_batches = double_mnist.train.labels.shape[0] / batchsize
    numiter = numepochs * num_train_batches
    num_test_batches = double_mnist.test.labels.shape[0] / test_batchsize
    num_valid_batches = double_mnist.validation.labels.shape[0] / test_batchsize
    print("Epochs: %d \t Training batches: %d \t Iterations: %d \t Mode: %s"\
                    %(numepochs, num_train_batches, numiter, mode))
    
    start_time = time.time()
        
    ## TRAINING ITERATIONS
    for i in range(int(numiter)):

		# Dropout probabilities for this iteration
        if 'switch_dropout' not in mode:
		    _prob_input = _prob(i, gamma, p_input)
		    _prob_conv = _prob(i, gamma, p_conv)
		    _prob_fc = _prob(i, gamma, p_fc)
        elif mode == 'switch_dropout':
		    current = double_mnist.train.epochs_completed
		    _prob_input = _prob(current, gamma, p_input)
		    _prob_conv = _prob(current, gamma, p_conv)
		    _prob_fc = _prob(current, gamma, p_fc)
        
        # calculate accuracies and cost every 500 iterations
        if i%500 == 0 and i != 0:
            # calculate TRAIN  accuracy on the SINGLE BATCH
            train_accuracy, xentropy = sess.run((accuracy, cross_entropy),feed_dict={
                                                    x:batch[0], y_: batch[1], 
                                                    keep_prob_input: 1.0, keep_prob_conv: 1.0, keep_prob_fc: 1.0}) # no dropout 
            accTrSet.append(train_accuracy)
            Xentr.append(xentropy)
            gammaValues.append(_prob_fc)

            # calculate TEST accuracy on the whole test set
            test_accuracy=0.        
            if TEST_EVAL:
                for j in range(int(num_test_batches)): # Must be done batchwise
                    batch = double_mnist.test.next_batch(test_batchsize)
                    test_accuracy += accuracy.eval(feed_dict={
                                                    x:batch[0], y_: batch[1], 
                                                    keep_prob_input: 1.0, keep_prob_conv: 1.0, keep_prob_fc: 1.0}) # no dropout
                test_accuracy = test_accuracy / num_test_batches
                accTeSet.append(test_accuracy)
 
            # calculate VALIDATION accuracy on the whole validation set
            valid_accuracy=0.
            if VALID:
                for j in range(int(num_valid_batches)): # Must be done batchwise 
                    batch = double_mnist.validation.next_batch(test_batchsize)
                    valid_accuracy += accuracy.eval(feed_dict={
                                                    x:batch[0], y_:batch[1], 
                                                    keep_prob_input: 1.0, keep_prob_conv: 1.0, keep_prob_fc: 1.0}) # no dropout
                valid_accuracy = valid_accuracy / num_valid_batches
                accValSet.append(valid_accuracy)
                #print("Droput prob: %f"%(prob))
        
                ## Early stopping
                #if len(accValSet)>5 and accValSet[-1]<accValSet[-2]:
                    #break
            duration = time.time()-start_time  
            start_time=time.time()
            print("step %d: \t cross entropy: %f \t training accuracy: %f \t test accuracy: %f \t valid accuracy: %f \t prob_fc: %f \t time: %f"\
                            %(i,  xentropy, train_accuracy, test_accuracy, valid_accuracy, _prob_fc, duration))
            
            #start = time.time()
        ## The actual training step
        batch = double_mnist.train.next_batch(batchsize)
        # SCHEDULING DROPOUT: no droput at first, tends to 0.5 as iterations increase
        train_step.run(feed_dict={
                                x: batch[0], y_: batch[1], 
                                keep_prob_input: _prob_input,    
                                keep_prob_conv: _prob_conv,   
                                keep_prob_fc: _prob_fc})    # CHANGE HERE
        
        #vals = bin_2_flat.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: _prob(i, gamma, p_hidden)})
        #print np.unique(_vals)
        
            
    #!# End of training iterations #
    
    
    # Finally test on the test set #
    ## Testing on small gpus must be done batch-wise to avoid OOM
    #test_accuracy=0
    #for j in range(num_test_batches):
        #batch = double_mnist.test.next_batch(test_batchsize)
        #test_accuracy += accuracy.eval(feed_dict={
        #x:batch[0], y_: batch[1],  keep_prob_input: 1.0, keep_prob_conv: 1.0, keep_prob_fc: 1.0}) # no dropout
    
    #print("test accuracy: %g"%(test_accuracy/num_test_batches))
    
    f = file(expDir+str(ii)+'_accuracies.pkl', 'w')
    cPickle.dump((accTrSet, accValSet, accTeSet, Xentr,gammaValues), f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    
    sess.close()

if __name__=="__main__":
    
    import sys
    import glob
    import time

    iteration = sys.argv[2]
    folder = sys.argv[1]
    expDir = glob.glob('./experiments/multiple_runs/'+ str(folder))[0]
    print expDir 
    print iteration

    start = time.time()
    with tf.device('/gpu:0'):
        trainModel(expDir+'/', iteration)
    print 'Duration:',time.time() - start

        
        
        
