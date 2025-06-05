import tensorflow as tf
import numpy as np

class GraphConvolution_DAPNet():
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self,feature_num,adj_size, out_features_dim,
                 features_co,adj_co,features_gene,adj_gene,features_icd,adj_icd):
        super(GraphConvolution_DAPNet, self).__init__()
        self.adj_size = adj_size
        self.out_features_dim = out_features_dim  # adj matrix
        self.feature_num = feature_num
        self.features_co = features_co
        self.raw_adj_co = adj_co

        self.features_gene = features_gene
        self.raw_adj_gene = adj_gene

        self.features_icd = features_icd
        self.raw_adj_icd = adj_icd

        self.gcn_num = 128

        self.adj_co = self.gen_adj(self.raw_adj_co)  # adj matrix
        self.adj_gene = self.gen_adj(self.raw_adj_gene)
        self.adj_icd = self.gen_adj(self.raw_adj_icd)

        self.lr = 0.00002
        self.build_net()
        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.5  # gpu memory 50%
        self.config.gpu_options.allow_growth = True

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
    def add_layer(self,inputs,in_size,out_size,activation_function,name):
        Weights = tf.Variable(tf.random_normal([in_size,out_size],mean=0.0,stddev=0.3),name='%s_Weights'%name)
        bias = tf.Variable(tf.zeros([1,out_size])+0.1,name='%s_bias'%name)
        wb = tf.add(tf.matmul(inputs,Weights),bias)
        if activation_function:
            return activation_function(wb)
        else:
            return wb

    def max_pool(self,input_2d, width, height):
        # expand 4d
        input_3d = tf.expand_dims(input_2d, 0)  # shape = [1,5,5]
        input_4d = tf.expand_dims(input_3d, 3)  # shape = [1,5,5,1]

        # pool
        pool_output = tf.nn.max_pool(input_4d, ksize = [1, height, width, 1], strides = [1, 1, 1, 1], padding = 'VALID')
        pool_output_2d = tf.squeeze(pool_output)  # shape = [4,4]

        return pool_output_2d
    def add_cnn_pool(self, inputs):
        my_filter = tf.Variable(tf.random_normal(shape = [1, 16, 1, 1]))
        input_2d = tf.expand_dims(inputs, 0)
        input_4d = tf.expand_dims(input_2d, 3)
        print(input_4d.shape)
        conv2 = tf.nn.conv2d(input_4d, filter = my_filter, strides = [1, 1, 8, 1], padding = "VALID")
        conv_output_1d = tf.squeeze(conv2)
        my_activation_output = tf.nn.relu(conv_output_1d)

        my_maxpool_output = self.max_pool(my_activation_output, width = 8, height = 1)
        return my_maxpool_output
    def logistic(self,inputs,in_size,out_size,activation_function,name):
        w = tf.Variable(tf.random_normal(shape = [in_size, out_size],mean=0.0,stddev=0.3), name = '%s_weights'%name)
        b = tf.Variable(tf.zeros([1, out_size])+0.1, name = '%s_bias'%name)
        logis = tf.add(tf.matmul(inputs,w),b)
        if activation_function:
            return activation_function(logis)
        else:
            return logis

    def build_net(self):

        with tf.name_scope('input'):
            self.input_x = tf.placeholder(tf.float32,[None,self.feature_num],name='input_x')
            self.input_y = tf.placeholder(tf.float32,[None,1],name='input_y')
            self.tf_is_training = tf.placeholder(tf.bool, None, name='tf_is_training')

        with tf.name_scope('raw_feature_fix'): # network embedding
            self.adj_w = tf.Variable(tf.random_normal([3051, 3051], mean = 0.0, stddev = 0.001), name = 'w1_icd')
            self.input_x_2 = tf.matmul(self.input_x, tf.cast(self.adj_w, tf.float32))
            self.w_input_score = tf.Variable(tf.random_normal([1, self.feature_num], mean = 0.5, stddev = 0.01), name = 'input_score')
            self.input_x_1=  self.input_x_2 * self.w_input_score
            self.fix_2 = self.add_cnn_pool(self.input_x_1)
            self.fix_3 = self.add_cnn_pool(self.fix_2)
            self.fix_4 = self.add_cnn_pool(self.input_x_1)
            self.fix_5 = self.add_cnn_pool(self.fix_4)
            self.x1 = tf.matmul(self.input_x_1, self.features_co)
            self.x2 = tf.matmul(self.input_x_1, self.features_gene)
            self.x3 = tf.matmul(self.input_x_1, self.features_icd)
            self.gcn_x = tf.concat([self.x1, self.x2, self.x3], axis=1)
            self.gcn_out_fix = self.add_layer(self.gcn_x, 1536, 76, tf.nn.tanh, 'gcn_out_fix')
        with tf.name_scope('concat_model_result'):
            self.x = tf.concat([self.gcn_out_fix,self.fix_3,self.fix_5],axis=1)

        with tf.name_scope('classfier'):#
            self.tf_x3 = self.add_layer(self.x,152,32,tf.nn.tanh,'tf_layer1')
            self.prediction = self.add_layer(self.tf_x3,32,1,tf.nn.sigmoid,'tf_layer3')

        with tf.name_scope('loss'):
            self.cross_entropy = -tf.reduce_mean((self.input_y * tf.log(self.prediction + 1e-9) +
                                                 (1 - self.input_y) * tf.log(1 - self.prediction + 1e-9)), name='loss')
            self.train_loss = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)

    def gen_adj(self,adj):
        A = adj
        size = len(A)
        # Get the degrees for each node
        degrees = []
        for node_adjaceny in A:
            num = 0
            for node in node_adjaceny:
                if node >= 1.0:
                    num = num + 1
            # Add an extra for the "self loop"
            num = num + 1
            degrees.append(num)

        # Create diagonal matrix D from the degrees of the nodes
        degrees = np.array(degrees)
        degrees = np.power(degrees, -0.5)
        D = np.diag(degrees)
        # Create an identity matrix of size x size
        I = np.eye(size)
        A_hat = A + I
        # Return A_hat
        adj = np.dot(np.dot(D, A_hat), D)
        return adj

