import tensorflow as tf
import numpy as np
import add_dependencies as ad # add some dependencies
import utils
import pdb

class stn(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.ds = config['ds']
        self.dt = config['dt']
        self.ns = config['ns']
        self.nl = config['nl']
        self.nu = config['nu']
        self.class_number = config['class_number']
        self.beta = config['beta']
        self.tau = config['tau']
        self.d = config['d']
        self.nt = self.nl+self.nu
        self.create_model()

    def create_model(self):
        #-----------------------------------------#        
        with tf.name_scope('inputs'):
            self.input_xs = tf.placeholder(tf.float32, [None, self.ds], name='input_xs')
            self.input_ys = tf.placeholder(tf.int32, [None, self.class_number], name='input_ys')
            self.input_xl = tf.placeholder(tf.float32, [None, self.dt], name='input_xl')
            self.input_yl = tf.placeholder(tf.int32, [None, self.class_number], name='input_yl')
            self.input_xu = tf.placeholder(tf.float32, [None, self.dt], name='input_xu')
            self.input_yu = tf.placeholder(tf.int32, [None, self.class_number], name='input_yu')
            self.learning_rate = tf.placeholder(tf.float32, [], name='lr')
            self.t = tf.placeholder(tf.float32, [], name='t') # the iter number
            self.T1 = tf.placeholder(tf.float32, [], name='T1') 
            self.T2 = tf.placeholder(tf.float32, [], name='T2')
            self.input_xt = tf.concat([self.input_xl, self.input_xu], 0, name='input_xt')
            self.input_ya = tf.concat([self.input_ys, self.input_yl], 0, name='input_ya')
        #------------------------------------------#
        # set the number of each layer
        s_h = (self.ds+self.d)/2 # 2 layers
        #----------------------------#
        # set the number of each layer
        t_h = (self.dt+self.d)/2 # 2 layers
        #------------------------------------------#
        # set the parameters of each layer
        n_w = {
           's_w1': tf.Variable(tf.truncated_normal([self.ds, s_h], stddev=0.01)),
           's_w2': tf.Variable(tf.truncated_normal([s_h, self.d], stddev=0.01)),
            #-----------------------------------------------------------------#
            't_w1': tf.Variable(tf.truncated_normal([self.dt, t_h], stddev=0.01)),
            't_w2': tf.Variable(tf.truncated_normal([t_h, self.d], stddev=0.01)),
        }
        n_b = {
            's_b1': tf.Variable(tf.truncated_normal([s_h], stddev=0.01)),
            's_b2': tf.Variable(tf.truncated_normal([self.d], stddev=0.01)),
            #-----------------------------------------------------------#
            't_b1': tf.Variable(tf.truncated_normal([t_h], stddev=0.01)),
            't_b2': tf.Variable(tf.truncated_normal([self.d], stddev=0.01)),
        }
        #------------------------------------------#
        # build projection network phi_s(Xs)
        self.projection_xs = utils.build_s(self.input_xs, n_w, n_b, tf.nn.leaky_relu)
        # build projection network phi_t(Xt)
        self.projection_xt = utils.build_t(self.input_xt, n_w, n_b, tf.nn.leaky_relu)
        self.projection_xl = tf.slice(self.projection_xt, [0, 0], [self.nl, -1])
        self.projection_xu = tf.slice(self.projection_xt, [self.nl, 0], [self.nu, -1]) 
        # connecting all data so that plotting tsne
        self.all_data = tf.concat([self.projection_xs, self.projection_xt], 0)
        #------------------------------------------#
        # set the parameters of classifier layer 
        f_w = tf.Variable(tf.truncated_normal([self.d, self.class_number], stddev=0.01))
        f_b = tf.Variable(tf.truncated_normal([self.class_number], stddev=0.01))
        self.f_x_logits = utils.add_layer(self.all_data, f_w, f_b, tf.nn.leaky_relu)
        self.f_xa_logits = tf.slice(self.f_x_logits, [0, 0], [self.ns+self.nl, -1]) # extract f_xa_logists from f_x_logists, xa is all labeled data
        self.f_xu_logits = tf.slice(self.f_x_logits, [self.ns+self.nl, 0], [self.nu, -1]) # extract f_xu_logists from f_x_logists
        self.pseudo_yu = tf.nn.softmax(self.f_xu_logits)
        #------------------------------------------#
        with tf.name_scope('loss_f_xa'):
            self.loss_f_xa = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_ya, logits=self.f_xa_logits))
        #------------------------------------------#
        #a1 = (self.t < self.T1, lambda: tf.constant(0, tf.float32))
        #b1 = (self.t >= self.T2, lambda: tf.constant(1, tf.float32))
        #pairs1 = [a1, b1]
        #default1 = lambda: (self.t) / (self.T2)
        #self.t0 = tf.case(pairs1, default1)

        # in this paper, we use this weighting scheme
        self.t0 = (self.t) / (self.T2)

        # MMD loss: computer_mmd_loss(xs, xl, xu, ys, yl, pseudo_yu, alpha, class_number, dt)
        with tf.name_scope('mmd_loss'):
            self.margin_loss, self.conditional_loss, self.mmd_loss = utils.compute_mmd_loss(self.projection_xs, 
                self.projection_xl, self.projection_xu, self.input_ys, self.input_yl, self.pseudo_yu, self.class_number, self.nu, self.t0)
       #------------------------------------------#
        # reguralization term
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.tau)(n_w['s_w1']))
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.tau)(n_w['s_w2']))
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.tau)(n_w['t_w1']))
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.tau)(n_w['t_w2']))
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.tau)(f_w))
        self.reg = tf.add_n(tf.get_collection("loss"))
        #------------------------------------------#

        with tf.name_scope('total_loss'):
            self.total_loss = self.loss_f_xa \
                              + self.beta*self.mmd_loss \
                              + self.reg
        #------------------------------------------#
        # the accuracy of xa
        pred_xa = tf.nn.softmax(self.f_xa_logits)
        correct_pred_xa = tf.equal(tf.argmax(self.input_ya,1), tf.argmax(pred_xa,1))
        self.xa_acc = tf.reduce_mean(tf.cast(correct_pred_xa, tf.float32))
        #------------------------------------------#
        self.f_xs_logits = tf.slice(self.f_xa_logits, [0, 0], [self.ns, -1]) # extract f_xs_logists from f_xa_logists
        self.f_xl_logits = tf.slice(self.f_xa_logits, [self.ns, 0], [self.nl, -1]) # extract f_xl_logists from f_xa_logists
        # the accuracy of xs
        pred_xs = tf.nn.softmax(self.f_xs_logits)
        correct_pred_xs = tf.equal(tf.argmax(self.input_ys,1), tf.argmax(pred_xs,1))
        self.xs_acc = tf.reduce_mean(tf.cast(correct_pred_xs, tf.float32))
        # the accuracy of xl
        pred_xl = tf.nn.softmax(self.f_xl_logits)
        correct_pred_xl = tf.equal(tf.argmax(self.input_yl,1), tf.argmax(pred_xl,1))
        self.xl_acc = tf.reduce_mean(tf.cast(correct_pred_xl, tf.float32))
        # the accuracy of xu
        correct_pred_xu = tf.equal(tf.argmax(self.input_yu,1), tf.argmax(self.pseudo_yu,1))
        self.xu_acc = tf.reduce_mean(tf.cast(correct_pred_xu, tf.float32))
        #------------------------------------------#
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)
        #self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.total_loss)
        #self.train_step = tf.train.MomentumOptimizer(self.learning_rate, 0.03).minimize(self.total_loss)

        #writer = tf.summary.FileWriter("log/", self.sess.graph)
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        self.sess.run(init)

