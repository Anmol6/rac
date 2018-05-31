import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process
import ipdb
import matplotlib.pyplot as plt
from sklearn import decomposition
from tensorboardX import SummaryWriter
#import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
writer = SummaryWriter()
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D

#Simulation parameters


#============================================================================================#
# Utilities
#============================================================================================#


def weigh_params(params, weights):
    bsize = tf.shape(weights)[0]
    weighted_param = tf.reshape(weights[:,0],[-1,1,1])*tf.tile(tf.expand_dims(params[0],axis=0),[bsize,1,1])
    for i in range(len(params)-1):
        weighted_param += tf.reshape(weights[:,i+1],[-1,1,1])*tf.tile(tf.expand_dims(params[i+1],axis=0),[bsize,1,1])
    return weighted_param

def linear(w,b,x, expand_dims = True):
    if expand_dims:
        s1 =  tf.matmul(tf.expand_dims(x,axis=1),w) + b
    else:
        s1 =  tf.matmul(x,w) + b
    s2 = s1#tf.squeeze(s1,axis=1)
    print('lol')
    print((s1).shape)
    print(b.shape)
    return s1


def create_params(num_experts, input_size, output_size):
    w_params = []
    b_params = []    
    for i in range(num_experts):
        w_params.append(tf.Variable(tf.truncated_normal([input_size, output_size])))
        b_params.append(tf.Variable(tf.zeros([1,output_size])))
    return w_params, b_params

def build_mlp(
        input_placeholder, 
        output_size,
        input_size,
        num_experts=2,
        scope='policy', 
        n_layers=2,
        size=32, 
        activation=tf.tanh,
        output_activation=None,
        mix=True
        ):
    x = input_placeholder
    #x.set_shape([bsize,input_size]) 
    #gating network
    print('mixing: ' + str(mix))
    if mix==False:
        with tf.variable_scope(scope):
            l = x
            for i in range(n_layers):
                l = tf.layers.dense(l, size, name='l'+str(i+1), activation=activation)
            out = tf.layers.dense(l, output_size, activation=output_activation, name='out')
        return out

    with tf.variable_scope(scope):
        with tf.variable_scope('mixing_network'):
            l = x#tf.concat([tf.expand_dims(x[:,8],axis=-1), x[:,13:]], axis=-1)
            for i in range(n_layers):
                l = tf.layers.dense(l, size, name='l'+str(i+1), activation=activation)
            mixing_weights = tf.layers.dense(l, num_experts, activation=tf.nn.softmax, name='out')
        #mixing_weights = tf.reshape(mixing_weights, [-1,1,1])
        with tf.variable_scope('experts'):
            #x = tf.concat([x[:,:8], x[:,9:13]],axis=-1)
            #input_size = 12
            w_params_1, b_params_1 = create_params(num_experts, input_size, size)
            w_params_2, b_params_2 = create_params(num_experts, size, size)#output_size)
            w_params_3, b_params_3 = create_params(num_experts, size, output_size)

            weighted_w1 = weigh_params(w_params_1,mixing_weights)
            weighted_b1 = weigh_params(b_params_1,mixing_weights)
            
            print(x.shape)
            l1 = tf.tanh(linear(weighted_w1,weighted_b1,x))

            weighted_w2 = weigh_params(w_params_2, mixing_weights)
            weighted_b2 = weigh_params(b_params_2, mixing_weights)
            l2 = tf.tanh(linear(weighted_w2,weighted_b2,l1, expand_dims=False))
            #l2 = tf.squeeze(l2,axis=1) 


            weighted_w3 = weigh_params(w_params_3, mixing_weights)
            weighted_b3 = weigh_params(b_params_3, mixing_weights)
            l3 = tf.tanh(linear(weighted_w3,weighted_b3,l2, expand_dims=False))
            l3 = tf.squeeze(l3,axis=1) 

            #ipdb.set_trace()

        return l3, w_params_1, weighted_w1, b_params_1, weighted_b1, mixing_weights


    
    
    
    return out 



def pathlength(path):
    return len(path["reward"])



#============================================================================================#
# Policy Gradient
#============================================================================================#

def train_PG(exp_name='',
             env_name='CartPole-v0',
             n_iter=100, 
             gamma=1.0, 
             min_timesteps_per_batch=1000, 
             max_path_length=None,
             learning_rate=5e-3, 
             reward_to_go=True, 
             animate=True, 
             logdir=None, 
             normalize_advantages=True,
             nn_baseline=False, 
             seed=0,
             # network arguments
             n_layers=1,
             size=32,
             len_control=None,
             controller_dim= 12,
             n_layers_controller=1,
             mix = True
             ):

    start = time.time()

    # Configure output directory for logging
    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Make the gym environment
    env = gym.make(env_name)
    
    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    #========================================================================================#
    # Notes on notation:
    # 
    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in the function
    # 
    # Prefixes and suffixes:
    # ob - observation 
    # ac - action
    # _no - this tensor should have shape (batch size /n/, observation dim)
    # _na - this tensor should have shape (batch size /n/, action dim)
    # _n  - this tensor should have shape (batch size /n/)
    # 
    # Note: batch size /n/ is defined at runtime, and until then, the shape for that axis
    # is None
    #========================================================================================#

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    #========================================================================================#
    #                           ----------SECTION 4----------
    # Placeholders
    # 
    # Need these for batch observations / actions / advantages in policy gradient loss function.
    #========================================================================================#

    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
    if discrete:
        sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) 
    else:
        sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32) 

    # Define a placeholder for advantages
    sy_adv_n = tf.placeholder(shape=[None,1], name="adv_estimate", dtype=tf.float32)


    print('is_discrete:' + str(discrete))

    #========================================================================================#
    #                           ----------SECTION 4----------
    # Networks
    # 
    # Make symbolic operations for
    #   1. Policy network outputs which describe the policy distribution.
    #       a. For the discrete case, just logits for each action.
    #
    #       b. For the continuous case, the mean / log std of a Gaussian distribution over 
    #          actions.
    #
    #      Hint: use the 'build_mlp' function you defined in utilities.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ob_no'
    #
    #   2. Producing samples stochastically from the policy distribution.
    #       a. For the discrete case, an op that takes in logits and produces actions.
    #
    #          Should have shape [None]
    #
    #       b. For the continuous case, use the reparameterization trick:
    #          The output from a Gaussian distribution with mean 'mu' and std 'sigma' is
    #
    #               mu + sigma * z,         z ~ N(0, I)
    #
    #          This reduces the problem to just sampling z. (Hint: use tf.random_normal!)
    #
    #          Should have shape [None, ac_dim]
    #
    #      Note: these ops should be functions of the policy network output ops.
    #
    #   3. Computing the log probability of a set of actions that were actually taken, 
    #      according to the policy.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ac_na', and the 
    #      policy network output ops.
    #   
    #========================================================================================#
    

    
    if (len_control is not None):
        feature_size = controller_dim
        sy_ob_controller_no = tf.placeholder(shape=[None, ob_dim], name="controlling_state", dtype=tf.float32)
        control_params = build_mlp(sy_ob_controller_no, (feature_size+1)*ac_dim, n_layers=n_layers_controller, scope='lol1', mix=mix)
        sy_policy_features_output = build_mlp(sy_ob_no-sy_ob_controller_no, feature_size, n_layers=n_layers, output_activation=tf.nn.tanh,scope='lol2', mix=mix) 
        t_mat = tf.reshape(control_params, (tf.shape(control_params)[0], ac_dim, feature_size+1))
        #ipdb.set_trace()
        sy_policy_output = tf.squeeze(tf.matmul(t_mat[:,:,:-1],tf.expand_dims(sy_policy_features_output,axis=-1)),axis=-1)+t_mat[:,:,-1]#TODO: combine control_params and output of low-level policy features
        #sy_policy_output = tf.squeeze(sy_policy_output, axis=-1) 
        #sy_policy_output=tf.nn.relu(sy_policy_output)
    else:
        if mix==True:
            sy_policy_output, wp,ww, bp,bw, mixing_weights = build_mlp(sy_ob_no, ac_dim, ob_dim, n_layers=n_layers,mix=mix)
        else:
            sy_policy_output = build_mlp(sy_ob_no, ac_dim, ob_dim, n_layers=n_layers,mix=mix)
    if discrete:
        # YOUR_CODE_HERE
        sy_logits_na = sy_policy_output
        sy_sampled_ac = tf.multinomial(sy_logits_na, num_samples=1) #TODO # Hint: Use the tf.multinomial op
        sy_sampled_ac = tf.reshape(sy_sampled_ac, [-1])
        sy_logprob_n = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sy_ac_na, logits=sy_logits_na )#tf.gather_nd(tf.log(sy_softmax_na), sy_ac_na)
        sy_logprob_n = tf.expand_dims(sy_logprob_n, axis=1)
    else:
        # YOUR_CODE_HERE
        sy_mean = sy_policy_output
        sy_logstd = tf.Variable(tf.zeros([1, ac_dim], dtype=tf.float32),name='std') # logstd should just be a trainable variable, not a network output.
        sy_sampled_ac = sy_mean + tf.multiply(tf.exp(sy_logstd),tf.random_normal(shape=tf.shape(sy_mean))) 
        sy_logprob_n =  -tf.square((sy_ac_na-sy_mean)/tf.exp(sy_logstd))# Hint: Use the log probability under a multivariate gaussian. 




    #========================================================================================#
    #                           ----------SECTION 4----------
    # Loss Function and Training Operation
    #========================================================================================#

    loss = -tf.reduce_mean(tf.multiply(sy_logprob_n,sy_adv_n)) # Loss function that we'll differentiate to get the policy gradient.
    mixing_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "policy/mixing_network")
    expert_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "policy/experts")
   
    if (mix==True):
        update_op_mixing = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=mixing_vars)
        update_op_experts = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=expert_vars)



    update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)


    #========================================================================================#
    #                           ----------SECTION 5----------
    # Optional Baseline
    #========================================================================================#

    if nn_baseline:
        baseline_prediction = tf.squeeze(build_mlp(
                                sy_ob_no, 
                                1,
                                None,
                                scope="nn_baseline",
                                n_layers=n_layers,
                                size=size, mix=False))
        # Define placeholders for targets, a loss function and an update op for fitting a 
        # neural network baseline. These will be used to fit the neural network baseline. 
        baseline_targets = tf.placeholder(shape=[None,1], name="baseline_targets", dtype=tf.float32)
        baseline_loss = tf.nn.l2_loss(baseline_targets-baseline_prediction)
        baseline_op = tf.train.AdamOptimizer(learning_rate).minimize(baseline_loss)
        #baseline_update_op = tf.train.AdamOptimizer(learning_rate).minimize

    
    #========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    #========================================================================================#

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`
    #saver = tf.train.Saver()
    
    load = False
    save_model = False
    load_path = ""
    if load:
        saver.restore(sess, load_path)

    else:
        tf.global_variables_initializer().run() #pylint: disable=E1101



    #========================================================================================#
    # Training Loop
    #========================================================================================#
    #ipdb.set_trace()
    total_timesteps = 0
    #ipdb.set_trace()
    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)
        
        if(itr==0 and save_model):
            if mix==False:
                save_path = "no_moe"
            else:
                save_path = ""
            saver.save(sess, "/tflow_ckpts/" + exp_name +save_path +  ".ckpt")
            

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            
            if ((itr+1)%100==0 and mix==True):
                plt.figure(1)                
                plt.subplot(211)
                plot(np.array(mw_buffer))
                pca = decomposition.PCA(n_components=2)
                pca.fit(np.array(ob_buffer))
                ob_pca = pca.transform(np.array(ob_buffer))
                print(np.array(mw_buffer).shape)
                print(ob_pca.shape)
                plt.subplot(212)
                plt.scatter(ob_pca[:,0],ob_pca[:,1], c=np.array(mw_buffer),s=50)
                #plt.subplot(213)
                #plt.plot(ob_pca[:,0],ob_pca[:,1])
                #ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
                #ax.scatter(ob_pca[:,0],ob_pca[:,1],ob_pca[:,2], c=(np.array(mw_buffer)),s=50)
                draw()
                show()

            ob = env.reset()
            mw_buffer = []
            ob_buffer = []
            obs, acs, rewards, obs_controller = [], [], [], []
            prev_control_ob = ob
            curr_control_len = 0 
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and animate)
            steps = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                obs.append(ob)
                if len_control is not None:
                    if(curr_control_len%len_control==0):
                        control_ob = ob
                        prev_control_ob = ob
                        curr_control_len=0
                    else:
                        control_ob = prev_control_ob
                    obs_controller.append(control_ob)
                    curr_control_len+=1
                    #ipdb.set_trace()
                    ac,mw_l = sess.run([sy_sampled_ac, mixing_weights], feed_dict={sy_ob_no:ob[None], sy_ob_controller_no:control_ob[None]})
                else:
                    if(mix==True):
                        ac, mw_l = sess.run([sy_sampled_ac,mixing_weights], feed_dict={sy_ob_no : ob[None]})
                    else:
                        ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
                
                ac = ac[0]
                if(mix==True):
                    mw_buffer.append(mw_l[0][0])
                    ob_buffer.append(ob)
                #writer.add_scalar('tboard/mw_expert0', mw_l[0][0], steps)
                acs.append(ac)
                #ipdb.set_trace()
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    break
            
            path = {"observation" : np.array(obs), 
                    "reward" : np.array(rewards), 
                    "action" : np.array(acs),
                    "observation_controller": np.array(obs_controller)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        if len_control is not None:
            ob_controller_no = np.concatenate([path["observation_controller"] for path in paths])

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Computing Q-values
        #
        # Your code should construct numpy arrays for Q-values which will be used to compute
        # advantages (which will in turn be fed to the placeholder you defined above). 
        #
        # Recall that the expression for the policy gradient PG is
        #
        #       PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]
        #
        # where 
        #
        #       tau=(s_0, a_0, ...) is a trajectory,
        #       Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
        #       and b_t is a baseline which may depend on s_t. 
        #
        # You will write code for two cases, controlled by the flag 'reward_to_go':
        #
        #   Case 1: trajectory-based PG 
        #
        #       (reward_to_go = False)
        #
        #       Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over 
        #       entire trajectory (regardless of which time step the Q-value should be for). 
        #
        #       For this case, the policy gradient estimator is
        #
        #           E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]
        #
        #       where
        #
        #           Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.
        #
        #       Thus, you should compute
        #
        #           Q_t = Ret(tau)
        #
        #   Case 2: reward-to-go PG 
        #
        #       (reward_to_go = True)
        #
        #       Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
        #       from time step t. Thus, you should compute
        #
        #           Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        #
        #
        # Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
        # like the 'ob_no' and 'ac_na' above. 
        #
        #====================================================================================#

        # YOUR_CODE_HERE
        q_n=[]
        
        if (reward_to_go):
            for path in paths:
                pathlen = pathlength(path)
                path_q_n = []
                trajectory_discounted_reward = 0.0
                for t in range(pathlen):
                    path_q_n.append(path["reward"][pathlen-t-1]+gamma*trajectory_discounted_reward)
                    trajectory_discounted_reward = path_q_n[-1]
                q_n.append(np.expand_dims(np.flip(np.array(path_q_n), axis=0),axis=1))


        else:
            for path in paths:
                pathlen = pathlength(path)
                path_q_n = []
                trajectory_total_discounted_reward = 0.0
                for t in range(pathlen):
                    trajectory_total_discounted_reward+=(gamma**t)*path["reward"][t]
                q_n.append(np.full((pathlen,1), trajectory_total_discounted_reward))
                
        

        q_n = np.concatenate(q_n, axis=0)
        
        #====================================================================================#
        #                           ----------SECTION 5----------
        # Computing Baselines
        #====================================================================================#

        if nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            #
            # Hint #bl1: rescale the output from the nn_baseline to match the statistics
            # (mean and std) of the current or previous batch of Q-values. (Goes with Hint
            # #bl2 below.)

            b_n = sess.run(baseline_prediction, feed_dict={sy_ob_no:ob_no})
              
            #if normalize_advantages:
            b_n = (b_n - np.mean(q_n))/(np.std(q_n)+1e-6)    
            #ipdb.set_trace()
            adv_n = q_n - np.expand_dims(b_n,axis=-1) + 1e-6
        else:
            adv_n = q_n.copy()

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Advantage Normalization
        #====================================================================================#

        if normalize_advantages:
            adv_n = adv_n - np.mean(adv_n)
            adv_n = adv_n/(np.std(adv_n) + 1e-6)
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1. 
            
            pass


        #====================================================================================#
        #                           ----------SECTION 5----------
        # Optimizing Neural Network Baseline
        #====================================================================================#
        if nn_baseline:
            # ----------SECTION 5----------
            # If a neural network baseline is used, set up the targets and the inputs for the 
            # baseline. 
            # 
            # Fit it to the current batch in order to use for the next iteration. Use the 
            # baseline_update_op you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the 
            # targets to have mean zero and std=1. (Goes with Hint #bl1 above.)

            # YOUR_CODE_HERE
            b_n_targets = q_n
            
            

            pass

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Performing the Policy Update
        #====================================================================================#

        # Call the update operation necessary to perform the policy gradient update based on 
        # the current batch of rollouts.
        # 
        # For debug purposes, you may wish to save the value of the loss function before
        # and after an update, and then log them below. 

        # YOUR_CODE_HERE
        #ipdb.set_trace() 
        #if discrete:
        #    ac_na = np.stack([np.arange(ac_na.shape[0]),ac_na], axis=1)
        
        #ipdb.set_trace()
        if len_control is not None:
            _, loss_scalar = sess.run([update_op, loss], feed_dict={sy_ob_no:ob_no, sy_ac_na: ac_na, sy_adv_n: adv_n, sy_ob_controller_no:ob_controller_no})
       
        else:
            #ipdb.set_trace()
            if (mix==True):
                _, loss_scalar,mixing_weights_  = sess.run([update_op, loss, mixing_weights], feed_dict={sy_ob_no:ob_no, sy_ac_na: ac_na, sy_adv_n: adv_n})
                ''' 
                if(itr%10<3):
                    _, loss_scalar,mixing_weights_  = sess.run([update_op_mixing, loss, mixing_weights], feed_dict={sy_ob_no:ob_no, sy_ac_na: ac_na, sy_adv_n: adv_n})
                else:
                    _, loss_scalar,mixing_weights_  = sess.run([update_op, loss, mixing_weights], feed_dict={sy_ob_no:ob_no, sy_ac_na: ac_na, sy_adv_n: adv_n})
 
                '''
                '''
                if(itr%10==0):
                    plot(mixing_weights_[:,0])
                    draw()
                    show()
                '''
            else:
                _, loss_scalar  = sess.run([update_op, loss], feed_dict={sy_ob_no:ob_no, sy_ac_na: ac_na, sy_adv_n: adv_n})
        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageLoss", loss_scalar) 
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        if mix==True:
            logz.log_tabular("expert1averageproportion", np.mean(mixing_weights_[:,0]))
        logz.dump_tabular()
        logz.pickle_tf_vars()
        

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=300)
    parser.add_argument('--batch_size', '-b', type=int, default=500)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--h_controller_steps', '-k', type=int, default=None)
    parser.add_argument('--dont_mix', '-dm', action='store_true') 
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)
        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline, 
                seed=seed,
                n_layers=args.n_layers,
                size=args.size,
                len_control=args.h_controller_steps,
                mix = not args.dont_mix
                )
        train_func()
        # Awkward hacky process runs, because Tensorflow does not like
        # repeatedly calling train_PG in the same thread.
        #p = Process(target=train_func, args=tuple())
        #p.start()
        #p.join()
        

if __name__ == "__main__":
    main()
