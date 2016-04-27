from utils import *
import numpy as np
import random
import tensorflow as tf
import time
import os
import logging
import gym
from gym import envs, scoreboard
from gym.spaces import Discrete, Box
import prettytensor as pt
from space_conversion import SpaceConversionEnv
import tempfile
import sys

eps = 1e-6

config = dict2(**{
    "timesteps": 10,
    "bs": 100,
    "gamma": 0.95,
    "max_kl": 0.05})


class TRPO_RNN_Agent(object):

    def __init__(self, envs):
        self.envs = envs
        if not isinstance(envs[0].observation_space, Box) or \
           not isinstance(envs[0].action_space, Discrete):
            print("Incompatible spaces.")
            exit(-1)
        self.session = tf.Session()
        n = envs[0].observation_space.shape[0]
        timesteps = config.timesteps
        bs = config.bs
        self.end_count = 0
        self.train = True
        self.obs_single = obs_single = tf.placeholder(dtype, shape=[bs, n], name="obs_single")
        self.obs_multi = obs_multi = tf.placeholder(dtype, shape=[timesteps, bs, n], name="obs_multi")
        self.done = done = tf.placeholder(dtype, shape=[timesteps, bs], name="done")  
        self.action = action = tf.placeholder(tf.int64, shape=[timesteps, bs], name="action")  
        self.advant = advant = tf.placeholder(dtype, shape=[timesteps, bs], name="advant")  
        self.state_size = 60
        self.state = state = tf.placeholder(dtype, shape=[bs, self.state_size], name="state")  
        self.oldaction_dist = oldaction_dist = tf.placeholder(dtype, shape=[timesteps, bs, envs[0].action_space.n], name="oldaction_dist")

        # Create neural network.

        with tf.variable_scope("basic_rnn") as scope:
            rnn = tf.nn.rnn_cell.BasicRNNCell(self.state_size)
            output, self.new_state = rnn(obs_single, state)
            action_dist_n, _ = (pt.wrap(output).
                           softmax_classifier(envs[0].action_space.n))  
            action_dist_n = tf.minimum(action_dist_n, 1.0 - eps)
            action_dist_n = tf.maximum(action_dist_n, eps)
            action_dist_n /= tf.expand_dims(tf.reduce_sum(action_dist_n, 1), 1)

        self.action_dist_n = action_dist_n
        surr, kl, ent, kl_firstfixed = 0.0, 0.0, 0.0, 0.0
        print("Unrolling RNN")
        for i in range(config.timesteps):
            sys.stdout.write(".")
            sys.stdout.flush()
            with tf.variable_scope(scope, reuse=True):
                o = tf.expand_dims(obs_multi[i, :, :], 0)
                o = tf.reshape(o, (bs, n))
                output, state = rnn(o, state)
                d = tf.reshape(done[i, :], (bs, 1))
                d = tf.tile(d, (1, self.state_size))
                state = tf.select(tf.less(d, d * 0.0 + 1e-4), state, state * 0.0)
                action_dist_n, _ = (pt.wrap(output).
                               softmax_classifier(envs[0].action_space.n))

                action_dist_n = tf.minimum(action_dist_n, 1.0 - eps)
                action_dist_n = tf.maximum(action_dist_n, eps)
                action_dist_n /= tf.expand_dims(tf.reduce_sum(action_dist_n, 1), 1)

                p_n = slice_2d(action_dist_n, tf.range(0, bs), action[i, :])
                oldp_n = slice_2d(oldaction_dist[i, :, :], tf.range(0, bs), action[i, :])

                ratio_n = p_n / oldp_n
                surr += -tf.reduce_sum(ratio_n * advant[i, :]) / float(timesteps * bs)
                single_kl = oldaction_dist[i, :, :] * tf.log(oldaction_dist[i, :, :] / action_dist_n)
                single_kl = tf.select(tf.greater(oldaction_dist[i, :, :], eps), single_kl, tf.stop_gradient(oldaction_dist[i, :, :] * 0.0))
                kl += tf.reduce_sum(single_kl) / float(timesteps * bs)

                single_ent = -action_dist_n * tf.log(action_dist_n)
                single_ent = tf.select(tf.greater(action_dist_n, eps), single_ent, tf.stop_gradient(single_ent * 0.0))
                ent += tf.reduce_sum(single_ent) / float(timesteps * bs)

                single_kl_first = tf.stop_gradient(action_dist_n) * tf.log(tf.stop_gradient(action_dist_n) / action_dist_n)
                single_kl_first = tf.select(tf.greater(action_dist_n, eps), single_kl_first, tf.stop_gradient(single_kl_first * 0.0))
                kl_firstfixed += tf.reduce_sum(single_kl_first) / float(timesteps * bs)

        self.entropy = ent
        self.kl = kl
        self.surr = surr
        var_list = tf.trainable_variables()
        self.losses = [surr, kl, ent]
        self.pg = flatgrad(surr, var_list)
        # KL divergence where first arg is fixed
        # replace old->tf.stop_gradient from previous kl
        grads = tf.gradients(kl_firstfixed, var_list)
        self.flat_tangent = tf.placeholder(dtype, shape=[None])
        shapes = map(var_shape, var_list)
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size
        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        self.fvp = flatgrad(gvp, var_list)
        self.gf = GetFlat(self.session, var_list)
        self.sff = SetFromFlat(self.session, var_list)
        self.session.run(tf.initialize_variables(var_list))
        self.vf = VF(self.session)

    def learn(self):
        start_time = time.time()
        numeptotal = 0
        pathss = None
        i = 0
        self.state_n = np.zeros((config.bs, self.state_size))
        self.rewards_sum = np.zeros(config.bs)
        while True:
            # Generating paths.
            print("\nGathering trajectories")
            
            pathss = rollout(self.envs, self, config.timesteps, config.bs)

            # Computing returns and estimating advantage function.
            obs_n, done_n, action_n, advant_n, action_dist_n, baseline_n, returns_n, init_state_n = [], [], [], [], [], [], [], []
            for paths in pathss:
                for path in paths:
                    path["baseline"] = self.vf.predict(path)
                    path["returns"] = discount(path["rewards"], config.gamma)
                    path["advants"] = path["returns"] - path["baseline"]
                    baseline_n += path["baseline"].tolist()
                    returns_n += path["returns"].tolist()

                # Updating policy.
                init_state_n.append(np.concatenate([path["state"] for path in paths])[-config.timesteps])
                action_dist_n.append(np.concatenate([path["action_dists"] for path in paths])[-config.timesteps:, :])
                obs_n.append(np.concatenate([path["obs"] for path in paths])[-config.timesteps:, :])
                done_n.append(np.concatenate([path["done"] for path in paths])[-config.timesteps:])
                action_n.append(np.concatenate([path["actions"] for path in paths])[-config.timesteps:])
                advant_n.append(np.concatenate([path["advants"] for path in paths])[-config.timesteps:])

            paths = [path for paths in pathss for path in paths]
            print "\n********** Iteration %i ************" % i
            i += 1
            episoderewards = np.array(
                [path["rewards_sum"].sum() for path in paths])
            episoderewards = np.array(episoderewards)

            print("Average sum of rewards per episode = %f" % episoderewards.mean(), self.envs[0]._env.spec.reward_threshold)
            if episoderewards.mean() > self.envs[0]._env.spec.reward_threshold:
                self.train = False
                print("Skipping")
                continue
            if not self.train:
                print(episoderewards)
                self.end_count += 1
                if self.end_count > 50:
                    break
                continue
            print("Training")

            objs = ["obs_n", "done_n", "action_n", "advant_n", "action_dist_n", "init_state_n"]

            feed = {self.obs_multi: obs_n,
                    self.done: done_n,
                    self.action: action_n,
                    self.advant: advant_n,
                    self.oldaction_dist: action_dist_n}
            for k in feed.keys():
                obj = feed[k]
                obj = np.concatenate([np.expand_dims(o, 1) for o in obj], 1)
                feed[k] = obj
            feed[self.state] = init_state_n


            feed[self.advant] -= feed[self.advant].mean()
            feed[self.advant] /= (feed[self.advant].std() + 1e-8)

            # Computing baseline function for next iter.
            self.vf.fit(paths)
            thprev = self.gf()

            def fisher_vector_product(p):
                feed[self.flat_tangent] = p
                return self.session.run(self.fvp, feed)

            g = self.session.run(self.pg, feed_dict=feed)
            stepdir = conjugate_gradient(fisher_vector_product, -g)
            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / config.max_kl)
            fullstep = stepdir / lm
            neggdotstepdir = -g.dot(stepdir)

            def loss(th):
                self.sff(th)
                return self.session.run(self.losses[0], feed_dict=feed)
            theta = linesearch(loss, thprev, fullstep, neggdotstepdir / lm)
            norm = np.sqrt(np.sum(fullstep * fullstep))
            theta = thprev + fullstep
            self.sff(theta)

            surrafter, kl, entropy = self.session.run(self.losses, feed_dict=feed)   
            stats = []
            for path in paths:
                numeptotal += len(path["rewards"])
            stats.append(("Total number of timesteps", numeptotal))
            stats.append(("Time elapsed", "%.2f mins" % ((time.time() - start_time) / 60.0)))
            stats.append(("Entropy", entropy))
            stats.append(("KL between old and new distribution", kl))
            exp = explained_variance(np.array(baseline_n), np.array(returns_n))
            stats.append(("Baseline explained", exp))
            stats.append(("Surrogate loss after", surrafter))
            stats.append(("Average sum of rewards per episode", episoderewards.mean()))
            stats.append(("Max sum of rewards per episode", episoderewards.max()))
            for k, v in stats:
                print(k + ": " + " " * (40 - len(k)) + str(v))
            if entropy != entropy:
                exit(-1)
            if episoderewards.mean() > 0.8 * self.envs[0]._env.spec.reward_threshold:
                config.train = False
            if exp > 0.8 or entropy < 0.05:
                self.train = False

    def act(self, obs, timesteps_sofar):
        inp = {self.obs_single: obs, self.state: self.state_n}
        action_dist_n, new_state_n = self.session.run([self.action_dist_n, self.new_state], inp)
        for j in range(timesteps_sofar.shape[0]):
            if timesteps_sofar[j]:
                self.state_n[j, :] = copy.copy(new_state_n[j, :])
        if self.train:
            action = cat_sample(action_dist_n)
        else:
            action = np.argmax(action_dist_n, 1)
        return action, action_dist_n


training_dir = tempfile.mkdtemp()
logging.getLogger().setLevel(logging.DEBUG)
if len(sys.argv) > 1:
    task = sys.argv[1]
else:
    task = "Copy-v0"

envs = []
for i in range(config.bs):
    env = gym.envs.make(task)
    env.monitor.start(training_dir)
    env = SpaceConversionEnv(env, Box, Discrete)
    envs.append(env)
    if i != 0:
        env.monitor.configure(video_callable=lambda episode_id: False)

agent = TRPO_RNN_Agent(envs)
agent.learn()
for env in envs:
    env.monitor.close()
print("Training dir: %s" % training_dir)
gym.upload(training_dir, algorithm_id='trpo_rnn')


