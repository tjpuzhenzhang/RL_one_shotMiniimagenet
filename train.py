from __future__ import division

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import os
import model
import task_generator as tg
import utils
import random
from scipy import stats

BATCH_SIZE = 128
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001
EPSILON = 0.1


class Trainer:

    def __init__(self, state_dim, action_dim, action_lim, ram, device='cpu'):
        """
        :param state_dim: Dimensions of state (int)
        :param action_dim: Dimension of action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :param ram: replay memory buffer object
        :return:
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.ram = ram
        self.iter = 0
        self.device = device
        # self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)

        self.actor = model.Actor(
            self.state_dim, self.action_dim, self.action_lim).to(device)
        self.target_actor = model.Actor(
            self.state_dim, self.action_dim, self.action_lim).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), LEARNING_RATE)

        self.critic = model.Critic(self.state_dim, self.action_dim).to(device)
        self.target_critic = model.Critic(
            self.state_dim, self.action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), LEARNING_RATE)

        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)

    def get_exploitation_action(self, state):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        # state = torch.from_numpy(state)
        action = self.target_actor.forward(state).detach()
        return action.data.numpy()

    def get_exploration_action(self, state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        action = self.actor(state).detach()
        new_action = action.int()
        if random.random() < EPSILON:
            return torch.tensor(random.sample(
                [i for i in range(0, self.action_lim)], self.action_dim), dtype=torch.int, device=self.device)
        # new_action = action.data.numpy() + (self.noise.sample()
        #                                     * self.action_lim)
        return new_action

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        s1, a1, r1, s2 = self.ram.sample(BATCH_SIZE)

        s1 = torch.from_numpy(s1).to(self.device)
        a1 = torch.from_numpy(a1).to(self.device)
        r1 = torch.from_numpy(r1).to(self.device)
        s2 = torch.from_numpy(s2).to(self.device)

        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        a2 = self.target_actor.forward(s2).detach()
        next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
        # y_exp = r + gamma*Q'( s2, pi'(s2))
        y_expected = (r1 + GAMMA*next_val).squeeze()
        # y_pred = Q( s1, a1)
        y_predicted = torch.squeeze(self.critic.forward(s1, a1))
        # compute critic loss, and update the critic
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------
        pred_a1 = self.actor.forward(s1)
        loss_actor = -1*torch.sum(self.critic.forward(s1, pred_a1))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        utils.soft_update(self.target_actor, self.actor, TAU)
        utils.soft_update(self.target_critic, self.critic, TAU)

        # if self.iter % 100 == 0:
        #     print('Iteration :- ', self.iter, ' Loss_actor :- ',
        #           loss_actor.data.numpy(), ' Loss_critic :- ',
        #           loss_critic.data.numpy())
        # self.iter += 1

    def save_models(self, episode_count):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        torch.save(self.target_actor.state_dict(), './models/' +
                   str(episode_count) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), './models/' +
                   str(episode_count) + '_critic.pt')
        print('models saved successfully')

    def load_models(self, episode):
        """
        loads the target actor and critic models, and copies them onto actor
                                and critic models
        :param episode: the count of episodes iterated (used to find the file
                                name)
        :return:
        """
        self.actor.load_state_dict(torch.load(
            './models/' + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load(
            './models/' + str(episode) + '_critic.pt'))
        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)
        print('models loaded succesfully')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m, h


class RNTrainer(object):
    def __init__(self, metatrain_folders, metatest_folders,
                 feature_dim, relation_dim, class_num, sample_num_per_class,
                 batch_num_per_class, device='cpu'):
        self.metatrain_folders = metatrain_folders
        self.metatest_folders = metatest_folders
        self.feature_dim = feature_dim
        self.relation_dim = relation_dim
        self.class_num = class_num
        self.sample_num_per_class = sample_num_per_class
        self.batch_num_per_class = batch_num_per_class
        self.device = device

        print("init neural networks")

        self.feature_encoder = model.CNNEncoder()
        self.relation_network = model.RelationNetwork(
            feature_dim, relation_dim)

        self.feature_encoder.apply(weights_init)
        self.relation_network.apply(weights_init)

        self.feature_encoder.to(device)
        self.relation_network.to(device)

        self.feature_encoder_optim = torch.optim.Adam(
            self.feature_encoder.parameters(), lr=LEARNING_RATE)
        self.feature_encoder_scheduler = StepLR(
            self.feature_encoder_optim, step_size=100000, gamma=0.5)
        self.relation_network_optim = torch.optim.Adam(
            self.relation_network.parameters(), lr=LEARNING_RATE)
        self.relation_network_scheduler = StepLR(
            self.relation_network_optim, step_size=100000, gamma=0.5)

        self.mse = nn.MSELoss().to(self.device)

    def optimize(self, episode, class_folders_ids):
        task = tg.MiniImagenetTask(class_folders_ids, self.metatrain_folders, self.class_num,
                                   self.sample_num_per_class, self.batch_num_per_class)
        sample_dataloader = tg.get_mini_imagenet_data_loader(
            task, num_per_class=self.sample_num_per_class, split="train", shuffle=False)
        batch_dataloader = tg.get_mini_imagenet_data_loader(
            task, num_per_class=self.batch_num_per_class, split="test", shuffle=True)

        # sample datas
        samples, sample_labels = sample_dataloader.__iter__().next()
        batches, batch_labels = batch_dataloader.__iter__().next()
        samples = samples.to(self.device)
        batches = batches.to(self.device)
        # calculate features
        sample_features = self.feature_encoder(samples)  # 5x64*5*5
        batch_features = self.feature_encoder(batches)  # 20x64*5*5

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(
            0).repeat(self.batch_num_per_class*self.class_num, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(
            self.sample_num_per_class*self.class_num, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        relation_pairs = torch.cat(
            (sample_features_ext, batch_features_ext), 2).view(-1, self.feature_dim*2, 19, 19)
        relations = self.relation_network(
            relation_pairs).view(-1, self.class_num*self.sample_num_per_class)

        mse = nn.MSELoss().to(self.device)
        one_hot_labels = torch.zeros(
            self.batch_num_per_class*self.class_num, self.class_num).scatter_(1, batch_labels.view(-1, 1), 1).to(self.device)
        loss = mse(relations, one_hot_labels)

        # training

        self.feature_encoder.zero_grad()
        self.relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.relation_network.parameters(), 0.5)

        self.feature_encoder_optim.step()
        self.relation_network_optim.step()

        if (episode+1) % 100 == 0:
            print("episode:", episode + 1, "loss", loss.item())

        return relations.view(-1).detach(), -loss.item()

    def validate(self, test_episode):
        print("Testing...")
        accuracies = []
        for i in range(test_episode):
            total_rewards = 0
            counter = 0
            class_folders_ids = random.sample(
                [i for i in range(0, len(self.metatest_folders))], self.class_num)
            task = tg.MiniImagenetTask(class_folders_ids,
                                       self.metatest_folders, self.class_num, 1, 15)
            sample_dataloader = tg.get_mini_imagenet_data_loader(
                task, num_per_class=1, split="train", shuffle=False)

            num_per_class = 3
            test_dataloader = tg.get_mini_imagenet_data_loader(
                task, num_per_class=num_per_class, split="test", shuffle=True)
            sample_images, sample_labels = sample_dataloader.__iter__().next()
            sample_images = sample_images.to(self.device)
            for test_images, test_labels in test_dataloader:
                test_images = test_images.to(self.device)
                batch_size = test_labels.shape[0]
                # calculate features
                sample_features = self.feature_encoder(sample_images)  # 5x64
                test_features = self.feature_encoder(test_images)  # 20x64

                # calculate relations
                # each batch sample link to every samples to calculate relations
                # to form a 100x128 matrix for relation network
                sample_features_ext = sample_features.unsqueeze(
                    0).repeat(batch_size, 1, 1, 1, 1)
                test_features_ext = test_features.unsqueeze(
                    0).repeat(1*self.class_num, 1, 1, 1, 1)
                test_features_ext = torch.transpose(
                    test_features_ext, 0, 1)
                relation_pairs = torch.cat(
                    (sample_features_ext, test_features_ext), 2).view(-1, self.feature_dim*2, 19, 19)
                relations = self.relation_network(
                    relation_pairs).view(-1, self.class_num)

                _, predict_labels = torch.max(relations.data, 1)

                rewards = [1 if predict_labels[j] ==
                           test_labels[j] else 0 for j in range(batch_size)]

                total_rewards += np.sum(rewards)
                counter += batch_size
            accuracy = total_rewards/1.0/counter
            accuracies.append(accuracy)

        test_accuracy, h = mean_confidence_interval(accuracies)
        print("test accuracy:", test_accuracy, "h:", h)

        return test_accuracy

    def load_models(self):
        if os.path.exists(str("./models/miniimagenet_feature_encoder_" + str(self.class_num) + "way_" +
                              str(self.sample_num_per_class) + "shot.pkl")):
            self.feature_encoder.load_state_dict(torch.load(str(
                "./models/miniimagenet_feature_encoder_" + str(self.class_num) + "way_" +
                str(self.sample_num_per_class) + "shot.pkl")))
            print("load feature encoder success")
        if os.path.exists(str("./models/miniimagenet_relation_network_" + str(self.class_num)
                              + "way_" + str(self.sample_num_per_class) + "shot.pkl")):
            self.relation_network.load_state_dict(torch.load(str(
                "./models/miniimagenet_relation_network_" + str(self.class_num) + "way_" +
                str(self.sample_num_per_class) + "shot.pkl")))
            print("load relation network success")

    def save_models(self):
        # save networks
        torch.save(self.feature_encoder.state_dict(),
                   str("./models/miniimagenet_feature_encoder_" +
                       str(self.class_num) + "way_" + str(self.sample_num_per_class) + "shot.pkl"))
        torch.save(self.relation_network.state_dict(),
                   str("./models/miniimagenet_relation_network_" +
                       str(self.class_num) + "way_" + str(self.sample_num_per_class) + "shot.pkl"))

