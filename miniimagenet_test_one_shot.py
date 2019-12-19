from __future__ import division

import torch

import task_generator as tg
import train

metatrain_character_folders, metatest_character_folders = tg.mini_imagenet_folders()

FEATURE_DIM = 64
RELATION_DIM = 8
CLASS_NUM = 5
SAMPLE_NUM_PER_CLASS = 1
BATCH_NUM_PER_CLASS = 15
EPISODE = 10
TEST_EPISODE = 600

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

rn_trainer = train.RNTrainer(metatrain_character_folders, metatest_character_folders,
                             FEATURE_DIM, RELATION_DIM, CLASS_NUM,
                             SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS, DEVICE)

rn_trainer.load_models()
last_accuracy = 0.0
for r in range(EPISODE):
    test_accuracy = rn_trainer.validate(TEST_EPISODE)

print('Completed episodes')

