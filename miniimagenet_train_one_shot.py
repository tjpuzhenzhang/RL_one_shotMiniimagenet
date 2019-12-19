from __future__ import division

import gc
import random

import torch

import buffer
import task_generator as tg
import train

metatrain_character_folders, metatest_character_folders = tg.mini_imagenet_folders()

MAX_STEPS = 10000
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
S_DIM = 375
A_DIM = 5
A_MAX = len(metatrain_character_folders)

FEATURE_DIM = 64
RELATION_DIM = 8
CLASS_NUM = 5
SAMPLE_NUM_PER_CLASS = 1
BATCH_NUM_PER_CLASS = 15
EPISODE = 500000
TEST_EPISODE = 600

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram, DEVICE)
rn_trainer = train.RNTrainer(metatrain_character_folders, metatest_character_folders,
                             FEATURE_DIM, RELATION_DIM, CLASS_NUM,
                             SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS, DEVICE)


class_folders_ids = random.sample([i for i in range(0, A_MAX)], CLASS_NUM)
state, _ = rn_trainer.optimize(0, class_folders_ids)
last_accuracy = 0.0
for r in range(EPISODE):
    action = trainer.get_exploration_action(state)
    # if _ep%5 == 0:
    # 	# validate every 5th episode
    # 	action = trainer.get_exploitation_action(state)
    # else:
    # 	# get action based on observation, use exploration policy here
    # 	action = trainer.get_exploration_action(state)

    new_observation, reward = rn_trainer.optimize(r, action)

    if (r+1) % MAX_STEPS == 0:
        test_accuracy = rn_trainer.validate(TEST_EPISODE)
        if test_accuracy > last_accuracy:
            rn_trainer.save_models()
            print("save networks for episode:", r)
            last_accuracy = test_accuracy
    # # dont update if this is validation
    # if _ep%50 == 0 or _ep>450:
    # 	continue

    new_state = new_observation
    # push this exp in ram
    ram.add(state.cpu().numpy(), action.cpu().numpy(),
            reward, new_state.cpu().numpy())

    observation = new_observation

    # perform optimization
    trainer.optimize()

    if (r+1) % MAX_STEPS == 0:
        gc.collect()
        trainer.save_models(r)


print('Completed episodes')

