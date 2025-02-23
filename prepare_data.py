import os
import shutil
from tqdm import tqdm

root_dir = './data/VeRi_CARLA_dataset/'
out_dir = './data/prepared_VeRi_CARLA_dataset/'
train_test_split = 0.8

############### Classify images by vehicle id ################

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print('Classifying images by vehicle id...')

for d in os.listdir(root_dir):
    if not os.path.isdir(os.path.join(root_dir, d)):
        continue
    current_path = os.path.join(root_dir, d)
    print(f'Processing {current_path}')
    for f in tqdm(os.listdir(current_path)):
        if not os.path.isfile(os.path.join(current_path, f)):
            continue
        f_name, f_ext = os.path.splitext(f)
        if f_ext.lower() in ['.jpg', '.jpeg', '.png']:
            # From dataset documentation: If the filename is `20220711212617_24_9.jpg`, then `20220711212617` is the datetime, `24` is the camera id and `9` represents the vehicle id. 
            vehicle_id = f_name.split('_')[-1]
            vehicle_dir = os.path.join(out_dir, vehicle_id)
            if not os.path.exists(vehicle_dir):
                os.makedirs(vehicle_dir)
            # Copy the file to the new directory
            shutil.copy(os.path.join(current_path, f), os.path.join(vehicle_dir, f))

print('Completed classifying images by vehicle id.')

############### Create train and test sets ################

print('Creating train and test sets...')

n_train_imgs = 0
n_test_imgs = 0

vehicle_ids = os.listdir(out_dir)
train_dir = os.path.join(out_dir, 'train')
test_dir = os.path.join(out_dir, 'test')

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

split_idx = int(len(vehicle_ids) * train_test_split)
train_vehicle_ids = vehicle_ids[:split_idx]
test_vehicle_ids = vehicle_ids[split_idx:]

for vehicle_id in train_vehicle_ids:
    n_train_imgs += len(os.listdir(os.path.join(out_dir, vehicle_id)))
    shutil.move(os.path.join(out_dir, vehicle_id), os.path.join(train_dir, vehicle_id))

for vehicle_id in test_vehicle_ids:
    n_test_imgs += len(os.listdir(os.path.join(out_dir, vehicle_id)))
    shutil.move(os.path.join(out_dir, vehicle_id), os.path.join(test_dir, vehicle_id))

print('Completed creating train and test sets.')

print('Data preparation summary:')
print('-------------------------')
print(f'Total vehicle identities: {len(vehicle_ids)}')
print(f'Number of vehicle IDs in train set: {len(train_vehicle_ids)}')
print(f'Number of vehicle IDs in test set: {len(test_vehicle_ids)}')
print(f'Total images in train set: {n_train_imgs}')
print(f'Total images in test set: {n_test_imgs}')
print(f'Train-test ratio by vehicle IDs: {len(train_vehicle_ids) / (len(train_vehicle_ids) + len(test_vehicle_ids)):.2f}')
print(f'Train-test ratio by images: {n_train_imgs / (n_train_imgs + n_test_imgs):.2f}')
print(f'Train set saved to: {train_dir}')
print(f'Test set saved to: {test_dir}')

############################## LAST OUTPUT ##############################

# Classifying images by vehicle id...
# Processing ./data/VeRi_CARLA_dataset/image_query
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 424/424 [00:00<00:00, 2780.18it/s]
# Processing ./data/VeRi_CARLA_dataset/image_gallery
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3823/3823 [00:01<00:00, 3181.83it/s]
# Processing ./data/VeRi_CARLA_dataset/image_train
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50949/50949 [00:17<00:00, 2905.81it/s]
# Completed classifying images by vehicle id.
# Creating train and test sets...
# Completed creating train and test sets.
# Data preparation summary:
# -------------------------
# Total vehicle identities: 605
# Number of vehicle IDs in train set: 484
# Number of vehicle IDs in test set: 121
# Total images in train set: 43645
# Total images in test set: 11551
# Train-test ratio by vehicle IDs: 0.80
# Train-test ratio by images: 0.79
# Train set saved to: ./data/prepared_VeRi_CARLA_dataset/train
# Test set saved to: ./data/prepared_VeRi_CARLA_dataset/test