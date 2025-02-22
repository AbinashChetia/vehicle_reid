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
    shutil.move(os.path.join(out_dir, vehicle_id), os.path.join(train_dir, vehicle_id))

for vehicle_id in test_vehicle_ids:
    shutil.move(os.path.join(out_dir, vehicle_id), os.path.join(test_dir, vehicle_id))

print('Completed creating train and test sets.')

print('Data preparation summary:')
print('-------------------------')
print(f'Total vehicle identities: {len(vehicle_ids)}')
print(f'Length of train set: {len(train_vehicle_ids)}')
print(f'Length of test set: {len(test_vehicle_ids)}')
print(f'Train set saved to: {train_dir}')
print(f'Test set saved to: {test_dir}')