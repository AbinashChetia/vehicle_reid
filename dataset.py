from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import os

class VehicleReIDDataset(Dataset):
    def __init__(self, name, root_dir, labels_file=None, transform=None):
        """
        root_dir: Directory containing images in structure:
        transform: Image transformations
        """
        if name not in ['prep_VeRi_CARLA', 'VeRi_CARLA', 'VRIC']:
            raise NotImplementedError(f'{name} dataset is not supported yet.')
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        if name == 'prep_VeRi_CARLA':
            '''
            root_dir/
                    ├── vehicle_id1/
                    │   ├── img1.jpg
                    │   ├── img2.jpg
                    ├── vehicle_id2/
                    │   ├── img3.jpg
                    │   ├── img4.jpg
            '''
            self.vehicle_ids = os.listdir(root_dir)
            # Collect all image paths and labels
            for vehicle_id in self.vehicle_ids:
                if not os.path.isdir(os.path.join(root_dir, vehicle_id)):
                    continue
                vehicle_folder = os.path.join(root_dir, vehicle_id)
                images = os.listdir(vehicle_folder)
                for img in images:
                    self.image_paths.append(os.path.join(vehicle_folder, img))
                    self.labels.append(vehicle_id)
            # Convert vehicle IDs to numerical labels
            self.label_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}
            self.labels = [self.label_to_idx[label] for label in self.labels]

        elif name == 'VeRi_CARLA':
            '''
            root_dir/
                    ├── datetime1_cameraID1_vehicleID1.jpg
                    ├── datetime2_cameraID2_vehicleID2.jpg
            '''
            img_filenames = os.listdir(root_dir)
            self.vehicle_ids = list(set([img_filename.split('_')[-1] for img_filename in img_filenames]))
            for img_filename in img_filenames:
                self.image_paths.append(os.path.join(root_dir, img_filename))
                self.labels.append(img_filename.split('_')[-1].split('.')[0])
            # Convert vehicle IDs to numerical labels
            self.label_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}
            self.labels = [self.label_to_idx[label] for label in self.labels]
        elif name == 'VRIC':
            '''
            VRIC_dataset/
                        ├── gallery_images/
                        │   ├── img1.jpg
                        │   ├── img2.jpg
                        ├── probe_images/
                        │   ├── img1.jpg
                        │   ├── img2.jpg
                        |── train_images/
                        │   ├── img1.jpg
                        │   ├── img2.jpg
                        |── vric_gallery.txt - [img_path] [label] [camera_id]
                        |── vric_probe.txt - [img_path] [label] [camera_id]
                        |── vric_train.txt - [img_path] [label] [camera_id]
                        |── README.txt
            '''
            if labels_file is None:
                raise ValueError('Labels file must be provided for VRIC dataset.')
            with open(labels_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    img_path, label, _ = line.strip().split()
                    self.image_paths.append(os.path.join(root_dir, img_path))
                    self.labels.append(label)
            # Convert vehicle IDs to numerical labels
            self.label_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}
            self.labels = [self.label_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    
def dataloader_train_test_split(full_dataset, test_split=0.2, shuffle_dataset=True, batch_size=32, random_seed=42):
    '''
    Splits a PyTorch DataLoader into training and test DataLoaders.

    Parameters:
    -----------
    full_dataset: torch.utils.data.Dataset
        The original PyTorch Dataset.
    test_split: float
        The fraction of the dataset to use for testing.
    shuffle_dataset: bool
        Whether to shuffle the dataset before splitting.
    batch_size: int
        Batch size for the created DataLoaders.
    random_seed: int
        Random seed for reproducibility.

    Returns:
    --------
    train_loader: torch.utils.data.Dataloader
        DataLoader for the training set.
    val_loader: torch.utils.data.Dataloader
        DataLoader for the validation set.
    '''

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    test_size = int(test_split * dataset_size)
    train_indices, test_indices = indices[test_size:], indices[:test_size]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, test_loader