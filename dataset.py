from torch.utils.data import Dataset
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