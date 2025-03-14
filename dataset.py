from torch.utils.data import Dataset
from PIL import Image
import os

class VehicleReIDDataset(Dataset):
    def __init__(self, name, root_dir, transform=None):
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
            pass
        elif name == 'VRIC':
            pass

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label