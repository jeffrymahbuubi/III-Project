from pathlib import Path
import glob
from PIL import Image

from sklearn.model_selection import train_test_split, KFold

from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms


class SignalDataset2D(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.image_files = []
        self.labels = []

        # Verify the existence of class directories (0-7)
        for class_label in range(8):  # Check for classes 0-7
            class_dir = self.data_dir / str(class_label)
            if not class_dir.exists() or not class_dir.is_dir():
                raise FileNotFoundError(f"Class directory '{class_label}' does not exist in {data_dir}.")

            # Load all .png files for the class
            class_files = list(class_dir.glob("*.png"))
            self.image_files.extend(class_files)
            self.labels.extend([class_label] * len(class_files))

        if not self.image_files:
            raise ValueError(f"No images found in the dataset directory: {data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = Image.open(self.image_files[index]).convert("RGB")
        label = self.labels[index]

        # Apply transforms if provided, otherwise return original image
        if self.transforms:
            image = self.transforms(image)

        return image, label

def get_model_path(indices, model_name, model_dir, model_suffix="*.pth"):
    if indices < 0 or indices >= len(model_name):
        raise ValueError("Index out of range.")
    model_folder = Path(model_dir) / model_name[indices]
    pth_files = glob.glob(str(model_folder / model_suffix))

    if not pth_files:
        raise FileNotFoundError(f"No '.pth' files found in {model_folder}.")

    print(f"Loaded Model from {pth_files[0]}")
    return pth_files[0]  # Return the first matching `.pth` file

def get_data(data_dir, batch_size=8, test_size=0.2, num_workers=0, random_state=42):
    # Define transforms for the images
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
    ])

    # Initialize the dataset
    dataset = SignalDataset2D(data_dir=data_dir, transforms=data_transforms)

    # Create indices for train-test split
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=dataset.labels
    )

    # Create subsets for train and validation
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # Create DataLoader objects
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def get_data_kfold(data_dir, batch_size=8, num_workers=0, num_folds=5, fold_index=0, random_state=42):
    # Define transforms for the images
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
    ])

    # Initialize the dataset
    dataset = SignalDataset2D(data_dir=data_dir, transforms=data_transforms)

    # Prepare the labels for stratification
    labels = dataset.labels

    # Set up StratifiedKFold
    skf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)

    # Get the train/val indices for the chosen fold
    all_splits = list(skf.split(range(len(dataset)), labels))

    if fold_index < 0 or fold_index >= num_folds:
        raise ValueError(f"Invalid fold_index {fold_index}. Must be between 0 and {num_folds - 1}.")

    train_indices, val_indices = all_splits[fold_index]

    # Create subsets for train and validation
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # Create DataLoader objects
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader