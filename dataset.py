import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, img_name)).convert("L")

        if self.transform:
            image_np = np.array(image)
            mask_np = (np.array(mask) > 127).astype('float32')
            augmented = self.transform(image=image_np, mask=mask_np)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask
def predict_large_image(image, model, device, crop_size=255, stride=128, threshold=0.5):
    # Load image
    image_np = np.array(image)

    h, w, _ = image_np.shape
    pad_h = (crop_size - h % stride) % stride
    pad_w = (crop_size - w % stride) % stride

    # Pad image to fit complete stride steps
    padded_img = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    padded_h, padded_w, _ = padded_img.shape

    # Output mask accumulation and count map
    pred_sum = np.zeros((padded_h, padded_w), dtype=np.uint16)
    pred_count = np.zeros((padded_h, padded_w), dtype=np.uint16)

    transform = A.Compose([
        A.Resize(crop_size, crop_size),
        A.Normalize(),
        ToTensorV2(),
    ])

    model.eval()
    with torch.no_grad():
        for i in range(0, padded_h - crop_size + 1, stride):
            for j in range(0, padded_w - crop_size + 1, stride):
                crop = padded_img[i:i+crop_size, j:j+crop_size, :]
                augmented = transform(image=crop, mask=np.zeros((crop_size, crop_size)))
                crop_tensor = augmented["image"].unsqueeze(0).to(device)

                pred = model(crop_tensor)
                pred = torch.sigmoid(pred)
                pred_bin = (pred > threshold).float()
                pred_bin_np = pred_bin.squeeze().cpu().numpy()

                # Accumulate results
                pred_sum[i:i+crop_size, j:j+crop_size] += pred_bin_np.astype(np.uint16)
                pred_count[i:i+crop_size, j:j+crop_size] += 1

    # Majority voting
    final_mask = (pred_sum >= (pred_count / 2)).astype(np.uint8) *  255

    # Crop to original size
    final_mask = final_mask[:h, :w]

    return final_mask
    
import segmentation_models_pytorch as smp

model = smp.UnetPlusPlus(
    encoder_name="resnet34",        # can be 'efficientnet-b0', etc.
    encoder_weights="imagenet",     # use "None" if training from scratch
    in_channels=3,
    classes=1,                      # binary segmentation
)
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2(),
], additional_targets={"mask": "mask"})

# Datasets and loaders
train_dataset = SegmentationDataset("dataset/images", "dataset/masks", transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model, loss, optimizer
model = model.to(device)
loss_fn = smp.losses.DiceLoss("binary")
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(15):
    print(f"\nEpoch {epoch + 1}/15")
    model.train()
    batch_count = 0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1).float()

        preds = model(images)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_count += 1
        if batch_count % 10 == 0:
            print(f"  Batch {batch_count}, Loss: {loss.item():.4f}")
from torch.utils.data import DataLoader
import torch
import os
import numpy as np
from torchvision.utils import save_image

# Create test dataset
test_dataset = SegmentationDataset("dataset/test", "dataset/test_mask", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Set model to eval mode
model.eval()

# Optional: make a folder to save predicted masks
os.makedirs("predictions", exist_ok=True)

# For evaluation
dice_scores = []

with torch.no_grad():
    from torchvision.transforms.functional import to_pil_image

    from PIL import Image

for i, (image, mask) in enumerate(test_loader):
    image_tensor = image.squeeze(0).cpu()  # [3, H, W]
    pil_image = to_pil_image(image_tensor)

    # Predict using sliding window
    pred = predict_large_image(pil_image, model, device, crop_size=256, stride=128, threshold=0.5)

    # Convert np array (0 or 255) to float tensor in range [0.0, 1.0]
    pred_tensor = torch.from_numpy(pred / 255.0).unsqueeze(0).to(device).float()  # [1, H, W]

    # Save prediction
    save_image(pred_tensor, f"predictions/pred_{i}.png")

    # Prepare ground truth mask
    mask = mask.to(device).unsqueeze(1).float()  # [1, 1, H, W]

    # Resize mask to match prediction shape if needed
    if mask.shape[-2:] != pred_tensor.shape[-2:]:
        mask = torch.nn.functional.interpolate(mask, size=pred_tensor.shape[-2:], mode='nearest')

    # Compute Dice score
    intersection = (pred_tensor * mask).sum()
    union = pred_tensor.sum() + mask.sum()
    dice = (2. * intersection) / (union + 1e-7)
    dice_scores.append(dice.item())

    print(f"Test image {i}: Dice score = {dice.item():.4f}")


# Summary
mean_dice = np.mean(dice_scores)
torch.save(model.state_dict(), "unetpp_trained.pth")
print(f"\nAverage Dice score on test set: {mean_dice:.4f}")
