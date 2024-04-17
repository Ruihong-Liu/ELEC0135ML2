"""
This is the code for task 1 of image super resolution task, Using EDSR
First of all stating with import all the Libaray
"""
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

"""
First part is to import all the images and pre-processed it suitable for further training
"""
class SuperResolutionDataset(Dataset):
    
    def __init__(self, lr_dir, hr_dir, scale_suffix='x2', transform=None):
        # path of low resolution image
        self.lr_dir = lr_dir 
        # path of high resolution image
        self.hr_dir = hr_dir 
        self.scale_suffix = scale_suffix 
        self.transform = transform
        self.lr_images = os.listdir(self.lr_dir)
    
    def __len__(self):
        return len(self.lr_images)
    
    def __getitem__(self, idx):
        lr_image_name = self.lr_images[idx]
        # remove the suffix from low resolution image name
        hr_image_name = lr_image_name.replace(self.scale_suffix, '')
        lr_image_path = os.path.join(self.lr_dir, lr_image_name)
        hr_image_path = os.path.join(self.hr_dir, hr_image_name)
        # use pillow to convert RGB value
        lr_image = Image.open(lr_image_path).convert('RGB')
        hr_image = Image.open(hr_image_path).convert('RGB')
        # check if the file is empty
        if self.transform is not None:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)
        return lr_image, hr_image
# reshape value
image_size = (256, 256)

# resize the image and transform to Tensor
transform = transforms.Compose([
    transforms.Resize(image_size), 
    transforms.ToTensor(),
])
"""
Path here need to be changed
"""
# training data
train_dataset = SuperResolutionDataset(lr_dir=r'Datasets\DIV2K_train_LR_bicubic_X2\DIV2K_train_LR_bicubic\X2', hr_dir=r'Datasets\DIV2K_train_HR\DIV2K_train_HR', scale_suffix='x2', transform=transform)
validation_dataset = SuperResolutionDataset(lr_dir=r'Datasets\DIV2K_valid_LR_bicubic_X2\DIV2K_valid_LR_bicubic\X2', hr_dir=r'Datasets\DIV2K_valid_HR\DIV2K_valid_HR', scale_suffix='x2', transform=transform)
# validation data
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False)

"""
Sample image printing and shown
"""
# for shown random low and high resolution images
def show_image_pairs(lr_images, hr_images, num_pairs):
    if num_pairs > len(lr_images):
        print(f"Requested {num_pairs} pairs, but only {len(lr_images)} pairs are available.")
        num_pairs = len(lr_images)
    
    fig, axes = plt.subplots(num_pairs, 2, figsize=(10, 5 * num_pairs))
    
    for i in range(num_pairs):
        if num_pairs > 1:
            ax1, ax2 = axes[i]
        else:
            ax1, ax2 = axes
        
        # show low resolution image
        lr_image = lr_images[i].permute(1, 2, 0)
        ax1.imshow(lr_image)
        ax1.set_title('Low Resolution')
        ax1.axis('off')
        
        # show high resolution image
        hr_image = hr_images[i].permute(1, 2, 0)
        ax2.imshow(hr_image)
        ax2.set_title('High Resolution')
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig("image.png")

# load data
dataiter = iter(train_loader)
lr_images, hr_images = next(dataiter)

# show images
show_image_pairs(lr_images, hr_images, num_pairs=1)
"""
Model structure desgin 
"""
# class of Residual Block structure conv__ReLU__Conv
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x 
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual

class EDSR(nn.Module):
    def __init__(self, scale_factor=2, num_channels=3, num_features=64, num_res_blocks=40):
        super(EDSR, self).__init__()
        self.entry_layer = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_res_blocks)])
        self.mid_layer = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * scale_factor ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.entry_layer(x)
        residual = x
        x = self.res_blocks(x)
        x = self.mid_layer(x) + residual
        x = self.upsample(x)
        x = self.output_layer(x)
        return x
"""
PSRN calculation
"""
def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    return 20 * torch.log10(max_val / torch.sqrt(mse))
"""
Model training
"""
# Setting up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently using：{'GPU' if device.type == 'cuda' else 'CPU'}")

# Model, loss function, and optimizer setup
model = EDSR().to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training parameters
num_epochs = 20
train_losses, val_losses, val_psnrs = [], [], []

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0     
    train_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}, Train')
    for lr_imgs, hr_imgs in train_iterator:
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
        optimizer.zero_grad()
        outputs = model(lr_imgs)
        hr_imgs_resized = F.interpolate(hr_imgs, size=(outputs.size(2), outputs.size(3)), mode='bilinear', align_corners=False)
        loss = criterion(outputs, hr_imgs_resized)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_iterator.set_postfix(loss=loss.item())
    train_losses.append(train_loss / len(train_loader))

    model.eval()
    val_loss, total_psnr = 0.0, 0.0
    validation_iterator = tqdm(validation_loader, desc=f'Epoch {epoch+1}/{num_epochs}, Validation ')
    with torch.no_grad():
        for lr_imgs, hr_imgs in validation_iterator:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            outputs = model(lr_imgs)
            hr_imgs_resized = F.interpolate(hr_imgs, size=(outputs.size(2), outputs.size(3)), mode='bilinear', align_corners=False)
            loss = criterion(outputs, hr_imgs_resized)
            val_loss += loss.item()
            total_psnr += psnr(outputs, hr_imgs_resized).item()
            validation_iterator.set_postfix(loss=loss.item(), psnr=total_psnr)
    avg_val_loss = val_loss / len(validation_loader)
    avg_psnr = total_psnr / len(validation_loader)
    val_losses.append(avg_val_loss)
    val_psnrs.append(avg_psnr)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train loss: {train_losses[-1]:.4f}, validation loss: {avg_val_loss:.4f}, PSNR: {avg_psnr:.2f}")

"""
ploting results
"""

def plot_metrics(train_losses, val_losses, val_psnrs):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(8, 5))
    ## plotting of loss function
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("Task 1 result L1 loss funcion .png")

    # plotting of PSNR
    plt.figure(figsize=(8, 5))  # 限制大小为10x5英寸
    plt.plot(epochs, val_psnrs, 'go-', label='Validation PSNR')
    plt.title('Validation PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.legend()
    plt.savefig("Task 1 PSRN result .png")

# plot images
plot_metrics(train_losses, val_losses, val_psnrs)

"""
Testing the modle using other data
"""

#testing data path
test_dataset = SuperResolutionDataset(
    lr_dir=r'Datasets\DIV2K_valid_LR_unknown_X2\DIV2K_valid_LR_unknown\X2',
    hr_dir=r'Datasets\DIV2K_valid_HR\DIV2K_valid_HR',
    scale_suffix='x2',  # scale factor
    transform=transform
)

# load data
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# testing model 
def evaluate_model(model, test_loader, device):
    model.eval()
    test_loss, total_psnr = 0.0, 0.0

    with torch.no_grad():
        for lr_imgs, hr_imgs in tqdm(test_loader, desc='Testing'):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            outputs = model(lr_imgs)
            hr_imgs_resized = F.interpolate(hr_imgs, size=(outputs.size(2), outputs.size(3)), mode='bilinear', align_corners=False)
            loss = criterion(outputs, hr_imgs_resized)
            test_loss += loss.item()
            total_psnr += psnr(outputs, hr_imgs_resized).item()
    
    avg_test_loss = test_loss / len(test_loader)
    avg_psnr = total_psnr / len(test_loader)
    print(f"Test loss: {avg_test_loss:.4f}, Test PSNR: {avg_psnr:.2f}")

# testing
evaluate_model(model, test_loader, device)

"""
Plot image after super resolution
"""
def show_prediction_comparison(model, data_loader, num_pairs, device):
    model.eval()  
    fig, axes = plt.subplots(num_pairs, 2, figsize=(10, 5 * num_pairs))  # iamge size

    with torch.no_grad():
        for i, (lr_imgs, hr_imgs) in enumerate(data_loader):
            if i >= num_pairs:
                break
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            outputs = model(lr_imgs)  # format predicted image
            hr_imgs_resized = F.interpolate(hr_imgs, size=(outputs.size(2), outputs.size(3)), mode='bilinear', align_corners=False)

            hr_real = hr_imgs_resized[0].permute(1, 2, 0).cpu().numpy()
            hr_predicted = outputs[0].permute(1, 2, 0).cpu().numpy()

            if num_pairs > 1:
                ax1, ax2 = axes[i]
            else:
                ax1, ax2 = axes

            ax1.imshow(hr_real)
            ax1.set_title('High Resolution (True)')
            ax1.axis('off')

            ax2.imshow(hr_predicted)
            ax2.set_title('High Resolution (Predicted)')
            ax2.axis('off')

    plt.tight_layout()
    plt.savefig("Task 1 prediction_comparison .png")

# compare preficted image and true value
show_prediction_comparison(model, validation_loader, num_pairs=1, device=device)