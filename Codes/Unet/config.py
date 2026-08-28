import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 128

BATCH_SIZE = 8

LR = 1e-3

EPOCHS = 40

NUM_CLASSES = 1

TRAIN_IMG_DIR = "Datasets/train_images"
TRAIN_MASK_DIR = "Datasets/train_masks"

TEST_IMG_DIR = "Datasets/test_images"
TEST_MASK_DIR = "Datasets/test_masks"


MODEL_SAVE_PATH = "unet_kvasir.pth"


