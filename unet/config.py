NUM_FRAMES = 22
IN_CHANNELS = 3  # RGB
NUM_CLASSES = 49  # 48 object classes + 1 background class
BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
MAX_PATIENCE = 3
EARLY_STOP = True

MODEL_NAME = 'best_unet_weights.pth'

# use absolute paths to avoid FileNotFound errors
TRAIN_DATA_DIR = '/scratch/rn2214/Deep_Learning_Final_BS3/data/train'
VAL_DATA_DIR = '/scratch/rn2214/Deep_Learning_Final_BS3/data/val'
UNLABELED_DATA_DIR = '/scratch/rn2214/Deep_Learning_Final_BS3/data/unlabeled'
LABELED_OUT_DIR = '/scratch/rn2214/labeled'
