SEED = 42
DEFAULT_DATA_PATH = "/scratch/tk3309/dl_data/dataset/"
DEFAULT_MODEL_CONFIG = {
    # For MetaVP models, the most important hyperparameters are: 
    # N_S, N_T, hid_S, hid_T, model_type
    'in_shape': [11, 3, 160, 240],
    'hid_S': 64,
    'hid_T': 512,
    'N_S': 4,
    'N_T': 8,
    'model_type': 'gSTA',
}