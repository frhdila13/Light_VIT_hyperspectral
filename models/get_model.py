from .cnn2d import cnn2d
from .sprn import SPRN
from .cnn3d import cnn3d
from .hybridsn import hybridsn
from .spectralformer import spectralformer
from .ssftt import ssftt
from .gaht import gaht
from .gscvit import gscvit
from .morphFormer import morphFormer
from .caevt import caevt
def get_model(model_name, dataset_name, patch_size, num_classes, num_bands):
    if model_name == 'cnn2d':
        model = gscvit(num_classes=num_classes, num_bands=num_bands, patch_size=patch_size)

    elif model_name == 'sprn':
        model = SPRN(num_classes=num_classes, num_bands=num_bands, patch_size=patch_size)

    elif model_name == 'cnn3d':
        model = gscvit(num_classes=num_classes, num_bands=num_bands, patch_size=patch_size)

    elif model_name == 'hybridsn':
        model = hybridsn(num_classes=num_classes, num_bands=num_bands, patch_size=patch_size)

    elif model_name == 'spectralformer':
        model = spectralformer(num_classes=num_classes, num_bands=num_bands, patch_size=patch_size)

    elif model_name == 'ssftt':
        model = ssftt(num_classes=num_classes, num_bands=num_bands, patch_size=patch_size)

    elif model_name == 'gaht':
        model = gaht(num_classes=num_classes, num_bands=num_bands, patch_size=patch_size)

    elif model_name == 'morphFormer':
        model = morphFormer(16, 80, 10, False, 8)

    elif model_name == 'gscvit':
        model = gscvit(num_classes=num_classes, num_bands=num_bands, patch_size=patch_size)

    elif model_name == 'caevt':
        model = caevt(num_classes=num_classes, num_bands=num_bands, patch_size=patch_size)

    else:
        raise KeyError("{} model is not supported yet".format(model_name))

    return model

