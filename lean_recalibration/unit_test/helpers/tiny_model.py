"""
Tiny CNN model factory used by the test suite.

The four target scripts (main_store_cav.py, main_recalib_custom_by_loading_cav.py,
utils_sensitivity_multiclass.py) call ``torch.load(path, weights_only=False)`` and expect
a *full* pickled ``torch.nn.Module`` object (not a state_dict). To keep the test suite:

  * fast (no real VGG16/ResNet50 forward/backward passes),
  * offline (no torchvision pretrained-weight downloads),
  * robust to unpickling in a fresh subprocess (a custom ``nn.Module`` subclass defined in
    this test package would need to be importable, under the exact same dotted path, by
    whatever process later calls ``torch.load`` -- fragile across OSes/CWDs),

the model below is built *exclusively* from built-in ``torch.nn`` classes
(``Sequential``, ``Conv2d``, ``ReLU``, ``MaxPool2d``, ``AdaptiveAvgPool2d``, ``Flatten``,
``Linear``). Those classes always live under ``torch.nn`` which is guaranteed to be
importable anywhere ``torch`` itself is importable, so pickling/unpickling never depends on
this test package being on ``sys.path``.

Layer names produced (as returned by ``utils.get_model_layers``, which scans for
``nn.Conv2d``/``nn.MaxPool2d`` submodules):
    features.0  -> Conv2d(in_channels, 8)
    features.2  -> MaxPool2d
    features.3  -> Conv2d(8, 16)
    features.5  -> MaxPool2d
    features.6  -> Conv2d(16, 16)
"""

from collections import OrderedDict
import os

import torch
import torch.nn as nn

# Names emitted by utils.get_model_layers(build_tiny_model()) -- kept in one place so
# both the data-generation step and the test/config files agree on the exact strings.
CONV_LAYER_NAMES = ["features.0", "features.2", "features.3", "features.5", "features.6"]

# Model name used everywhere in the tests. main_store_cav.py hard-codes a whitelist of
# accepted --model_name values; "vgg16" is the one we reuse (the *architecture* loaded at
# that path is our tiny network, not real VGG16 -- these scripts never validate architecture,
# they just torch.load() whatever object is on disk).
TEST_MODEL_NAME = "vgg16"


def build_tiny_model(num_classes: int = 3, in_channels: int = 3, seed: int = 42) -> nn.Sequential:
    """Build a tiny, fully deterministic CNN with named 'features'/'classifier' blocks.

    Only uses built-in torch.nn layers so the returned object can be torch.save()'d and
    reloaded via torch.load(..., weights_only=False) in any other process without needing
    this module to be importable there.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = nn.Sequential(OrderedDict([
            ("features", nn.Sequential(
                nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),   # features.0
                nn.ReLU(inplace=True),                                  # features.1
                nn.MaxPool2d(2),                                        # features.2
                nn.Conv2d(8, 16, kernel_size=3, padding=1),             # features.3
                nn.ReLU(inplace=True),                                  # features.4
                nn.MaxPool2d(2),                                        # features.5
                nn.Conv2d(16, 16, kernel_size=3, padding=1),            # features.6
                nn.ReLU(inplace=True),                                  # features.7
                nn.AdaptiveAvgPool2d((4, 4)),                           # features.8
            )),
            ("classifier", nn.Sequential(
                nn.Flatten(),                        # classifier.0
                nn.Linear(16 * 4 * 4, 32),            # classifier.1
                nn.ReLU(inplace=True),                # classifier.2
                nn.Linear(32, num_classes),           # classifier.3
            )),
        ]))
    del generator
    model.eval()
    return model


def save_tiny_model(model: nn.Module, path: str) -> str:
    """torch.save the full model object (not a state_dict) at `path`, creating parent dirs."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model, path)
    return path


def build_and_save(path: str, num_classes: int = 3, in_channels: int = 3, seed: int = 42) -> str:
    model = build_tiny_model(num_classes=num_classes, in_channels=in_channels, seed=seed)
    return save_tiny_model(model, path)
