from typing import List
from dataclasses import dataclass, field
from tensorflow_caney.model.config.cnn_config import Config


@dataclass
class CloudMaskConfig(Config):

    test_classes: List[str] = field(
        default_factory=lambda: ['no-cloud', 'cloud', 'thin-cloud']
    )
