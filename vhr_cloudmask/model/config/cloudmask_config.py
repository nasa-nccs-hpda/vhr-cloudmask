from typing import List
from dataclasses import dataclass, field
from tensorflow_caney.model.config.cnn_config import Config


@dataclass
class CloudMaskConfig(Config):

    # model parameters from Huggingface
    hf_repo_id: str = 'nasa-cisto-data-science-group/vhr-cloudmask'
    hf_model_filename: str = 'cloudmask-vietnam-senegal-46-0.04.hdf5'

    # test classes definition for validation
    test_classes: List[str] = field(
        default_factory=lambda: ['no-cloud', 'cloud', 'thin-cloud']
    )
