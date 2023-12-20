import sys
import pytest
from omegaconf import OmegaConf
from vhr_cloudmask import CloudMaskConfig


@pytest.mark.parametrize(
    "config_filename, expected_experiment_name, expected_experiment_type",
    [(
        'tests/data/cloudmask_test.yaml',
        'vhr-cloudmask',
        'cloudmask'
    )]
)
def test_cloudmask_pipeline(
            config_filename: str,
            expected_experiment_name: str,
            expected_experiment_type: str
        ):

    # initialize cloud mask object
    

    #schema = OmegaConf.structured(Config)
    #conf = OmegaConf.load(filename)

    #try:
    #    conf = OmegaConf.merge(schema, conf)
    #except BaseException as err:
    #    sys.exit(f"ERROR: {err}")

    #assert expected_experiment_name == conf.experiment_name
    #assert expected_experiment_type == conf.experiment_type
