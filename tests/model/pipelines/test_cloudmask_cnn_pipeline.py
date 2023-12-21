import pytest
from vhr_cloudmask import CloudMaskPipeline


@pytest.mark.parametrize(
    "config_filename, output_dir, inference_regex_list, " +
    "expected_experiment_name, expected_experiment_type",
    [(
        'tests/data/cloudmask_test.yaml',
        'tests/output',
        ['tests/data/*.tif'],
        'vhr-cloudmask',
        'cloudmask'
    )]
)
def test_cloudmask_pipeline(
            config_filename: str,
            output_dir: str,
            inference_regex_list: list,
            expected_experiment_name: str,
            expected_experiment_type: str
        ):

    # initialize cloud mask object
    cloudmask_pipeline = CloudMaskPipeline(
        config_filename,
        None,
        None,
        output_dir,
        inference_regex_list
    )

    assert expected_experiment_name == \
        cloudmask_pipeline.conf.experiment_name
    assert expected_experiment_type == \
        cloudmask_pipeline.conf.experiment_type
