# Usage

The NAVSIM toolkit is used for generating data input for auto-regressive model, and evaluate the output of auto-regressive model.

1. Generate inference json files on `navtrain` or `navtest` for DrivingGPT input:

    ```bash
    python get_json_for_drivinggpt_inference.py
    ```

2. Transform vision and action tokens of single scene from DrivingGPT to vision feature and actual actions:

    ```bash
    python load_va_feature.py
    ```

3. Evaluate the DrivingGPT planning result:

    ```bash
    cd navsim
    bash scripts/evaluation/run_external_agent_pdm_score_evaluation.sh
    ```