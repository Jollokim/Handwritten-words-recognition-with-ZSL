# Handwritten words recognition with ZSL

## Preparing the python environment
python version: Python 3.10.12 (but might work with others as well)
### Create the python environment:

    python3 -m venv <environment_path>

### Installing dependecies
Please activate your environment before doing the next step. The environment activation method is different based on your operating system.

    pip install -r requirements.txt

### Training example:

    python main.py --name ResNet18Phosc_samplerun --mode train --model ResNet18Phosc --epochs 5  --train_csv image_data/IAM_Data/IAM_train.csv --train_folder image_data/IAM_Data/IAM_train --valid_csv image_data/IAM_Data/IAM_valid.csv --valid_folder image_data/IAM_Data/IAM_valid

### Testing example:

    python main.py --name ResNet18Phosc_samplerun --mode test --testing_mode zsl --model ResNet18Phosc --pretrained_weights ResNet18Phosc_samplerun/epoch1.pt --test_csv_seen image_data/IAM_Data/IAM_test_seen.csv --test_folder_seen image_data/IAM_Data/IAM_test --test_csv_unseen image_data/IAM_Data/IAM_test_unseen.csv --test_folder_unseen image_data/IAM_Data/IAM_test

### Best acvhieved results on IAM

seen: 93.78%, unseen: 87.32%

