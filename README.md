# Gated Convolutional Channel-wise Self-Attention Network (GC<sup>2</sup>SA-Net)
## Self-Attentive Contrastive Learning for Conditioned Periocular and Face Biometrics

The project directories are as follows:

- configs: Dataset path configuration file and hyperparameters.
    * datasets_config.py - Directory path for dataset files. Change 'main' in 'main_path' dictionary to point to dataset, e.g., `/home/tiongsik/gc2sa_net/data` (without slash).
    * params.py - Adjust hyperparameters and arguments in this file for training. 
- data: Dataloader functions and preprocessing.
    * __**INSERT DATASET HERE**__
    * _CMC and ROC data dictionaries are generated in this directory._
    * data_loader.py - Generate training and testing PyTorch dataloader. Adjust the augmentations etc. in this file. Batch size of data is also determined here, based on the values set in `params.py`.
- eval: Evaluation metrics (identification and verification). Also contains CMC and ROC evaluations.
    * cmc_eval.py - Evaluates and generates Cumulative Matching Characteristic (CMC) curve, which are saved as `.pt` files in `data` directory. Use these `.pt` files to generate CMC curves.
    * identification.py - Evaluates Rank-1 Identification Rate (IR).
    * plot_cmc_roc_sota.ipynb - Notebook to plot CMC and ROC curves side-by-side, based on generated `.pt` files from `cmc_eval.py` and `roc_eval.py`. Graph is generated in `graphs` directory.
    * roc_eval.py - Evaluates and generates Receiver Operating Characteristic (ROC) curve, which are saved as `.pt` files in `data` directory. Use these `.pt` files to generate ROC curves.
    * verification.py - Evaluates Verification Equal Error Rate (EER).
- graphs: Directory where graphs are generated.
    * _CMC and ROC curve file is generated in this directory._
- logs: Directory where logs are generated.
    * _Logs will be generated in this directory. Each log folder will contain backups of training files with network files used._
- models: Directory to store pre-trained models. Trained models are also generated in this directory.
    * __**INSERT PRE-TRAINED MODELS HERE**__
    * _Trained models will also be stored in this directory._
- network: Contains loss functions and network related files.
    * `facexzoo_network` - Directory contains architecture files from FaceXZoo. Pretrained models can be downloaded from [FaceXZoo](https://github.com/JDAI-CV/FaceX-Zoo/tree/main/training_mode) repository on GitHub.
    * `SOTA` - Directory contains architecture files that are used for State-of-the-Art (SOTA) comparison, namely HA-ViT and CMB-Net. _Since HA-ViT has its own data loader and has a slightly different setting, the `HA_ViT` directory contains its own data loader and evaluation (identification+CMC and verification+ROC) files for simplicity._
    * ael_net.py - Architecture file for AELNet.
    * load_model.py - Loads pre-trained weights based on a given model.
    * logits.py - Contains some loss functions that are used.
- __training:__ Main files for training AELNet.
    * main.py - Main file to run for training. Settings and hyperparameters are based on the files in `configs` directory.
    * train.py - Training file that is called from `main.py`. Gets batch of dataloader and contains criterion for loss back-propagation.

### Pre-requisites (requirements):
Check `environment.yml` file, which was generated using `conda env export > environment.yml --no-builds` command. Else, check `requirements.txt` file which was generated using `pip list --format=freeze > requirements.txt` command. These files are not filtered, so there may be redundant packages.
Download dataset (training and testing) from [this link](https://www.dropbox.com/s/bfub8fmc44tvcxb/periocular_face_dataset.zip?dl=0). Password is _conditional\_biometrics_.

### Training:
1. Ensure that datasets are located in `data` directory. Configure `datasets_config.py` file to point to this data directory.
2. Change hyperparameters accordingly in `params.py` file. The set values used are the default.
3. Run `main.py` file. The training should start immediately.
4. Testing will be performed automatically after training is done, but it is possible to perform testing on an already trained model (see next section).

### Testing:
0. Pre-trained models for fine-tuning can be downloaded from [this link](https://www.dropbox.com/s/g8gn4x4wp0svyx5/pretrained_models.zip?dl=0). Password is _conditional\_biometrics_.
1. Based on the (pre-)trained models in the `models(/pretrained)` directory, load the correct model and the architecture (in `network` directory) using `load_model.py` file. Change the file accordingly in case of different layer names, etc.
2. Evaluation:
    * Cumulative Matching Characteristic (CMC) curve: Run `cmc_eval.py`. Based on the generated `.pt` files in `data` directory, run `plot_cmc_roc_sota.ipynb` to generate CMC graph.
    * Identification: Run `identification.py`. Only Rank-1 IR values will be displayed.
    * Receiver Operating Characteristic (ROC) curve: Run `roc_eval.py`. Based on the generated `.pt` files in `data` directory, run `plot_cmc_roc_sota.ipynb` to generate ROC graph.
    * Verification: Run `verification.py`. Only EER values will be displayed.

### Comparison with State-of-the-Art (SOTA) models

| Method | Rank-1 IR (%) <br> (Periocular Gallery) | Rank-1 IR (%) <br> (Face Gallery) | EER (%) <br> (Periocular-Face) |
| --- | --- | --- | --- |
| [CMB-Net](https://www.dropbox.com/s/apbejkd082dn0tp/CMB-Net.pth?dl=0) | 77.26 | 68.22 | 9.80 |
| [HA-ViT](https://www.dropbox.com/s/hzmsz7kuyvyrf75/HA-ViT.pth?dl=0) | 64.72 | 64.01 | 13.14 |
| [GC<sup>2</sup>-Net]() | 92.47 | 90.71 | 6.31 |
