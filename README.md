# Gated Convolutional Channel-wise Self-Attention Network (GC<sup>2</sup>SA-Net)
## Self-Attentive Contrastive Learning for Conditioned Periocular and Face Biometrics
### Published in Transaction of Information Forensics and Security (DOI: 10.1109/TIFS.2024.3361216)
[[Paper Link]](https://ieeexplore.ieee.org/document/10418204)

![Network Architecture](GC2SA_Net_Architecture.jpg?raw=true "GC<sup>2</sup>SA-Net")

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
    * grad_cam.py - Plot GradCAM images. For usage, store all images in a single folder, and change the path accordingly. More details of usage in the file's main function.
    * identification.py - Evaluates Rank-1 Identification Rate (IR).
    * plot_cmc_roc_sota.ipynb - Notebook to plot CMC and ROC curves side-by-side, based on generated `.pt` files from `cmc_eval.py` and `roc_eval.py`. Graph is generated in `graphs` directory.
    * plot_tSNE.ipynb - Notebook to plot t-SNE images based on the 10 identities of periocular-face toy examples. Example of text file (which correlates to the image paths) are in `data/tsne/img_lists`.
    * roc_eval.py - Evaluates and generates Receiver Operating Characteristic (ROC) curve, which are saved as `.pt` files in `data` directory. Use these `.pt` files to generate ROC curves.
    * verification.py - Evaluates Verification Equal Error Rate (EER).
- graphs: Directory where graphs are generated.
    * _CMC and ROC curve file is generated in this directory._
- logs: Directory where logs are generated.
    * _Logs will be generated in this directory. Each log folder will contain backups of training files with network files and hyperparameters used._
- models: Directory to store pre-trained models. Trained models are also generated in this directory.
    * __**INSERT PRE-TRAINED MODELS HERE. The base MobileFaceNet for fine-tuning the GC<sup>2</sup>SA-Net can be downloaded in [this link](https://www.dropbox.com/scl/fo/sx61beaupkwa1574fst2z/h?rlkey=onwf8vji3h20og0w7s6sxznlc&dl=0).**__
    * _Trained models will also be stored in this directory._
- network: Contains loss functions and network related files.
    * `facexzoo_network` - Directory contains architecture files from [FaceXZoo](https://github.com/JDAI-CV/FaceX-Zoo/tree/main/training_mode) repository on GitHub. All pre-trained weights are fairly trained. The weight files can be downloaded from the repository, or [here](https://www.dropbox.com/scl/fo/rnmj0n572gmfkshfplk6u/h?rlkey=lze6kbg2q0mcdrimz5qlkdjqw&dl=0).
    * `SOTA` - Directory contains architecture files that are used for State-of-the-Art (SOTA) comparison, namely PF-GLSR, CMB-Net, and HA-ViT (see table below). _Since HA-ViT has its own data loader and has a slightly different setting, the `HA_ViT` directory contains its own data loader and evaluation (identification+CMC and verification+ROC) files for simplicity._
    * flops_counter.py - Counter for Floating Point Operations (FLOPs) and number of parameters for architectures. They are stored in the files `macs_dict.pt` and `params_dict.pt`
    * gc2sa_net.py - Architecture file for GC<sup>2</sup>SA-Net.
    * load_model.py - Loads pre-trained weights based on a given model.
    * logits.py - Contains some loss functions that are used.
- __training:__ Main files for training GC<sup>2</sup>SA-Net.
    * main.py - Main file to run for training. Settings and hyperparameters are based on the files in `configs` directory.
    * train.py - Training file that is called from `main.py`. Gets batch of dataloader and contains criterion for loss back-propagation.

### Pre-requisites (requirements):
Check `requirements.txt` file which was generated using `pip list --format=freeze > requirements.txt` command. These files are slightly filtered, so there may be redundant packages.
Download dataset (training and testing) from [this link](https://www.dropbox.com/s/bfub8fmc44tvcxb/periocular_face_dataset.zip?dl=0). Password is _conditional\_biometrics_.

### Training:
1. Ensure that datasets are located in `data` directory. Configure `datasets_config.py` file to point to this data directory.
2. Change hyperparameters accordingly in `params.py` file. The set values used are the default.
3. Run `main.py` file. The training should start immediately.
4. Testing will be performed automatically after training is done, but it is possible to perform testing on an already trained model (see next section).

### Testing:
0. Pre-trained models for fine-tuning or testing can be downloaded from [this link](https://www.dropbox.com/s/g8gn4x4wp0svyx5/pretrained_models.zip?dl=0). Password is _conditional\_biometrics_.
1. Based on the (pre-)trained models in the `models(/pretrained)` directory, load the correct model and the architecture (in `network` directory) using `load_model.py` file. Change the file accordingly in case of different layer names, etc.
2. Evaluation:
    * Cumulative Matching Characteristic (CMC) curve: Run `cmc_eval.py`. Based on the generated `.pt` files in `data` directory, run `plot_cmc_roc_sota.ipynb` to generate CMC graph.
    * Identification: Run `identification.py`. Only Rank-1 IR values will be displayed.
    * Receiver Operating Characteristic (ROC) curve: Run `roc_eval.py`. Based on the generated `.pt` files in `data` directory, run `plot_cmc_roc_sota.ipynb` to generate ROC graph.
    * Verification: Run `verification.py`. Only EER values will be displayed.
3. Visualization:
    * Gradient-weighted Class Activation Mapping (Grad-CAM): Run `grad_cam.py`, based on the selected images that are stored in a directory. The images will be generated in the `graphs` directory.
    * t-distributed stochastic neighbor embedding (t-SNE) : Run the Jupyter notebook accordingly. Based on the generated text file (which is generated by the notebook as well) in `data/tsne/img_lists`, 10 toy identities are selected to plot the t-SNE points, which will be generated in the `graphs` directory.

### Comparison with State-of-the-Art (SOTA) models

| Method | Intra-Modal Rank-1 IR (%) <br> (Periocular) | Intra-Modal Rank-1 EER (%) <br> (Periocular) | Intra-Modal EER (%) <br> (Periocular Gallery) | Inter-Modal EER (%) <br> (Periocular-Face) |
| --- | --- | --- | --- | --- |
| [PF-GLSR](https://www.dropbox.com/scl/fo/o7rxtbws8g3fmhwkg0f08/h?rlkey=083q0xzibpsfubxmt3d31pa8d&dl=0) [(Paper)](https://ieeexplore.ieee.org/document/9159854) | 79.03 | 15.56 | N/A | N/A |
| [CMB-Net](https://www.dropbox.com/scl/fo/o7rxtbws8g3fmhwkg0f08/h?rlkey=083q0xzibpsfubxmt3d31pa8d&dl=0) [(Paper)](https://ieeexplore.ieee.org/document/9956636) | 86.96 | 9.62 | 77.26 | 9.80 |
| [HA-ViT](https://www.dropbox.com/scl/fo/o7rxtbws8g3fmhwkg0f08/h?rlkey=083q0xzibpsfubxmt3d31pa8d&dl=0) [(Paper)](https://ieeexplore.ieee.org/document/10068230) | 77.75 | 11.39 | 64.72 | 13.14 |
| [GC<sup>2</sup>SA-Net](https://www.dropbox.com/scl/fo/j7tfsk61jz6dch8hyl1hp/h?rlkey=b22nw4ff5kelu5ivti7ioy1mr&dl=0) | 93.63 | 6.39 | 90.77 | 6.50 |