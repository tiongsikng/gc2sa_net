# Gated Convolutional Channel-wise Self-Attention Network (GC<sup>2</sup>SA-Net)
## Self-Attentive Contrastive Learning for Conditioned Periocular and Face Biometrics
### Published in Transaction of Information Forensics and Security (DOI: 10.1109/TIFS.2024.3361216)
[[Paper Link]](https://ieeexplore.ieee.org/document/10418204)

![Network Architecture](GC2SA_Net_Architecture.jpg?raw=true "GC<sup>2</sup>SA-Net")

The project directory is as follows:

- configs: Dataset path configuration file and hyperparameters.
    * datasets_config.py - Directory path for dataset files. Change 'main' in 'main_path' dictionary to point to dataset, e.g., `/home/tiongsik/gc2sa_net/data` (without slash).
    * params.py - Adjust hyperparameters and arguments in this file for training. 
- data: Dataloader functions and preprocessing.
    * __**INSERT DATASET HERE**__
    * _CMC and ROC data dictionaries are generated in this directory._
    * data_loader.py - Generate training and testing PyTorch dataloader. Adjust the augmentations etc. in this file. Batch size of data is also determined here, based on the values set in `params.py`.
- eval: Evaluation metrics (identification and verification). Also contains CMC and ROC evaluations.
    * cmc_eval_identification.py - Evaluates Rank-1 Identification Rate (IR) and generates Cumulative Matching Characteristic (CMC) curve, which are saved as `.pt` files in `data` directory. Use these `.pt` files to generate CMC curves.
    * grad_cam.py - Plot GradCAM images. For usage, store all images in a single folder, and change the path accordingly. More details of usage in the file's main function.
    * plot_cmc_roc_sota.ipynb - Notebook to plot CMC and ROC curves side-by-side, based on generated `.pt` files from `cmc_eval.py` and `roc_eval.py`. Graph is generated in `graphs` directory.
    * plot_tSNE.ipynb - Notebook to plot t-SNE images based on the 10 identities of periocular-face toy examples. Example of text file (which correlates to the image paths) are in `data/visualization/tsne/img_lists`.
    * roc_eval_verification.py - Evaluates Verification Equal Error Rate (EER) and generates Receiver Operating Characteristic (ROC) curve, which are saved as `.pt` files in `data` directory. Use these `.pt` files to generate ROC curves.
- graphs: Directory where graphs are generated.
    * _CMC and ROC curve file is generated in this directory._
- logs: Directory where logs are generated.
    * _Logs will be generated in this directory. Each log folder will contain backups of training files with network files and hyperparameters used._
- models: Directory to store pre-trained models. Trained models are also generated in this directory.
    * __**INSERT PRE-TRAINED MODELS HERE. The base MobileFaceNet for fine-tuning the GC<sup>2</sup>SA-Net can be downloaded in [this link](https://www.dropbox.com/scl/fi/l3k1h3tc12vy7puargfc3/MobileFaceNet_1024.pt?rlkey=m9zock9slmaivhij6sptjyzl6&st=jy9cb6oj&dl=0).**__
    * _Trained models will also be stored in this directory._
- network: Contains loss functions and network related files.
    * gc2sa_net.py - Architecture file for GC<sup>2</sup>SA-Net.
    * load_model.py - Loads pre-trained weights based on a given model.
    * logits.py - Contains some loss functions that are used.
- __training:__ Main files for training GC<sup>2</sup>SA-Net.
    * main.py - Main file to run for training. Settings and hyperparameters are based on the files in `configs` directory.
    * train.py - Training file that is called from `main.py`. Gets batch of dataloader and contains criterion for loss back-propagation.

### Pre-requisites:
- <b>Environment: </b>Check `requirements.txt` file which was generated using `pip list --format=freeze > requirements.txt` command for the environment requirement. These files are slightly filtered manually, so there may be redundant packages.
- <b>Dataset: </b> Download dataset (training and testing) from [this link](https://www.dropbox.com/s/bfub8fmc44tvcxb/periocular_face_dataset.zip?dl=0). Password is _conditional\_biometrics_.
Ensure that datasets are located in `data` directory. Configure `datasets_config.py` file to point to this data directory by changing main path.
- <b>Pre-trained models: </b>(Optional) The pre-trained MobileFaceNet model for fine-tuning or testing can be downloaded from [this link](https://www.dropbox.com/scl/fi/l3k1h3tc12vy7puargfc3/MobileFaceNet_1024.pt?rlkey=m9zock9slmaivhij6sptjyzl6&st=jy9cb6oj&dl=0).

### Training: 
1. Change hyperparameters accordingly in `params.py` file. The set values used are the default, but it is possible to alternatively change them when running the python file.
2. Run `python training/main.py`. The training should start immediately.
3. Testing will be performed automatically after training is done, but it is possible to perform testing on an already trained model (see next section).

### Testing:
1. Based on the (pre-)trained models in the `models(/pretrained)` directory, load the correct model and the architecture (in `network` directory) using `load_model.py` file. Change the file accordingly in case of different layer names, etc.
2. Evaluation:
    * Identification / Cumulative Matching Characteristic (CMC) curve: Run `cmc_eval_identification.py`. Based on the generated `.pt` files in `data` directory, run `plot_cmc_roc_sota.ipynb` to generate CMC graph.
    * Verification / Receiver Operating Characteristic (ROC) curve: Run `roc_eval_verification.py`. Based on the generated `.pt` files in `data` directory, run `plot_cmc_roc_sota.ipynb` to generate ROC graph.
3. Visualization:
    * Gradient-weighted Class Activation Mapping (Grad-CAM): Run `grad_cam.py`, based on the selected images that are stored in a directory. The images will be generated in the `graphs` directory.
    * t-distributed stochastic neighbor embedding (t-SNE) : Run the Jupyter notebook accordingly. Based on the included text file in `data/visualization/tsne/img_lists`, 10 toy identities are selected to plot the t-SNE points, which will be generated in the `graphs` directory.

### Comparison with State-of-the-Art (SOTA) models

| Method | Intra-Modal Rank-1 IR (%) <br> (Periocular) | Intra-Modal Rank-1 EER (%) <br> (Periocular) | Intra-Modal EER (%) <br> (Periocular Gallery) | Inter-Modal EER (%) <br> (Periocular-Face) |
| --- | --- | --- | --- | --- |
| PF-GLSR [(Weights)](https://www.dropbox.com/scl/fo/gc7lnp66p706ecfr3exz2/AF6Jx_LKAeDOaKqDr2rbtMk?rlkey=skqp1kbwrd3uua1fk68qgmu01&st=dyunrk9r&dl=0) [(Paper)](https://ieeexplore.ieee.org/document/9159854) | 79.03 | 15.56 | - | - |
| CMB-Net [(Weights)](https://www.dropbox.com/scl/fo/h3grey98yeh0ir7i82lbd/AINQZy8eAEU3F4rXJm50MCE?rlkey=h0i1vv0a36uu4xsd2s41bdnaf&st=3ws0bo5q&dl=0) [(Paper)](https://ieeexplore.ieee.org/document/9956636) | 86.96 | 9.62 | 77.26 | 9.80 |
| HA-ViT [(Weights)](https://www.dropbox.com/scl/fo/crjb30rnxe95e6cdbolsk/AFT0bjj1-OzFuRTrictlAuQ?rlkey=rmpe6mriebl5l051pcfatog11&st=os5z2084&dl=0) [(Paper)](https://ieeexplore.ieee.org/document/10068230) | 77.75 | 11.39 | 64.72 | 13.14 |
| GC<sup>2</sup>SA-Net [(Weights)](https://www.dropbox.com/scl/fo/j7tfsk61jz6dch8hyl1hp/h?rlkey=b22nw4ff5kelu5ivti7ioy1mr&dl=0) [(Paper)](https://ieeexplore.ieee.org/document/10418204) | 93.63 | 6.39 | 90.77 | 6.50 |