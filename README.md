<h1 align="center">
    Gated Convolutional Channel-wise Self-Attention Network (GC<sup>2</sup>SA-Net)
</h1>
<h2 align="center">
    Self-Attentive Contrastive Learning for Conditioned Periocular and Face Biometrics
</h2>
<h3 align="center">
    Published in Transaction of Information Forensics and Security (DOI: 10.1109/TIFS.2024.3361216) </br>
    <a href="https://ieeexplore.ieee.org/document/10418204"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a>
</h3>
<br/>

![Network Architecture](GC2SA_Net_Architecture.jpg?raw=true "GC2SA_Net")
<br/></br>

## Pre-requisites:
- <b>Environment: </b>Check `requirements.txt` file which was generated using `pip list --format=freeze > requirements.txt` command for the environment requirement. These files are slightly filtered manually, so there may be redundant packages.
- <b>Dataset: </b> Download dataset (training and testing) from [this link](https://www.dropbox.com/s/bfub8fmc44tvcxb/periocular_face_dataset.zip?dl=0). Password is _conditional\_biometrics_.
Ensure that datasets are located in `data` directory. Configure `datasets_config.py` file to point to this data directory by changing main path.
- <b>Pre-trained models: </b>(Optional) The pre-trained MobileFaceNet model for fine-tuning or testing can be downloaded from [this link](https://www.dropbox.com/scl/fi/l3k1h3tc12vy7puargfc3/MobileFaceNet_1024.pt?rlkey=m9zock9slmaivhij6sptjyzl6&st=jy9cb6oj&dl=0).

## Training: 
1. Change hyperparameters accordingly in `params.py` file. The set values used are the default, but it is possible to alternatively change them when running the python file.
2. Run `python training/main.py`. The training should start immediately.
3. Testing will be performed automatically after training is done, but it is possible to perform testing on an already trained model (see next section).

## Testing:
1. Based on the (pre-)trained models in the `models(/pretrained)` directory, load the correct model and the architecture (in `network` directory) using `load_model.py` file. Change the file accordingly in case of different layer names, etc.
2. Evaluation:
    * Identification / Cumulative Matching Characteristic (CMC) curve: Run `cmc_eval_identification.py`. Based on the generated <code>.pt</code> files in `data` directory, run `plot_cmc_roc_sota.ipynb` to generate CMC graph.
    * Verification / Receiver Operating Characteristic (ROC) curve: Run `roc_eval_verification.py`. Based on the generated <code>.pt</code> files in `data` directory, run `plot_cmc_roc_sota.ipynb` to generate ROC graph.
3. Visualization:
    * Gradient-weighted Class Activation Mapping (Grad-CAM): Run `grad_cam.py`, based on the selected images that are stored in a directory. The images will be generated in the `graphs` directory.
    * t-distributed stochastic neighbor embedding (t-SNE) : Run the Jupyter notebook accordingly. Based on the included text file in `data/visualization/tsne/img_lists`, 10 toy identities are selected to plot the t-SNE points, which will be generated in the `graphs` directory.

## Comparison with State-of-the-Art (SOTA) models

| Method | Intra-Modal Rank-1 IR (%) <br> (Periocular) | Intra-Modal Rank-1 EER (%) <br> (Periocular) | Intra-Modal IR (%) <br> (Periocular Gallery) | Inter-Modal EER (%) <br> (Periocular-Face) |
| --- | --- | --- | --- | --- |
| PF-GLSR <a href="https://ieeexplore.ieee.org/document/9159854"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a> <br> <a href="https://www.dropbox.com/scl/fo/gc7lnp66p706ecfr3exz2/AF6Jx_LKAeDOaKqDr2rbtMk?rlkey=skqp1kbwrd3uua1fk68qgmu01&st=dyunrk9r&dl=0"> <img src="https://img.shields.io/badge/pre--trained%20weights-8A2BE2" alt="Pre-trained Weights"> </a> | 79.03 | 15.56 | - | - |
| <a href="https://github.com/tiongsikng/cb_net" target="_blank" rel="noopener noreferrer"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/github.svg" width="20" height="20">CMB-Net</a> <a href="https://ieeexplore.ieee.org/document/9956636"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a> <br> <a href="https://www.dropbox.com/scl/fo/h3grey98yeh0ir7i82lbd/AINQZy8eAEU3F4rXJm50MCE?rlkey=h0i1vv0a36uu4xsd2s41bdnaf&st=3ws0bo5q&dl=0"> <img src="https://img.shields.io/badge/pre--trained%20weights-8A2BE2" alt="Pre-trained Weights"> </a> | 86.96 | 9.62 | 77.26 | 9.80 |
| <a href="https://github.com/MIS-DevWorks/HA-ViT" target="_blank" rel="noopener noreferrer"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/github.svg" width="20" height="20">HA-ViT</a> <a href="https://ieeexplore.ieee.org/document/10068230"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a> <br> <a href="https://www.dropbox.com/scl/fo/crjb30rnxe95e6cdbolsk/AFT0bjj1-OzFuRTrictlAuQ?rlkey=rmpe6mriebl5l051pcfatog11&st=os5z2084&dl=0"> <img src="https://img.shields.io/badge/pre--trained%20weights-8A2BE2" alt="Pre-trained Weights"> </a> | 77.75 | 11.39 | 64.72 | 13.14 |
| <a href="https://github.com/tiongsikng/gc2sa_net" target="_blank" rel="noopener noreferrer"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/github.svg" width="20" height="20">GC<sup>2</sup>SA-Net</a> <a href="https://ieeexplore.ieee.org/document/10418204"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a> <br> <a href="https://www.dropbox.com/scl/fo/z0sxpfbzmgp76erlcjxij/AIthSVT0Ju6VNeZupjtju1Y?rlkey=k8ivz5l1gv464e4dbxvfu40gc&e=1&st=0yt7hmr1&dl=0"> <img src="https://img.shields.io/badge/pre--trained%20weights-8A2BE2" alt="Pre-trained Weights"> </a> | 93.63 | 6.39 | 90.77 | 6.50 |

### The project directory is as follows:
<pre>
├── configs: Dataset path configuration file and hyperparameters.
│   ├── datasets_config.py - Directory path for dataset files. Change 'main' in 'main_path' dictionary to point to dataset, e.g., <code>/home/gc2sa_net/data</code> (without slash).
│   └── params.py - Adjust hyperparameters and arguments in this file for training. 
├── data: Dataloader functions and preprocessing.
│   ├── <i><b>[INSERT DATASET HERE.]</i></b>
│   ├── <i>The <code>.pt</code> files to plot the CMC and ROC graphs will be generated in this directory.</i>
│   └── data_loader.py - Generate training and testing PyTorch dataloader. Adjust the augmentations etc. in this file. Batch size of data is also determined here, based on the values set in <code>params.py</code>.
├── eval: Evaluation metrics (identification and verification). Also contains CMC and ROC evaluations.
│   ├── cmc_eval_identification.py - Evaluates Rank-1 Identification Rate (IR) and generates Cumulative Matching Characteristic (CMC) curve, which are saved as <code>.pt</code> files in <code>data</code> directory. Use these <code>.pt</code> files to generate CMC curves.
│   ├── grad_cam.py - Plot GradCAM images. For usage, store all images in a single folder, and change the path accordingly. More details of usage in the file's main function.
│   ├── plot_cmc_roc_sota.ipynb - Notebook to plot CMC and ROC curves side-by-side, based on generated <code>.pt</code> files from <code>cmc_eval.py</code> and <code>roc_eval.py</code>. Graph is generated in <code>graphs</code> directory.
│   ├── plot_tSNE.ipynb - Notebook to plot t-SNE images based on the 10 identities of periocular-face toy examples. Example of text file (which correlates to the image paths) are in <code>data/visualization/tsne/img_lists</code>.
│   └── roc_eval_verification.py - Evaluates Verification Equal Error Rate (EER) and generates Receiver Operating Characteristic (ROC) curve, which are saved as <code>.pt</code> files in <code>data</code> directory. Use these <code>.pt</code> files to generate ROC curves.
├── graphs: Directory where graphs and visualization evaluations are generated.
│   └── <i>CMC and ROC curve file is generated in this directory. Some evaluation images are also generated in this directory.</i>
├── logs: Stores logs based on 'Method' and 'Remarks' based on config files, with time.
│   └── <i>Logs will be generated in this directory. Each log folder will contain backups of training files with network files and hyperparameters used.</i>
├── models: Directory to store pretrained models, and also where models are generated.
│   ├── <i><b>[INSERT PRE-TRAINED MODELS HERE.]</i></b>
│   ├── <i><b>The base MobileFaceNet for fine-tuning the GC<sup>2</sup>SA-Net can be downloaded in <a href="https://www.dropbox.com/scl/fi/l3k1h3tc12vy7puargfc3/MobileFaceNet_1024.pt?rlkey=m9zock9slmaivhij6sptjyzl6&st=jy9cb6oj&dl=0">this link</a>.</i></b>
│   └── <i>Trained models will be generated in this directory.</i>
├── network: Contains loss functions, and network related files.
│   ├── gc2sa_net.py - Architecture file for GC<sup>2</sup>SA-Net.
│   ├── load_model.py - Loads pre-trained weights based on a given model.
│   └── logits.py - Contains some loss functions that are used.
└── <i>training:</i> Main files for training.
    ├── main.py - Main file to run for training. Settings and hyperparameters are based on the files in <code>configs</code> directory.
    └── train.py - Training file that is called from <code>main.py</code>. Gets batch of dataloader and contains criterion for loss back-propagation.
</pre>

#### Citation for this work:
```
@ARTICLE{gc2sa_net,
  author={Ng, Tiong-Sik and Chai, Jacky Chen Long and Low, Cheng-Yaw and Beng Jin Teoh, Andrew},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Self-Attentive Contrastive Learning for Conditioned Periocular and Face Biometrics}, 
  year={2024},
  volume={19},
  number={},
  pages={3251-3264},
  keywords={Face recognition;Faces;Biometrics (access control);Feature extraction;Biological system modeling;Self-supervised learning;Correlation;Biometrics;face;periocular;channel-wise self-attention;modality alignment loss;intra-modal matching;inter-modal matching},
  doi={10.1109/TIFS.2024.3361216}
}
```