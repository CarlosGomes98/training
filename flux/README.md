# 1. Problem 

Text to Image - Flux.1-schnell.

[Torchtitan](https://github.com/pytorch/torchtitan) provides an implementation of the Flux model from [Black Forest Labs](https://bfl.ai/). The relevant files are under `torchtitan/experiments/flux`.
These files plug in to the rest of torchtitan.

```
@inproceedings{
   liang2025torchtitan,
   title={TorchTitan: One-stop PyTorch native solution for production ready {LLM} pretraining},
   author={Wanchao Liang and Tianyu Liu and Less Wright and Will Constable and Andrew Gu and Chien-Chin Huang and Iris Zhang and Wei Feng and Howard Huang and Junjie Wang and Sanket Purandare and Gokul Nadathur and Stratos Idreos},
   booktitle={The Thirteenth International Conference on Learning Representations},
   year={2025},
   url={https://openreview.net/forum?id=SFN6Wm7YBI}
}
```

# 2. Directions
### Steps to configure machine
To use this repository, please ensure your system can run docker containers and has appropriate GPU support (e.g. for CUDA GPUs, please make sure the appropriate drivers are set up)

**For all instructions that follow, make sure you are in the `flux/torchtitan` directory.**

Without docker, follow the instructions to install torchtitan and additionally install `requirements-mlperf.txt` and `torchtitan/experiments/flux/requirements.txt`.

### Container setup
To build the container:
```docker build -t <tag> -f Dockerfile .```
To run the container:
```
docker run -it --rm \
--gpus all --ulimit memlock=-1 --ulimit stack=67108864 \
--network=host --ipc=host \
-v ~/.ssh:/root/.ssh \
-v <path for dataset storage>:/dataset \
<tag> bash
```
Note: it's recommended to map your .ssh folder to inside the container, so that it's easier for the code to set up remote cluster access.

### Steps to download and verify data

#### CC12M dataset
```
docker run -it --rm \
--network=host --ipc=host \
-v ~/.ssh:/root/.ssh \
-v <path for dataset storage>:/dataset \
<tag> python -c "from datasets import load_dataset; load_dataset('pixparse/cc12m-wds', split='train', num_proc=8).save_to_disk('/dataset/cc12m')"
```

#### COCO-2014 subset
1. Run the container
2. download coco-2014 validation dataset: `DOWNLOAD_PATH=/dataset/coco2014 bash torchtitan/experiments/flux/scripts/coco-2014-validation-download.sh`
3. create the validation subset, and resize the images to 256x256: `bash torchtitan/experiments/flux/scripts/coco-2014-validation-split-resize.sh --input-images-path /dataset/coco2014/val2014 --input-coco-captions /dataset/coco2014/annotations/captions_val2014.json --output-images-path /dataset/coco2014/val2014_256x256_30k --output-tsv-file /dataset/coco2014/val2014_30k.tsv`
4. Prepare the data in hf format: `python -c "from datasets import load_dataset; load_dataset('/dataset/coco2014/val2014_256x256_30k', split='validation', num_proc=8).save_to_disk('/dataset/coco', max_shard_size='0.5GB')"`
5. (Optional) remove the unprocessed dataset to reclaim space: `rm -r /dataset/coco2014`

#### Download the autoencoder
Finally, download the autoencoder model from HuggingFace with your own access token:
```bash
python torchtitan/experiments/flux/scripts/download_autoencoder.py --repo_id black-forest-labs/FLUX.1-schnell --ae_path ae.safetensors --hf_token <your_access_token>
```

### Steps to run and time
All steps below are assumed to be run inside the container. 

The first time this is executed, checkpoints for the text encoders will automatically be downloaded from HF.
To prevent this from happening every time, we encourage users to create a directory to be used as the HF cache and mount
it to the container, as below.

```
docker run -it --rm \
--gpus all --ipc=host --ulimit memlock=-1 \
--ulimit stack=67108864 \
--network=host --ipc=host \
-v ~/.ssh:/root/.ssh \
-v <desired huggingface cache directory>:/root/.cache
-v <path to cc12m dataset>:/dataset/cc12m \
-v <path to coco dataset>:/dataset/coco
<tag> bash
```

#### Basic run
`CONFIG=torchtitan/experiments/flux/train_configs/flux_schnell_model.toml NGPU=1 bash torchtitan/experiments/flux/run_train.sh --training.dataset=cc12m_wds_disk --eval.dataset=coco`.

#### Longer run
**For longer runs, we expect a system with a slurm-based cluster.**
```source torchtitan/experiments/flux/configs/config_08x08x16.sh; export CONT=<tag>; export DATAROOT=<path_to_dataroot> sbatch -N <number of nodes> -t <time> run.sub $PARAMS```

`DATAROOT` should be set to the path where data resides. e.g. `${DATAROOT}/cc12m_disk` should point to the CC12M training dataset.

Given the substantial variability among Slurm clusters, users are encouraged to review and adapt these scripts to fit their specific cluster specifications.

In any case, the dataset and checkpoints are expected to be available to all the nodes.

# 3. Dataset/Environment
### Publication/Attribution
We use the CC12M dataset available at https://huggingface.co/datasets/pixparse/cc12m-wds

```
@inproceedings{changpinyo2021cc12m,
  title = {{Conceptual 12M}: Pushing Web-Scale Image-Text Pre-Training To Recognize Long-Tail Visual Concepts},
  author = {Changpinyo, Soravit and Sharma, Piyush and Ding, Nan and Soricut, Radu},
  booktitle = {CVPR},
  year = {2021},
}
```
We use the COCO2014 dataset for validation.

```
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={Computer vision--ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```

### Data preprocessing
For both datasets, images are resized to 256x256 using a bicubic interpolation.

The full CC12M dataset is used.
The COCO-2014-validation dataset consists of 40,504 images and 202,654 annotations. 
However, our benchmark uses only a subset of 30,000 images and annotations chosen at random with a preset seed. 


### Training data order
The data is read in the same deterministic order each time.
### Test data order
The data is randomly shuffled at each epoch. However, a random seed is fixed, making this deterministic.

# 4. Model
### Publication/Attribution
This model largely follows the Flux.1-schnell model, as implemented by torchtitan.
In turn, the model code is largely based on the model open-sourced in [huggingface](https://huggingface.co/black-forest-labs/FLUX.1-schnell) by [Black Forest Labs](https://bfl.ai/).

```
@inproceedings{esser2024scaling,
  title={Scaling rectified flow transformers for high-resolution image synthesis},
  author={Esser, Patrick and Kulal, Sumith and Blattmann, Andreas and Entezari, Rahim and M{\"u}ller, Jonas and Saini, Harry and Levi, Yam and Lorenz, Dominik and Sauer, Axel and Boesel, Frederic and others},
  booktitle={Forty-first international conference on machine learning},
  year={2024}
}
```

### List of layers 

| **Component** | **Architecture** | **Parameters** | **Technical Details** |
|---------------|------------------|----------------|----------------------|
| **Text Encoders (Frozen)** | | | |
| └ [VIT-L CLIP text encoder](https://huggingface.co/openai/clip-vit-large-patch14) | Transformer | ~123M | Max sequence length: 77 tokens |
| | | | Output dimension: 768 |
| └ [T5-XXL](https://huggingface.co/google/t5-v1_1-xxl) | Transformer | ~11B | Max sequence length: 256 tokens |
| | |  | Output dimension: 4096 |
| **Image Encoder (Frozen)** | | | |
| └ [VAE (Variational AutoEncoder)](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | CNN | ~84M | Downscaling factor: 8 (256→32) |
| | | | Channel depth: 16 |
| **Diffusion Transformer** | | | |
| └ [Flux Diffusion Transformer](https://github.com/black-forest-labs/flux/) | Multimodal Diffusion Transformer (MMDiT) | ~11.9B |
| | **Double Stream Blocks** | | **19 layers** |
| | **Single Stream Blocks** | | **38 layers** |
| | | | 24 attention heads per layer | 
| | | | Hidden dimension: 3072 |
| | | | MLP ratio: 4.0 | | Processes 64 input channels |

### Loss function
The MSE calculated over latents is used for the loss
### Optimizer
AdamW

# 5. Quality
### Quality metric
Validation loss averaged over 8 equidistant time steps, as described in [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/pdf/2403.03206)
### Quality target
TODO: tbd
### Evaluation frequency
TODO: tbd
### Evaluation thoroughness
30,000 samples (the full validation dataset)
