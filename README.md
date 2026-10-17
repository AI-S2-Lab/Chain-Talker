# Chain-Talker

## Introduction

This is an implementation of the following paper. [《Chain-Talker: Chain Understanding and Rendering for Empathetic Conversational Speech Synthesis》](https://arxiv.org/pdf/2505.12597) (Accepted by Findings of ACL 2025)

Yifan Hu, [Rui Liu \*](https://ttslr.github.io/), [Yi Ren](https://rayeren.github.io/), Xiang Yin, [Haizhou Li](https://www.colips.org/~eleliha/).

\* Corresponding author.

## 1. Environment Setup

### 1.1 Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y build-essential git wget ca-certificates bzip2 ffmpeg libsndfile1 libgomp1
```

### 1.2 Create a Conda Environment

```bash
wget -O /tmp/chain_talker_miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/chain_talker_miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
```

For installation instructions, see the [official Conda installation guide for Linux](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html).

Initialize the current Bash session, then create and activate a Python 3.10 environment:

```bash
eval "$(conda shell.bash hook)"
conda create -n chain_talker python=3.10 --override-channels -c conda-forge -y
conda activate chain_talker
python --version
python -m pip --version
```

### 1.3 Install PyTorch and Project Dependencies

Ensure that the `chain_talker` environment is active, navigate to the project root directory, and run the following commands in order:

```bash
python -m pip install --upgrade pip "setuptools<81" wheel

python -m pip install torch==2.7.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install --no-build-isolation -r requirements.txt

python -m pip check
```

Install the matching CUDA builds of PyTorch and Torchaudio first, followed by the remaining dependencies. See the [official PyTorch 2.7.1 installation instructions](https://pytorch.org/get-started/previous-versions/#v271) for the versions and CUDA wheel source.


## 2. Data Preparation (DailyTalk Example)

### 2.1 Raw Data and Prepared Data

Place the original files in [data/dailytalk_raw](data/dailytalk_raw) using the following structure:

```text
data/dailytalk_raw/
├── train/
│   ├── 2094/
│   │   ├── 0_1_d2094.wav
│   │   ├── 0_1_d2094.lab
│   │   ├── 0_1_d2094.caption.lab
│   │   ├── 1_0_d2094.wav
│   │   ├── 1_0_d2094.lab
│   │   ├── 1_0_d2094.caption.lab
│   │   └── ...
│   └── ...
├── test/
│   ├── 1140/
│   │   └── ...
│   └── ...
└── valid/
    ├── 222/
    │   └── ...
    └── ...
```

Each numbered directory corresponds to a complete dialogue. The files for each utterance use the same filename prefix:

| File | Contents |
|---|---|
| `<turn_id>_<speaker_id>_d<dialogue_id>.wav` | Original speech recording for the utterance |
| `<turn_id>_<speaker_id>_d<dialogue_id>.lab` | Transcript of the corresponding speech |
| `<turn_id>_<speaker_id>_d<dialogue_id>.caption.lab` | Empathetic caption for the utterance |

From the project root, generate the manifests:

```bash
python -m chain_talker prepare --input data/dailytalk_raw
```

This command creates only the manifest files listed below in `data/dailytalk/`. Paths in the manifests refer to the original files, for example `../dailytalk_raw/train/2094/0_1_d2094.wav`. Keep the raw and prepared directories in the same relative layout when moving the project.

Directory structure after processing:

```text
data/dailytalk/
├── utterances.jsonl
├── dialogue_splits.json
├── train.jsonl
├── valid.jsonl
├── test.jsonl
├── source_manifest.json
└── prepare_report.json
```

### 2.2 Download Pretrained Models for Feature Extraction

```bash
python tools/download_models.py
```

### 2.3 Extract Training Features

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m chain_talker features \
  --data data/dailytalk \
  --caption-device cuda \
  --onnx-device cpu \
  --workers 4 \
  --verify-cache
```

All features are saved in the same directory:

```text
data/dailytalk/
├── features/<feature_version>/<utterance_id>.npz
├── features_index.json
└── features_report.json
```

## 3. Training

### 3.1 Train EmGPT

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m chain_talker train \
  --component llm \
  --data data/dailytalk \
  --output runs/dailytalk_llm \
  --device cuda --precision bf16 \
  --optimizer adamw --trainable all \
  --batch-size 4 --accum-grad 1 --epochs 200 \
  --lr 1e-5 --new-lr 1e-4 \
  --max-tokens 4096 \
  --valid-batches 0 --save-every 1000 --seed 1986 \
  --log-dir /root/tf-logs --run-name dailytalk_llm
```

### 3.2 Train Synthesizer

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m chain_talker train \
  --component flow \
  --data data/dailytalk \
  --output runs/dailytalk_flow \
  --device cuda --precision bf16 \
  --optimizer adamw --trainable all \
  --batch-size 4 --accum-grad 1 --epochs 200 \
  --lr 1e-5 --new-lr 1e-4 \
  --valid-batches 0 --save-every 1000 --seed 1986 \
  --log-dir /root/tf-logs --run-name dailytalk_flow
```

### 3.3 Checkpoints, Resuming Training, and TensorBoard

Training results are saved in `runs/dailytalk_llm` and `runs/dailytalk_flow`, respectively:

| File | Purpose |
|---|---|
| `last.pt` | Most recently saved model checkpoint for inference or resuming training |
| `last.optimizer.pt` | Corresponding optimizer, mixed precision scaler, and random number generator states; must be retained alongside the model checkpoint to resume training |
| `run_config.json` | Configuration and data/feature versions for this training run |
| `weight_load_report.json` | Pretrained weight loading report |
| `train_metrics.jsonl` | Training metrics |
| `training_report.json` | Training progress and validation loss at the most recent checkpoint |

TensorBoard logs are written to `/root/tf-logs/llm/dailytalk_llm` and `/root/tf-logs/flow/dailytalk_flow`, respectively, and include training loss, validation loss, learning rates, gradient statistics, and GPU memory metrics. Start TensorBoard in another terminal:

```bash
tensorboard --logdir /root/tf-logs --port 6006
```

## 4. Inference

### 4.1 Export an Inference Input from the Test Set

```bash
python -m chain_talker make-job \
  --data data/dailytalk \
  --manifest data/dailytalk/test.jsonl \
  --index 0 \
  --output inference/test_job.json
```

Candidates are sorted by the duration of the target speech recording. `--index 0` selects the first candidate. Specify another valid index, or use `--target <full_utterance_id_from_utterances.jsonl>` to select a target utterance.

### 4.2 Generate a Caption and Speech

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m chain_talker infer \
  --job inference/test_job.json \
  --data data/dailytalk \
  --llm-checkpoint runs/dailytalk_llm/last.pt \
  --flow-checkpoint runs/dailytalk_flow/last.pt \
  --device cuda --precision bf16 \
  --max-caption-tokens 128 \
  --max-speech-tokens 750 --min-speech-tokens 10 \
  --seed 1986 \
  --output inference/result.wav
```

### 4.3 Output Files

After successful inference, the following five output files are saved:

| File | Contents |
|---|---|
| `inference/result.wav` | Generated speech |
| `inference/result.caption.txt` | Empathetic caption predicted by EmGPT |
| `inference/result.groundtruth.wav` | Ground truth speech for the target utterance from the dataset |
| `inference/result.groundtruth.caption.txt` | Ground truth empathetic caption for the target utterance from the dataset |
| `inference/result.json` | Input dialogue information, predictions, and ground truth target information; the predicted `caption_token` and `speech_token` arrays are also stored here |


## Citing
To cite this repository:
```bibtex
@inproceedings{hu2025chain,
  title={Chain-talker: Chain understanding and rendering for empathetic conversational speech synthesis},
  author={Hu, Yifan and Liu, Rui and Ren, Yi and Yin, Xiang and Li, Haizhou},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  pages={1988--2003},
  year={2025}
}

```

## Author

E-mail：hyfwalker@163.com
