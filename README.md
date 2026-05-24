# TRACER

This repository contains the anonymized implementation for table semantic annotation experiments on GitTables and SemTab. The code is organized by dataset and task: GitTables scripts are used for column type annotation, while SemTab scripts are used for column pair annotation.

## Directory structure

```text
TRACER/
├── GitTables/
│   ├── data/
│   │   └── precompute_entropy.py
│   ├── eval/
│   │   └── evaluate.py
│   ├── train/
│   │   └── train.py
│   └── utils.py
├── SemTab/
│   ├── data/
│   │   ├── build_data.py
│   │   └── split_data.py
│   ├── eval/
│   │   └── evaluate.py
│   ├── train/
│   │   └── train.py
│   ├── data_config_template.json
│   └── mapping.json
├── README.md
└── requirements.txt
```

## Environment

Install the required packages with:

```bash
pip install -r requirements.txt
```

The main dependencies include PyTorch, Transformers, Sentence-Transformers, pandas, NumPy, scikit-learn, and tqdm. A CUDA-enabled GPU is recommended for entropy precomputation and model training.

## Datasets

The datasets are not included in this repository. Please download them from the official sources and place them outside the code directory.

| Dataset | Usage | Source |
|---|---|---|
| GitTables | Column type annotation | [GitTables](https://gittables.github.io/) |
| SemTab 2019 | Column pair annotation | [Zenodo record 3518539](https://zenodo.org/records/3518539) |
| SemTab 2020 | Column pair annotation | [Zenodo record 4282879](https://zenodo.org/records/4282879) |

All paths in the commands below use `path/to/...` placeholders. Replace them with your local dataset, model, and output paths.

## GitTables experiments

### 1. Precompute entropy values

`GitTables/data/precompute_entropy.py` computes entropy values for input table cells and saves a pickle file containing a dictionary of the form:

```python
{text: entropy_value}
```

Example command:

```bash
python GitTables/data/precompute_entropy.py \
  --input_path path/to/gittables/input_data \
  --label_file path/to/gittables/label_frequencies.json \
  --model_path path/to/sentence_transformer \
  --output_file path/to/entropy_lookup.pkl
```

The input path can be either a file or a directory containing table files. Supported file formats depend on the preprocessing script and may include JSON, JSONL, CSV, or TSV files.

### 2. Train the GitTables reader

The training script expects a preprocessed pickle file with the following keys:

```text
train
dev
label_encoder
```

The `train` and `dev` entries should be pandas DataFrames containing at least:

```text
table_id, data, label
```

Example command:

```bash
python GitTables/train/train.py \
  --data_file path/to/gittables_dataset.pkl \
  --output_dir path/to/checkpoints \
  --model_name path/to/pretrained_model \
  --lr 1e-5 \
  --batch_size 64 \
  --epochs 20 \
  --num_workers 0
```

The best checkpoint will be saved under the specified output directory.

### 3. Evaluate the GitTables reader

The evaluation script expects the same pickle format, with an additional `test` split:

```text
test
label_encoder
```

The `test` DataFrame should contain at least:

```text
table_id, data, label
```

Example command:

```bash
python GitTables/eval/evaluate.py \
  --data_pkl path/to/gittables_dataset.pkl \
  --reader_model_name path/to/pretrained_model \
  --reader_checkpoint path/to/checkpoints/best_model.pt \
  --batch_size 128 \
  --num_workers 0
```

## SemTab experiments

The SemTab pipeline is used for column pair annotation. It converts subject-object column pairs into serialized row-pair text and trains a relation classifier.

### 1. Prepare the data configuration

Copy the template file and edit the paths:

```bash
cp SemTab/data_config_template.json path/to/data_config.json
```

The configuration file should contain entries like:

```json
[
  {
    "name": "2019_R1",
    "gt_path": "path/to/SemTab2019/GT/CPA/CPA_Round1_gt.csv",
    "table_dir": "path/to/SemTab2019/Round1/tables",
    "need_mapping": false
  },
  {
    "name": "2020_R1",
    "gt_path": "path/to/SemTab2020/GT/CPA/CPA_Round1_gt.csv",
    "table_dir": "path/to/SemTab2020/Round1/tables",
    "need_mapping": true
  }
]
```

For SemTab 2020, relation labels may need to be mapped into the target label space. The mapping file is expected at:

```text
SemTab/mapping.json
```

You can also provide another mapping path through the command line.

### 2. Build row-pair CPA data

`SemTab/data/build_data.py` supports entropy precomputation and dataset construction. The row selection strategy uses subject-object columns for entropy aggregation, and each retained row is serialized with the fixed format:

```text
[ROW] subject: ... ; object: ...
```

Example command:

```bash
python SemTab/data/build_data.py \
  --step all \
  --data_config path/to/data_config.json \
  --mapping_file SemTab/mapping.json \
  --output_dir path/to/semtab_output \
  --entropy_file path/to/entropy.pkl \
  --sentence_model_name path/to/sentence_transformer \
  --k 15 \
  --entropy_ratio 0.7 \
  --head_rows 2 \
  --num_folds 5 \
  --num_workers 16
```

The output directory will contain fold-level files:

```text
path/to/semtab_output/
├── fold_0/
│   ├── train.pkl
│   ├── val.pkl
│   ├── test.pkl
│   └── labels.json
├── fold_1/
│   └── ...
└── ...
```

You can also run entropy precomputation and data construction separately:

```bash
python SemTab/data/build_data.py \
  --step entropy \
  --data_config path/to/data_config.json \
  --mapping_file SemTab/mapping.json \
  --entropy_file path/to/entropy.pkl \
  --sentence_model_name path/to/sentence_transformer
```

```bash
python SemTab/data/build_data.py \
  --step build \
  --data_config path/to/data_config.json \
  --mapping_file SemTab/mapping.json \
  --output_dir path/to/semtab_output \
  --entropy_file path/to/entropy.pkl
```

### 3. Optional split-only preprocessing

`SemTab/data/split_data.py` can be used when you only need to merge and split the SemTab CPA metadata without row-pair serialization.

Example command:

```bash
python SemTab/data/split_data.py \
  --data_config path/to/data_config.json \
  --mapping_file SemTab/mapping.json \
  --output_dir path/to/split_output \
  --num_folds 5
```

### 4. Train the SemTab CPA reader

Example command:

```bash
python SemTab/train/train.py \
  --data_dir path/to/semtab_output/fold_0 \
  --output_dir path/to/checkpoints \
  --model_name path/to/pretrained_model \
  --batch_size 64 \
  --epochs 30 \
  --lr 3e-5 \
  --num_workers 0
```

The training script uses standard cross-entropy loss and saves:

```text
best_model.pt
best_checkpoint.pt
history.json
training_summary.json
labels.json
```

### 5. Evaluate the SemTab CPA reader

Example command:

```bash
python SemTab/eval/evaluate.py \
  --data_dir path/to/semtab_output/fold_0 \
  --checkpoint_path path/to/checkpoints/best_model.pt \
  --model_name path/to/pretrained_model \
  --split test \
  --batch_size 128 \
  --num_workers 0
```

The script reports accuracy, micro-F1, macro-F1, and weighted-F1.

## Notes on anonymous release

This repository only contains code and configuration templates. It does not include:

```text
raw datasets
processed pickle files
pretrained models
sentence-transformer checkpoints
training checkpoints
prediction outputs
logs
```

Before pushing the repository, check that no local paths, user names, checkpoints, or dataset files are included.

Recommended `.gitignore` entries:

```gitignore
__pycache__/
*.pyc
*.pkl
*.pt
*.pth
*.bin
*.safetensors
*.log
*.csv
*.tsv
*.jsonl

checkpoints/
outputs/
logs/
data/
dataset/
datasets/
models/
pretrained_models/
wandb/
runs/

.DS_Store
.vscode/
.idea/
```

Do not ignore `SemTab/data_config_template.json` or `SemTab/mapping.json` if they are intended to be included in the repository.

## Citation

If you use GitTables or SemTab, please cite the corresponding dataset papers or official dataset records according to their instructions.
