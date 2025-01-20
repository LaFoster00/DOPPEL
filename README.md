# DOPPEL
D.O.P.P.E.L. - Detection of People through Evaluated Likelihood

## Requirements
### DOPPEL
* Linux is supported.
* 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of memory.
* 64-bit Python 3.9 (Should be part of the conda environment) and tensorflow 2.16.1. See https://www.tensorflow.org/install for Tensorflow install instructions.
* Python libraries: see [generator_cuda.yml](./environment-cuda.yml) for exact library dependencies.  You can use the following commands with Miniconda3 or Anaconda to create and activate your DOPPEL Python environment:
  - `conda env create -f generator_cuda.yml`
  - `conda activate DOPPEL`
* Download the VGG-Face2 dataset from [Academic Torrents](https://academictorrents.com/details/535113b8395832f09121bc53ac85d7bc8ef6fa5b)
  and copy it into ```./data``` (create the directory if needed)
  * Executing the training or evaluation scripts will automatically extract the dataset.

## Training
Make sure you activated the 'DOPPEL' conda environment

To train the model, execute the following command:
``` bash
python model.py
```
```
Options:                                                                                                                                                                                           
  --epochs INTEGER                Number of epochs. (50)
  --batch_size INTEGER            Batch size. (16)
  --image_dim <INTEGER INTEGER>...
                                  Image dimensions. (224, 224)
  --learning_rate FLOAT           Learning rate. (0.0001)
  --limit_images INTEGER          Limit image comparisons per person. (15, -1 for all)
  --num_train_classes INTEGER     Number of training classes (Persons). (-1 for all)
  --num_test_classes INTEGER      Number of test classes (Persons). (-1 for all)
  --data_dir TEXT                 Directory containing the dataset. (data/VGG-Face2/data)
  --help                          Show this message and exit.
```

## Inference
Make sure you activated the 'DOPPEL' conda environment

To evaluate pictures of persons and plot the result, execute the following command:
``` bash
python show_samples.py
```
```
Options:
  --data_dir PATH                 Directory containing the dataset. (data/VGG-Face2/data)
  --image_size <INTEGER INTEGER>...
                                  Image dimensions. (224, 224)
  --num_pairs INTEGER             Number of pairs to generate and plot. (4)
  --model_path PATH               Path to the saved models. (saved_models) e.g. saved_models/DOPPEL_Contrastive_Embedding_20250119_211235.keras
  --help                          Show this message and exit.

```