
# Stylish TTS (Text-To-Speech) System For Model Training
<a target="_blank" href="https://colab.research.google.com/github/Stylish-TTS/stylish-tts/blob/main/notebook/train-stylish.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
<a target="_blank" href="https://kaggle.com/kernels/welcome?src=https://github.com/Stylish-TTS/stylish-tts/blob/main/notebook/train-stylish.ipynb">
  <img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"/>
</a>

# Quick Links:
1. [What is Stylish TTS?](#1-what-is-stylish-tts)
    1. [Overview](#11-overview)
    2. [Current Status](#12-current-status)
2. [Installation](#2-installation)
    1. [Inference](#21-inference)
    2. [Training](#22-training)
3. [Training Your Model](#3-training-your-model)
    1. [Creating your config.yml file](#31-creating-your-configyml-file)
    2. [Preparing Your Dataset](#32-preparing-your-dataset)
    3. [Generating Pitch and Alignment Data](#33-generating-pitch-and-alignment-data)
    4. [Starting a Training Run](#34-starting-a-training-run)
    5. [(Optional) Loading a Checkpoint](#35-optional-loading-a-checkpoint)
    6. [Exporting to ONNX (for Deployment and Inference)](#36-exporting-to-onnx-for-deployment-and-inference)
4. [Other Forms of Model Training](#4-other-forms-of-model-training)
    1. [Training New Languages](#41-training-new-languages)
5. [Roadmap to v1.0 Release](#5-roadmap-to-v10-release)
6. [License](#6-license)
7. [Citations](#7-citations)

# 1. What is Stylish TTS?

### 1.1 Overview
- Stylish TTS is a lightweight, high-performance Text-To-Speech (TTS) system for training TTS models that are suitable for offline local use. Our focus has been to reduce training cost and difficulty while maintaining quality.
- The architecture was based on [StyleTTS 2](https://github.com/yl4579/StyleTTS2), but has now diverged substantially.
- The current goal is to allow training high quality, single-speaker text-to-speech models (rather than zero-shot voice cloning) as well as offering consistent text-to-speech results for long-form text and screen reading.

### 1.2 Current Status
- Stylish TTS is currently in Alpha/testing. It has been mostly tested with Linux and NVidia hardware, though it should be feasible to train it work using other graphics cards or even on CPU. We are currently working on adding support for Mac hardware. Stylish TTS is approaching the [v1.0 release](#5-roadmap-to-v10-release). Feel free to try it now, though remember that we still might make breaking changes before release and many scenarios are untested.


# 2. Installation

### 2.1 Inference

Complete models are converted to ONNX and so you will need a version of the onnx runtime.
TBD: More dependencies as we flesh out inference

### 2.2 Training:

If you are using a notebook-based environment (e.g., Colab or Kaggle), launch the training notebook via:

<a target="_blank" href="https://colab.research.google.com/github/Stylish-TTS/stylish-tts/blob/main/notebook/train-stylish.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
<a target="_blank" href="https://kaggle.com/kernels/welcome?src=https://github.com/Stylish-TTS/stylish-tts/blob/main/notebook/train-stylish.ipynb">
  <img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"/>
</a>

Direct notebook link: 
[train-stylish.ipynb](https://github.com/Stylish-TTS/stylish-tts/blob/main/notebook/train-stylish.ipynb)

Instructions are provided for both `uv` or `pip`. Install your preferred Python package manager and refer to the associated instructions.

<details>
	<summary>📘 <b>Installation via uv</b></summary>

```
# Create a folder for your uv project
mkdir my-training-dir
cd my-training-dir

# stylish-tts currently uses Python 3.12
uv init --python 3.12

# Clone the stylish-tts source somewhere (TODO: Fix this when we upload a package)
git clone https://github.com/Stylish-TTS/stylish-tts.git

# Install pytorch and onnx.
# Use onnxruntime-gpu if you want to do test inference with a GPU.
uv add torch torchaudio onnxruntime requests beautifulsoup4

# Auto-detect OS, Python, and PyTorch versions to fetch and install the compatible k2 wheel.
uv run stylish-tts/get_k2_whl.py > k2_whl.txt && uv add -r k2_whl.txt

# Verify that k2 was installed successfully
uv run python -c "import k2; print('k2 installed successfully')"

# Install stylish-tts as a local editable package
# Automatically rebuilds if contents change
# IMPORTANT: Don't forget the trailing slash /
uv add --editable stylish-tts/
```

</details>

<details>
	<summary>📘 <b>Installation via pip</b></summary>

```
# stylish-tts currently uses Python 3.12
# Use pyenv or equivalent to ensure the venv uses Python 3.12
# pyenv install 3.12.7 && pyenv local 3.12.7

# Ensure pip is installed via your package manager

# Create a folder for your uv project
mkdir my-training-dir
cd my-training-dir

# Set up virtual environment
python3.12 -m venv venv
# Activate virtual environment (needs to be done every time you begin a new session)
source venv/bin/activate

# Clone the stylish-tts source somewhere (TODO: Fix this when we upload a package)
git clone https://github.com/Stylish-TTS/stylish-tts.git

# Install pytorch and onnx.
# Use onnxruntime-gpu if you want to do test inference with a GPU.
pip install torch torchaudio onnxruntime requests beautifulsoup4

# Auto-detect OS, Python, and PyTorch versions to fetch and install the compatible k2 wheel.
python stylish-tts/get_k2_whl.py > k2_whl.txt && pip install -r k2_whl.txt

# Verify that k2 was installed successfully
python -c "import k2; print('k2 installed successfully')"

# Install stylish-tts as a local editable package from the stylish-tts/ directory.
# Automatically rebuilds if contents change.
# IMPORTANT: Don't forget the trailing slash /
pip install -e stylish-tts/
```

</details>

### Extras

<details>
  <summary>📘 <b>Optional Dependency: tensorboard</b></summary>

`tensorboard` is a separate application you can run which lets you see graphs of your loss functions and listen to samples produced at each validation.
- Data for tensorboard is generated in your output directories regardless of whether tensorboard is installed.
- tensorboard creates a HTTP server on localhost and provides a URL you can point your browser at to check on your training.
- You can also see training data showing up in the `train.log` file, but this tends to be less useful than the interactive graphs provided by tensorboard.

</details>

<details>
  <summary>📘 <b>PyTorch and old GPU cards</b></summary>

The latest versions of PyTorch can drop support for older GPUs. If you have an old GPU:
- Use the latest version of `torch` and `torchaudio` which supports your GPU. If your GPU is older and requires you to use an older version of `torch`, make sure to use the same version of `torchaudio` during training.
- If you are using an older version of `torch` for training, make a separate virtual environment/directory to use during model conversion and use a device of `cpu` and the latest version of torch for project conversion. This will ensure that you are converting using the most up-to-date version of Torch Dynamo.

</details>


# 3. Training Your Model

In order to train your model, you will need:
- A GPU (or a CPU and plenty of time) with PyTorch support and at least 16GB of VRAM
- Appropriate drivers and software for your GPU installed
- A [Dataset](#32-preparing-your-dataset) for your model with at least 25 hours of text/audio pairs
	- You will be training a base model from scratch. So just a few minutes of data is not enough.

## 3.1 Create your config.yml file
- You will need your own `config.yml` file (say `my_config.yml`) created from the template [here](https://github.com/Stylish-TTS/stylish-tts/blob/main/config/config.yml). You can store it anywhere, like at the root of your project.


### Training Configuration

```
	training:
	  log_interval: 1000
	  save_interval: 5000
	  val_interval: 5000
	  device: "cuda"
	  mixed_precision: "no"
	  vram_reserve: 200
	  data_workers: 32
```

- How often to log training data, how often to save a checkpoint, and how often to run a validation to see how the model is doing can all be set to your personal preference of impatience vs. overhead. `log_interval`, `save_interval`, and `val_interval` are all in steps.
- You will need to specify the `device` ("cuda", "mps", "cpu" or whatever will work with your torch installation).
- `vram_reserve` allocates an extra block of memory when probing how big batch sizes can be. It reduces the odds of having out-of-memory events.
- `data_workers` sets how many processes will be used to prepare data for training. If your GPU is under-utilized and you have spare CPU cores, you can set this higher to go faster.
  - **Note:** For Mac users, set `data_workers` to 0 for now. There is a known issue with dataloader concurrency that we are looking into.

### Training Plan Configuration

```
training_plan:
  alignment:
    epochs: 20
    probe_batch_max: 128
    lr: 1e-5
  acoustic:
    epochs: 10
    probe_batch_max: 16
    lr: 1e-4
  textual:
    epochs: 10
    probe_batch_max: 16
    lr: 3e-5
  style:
    epochs: 50
    probe_batch_max: 128
    lr: 1e-5
  duration:
    epochs: 50
    probe_batch_max: 128
    lr: 1e-4
```

The `training_plan` section provides parameters for each stage of training. With more testing we'll provide concrete guidelines for each of this.
- Generally, leave the learning rate (`lr`) alone as this has been tuned.
- If you have a small dataset, you will want to increase the number of `epochs`.
  - `acoustic` and `textual` stages are SLOW. The other stages are FAST.
  - Generally you want the same number of epochs for `acoustic` and `textual`.
  - The number above assume a 100-hour dataset. Make them proportionally larger for smaller datasets.
- If you have a GPU with a lot of VRAM, you will want to have higher `probe_batch_max`. TODO: Guidance
- If you are training using a CPU or are using an architecture with unified memory (like Mac), set `probe_max_batch` to 2 for every stage. The batch probing works by pushing until it runs out of memory. That will make you sad if this is system memory instead of VRAM memory.

### Dataset Configuration

- The `path` should be the root of your dataset, and the various other paths in this section are relative to that root. If you use the default file and directory names, your directory structure will look something like this:
  ```
  dataset:
    # All paths in this section are relative to the main path
    path: "../my_dataset/"
    train_data: "training-list.txt"
    val_data: "validation-list.txt"
    wav_path: "wav-dir"
    pitch_path: "pitch.safetensors"
    alignment_path: "alignment.safetensors"
    alignment_model_path: "alignment_model.safetensors"
  ```

- The structure of your dataset folder from this example would look like this:
  ```
  ../my_dataset/             # Root
  |
  +-> training-list.txt             # Training list file (described below)
  |
  +-> validation-list.txt           # Validation list file (described below)
  |
  +-> wav-dir                       # Folder with audio wav files, one for each segment
  |   |
  |   +-> something.wav
  |   |
  |   +-> other.wav
  |   |
  |   +-> ...
  |
  +-> pitch.safetensors             # Pre-cached segment pitches (file gets generated at `pitch_path` in `my_config.yml`. See below)
  |
  +-> alignment.safetensors         # Pre-cached alignments (file gets generated at `alignment_path` in `my_config.yml`. See below)
  |
  +-> alignment_model.safetensors   # Model for generating alignments. You will train this (gets generated at `alignment_model_path`  in `my_config.yml`. See below)

  ```

### Remaining Configuration

- The `validation` section allows you to adjust which samples are exported to tensorflow.
- The `loss_weight` section provides relative weights for different kinds of loss. These are tuned and should not be changed unless you know what you are doing.


## 3.2 Preparing Your Dataset

- Note: A sample dataset can be found at [sample_dataset/](sample_dataset/). Please note that this has been provided as a reference only, and will in no way be sufficient to train a model.

- A dataset consists of many segments. Each segment has a written text and an audio file where that text is spoken by a reader.
- Your dataset should have the following files:
  - your Training List (corresponding to train_data in `my_config.yml`)
  - your Validation List (corresponding to val_data in `my_config.yml`)
  - your Folder with audio wav files (resampled at 24 khz, mono), one for each segment (corresponding to wav_path in `my_config.yml`)

- Segment Length Distribution:
  - Segments must have 510 phonemes or less (currently not a hard limit, but recommended).
  - Audio segments must be at least 0.25 seconds long (hard limit).
  - The upper limit on audio length is determined by your VRAM and the training stage. If you have enough VRAM, you can include even longer segments, though there are diminishing returns in the usefulness of very long segments.
  - Generally speaking, you will want to have a distribution of segments between 0.25 seconds and 10 seconds long.
    - If your range doesn't cover the shortest lengths, your model will sound worse when doing short utterances of one word or a few words.
    - If your range doesn't cover longer lengths which include multiple sentences, your model will tend to skip past punctuation too quickly.

- Training List and Validation List:
  - Training and Validation lists are a series of lines in the following format: `<filename>|<phonemes>|<speaker-id>|<plaintext>`
  - The training list should consist of roughly 99% of your segments and the validation list should consist of the remainder.
  - Examples of entries in the Training and Validation lists:
      - `1.wav|ɔnðə kˈɑːntɹɛɹi|0|On the contrary`
      - `2.wav|fɚðə fˈɜːst tˈaɪm|0|For the first time`
  - The filename is the name of the file for the segment audio. It should be a .wav file (24 khz, mono) in the wav_path folder specified in your `my_config.yml`.
  - The phonemes are the IPA representation of how your segment text is pronounced. You may use `espeak-ng` (or a similar G2P system) to create phonemes corresponding to each audio file.
  - Speaker ID is an arbitrary integer which should be applied to every segment that has the same speaker. For single-speaker datasets, this will typically always be '0'.
  - The plaintext is the original text transcript of your utterance before phonemization. It does not need to be tokenized or normalized, but obviously should not include the '|' character, which is to be used as the separator.


## 3.3 Generating Pitch Data

- Stylish TTS uses a pre-cached ground truth pitch (F0) for all your segments. To generate these pitches, run:

   **uv:**
   ```
   uv run stylish-train pitch /path/to/your/config.yml --workers 16
   ```
   **pip:**
   ```
   stylish-train pitch /path/to/your/config.yml --workers 16
   ```


- The number of workers should be approximately equal to the number of cores on your machine.
- By default, Harvest, which is a CPU-based system, is used to extract pitch. If you find this to be too slow, there is also a GPU-based option available by passing `--method rmvpe` from the command line. When finished, it will write the pre-cached segment pitches at the `pitch_path` file path specified by your `my_config.yml`.


## 3.3 Generating Alignment Data

### Train Alignment Model

- Alignment data is also pre-cached. This is a multi-step process, but only needs to be done ONCE for your dataset, after which you can just use the cached results (similar to the generated pitch data).
- First, you need to train your own alignment model:

   **uv:**
   ```
   uv run stylish-train train-align /path/to/your/config.yml --out /path/to/your/output
   ```
   **pip:**
   ```
   stylish-train train-align /path/to/your/config.yml --out /path/to/your/output
   ```

<details>
<summary>📘 <b>Expectations during alignment pre-training</b></summary>

- Expectations during alignment pre-training:
  - In this stage, a special adjustment is made to the training parameters at the end of each epoch.
  - This adjustment means there will be a discontinuity in the training curve between epochs. This adjustment will eventually make the loss turn NEGATIVE. This is normal. If your training align_loss does not eventually turn negative, that is a sign that you likely need to train more.
  - At each validation step, both an un-adjusted align_loss and a confidence score are generated. 
    - align_loss should be going down.
    - Confidence score should be going up.
    - You want to pick a number of epochs so that these scores reach the knee in their curve. Do not keep training forever just because they are slowly improving. If you run into issues where things are not converging later, it is likely that you need to come back to this step and train a different amount to hit that "knee" in the loss curve.
  - During alignment pre-training, we ALSO train on the validation set. This is usually a very, very bad thing in Machine Learning (ML). But in this case, the alignment model will never be used for aligning out-of-distribution segments. Doing this gives us a more representative sample for acoustic and textual training and does not have any other effects on overall training.
</details>

- The `--out` option is where logs and checkpoints will end up.
- Once the alignment stage completes, it will provide a trained model at the file specified in your `my_config.yml`.
- This is a MODEL, not the alignments themselves.

### Use Alignment Model

- We will use this model to generate the alignments:

   **uv:**
   ```
   uv run stylish-train align /path/to/your/config.yml
   ```
   **pip:**
   ```
   stylish-train align /path/to/your/config.yml
   ```

- This generates the actual cached alignments for all the segments in both the training and validation data as configured in your config.yml. It outputs its results to the alignment file from your `my_config.yml`.


<details>
<summary>📘 <b>OPTIONAL: Culling Bad Alignments</b></summary>

- OPTIONAL: Culling Bad Alignments
  - Running `stylish-tts align` generates a "confidence value" score for every segment it processes. These scores are written to files in your dataset `path`.
  - Confidence is not a guarantee of accuracy, because the model could be confidently wrong. But it is a safe bet that the segments that it is the least confident about either:
    - Have a problem (perhaps the text doesn't match the audio) or
    - Are just a bad fit for the model's heuristics.
  - Culling the segments with the least confidence will make your model converge faster, though it also means it will use less training data overall.
  - We have found that culling the 10% with the lowest confidence scores is a good balance.
</details>

- Note: All of the commands above (for Pitch and Alignment) should only need to be done ONCE per dataset, as long as the dataset does not change. Once they are done, their results are kept in your dataset directory. Now we begin ACTUALLY training.


## 3.4 Start a Training Run

- Here is a typical command to start off a new training run:

   **uv:**
   ```
   uv run stylish-train train /path/to/your/config.yml --out /path/to/your/output
   ```
   **pip:**
   ```
   stylish-train train /path/to/your/config.yml --out /path/to/your/output
   ```

- All checkpoint, training logs, and tensorboard data are sent to the path you specify with `--out`.
- Make sure to have plenty of disk space available here as checkpoints can take a large amount of storage.

- Expectations During Training
  - It will take a LONG time to run this script. So, it is a good idea to run using `screen` or `tmux` to have a persistent shell that won't disappear if you get disconnected or close the window.
  - Training happens over the course of four stages:
    - The four main stages of training are `acoustic`, `textual`, `style`, and `duration`. 
    - When you begin training, it will start with the `acoustic` stage by default.
    - As each stage ends, the next will automatically begin.
    - You can specify a stage with the `--stage` option, which is necessary if you are resuming from a checkpoint.
  - Stages advance automatically and a checkpoint is created at the end of every stage before moving to the next. Other checkpoints will be saved and validations will be periodically run based on your `my_config.yml` settings.
  - Each stage will have its own sub-directory of `out`, and its own training log and tensorboard graphs/samples.

<details>
<summary>📘 <b>Expectations for Acoustic training</b></summary>

- Acoustic training is about training the fundamental acoustic speech prediction models which feed into the vocoder. We 'cheat' by feeding these models parameters derived directly from the audio segments. The pitch, energy, and alignments all come from our target audio. Pitch and energy are still being trained here, but they are not being used to generate predicted audio.
- The main loss figure to look at is `mel` which is a perceptual similarity of the generated audio to the ground truth. It should slowly decrease during training, but the exact point at which it converges will depend on your dataset. The other loss figures can generally be ignored and may not vary much during training.
- By the end of acoustic training, the samples should sound almost identical to ground-truth. These are probably going to be the best-sounding samples you listen to. But of course this is because it is doing the easiest version of the task.

</details>

<details>
<summary>📘 <b>Expectations for Textual training</b></summary>

- In textual training, the acoustic speech prediction is frozen while the focus of training becomes pitch and energy. An acoustic style model still 'cheats' by using audio to generate a prosodic style. This style along with the base text are what is used to calculate the pitch and energy values for each time location.
- Here, `mel`, `pitch`, and `energy` losses are all important. You should expect mel loss to always be much higher in this stage than the acoustic stage. And it will only very gradually go down. Since there are three losses here, keeping an eye on total loss is more useful. It will be a lot less stable than in acoustic, but there is still a clear trend downwards.
- As training goes on, the voice should sound less strained, less 'warbly', and more natural. Make sure you are listening for the tone of the sound and how loud it is rather than strict prosody because the samples are still using the ground truth alignment.

</details>

<details>
<summary>📘 <b>Expectations for Style training</b></summary>

- Here the only 'cheating' we do is to use the ground-truth alignment. The predicted pitch and energy are used to directly predict the audio. A textual style encoder is trained to produce the same outputs as the acoustic model from the previous stage.
- Aside from that, the training regimen should look a lot like the previous stage. `mel`, `pitch`, and `energy` should all trend downward but expect `mel` to be higher than the previous stage.

</details>

<details>
<summary>📘 <b>Expectations for Duration training</b></summary>

- The final stage of training removes our last 'cheat' and trains the duration predictor to try to replicate the prosody of the original. The other models are frozen. All samples use only values predicted from the text.
- The `duration` and `duration_ce` losses should both slowly go down. The main danger here is overfitting. So if you see validation loss stagnate or start going up you should stop training even if training loss is still going down. It is expected that one of the losses might plateau before the other.
- When you listen to samples, you will get the same version you'd expect to hear during inference. Listen to make sure the voice as a whole is not going to fast or slow or just going past punctuation without pausing. You should no longer expect it to mirror the ground truth exactly, but it should have generalized to the point where it is a plausible and expressive reading. As training proceeds, it should sound more and more like fluent prosody. If there are still pitch or energy issues like warbles or loudness or tone, then those won't be fixed in this stage and you may need to train more in Textual or Acoustic before trying Duration training.

</details>


### 3.5 (Optional) Loading a Checkpoint

<details>
<summary>📘 <b>What is a Checkpoint?</b></summary>

  What is a Checkpoint?
  - A checkpoint is a disk snapshot of training progress.
  - Our checkpoints contain:
    - Model weights for all models
    - Optimizer state
    - Training metadata (current epoch, step count, etc.)
  - You can use the checkpoint to resume in case training is interrupted.
  - After training is complete, you will use the checkpoint to generate the ONNX model for inference.
</details>

- To load a checkpoint:

   **uv:**
    ```
    uv run stylish-train train /path/to/your/config.yml --stage <stage> --out /path/to/your/output --checkpoint /path/to/your/checkpoint
    ```
   **pip:**
    ```
    stylish-train train /path/to/your/config.yml --stage <stage> --out /path/to/your/output --checkpoint /path/to/your/checkpoint
    ```

- You can load a checkpoint from any stage via the `--checkpoint` argument.
- You still need to set `--stage` appropriately to one of "alignment|acoustic|textual|duration".
  - If you set it to the same stage as the checkpoint loaded from, it will continue in that stage at the same step number and epoch.
  - If it is a different stage, it will train the entire stage.

- Please note that Stylish TTS checkpoints are NOT compatible with StyleTTS 2 checkpoints.


### 3.6 Exporting to ONNX (for deployment and inference)

- ONNX (Open Neural Network Exchange) is an open standard format for representing machine learning models.
- Only the models actually needed during inference are exported.
- They provide a self-contained standalone version of the model optimized for inference.
- This command will export two ONNX files, one for predicting duration and the other for predicting speech.

   **uv:**
   ```
   uv run stylish-train convert /path/to/your/config.yml --duration /path/to/your/duration.onnx --speech /path/to/your/speech.onnx --checkpoint /path/to/your/checkpoint
   ```
   **pip:**
   ```
   stylish-train convert /path/to/your/config.yml --duration /path/to/your/duration.onnx --speech /path/to/your/speech.onnx --checkpoint /path/to/your/checkpoint
   ```

- Using the ONNX model for Inference:

  **uv:**
  ```
  uv run stylish-tts speak /path/to/your/output/speech.onnx < /your/phonemes.txt
  ```
  **pip:**
  ```
  uv run stylish-tts speak /path/to/your/output/speech.onnx < /your/phonemes.txt
  ```

  Your file should contain phonemized text, one utterance per line. The utterances will be automatically concatenated together. Look at the `tts/cli.py` and `tts/stylish_model.py` files to see how this is implemented and you can make your own inference workflow using those as your starting point.


# 4. Other Forms of Model Training

### 4.1 Training New Languages

- Grapheme to Phoneme (G2P)
  - Grapheme-to-phoneme conversion (G2P) is the task of converting graphemes (the text we write) to phonemes (the pronunciation of that text as encoded in IPA characters).
  - Each language has its own phonetic rules, and therefore, requires a distinct G2P system. Accurate G2P is critical for the performance of text-to-speech (TTS).
  - The most effective G2P systems are typically tailored to specific languages. These can often be found in research papers focused on phonetics or TTS—try searching for terms like "[language] G2P/TTS site:arxiv.org" or "[language] G2P site:github.com". Libraries such as [misaki](https://github.com/hexgrad/misaki/) may also provide such G2P systems.
  - A commonly used multilingual G2P system is `espeak-ng`, though its accuracy can vary depending on the language. In some cases, a simple approach - using word-to-phoneme mappings from sources like Wiktionary - can be sufficient.

- Adjust model.yml

    <details>
    <summary>(Expand to read) What is model.yml used for?</summary>

    What is model.yml used for?
    - [model.yml](src/stylish_tts/train/config/model.yml) holds the hyperparameters to the model.
    - Most of the time, you will use this by default. However, in rare circumstances, it can be edited, say, if you want to experiment with different options or need to change a specific aspect of the model.
    - ---
    </details>

  - If the G2P doesn't share the same phonetic symbol set in `model.yml`, change the `symbol` section and `text_encoder.tokens`.
  - `text_encoder.tokens` should be equal to length of `symbol.pad` + `symbol.punctuation` + `symbol.letters` + `symbol.letters_ipa`
    ```
    ...
    text_encoder:
      tokens: 178 # number of phoneme tokens
      hidden_dim: 192
      filter_channels: 768
      heads: 2
      layers: 6
      kernel_size: 3
      dropout: 0.1

    ...

    symbol:
      pad: "$"
      punctuation: ";:,.!?¡¿—…\"()“” "
      letters: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
      letters_ipa: "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁᵊǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

    ```


# 5. Roadmap to v1.0 Release
Pending tasks:
- [x] Rework the CLI (Command Line Interface)
- [x] Merge disc-opt into main
- [x] Import pitch cache script and make it use a concurrent.futures worker pool
- [x] Do proper stage detection in dataloader to prevent mixups with precached alignment/pitch
- [x] Verify final model architecture
- [x] Verify ONNX conversion
- [x] Make sure it can work as a PyPi package
- [ ] Replace checkpointing with safetensors instead of accelerator checkpoint
- [ ] Remove dependency on accelerator
- [ ] Audit dependencies
- [ ] Audit and fix any remaining torch warnings
- [x] Move test_onnx to stylish-tts module and remake it into at least a barebones inferencer.
- [ ] Update this README with Sample / Demo audio clips


# 6. License
- All original code in this repository is <b>MIT-licensed.</b>
- Most code from other sources is <b>MIT-licensed.</b>
- A BSD license is included as a comment for the limited amount of code that was <b>BSD-licensed.</b>


# 7. Citations
<details>
<summary>View Citations</summary>

- The foundation of this work is StyleTTS and StyleTTS 2
  - "StyleTTS: A Style-Based Generative Model for Natural and Diverse Text-to-Speech Synthesis" by Yinghao Aaron Li, Cong Han, Nima Mesgarani [Paper](https://arxiv.org/abs/2205.15439) [Code](https://github.com/yl4579/StyleTTS)
  - "StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models" by Yinghao Aaron Li, Cong Han, Vinay S. Raghavan, Gavin Mischler, Nima Mesgarani [Paper](https://arxiv.org/abs/2306.07691) [Code](https://github.com/yl4579/StyleTTS2)

- Discriminators
  - "Improve GAN-based Neural Vocoder using Truncated Pointwise Relativistic Least Square GAN" by Yanli Li, Congyi Wang [Paper](https://dl.acm.org/doi/abs/10.1145/3573834.3574506)
  - Some code adapted from "Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis" [Paper](https://arxiv.org/pdf/2306.00814.pdf) [Code](https://github.com/gemelo-ai/vocos)
  - Discriminator Regulator: "Mind the (optimality) Gap: A Gap-Aware Learning Rate Scheduler for
  Adversarial Nets" by Hussein Hazimeh, Natalia Ponomareva [Paper](https://arxiv.org/abs/2302.00089) [Code](https://github.com/google-research/google-research/blob/master/adversarial_nets_lr_scheduler/demo.ipynb)
  - Only use MRD discriminator: "GAN Vocoder: Multi-Resolution Discriminator Is All You Need" by Jaeseong You, Dalhyun Kim, Gyuhyeon Nam, Geumbyeol Hwang, Gyeongsu Chae [Paper](https://www.isca-archive.org/interspeech_2021/you21b_interspeech.pdf)

- Text Alignment
  - "Less Peaky and More Accurate CTC Forced Alignment by Label Priors" by Ruizhe Huang, Xiaohui Zhang, Zhaoheng Ni, Li Sun, Moto Hira, Jeff Hwang, Vimal Manohar, Vineel Pratap, Matthew Wiesner, Shinji Watanabe, Daniel Povey, Sanjeev Khudanpur [Paper](https://arxiv.org/abs/2406.02560v3) [Code](https://github.com/huangruizhe/audio/tree/aligner_label_priors/examples/asr/librispeech_alignment)
  - "Evaluating Speech–Phoneme Alignment and Its Impact on Neural Text-To-Speech Synthesis" by Frank Zalkow, Prachi Govalkar, Meinard Müller, Emanuël A. P. Habets, and Christian Dittmar [Paper](https://ieeexplore.ieee.org/document/10097248) [Supplement](https://www.audiolabs-erlangen.de/resources/NLUI/2023-ICASSP-eval-alignment-tts)
  - "Phoneme-to-Audio Alignment with Recurrent Neural Networks for Speaking and Singing Voice" by Yann Teytaut, Axel Roebel [Paper](https://www.isca-archive.org/interspeech_2021/teytaut21_interspeech.html)

- Pitch Extraction
  - "Harvest: A high-performance fundamental frequency estimator from speech signals" by Masanori Morise [Paper](https://www.isca-archive.org/interspeech_2017/morise17b_interspeech.pdf)
  - "RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music" by Haojie Wei, Xueke Cao, Tangpeng Dan, Yueguo Chen [Paper](https://arxiv.org/abs/2306.15412)

- Text Encoding
  - Taken from "Matcha-TTS: A fast TTS architecture with conditional flow matching", by Shivam Mehta, Ruibo Tu, Jonas Beskow, Éva Székely, and Gustav Eje Henter [Paper](https://arxiv.org/abs/2309.03199) [Code](https://github.com/shivammehta25/Matcha-TTS)
  - Originally from "Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search" by Jaehyeon Kim, Sungwon Kim, Jungil Kong, and Sungroh Yoon [Paper](https://arxiv.org/abs/2005.11129) [Code](https://github.com/jaywalnut310/glow-tts)

- Vocoder is a hybrid model with inspiration from several sources
  <!-- - Backbone: "RingFormer: A Neural Vocoder with Ring Attention and Convolution-Augmented Transformer" by Seongho Hong, Yong-Hoon Choi [Paper](https://arxiv.org/abs/2501.01182) [Code](https://github.com/seongho608/RingFormer) -->
  <!-- - Harmonics Generation: "Neural Source-Filter Waveform Models for Statistical Parametric Speech Synthesis" by Wang, X., Takaki, S. & Yamagishi, J. [Paper](https://ieeexplore.ieee.org/document/8915761) [Code](https://github.com/nii-yamagishilab/project-CURRENNT-scripts/tree/master/waveform-modeling/project-NSF-v2-pretrained) -->
  - "APNet2: High-quality and High-efficiency Neural Vocoder with Direct Prediction of Amplitude and Phase Spectra" by Hui-Peng Du, Ye-Xin Lu, Yang Ai, Zhen-Hua Ling [Paper](https://arxiv.org/abs/2311.11545)
  - "LightVoc: An Upsampling-Free GAN Vocoder Based On Conformer And Inverse Short-time Fourier Transform" by Dinh Son Dang, Tung Lam Nguyen, Bao Thang Ta, Tien Thanh Nguyen, Thi Ngoc Anh Nguyen, Dang Linh Le, Nhat Minh Le, Van Hai Do [Paper](https://www.isca-archive.org/interspeech_2023/dang23b_interspeech.pdf)
  - For phase loss and serial AP architecture (even though we found quality is better with discriminator and also the phase loss): "Is GAN Necessary for Mel-Spectrogram-based Neural Vocoder?" by Hui-Peng Du, Yang Ai, Rui-Chen Zheng, Ye-Xin Lu, Zhen-Hua Ling [Paper](https://arxiv.org/pdf/2508.07711)
  - For anti-wrapping phase loss: "Neural Speech Phase Prediction based on Parallel Estimation Architecture and Anti-Wrapping Losses" by Yang Ai, Zhen-Hua Ling [Paper](https://arxiv.org/abs/2211.15974)
  - Attention code from Conformer implementation by Lucidrains [Code](https://github.com/lucidrains/conformer/blob/fc70d518d3770788d17a5d9799e08d23ad19c525/conformer/conformer.py#L66)

- Duration prediction
  <!-- - Ordinal regression loss: "Class Distance Weighted Cross-Entropy Loss for Ulcerative Colitis Severity Estimation" by Gorkem Polat, Ilkay Ergenc, Haluk Tarik Kani, Yesim Ozen Alahdab, Ozlen Atug, Alptekin Temizel [Paper](https://arxiv.org/abs/2202.05167)
  -->
  - "Conformal Prediction Sets for Ordinal Classification" by Prasenjit Dey, Srujana Merugu, Sivaramakrishnan (Siva) Kaveri [Paper](https://www.amazon.science/publications/conformal-prediction-sets-for-ordinal-classification)

- ONNX Compatibility
  - Kokoro [Code](https://github.com/hexgrad/kokoro)
  - Custom STFT Contributed to Kokoro by [Adrian Lyjak](https://github.com/adrianlyjak)
  - Loopless Duration Contributed to Kokoro by [Hexgrad](https://github.com/hexgrad)

</details>
