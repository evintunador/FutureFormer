# FutureFormer 
## about
*WIP - Code does not currently function and no models are trained*

This model is designed as a kind of expansion upon next-token prediction that encourages the model to think about what it wants to output further into the future of the sequence at lower & lower resolutions as the prediction gets further out, and then attend to its thoughts about the future when generating the next token. Hopefully this is kind of allowing the model to plan ahead in a way

This repo is built off of [templateGPT](https://github.com/evintunador/templateGPT) and part of a larger project of mine called [micro_model_sandbox]() that's basically a hub for all the novel model experiments I do, with the goal of facilitating easy comparison between the different models. Basically for each of those experiments I just use this very repo as a template to start editing, and then once I'm happy with the project (or if I've just abandoned it but it's moderately functional) I add it to the sandbox.

## getting started
1. clone the repository
2. `cd` to the folder
3. setup a virtual environment unless you're an agent of chaos
4. `pip install -r requirements.txt`
5. edit values in `config.py` to suit your liking. This might involve a lot of trial and error if you don't know what you're doing, either due to errors from incompatible parameter configurations or from going over your available vram amount. Checkout the config files for each already trained model to get an idea of what reasonable values look like
6. Hop into `train.py` and run every cell before the final one. There's a cell where if you set the `if` statement to `True` then you'll be able to visualize what the learning rate is going to look like over the course of training (which you determined over in `config.py`)
7. If you like the look of the model you trained, run that final cell to save it. I recommend going into `trained/` and changing the folder name if you didn't already do so when messing with the config since the default is just going to be the date & time that its training begun, which is ugly boring and confusing
8. If you ever want to just test out a model you've already made then hop on over into `inference.ipynb` and run all the cells.
9. If you've trained multiple models, you can compare them in `model_comparison.ipynb` as long as you remember to use the third cell to specify which models you want to compare. It'll look at loss curves over the course of training and teacher-forcing topk accuracy rate
10. This step could really go anywhere, but if you're trying to learn how transformers work then along with reading the code in `modules/` you can use `test_modules.ipynb` to visualize how the tensor shapes change.

## file structure
- `old version 2024-05-24`: contains files from how the repo was before I refactors based on templateGPT and according to a new idea
- `modules/`: where all of the code for the actual model goes
    - `layer.py`: defines each residual connection layer of our GPT
    - `logging.py`: defines the `LoggingModule` class, a wrapper that you should use instead of pytorch's `nn.module` in order to facilitate easy demonstration of how tensor shapes change throughout a given module
    - `mlp.py`: a two-layer multi-layer perceptron with an optional gate and either [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html), [GeLU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html), or [SiLU](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html) nonlinearities, all configurable in `config.py`. Adding more nonlinearities is also absurdly easy
    - `model.py`: the primary class for our GPT
    - `mqa.py`: [multi-query attention](https://arxiv.org/abs/1911.02150) with pre-computed [rotary positional encodings](https://arxiv.org/abs/2104.09864)
    - `norm.py`: a norm module with an optional affine layer that allows you to switch between [RMSNorm](https://arxiv.org/abs/1910.07467), [LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) and [CosineNorm](https://arxiv.org/pdf/1702.05870) easily using a setting over in `config.py`. Adding different normalization methods is also absurdly easy
- `tokenizers/`: a folder where you store your tokenizers
    - `bpe_v1/`: a [byte-pair encoding](https://huggingface.co/learn/nlp-course/chapter6/5) tokenizer except I use the characters that show up in TinyStories instead of bytes. If you want to train a model that's comparable with those in templateGPT use this, but if you're training a new model and don't care about comparing it against the pre-existing ones then I recommend using `bpe_v2/`
        - `build.ipynb`: the notebook where i built my bpe tokenizers. My pairing rules could certainly be improved upon
        - `tokenizer.py`: an overly-simplistic and annoyingly inefficient tokenizer with bos & eos tokens, post-sequence padding, and a `display` function to help you visualize how a given string is broken down
        - `models/`
            - `{95, 128, 256, 512, 1024, 2048, 4096, 8192}.model`: different tokenizer sizes, each a subset of the next. the 95 one is character-wise tokenization
    - `bpe_v2`: a slightly updated version of the one above that uses GPT4's regex instead of my own shitty from-scratch rules. Not currently implemented in any models
        - `...`
- `trained/`
    - `FutureFormer_?m_?/`: none yet currently trained, but this folder will get created eventually
        - `model_config.json`: hyperparameters of the model
        - `model.pth`: weights of the model
        - `train_config.json`: hyperparameters of the training loop used
        - `log_data.csv`: a record of loss and a couple other key metrics over the course of training
- `inference.ipynb`: open this notebook if you just want to see the output of one of the models
- `model_comparison.ipynb`: open this notebook to compare different models against each other. includes loss curve plots and topk teacher-forcing accuracy rate
- `testing_modules.ipynb`: creates easy printouts that allow you to follow the progression of tensor shapes for demonstration & debugging purposes of all the modules in `model.py`. If you're building new modules for a novel architecture idea you have then this notebook will be of extreme value to you in debugging & visualization
- `train.ipynb`: open this notebook to train a new model
- `config.py`: all of the editable model and training settings
- `inference.py`: functions for performing inference used in multiple `.ipynb` files
- `model_comparison.py`: functions for comparing models used in `model_comparison.ipynb`
- `requirements.txt` - I should probably change this to only include the packages that are actually necessary and not be so strict on versions. The command I used to get this list is `pip freeze | grep -v " @ file://" > requirements.txt` and then I deleted the version numbers, lmk if you know of a better method
- `tools.py`: A variety of functions & classes that don't fit elsewhere and/or are used by more than one of the jupyter notebooks. I should prolly find a better way to organize these
- `train.py`: functions for training a model, used in `train.ipynb`

## definite eventual TODOs
- [ ] build out according to the new idea. basically instead of next-token prediction, I want the model to predict the next token, then a vector that is a pooled combination of the 2nd and 3rd tokens, then a vector that's pooled from 4th through 7th, then for 8th through 15th, etc. It does this for every time period, and we let it cross-attend to its previous far-into-the-future predictions. Not sure if this can be doen with a simple linear or MLP at the end of the final residual state, or if it's gonna have to be off-set by some number of layers or something. 
    - [ ] multi-scale pooling mechanism
    - [ ] multi-scale output layer
    - [ ] multi-scale cosine similarity loss function
    - [x] cross-attention mechanism
    - [ ] edit `generate()` function to pass through future predictions
    - [ ] create alternative options for 
        - [x] pooling mechanisms (sum, max, plus, linear, norm, etc)
        - [ ] output layer (simple linear, MLP, or full transformer layer)
        - [ ] add MSE & MAE to loss function
- [ ] train model(s)

### potential future TODOs
- [ ] can I get one single module that does both self and cross attention? I think i saw that in a diffusion paper, not sure if it makes sense here
- [ ] write a paper IF it works?

## how to contribute
Other than the above TODO lists, appreciated contributions include:
- bug fixes
- adding more detailed comment explanations of what the code is doing
- general readability edits
- efficiency edits
- editing the code in `modules/` to take better advantage of the `LoggingModule`. This means splitting up each class into more and tinier functions
- training more models (especially if they're bigger than what's already here!)

Because I'm not super knowledgeable on how collaborating on git projects works and I tend to edit directly on the main branch, please reach out and communicate with me about any edits you plan to make so that I can avoid editing the same files. [Click here to join my discord server](https://discord.gg/hTYQyDPpr9)

## check me out
- guides on how to build miniature versions of popular models from scratch, with a hand-holding walkthrough of every single tensor operation: [minGemma](https://github.com/evintunador/minGemma), [minGrok](https://github.com/evintunador/minGrok), and [minLlama3](https://github.com/evintunador/minLlama3). Future versions of these guides will use this Repo as a template
- [my YouTube channel](https://www.youtube.com/@Tunadorable)
- my [other links](https://linktr.ee/tunadorable)
