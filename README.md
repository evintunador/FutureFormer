# FutureFormer 
## about
*WIP - Code does not currently function and no models are trained*

This model is designed as a kind of expansion upon next-token prediction that encourages the model to think about what it wants to output further into the future of the sequence at lower & lower resolutions as the prediction gets further out, and then attend to its thoughts about the future when generating the next token. Hopefully this is kind of allowing the model to plan ahead in a way

This repo is built off of [templateGPT](https://github.com/evintunador/templateGPT) and part of a larger project of mine called [micro_model_sandbox]() that's basically a hub for all the novel model experiments I do, with the goal of facilitating easy comparison between the different models. Basically for each of those experiments I just use this very repo as a template to start editing, and then once I'm happy with the project (or if I've just abandoned it but it's moderately functional) I add it to the sandbox

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
- `old version 2024-05-24`: contains files from how the repo was before I refactored based on templateGPT and according to a new idea
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
    - `untrained_test_0.1m/`: an untrained 0.1m parameter model meant for testing purposes
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
- [ ] run through all files looking for more TODOs
- [x] build out according to the new idea. basically instead of next-token prediction, I want the model to predict the next token, then a vector that is a pooled combination of the 2nd and 3rd tokens, then a vector that's pooled from 4th through 7th, then for 8th through 15th, etc. It does this for every time period, and we let it cross-attend to its previous far-into-the-future predictions. More specifically, the final layer does NTP and cross-attends to the pooled vector of 2nd & 3rd tokens, the second to last layer predicts the pooled vec of 2&3 by cross-attending to pool of 4thru7, etc.
- [x] *figure out if i've got an information leak*
    - [x] disable kv cache for certain self-attention layers to prevent info leak
- [x] cross-attention mechanism
    - [x] combine self-attention & cross-attention into one module
- [x] pooling mechanisms (sum, max, plus, linear, norm, etc)
	- [x] make pooling choosable thru config
    - [ ] figure out how to make queries in `SelfAttentionPooling` input-dependent
    - [ ] it might be smart to find more ways to reduce parameter counts by weight tying. for example, instead of creating many separate pooling modules we can just create one pooling module capable of handling the max length and then when lesser lengths go in they just don't use the entire module. however those earlier weight matrices would now have to very different tasks, some involving compression of a few near term tokens and some of many far term tokens, which i don't think would be conducive to a good learning environment, so maybe drop the idea?
- [x] figure out BCELoss
- [x] forward.train function
- [x] implement a hyperparameter to de-prioritize fs loss
- [x] forward.inference function
- [ ] ~~fix kv caching lost vectors in later layers~~
    - [x] temporarily disable kv caching bc i'm lazy
    - [ ] eventually bring back kv caching if this model does well. honestly it's such a simple fix i'm just lazy
- [x] in model.py move pooling choice into its own function for readability
- [ ] put more stuff into loss.py? it's pretty sparse
- [ ] find a better way to assert max future sight lookahead
- [ ] train an initial attempt

### potential future TODOs
- [ ] should i set cross-attention hyperparameters different from self-attention? fewer heads?
- [ ] IF the first tests are somewhat successful
    - [ ] so the final output logit tensor shape (b,t,v) are pretty huge which is an issue in terms of ram. in [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) they overcame this issue by messing with the optimizer to do the bulk shared part of the model together but each part of the parameters specific to a given output logit tensor on its own sequentially, however they've not open-sourced their code. i've either gotta take the ram hit, lower my vocabulary size, or figure out how to mess with the optimizer like they did
    - [ ] scale up & train on a friend's gaming pc
    - [ ] write a paper?

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
- guides on how to build miniature versions of popular models from scratch, with a hand-holding walkthrough of every single tensor operation: [minGemma](https://github.com/evintunador/minGemma), [minGrok](https://github.com/evintunador/minGrok), and [minLlama3](https://github.com/evintunador/minLlama3)
- [my YouTube channel](https://www.youtube.com/@Tunadorable)
- my [other links](https://linktr.ee/tunadorable)

## potential citations
- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737): shows that predicting further into the future can help model performance. Notably, they found that it actually hurt until the model reached a critical point in size which means that i should probably not be dismayed if FutureFormer doesn't work at this absurdly small scale. also of note is how they pointed out that the final logit tensors (b,t,v) are really big so when you've got more then one suddenly you've gotta start thinking about ram. Their solution to this was to record & apply the gradient for each relevant final logit tensor & it's task-specific transformer layer sequentially while holding the parameters of the shared body of the model and then applying all of those all at once. this might make sense for me to do which would suck cuz it'd mean that i'd have to learn how to mess with optimizers which sounds like a lot of work. ugh this is annoying, tbh i'd likely just use a smaller vocab size for these initial experiments and then deal with it later. i wonder if they open-sourced their code on how they messed with their optimizer? 
- [Language Reconstruction with Brain Predictive Coding from fMRI Data](https://arxiv.org/abs/2405.11597): They cited a couple papers i need to look into on *Predictive Coding Theory* which states that the human brain predicts out into the future past the next instant moment. In previous papers they put patients into an fMRI and have them listen to a story then run those fMRI scans through a transformer encoder-decoder to see if they can decode the speech heard by the human. In this paper they expanded that idea by adding a second encoder-decoder side network which had the job of predicting multiple time periods into the future instead of regular NTP and did in fact see benefits in performance. Notably like the above paper they found the same ideal future prediction time period of $n=4$ tokens; FutureFormer isn't working in terms of tokens but still it's interesting that their results matched up so well
- Some paper I read recently (today is [[2024-06-04]]) mentioned that layers before the final were actually better at predicting further out time periods, which aligns very well with the plans here for FutureFormer. Not sure which paper though it might've been one of the above two