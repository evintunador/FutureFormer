# FutureFormer

This is a slightly biologically inspired architecture & inference algorithm. The idea is that in the brain, your perception is not only composed from raw input; rather, your brain's own predictions of the future shape what you perceive. Here with FutureFormer, the model is trained with an attention mask that allows certain tokens in the sequence to see some pre-set number of tokens into the future. Then during inference, we first compose a sequence using regular next-token prediction, and then refine earlier tokens in the sequence based on later tokens. Here is a regular next-token prediction attention mask for a sequence length of 5:
```
[[1,0,0,0,0],
[1,1,0,0,0],
[1,1,1,0,0],
[1,1,1,1,0],
[1,1,1,1,1]]
```
and here is our future-token prediction attention mask on a sequence length of 5 that can see one token in the future past the one it's supposed to predict
```
[[1,0,1,0,0],
[1,1,0,1,0],
[1,1,1,0,1],
[1,1,1,1,0],
[1,1,1,1,1]]
```
and here is our future-token prediction attention mask on a sequence length of 5 that can see two tokens into the future past the one it's supposed to predict
```
[[1,0,1,1,0],
[1,1,0,1,1],
[1,1,1,0,1],
[1,1,1,1,0],
[1,1,1,1,1]]
```

File guide:
- `FutureFormer.ipynb` contains all of the interesting code. basically ignore everything else
- `input.txt` is just TinyShakespeare
- `tokenizers/tokenizer.model` was pre-constructed for use here. Its construction can be found [here](https://github.com/evintunador/base_model.git)
- the files in `models/` are all trained with a batch size of 32 over 5,000 iterations and use the parameters listed in their filenames.