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

# this shit don't work
for now i think i'm going to abandon this project until/unless i can think of a way to rework it that is fundamentally better from the ground up. i think most of these lessons were silly & due to my own lack of experience/understanding
- when you splice the (max_context_len, max_context_len) mask matrix down to (input_seq_len, input_seq_len) during inference you're removing access to the future for tokens that were trained to expect it, thereby making them go haywire. a potential solution might be to randomly train with the regular NTP mask half the time so that the model is encouraged to learn both NTP and future-sight representations and not know which to expect. 
- my splicing during the inference function is definitely still messed up
- my fast version of the prediction algorithm has too much distributional shift; i need to see how the full version performs
- this whole idea really only makes sense for a single layer transformer. because the transformer is multi-layer, you end up with information about the correct answer for a given token propogating through multiple layers and thus allowing the model to cheat. the other problems can be worked around, but I don't think this one can be