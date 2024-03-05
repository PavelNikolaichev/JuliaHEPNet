# Evaluation excersise for JuliaHEP
That's my simple solution for GSoC eval excersise for JuliaHEP.\
I'm using primitive NN architecture, but since that's eval excersise and requirements said so, I don't think it's a good thing to overcomplicate my solution anyway.\
Actually I'm sure that even NN will be too complex already, boosting trees should perform better without NN-s disadvantages straight out of the box

# Table of Contents
1. [Introduction](#Introduction)
2. [Installation and Usage](#Installation)
3. [Other ideas and models](#Other-ideas-and-models)

# Introduction

# Installation
Just run ```activate .``` from julia pkg manager to set up local envirnoment.\
After that you can run `train.jl` and `predict.jl` with arguments that you want for corresponding operations:\
For `train.ji`:
```bash
julia --project src/predict.jl --data_dir dataset/dataset.csv --verbose --epochs 10 --gpu --output_model_path <output_model_name.jld2>
```
For `predict.ji`:
```bash
julia --project src/predict.jl --data_dir dataset/dataset.csv --gpu --checkpoint_path model.jld2
```

# Other ideas and models
Usually data from CERN is divided by runs, that means that we can have a lots of information that is actually more or less dependent on each other(for example, if we know that only 3 different particles are in each run, we can classify which particles have interacted in each registered event during each run, since for each run we can assume that each particle can interact only one time or so.).\
I think this data might actually be the same, so that's why it might be a better tactic to use something like RNN for each run rather than trying to fit lots of points that might vary from different runs. But that's only assumption, since the data doesn't have information about runs.\
Also it's worth checking out boosting trees for this classification, since they are more stable and easier to optimize rather then setting up NN's architecture and hyperparameters. I haven't tried them because I am kinda lazy and this is only test task.