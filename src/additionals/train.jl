using Flux
using Flux.Data: DataLoader
using Flux: train!
using IterTools: ncycle 

"""
    train_model(model, train_data, epochs, loss, optimizer; verbose=false, batchsize=100)

Trains the model with the given data and optimizer. Note that DataLoader is created inside the function.

# Arguments
- `model`: the model to be trained
- `train_data`: training data, should be a tuple of (X, y)
- `epochs`: the number of epochs
- `loss`: the loss function
- `optimizer`: the optimizer
- `verbose`: if verbose is true, prints the loss each epoch
- `batchsize`: the batch size

# Examples
```julia
train!(model, train_data, epochs, loss, optimizer)
```
"""
function train_model(model, train_data, epochs, loss, optimizer; verbose=false, batchsize=100)
    # Not the best way make dataloader for train, let's not think about it too much, eval task moment
    dataset = DataLoader(train_data, batchsize=batchsize, shuffle=true)

    for epoch in 1:epochs
        Flux.train!(loss, Flux.params(model), ncycle(dataset, length(dataset) ÷ batchsize), optimizer)
        if verbose
            @info "Epoch $epoch and loss $(loss(train_data...))"
        end
    end
end
