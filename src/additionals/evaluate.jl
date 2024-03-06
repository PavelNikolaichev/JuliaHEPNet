using Statistics


"""
Simple accuracy metric function that is used to evaluate the model

# Arguments
- `model`: the model
- `X`: the features
- `y`: the labels
- `round_output`: if true, rounds the predictions to 0 or 1. Technically use this only if your NN doesn't have rounded output.

# Returns
- `accuracy`: the accuracy of the model in the range [0, 1]

# Examples
```julia
accuracy(model, X, y)
```
"""
function accuracy(model, X, y; round_output=false)
    preds = round_output ? round.(Int, model(X)) : model(X)

    return mean(preds .== y)
end