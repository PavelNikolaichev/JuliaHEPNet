using Flux

"""
    build_model(input_size, hidden_size)

Build a simple NN model for binary classification.

# Returns
- `model`: the model

# Examples
```julia
model = build_model()
```
"""
function build_model()
    return Chain(
        Dense(3, 64, σ),
        # Dense(64, 512, σ),
        # Dense(512, 64, σ),
        Dense(64, 1, σ),
    ) |> f64 # Actually consumes more memory, if model architecture is too big, consider converting input types instead
end