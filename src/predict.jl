using ArgParse
using Flux, JLD2, CUDA, CSV, DataFrames

include("additionals/data.jl")
include("additionals/model.jl")
include("additionals/train.jl")
include("additionals/evaluate.jl")

"""
    load_checkpoint(model, checkpoint_path)

Load a checkpoint and return the model

# Arguments
- `model`: Flux model
- `checkpoint_path`: path to the checkpoint to load. If it's empty, model will be returned as is.

# Returns
- `model`: the model

# Examples
```julia
model = load_checkpoint(model, "model.jld2")
```
"""
function load_checkpoint(model, checkpoint_path="model.jld2")
    if (checkpoint_path == "")
        return model
    end

    model_state = JLD2.load(checkpoint_path, "model_state")
    Flux.loadmodel!(model, model_state);

    return model
end


"""
    evaluate(model, X, output_path="output.csv")

Evaluate the model and save the predictions to the output path as csv

# Arguments
- `model`: the model
- `X`: the features
- `output_path`: the path to save the predictions

# Examples
```julia
evaluate(model, X)
```
"""
function evaluate(model, X, output_path="output.csv")
    # Transfer to CPU for scalar indexing
    preds = round.(Int, model(X))' |> cpu
    preds = map(pred -> pred == 0 ? "b" : "s", preds)

    CSV.write(output_path, DataFrame(preds, ["class"]))

    return preds
end

function main(args)
    model = build_model()

    model = load_checkpoint(model, args["checkpoint_path"])

    
    X = load_eval_data(args["data_dir"])

    if (CUDA.functional() && args["gpu"])
        model = model |> gpu
        X = X |> gpu
    end
    
    evaluate(model, X)
end

# Parse command-line arguments
s = ArgParseSettings()
@add_arg_table s begin
    "--data_dir"
        help = "Path to the dataset directory"
        required = true
    "--checkpoint_path"
        help = "Path to the trained checkpoint file"
        required = true
    "--gpu"
        help = "Use GPU for inference"
        action = :store_true
end
args = parse_args(s)

main(args)