using ArgParse
using Flux, JLD2, CUDA

include("additionals/data.jl")
include("additionals/model.jl")
include("additionals/train.jl")
include("additionals/evaluate.jl")


"""
    save_model(model, checkpoint_path)

Save the model to the checkpoint path

# Arguments
- `model`: the model
- `checkpoint_path`: the path to save the checkpoint

# Examples
```julia
save_model(model, "model.jld2")
```
"""
function save_model(model, checkpoint_path="model.jld2")
    model = cpu(model)

    jldsave(checkpoint_path; model_state=Flux.state(model))
end

function main(args)
    X, y = load_data(args["data_dir"])

    (train_X, train_y), (test_X, test_y) = split_data(X, y, 0.8)
    if args["gpu"]
        train_X = train_X |> gpu
        train_y = train_y |> gpu
        test_X = test_X |> gpu
        test_y = test_y |> gpu
    end

    model = build_model()
    if args["gpu"]
        model = model |> gpu
    end
    
    loss(x, y) = Flux.binarycrossentropy(model(x), y)
    optimizer = ADAM()

    train_model(model, (train_X, train_y), args["epochs"], loss, optimizer, verbose=args["verbose"], batchsize=args["batchsize"])

    train_acc = accuracy(model, train_X, train_y, round_output=true)
    test_acc = accuracy(model, test_X, test_y, round_output=true)

    println("Train accuracy: ", train_acc)
    println("Test accuracy: ", test_acc)

    save_model(model, args["output_model_path"])
end

# Parse command-line arguments
s = ArgParseSettings()
@add_arg_table s begin
    "--data_dir"
        help = "Path to the dataset directory"
        required = true
    "--epochs"
        help = "Number of epochs for training"
        arg_type = Int
        default = 10
    "--gpu"
        help = "Use GPU for training"
        action = :store_true
    "--verbose"
        help = "Print training progress"
        action = :store_true
    "--batchsize"
        help = "Batch size for training"
        arg_type = Int
        default = 100
    "--output_model_path"
        help = "Path to save the trained model"
        default = "model.jld2"
end
args = parse_args(s)

main(args)