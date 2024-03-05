using CSV
using DataFrames
using Random: randperm

"""
    load_data(filename)

Load data from CSV file.

# Arguments
- `filename`: path to CSV file

# Returns
- `X`: features
- `y`: labels

# Examples
```julia
X, y = load_data("data.csv")
```
"""
function load_data(filename)
    df = CSV.read(filename, DataFrame)
    labels = df[:, :class]
    y = reshape(map(label -> label == "b" ? 0 : 1, labels), 1, length(labels))
    X = Matrix(select(df, Not(:class)))'
    return X, y
end

"""
    load_eval_data(filename)

Load data from CSV file for evaluation, skipping the `class` column if there's one.

# Arguments
- `filename`: path to CSV file

# Returns
- `X`: features

# Examples
```julia
X = load_eval_data("data.csv")
```
"""
function load_eval_data(filename)
    df = CSV.read(filename, DataFrame)
    if "class" in names(df)
        df = select(df, Not(:class))
    end
    
    x = Matrix(df)'
    return x
end

"""
    split_data(X, y, train_ratio)

Split data into training and testing sets.

# Arguments
- `X`: features
- `y`: labels
- `train_ratio`: ratio of training data

# Returns
- `train_X`: training features
- `train_y`: training labels
- `test_X`: testing features
- `test_y`: testing labels

# Examples
```julia
train_X, train_y, test_X, test_y = split_data(X, y, 0.8)
```
"""
function split_data(X, y, train_ratio=0.8)
    n = size(X, 2)
    train_indices = randperm(n)[1:floor(Int, n * train_ratio)]
    test_indices = setdiff(1:n, train_indices)

    train_X = X[:, train_indices]
    train_y = y[:, train_indices]

    test_X = X[:, test_indices]
    test_y = y[:, test_indices]

    return (train_X, train_y), (test_X, test_y)
end
