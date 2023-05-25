abstract type GraphNode end
abstract type Operator <: GraphNode end

struct Constant{T} <: GraphNode
    output :: T
end

mutable struct Variable <: GraphNode
    output :: Any
    gradient :: Any
    name :: String
    Variable(output; name="?") = new(output, nothing, name)
end

mutable struct ScalarOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name :: String
    ScalarOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name :: String
    BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

import Base: show, summary
show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "op ", x.name, "(", F, ")");
show(io::IO, x::BroadcastedOperator{F}) where {F} = print(io, "op.", x.name, "(", F, ")");
show(io::IO, x::Constant) = print(io, "const ", x.output)
show(io::IO, x::Variable) = begin
    print(io, "var ", x.name);
    print(io, "\n ┣━ ^ "); summary(io, x.output)
    print(io, "\n ┗━ ∇ ");  summary(io, x.gradient)
end

function visit(node::GraphNode, visited, order) 
    if node ∉ visited
        push!(visited, node)
        push!(order, node)
    end 
end

function visit(node::Operator, visited, order) 
    if !(node in visited)
        push!(visited, node)
        @inbounds for input in node.inputs
            visit(input, visited, order)
        end
        push!(order, node)
    end 
end

function topological_sort(head::GraphNode) 
    visited = Vector{GraphNode}()
    order = Vector{GraphNode}()
    visit(head, visited, order)
    return order
end

reset!(node::Constant) = nothing
reset!(node::Variable) = node.gradient = nothing
reset!(node::Operator) = node.gradient = nothing

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
compute!(node::Operator) = node.output = forward(node, [input.output for input in node.inputs]...)

function forward!(order::Vector)
    @inbounds  for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end

update!(node::Constant, gradient) = nothing
update!(node::GraphNode, gradient) = if isnothing(node.gradient)
    node.gradient = gradient else node.gradient .+= gradient
end

function backward!(order::Vector; seed=1.0)
    result = last(order)
    result.gradient = seed
    @assert length(result.output) == 1
    @inbounds  for node in reverse(order)
        backward!(node)
    end
    return nothing
end

function backward!(node::Constant) end
function backward!(node::Variable) end
function backward!(node::Operator)
    inputs = node.inputs
    gradients = backward(node, [input.output for input in inputs]..., node.gradient)
    @inbounds for (input, gradient) in zip(inputs, gradients)
        update!(input, gradient)
    end
    return nothing
end

import LinearAlgebra: diagm
import LinearAlgebra: mul!

import Base: *
*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = return A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x, y) = return @. x * y
function backward(node::BroadcastedOperator{typeof(*)}, x, y, g)
    o = ones(length(node.output))
    @views Jx = diagm(vec(y .* o))
    @views Jy = diagm(vec(x .* o))
    gJx = Jx' * g
    gJy = Jy' * g
    @inbounds mul!(gJx, Jx', g)
    @inbounds mul!(gJy, Jy', g)
    return tuple(gJx, gJy)
end

import Base: exp
Base.Broadcast.broadcasted(exp, x::GraphNode) = BroadcastedOperator(exp, x)
forward(::BroadcastedOperator{typeof(exp)}, x) = return exp.(x)
backward(node::BroadcastedOperator{typeof(exp)}, x, grad) = let
    J = diagm(vec(node.output))
    return (J' * grad,)
end

import Base: sum
sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x) = return sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, g) = let
    o = ones(length(x))
    J = o'
    tuple(J' * g)
end

Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x, y) = return @. x / y
function backward(node::BroadcastedOperator{typeof(/)}, x, y::Real, g)
    o = ones(length(node.output))
    Jx = diagm(vec(@. o / y))
    Jy = @. (-x / y ^2)
    mul!(g, Jy, g)
    (Jx' * g, )
end


import Base: log
Base.Broadcast.broadcasted(log, x::GraphNode) = BroadcastedOperator(log, x)
forward(::BroadcastedOperator{typeof(log)}, x) = return log.(x)
backward(::BroadcastedOperator{typeof(log)}, x, g) = (g ./ x',)

using LinearAlgebra
function convOperation(I, K)
    n, m = size(I) .- size(K) .+ 1
    J = zeros(eltype(I), n, m)
    @inbounds for i in 1:n, j in 1:m
        view_I = view(I, i:i+1, j:j+1)
        J[i, j] = dot(view_I, K)
    end
    return J
end

convLayer(x::GraphNode, k::GraphNode) = BroadcastedOperator(convLayer, x, k)
forward(::BroadcastedOperator{typeof(convLayer)}, x, k) = let
    return convOperation(x, k)
end
using LinearAlgebra
function backward(node::BroadcastedOperator{typeof(convLayer)}, I, K, g)
    kh, kw = size(K)
    fgrad = zeros(eltype(I), size(K))
    outh, outw = size(node.output)
    @inbounds for i in 1:outh, j in 1:outw
        @inbounds mul!(view(fgrad, :, :), g[i, j], view(I, i:i+kh-1, j:j+kw-1))
    end    
    return fgrad
end

function convlayerInit(x, k , activation) return activation(convLayer(x, k)) end
function convlayerInit(x, k) return convLayer(x, k) end

function flatten(input)       
    return reshape(input, (:, 1))
end
flatten(x::GraphNode) = BroadcastedOperator(flatten, x)
forward(::BroadcastedOperator{typeof(flatten)}, x) = return flatten(x)
backward(::BroadcastedOperator{typeof(flatten)}, x, grad) = let
    result = reshape(grad, size(x))
    tuple(result)
end

function dense(w, x, activation) return activation(w * x) end
function dense(w, x) return w * x end

relu(x::GraphNode) = BroadcastedOperator(relu, x)
forward(::BroadcastedOperator{typeof(relu)}, x) = return max.(0, x)
backward(::BroadcastedOperator{typeof(relu)}, x, g) = let 
        result = @. g * (x >= 0)
        return tuple(result)
 end

elu(x::GraphNode) = BroadcastedOperator(elu, x)
forward(::BroadcastedOperator{typeof(elu)}, x) = let 
    return @. (x >= 0) * x + 1 * (exp(x) - 1) * (x < 0)
end
backward(::BroadcastedOperator{typeof(elu)}, x, g) = let
    grad = similar(x)
    @. grad = g * ifelse(x >= 0, 1, exp(x))
    return (grad,)
end

softmax(x::GraphNode) = BroadcastedOperator(softmax, x)
forward(::BroadcastedOperator{typeof(softmax)}, x) = let 
    return @. exp(x) / sum(exp(x))
end
backward(node::BroadcastedOperator{typeof(softmax)}, x, g) = let
        y = ones(length(node.output))
        J = @. diagm(vec(node.output * y)) - node.output * node.output'
        tuple(J' * g)
end

function cross_entropy_loss(x::GraphNode, y::GraphNode)
        return sum(@. y * log(exp(x))) * Constant(-1.0)
end

using MLDatasets
using Flux: onehotbatch
function load_fashion_mnist()

    train_data, train_labels = FashionMNIST(split=:train)[1:100]
    test_data, test_labels = FashionMNIST(split=:test)[1:10]

    train_data = train_data ./ 255.0
    test_data = test_data ./ 255.0

    train_data = reshape(train_data, (:, 1, 28, 28))
    test_data = reshape(test_data, (:, 1, 28, 28))

    train_labels = onehotbatch(train_labels, 0:9)
    test_labels = onehotbatch(test_labels, 0:9)

    X = Variable(train_data, name="images")
    y = Variable(train_labels, name = "labels")

    return X, y, test_data, test_labels
end

function net(x::Variable, wc::Variable, wd::Variable, wo::Variable, y::Variable)
        c = convlayerInit(x, wc, relu)
        c.name = "conv layer"
        f = flatten(c)
        d1 = dense(wd, f, elu)
        d1.name = "dense leyer"
        d2 = dense(wo, d1, relu)
        d2.name = "output"
        E = cross_entropy_loss(y, d2)
        E.name = "loss"
    
        return topological_sort(E)
end

function inicializeTestData()
        Wc  = Variable(rand(2,2), name="Wagi conv")
        Wd  = Variable(rand(10,729), name="Wagi dense")
        Wo  = Variable(rand(10,10), name="Wagi out")
        X, Y, test_data, test_labels = load_fashion_mnist()  
        return Wc,Wd,Wo,X, Y
end

function trainSGD(epochs::Int, learning_rate::Real, expectedValueOfLoss::Real)
    Wc, Wd, Wo, X, Y = inicializeTestData()
    forwardVal = 0.0
    sgn = 0.0
    @inbounds for epoch in 1:epochs
        @inbounds for j in 1:size(X.output, 1)
            x = Variable(X.output[j, 1, :, :])
            y = Variable(Y.output[:, j])
            graph = net(x, Wc, Wd, Wo, y)
            forwardVal = forward!(graph)
            if abs(forwardVal) < expectedValueOfLoss
                print("Epoch nr. ", epoch, " Loss: ", round(forwardVal, digits=6), "\n")
                return
            end
            backward!(graph)
            sgn = sign(forwardVal)
            Wc.output .-= learning_rate * sgn * Wc.gradient
            Wd.output .-= learning_rate * sgn * Wd.gradient
            Wo.output .-= learning_rate * sgn * Wo.gradient
        end
        print("Epoch nr. ", epoch, " Loss: ", round(forwardVal, digits=6), "\n")
    end
end

trainSGD(100, 0.001, 0.001)