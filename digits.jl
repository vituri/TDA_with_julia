using MLJ
using DataFrames

SVMClassifier = @load SVMClassifier pkg=MLJScikitLearnInterface
scitype(y)

y = coerce(y, Multiclass)
X2 = DataFrame(X_all, :auto)

model = SVMClassifier(C=3.0, degree = 2)
mach = machine(model, X2, y)
train, test = partition(eachindex(y), 0.7); # 70:30 split
fit!(mach, rows=train);
yhat = predict(mach, X2[train,:]);
(yhat .== y[train]) |> mean
yhat = predict(mach, X2[test,:]);
(yhat .== y[test]) |> mean

X

scitype(y)

models()
models(matching(X, y))

X2 = X'