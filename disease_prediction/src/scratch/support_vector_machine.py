"""
SVM: Linear Model

	w.x_i - b >= 1	if y_i = 1
	w.x_i - b <= -1	if y_i = -1

	y_i*(w.x_i - b) ?= 1

Loss Function: Hinge Loss

	l = max(0,1-y_i*(w.x_i - b))

	l = {
		0 			if y.f(x) >= 1
		1 - y.f(x)	otherwise
	}

Add Regularization

	J = lambda*||w||^2 + 1/n * sum(max(0,1-y_i*(w.x_i - b)))

	if y_i.f(x) >= 1:
		J_i = lambda||w||^2
	else:
		J_i = lambda||w||^2 + 1 - y_i*(w.x_i - b)


Gradient

"""