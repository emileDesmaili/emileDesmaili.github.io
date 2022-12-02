---
title: 'Numerical Convex Optimization'
date: 2022-12-01
permalink: /posts/2022/12/optim-blog/
tags:
  - homework
  - optimization
  - applied math
---

### Getting inside the black box of vanilla convex optimization

For many applied data scientists, optimization is still a black box that is the magic used when calling some fitting routine

```
model = Model()
model.fit(optimizer='gradient_descent')
```

I will explain two very common descent methods for unconstrained optimization and implement them in python from scratch

### Descent methods, the general principle

Descent methods consist of looking for 

$$x^* = argmin f(x) $$

This is done by specifying a strating value, and descending in a certain direction ($\Delta$) at a certain rate ($t$)

$$ x^{k+1} = x^{k}-t\Delta_k $$

### Looking for t with backtracking line search

the first ingredient we need is need is step size $t$. We will use the backtracking line search algorithm for that. 

```
def backtracking_line_search(fun, x, grad, delta_x, alpha, beta):
    t = 1
    prod = grad(x).T @ delta_x
    while fun((x + (t * delta_x))) > fun(x) + (alpha * t * prod): #stopping criterion from slides
        t *= beta

    return t
```

### Gradient Descent

With gradient descent the direction is given by the gradient of the objective function, so the update at each step becomes:
$$ x^{k+1}= x^{k}-t\nabla f(x^{k}) $$

Tha criterion to stop is reached when $$||\nabla f(x)|| < \epsilon $$
with $$\epsilon$$ > 0  specified by the user

Here is the code, it returns all the values of the sequence $x^{k}$:

```
def gd(fun, x0, grad, alpha=0.1, beta=0.7, epsilon=1e-3):
    x_new = x0
    x_ = [x0]
    while np.linalg.norm(grad(x_new)) > epsilon:  #stopping criterion

        delta_x = -grad(x_new)
        t = backtracking_line_search(fun, x_new, grad, delta_x, alpha, beta) 
        x_new = x_new + (t * delta_x)
        x_.append(x_new)
    else:
        return np.array(x_)
```

### Newton's method

Newton's method is a **second-order** method, meaning it uses more information than gradient descent which uses only the gradient, so first-order information. 

Here we update the diretion using the gradient and the Hessian matrix of f. This means the algorithm converges faster (for quadratic functions, in 1 step), because it has more information when descending, but it has to invert the Hessian matrix, which is computationally intensive - $ \Omega(n^3)$. To alleviate that, one could also decide ot update the descent every $r$ steps (done by adding a simple conditional statement in the code).

The descent is given by:

$$ x^{k+1} = x^{k}-t\nabla^2 f(x^{k})^{-1} \nabla f(x^{k}) $$

The stopping criterion is reached when $$\lambda^2/2 < \epsilon$$

$\lambda^2$ here is the Newton decrement. The formula for it is as follows:

$$ \lambda^2 = \nabla f(x^{k})^T\nabla^2 f(x^{k})^{-1}\nabla f(x^{k}) $$

```
def newton(fun, x0, grad, hess, alpha=0.1, beta=0.7, epsilon=1e-10, r=3):
    x_new=x0
    x_=[x0]
    i = 0
    D = np.linalg.inv(hess(x_new)) # starting value for Dk

    while np.dot((np.dot(grad(x_new).T,D)),grad(x_new))/2 > epsilon:  #stopping criterion
        
        if i % r == 0:                      # only update D modulo r
            D = np.linalg.inv(hess(x_new))
        delta_x = -D @ grad(x_new)
        t = backtracking_line_search(fun, x_new, grad, delta_x, alpha, beta) 
        x_new = x_new + (t * delta_x)
        x_.append(x_new)
        i+=1
    else:
        return np.array(x_)

```


### Application 

We will optimize the following non-quadratic function

$$ f(x_1,x_2) = e^{x_1+3x_2-0.1} + e^{x_1-3x_2-0.1} + e^{-x_1-0.1} $$


```
f = lambda x: (np.exp(x[0]+3*x[1]-0.1) + np.exp(x[0]-3*x[1]-0.1) + np.exp(-x[0]-0.1))

#manually derived gradient 

def grad(x):
    dfdx1 = np.exp(x[0]+3*x[1]-0.1) + np.exp(x[0]-3*x[1]-0.1) - np.exp(-x[0]-0.1)
    dfdx2 = 3*np.exp(x[0]+3*x[1]-0.1) - 3*np.exp(x[0]-3*x[1]-0.1)
    return np.array([dfdx1,dfdx2])

#hessian

def hess(x):
    dfdx1x1 = np.exp(x[0]+3*x[1]-0.1) + np.exp(x[0]-3*x[1]-0.1) + np.exp(-x[0]-0.1)
    dfdx1x2 = 3*np.exp(x[0]+3*x[1]-0.1) -3*np.exp(x[0]-3*x[1]-0.1)
    dfdx2x2 = 9*np.exp(x[0]+3*x[1]-0.1) + 9*np.exp(x[0]-3*x[1]-0.1)
    dfdx2x1 = 3*np.exp(x[0]+3*x[1]-0.1) - 3*np.exp(x[0]-3*x[1]-0.1)
    return np.array([[dfdx1x1,dfdx1x2],[dfdx2x1,dfdx2x2]])
```

#### Gradient Descent plot

![gd](http://emileDesmaili.github.io/images/blog_optim/gdplot.png)

#### Newton plots

Here I plot the two ways of implementing Newton's method. 

![newton](http://emileDesmaili.github.io/images/blog_optim/newton.png)

As we can see, Newton's method converges faster than gradient descent