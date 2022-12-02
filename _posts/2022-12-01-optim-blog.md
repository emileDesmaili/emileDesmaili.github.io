---
title: 'Numerical Convex Optimization'
date: 2022-12-01
permalink: /posts/2022/12/optim-blog/
tags:
  - homework
  - optimization
  - applied math
---

### Getting inside the black box of vanilla convex Optimization

For many applied data scientists, optimization is still a black box that is the magic used when calling some fitting routine

'''
model = Model()
model.fit(optimizer='gradient_descent')

'''

I will explain two very common descent methods for unconstrained optimization and implement them in python from scratch

### Descent methods, the general principle

Descent methods consist of looking for 

$$x^* = argmin f(x) $$

This is done by specifying a strating value, and descending in that direction

$$ x^(k) = x(k-1)-t*\Delta_t $$

### Looking for t with backtracking line search

the first ingredient we need is need is step size $t$. We will use the backtracking line search algorithm for that. 

'''


'''

### Gradient Descent

With


### The function used

We will optimize the following non-quadratic function

$$ f(x_1,x_2) = e^(x_1+3x_2-0.1) + e^(x_1-3x_2-0.1) + e^(-x_1-0.1) $$
'''

f = lambda x: (np.exp(x[0]+3*x[1]-0.1) + np.exp(x[0]-3*x[1]-0.1) + np.exp(-x[0]-0.1))
#manually derived gradient 
def grad(x):
    dfdx1 = np.exp(x[0]+3*x[1]-0.1) + np.exp(x[0]-3*x[1]-0.1) - np.exp(-x[0]-0.1)
    dfdx2 = 3*np.exp(x[0]+3*x[1]-0.1) - 3*np.exp(x[0]-3*x[1]-0.1)
    return np.array([dfdx1,dfdx2])

'''