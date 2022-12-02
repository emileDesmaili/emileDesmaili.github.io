---
title: 'Numerical Convex Optimization'
date: 2022-12-01
permalink: /posts/2022/12/optim_blog/
tags:
  - homework
  - optimization
  - applied math
---

### Getting inside the black box of vanilla convex Optimization

for many applied data scientists, optimization is still a black box that is the magic used when calling some fitting routine

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



