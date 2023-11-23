# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 20:51:18 2023

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
import torch

# For a set p of N numbers, what is the set q s.t. the function:
# H(p,q) = - \sum{i=1}{N} p_i \ln(q_i) 
# is minismised, constrained by \sum p_i = \sum q_i

# we can examine a few sets

p = np.array([5,1,4,6,2,4])
q1 = p
q2 = np.array([3,7,1,4,1,6])
q3 = np.array([2,5,7,2,1,5])

H1 = -sum(p*np.log(q1))
H2 = -sum(p*np.log(q2))
H3 = -sum(p*np.log(q3))
print(f"H1 is {H1}, H2 is {H2}, H3 is {H3}") # note H1 < H2 < H3

# We can prove with Lagrange multipliers that the optimal set is p=q. Try it as a reminder exercise if you ever forget.
# n.b. if you work this through again, you'll notice it works for p and q being functions rather than constants.
# For a reminder, check the lagrange multiplier section of Year 1 Maths notes.

# so we know that the cross entropy in some sense measures the difference between p and q. How do we use this?

#CLASSIFICATION
# Let's say you're trying to identify a certain input as belonging to one of 5 categories
# We will define the "true" likelihood of an image 'x' belonging to a class 'i' as p_i. 
# The classifier needs to create a function f s.t. f(x) = q, where q should try to approach p
# Therefore p,q are really pdf's, and as such the sum of all p_i and q_i should equal the same value, namely 1. This is the source of the constraint.
# To provide an example, you would have a vector p for each image, with a weighting in each row signifying the probability of belonging to that class. Say we're choosing between a dog, wolf or hyena. For a given image the p vector may be [0.5 0.35 0.15]

#[NOTE TO SELF, start using jupyter notebooks to help with the formatting here?]

# We have found a function wherein it is minimised when p=q, and where sum of p_i = sum of q_i
# This is perfect for this application.

# Let's see a simple example:
    
p = np.zeros(10); p[4] = 1
q = np.random.rand(10)
q = q/sum(q) #normalise
H_example_1 = -sum(p*np.log(q))
print(f"The cross entropy for this example is {H_example_1}")

# since we know where q should be highest, let's artifically inflate that element.

q[4] = 200
q = q/sum(q) #renormalising
H_example_2 = -sum(p*np.log(q))
print(f"The manually adjusted cross entropy for this example is {H_example_2}")

# This just confirms what we expected. How do we extend this to N images?
# We may imagine c~ as the true class' index. So let's take some 5 images where c~(n) is known
# So we will have q(c~(n)) as the __predicted__ probability of belonging to the correct class.
# Let L(p,q) = \sum H(p_n,q_n) from n=i to N. Since we have defined c~(n), all other p will be 0, so this simplifies to - \sum /ln(q_n(c~(n)))

p = np.zeros((4,10))
p[0,4]=1
p[1,2]=1
p[2,5]=1
p[3,9]=1
print(p)

q = np.random.rand(40).reshape(4,10)
yhat = q
i=0
for line in q:
    line = line/sum(line)
    q[i] = line
    i+=1
    
# alternative, quicker method:
q2 = q/np.expand_dims(np.sum(q,axis=1), axis=1)

H_example_3 = (np.log(q[p>0])) 
L = -sum(H_example_3)
print(f"The cross entropy for this example is {L}")

H_example_4 = (np.log(q2[p>0])) 
L2 = -sum(H_example_4)
print(f"The cross entropy for this example is {L2}")

# The normalisation is done differently in PyTorch.
# If \hat y_n is the vector output of f(x), where len(\hat y_n) = no. of classes
# q is normalised as q_n(c) = exp[\hat y_n(c)] / \Sum exp[\hat y_n(c')] from c'=0 to C
# remember n is the image index and c is the class index.

q3 = np.exp(q)
q3 = q3/np.expand_dims(np.sum(q3,axis=1), axis=1)
H_example_5 = (np.log(q3[p>0])) 
L3 = -sum(H_example_5)
print(f"The cross entropy for this exponentially normalised example is {L3}")

# equivalent in pytorch

L = torch.nn.CrossEntropyLoss(reduction='sum')
print(L((torch.tensor(yhat)), torch.tensor(p, dtype= torch.float)))

#TBC