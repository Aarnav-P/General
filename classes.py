# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:46:45 2023

@author: User
"""
# What is a class?
# A class is similar to an object IRL, which we may describe similarly.

# e.g. what constitutes a person?

class Person:
    def __init__(self,age,weight,height,name,slogan):
        self.age = age
        self.weight = weight
        self.height = height
        self.name = name
        self.slogan = slogan
    def talk(self):
        print("Did you know penguins have their knees above their hips?")
    def joke(self):
        print("Doctor doctor, I feel like I'm losing my mind whenever I walk downstairs!")
        print("Sounds like a descent into madness")
# Now we could assign this object type to something 

user = Person(19,61,171,"Aarnav","Am I crazy or -")

print(user.slogan)
user.joke()
user.talk()

# effectively any object like list, tuple, etc is pretty much a class
# self is essentially a way of calling the instance simply.
