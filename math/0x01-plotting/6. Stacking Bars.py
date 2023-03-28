#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

# your code here
columns=['Farrah', 'Fred', 'Felicia']
rows=['apples', 'bananas', 'oranges', 'peaches']
colors=['red', 'yellow', '#ff8000', '#ffe5b4']

plt.title('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')
plt.ylim((0,80))
plt.yticks(np.arange(0,90,10))
stacked = np.zeros(len(fruit[0]))

for i in range(len(fruit)):
    plt.bar(columns,fruit[i],color=colors[i],label=rows[i],width=0.5,bottom = stacked)
    stacked = stacked + fruit[i]

plt.legend()

plt.show()