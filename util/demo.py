vocab = list(open('vocab.txt'))[0]

print(len(vocab))
import numpy as np 
np.random.seed(1234)
idx = np.random.randint(len(vocab))
print(idx)