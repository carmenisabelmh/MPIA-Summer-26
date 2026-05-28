import numpy as np
import matplotlib.pyplot as plt

# Load the loss
loss = np.load('loss_curve 20260528 n16000 lr5e-4 2restarts.npy')

# Plot loss

plt.title('Test 3: Loss Over 16000 Steps with a LR of 5e-4 over 2 Cosine Annealing Restarts')
plt.xlabel('Number of Steps')
plt.ylabel('Loss')
plt.plot(loss, color = 'pink')
plt.savefig('Test 3 20260528 n16000 lr5e-4 2restarts.')
plt.show()



