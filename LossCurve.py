# import numpy as np
# import matplotlib.pyplot as plt

# # Load the loss
# loss = np.load('loss_curve.npy')

# # Plot loss

# plt.title('Test 6: Loss Over 20000 Steps with a LR of 1e-4 over 4 Cosine Annealing Restarts')
# plt.xlabel('Number of Steps')
# plt.ylabel('Loss')
# plt.yscale(f'log')
# plt.plot(loss[::1000], color = 'pink')
# plt.savefig('Test 6 20260528 n20000 lr1e-4 4restarts')
# plt.show()



import numpy as np
import matplotlib.pyplot as plt

def smooth(y, window=50):
    return np.convolve(y, np.ones(window)/window, mode='valid')

files = {
    'Test 1: lr 1e-4 r 1': 'loss_curve 20260527 n8000 lr1e-4.npy',
    'Test 2: lr 1e-4 r 2': 'loss_curve 20260528 n16000 lr1e-4 2restarts.npy',
    'Test 3: lr 5e-4 r 2': 'loss_curve 20260528 n16000 lr5e-4 2restarts.npy',
    'Test 4: lr 5e-4 r 1': 'loss_curve 20260528 n16000 lr5e-4.npy',
    'Test 5: lr 1e-5 r 4': 'loss_curve 20260528 n16000 lr5e-5 4restarts.npy',
    'Test 6: lr 5e-4 r 4': 'loss_curve 20260528 n20000 lr5e-4 4restarts.npy'

}

plt.figure(figsize=(12, 6))

for label, filepath in files.items():
    loss = np.load(filepath)
    plt.plot(smooth(loss[::100], window=50), label=label, alpha=0.8)

plt.title('Loss Curves Comparison')
plt.xlabel('Number of Steps')
plt.ylabel('Loss')
plt.yscale(f'log')
plt.legend()
plt.tight_layout()
plt.savefig('loss_curves_comparison.png', dpi=150)
plt.show()
