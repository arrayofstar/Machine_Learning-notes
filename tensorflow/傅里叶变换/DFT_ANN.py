"""
link: https://sidsite.com/posts/fourier-nets/
https://gist.github.com/endolith/98863221204541bf017b6cae71cb0a89

Train a neural network to implement the discrete Fourier transform
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

N = 64
batch = 10000

# Generate random input data and desired output data
sig = np.random.randn(batch, N) + 1j*np.random.randn(batch, N)
F = np.fft.fft(sig, axis=-1)

# First half of inputs/outputs is real part, second half is imaginary part
X = np.hstack([sig.real, sig.imag])
Y = np.hstack([F.real, F.imag])

# Create model with no hidden layers, same number of outputs as inputs.
# No bias needed.  No activation function, since DFT is linear.
model = Sequential([Dense(N*2, input_dim=N*2, use_bias=False)])
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=300, batch_size=100)

# Confirm that it works
data = np.arange(N)


def ANN_DFT(x):
    if len(x) != N:
        raise ValueError(f'Input must be length {N}')
    pred = model.predict(np.hstack([x.real, x.imag])[np.newaxis])[0]
    result = pred[:N] + 1j*pred[N:]
    return result


ANN = ANN_DFT(data)
FFT = np.fft.fft(data)
print(f'ANN matches FFT: {np.allclose(ANN, FFT)}')

# Heat map of neuron weights
plt.imshow(model.get_weights()[0], vmin=-1, vmax=1, cmap='coolwarm')
plt.show()