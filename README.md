# DAS-N2N

## Installation:

To install package + core dependencies: clone the repository, navigate to the directory and run:
```
pip install .
```

After, to install optional dependencies (e.g., for running example notebooks), run:
```
pip install dasn2n[optional]
pip install dasn2n[jupyter] # If jupyter lab also required
```

## Example usage:

```
from dasn2n import dasn2n

model = dasn2n() # Initialise model
model.load_weights() # Load default weights
# model = model.to('cuda') # Uncomment if CUDA/GPU available

data_denoised = model.denoise_numpy(data) # Denoise 2D numpy array (data) containing DAS data
```
