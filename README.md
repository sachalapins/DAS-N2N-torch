# dasn2n

## Installation:

To install package + core dependencies, run:
```
pip install .
```

After, to install optional dependencies (e.g., for running example notebooks), run:
```
pip install DASdenoiser[optional]
pip install DASdenoiser[jupyter] # If jupyter lab also required
```

## Example usage:

```
from dasn2n import dasn2n

model = dasn2n() # Initialise model
model.load_weights() # Load default weights
model = model.to('cuda') # If CUDA/GPU available

data_denoised = model.denoise(data) # Denoise 2D numpy array (data) containing DAS data
```