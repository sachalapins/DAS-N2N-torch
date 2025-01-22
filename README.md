# DAS-N2N
Code repository for DAS denoising model described in paper "DAS-N2N: Machine learning Distributed Acoustic Sensing (DAS) signal denoising without clean data" ([https://doi.org/10.1093/gji/ggad460](https://doi.org/10.1093/gji/ggad460)).

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
from dasn2n import DASN2N # Import main model class

model = DASN2N() # Initialise model
model.load_weights() # Load default weights (currently the ones from the paper)
# model = model.to('cuda') # Uncomment if CUDA-enabled GPU available
# model = model.to('mps') # Uncomment if Apple GPU available

data_denoised = model.denoise_numpy(data) # Denoise 2D numpy array (data) containing DAS data
```

See notebooks in `examples` directory for more guidance.

If you would like further guidance on training/implementing a DAS-N2N model for your own DAS data, please feel free to get in touch: [sacha.lapins@bristol.ac.uk](mailto:sacha.lapins@bristol.ac.uk)
