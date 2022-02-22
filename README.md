# How to use our wav2vec 2.0 dimensional emotion model

## Installation

Create and activate Python virtual environment. 
Then install dependencies.

```
$ pip install -r requirements
```

## Quick start

Load model and test on random signal.

```python
import audeer
import audonnx
import numpy as np


url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
cache_root = audeer.mkdir('cache')
model_root = audeer.mkdir('model')

archive_path = audeer.download_url(url, cache_root, verbose=True)
audeer.extract_archive(archive_path, model_root)
model = audonnx.load(model_root)

sampling_rate = 16000
signal = np.random.normal(sampling_rate).astype(np.float32)
model(signal, sampling_rate)
```
```
{'hidden_states': array([[-0.00711814,  0.00615957, -0.00820673, ...,  0.00666412,
          0.00952989,  0.00269193]], dtype=float32),
 'logits': array([[0.6717072 , 0.6421313 , 0.49881312]], dtype=float32)}
```

### Notebook

For a detailed introduction, please check out the notebook.

```
$ jupyter notebook notebook.ipynb 
```
