# How to use our public dimensional emotion model

An introduction to our model for 
dimensional speech emotion recognition based on
[wav2vec 2.0](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/).
The model is available from 
[doi:10.5281/zenodo.6221127](https://doi.org/10.5281/zenodo.6221127)
and released under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The model was created
by fine-tuning the pre-trained
[wav2vec2-large-robust](https://huggingface.co/facebook/wav2vec2-large-robust)
model on
[MSP-Podcast](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html)
(v1.7).

## Quick start

Create / activate Python virtual environment and install 
[audonnx](https://github.com/audeering/audonnx).

```
$ pip install audonnx
```

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
signal = np.random.normal(size=sampling_rate).astype(np.float32)
model(signal, sampling_rate)
```
```
{'hidden_states': array([[-0.00711814,  0.00615957, -0.00820673, ...,  0.00666412,
          0.00952989,  0.00269193]], dtype=float32),
 'logits': array([[0.6717072 , 0.6421313 , 0.49881312]], dtype=float32)}
```

The hidden states might be used as embeddings
for related speech emotion recognition tasks.
The order in the logits output is:
arousal,
dominance,
valence.

## Tutorial

For a detailed introduction, please check out the [notebook](./notebook.ipynb).

```
$ pip install -r requirements.txt
$ jupyter notebook notebook.ipynb 
```

## Citation

TBA
