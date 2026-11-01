# CoDeX

CoDeX contains learned data compression tools for JAX.

You can use this library to build your own ML models with end-to-end optimized data
compression built in. It's useful to find storage-efficient representations of your data
(images, features, examples, etc.) while only sacrificing a small fraction of model
performance.

For a more in-depth introduction from a classical data compression perspective, consider
our [paper on nonlinear transform coding](https://arxiv.org/abs/2007.03034), or watch
@jonaballe's [talk on learned image
compression](https://www.youtube.com/watch?v=x_q7cZviXkY). For an introduction to lossy
data compression from a machine learning perspective, take a look at @yiboyang's [review
paper](https://arxiv.org/abs/2202.06533).

## Documentation & getting help

Please post all questions or comments on
[Discussions](https://github.com/balle-lab/google/codex/discussions). Only file
[Issues](https://github.com/balle-lab/codex/issues) for actual bugs or feature requests.
On Discussions, you may get a faster answer, and you help other people find the question
or answer more easily later.

## Installation

To install CoDeX via `pip`, run the following command:

```bash
python -m pip install git+https://github.com/balle-lab/codex.git
```

To test that the installation works correctly, you can run the unit tests with:

```bash
python -m pip install pytest chex tensorflow-probability flaxmodels
pytest --pyargs --ignore-glob="**/range_ans_*" codex
```

Once the command finishes, you should see a message ```74 passed in 21.37s``` or similar
in the last line.

## Usage

We recommend importing the library from your Python code and using it as follows:

```python
import codex as cdx

... = cdx.ops.ste_round(...)
```

Entropy models can be imported with support for
[Equinox](https://github.com/patrick-kidger/equinox) or for
[Flax](https://github.com/google/flax):

```python
from codex.ems import equinox as ems
# or:
from codex.ems import flax as ems

... = ems.RealMappedFourierEntropyModel(...)
```

## Citations

If you use this library for scientific or research purposes, please refer to the
docstrings in the individual functions and classes to see which papers to cite.
