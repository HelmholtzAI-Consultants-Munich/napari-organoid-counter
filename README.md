# Napari Organoid Counter - Version 0.2.6 is out! 

[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-organoid-counter)](https://napari-hub.org/plugins/napari-organoid-counter)
![stability-stable](https://img.shields.io/badge/stability-stable-green.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6457903.svg)](https://doi.org/10.5281/zenodo.6457903)
[![License](https://img.shields.io/pypi/l/napari-organoid-counter.svg?color=green)](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-organoid-counter.svg?color=green)](https://pypi.org/project/napari-organoid-counter)
[![Python Version](https://img.shields.io/badge/python-3.10-blue)](https://python.org)
[![tests](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/workflows/tests/badge.svg)](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/actions)
[![codecov](https://codecov.io/gh/HelmholtzAI-Consultants-Munich/napari-organoid-counter/branch/main/graph/badge.svg)](https://codecov.io/gh/HelmholtzAI-Consultants-Munich/napari-organoid-counter)


A napari plugin to automatically count lung organoids from microscopy imaging data. Note: this plugin only supports single channel grayscale images.

***Hold it for the demo!***

![Demo](readme-content/demo-v026.gif)

This demo showcases the features of Version 0.2, while the new functionalities of the latest version are detailed in the "Latest version features" section.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.


## Installation

This plugin has been tested with python 3.10 - you may consider using conda or mamba to create your dedicated environment before running the `napari-organoid-counter`.

1. You can install `napari-organoid-counter` via [pip](https://pypi.org/project/napari-organoid-counter/):

    ``` pip install napari-organoid-counter```

   To install for a developer:

    ```git clone https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter```
    ```pip install -e .```

For installing on a Windows machine directly from within napari, follow the instuctions [here](readme-content/How%20to%20install%20on%20a%20Windows%20machine.pdf).

## Latest version features
Checkout *Features of the latest version* [here](.napari/DESCRIPTION.md#whats-new-in-v3).

## How to use?
1. To launch the plugin directly from your terminal:
```bash
napari -w napari-organoid-counter
```

2. For convenience you can add a shell alias  by adding this to your `~/.zshrc` (macOS) or `~/.bashrc` (Linux):
```bash
alias organoid='napari -w napari-organoid-counter'
```

Then reload your shell config:
```bash
source ~/.zshrc   # macOS
source ~/.bashrc  # Linux
```

You can now launch the plugin simply by running:
```bash
organoid
```

3. Alternatively, you can also start napari manually and select the plugin from the drop down menu.

For more information on this plugin, its intended audience, and a Quickstart guide go to our [Quickstart guide](.napari/DESCRIPTION.md#quickstart).

## Contributing

Contributions are very welcome. Tests can be run with [pytest], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-organoid-counter" is free and open source software

## Dependencies


```napari-organoid-counter``` uses the ```BioIO```<sup>[1]</sup> for reading and processing images.

[1] Eva Maxfield Brown, Dan Toloudis, Jamie Sherman, Madison Swain-Bowden, Talley Lambert, Sean Meharry, Brian Whitney, AICSImageIO Contributors (2023). BioIO: Image Reading, Metadata Conversion, and Image Writing for Microscopy Images in Pure Python [Computer software]. GitHub. https://github.com/bioio-devs/bioio

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/issues

[napari]: https://github.com/napari/napari
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

## Citing

If you use this plugin for your work, please cite it using the following:

> Christina Bukas, Harshavardhan Subramanian, & Marie Piraud. (2023). HelmholtzAI-Consultants-Munich/napari-organoid-counter: v0.2.0 (v0.2.0). Zenodo. https://doi.org/10.5281/zenodo.7859571
> 
bibtex:
```
@software{christina_bukas_2022_6457904,
  author       = {Christina Bukas, Harshavardhan Subramanian, & Marie Piraud},
  title        = {{HelmholtzAI-Consultants-Munich/napari-organoid- 
                   counter: second release of the napari plugin for lung
                   organoid counting}},
  month        = apr,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.2.0},
  doi          = {10.5281/zenodo.7859571},
  url          = {https://doi.org/10.5281/zenodo.7859571}
}
```
