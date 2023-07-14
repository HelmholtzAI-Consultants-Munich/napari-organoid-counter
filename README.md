# Napari Organoid Counter - Version 0.2 is out! 

![stability-stable](https://img.shields.io/badge/stability-stable-green.svg)
[![DOI](https://zenodo.org/badge/476715320.svg)](https://zenodo.org/badge/latestdoi/476715320)
[![License](https://img.shields.io/pypi/l/napari-organoid-counter.svg?color=green)](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-organoid-counter.svg?color=green)](https://pypi.org/project/napari-organoid-counter)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://python.org)
[![tests](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/workflows/tests/badge.svg)](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/actions)
[![codecov](https://codecov.io/gh/HelmholtzAI-Consultants-Munich/napari-organoid-counter/branch/main/graph/badge.svg)](https://codecov.io/gh/HelmholtzAI-Consultants-Munich/napari-organoid-counter)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-organoid-counter)](https://napari-hub.org/plugins/napari-organoid-counter)

A napari plugin to automatically count lung organoids from microscopy imaging data. *Note that this plugin only supports single channel grayscale images.*

![Alt Text](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/blob/main/readme-content/demo-plugin-v2.gif)

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.


## Installation

You can install `napari-organoid-counter` via [pip]:

    pip install napari-organoid-counter


To install latest development version :

    pip install git+https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter.git
    
    
For the dev branch you can clone this repo and install with:

    pip install -e .  

Then run napari on your terminal.


## What's new in v2?
Checkout our *What's New in v2* [here](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/blob/main/.napari/DESCRIPTION.md#whats-new-in-v2).

## How to use?
After installing, you can start napri (either by typing ```napari``` in your terminal or by launching the application) and select the plugin from the drop down menu.

For more information on this plugin, its' intended audience, as well as Quickstart guide go to our [Quickstart guide](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/blob/main/.napari/DESCRIPTION.md#quickstart).

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-organoid-counter" is free and open source software

## Dependencies


```napari-organoid-counter``` uses the ```napari-aicsimageio```<sup>[1]</sup> <sup>[2]</sup> plugin for reading and processing CZI images.

[1] Eva Maxfield Brown, Dan Toloudis, Jamie Sherman, Madison Swain-Bowden, Talley Lambert, AICSImageIO Contributors (2021). AICSImageIO: Image Reading, Metadata Conversion, and Image Writing for Microscopy Images in Pure Python [Computer software]. GitHub. https://github.com/AllenCellModeling/aicsimageio

[2] Eva Maxfield Brown, Talley Lambert, Peter Sobolewski, Napari-AICSImageIO Contributors (2021). Napari-AICSImageIO: Image Reading in Napari using AICSImageIO [Computer software]. GitHub. https://github.com/AllenCellModeling/napari-aicsimageio

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
[tox]: https://tox.readthedocs.io/en/latest/
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

