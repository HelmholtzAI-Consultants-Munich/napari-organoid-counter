# napari-organoid-counter

[![DOI](https://zenodo.org/badge/476715320.svg)](https://zenodo.org/badge/latestdoi/476715320)
[![License](https://img.shields.io/pypi/l/napari-organoid-counter.svg?color=green)](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-organoid-counter.svg?color=green)](https://pypi.org/project/napari-organoid-counter)
[![Python Version](https://img.shields.io/pypi/pyversions/{{cookiecutter.plugin_name}}.svg?color=green)](https://python.org)
[![tests](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/workflows/tests/badge.svg)](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/actions)
[![codecov](https://codecov.io/gh/HelmholtzAI-Consultants-Munich/napari-organoid-counter/branch/main/graph/badge.svg)](https://codecov.io/gh/HelmholtzAI-Consultants-Munich/napari-organoid-counter)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-organoid-counter)](https://napari-hub.org/plugins/napari-organoid-counter)

A napari plugin to automatically count lung organoids from microscopy imaging data. The original implementation can be found in the [Organoid-Counting](https://github.com/HelmholtzAI-Consultants-Munich/Organoid-Counting) repository, which has been adapted here to work as a napari plugin.

![Alt Text](https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter/blob/main/readme-content/demo-plugin.gif)

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/stable/index.html
-->

## Installation

You can install `napari-organoid-counter` via [pip]:

    pip install napari-organoid-counter



To install latest development version :

    pip install git+https://github.com/HelmholtzAI-Consultants-Munich/napari-organoid-counter.git


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-organoid-counter" is free and open source software

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
```
@software{christina_bukas_2022_6457904,
  author       = {Christina Bukas},
  title        = {{HelmholtzAI-Consultants-Munich/napari-organoid- 
                   counter: first release of napari plugin for lung
                   organoid counting}},
  month        = apr,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {v0.1.0-beta},
  doi          = {10.5281/zenodo.6457904},
  url          = {https://doi.org/10.5281/zenodo.6457904}
}
```
