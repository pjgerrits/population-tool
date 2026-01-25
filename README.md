# Population Tool

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mapaction-population-tool.streamlit.app/)
[![license](https://img.shields.io/github/license/OCHA-DAP/pa-aa-toolbox.svg)](https://github.com/mapaction/population-tool/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

**NOTE: The Streamlet app is not currently operational. Please contact the team on MapAction slack for more information.**

This repository contains a Streamlit app that enables users to extract population information (aggregated and disaggregated by age and gender) for specific areas defined by shapefiles.

The population information is retrieved from <a href='https://www.worldpop.org/'>WorldPop</a>, an open-access data repository that focuses on generating and providing detailed population data for different regions and countries around the world. The aim of WorldPop is to improve our understanding of human population dynamics and their interactions with social, economic, and environmental factors. The organization utilizes various sources of data, including census records, satellite imagery, household surveys, and other demographic information, to estimate population distribution and demographic characteristics at high spatial resolutions. WorldPop employs advanced spatial modeling techniques and statistical methods to generate population estimates at fine-scale resolutions (100 meters in this case).

The accuracy of the results provided by the tool may vary depending on the region due to limitations of the WorldPop's population data, in terms of spatial and temporal resolution, as well as inherent uncertainty resulting from modeling and statistical techniques used for estimation.


## Usage

#### Requirements

The Python version currently used is 3.9. Please install all packages from
``requirements.txt``:

```shell
pip install -r requirements.txt
```

#### Google Earth Engine authentication

[Sign up](https://signup.earthengine.google.com/) for a Google Earth Engine account, if you don't already have one. Open a terminal window, type `python` and then paste the following code:

```python
import ee
ee.Authenticate()
```

Log in to your Google account to obtain the authorization code and paste it back into the terminal. Once you press "Enter", an authorization token will be saved to your computer under the following file path (depending on your operating system):

- Windows: `C:\\Users\\USERNAME\\.config\\earthengine\\credentials`
- Linux: `/home/USERNAME/.config/earthengine/credentials`
- MacOS: `/Users/USERNAME/.config/earthengine/credentials`

The credentials will be used when initialising Google Earth Engine in the app.

#### Run the app

Open a terminal and run

```shell
streamlit run app/Home.py
```

A new browser window will open and you can start using the tool.

## Contributing

#### Pre-commit

All code is formatted according to
[black](https://github.com/psf/black) and [flake8](https://flake8.pycqa.org/en/latest) guidelines. The repo is set-up to use [pre-commit](https://github.com/pre-commit/pre-commit). Please run ``pre-commit install`` the first time you are editing. Thereafter all commits will be checked against black and flake8 guidelines.

To check if your changes pass pre-commit without committing, run:

```shell
pre-commit run --all-files
```
