# Introduction

---

This repository contains the main source code for the Köppen-Geiger Sentinel-2 Satellite Climate Classification final project for CS 6140: Machine Learning.
Before setting up the project, make sure you have downloaded the dataset [here](https://northeastern-my.sharepoint.com/:f:/r/personal/salchert_b_northeastern_edu/Documents/cs6140_climate_project_data?csf=1&web=1&e=r2y4k5).

To ensure the project sees the dataset, please make sure that the dataset folder is named `data` and is located inside the project root directory.
The relative filepath should look like this:

`KoppenSatelliteClimateClassification\data`

# Setup

---

To set up the project, first you must create the Conda environment with the required dependencies.
First, make sure Conda is installed.
You can find the installation instructions [here](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html).

We recommend using PyCharm as your IDE for running the project, though any IDE should work.
Open your terminal and navigate to the project root directory.
Then, run

`conda create env -f environment.yml`

to create the Conda environment for the project.

After creating the environment, activate it using the command:

`conda activate climate_project_env`

PyTorch can create some conflicts when included in the `environment.yml` file, so you will need to install it manually using the following Pip command:

`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`

At this point, the environment setup is complete.
If you are using PyCharm, navigate to Settings → Python → Interpreter and add a new local interpreter.
Choose "Select existing", select the type as "Conda", and then select `climate_project_env` as the environment.
Press "OK" on all windows.

The project is now ready to run!

# Usage

---

