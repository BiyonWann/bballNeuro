# bballNeuro

This project trains a competitive learning network on the court locations of NBA shots. The network is never told where the 3-point line is. We check whether it finds the line on its own, and how its map of the court changes from 2000-01 to 2024-25.

## Set up locally

You need Python 3.11 or newer.

1. Clone the repo and move into it.

```
git clone https://github.com/BiyonWann/bballNeuro.git
cd bballNeuro
```

2. Create a virtual environment.

```
python -m venv .venv
```

3. Activate it. On macOS or Linux:

```
source .venv/bin/activate
```

On Windows:

```
.venv\Scripts\activate
```

4. Install the packages. `notebook` is only needed to open the notebook in Jupyter.

```
pip install -r requirements.txt -r requirements-dev.txt
pip install notebook
```

`requirements.txt` holds numpy, pandas, matplotlib and scikit-learn. `requirements-dev.txt` adds pytest, ruff, nbconvert and ipykernel.

## Run the notebook

Run the notebook from the repo root, so Python can find the `threept` folder.

To work in the notebook, start Jupyter and open `notebook.ipynb`:

```
jupyter notebook
```

To run it top to bottom and save the result in `executed.ipynb`:

```
jupyter nbconvert --to notebook --execute notebook.ipynb --output executed.ipynb
```

The first run downloads the four season files, about 160 MB once unpacked, so it needs internet. The whole notebook takes about two minutes.

## Where the data comes from

The shots come from the public repository `shufinskiy/nba_data`, which mirrors the NBA's shot-chart data. The project uses four seasons, named by the year they start: `shotdetail_2000` is 2000-01, then 2008, 2015 and 2024.

The code downloads each season the first time it is needed. It extracts `shotdetail_<year>.csv` into `data/` and deletes the archive. Later runs read the saved file and skip the network. To download a season again, delete its CSV. Git ignores `data/`.

If the season files are already in another folder, set `BBALL_DATA_DIR` to that folder before you start the notebook. Without it, the notebook uses `data/`.

On macOS or Linux:

```
BBALL_DATA_DIR=/path/to/your/data jupyter notebook
```

On Windows (cmd):

```
set BBALL_DATA_DIR=C:\path\to\your\data
jupyter notebook
```

## Run it in Google Colab

1. Public repo: open colab.research.google.com, choose File, then Open notebook, then the GitHub tab, and enter `BiyonWann/bballNeuro`. Choose Runtime, then Run all. The first cell clones the repo and installs the requirements.
2. Private repo: the clone will fail, so upload the files instead. Zip the project folder, upload it in Colab's Files panel and unzip it. Add a cell at the top with the line below, then choose Run all. The first cell sees the files and only installs the requirements.

```
%cd <your folder>
```

3. The data downloads into the Colab session. It is lost when the runtime resets, so the next session downloads it again.

## What is in the repo

```
notebook.ipynb         the experiments and plots, imports everything from threept/
threept/
  data.py              download, clean, convert to feet and polar, split by game
  network.py           the competitive learning network, written in numpy
  metrics.py           evaluation metrics and baselines
  court.py             half-court drawing and plots
  experiments.py       helpers that run and summarize the experiments
tests/                 one test file per module, small fixtures in tests/fixtures/
docs/proposal.md       the project proposal
requirements.txt       packages the notebook needs
requirements-dev.txt   packages for tests, linting and running the notebook
data/                  downloaded season files, created on first run, not in git
```

## Run the tests

```
pytest
ruff check .
ruff format --check .
```

The tests never download anything, so they run on small fixtures. CI runs the same checks on every push and pull request.
