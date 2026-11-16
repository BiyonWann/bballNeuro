# bballNeuro

Can a competitive learning network find the NBA 3-point line using only where shots were taken? This project trains one on shot locations and never tells it where any line is.

It then trains the same network season by season, from 2000-01 to 2024-25, to see how its map of the court shifts as the league moved toward the 3. The full plan is in `docs/proposal.md`.

## Set up locally

You need Python 3.11 or newer.

```
git clone https://github.com/BiyonWann/bballNeuro.git
cd bballNeuro
python -m venv .venv
```

Activate the environment. On macOS or Linux:

```
source .venv/bin/activate
```

On Windows:

```
.venv\Scripts\activate
```

Then install the packages:

```
pip install -r requirements.txt -r requirements-dev.txt
pip install notebook
```

`requirements.txt` holds numpy, pandas, matplotlib and scikit-learn. `requirements-dev.txt` adds pytest, ruff and the tools that run a notebook from the command line. The `notebook` package is only needed if you want to open the notebook in Jupyter.

## Run the notebook

To work in the notebook, start Jupyter and open `notebook.ipynb`:

```
jupyter notebook
```

To run it from top to bottom without opening it, write the executed copy to `executed.ipynb`:

```
jupyter nbconvert --to notebook --execute notebook.ipynb --output executed.ipynb
```

The first run downloads the data, so it needs an internet connection and takes longer.

## Where the data comes from

The shot data comes from the public repository `shufinskiy/nba_data`, which mirrors the NBA's shot-chart data. The project uses four seasons, named by the year they start: `shotdetail_2000` is 2000-01, then 2008, 2015 and 2024.

You do not download anything by hand. The first time a season is needed, the code downloads its archive, extracts `shotdetail_<year>.csv` into the `data/` folder and deletes the archive. After that it reads the saved file and does not touch the network. To fetch a season again, delete its CSV from `data/`. The `data/` folder is not committed to git.

If you already have the season files in another folder, set the optional `BBALL_DATA_DIR` environment variable to that folder before you start the notebook. The notebook reads it and uses `data/` when it is not set.

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

1. The Colab setup cell clones this repository, so the repository must be public. If it is private, the clone fails. In that case upload the project files to the Colab session instead. Zip the folder, open the Files panel in Colab, upload the zip and unzip it there. Name the unzipped folder something other than `bballNeuro`, because the setup cell would try to create that name. Then, in a new cell at the top, run `%cd your_folder_name` and `!pip install -r requirements.txt`, and skip the original setup cell when you run the rest.
2. Open Colab at https://colab.research.google.com and choose File, then Open notebook.
3. Pick the GitHub tab and enter `BiyonWann/bballNeuro`, or upload `notebook.ipynb` directly.
4. With a public repository, choose Runtime, then Run all. The first cell clones the repository and installs `requirements.txt`. Locally that cell does nothing. With uploaded files, run the cells one by one from the step 1 cell onward instead, skipping the original setup cell.
5. The data downloads into the Colab session. It is lost when the runtime resets, so the next session downloads it again.

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

The tests never use the network. They run on small fixtures, and the same checks run on every push and pull request.
