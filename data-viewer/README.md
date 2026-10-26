# MATLAB Waveform Viewer

A small Python program that displays numeric waveforms from MATLAB `.mat`
files in a browser. It looks for files in your macOS `~/Downloads` folder and
runs a local web server on your computer.

## Features

- Lists `.mat` files from `~/Downloads`; the data files are not included here.
- Detects numeric signals in MATLAB records and lets you select a signal or
  channel. For the `acc_signal_database.mat` dataset, the selector includes
  each subject's posture and rest recordings.
- Applies a one-cycle moving-average filter to reduce 50 Hz and its harmonics.
  The sample rate must be a multiple of 50 Hz.
- Plots three traces on one shared time axis:
  1. The filtered waveform.
  2. The point-by-point absolute value of the filtered waveform.
  3. A trigger trace that goes to 0.04 for 250 ms when the absolute value
     reaches 0.04. It rearms after the absolute value falls below 0.04.
- Lets you display 1, 5, or 10 seconds at a time and scroll through a recording.
- Opens a selected file in a separate browser tab. The compact controls remain
  at the top while scrolling.
- Provides an **End viewer** button to stop the local server.

## Requirements

- macOS (the default data folder is `~/Downloads`).
- Python 3.10 or newer.
- The packages listed in `requirements.txt` (NumPy, Matplotlib, and SciPy are
  used by the program; SciPy also reads the MAT files).

## Setup and run

Open Terminal in this project folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python test1.py
```

The program opens the viewer in your default browser. If no `.mat` files appear,
place one in `~/Downloads` and restart the program. Select a file, then use the
**Signal or channel** menu and display controls above the plot.

To stop the program, click **End viewer** in the browser or press Control+C in
the Terminal running the program.

## Data and network behavior

The viewer binds its web server to `127.0.0.1`, so it is available only from
the same computer. MAT files are read from `~/Downloads` and are not uploaded
by this program. The MAT data itself is not committed with the source.
