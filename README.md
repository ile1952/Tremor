# Tremor Measurement System

This repository brings together the software components for collecting and
reviewing tremor measurements.

## Projects

| Folder | Purpose | Status |
| --- | --- | --- |
| [`data-viewer/`](./data-viewer/) | Python browser-based viewer for MATLAB waveform files | Available |
| [`esp32-firmware/`](./esp32-firmware/) | ESP32 firmware to collect sensor data and broadcast it over Bluetooth | Planned |
| [`maui-app/`](./maui-app/) | .NET MAUI app to receive and display the Bluetooth data on Mac or iPhone | Planned |
| [`docs/`](./docs/) | Shared system and Bluetooth protocol documentation | In progress |

The folders keep each component independent while allowing shared documentation
and protocol decisions to live alongside the code.

## Getting started

To run the current Python viewer, follow the instructions in
[`data-viewer/README.md`](./data-viewer/README.md). The viewer reads MATLAB
`.mat` files from `~/Downloads`; those data files are not included in this
repository.

## Planned Bluetooth integration

The ESP32 firmware and MAUI app will need to agree on a Bluetooth transport,
device identification, packet format, sample rate, units, and connection
behavior. Record the decisions in [`docs/bluetooth-protocol.md`](./docs/bluetooth-protocol.md)
before implementing the shared interface.

## Data and privacy

Do not commit patient recordings, credentials, device keys, or other sensitive
data. The root `.gitignore` excludes MATLAB files and common local development
artifacts.
