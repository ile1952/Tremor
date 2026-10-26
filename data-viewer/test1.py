"""Browse MATLAB waveform files in Downloads using a local web viewer."""

import base64
import html
import io
import math
import secrets
import sys
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

import numpy as np
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d


DOWNLOADS_DIR = Path.home() / "Downloads"
MAX_PLOT_POINTS = 10_000
FILTER_FREQUENCY_HZ = 50.0
TRIGGER_THRESHOLD = 0.04
TRIGGER_PULSE_SECONDS = 0.250
WINDOW_DURATIONS = (1, 5, 10)
DEFAULT_WINDOW_DURATION = 10
SHUTDOWN_TOKEN = secrets.token_urlsafe(32)


@dataclass(frozen=True)
class Waveform:
    label: str
    data_path: tuple[str | int | tuple[str, int], ...]
    sample_rate: float | None


def list_mat_files() -> list[Path]:
    """Return MATLAB files in the user's Downloads folder, sorted by name."""
    if not DOWNLOADS_DIR.is_dir():
        return []
    return sorted(
        (
            path
            for path in DOWNLOADS_DIR.iterdir()
            if path.is_file() and path.suffix.lower() == ".mat"
        ),
        key=lambda path: path.name.casefold(),
    )


def find_waveforms(mat_data: dict[str, object]) -> list[Waveform]:
    """Find selectable numeric waveforms and their paths in loaded MAT data."""
    waveforms: list[Waveform] = []

    def add_arrays(
        data: object,
        path: tuple[str | int | tuple[str, int], ...],
        label: str,
        sample_rate: float | None,
    ) -> None:
        array = np.asarray(data)
        if array.dtype.kind not in "iuf" or array.size < 2:
            return
        if array.ndim == 1:
            waveforms.append(Waveform(label, path, sample_rate))
        elif array.ndim == 2 and 1 in array.shape:
            waveforms.append(Waveform(label, path, sample_rate))
        elif array.ndim == 2 and array.shape[1] <= 16:
            for column in range(array.shape[1]):
                waveforms.append(
                    Waveform(
                        f"{label} · column {column + 1}",
                        path + (("index", column),),
                        sample_rate,
                    )
                )
        elif array.ndim == 2 and array.shape[0] <= 16:
            for row in range(array.shape[0]):
                waveforms.append(
                    Waveform(
                        f"{label} · row {row + 1}",
                        path + (("index", row),),
                        sample_rate,
                    )
                )

    for variable_name, value in mat_data.items():
        if variable_name.startswith("__"):
            continue
        if isinstance(value, list) and value and all(
            isinstance(record, dict) for record in value
        ):
            for record_index, record in enumerate(value):
                subject = record.get("subject", record_index + 1)
                hand = record.get("hand")
                prefix = f"Record {record_index + 1} · subject {subject}"
                if hand:
                    prefix += f" · {hand} hand"
                sample_rate_value = record.get("Fs")
                sample_rate = (
                    float(sample_rate_value)
                    if isinstance(sample_rate_value, (int, float, np.number))
                    and sample_rate_value > 0
                    else None
                )
                for field_name, field_value in record.items():
                    add_arrays(
                        field_value,
                        (variable_name, record_index, field_name),
                        f"{prefix} · {field_name}",
                        sample_rate,
                    )
        else:
            add_arrays(value, (variable_name,), variable_name, None)

    return waveforms


def resolve_waveform(mat_data: dict[str, object], waveform: Waveform) -> np.ndarray:
    """Resolve a selected waveform and apply the 50 Hz averaging filter."""
    value: object = mat_data
    for part in waveform.data_path:
        if isinstance(part, str):
            if not isinstance(value, dict):
                raise ValueError("The selected MATLAB variable has an unexpected format.")
            value = value[part]
        elif isinstance(part, int):
            if not isinstance(value, list):
                raise ValueError("The selected MATLAB record has an unexpected format.")
            value = value[part]
        elif isinstance(part, tuple) and part[0] == "index":
            value = np.asarray(value)[part[1]]
        else:
            raise ValueError("The selected MATLAB data has an unexpected index.")

    data = np.asarray(value).squeeze()
    if data.ndim != 1 or data.dtype.kind not in "iuf":
        raise ValueError("The selected MATLAB data is not a numeric waveform.")
    if waveform.sample_rate is None:
        raise ValueError("This signal does not include a usable sample rate.")
    return apply_50hz_averaging_filter(data, waveform.sample_rate)


def apply_50hz_averaging_filter(data: np.ndarray, sample_rate: float) -> np.ndarray:
    """Apply a one-cycle moving average when the sample rate is a 50 Hz multiple."""
    samples_per_cycle = round(sample_rate / FILTER_FREQUENCY_HZ)
    if samples_per_cycle < 1 or not math.isclose(
        sample_rate / samples_per_cycle,
        FILTER_FREQUENCY_HZ,
        rel_tol=1e-9,
    ):
        raise ValueError(
            "The sample rate must be a multiple of 50 Hz to apply the one-cycle "
            "averaging filter."
        )
    return uniform_filter1d(
        data,
        size=samples_per_cycle,
        mode="reflect",
    )


def select_window(
    data: np.ndarray,
    sample_rate: float,
    duration_seconds: int,
    start_seconds: float,
) -> tuple[np.ndarray, float, float, int]:
    """Select a bounded time window and return its offset and sample index."""
    total_duration = data.size / sample_rate
    max_start = max(0.0, total_duration - duration_seconds)
    actual_start = min(max(0.0, start_seconds), max_start)
    first_sample = int(actual_start * sample_rate)
    window_samples = max(1, int(duration_seconds * sample_rate))
    window = data[first_sample : min(data.size, first_sample + window_samples)]
    return window, max_start, actual_start, first_sample


def render_waveform_image(
    waveform: Waveform,
    data: np.ndarray,
    duration_seconds: int,
    start_seconds: float,
) -> tuple[str, int, float, float, int]:
    """Render filtered, absolute-value, and threshold-trigger traces as SVG."""
    if waveform.sample_rate is None or waveform.sample_rate <= 0:
        raise ValueError("This signal does not include a usable sample rate.")

    sample_rate = waveform.sample_rate
    window, max_start, actual_start, first_sample = select_window(
        data, sample_rate, duration_seconds, start_seconds
    )
    trigger_trace = calculate_trigger_trace(data, sample_rate)
    windowed_trigger, _, _, _ = select_window(
        trigger_trace,
        sample_rate,
        duration_seconds,
        start_seconds,
    )
    step = max(1, int(np.ceil(window.size / MAX_PLOT_POINTS)))
    plot_data = window[::step]
    absolute_data = np.abs(window)[::step]
    trigger_points = windowed_trigger[::step]
    window_start_time = first_sample / sample_rate
    window_end_time = (first_sample + window.size) / sample_rate
    x_values = window_start_time + np.arange(0, window.size, step) / sample_rate

    figure = Figure(figsize=(11, 5), tight_layout=True)
    axis = figure.subplots()
    axis.plot(
        x_values,
        plot_data,
        color="#4b5fc7",
        linewidth=0.8,
        label="Filtered waveform",
    )
    axis.plot(
        x_values,
        absolute_data,
        color="#d15b43",
        linewidth=1.2,
        label="Absolute value",
    )
    axis.plot(
        x_values,
        trigger_points,
        color="#348a60",
        linewidth=1.4,
        label="0.04 trigger pulse (250 ms)",
    )
    axis.set_title(f"Filtered waveform, absolute value, and trigger · {waveform.label}")
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Amplitude")
    axis.set_xlim(window_start_time, window_end_time)
    axis.grid(True, alpha=0.3)
    axis.legend()
    plot_buffer = io.BytesIO()
    FigureCanvasSVG(figure).print_svg(plot_buffer)
    encoded_image = base64.b64encode(plot_buffer.getvalue()).decode("ascii")
    return (
        encoded_image,
        plot_data.size,
        max_start,
        actual_start,
        window.size,
    )


def calculate_trigger_trace(data: np.ndarray, sample_rate: float) -> np.ndarray:
    """Create 250 ms pulses at threshold crossings, rearming below the threshold."""
    if sample_rate <= 0:
        raise ValueError("This signal does not include a usable sample rate.")
    pulse_samples = round(sample_rate * TRIGGER_PULSE_SECONDS)
    if pulse_samples < 1:
        raise ValueError("The sample rate is too low for a 250 ms trigger pulse.")

    absolute_data = np.abs(data)
    trigger_trace = np.zeros(data.size, dtype=np.float64)
    armed = True
    pulse_end = 0
    for index, value in enumerate(absolute_data):
        if index < pulse_end:
            continue
        if value < TRIGGER_THRESHOLD:
            armed = True
        elif armed:
            pulse_end = min(index + pulse_samples, data.size)
            trigger_trace[index:pulse_end] = TRIGGER_THRESHOLD
            armed = False
    return trigger_trace


def render_page(
    mat_files: list[Path],
    selected_file: Path | None = None,
    waveforms: list[Waveform] | None = None,
    selected_waveform: int | None = None,
    message: str = "",
    plot_image: str = "",
    plotted_points: int = 0,
    sample_count: int = 0,
    duration_seconds: int = DEFAULT_WINDOW_DURATION,
    start_seconds: float = 0.0,
    max_start_seconds: float = 0.0,
    total_duration_seconds: float = 0.0,
) -> bytes:
    """Build the file-selection or waveform-viewer HTML page."""
    file_cards = []
    for mat_file in mat_files:
        safe_name = html.escape(mat_file.name)
        file_cards.append(
            f'<a class="file-card" href="/choose?file={quote(mat_file.name)}#plot" target="_blank" rel="noopener">'
            f'<span class="mat-icon" aria-hidden="true">MAT</span>'
            f'<span class="file-name">{safe_name}</span>'
            f'<span class="file-size">{mat_file.stat().st_size / (1024 * 1024):.1f} MB</span>'
            "</a>"
        )

    waveform_section = ""
    if selected_file is not None and waveforms is not None:
        options = []
        for index, waveform in enumerate(waveforms):
            selected = " selected" if index == selected_waveform else ""
            options.append(
                f'<option value="{index}"{selected}>{html.escape(waveform.label)}</option>'
            )
        if options:
            duration_options = "".join(
                f'<option value="{duration}"'
                f'{" selected" if duration == duration_seconds else ""}>'
                f'{duration} second{"s" if duration != 1 else ""}</option>'
                for duration in WINDOW_DURATIONS
            )
            window_controls = (
                f"""
                <div class="control-field">
                  <label for="duration">Visible window</label>
                  <select id="duration" name="duration">{duration_options}</select>
                </div>
                <div class="control-field scroll-control">
                  <label for="start">Scroll through recording
                    <span id="position">{start_seconds:.1f} seconds</span>
                  </label>
                  <div class="scroll-inputs">
                    <input id="start" name="start" type="range" min="0"
                      max="{max_start_seconds:.1f}" step="0.1" value="{start_seconds:.1f}"
                      data-total-duration="{total_duration_seconds:.1f}">
                    <div class="scroll-buttons">
                      <button type="button" class="secondary" id="earlier">← Earlier</button>
                      <button type="button" class="secondary" id="later">Later →</button>
                    </div>
                  </div>
                </div>
                """
                if waveforms[selected_waveform or 0].sample_rate
                else '<p class="notice">This signal has no sample-rate metadata, so its duration cannot be shown in seconds.</p>'
            )
            waveform_section = f"""
            <section class="waveform-panel">
              <form class="display-form" action="/plot#plot" method="get">
                <input type="hidden" name="file" value="{html.escape(selected_file.name, quote=True)}">
                <div class="control-field signal-control">
                  <label for="waveform">Signal or channel</label>
                  <select id="waveform" name="waveform">{"".join(options)}</select>
                </div>
                {window_controls}
                <button type="submit">Update display</button>
              </form>
            </section>
            """
        else:
            waveform_section = (
                '<p class="notice">No numeric waveform arrays were found in this file.</p>'
            )

    plot_section = ""
    if plot_image and selected_waveform is not None and waveforms is not None:
        label = html.escape(waveforms[selected_waveform].label, quote=True)
        plot_section = f"""
        <section class="plot-panel" id="plot">
          <h2>Filtered waveform, absolute value, and trigger</h2>
          <img alt="Filtered waveform, its absolute value, and a 0.04 trigger pulse for {label}, shown as three traces on one graph with a shared time axis"
            src="data:image/svg+xml;base64,{plot_image}">
          <p>Showing {plotted_points:,} points from {sample_count:,} samples.
            All traces share one time and amplitude axis. The second trace is the point-by-point absolute value; the third goes to 0.04 for 250 ms when the absolute value crosses 0.04, then waits for the absolute value to fall below 0.04 before it can trigger again.</p>
        </section>
        """

    notice = f'<p class="notice" role="alert">{html.escape(message)}</p>' if message else ""
    file_browser = ""
    if selected_file is None:
        file_browser = f"""
        <section class="file-list" aria-label="MAT files">
          {"".join(file_cards)}
        </section>
        {('<p class="empty">No .mat files were found in your Downloads folder.</p>' if not mat_files else "")}
        """
    back_link = (
        f'<a class="back viewer-back" href="/">{html.escape("← Back to all MAT files")}</a>'
        if selected_file
        else ""
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{"Waveform viewer" if selected_file else "MAT waveform viewer"}</title>
  <style>
    :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; background: #f3f6fb; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 42px 20px; }}
    main {{ max-width: 900px; margin: 0 auto; }}
    .eyebrow {{ margin: 0 0 9px; color: #5369c5; font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }}
    h1 {{ margin: 0; font-size: clamp(1.8rem, 5vw, 2.5rem); letter-spacing: -.04em; }}
    .intro {{ margin: 10px 0 26px; color: #596579; line-height: 1.6; }}
    h2 {{ margin: 0 0 14px; font-size: 1.1rem; }}
    .file-list {{ display: grid; gap: 10px; }}
    .file-card {{ display: flex; align-items: center; gap: 14px; padding: 15px; border: 1px solid #e1e7f0; border-radius: 13px; background: white; color: inherit; text-decoration: none; box-shadow: 0 5px 18px #2334580a; }}
    .file-card:hover {{ border-color: #9ba8ed; background: #fbfbff; }}
    .mat-icon {{ display: grid; width: 46px; height: 46px; flex: 0 0 46px; place-items: center; border-radius: 11px; background: #edf0ff; color: #465dc0; font-size: .72rem; font-weight: 800; }}
    .file-name {{ flex: 1; overflow-wrap: anywhere; font-weight: 650; }}
    .file-size {{ color: #69758a; font-size: .88rem; white-space: nowrap; }}
    .waveform-panel, .plot-panel {{ margin-top: 24px; padding: 20px; border: 1px solid #e1e7f0; border-radius: 14px; background: white; }}
    .waveform-panel {{ position: sticky; top: 0; z-index: 2; margin-top: 12px; padding: 12px 16px; box-shadow: 0 6px 18px #23345812; }}
    form {{ display: flex; align-items: end; gap: 12px; flex-wrap: wrap; }}
    .display-form {{ display: grid; grid-template-columns: minmax(180px, 1fr) minmax(130px, auto) auto; gap: 8px 12px; align-items: end; }}
    .display-form > input[type="hidden"] {{ display: none; }}
    .control-field {{ min-width: 0; }}
    .control-field label {{ display: block; margin-bottom: 4px; color: #596579; font-size: .84rem; }}
    select {{ min-width: 0; width: 100%; padding: 8px 10px; border: 1px solid #ccd4e2; border-radius: 9px; background: white; color: inherit; }}
    button {{ padding: 11px 16px; border: 0; border-radius: 9px; background: #4b5fc7; color: white; font: inherit; font-weight: 650; cursor: pointer; }}
    button:hover {{ background: #394cae; }}
    .scroll-control {{ grid-column: 1 / -1; }}
    .scroll-inputs {{ display: flex; align-items: center; gap: 12px; }}
    input[type="range"] {{ min-width: 0; flex: 1; accent-color: #4b5fc7; }}
    .scroll-buttons {{ display: flex; gap: 8px; }}
    .scroll-buttons button, .display-form > button {{ padding: 8px 12px; }}
    button.secondary {{ border: 1px solid #ccd4e2; background: #f7f8fc; color: #35415a; }}
    button.secondary:hover {{ background: #e9ecf5; }}
    .end-form {{ margin-top: 30px; }}
    button.end-button {{ background: #8d3440; }}
    button.end-button:hover {{ background: #702934; }}
    .plot-panel img {{ display: block; width: 100%; height: auto; }}
    .plot-panel {{ border-left: 4px solid #4b5fc7; scroll-margin-top: 195px; }}
    .notice {{ padding: 12px 15px; border-radius: 10px; background: #fff0ed; color: #9b3024; }}
    .empty {{ padding: 20px; border: 1px dashed #bbc5d7; border-radius: 12px; color: #596579; text-align: center; }}
    .back {{ display: inline-block; margin-top: 18px; color: #465dc0; }}
    .viewer-back {{ margin: 10px 0 0; }}
    #position {{ margin-left: 8px; color: #465dc0; font-weight: 650; }}
    @media (max-width: 540px) {{
      body {{ padding: 20px 14px; }}
      .file-card {{ flex-wrap: wrap; }}
      .file-size {{ margin-left: 60px; }}
      .display-form {{ grid-template-columns: minmax(0, 1fr) auto; }}
      .signal-control {{ grid-column: 1 / -1; }}
      .scroll-inputs {{ flex-wrap: wrap; }}
      .scroll-buttons {{ width: 100%; }}
      .scroll-buttons button {{ flex: 1; }}
    }}
  </style>
</head>
<body>
  <main>
    <p class="eyebrow">Local MAT viewer</p>
    <h1>{"Waveform viewer" if selected_file else "Select a waveform file"}</h1>
    <p class="intro">{"A one-cycle moving average is applied to reduce 50 Hz and its harmonics." if selected_file else f'Choose a MATLAB file from <code>{html.escape(str(DOWNLOADS_DIR))}</code>, then select a signal to display. A one-cycle moving average is applied to reduce 50 Hz and its harmonics.'}</p>
    {notice}
    {back_link}
    {file_browser}
    {waveform_section}
    {plot_section}
    <form class="end-form" action="/shutdown" method="post">
      <input type="hidden" name="token" value="{SHUTDOWN_TOKEN}">
      <button type="submit" class="end-button">End viewer</button>
    </form>
  </main>
</body>
<script>
  const startSlider = document.getElementById("start");
  if (startSlider) {{
    const position = document.getElementById("position");
    const duration = document.getElementById("duration");
    const form = startSlider.form;
    const showPosition = () => {{ position.textContent = `${{Number(startSlider.value).toFixed(1)}} seconds`; }};
    const updateWindowLimit = () => {{
      startSlider.max = Math.max(0, Number(startSlider.dataset.totalDuration) - Number(duration.value)).toFixed(1);
      startSlider.value = Math.min(Number(startSlider.value), Number(startSlider.max));
      showPosition();
    }};
    startSlider.addEventListener("input", showPosition);
    duration.addEventListener("change", updateWindowLimit);
    const moveWindow = (direction) => {{
      const step = Number(duration.value);
      startSlider.value = Math.max(0, Math.min(Number(startSlider.max), Number(startSlider.value) + direction * step));
      showPosition();
      form.requestSubmit();
    }};
    document.getElementById("earlier").addEventListener("click", () => moveWindow(-1));
    document.getElementById("later").addEventListener("click", () => moveWindow(1));
  }}
</script>
</html>
"""
    return page.encode("utf-8")


class MatViewerRequestHandler(BaseHTTPRequestHandler):
    """Serve local MAT-file selection, waveform plots, and viewer shutdown."""

    def respond_html(self, status: int, content: bytes) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.log_message("Browser disconnected before the response completed")

    def selected_file(self) -> Path | None:
        """Resolve the requested filename only if it is in Downloads."""
        name = parse_qs(urlsplit(self.path).query).get("file", [""])[0]
        if not name or Path(name).name != name:
            return None
        return next((path for path in list_mat_files() if path.name == name), None)

    def do_GET(self) -> None:
        """Serve the file list, selected waveform, or updated plot."""
        request = urlsplit(self.path)
        if request.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        mat_files = list_mat_files()
        if request.path == "/":
            self.respond_html(200, render_page(mat_files))
            return

        if request.path not in {"/choose", "/plot"}:
            self.send_error(404, "Page not found")
            return

        selected_file = self.selected_file()
        if selected_file is None:
            self.respond_html(404, render_page(mat_files, message="That MAT file is not in Downloads."))
            return

        try:
            mat_data = loadmat(selected_file, simplify_cells=True)
            waveforms = find_waveforms(mat_data)
        except (OSError, ValueError, NotImplementedError) as error:
            self.respond_html(
                400,
                render_page(
                    mat_files,
                    selected_file,
                    message=f"Could not read this MAT file: {error}",
                ),
            )
            return

        if request.path == "/choose":
            if waveforms:
                waveform = waveforms[0]
                if waveform.sample_rate:
                    try:
                        data = resolve_waveform(mat_data, waveform)
                        duration_seconds = DEFAULT_WINDOW_DURATION
                        (
                            plot_image,
                            plotted_points,
                            max_start_seconds,
                            start_seconds,
                            sample_count,
                        ) = render_waveform_image(waveform, data, duration_seconds, 0.0)
                    except ValueError as error:
                        page = render_page(
                            mat_files,
                            selected_file,
                            waveforms,
                            message=f"Could not filter or display this signal: {error}",
                        )
                    else:
                        page = render_page(
                            mat_files,
                            selected_file,
                            waveforms,
                            0,
                            plot_image=plot_image,
                            plotted_points=plotted_points,
                            sample_count=sample_count,
                            duration_seconds=duration_seconds,
                            start_seconds=start_seconds,
                            max_start_seconds=max_start_seconds,
                            total_duration_seconds=data.size / waveform.sample_rate,
                        )
                else:
                    page = render_page(
                        mat_files,
                        selected_file,
                        waveforms,
                        message="The selected signal does not include a usable sample rate.",
                    )
            else:
                page = render_page(mat_files, selected_file, waveforms)
            self.respond_html(200, page)
            return

        try:
            waveform_index = int(parse_qs(request.query).get("waveform", [""])[0])
            duration_seconds = int(
                parse_qs(request.query).get(
                    "duration",
                    [str(DEFAULT_WINDOW_DURATION)],
                )[0]
            )
            start_seconds = float(parse_qs(request.query).get("start", ["0"])[0])
            if duration_seconds not in WINDOW_DURATIONS:
                raise ValueError("Choose a window of 1, 5, or 10 seconds.")
            if not math.isfinite(start_seconds):
                raise ValueError("The scroll position must be a finite number.")
            waveform = waveforms[waveform_index]
            data = resolve_waveform(mat_data, waveform)
        except (ValueError, IndexError, KeyError) as error:
            self.respond_html(
                400,
                render_page(
                    mat_files,
                    selected_file,
                    waveforms,
                    message=f"Could not display that waveform: {error}",
                ),
            )
            return

        if waveform.sample_rate is None:
            self.respond_html(
                400,
                render_page(
                    mat_files,
                    selected_file,
                    waveforms,
                    waveform_index,
                    message="This signal does not include a usable sample rate.",
                ),
            )
            return

        (
            plot_image,
            plotted_points,
            max_start_seconds,
            start_seconds,
            sample_count,
        ) = render_waveform_image(waveform, data, duration_seconds, start_seconds)
        page = render_page(
            mat_files,
            selected_file,
            waveforms,
            waveform_index,
            plot_image=plot_image,
            plotted_points=plotted_points,
            sample_count=sample_count,
            duration_seconds=duration_seconds,
            start_seconds=start_seconds,
            max_start_seconds=max_start_seconds,
            total_duration_seconds=data.size / waveform.sample_rate,
        )
        self.respond_html(200, page)

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")

    def do_POST(self) -> None:
        """Stop the local server only for an authorized shutdown request."""
        if urlsplit(self.path).path != "/shutdown":
            self.send_error(404, "Page not found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            form = parse_qs(self.rfile.read(content_length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self.send_error(400, "Invalid shutdown request")
            return

        token = form.get("token", [""])[0]
        if not secrets.compare_digest(token, SHUTDOWN_TOKEN):
            self.send_error(403, "Shutdown request not authorized")
            return

        page = (
            b"<!doctype html><html lang='en'><meta charset='utf-8'>"
            b"<title>Viewer stopped</title><body><h1>Viewer stopped</h1>"
            b"<p>You can close this browser tab.</p></body></html>"
        )
        self.respond_html(200, page)
        threading.Thread(target=self.server.shutdown, daemon=True).start()


def main() -> int:
    """Start the loopback-only server and open the viewer in a browser."""
    mat_files = list_mat_files()
    print(f"Looking for .mat files in: {DOWNLOADS_DIR}")
    if not mat_files:
        print("No .mat files found. Add a MATLAB file to Downloads and restart.")

    with ThreadingHTTPServer(("127.0.0.1", 0), MatViewerRequestHandler) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Opening the local waveform viewer: {url}")
        print("Use the End viewer button in the browser to stop the viewer.")
        webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nWaveform viewer stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
