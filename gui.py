import base64
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import perf_counter
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from laser_engine import FrameResult, LaserShooterEngine, ShooterConfig, Shot
from video_source import (
    discover_camera_ids,
    open_first_available_camera,
    open_video_capture,
)


APP_TITLE = "Laser Shooter"
DISPLAY_TITLE = "Laser Shooter - Target Display"
PREVIEW_SIZE = (620, 540)


class FrameWorker:
    """Own the camera and image processing away from Tk's event thread."""

    def __init__(
        self,
        source: int | str | None,
        config: ShooterConfig,
        player_name: str,
    ):
        self.source = source
        self.config = config
        self.player_name = player_name
        self.preview_frames: Queue[object] = Queue(maxsize=1)
        self.frames: Queue[FrameResult] = Queue(maxsize=1)
        self.events: Queue[tuple[str, object]] = Queue()
        self.stop_event = Event()
        self.reset_event = Event()
        self.threshold_lock = Lock()
        self.threshold = config.white_threshold
        self.thread = Thread(target=self._run, name="laser-camera", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def request_reset(self) -> None:
        self.reset_event.set()
        self._discard_pending_frame()

    def update_threshold(self, threshold: int) -> None:
        with self.threshold_lock:
            self.threshold = threshold

    def latest_frame(self) -> FrameResult | None:
        latest = None
        while True:
            try:
                latest = self.frames.get_nowait()
            except Empty:
                return latest

    def latest_preview(self) -> object | None:
        latest = None
        while True:
            try:
                latest = self.preview_frames.get_nowait()
            except Empty:
                return latest

    def _run(self) -> None:
        capture: cv2.VideoCapture | None = None
        reason = "stopped"
        detail = "Stopped."

        try:
            opened_source = self.source
            if self.source is None:
                capture, opened_source = open_first_available_camera()
            else:
                capture = open_video_capture(self.source)
            engine = LaserShooterEngine(self.config)
            self.events.put(("started", opened_source))
            fps_started_at = perf_counter()
            fps_frame_count = 0

            while not self.stop_event.is_set():
                if self.reset_event.is_set():
                    engine.reset()
                    self.reset_event.clear()

                with self.threshold_lock:
                    threshold = self.threshold
                engine.update_threshold(threshold)

                success, frame = capture.read()
                if not success:
                    if not self.stop_event.is_set():
                        reason = "ended"
                        detail = (
                            "The webcam stopped returning frames."
                            if isinstance(opened_source, int)
                            else "Video playback finished."
                        )
                    break

                self._publish_latest_preview(frame)
                result = engine.process_frame(frame, self.player_name)
                fps_frame_count += 1
                fps_elapsed = perf_counter() - fps_started_at
                if fps_elapsed >= 0.5:
                    self.events.put(("fps", fps_frame_count / fps_elapsed))
                    fps_started_at = perf_counter()
                    fps_frame_count = 0
                for shot in result.new_shots:
                    self.events.put(("shot", shot))
                self._publish_latest(result)
        except Exception as exc:
            reason = "error"
            detail = str(exc)
        finally:
            if capture is not None:
                capture.release()
            self.events.put(("finished", (reason, detail)))

    def _publish_latest(self, result: FrameResult) -> None:
        try:
            self.frames.put_nowait(result)
        except Full:
            self._discard_pending_frame()
            try:
                self.frames.put_nowait(result)
            except Full:
                pass

    def _publish_latest_preview(self, frame: object) -> None:
        try:
            self.preview_frames.put_nowait(frame)
        except Full:
            try:
                self.preview_frames.get_nowait()
            except Empty:
                pass
            try:
                self.preview_frames.put_nowait(frame)
            except Full:
                pass

    def _discard_pending_frame(self) -> None:
        try:
            self.frames.get_nowait()
        except Empty:
            pass


class LaserShooterApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1400x860")
        self.root.minsize(1050, 700)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.option_add("*Font", ("Segoe UI", 10))

        self.worker: FrameWorker | None = None
        self.poll_after_id: str | None = None
        self.camera_scan_results: Queue[tuple[list[int], str | None]] = Queue(maxsize=1)
        self.camera_scan_in_progress = False
        self.running = False
        self.player_name = "Player"

        self.player_var = tk.StringVar(value="Player")
        self.source_type_var = tk.StringVar(value="Webcam")
        self.camera_id_var = tk.StringVar(value="Auto")
        self.video_path_var = tk.StringVar()
        self.threshold_var = tk.IntVar(value=160)
        self.target_width_var = tk.StringVar(value="595")
        self.target_height_var = tk.StringVar(value="842")
        self.inner_radius_var = tk.StringVar(value="28")
        self.outer_radius_var = tk.StringVar(value="293")
        self.max_shots_var = tk.StringVar(value="10")
        self.total_score_var = tk.StringVar(value="0")
        self.shot_count_var = tk.StringVar(value="0 / 10")
        self.remaining_shots_var = tk.StringVar(value="10")
        self.fps_var = tk.StringVar(value="FPS: --")
        self.status_var = tk.StringVar(value="Ready. Configure the camera and press Start.")
        self.display_has_target = False
        self.latest_display_target: object | None = None
        self.display_resize_after_id: str | None = None
        self.display_shots: list[Shot] = []
        self.active_config: ShooterConfig | None = None

        self._build_style()
        self._build_ui()
        self._build_display_window()
        self._update_source_controls()
        self.threshold_var.trace_add("write", self._on_threshold_changed)
        self.root.after(200, self._show_display_window)
        self.poll_after_id = self.root.after(30, self._poll_background_work)

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        available_themes = style.theme_names()
        if "vista" in available_themes:
            style.theme_use("vista")
        style.configure("Status.TLabel", padding=(10, 7))
        style.configure("Score.TLabel", font=("Segoe UI", 22, "bold"))
        style.configure("Heading.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure(
            "Fps.TLabel", font=("Consolas", 12, "bold"), foreground="#2563eb"
        )

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(self.root, text="Session and camera")
        controls.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        controls.columnconfigure(9, weight=1)

        ttk.Label(controls, text="Player").grid(row=0, column=0, padx=(10, 4), pady=8)
        self.player_entry = ttk.Entry(controls, textvariable=self.player_var, width=15)
        self.player_entry.grid(row=0, column=1, padx=(0, 12), pady=8)

        ttk.Label(controls, text="Source").grid(row=0, column=2, padx=(0, 4), pady=8)
        self.source_combo = ttk.Combobox(
            controls,
            textvariable=self.source_type_var,
            values=("Webcam", "Video file"),
            state="readonly",
            width=12,
        )
        self.source_combo.grid(row=0, column=3, padx=(0, 12), pady=8)
        self.source_combo.bind("<<ComboboxSelected>>", self._update_source_controls)

        ttk.Label(controls, text="Camera ID").grid(row=0, column=4, padx=(0, 4), pady=8)
        self.camera_combo = ttk.Combobox(
            controls,
            textvariable=self.camera_id_var,
            values=("Auto", "0"),
            width=6,
            state="normal",
        )
        self.camera_combo.grid(row=0, column=5, padx=(0, 4), pady=8)
        self.detect_button = ttk.Button(
            controls, text="Detect", command=self._scan_cameras
        )
        self.detect_button.grid(row=0, column=6, padx=(0, 12), pady=8)

        self.video_entry = ttk.Entry(
            controls, textvariable=self.video_path_var, width=28
        )
        self.video_entry.grid(row=0, column=7, padx=(0, 4), pady=8)
        self.browse_button = ttk.Button(
            controls, text="Browse...", command=self._browse_video
        )
        self.browse_button.grid(row=0, column=8, padx=(0, 12), pady=8)

        ttk.Label(
            controls, textvariable=self.fps_var, style="Fps.TLabel", width=10
        ).grid(row=0, column=9, padx=(0, 12), pady=8)

        self.start_button = ttk.Button(controls, text="Start", command=self.start)
        self.start_button.grid(row=0, column=10, padx=4, pady=8)
        self.stop_button = ttk.Button(
            controls, text="Stop", command=self.stop, state="disabled"
        )
        self.stop_button.grid(row=0, column=11, padx=4, pady=8)
        self.reset_button = ttk.Button(controls, text="Reset round", command=self.reset_round)
        self.reset_button.grid(row=0, column=12, padx=4, pady=8)
        self.display_button = ttk.Button(
            controls, text="Show display", command=self._show_display_window
        )
        self.display_button.grid(row=0, column=13, padx=(4, 10), pady=8)

        ttk.Label(controls, text="Laser threshold").grid(
            row=1, column=0, padx=(10, 4), pady=(0, 10)
        )
        self.threshold_scale = tk.Scale(
            controls,
            from_=0,
            to=255,
            orient="horizontal",
            variable=self.threshold_var,
            showvalue=True,
            resolution=1,
            length=220,
            highlightthickness=0,
        )
        self.threshold_scale.grid(
            row=1, column=1, columnspan=3, sticky="w", padx=(0, 12), pady=(0, 6)
        )

        settings = (
            ("Target W", self.target_width_var),
            ("Target H", self.target_height_var),
            ("Inner radius", self.inner_radius_var),
            ("Outer radius", self.outer_radius_var),
            ("Max shots", self.max_shots_var),
        )
        self.config_entries: list[ttk.Entry] = []
        column = 4
        for label, variable in settings:
            ttk.Label(controls, text=label).grid(
                row=1, column=column, padx=(0, 4), pady=(0, 10)
            )
            entry = ttk.Entry(controls, textvariable=variable, width=7)
            entry.grid(
                row=1, column=column + 1, padx=(0, 12), pady=(0, 10)
            )
            self.config_entries.append(entry)
            column += 2

        content = ttk.Panedwindow(self.root, orient="horizontal")
        content.grid(row=1, column=0, sticky="nsew", padx=10, pady=6)

        viewer = ttk.Frame(content)
        score_panel = ttk.Frame(content, width=280)
        content.add(viewer, weight=5)
        content.add(score_panel, weight=1)

        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(0, weight=1)
        self.notebook = ttk.Notebook(viewer)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        live_tab = ttk.Frame(self.notebook)
        calibration_tab = ttk.Frame(self.notebook)
        self.notebook.add(live_tab, text="Live target")
        self.notebook.add(calibration_tab, text="Threshold calibration")

        live_tab.columnconfigure(0, weight=1)
        live_tab.columnconfigure(1, weight=1)
        live_tab.rowconfigure(1, weight=1)
        ttk.Label(live_tab, text="Camera and marker detection", style="Heading.TLabel").grid(
            row=0, column=0, pady=(10, 5)
        )
        ttk.Label(live_tab, text="Rectified target", style="Heading.TLabel").grid(
            row=0, column=1, pady=(10, 5)
        )
        self.preview_label = self._make_image_panel(
            live_tab, "Camera preview appears after Start"
        )
        self.preview_label.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(0, 8))
        self.target_label = self._make_image_panel(
            live_tab, "The target appears after all 4 markers are detected"
        )
        self.target_label.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=(0, 8))

        calibration_tab.columnconfigure(0, weight=1)
        calibration_tab.rowconfigure(1, weight=1)
        ttk.Label(
            calibration_tab,
            text="Move the threshold slider until only the laser dot remains white",
            style="Heading.TLabel",
        ).grid(row=0, column=0, pady=10)
        self.mask_label = self._make_image_panel(
            calibration_tab, "Threshold preview appears after Start"
        )
        self.mask_label.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        score_panel.columnconfigure(0, weight=1)
        score_panel.rowconfigure(4, weight=1)
        ttk.Label(score_panel, text="Score", style="Heading.TLabel").grid(
            row=0, column=0, pady=(4, 2)
        )
        ttk.Label(
            score_panel, textvariable=self.total_score_var, style="Score.TLabel"
        ).grid(row=1, column=0, pady=(0, 2))
        ttk.Label(score_panel, textvariable=self.shot_count_var).grid(
            row=2, column=0, pady=(0, 10)
        )

        columns = ("number", "score", "position")
        self.score_tree = ttk.Treeview(
            score_panel, columns=columns, show="headings", height=20
        )
        self.score_tree.heading("number", text="#")
        self.score_tree.heading("score", text="Score")
        self.score_tree.heading("position", text="Position")
        self.score_tree.column("number", width=40, anchor="center", stretch=False)
        self.score_tree.column("score", width=65, anchor="center", stretch=False)
        self.score_tree.column("position", width=125, anchor="center")
        self.score_tree.grid(row=4, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            score_panel, orient="vertical", command=self.score_tree.yview
        )
        scrollbar.grid(row=4, column=1, sticky="ns")
        self.score_tree.configure(yscrollcommand=scrollbar.set)

        status = ttk.Label(
            self.root,
            textvariable=self.status_var,
            style="Status.TLabel",
            relief="sunken",
            anchor="w",
        )
        status.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

    def _build_display_window(self) -> None:
        self.display_window = tk.Toplevel(self.root)
        self.display_window.title(DISPLAY_TITLE)
        self.display_window.geometry("1280x800+80+80")
        self.display_window.minsize(800, 500)
        self.display_window.configure(bg="#060b16")
        self.display_window.protocol("WM_DELETE_WINDOW", self._hide_display_window)
        self.display_window.columnconfigure(0, weight=1)
        self.display_window.columnconfigure(1, minsize=300)
        self.display_window.rowconfigure(0, weight=1)

        target_panel = tk.Frame(self.display_window, bg="#111827")
        target_panel.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        target_panel.columnconfigure(0, weight=1)
        target_panel.rowconfigure(0, weight=1)
        self.display_target_label = self._make_image_panel(
            target_panel,
            "과녁을 비추면\n피격 상태가 여기에 표시됩니다",
        )
        self.display_target_label.configure(font=("Segoe UI", 20, "bold"))
        self.display_target_label.grid(row=0, column=0, sticky="nsew")
        self.display_target_label.bind("<Configure>", self._on_display_target_resized)

        stats_panel = tk.Frame(self.display_window, bg="#0b1220", width=300)
        stats_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        stats_panel.grid_propagate(False)
        stats_panel.columnconfigure(0, weight=1)
        stats_panel.rowconfigure(0, weight=1)
        stats_panel.rowconfigure(1, weight=1)

        self._build_large_stat(
            stats_panel,
            row=0,
            title="남은 총알",
            variable=self.remaining_shots_var,
            color="#38bdf8",
        )
        self._build_large_stat(
            stats_panel,
            row=1,
            title="점수",
            variable=self.total_score_var,
            color="#facc15",
        )

    @staticmethod
    def _build_large_stat(
        parent: tk.Misc,
        row: int,
        title: str,
        variable: tk.StringVar,
        color: str,
    ) -> None:
        panel = tk.Frame(parent, bg="#0b1220")
        panel.grid(row=row, column=0, sticky="nsew", padx=12, pady=12)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)
        panel.rowconfigure(1, weight=2)
        tk.Label(
            panel,
            text=title,
            bg="#0b1220",
            fg="#cbd5e1",
            font=("Segoe UI", 24, "bold"),
        ).grid(row=0, column=0, sticky="s", pady=(10, 0))
        tk.Label(
            panel,
            textvariable=variable,
            bg="#0b1220",
            fg=color,
            font=("Segoe UI", 76, "bold"),
        ).grid(row=1, column=0, sticky="n", pady=(0, 10))

    def _show_display_window(self) -> None:
        self.display_window.deiconify()
        self.display_window.lift()

    def _hide_display_window(self) -> None:
        self.display_window.withdraw()

    def _on_display_target_resized(self, _event: object) -> None:
        if self.latest_display_target is None:
            return
        if self.display_resize_after_id is not None:
            self.root.after_cancel(self.display_resize_after_id)
        self.display_resize_after_id = self.root.after(
            80, self._redraw_display_target
        )

    def _redraw_display_target(self) -> None:
        self.display_resize_after_id = None
        if self.latest_display_target is None:
            return
        display_width = max(100, self.display_target_label.winfo_width() - 20)
        display_height = max(100, self.display_target_label.winfo_height() - 20)
        self._show_image(
            self.display_target_label,
            self.latest_display_target,
            (display_width, display_height),
        )

    @staticmethod
    def _render_clean_target(
        config: ShooterConfig, shots: list[Shot]
    ) -> np.ndarray:
        """Render the recognized scoring area without camera pixels or markers."""
        side = min(config.target_width, config.target_height)
        left = (config.target_width - side) // 2
        top = (config.target_height - side) // 2
        center = (config.target_width // 2 - left, config.target_height // 2 - top)
        image = np.full((side, side, 3), 255, dtype=np.uint8)

        for radius in config.ring_radii:
            cv2.circle(image, center, radius, (20, 20, 20), 3, cv2.LINE_AA)

        for index, shot in enumerate(shots):
            point = (shot.x - left, shot.y - top)
            if not 0 <= point[0] < side or not 0 <= point[1] < side:
                continue
            is_latest = index == len(shots) - 1
            color = (0, 200, 0) if is_latest else (0, 0, 230)
            cv2.circle(image, point, 9 if is_latest else 7, (20, 20, 20), -1)
            cv2.circle(image, point, 7 if is_latest else 5, color, -1)

        return image

    @staticmethod
    def _make_image_panel(parent: tk.Misc, text: str) -> tk.Label:
        return tk.Label(
            parent,
            text=text,
            bg="#111827",
            fg="#e5e7eb",
            justify="center",
            anchor="center",
        )

    def _update_source_controls(self, _event: object | None = None) -> None:
        webcam_selected = self.source_type_var.get() == "Webcam"
        camera_state = "normal" if webcam_selected and not self.running else "disabled"
        self.camera_combo.configure(state=camera_state)
        detect_state = (
            "normal"
            if webcam_selected and not self.running and not self.camera_scan_in_progress
            else "disabled"
        )
        self.detect_button.configure(state=detect_state)
        video_state = "normal" if not webcam_selected and not self.running else "disabled"
        self.video_entry.configure(state=video_state)
        self.browse_button.configure(state=video_state)

    def _scan_cameras(self) -> None:
        if self.camera_scan_in_progress or self.running:
            return

        self.camera_scan_in_progress = True
        self.detect_button.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.status_var.set("Detecting available camera IDs in the background...")

        def scan() -> None:
            try:
                camera_ids = discover_camera_ids()
                result = (camera_ids, None)
            except Exception as exc:
                result = ([], str(exc))

            try:
                self.camera_scan_results.put_nowait(result)
            except Full:
                pass

        Thread(target=scan, name="camera-discovery", daemon=True).start()

    def _apply_camera_scan_result(
        self, camera_ids: list[int], error: str | None
    ) -> None:
        self.camera_scan_in_progress = False
        values = ("Auto", *(str(camera_id) for camera_id in camera_ids))
        self.camera_combo.configure(values=values)

        if error:
            self.status_var.set(f"Camera detection failed: {error}")
        elif camera_ids:
            current_id = self.camera_id_var.get()
            if current_id not in values:
                self.camera_id_var.set("Auto")
            detected = ", ".join(values[1:])
            self.status_var.set(f"Detected camera IDs: {detected}")
        else:
            self.status_var.set(
                "No camera was detected automatically. You can still enter an ID manually."
            )

        if not self.running:
            self.start_button.configure(state="normal")
        self._update_source_controls()

    def _browse_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a test video",
            filetypes=(
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*"),
            ),
        )
        if path:
            self.video_path_var.set(path)

    def _read_config(self) -> ShooterConfig:
        try:
            config = ShooterConfig(
                target_width=int(self.target_width_var.get()),
                target_height=int(self.target_height_var.get()),
                inner_radius=int(self.inner_radius_var.get()),
                outer_radius=int(self.outer_radius_var.get()),
                white_threshold=int(self.threshold_var.get()),
                max_shots=int(self.max_shots_var.get()),
            )
        except ValueError as exc:
            raise ValueError("All target settings must be whole numbers.") from exc
        config.validate()
        return config

    def _read_source(self) -> int | str | None:
        if self.source_type_var.get() == "Webcam":
            camera_id = self.camera_id_var.get().strip()
            if camera_id.casefold() == "auto":
                return None
            try:
                source = int(camera_id)
            except ValueError as exc:
                raise ValueError("Camera ID must be Auto or a whole number.") from exc
            return source

        path = Path(self.video_path_var.get().strip())
        if not path.is_file():
            raise ValueError("Select an existing video file.")
        return str(path)

    def start(self) -> None:
        if self.running:
            return

        try:
            config = self._read_config()
            source = self._read_source()
        except ValueError as exc:
            messagebox.showerror(APP_TITLE, str(exc), parent=self.root)
            self.status_var.set(f"Could not start: {exc}")
            return

        self.player_name = self.player_var.get().strip() or "Player"
        self.active_config = config
        self.worker = FrameWorker(source, config, self.player_name)
        self.running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._set_settings_state("disabled")
        self._clear_scoreboard()
        self._reset_display_target()
        self.fps_var.set("FPS: --")
        self.status_var.set("Opening the camera in the background...")
        self.worker.start()

    def stop(self) -> None:
        if self.worker is None:
            return
        self.stop_button.configure(state="disabled")
        self.status_var.set("Stopping the camera...")
        self.worker.stop()

    def _set_settings_state(self, state: str) -> None:
        self.player_entry.configure(state=state)
        self.source_combo.configure(state="readonly" if state == "normal" else "disabled")
        self.camera_combo.configure(state="disabled" if state == "disabled" else "normal")
        self.detect_button.configure(state="disabled")
        self.video_entry.configure(state="disabled")
        self.browse_button.configure(state="disabled")
        for entry in self.config_entries:
            entry.configure(state=state)

    def _poll_background_work(self) -> None:
        self.poll_after_id = None

        try:
            camera_ids, error = self.camera_scan_results.get_nowait()
        except Empty:
            pass
        else:
            self._apply_camera_scan_result(camera_ids, error)

        worker = self.worker
        if worker is not None:
            while True:
                try:
                    event, payload = worker.events.get_nowait()
                except Empty:
                    break
                self._handle_worker_event(worker, event, payload)

            if self.worker is worker:
                preview = worker.latest_preview()
                result = worker.latest_frame()
                if result is not None:
                    self._render_result(result)
                elif preview is not None and self.notebook.index(
                    self.notebook.select()
                ) == 0:
                    self._show_image(self.preview_label, preview, PREVIEW_SIZE)

        self.poll_after_id = self.root.after(33, self._poll_background_work)

    def _handle_worker_event(
        self, worker: FrameWorker, event: str, payload: object
    ) -> None:
        if worker is not self.worker:
            return

        if event == "started":
            if isinstance(payload, int):
                self.status_var.set(
                    f"Camera {payload} started. Aim it at all four target markers."
                )
            else:
                self.status_var.set("Video started. Aim it at all four target markers.")
            return

        if event == "shot" and isinstance(payload, Shot):
            self.display_shots.append(payload)
            self.score_tree.insert(
                "",
                "end",
                values=(payload.number, payload.score, f"({payload.x}, {payload.y})"),
            )
            return

        if event == "fps" and isinstance(payload, (int, float)):
            self.fps_var.set(f"FPS: {payload:.1f}")
            return

        if event != "finished":
            return

        reason, detail = payload
        self.worker = None
        self.running = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self._set_settings_state("normal")
        self._update_source_controls()
        self.fps_var.set("FPS: --")
        self.status_var.set(str(detail))

        if reason == "error":
            messagebox.showerror(APP_TITLE, str(detail), parent=self.root)

    def _render_result(self, result: FrameResult) -> None:
        selected_tab = self.notebook.index(self.notebook.select())
        if selected_tab == 0:
            self._show_image(self.preview_label, result.preview, PREVIEW_SIZE)
            if result.target is None:
                self._clear_image(
                    self.target_label,
                    f"Aim at the full target\nMarkers detected: {result.marker_count}/4",
                )
            else:
                self._show_image(self.target_label, result.target, PREVIEW_SIZE)
        else:
            self._show_image(self.mask_label, result.threshold_mask, PREVIEW_SIZE)

        self.total_score_var.set(str(result.total_score))
        self.shot_count_var.set(
            f"{result.shot_count} / {result.max_shots} shots"
        )
        self.remaining_shots_var.set(
            str(max(0, result.max_shots - result.shot_count))
        )

        if result.target is not None and self.active_config is not None:
            self.latest_display_target = self._render_clean_target(
                self.active_config, self.display_shots
            )
            self._redraw_display_target()
            self.display_has_target = True
        elif not self.display_has_target:
            self._clear_image(
                self.display_target_label,
                f"과녁 마커를 찾는 중\n{result.marker_count}/4",
            )

        if result.round_complete:
            self.status_var.set(
                f"Round complete — total score: {result.total_score}. "
                "Press Reset round to continue."
            )
        elif result.marker_count == 4:
            self.status_var.set("Target locked. Ready for laser shots.")
        else:
            self.status_var.set(f"Searching for target markers: {result.marker_count}/4")

    def reset_round(self) -> None:
        if self.worker is not None:
            self.worker.request_reset()
        self._clear_scoreboard()
        self._reset_display_target()
        self.status_var.set("Round reset. Ready.")

    def _clear_scoreboard(self) -> None:
        for item in self.score_tree.get_children():
            self.score_tree.delete(item)
        self.total_score_var.set("0")
        try:
            maximum = int(self.max_shots_var.get())
        except ValueError:
            maximum = 10
        self.shot_count_var.set(f"0 / {maximum} shots")
        self.remaining_shots_var.set(str(maximum))

    def _reset_display_target(self) -> None:
        self.display_has_target = False
        self.display_shots.clear()
        self.latest_display_target = None
        if self.display_resize_after_id is not None:
            self.root.after_cancel(self.display_resize_after_id)
            self.display_resize_after_id = None
        self._clear_image(
            self.display_target_label,
            "과녁을 비추면\n피격 상태가 여기에 표시됩니다",
        )

    def _on_threshold_changed(self, *_args: object) -> None:
        if self.worker is not None:
            self.worker.update_threshold(self.threshold_var.get())

    @staticmethod
    def _show_image(label: tk.Label, frame: object, size: tuple[int, int]) -> None:
        image = frame
        height, width = image.shape[:2]
        scale = min(size[0] / width, size[1] / height)
        if scale != 1:
            image = cv2.resize(
                image,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
            )

        success, encoded = cv2.imencode(".png", image)
        if not success:
            return
        photo = tk.PhotoImage(data=base64.b64encode(encoded).decode("ascii"))
        label.configure(image=photo, text="")
        label.image = photo

    @staticmethod
    def _clear_image(label: tk.Label, text: str) -> None:
        label.configure(image="", text=text)
        label.image = None

    def close(self) -> None:
        if self.poll_after_id is not None:
            try:
                self.root.after_cancel(self.poll_after_id)
            except tk.TclError:
                pass
            self.poll_after_id = None
        if self.worker is not None:
            self.worker.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    LaserShooterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
