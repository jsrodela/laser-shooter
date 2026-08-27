import base64
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

from laser_engine import FrameResult, LaserShooterEngine, ShooterConfig
from video_source import open_video_capture


APP_TITLE = "Laser Shooter"
PREVIEW_SIZE = (620, 540)


class LaserShooterApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1400x860")
        self.root.minsize(1050, 700)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.option_add("*Font", ("Segoe UI", 10))

        self.capture: cv2.VideoCapture | None = None
        self.engine: LaserShooterEngine | None = None
        self.after_id: str | None = None
        self.running = False
        self.source_is_webcam = True
        self.player_name = "Player"

        self.player_var = tk.StringVar(value="Player")
        self.source_type_var = tk.StringVar(value="Webcam")
        self.camera_id_var = tk.StringVar(value="0")
        self.video_path_var = tk.StringVar()
        self.threshold_var = tk.IntVar(value=160)
        self.target_width_var = tk.StringVar(value="595")
        self.target_height_var = tk.StringVar(value="842")
        self.inner_radius_var = tk.StringVar(value="28")
        self.outer_radius_var = tk.StringVar(value="293")
        self.max_shots_var = tk.StringVar(value="10")
        self.total_score_var = tk.StringVar(value="0")
        self.shot_count_var = tk.StringVar(value="0 / 10")
        self.status_var = tk.StringVar(value="Ready. Configure the camera and press Start.")

        self._build_style()
        self._build_ui()
        self._update_source_controls()
        self.threshold_var.trace_add("write", self._on_threshold_changed)

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        available_themes = style.theme_names()
        if "vista" in available_themes:
            style.theme_use("vista")
        style.configure("Status.TLabel", padding=(10, 7))
        style.configure("Score.TLabel", font=("Segoe UI", 22, "bold"))
        style.configure("Heading.TLabel", font=("Segoe UI", 12, "bold"))

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
        self.camera_spinbox = ttk.Spinbox(
            controls, from_=0, to=20, textvariable=self.camera_id_var, width=5
        )
        self.camera_spinbox.grid(row=0, column=5, padx=(0, 12), pady=8)

        self.video_entry = ttk.Entry(
            controls, textvariable=self.video_path_var, width=28
        )
        self.video_entry.grid(row=0, column=6, padx=(0, 4), pady=8)
        self.browse_button = ttk.Button(
            controls, text="Browse...", command=self._browse_video
        )
        self.browse_button.grid(row=0, column=7, padx=(0, 12), pady=8)

        self.start_button = ttk.Button(controls, text="Start", command=self.start)
        self.start_button.grid(row=0, column=10, padx=4, pady=8)
        self.stop_button = ttk.Button(
            controls, text="Stop", command=self.stop, state="disabled"
        )
        self.stop_button.grid(row=0, column=11, padx=4, pady=8)
        self.reset_button = ttk.Button(controls, text="Reset round", command=self.reset_round)
        self.reset_button.grid(row=0, column=12, padx=(4, 10), pady=8)

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
        self.camera_spinbox.configure(state="normal" if webcam_selected else "disabled")
        self.video_entry.configure(state="disabled" if webcam_selected else "normal")
        self.browse_button.configure(state="disabled" if webcam_selected else "normal")

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

    def _read_source(self) -> int | str:
        if self.source_type_var.get() == "Webcam":
            try:
                source = int(self.camera_id_var.get())
            except ValueError as exc:
                raise ValueError("Camera ID must be a whole number.") from exc
            self.source_is_webcam = True
            return source

        path = Path(self.video_path_var.get().strip())
        if not path.is_file():
            raise ValueError("Select an existing video file.")
        self.source_is_webcam = False
        return str(path)

    def start(self) -> None:
        if self.running:
            return

        try:
            config = self._read_config()
            source = self._read_source()
            capture = open_video_capture(source)
        except (RuntimeError, ValueError) as exc:
            messagebox.showerror(APP_TITLE, str(exc), parent=self.root)
            self.status_var.set(f"Could not start: {exc}")
            return

        self.capture = capture
        self.engine = LaserShooterEngine(config)
        self.player_name = self.player_var.get().strip() or "Player"
        self.running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._set_settings_state("disabled")
        self._clear_scoreboard()
        self.status_var.set("Camera started. Aim it at all four target markers.")
        self.after_id = self.root.after(0, self._process_frame)

    def stop(self) -> None:
        self._stop_capture("Stopped.")

    def _stop_capture(self, status: str) -> None:
        self.running = False
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self._set_settings_state("normal")
        self._update_source_controls()
        self.status_var.set(status)

    def _set_settings_state(self, state: str) -> None:
        self.player_entry.configure(state=state)
        self.source_combo.configure(state="readonly" if state == "normal" else "disabled")
        self.camera_spinbox.configure(state="disabled" if state == "disabled" else "normal")
        self.video_entry.configure(state="disabled")
        self.browse_button.configure(state="disabled")
        for entry in self.config_entries:
            entry.configure(state=state)

    def _process_frame(self) -> None:
        self.after_id = None
        if not self.running or self.capture is None or self.engine is None:
            return

        success, frame = self.capture.read()
        if not success:
            status = (
                "The webcam stopped returning frames."
                if self.source_is_webcam
                else "Video playback finished."
            )
            self._stop_capture(status)
            return

        try:
            result = self.engine.process_frame(frame, self.player_name)
            self._render_result(result)
        except Exception as exc:
            self._stop_capture(f"Processing error: {exc}")
            messagebox.showerror(APP_TITLE, str(exc), parent=self.root)
            return

        self.after_id = self.root.after(15, self._process_frame)

    def _render_result(self, result: FrameResult) -> None:
        self._show_image(self.preview_label, result.preview, PREVIEW_SIZE)
        self._show_image(self.mask_label, result.threshold_mask, PREVIEW_SIZE)

        if result.target is None:
            self._clear_image(
                self.target_label,
                f"Aim at the full target\nMarkers detected: {result.marker_count}/4",
            )
        else:
            self._show_image(self.target_label, result.target, PREVIEW_SIZE)

        for shot in result.new_shots:
            self.score_tree.insert(
                "", "end", values=(shot.number, shot.score, f"({shot.x}, {shot.y})")
            )

        if self.engine is None:
            return
        self.total_score_var.set(str(self.engine.total_score))
        self.shot_count_var.set(
            f"{len(self.engine.shots)} / {self.engine.config.max_shots} shots"
        )

        if result.round_complete:
            self.status_var.set(
                f"Round complete — total score: {self.engine.total_score}. "
                "Press Reset round to continue."
            )
        elif result.marker_count == 4:
            self.status_var.set("Target locked. Ready for laser shots.")
        else:
            self.status_var.set(f"Searching for target markers: {result.marker_count}/4")

    def reset_round(self) -> None:
        if self.engine is not None:
            self.engine.reset()
        self._clear_scoreboard()
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

    def _on_threshold_changed(self, *_args: object) -> None:
        if self.engine is not None:
            self.engine.update_threshold(self.threshold_var.get())

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
        self._stop_capture("Closed.")
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    LaserShooterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
