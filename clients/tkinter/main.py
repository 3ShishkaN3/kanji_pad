import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import os
import sys
import numpy as np

# Настройка путей
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

try:
    from engine.data_models import NormalizedKanji, NormalizedStroke
    from engine.matcher import Matcher
    from engine.preprocessor import _normalize_kanji, _calculate_stroke_features
except ImportError as e:
    print(f"Ошибка: Не удалось импортировать компоненты движка: {e}")
    print(f"Убедитесь, что структура проекта верна и {PROJECT_ROOT} содержит папку 'engine'.")
    sys.exit(1)


class KanjiPadApp:
    """
    Интерактивное приложение для рисования и распознавания кандзи.
    """
    DATABASE_PATH = os.path.join(PROJECT_ROOT, 'assets', 'kanjivg_normalized.pkl')
    CANVAS_SIZE = 400
    NORMALIZATION_SIZE = 100

    def __init__(self, root):
        self.root = root
        self.root.title("Kanji Pad")

        try:
            self.database = self._load_database()
            self.matcher = Matcher(self.DATABASE_PATH)
        except RuntimeError as e:
            messagebox.showerror("Критическая ошибка", str(e))
            self.root.destroy()
            return

        self.mode = 'view'  # 'view' или 'draw'
        self.scale_factor = self.CANVAS_SIZE / self.NORMALIZATION_SIZE
        
        # Для режима просмотра
        self.current_kanji_to_view: NormalizedKanji | None = None
        self.current_stroke_index = 0

        # Для режима рисования
        self.user_drawing_raw: list[list[tuple[int, int]]] = []
        self.current_stroke_raw: list[tuple[int, int]] = []

        self._setup_ui()
        self._update_ui_for_mode()

    def _load_database(self) -> dict[str, NormalizedKanji]:
        """Загружает базу данных иероглифов из .pkl файла."""
        if not os.path.exists(self.DATABASE_PATH):
            raise RuntimeError(
                f"Файл базы данных не найден: {self.DATABASE_PATH}\n"
                "Запустите preprocessor.py для его создания."
            )
        with open(self.DATABASE_PATH, 'rb') as f:
            db = pickle.load(f)
            if not db:
                raise RuntimeError("База данных пуста. Проверьте работу preprocessor.py.")
            return db

    def _setup_ui(self):
        """Создает и размещает все элементы интерфейса."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Панель, разделяющая холст и результаты
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.grid(row=1, column=0, sticky="nsew", pady=5)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        #  Левая часть: Управление и Холст
        left_pane = ttk.Frame(paned_window, padding=5)
        paned_window.add(left_pane, weight=1)

        # Контролы (верхняя панель)
        controls_frame = ttk.Frame(left_pane)
        controls_frame.pack(fill=tk.X, pady=5)
        
        self.mode_button = ttk.Button(controls_frame, text="Переключить в режим рисования", command=self._toggle_mode)
        self.mode_button.pack(side=tk.LEFT, padx=5)
        
        # Контролы для режима просмотра
        self.viewer_controls = ttk.Frame(controls_frame)
        ttk.Label(self.viewer_controls, text="Кандзи:").pack(side=tk.LEFT)
        self.kanji_selector = ttk.Combobox(
            self.viewer_controls, values=sorted(list(self.database.keys())), state="readonly", width=8
        )
        self.kanji_selector.pack(side=tk.LEFT, padx=5)
        self.kanji_selector.bind("<<ComboboxSelected>>", self._on_kanji_select)
        
        # Контролы для режима рисования
        self.drawing_controls = ttk.Frame(controls_frame)
        self.recognize_button = ttk.Button(self.drawing_controls, text="Распознать", command=self._recognize_drawing)
        self.recognize_button.pack(side=tk.LEFT, padx=5)
        self.clear_button = ttk.Button(self.drawing_controls, text="Очистить", command=self._clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # Холст
        self.canvas = tk.Canvas(left_pane, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE, bg="white", relief="solid")
        self.canvas.pack(pady=5)

        # Правая часть: Результаты распознавания
        right_pane = ttk.Frame(paned_window, padding=5)
        paned_window.add(right_pane, weight=1)
        right_pane.columnconfigure(0, weight=1)
        right_pane.rowconfigure(1, weight=1)

        ttk.Label(right_pane, text="Результаты распознавания:", font="-weight bold").grid(row=0, column=0, sticky="w")
        
        cols = ("kanji", "distance", "confidence")
        self.results_tree = ttk.Treeview(right_pane, columns=cols, show="headings", height=10)
        self.results_tree.grid(row=1, column=0, sticky="nsew")
        
        self.results_tree.heading("kanji", text="Кандзи")
        self.results_tree.heading("distance", text="Расстояние")
        self.results_tree.heading("confidence", text="Уверенность")
        self.results_tree.column("kanji", width=80, anchor=tk.CENTER)
        self.results_tree.column("distance", width=120, anchor=tk.W)
        self.results_tree.column("confidence", width=120, anchor=tk.W)

    def _toggle_mode(self):
        """Переключает между режимами 'view' и 'draw'."""
        self.mode = 'draw' if self.mode == 'view' else 'view'
        self._update_ui_for_mode()
        self._clear_canvas()

    def _update_ui_for_mode(self):
        """Обновляет состояние UI в соответствии с текущим режимом."""
        if self.mode == 'view':
            self.mode_button.config(text="Переключить в режим рисования")
            # Показываем контролы для просмотра, прячем для рисования
            self.drawing_controls.pack_forget()
            self.viewer_controls.pack(side=tk.LEFT, padx=10)
            # Отвязываем события мыши
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            # Привязываем навигацию по штрихам
            self.root.bind("<Key>", self._handle_viewer_key_press)

        else: # mode == 'draw'
            self.mode_button.config(text="Переключить в режим просмотра")
            # Прячем контролы для просмотра, показываем для рисования
            self.viewer_controls.pack_forget()
            self.drawing_controls.pack(side=tk.LEFT, padx=10)
            # Привязываем события мыши для рисования
            self.canvas.bind("<Button-1>", self._on_mouse_press)
            self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
            self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
            # Отвязываем лишние события
            self.root.unbind("<Key>")
        self._clear_canvas()

    # Методы для режима рисовани

    def _on_mouse_press(self, event):
        self.current_stroke_raw = [(event.x, event.y)]

    def _on_mouse_drag(self, event):
        self.current_stroke_raw.append((event.x, event.y))
        if len(self.current_stroke_raw) > 1:
            self.canvas.create_line(self.current_stroke_raw[-2:], fill="black", width=3, capstyle="round")

    def _on_mouse_release(self, event):
        if len(self.current_stroke_raw) > 1:
            self.user_drawing_raw.append(self.current_stroke_raw)
        self.current_stroke_raw = []

    def _clear_canvas(self):
        """Очищает холст и все связанные данные."""
        self.canvas.delete("all")
        self.user_drawing_raw = []
        self.current_kanji_to_view = None
        for i in self.results_tree.get_children():
            self.results_tree.delete(i)

    def _recognize_drawing(self):
        """Запускает процесс распознавания нарисованного."""
        if not self.user_drawing_raw:
            messagebox.showinfo("Информация", "Сначала нарисуйте что-нибудь на холсте.")
            return

        # 1. Превращаем "сырой" рисунок в объект NormalizedKanji
        sampled_strokes = [[(x / self.CANVAS_SIZE * self.NORMALIZATION_SIZE, 
                             y / self.CANVAS_SIZE * self.NORMALIZATION_SIZE) 
                            for x, y in stroke] for stroke in self.user_drawing_raw]

        normalized_strokes = _normalize_kanji(sampled_strokes)
        stroke_features = [_calculate_stroke_features(s) for s in normalized_strokes]
        global_box, global_centroid = _get_global_features(normalized_strokes)

        user_drawing_obj = NormalizedKanji(
            character=None, # У пользовательского рисунка нет символа
            normalized_strokes=normalized_strokes,
            stroke_features=stroke_features,
            global_bounding_box=global_box,
            global_centroid=global_centroid,
            source_component_tree=None # type: ignore
        )

        # 2. Передаем объект в матчер
        results = self.matcher.recognize(user_drawing_obj, top_n=5)

        # 3. Отображаем результаты
        for i in self.results_tree.get_children():
            self.results_tree.delete(i)
        
        for res in results:
            self.results_tree.insert("", tk.END, values=(res.character, res.distance, f"{res.confidence:.2%}"))

    # Методы для режима просмотра

    def _on_kanji_select(self, event=None):
        char = self.kanji_selector.get()
        if char in self.database:
            self.current_kanji_to_view = self.database[char]
            self.current_stroke_index = self.current_kanji_to_view.stroke_count # Показываем сразу весь
            self._draw_viewer_kanji()

    def _handle_viewer_key_press(self, event):
        if not self.current_kanji_to_view: return
        if event.keysym == "Right" and self.current_stroke_index < self.current_kanji_to_view.stroke_count:
            self.current_stroke_index += 1
        elif event.keysym == "Left" and self.current_stroke_index > 0:
            self.current_stroke_index -= 1
        self._draw_viewer_kanji()

    def _draw_viewer_kanji(self):
        self.canvas.delete("all")
        if not self.current_kanji_to_view: return
        for i in range(self.current_stroke_index):
            stroke = self.current_kanji_to_view.normalized_strokes[i]
            scaled_points = [p * self.scale_factor for xy in stroke for p in xy]
            if len(scaled_points) >= 4:
                self.canvas.create_line(scaled_points, fill="black", width=3, capstyle="round")

def _get_global_features(all_strokes: list[NormalizedStroke]):
    # Вспомогательная функция
    if not all_strokes or not any(stroke for stroke in all_strokes): return (((0,0),(0,0)),(0,0))
    all_points = np.array([point for stroke in all_strokes for point in stroke])
    if all_points.size == 0: return (((0,0),(0,0)),(0,0))
    min_c, max_c = all_points.min(axis=0), all_points.max(axis=0)
    return ((tuple(min_c), tuple(max_c)), tuple(all_points.mean(axis=0)))

if __name__ == "__main__":
    root = tk.Tk()
    app = KanjiPadApp(root)
    root.mainloop()