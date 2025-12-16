import pickle
import numpy as np
from .data_models import NormalizedKanji, RecognitionResult

# --- КОНСТАНТЫ ---
# Количество точек в штрихе для векторизации (чем больше, тем точнее форма, но медленнее)
POINTS_PER_STROKE = 32
# Максимальное кол-во штрихов, которое мы ожидаем в базе (для создания тензора фиксированного размера)
MAX_STROKES_IN_DB = 35 

class Matcher:
    """
    Класс, инкапсулирующий логику распознавания иероглифов.
    """
    def __init__(self, database_path: str):
        """
        Инициализирует Matcher, загружая базу данных.
        """
        self.database: dict[str, NormalizedKanji] = self._load_database(database_path)
        
        # Кэши для быстрого векторного доступа
        # self.db_tensor: (N_Kanji, MAX_STROKES, POINTS_PER_STROKE, 2)
        self.db_tensor = None 
        self.db_chars = []
        self.db_stroke_counts = []
        
        self._build_internal_caches()
        
        print(f"Matcher initialized with {len(self.database)} kanji entries.")

    def _load_database(self, path: str) -> dict[str, NormalizedKanji]:
        """Загружает базу данных из файла .pkl."""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Database file not found at '{path}'. Please run the preprocessor first.")
        except Exception as e:
            raise RuntimeError(f"Failed to load or parse database file '{path}': {e}")

    def recognize(self, user_drawing: NormalizedKanji, top_n: int = 5) -> list[RecognitionResult]:
        """
        Основной метод распознавания.
        """
        user_tensor, user_count = self._preprocess_user_input(user_drawing)
        
        if user_count == 0:
            return []

        # Логика предикшна: ищем иероглифы, у которых штрихов >= чем нарисовано
        # Эвристика: не больше чем +10 штрихов, чтобы отсечь совсем сложные
        mask = (self.db_stroke_counts >= user_count) & (self.db_stroke_counts <= user_count + 10)
        candidate_indices = np.where(mask)[0]
        
        if len(candidate_indices) == 0:
            return []

        # Shape: (N_Candidates, MAX_STROKES, 32, 2)
        subset_db = self.db_tensor[candidate_indices]

        # Нам нужно найти: насколько каждый штрих юзера похож на ЛУЧШИЙ штрих кандидата.
        
        # Инициализируем массив очков (расстояний) для кандидатов
        scores = np.zeros(len(candidate_indices), dtype=np.float32)

        # Цикл только по штрихам пользователя (их мало, обычно 1-10).
        # Внутренний цикл по базе (тысячи элементов) делает NumPy на C-скорости.
        for i in range(user_count):
            u_stroke = user_tensor[i]  # (32, 2)
            
            # Вычитаем штрих юзера из ВСЕХ штрихов ВСЕХ кандидатов сразу
            # subset_db: (N, MaxS, 32, 2)
            # u_stroke:  (32, 2) -> broadcast -> (N, MaxS, 32, 2)
            
            diff = subset_db - u_stroke
            
            # Евклидово расстояние: sqrt(sum(dx^2 + dy^2))
            # (N, MaxS, 32, 2) -> norm -> (N, MaxS, 32) -> sum -> (N, MaxS)
            # Суммируем расстояние по всем точкам штриха
            dists = np.sum(np.linalg.norm(diff, axis=3), axis=2)
            
            diff_rev = subset_db - u_stroke[::-1]
            dists_rev = np.sum(np.linalg.norm(diff_rev, axis=3), axis=2)
            
            # Берем минимум (прямое или обратное)
            dists = np.minimum(dists, dists_rev)
            
            # Для текущего штриха юзера находим САМЫЙ похожий штрих внутри каждого иероглифа
            # min по оси штрихов (axis=1) -> (N,)
            best_match_dists = np.min(dists, axis=1)
            
            # Добавляем к общему штрафу кандидата
            scores += best_match_dists

        # Делим на кол-во штрихов, чтобы получить среднее расстояние
        avg_scores = scores / user_count

        # Быстрая сортировка топ-N через argpartition
        k = min(top_n, len(avg_scores))
        if k == 0: return []
        
        # Получаем индексы лучших локально в subset
        best_local_indices = np.argpartition(avg_scores, k-1)[:k]
        
        # Сортируем эти топ-K (так как argpartition не гарантирует порядок)
        top_k_scores = avg_scores[best_local_indices]
        sorted_order = np.argsort(top_k_scores)
        
        results = []
        min_distance = top_k_scores[sorted_order[0]] + 1e-6 # Защита от деления на 0

        for idx in sorted_order:
            local_idx = best_local_indices[idx]
            global_idx = candidate_indices[local_idx]
            
            dist = top_k_scores[idx]
            char = self.db_chars[global_idx]
            
            # Ваша логика уверенности
            if dist <= min_distance:
                confidence = 1.0
            else:
                confidence = min_distance / dist

            results.append(RecognitionResult(
                character=char,
                distance=round(float(dist), 2),
                confidence=round(float(confidence), 4)
            ))

        return results

    def _build_internal_caches(self) -> None:
        """
        Превращает словарь NormalizedKanji в оптимизированный 4D тензор NumPy.
        Это выполняется 1 раз при старте сервера.
        """
        tensor_list = []
        chars_list = []
        counts_list = []

        print("Building vectorized cache...")

        for char, kanji in self.database.items():
            # Получаем сырые штрихи
            raw_strokes = [np.array(s, dtype=np.float32) for s in kanji.normalized_strokes]
            if not raw_strokes:
                continue

            # Нормализуем иероглиф (Scale & Center)
            norm_strokes = self._normalize_kanji_geometry(raw_strokes)
            
            # Создаем пустой массив штрихов, заполненный "бесконечностью"
            # Inf нужен, чтобы np.min() игнорировал пустые слоты
            padded_char = np.full((MAX_STROKES_IN_DB, POINTS_PER_STROKE, 2), np.inf, dtype=np.float32)
            
            # Заполняем реальными штрихами
            for i, stroke in enumerate(norm_strokes[:MAX_STROKES_IN_DB]):
                padded_char[i] = stroke

            tensor_list.append(padded_char)
            chars_list.append(char)
            counts_list.append(len(norm_strokes))

        # Финализация массивов
        self.db_tensor = np.array(tensor_list, dtype=np.float32)
        self.db_chars = np.array(chars_list)
        self.db_stroke_counts = np.array(counts_list, dtype=np.int32)

    def _preprocess_user_input(self, user_drawing: NormalizedKanji):
        """Подготовка ввода пользователя к тому же формату, что и база."""
        raw_strokes = [np.array(s, dtype=np.float32) for s in user_drawing.normalized_strokes]
        if not raw_strokes:
            return None, 0
            
        # Та же нормализация, что и для базы!
        norm_strokes = self._normalize_kanji_geometry(raw_strokes)
        return np.array(norm_strokes, dtype=np.float32), len(norm_strokes)

    def _normalize_kanji_geometry(self, strokes: list[np.ndarray]) -> list[np.ndarray]:
        """
        Центрирует и масштабирует весь иероглиф в квадрат (0,0)-(1,1).
        Также ресемплит каждый штрих до POINTS_PER_STROKE точек.
        """
        # Ресемплинг приводит к 32 точкам
        resampled = [self._resample_stroke(s, POINTS_PER_STROKE) for s in strokes]
        
        # общий Bounding Box иероглифа
        all_points = np.vstack(resampled)
        min_xy = all_points.min(axis=0)
        max_xy = all_points.max(axis=0)
        
        dims = max_xy - min_xy
        max_dim = np.max(dims)
        if max_dim == 0: max_dim = 1.0
        
        normalized = []
        for s in resampled:
            # Сдвиг в 0 + Масштаб
            s_norm = (s - min_xy) / max_dim
            normalized.append(s_norm)
            
        return normalized

    def _resample_stroke(self, stroke: np.ndarray, n: int) -> np.ndarray:
        """Интерполяция штриха к фиксированному числу точек N."""
        if len(stroke) == n: 
            return stroke
            
        dists = np.sqrt(np.sum(np.diff(stroke, axis=0)**2, axis=1))
        cum_dist = np.insert(np.cumsum(dists), 0, 0)
        total_len = cum_dist[-1]
        
        if total_len == 0:
            return np.tile(stroke[0], (n, 1))
            
        # Линейная интерполяция
        target_dists = np.linspace(0, total_len, n)
        x = np.interp(target_dists, cum_dist, stroke[:, 0])
        y = np.interp(target_dists, cum_dist, stroke[:, 1])
        
        return np.stack((x, y), axis=1)