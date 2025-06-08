import pickle
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from .data_models import NormalizedKanji, RecognitionResult

# --- Весовые коэффициенты для оценки расстояния ---
W_SHAPE = 1.0      # Форма штриха (DTW).
W_POSITION = 2.5   # Положение штриха на холсте (сравнение центроидов).
W_SIZE = 1.5       # Размер штриха (сравнение длин).


class Matcher:
    """
    Класс, инкапсулирующий всю логику распознавания иероглифов.
    Он сравнивает предоставленный рисунок с эталонными данными из
    предварительно обработанной базы KanjiVG.
    """
    def __init__(self, database_path: str):
        """
        Инициализирует Matcher, загружая базу данных с пре-рассчитанными фичами.

        Args:
            database_path: Путь к файлу .pkl, созданному preprocessor.py.
        """
        self.database: dict[str, NormalizedKanji] = self._load_database(database_path)
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
        Основной метод распознавания. Принимает нормализованный рисунок
        пользователя и возвращает N наиболее вероятных кандидатов.

        Args:
            user_drawing: Объект NormalizedKanji, представляющий рисунок пользователя.
            top_n: Количество лучших совпадений для возврата.

        Returns:
            Список объектов RecognitionResult, отсортированный по похожести.
        """
        # Этап 1: Быстрая фильтрация кандидатов по числу штрихов.
        candidates = self._filter_candidates(user_drawing)
        if not candidates:
            return []

        # Этап 2: Расчет "расстояния" для каждого кандидата.
        scores = []
        for candidate in candidates:
            distance = self._calculate_distance(user_drawing, candidate)
            # Исключаем кандидатов, которые не прошли базовую проверку (например, по числу штрихов).
            if distance != float('inf'):
                scores.append((candidate.character, distance))
        
        if not scores:
            return []

        # Этап 3: Сортировка по расстоянию и форматирование результата.
        scores.sort(key=lambda item: item[1])
        
        results = []
        # Лучший результат (с наименьшим расстоянием) используется для расчета уверенности.
        min_distance = scores[0][1]

        for char, dist in scores[:top_n]:
            # Уверенность: 1.0 для идеального совпадения, для остальных убывает.
            # Если min_distance равно 0, это значит лучший кандидат идеально совпал.
            if dist <= min_distance:
                confidence = 1.0
            else:
                # Простая относительная метрика уверенности.
                confidence = min_distance / dist
            
            results.append(RecognitionResult(
                character=char,
                distance=round(dist, 2),
                confidence=round(confidence, 4)
            ))
            
        return results

    def _filter_candidates(self, user_drawing: NormalizedKanji) -> list[NormalizedKanji]:
        """
        Отбирает из всей базы данных только тех кандидатов, которые
        имеют право на детальное сравнение.
        Основной и самый эффективный фильтр - по количеству штрихов.
        """
        user_stroke_count = user_drawing.stroke_count
        
        # abs(kanji.stroke_count - user_stroke_count) <= 1
        return [
            kanji for kanji in self.database.values()
            if kanji.stroke_count == user_stroke_count
        ]

    def _calculate_distance(self, drawing_A: NormalizedKanji, drawing_B: NormalizedKanji) -> float:
        """
        Вычисляет итоговое "расстояние" между двумя иероглифами.
        Это сердце алгоритма распознавания. Он использует взвешенную сумму
        расстояний по форме, положению и размеру для каждого штриха.
        """
        # Гарантируем, что сравниваем иероглифы с одинаковым числом штрихов.
        if drawing_A.stroke_count != drawing_B.stroke_count or drawing_A.stroke_count == 0:
            return float('inf')

        total_distance = 0.0

        # Сравниваем иероглифы поштрихово.
        for i in range(drawing_A.stroke_count):
            stroke_A = np.array(drawing_A.normalized_strokes[i])
            features_A = drawing_A.stroke_features[i]

            stroke_B = np.array(drawing_B.normalized_strokes[i])
            features_B = drawing_B.stroke_features[i]

            # 1. Расстояние по форме (Dynamic Time Warping).
            dist_shape, _ = fastdtw(stroke_A, stroke_B, dist=euclidean)

            # 2. Расстояние по положению (дистанция между центроидами штрихов).
            dist_position = euclidean(features_A.centroid, features_B.centroid)

            # 3. Расстояние по размеру (разница в длине штрихов).
            dist_size = abs(features_A.length - features_B.length)
            
            # 4. Вычисляем взвешенную сумму для текущей пары штрихов.
            stroke_distance = (
                (dist_shape * W_SHAPE) +
                (dist_position * W_POSITION) +
                (dist_size * W_SIZE)
            )
            total_distance += stroke_distance
        
        # Нормализуем итоговое расстояние на количество штрихов, чтобы
        # иероглифы с большим числом штрихов не получали систематически больший "штраф".
        return total_distance / drawing_A.stroke_count