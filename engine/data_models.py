# Файл: engine/data_models.py

from dataclasses import dataclass, field
from typing import TypeAlias, Tuple

Point: TypeAlias = Tuple[float, float]
BoundingBox: TypeAlias = Tuple[Point, Point]  # ((min_x, min_y), (max_x, max_y))
NormalizedStroke: TypeAlias = list[Point]


@dataclass(frozen=True)
class StrokeData:
    id_number: int
    path_data: str
    stroke_type: str | None

@dataclass(frozen=True)
class KanjiComponent:
    attributes: dict[str, str] = field(default_factory=dict)
    strokes: list[StrokeData] = field(default_factory=list)
    children: list['KanjiComponent'] = field(default_factory=list)


@dataclass(frozen=True)
class StrokeFeatures:
    """
    Предварительно вычисленные характеристики для ОДНОГО нормализованного штриха.
    Это позволит очень быстро сравнивать штрихи по разным параметрам.
    """
    # Ограничивающая рамка для этого конкретного штриха.
    bounding_box: BoundingBox
    # Координаты начальной и конечной точек.
    start_point: Point
    end_point: Point
    # Геометрический центр (центроид) штриха.
    centroid: Point
    # Суммарная длина штриха.
    length: float

@dataclass(frozen=True)
class NormalizedKanji:
    """
    Обогащенная структура для распознавания. Содержит все необходимые
    нормализованные данные и предварительно вычисленные фичи.
    """
    # Сам символ иероглифа.
    character: str
    
    # Список нормализованных штрихов (последовательностей точек).
    normalized_strokes: list[NormalizedStroke]
    
    # Список фич для каждого штриха. Индекс соответствует normalized_strokes.
    stroke_features: list[StrokeFeatures]
    
    # Глобальные фичи для всего иероглифа
    global_bounding_box: BoundingBox
    global_centroid: Point
    
    # Полное дерево компонентов
    source_component_tree: KanjiComponent

    @property
    def stroke_count(self) -> int:
        return len(self.normalized_strokes)

@dataclass(frozen=True)
class RecognitionResult:
    """Структура для возврата результата распознавания."""
    character: str
    # "Расстояние" до эталона. Чем меньше, тем лучше.
    distance: float
    #  уверенность
    confidence: float