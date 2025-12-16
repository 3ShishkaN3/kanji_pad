# Файл: engine/preprocessor.py

import os
import pickle
import numpy as np
from svgpathtools import parse_path
from tqdm import tqdm
from .data_models import (
    KanjiComponent, NormalizedKanji, NormalizedStroke, StrokeFeatures,
    StrokeData, Point, BoundingBox
)
from .svg_parser import parse_svg_file

def _calculate_stroke_features(stroke: NormalizedStroke) -> StrokeFeatures:
    """Вычисляет и возвращает все фичи для одного штриха."""
    if not stroke:
        # Возвращаем "пустой" объект на случай пустого штриха
        return StrokeFeatures(
            bounding_box=((0, 0), (0, 0)), start_point=(0, 0), end_point=(0, 0),
            centroid=(0, 0), length=0.0
        )
        
    points = np.array(stroke)
    
    # Bounding Box
    min_coords = tuple(points.min(axis=0))
    max_coords = tuple(points.max(axis=0))
    
    # Длина штриха (сумма расстояний между последовательными точками)
    length = np.sum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    
    return StrokeFeatures(
        bounding_box=(min_coords, max_coords),
        start_point=tuple(points[0]),
        end_point=tuple(points[-1]),
        centroid=tuple(points.mean(axis=0)),
        length=float(length)
    )

def _get_global_features(all_strokes: list[NormalizedStroke]) -> tuple[BoundingBox, Point]:
    """Вычисляет общие BoundingBox и центроид для всего иероглифа."""
    if not all_strokes:
        return (((0, 0), (0, 0)), (0, 0))
    
    all_points = np.array([point for stroke in all_strokes for point in stroke])
    if all_points.size == 0:
        return (((0, 0), (0, 0)), (0, 0))
        
    min_coords = tuple(all_points.min(axis=0))
    max_coords = tuple(all_points.max(axis=0))
    centroid = tuple(all_points.mean(axis=0))
    
    return ((min_coords, max_coords), centroid) # type: ignore

def _flatten_strokes(component: KanjiComponent) -> list[StrokeData]:
    """
    Рекурсивно собирает все штрихи из дерева компонентов в один
    плоский список, сохраняя их порядок.
    """
    # Собираем штрихи текущего уровня
    all_strokes = component.strokes[:]
    # Рекурсивно собираем штрихи из дочерних компонентов
    for child in component.children:
        all_strokes.extend(_flatten_strokes(child))
    return all_strokes


def _sample_path(path_data: str, num_points: int = 32) -> NormalizedStroke:
    """
    Преобразует строку SVG path в список из N точек (x, y).
    """
    path = parse_path(path_data)
    points = []
    for i in range(num_points):
        # t - параметр от 0.0 до 1.0, обозначающий положение на кривой
        t = i / (num_points - 1)
        point = path.point(t)
        # svgpathtools возвращает комплексные числа, где real=x, imag=y
        points.append((point.real, point.imag))
    return points


def _normalize_kanji(strokes: list[NormalizedStroke]) -> list[NormalizedStroke]:
    """
    Масштабирует и центрирует иероглиф, чтобы он вписывался
    в условный квадрат 100x100.
    """
    if not strokes or not any(strokes):
        return []

    # сбор всех точек в один numpy массив для эффективных вычислений
    all_points = np.array([point for stroke in strokes for point in stroke])

    min_coords = all_points.min(axis=0)
    max_coords = all_points.max(axis=0)

    # сдвиг всех точек так, чтобы левый верхний угол был в (0, 0)
    points_translated = all_points - min_coords

    size = max_coords - min_coords
    # Чтобы избежать деления на ноль для иероглифов, состоящих из одной точки
    size[size == 0] = 1

    scale = 100.0 / size.max()
    
    # масштабирование
    points_scaled = points_translated * scale

    # расчёт сдвига для центрирования
    scaled_size = size * scale
    offset = (100.0 - scaled_size) / 2.0
    
    # сдвиг для центрирования
    points_normalized = points_scaled + offset

    # точки обратно в структуру штрихов
    normalized_strokes = []
    start_index = 0
    for stroke in strokes:
        end_index = start_index + len(stroke)
        normalized_strokes.append([tuple(p) for p in points_normalized[start_index:end_index].tolist()])
        start_index = end_index
        
    return normalized_strokes


def create_database(svg_dir: str, output_path: str):
    """
    Обрабатывает все SVG-файлы из директории, нормализует их
    и сохраняет в единый файл базы данных.
    """
    print(f"Starting preprocessing of SVG files in '{svg_dir}'...")
    
    kanji_database = {}
    svg_files = [f for f in os.listdir(svg_dir) if f.endswith('.svg')]

    for filename in tqdm(svg_files, desc="Processing SVGs", unit="file"):
        filepath = os.path.join(svg_dir, filename)

        try:
            root_component = parse_svg_file(filepath)

            if not root_component.children:
                print(f"⚠️  No child component found in {filename}, skipping.")
                continue
            
            character_component = root_component.children[0]
            character = character_component.attributes.get('kvg:element')

            if not character:
                print(f"⚠️  No 'kvg:element' attribute on character group in {filename}, skipping.")
                continue

            all_raw_strokes = _flatten_strokes(root_component)
            all_raw_strokes.sort(key=lambda s: s.id_number)
            
            sampled_strokes = [_sample_path(stroke.path_data) for stroke in all_raw_strokes]
            normalized = _normalize_kanji(sampled_strokes)

            features_per_stroke = [_calculate_stroke_features(s) for s in normalized]
            
            global_box, global_centroid = _get_global_features(normalized)

            kanji_database[character] = NormalizedKanji(
                character=character,
                normalized_strokes=normalized,
                stroke_features=features_per_stroke,
                global_bounding_box=global_box,
                global_centroid=global_centroid,
                source_component_tree=root_component
            )

        except Exception as e:
            print(f"❌ Error processing '{filename}': {e}")
            continue
    with open(output_path, 'wb') as f:
        pickle.dump(kanji_database, f)
        
    print(f"\n✅ Database created successfully at '{output_path}' with {len(kanji_database)} entries.")