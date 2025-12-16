# -*- coding: utf-8 -*-
# usr/bin/env python3
# coding : utf-8
# PEP-8

import xml.etree.ElementTree as ET
from .data_models import KanjiComponent, StrokeData

# Объявление пространства имен для парсера
KVG_NS = "http://kanjivg.tagaini.net"
ET.register_namespace("kvg", KVG_NS)

def _parse_group_node(node: ET.Element, stroke_counter: dict) -> KanjiComponent:
    """Рекурсивная вспомогательная функция для парсинга тегов <g>."""
    
    attributes = {k.replace(f'{{{KVG_NS}}}', 'kvg:'): v 
                  for k, v in node.attrib.items() if KVG_NS in k}
    
    child_components = []
    child_strokes = []
    
    for child in node:
        if child.tag.endswith('g'):
            child_components.append(_parse_group_node(child, stroke_counter)) 
        elif child.tag.endswith('path'):
            stroke_counter['count'] += 1
            stroke_data = StrokeData(
                id_number=stroke_counter['count'],
                path_data=child.attrib.get('d', ''),
                stroke_type=child.attrib.get(f'{{{KVG_NS}}}type')
            )
            child_strokes.append(stroke_data)
            
    return KanjiComponent(
        attributes=attributes,
        strokes=child_strokes,
        children=child_components
    )

def parse_svg_file(filepath: str) -> KanjiComponent:
    """
    Главная функция модуля. Читает SVG-файл и возвращает его
    полное структурное представление в виде объекта KanjiComponent.
    
    Args:
        filepath: Путь к SVG-файлу.
        
    Returns:
        Корневой узел KanjiComponent, представляющий весь иероглиф.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    # Находим основной узел с путями штрихов
    paths_group = root.find(".//{http://www.w3.org/2000/svg}g[@id]")
    if paths_group is None:
        raise ValueError(f"Could not find main stroke group in {filepath}")
        
    # Используем словарь в качестве изменяемого счетчика для передачи по рекурсии
    stroke_counter = {'count': 0}
    
    return _parse_group_node(paths_group, stroke_counter)