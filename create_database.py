from engine.preprocessor import create_database

if __name__ == "__main__":
    import os
    import sys

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    SVG_DIR = os.path.join(PROJECT_ROOT, 'data')
    
    OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'assets', 'kanjivg_normalized.pkl')

    if not os.path.exists(SVG_DIR):
        print(f"Директория {SVG_DIR} не найдена. Убедитесь, что структура проекта верна.")
        sys.exit(1)

    create_database(SVG_DIR, OUTPUT_PATH)
    print(f"База данных успешно создана и сохранена в {OUTPUT_PATH}.")