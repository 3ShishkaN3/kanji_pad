# kanji_pad

<details>
<summary>🇷🇺 Русский</summary>

## Описание
kanji_pad — это экспериментальная платформа для офлайн-распознавания иероглифов кандзи по нарисованным пользователем штрихам. Репозиторий содержит движок нормализации, алгоритм сравнения и клиенты (Tkinter и gRPC).

> ⚠️ **Важное примечание:** В репозитории **отсутствуют** папки `data/` и `assets/`. Это сделано для соблюдения лицензионной чистоты (не затрагивая лицензию KanjiVG). Чтобы проект заработал, вам необходимо самостоятельно скачать исходные SVG-файлы и собрать базу.

## Основные возможности
- 🧠 Нормализация штрихов и извлечение геометрических признаков (`engine/preprocessor.py`).
- ⚙️ Векторизованный подбор ближайших иероглифов через `Matcher` (`engine/matcher.py`).
- 🖥️ Tkinter-клиент для рисования и распознавания.
- 🔌 gRPC API для интеграции (`engine/server.py`).

## Быстрый старт
1. **Установите зависимости:** `pip install -r requirements.txt`.
2. **Подготовьте данные:**
   - Создайте в корне проекта папку `data/`.
   - Скачайте SVG-файлы из репозитория [KanjiVG](https://github.com/KanjiVG/kanjivg) и поместите их в `data/`.
   - Создайте пустую папку `assets/`.
3. **Сгенерируйте базу:** Выполните `python create_database.py`. Скрипт обработает SVG и создаст файл `assets/kanjivg_normalized.pkl`.
4. **Запустите клиент:**
   - Tkinter GUI: `python clients/tkinter/main.py`.
   - gRPC сервер: `python engine/server.py`.

## Структура каталогов
На данный момент в репозитории содержатся только программные модули:
```
clients/        — клиенты (Tkinter, gRPC примеры)
engine/         — логика обработки, препроцессор и матчер
protos/         — описание gRPC контракта
create_database.py — скрипт сборки БД (требует наличия папок data/ и assets/)
```

## Лицензия и атрибуция
Проект распространяется по модифицированной Apache License 2.0. При любом публичном использовании необходимо указать автора: Reznik Danil Maksimovich (aka "Shishka"), согласно требованиям LICENSE и NOTICE.

</details>

<details>
<summary>🇬🇧 English</summary>

## Overview
kanji_pad is an experimental offline kanji recognition stack. It features a stroke-normalization engine, similarity matcher, and multiple clients.

> ⚠️ **Important Note:** The repository **does not** include `data/` and `assets/` folders. This is intentional to comply with **KanjiVG** licensing. You must manually download the source data and build the database yourself.

## Highlights
- 🧠 Stroke normalization & feature extraction pipeline (`engine/preprocessor.py`).
- ⚙️ Vectorized Matcher (`engine/matcher.py`) for finding candidates.
- 🖥️ Tkinter client for drawing and recognition.
- 🔌 gRPC API for backend integrations.

## Quick start
1. **Install deps:** `pip install -r requirements.txt`.
2. **Prepare data:**
   - Create a `data/` folder in the root directory.
   - Download SVG files from [KanjiVG](https://github.com/KanjiVG/kanjivg) and put them into `data/`.
   - Create an empty `assets/` folder.
3. **Build the database:** Run `python create_database.py`. This will process the SVGs and create `assets/kanjivg_normalized.pkl`.
4. **Run a client:**
   - Tkinter GUI: `python clients/tkinter/main.py`
   - gRPC server: `python engine/server.py`

## Repository layout
Currently, the repository only ships with the core logic and clients:
```
clients/        — UI and gRPC clients
engine/         — normalization, features, and matcher logic
protos/         — gRPC contract
create_database.py — DB generation script (requires data/ and assets/ folders)
```

## License & attribution
Released under modified Apache License 2.0. Any public-facing use must credit the author: Reznik Danil Maksimovich (aka "Shishka"), as enforced by LICENSE and NOTICE.

</details>

<details>
<summary>🇯🇵 日本語</summary>

## 概要
kanji_pad は、手描きストロークから漢字をオフラインで認識するための実験的プロジェクトです。正規化エンジン、マッチャー、およびクライアント（Tkinter、gRPC）が含まれています。

> ⚠️ **重要な注意点:** **KanjiVG** のライセンスに配慮し、本リポジトリには `data/` および `assets/` フォルダは含まれていません。利用には、ソースデータのダウンロードとデータベースのビルドをご自身で行う必要があります。

## 特長
- 🧠 `engine/preprocessor.py` による正規化と特徴量抽出。
- ⚙️ ベクトル化された `Matcher` による高速検索。
- 🖥️ Tkinter による描画・認識クライアント。
- 🔌 外部連携用 gRPC API。

## クイックスタート
1. **依存関係のインストール:** `pip install -r requirements.txt`
2. **データの準備:**
   - ルートディレクトリに `data/` フォルダを作成します。
   - [KanjiVG](https://github.com/KanjiVG/kanjivg) から SVG ファイルをダウンロードし、`data/` に配置します。
   - `assets/` フォルダを作成します。
3. **データベースの生成:** `python create_database.py` を実行します。`assets/kanjivg_normalized.pkl` が生成されます。
4. **実行:**
   - Tkinter GUI: `python clients/tkinter/main.py`
   - gRPC サーバー: `python engine/server.py`

## ディレクトリ構成
現在、リポジトリには以下のプログラムモジュールのみが含まれています。
```
clients/        — クライアント (Tkinter, gRPC)
engine/         — 処理ロジック、正規化、マッチャー
protos/         — gRPC インターフェース
create_database.py — DB 生成スクリプト (data/ と assets/ フォルダが必要)
```

## ライセンス
本プロジェクトは修正版 Apache License 2.0 の下で公開されています。公開利用の際は、作者 Reznik Danil Maksimovich ("Shishka") のクレジットを明記してください。

</details>
