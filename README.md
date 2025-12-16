# kanji_pad

<details>
<summary>🇷🇺 Русский</summary>

## Описание
kanji_pad — это экспериментальная платформа для офлайн-распознавания иероглифов кандзи по нарисованным пользователем штрихам. Репозиторий содержит движок нормализации штрихов, алгоритм сравнения с базой, а также несколько клиентов: настольное Tkinter-приложение и gRPC-сервис.

## Основные возможности
- 💠 Подготовленный датасет KanjiVG в формате `assets/kanjivg_normalized.pkl`.
- 🧠 Нормализация штрихов и извлечение геометрических признаков (`engine/preprocessor.py`).
- ⚙️ Векторизованный подбор ближайших иероглифов через `Matcher` (`engine/matcher.py`).
- 🖥️ Tkinter-клиент для рисования, просмотра и распознавания (`clients/tkinter`).
- 🔌 gRPC API для интеграции с другими сервисами (`clients/grpc`, `engine/server.py`).

## Быстрый старт
1. Установите зависимости: `pip install -r requirements.txt`.
2. Сгенерируйте базу (при необходимости): `python create_database.py`.
3. Запустите нужный клиент:
   - Tkinter GUI: `python clients/tkinter/main.py`.
   - gRPC сервер: `python engine/server.py`.

## Структура каталогов
```
assets/                 — подготовленные данные (KanjiVG, веса и др.)
clients/tkinter/        — настольный интерфейс
clients/grpc/           — пример клиента для gRPC
engine/                 — нормализатор, фичи и матчер
protos/recognition.proto — контракт gRPC сервиса
data/, create_database.py — вспомогательные скрипты и сырьевые данные
```

## Полезные ссылки
- GitHub: https://github.com/3ShishkaN3/kanji_pad
- GitVerse: https://gitverse.ru/shish/kanji_pad

## Лицензия и атрибуция
Проект распространяется по Apache License 2.0. При любом публичном использовании (в веб-интерфейсе, приложении, документации, упаковке и т.п.) необходимо указать автора: Reznik Danil Maksimovich (aka "Shishka"), согласно требованиям LICENSE и NOTICE.

</details>

<details>
<summary>🇬🇧 English</summary>

## Overview
kanji_pad is an experimental offline kanji recognition stack. It ships a stroke-normalization engine, similarity matcher, and multiple clients (Tkinter desktop UI and a gRPC endpoint) that compare user drawings against a prepared KanjiVG dataset.

## Highlights
- 💠 Preprocessed KanjiVG dataset stored in `assets/kanjivg_normalized.pkl`.
- 🧠 Stroke normalization & feature extraction pipeline in `engine/preprocessor.py`.
- ⚙️ Vectorized Matcher (`engine/matcher.py`) that finds the closest glyph candidates.
- 🖥️ Tkinter client for drawing, browsing, and recognition (`clients/tkinter`).
- 🔌 gRPC API for backend integrations (`clients/grpc`, `engine/server.py`).

## Quick start
1. Install deps: `pip install -r requirements.txt`.
2. Build the database if needed: `python create_database.py`.
3. Run a client:
   - Tkinter GUI: `python clients/tkinter/main.py`
   - gRPC server: `python engine/server.py`

## Repository layout
```
assets/                 — prepared datasets (KanjiVG, weights, etc.)
clients/tkinter/        — desktop UI
clients/grpc/           — example gRPC client
engine/                 — normalization, features, matcher
protos/recognition.proto — gRPC contract
data/, create_database.py — helper scripts & raw data
```

## Useful links
- GitHub: https://github.com/3ShishkaN3/kanji_pad
- GitVerse: https://gitverse.ru/shish/kanji_pad

## License & attribution
Released under Apache License 2.0. Any public-facing use (website, app, docs, packaging, etc.) must credit the author: Reznik Danil Maksimovich (aka "Shishka"), as enforced by LICENSE and NOTICE.

</details>

<details>
<summary>🇯🇵 日本語</summary>

## 概要
kanji_pad は、手描きストロークから漢字をオフラインで認識するための実験的スタックです。正規化エンジン、類似度マッチャー、Tkinter デスクトップ UI と gRPC サービスを含み、整形済みの KanjiVG データセットと照合します。

## 特長
- 💠 `assets/kanjivg_normalized.pkl` 内の前処理済み KanjiVG データ。
- 🧠 `engine/preprocessor.py` によるストローク正規化と特徴量抽出。
- ⚙️ 近似候補を検索するベクトル化 `Matcher` (`engine/matcher.py`).
- 🖥️ 描画・閲覧・認識用の Tkinter クライアント (`clients/tkinter`).
- 🔌 他サービス連携向け gRPC API (`clients/grpc`, `engine/server.py`).

## クイックスタート
1. 依存関係をインストール: `pip install -r requirements.txt`
2. 必要ならデータベース生成: `python create_database.py`
3. クライアントを実行:
   - Tkinter GUI: `python clients/tkinter/main.py`
   - gRPC サーバー: `python engine/server.py`

## ディレクトリ構成
```
assets/                 — 加工済みデータセット (KanjiVG など)
clients/tkinter/        — デスクトップ UI
clients/grpc/           — gRPC クライアント
engine/                 — 正規化・特徴量・マッチャー
protos/recognition.proto — gRPC インターフェース
```

## リンク
- GitHub: https://github.com/3ShishkaN3/kanji_pad
- GitVerse: https://gitverse.ru/shish/kanji_pad

## ライセンスとクレジット
本プロジェクトは Apache License 2.0 を採用しています。いかなる公開利用（Web、アプリ、ドキュメント、パッケージ等）でも、作者 Reznik Danil Maksimovich ("Shishka") を明示的にクレジットする必要があります。詳細は LICENSE と NOTICE を参照してください。

</details>
