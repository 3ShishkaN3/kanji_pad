import pickle
import numpy as np
from scipy.optimize import linear_sum_assignment
from .data_models import NormalizedKanji, RecognitionResult

POINTS_PER_STROKE = 32
MAX_STROKES_IN_DB = 35 

W_SHAPE = 1.0     
W_POSITION = 2.5   
W_SIZE = 1.5       

# Штраф за каждый лишний штрих в кандидате (для режима предикшна)
STROKE_COUNT_PENALTY = 0.15 

class Matcher:
    def __init__(self, database_path: str):
        self.database: dict[str, NormalizedKanji] = self._load_database(database_path)
        
        # Кэши для быстрого доступа
        self.db_tensor = None      # (N, MaxS, 32, 2) - геометрия
        self.db_features = None    # (N, MaxS, 4) - фичи: [CentroidX, CentroidY, Length]
        self.db_chars = []
        self.db_stroke_counts = []
        
        self._build_internal_caches()
        print(f"Matcher initialized with {len(self.database)} kanji entries. (High-Accuracy Mode)")

    def _load_database(self, path: str):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Database file not found at '{path}'. Please run the preprocessor first.")
        except Exception as e:
            raise RuntimeError(f"Failed to load or parse database file '{path}': {e}")

    def recognize(self, user_drawing: NormalizedKanji, top_n: int = 5, predictive_mode: bool = False) -> list[RecognitionResult]:
        """
        Основной метод распознавания.

        Args:
            user_drawing: Объект NormalizedKanji от пользователя.
            top_n: Количество результатов.
            predictive_mode: Если False (по умолчанию), сравнивает только с иероглифами
                             с таким же числом штрихов. Если True, ищет среди более сложных.
        """
        user_tensor, user_features, user_count = self._preprocess_user_input(user_drawing)
        if user_count == 0: return []

        if predictive_mode:
            mask = (self.db_stroke_counts >= user_count)
        else:
            mask = (self.db_stroke_counts == user_count)
            
        candidate_indices = np.where(mask)[0]
        if len(candidate_indices) == 0: return []

        scores = []

        for global_idx in candidate_indices:
            db_count = self.db_stroke_counts[global_idx]
            
            cost = self._calculate_distance(user_tensor, user_features, global_idx)
            
            if predictive_mode:
                stroke_diff = db_count - user_count
                cost *= (1 + stroke_diff * STROKE_COUNT_PENALTY)
                
            scores.append((global_idx, cost))

        scores.sort(key=lambda item: item[1])
        
        results = []
        if not scores: return []
        
        min_distance = scores[0][1] + 1e-6

        for glob_idx, dist in scores[:top_n]:
            char = self.db_chars[glob_idx]
            
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

    def _calculate_distance(self, u_tensor, u_features, db_index):
        """
        Воспроизводит вашу логику cost_matrix + linear_sum_assignment.
        """
        db_count = self.db_stroke_counts[db_index]
        u_count = len(u_tensor)
        
        if u_count > db_count:
            return float('inf')

        db_tensor = self.db_tensor[db_index][:db_count]
        db_features = self.db_features[db_index][:db_count]

        diff = u_tensor[:, None, :, :] - db_tensor[None, :, :, :]
        dist_shape = np.sum(np.linalg.norm(diff, axis=3), axis=2)
        diff_rev = u_tensor[:, None, :, :] - db_tensor[None, :, ::-1, :]
        dist_shape_rev = np.sum(np.linalg.norm(diff_rev, axis=3), axis=2)
        final_dist_shape = np.minimum(dist_shape, dist_shape_rev)

        u_pos = u_features[:, :2]
        db_pos = db_features[:, :2]
        dist_pos = np.linalg.norm(u_pos[:, None, :] - db_pos[None, :, :], axis=2)

        u_len = u_features[:, 2]
        db_len = db_features[:, 2]
        dist_size = np.abs(u_len[:, None] - db_len[None, :])

        cost_matrix = (
            (final_dist_shape * W_SHAPE) +
            (dist_pos * W_POSITION) +
            (dist_size * W_SIZE)
        )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_distance = cost_matrix[row_ind, col_ind].sum()

        return total_distance / u_count

    # Методы подготовки данных
    def _build_internal_caches(self):
        tensor_list, feat_list, chars_list, counts_list = [], [], [], []
        for char, kanji in self.database.items():
            raw_strokes = [np.array(s, dtype=np.float32) for s in kanji.normalized_strokes]
            if not raw_strokes: continue
            norm_strokes = self._normalize_kanji_geometry(raw_strokes)
            padded_geo = np.full((MAX_STROKES_IN_DB, POINTS_PER_STROKE, 2), np.nan, dtype=np.float32)
            padded_feat = np.full((MAX_STROKES_IN_DB, 3), np.nan, dtype=np.float32)
            for i, stroke in enumerate(norm_strokes[:MAX_STROKES_IN_DB]):
                padded_geo[i] = stroke
                centroid = np.mean(stroke, axis=0)
                length = np.sum(np.linalg.norm(stroke[1:] - stroke[:-1], axis=1))
                padded_feat[i] = [centroid[0], centroid[1], length]
            tensor_list.append(padded_geo)
            feat_list.append(padded_feat)
            chars_list.append(char)
            counts_list.append(len(norm_strokes))
        self.db_tensor = np.array(tensor_list)
        self.db_features = np.array(feat_list)
        self.db_chars = np.array(chars_list)
        self.db_stroke_counts = np.array(counts_list, dtype=np.int32)

    def _preprocess_user_input(self, user_drawing):
        raw_strokes = [np.array(s, dtype=np.float32) for s in user_drawing.normalized_strokes]
        if not raw_strokes: return None, None, 0
        norm_strokes = self._normalize_kanji_geometry(raw_strokes)
        feats = []
        for s in norm_strokes:
            centroid = np.mean(s, axis=0)
            length = np.sum(np.linalg.norm(s[1:] - s[:-1], axis=1))
            feats.append([centroid[0], centroid[1], length])
        return np.array(norm_strokes), np.array(feats), len(norm_strokes)

    def _normalize_kanji_geometry(self, strokes):
        resampled = [self._resample_stroke(s) for s in strokes]
        all_points = np.vstack(resampled)
        min_xy = all_points.min(axis=0)
        max_xy = all_points.max(axis=0)
        max_dim = np.max(max_xy - min_xy)
        if max_dim == 0: max_dim = 1.0
        return [(s - min_xy) / max_dim for s in resampled]

    def _resample_stroke(self, stroke, n=POINTS_PER_STROKE):
        if len(stroke) == n: return stroke
        dists = np.sqrt(np.sum(np.diff(stroke, axis=0)**2, axis=1))
        cum_dist = np.insert(np.cumsum(dists), 0, 0)
        total_len = cum_dist[-1]
        if total_len == 0: return np.tile(stroke[0], (n, 1))
        t = np.linspace(0, total_len, n)
        x = np.interp(t, cum_dist, stroke[:, 0])
        y = np.interp(t, cum_dist, stroke[:, 1])
        return np.stack((x, y), axis=1)