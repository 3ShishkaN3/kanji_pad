import grpc
import pickle
import sys, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

import recognition_pb2
import recognition_pb2_grpc
from engine.data_models import NormalizedKanji

def run_client(kanji_to_test: NormalizedKanji):
    """Запускает gRPC клиент и отправляет запрос на распознавание."""
    print("--- gRPC Client ---")
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = recognition_pb2_grpc.RecognitionServiceStub(channel)
        
        request = recognition_pb2.RecognitionRequest(top_n=5)
        
        for stroke in kanji_to_test.normalized_strokes:
            grpc_stroke = request.normalized_strokes.add()
            for p in stroke:
                grpc_stroke.points.add(x=p[0], y=p[1])

        for f in kanji_to_test.stroke_features:
            grpc_features = request.stroke_features.add()
            grpc_features.length = f.length
            grpc_features.start_point.x, grpc_features.start_point.y = f.start_point
            grpc_features.end_point.x, grpc_features.end_point.y = f.end_point
            grpc_features.centroid.x, grpc_features.centroid.y = f.centroid
            (min_x, min_y), (max_x, max_y) = f.bounding_box
            grpc_features.bounding_box.min.x, grpc_features.bounding_box.min.y = min_x, min_y
            grpc_features.bounding_box.max.x, grpc_features.bounding_box.max.y = max_x, max_y

        (min_x, min_y), (max_x, max_y) = kanji_to_test.global_bounding_box
        request.global_bounding_box.min.x, request.global_bounding_box.min.y = min_x, min_y
        request.global_bounding_box.max.x, request.global_bounding_box.max.y = max_x, max_y
        request.global_centroid.x, request.global_centroid.y = kanji_to_test.global_centroid

        print(f"Sending request to recognize kanji '{kanji_to_test.character}'...")
        response = stub.Recognize(request)

        print("\n--- Recognition Results ---")
        if not response.results:
            print("No results returned.")
        for res in response.results:
            print(
                f"Character: {res.character}\t "
                f"Distance: {res.distance:.2f}\t "
                f"Confidence: {res.confidence:.2%}"
            )

if __name__ == '__main__':
    db_path = os.path.join(PROJECT_ROOT, 'assets', 'kanjivg_normalized.pkl')
    with open(db_path, 'rb') as f:
        db = pickle.load(f)
    
    test_kanji_char = '猫' 
    if test_kanji_char in db:
        run_client(db[test_kanji_char])
    else:
        print(f"Кандзи '{test_kanji_char}' не найден в базе данных.")