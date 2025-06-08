import grpc
from concurrent import futures
import time

import recognition_pb2
import recognition_pb2_grpc

from engine.matcher import Matcher
from engine.data_models import NormalizedKanji, StrokeFeatures

DATABASE_PATH = 'assets/kanjivg_normalized.pkl'

class RecognitionServicer(recognition_pb2_grpc.RecognitionServiceServicer):
    """
    Класс-реализация нашего gRPC сервиса.
    Он связывает gRPC запросы с логикой Matcher.
    """
    def __init__(self):
        self.matcher = Matcher(DATABASE_PATH)
        print("gRPC servicer initialized with a Matcher instance.")

    def Recognize(self, request, context):
        """Обрабатывает входящий запрос на распознавание."""
        
        try:
            user_drawing = self._convert_request_to_kanji(request)
        except Exception as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f'Failed to parse request: {e}')
            return recognition_pb2.RecognitionResponse()

        results = self.matcher.recognize(user_drawing, top_n=request.top_n)

        response = recognition_pb2.RecognitionResponse()
        for res in results:
            response.results.add(
                character=res.character,
                distance=res.distance,
                confidence=res.confidence
            )
        
        return response

    def _convert_request_to_kanji(self, request: recognition_pb2.RecognitionRequest) -> NormalizedKanji:
        """Вспомогательный метод для конвертации типов."""
        strokes = [[(p.x, p.y) for p in s.points] for s in request.normalized_strokes]
        
        features = [
            StrokeFeatures(
                bounding_box=((f.bounding_box.min.x, f.bounding_box.min.y), (f.bounding_box.max.x, f.bounding_box.max.y)),
                start_point=(f.start_point.x, f.start_point.y),
                end_point=(f.end_point.x, f.end_point.y),
                centroid=(f.centroid.x, f.centroid.y),
                length=f.length
            ) for f in request.stroke_features
        ]
        
        g_box = ((request.global_bounding_box.min.x, request.global_bounding_box.min.y), 
                 (request.global_bounding_box.max.x, request.global_bounding_box.max.y))

        g_centroid = (request.global_centroid.x, request.global_centroid.y)

        return NormalizedKanji(
            character=None,
            normalized_strokes=strokes,
            stroke_features=features,
            global_bounding_box=g_box,
            global_centroid=g_centroid,
            source_component_tree=None
        )

def serve():
    """Запускает gRPC сервер."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    recognition_pb2_grpc.add_RecognitionServiceServicer_to_server(
        RecognitionServicer(), server
    )
    server.add_insecure_port('[::]:50051')
    print("Starting gRPC server on port 50051...")
    server.start()
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        print("Stopping server...")
        server.stop(0)

if __name__ == '__main__':
    serve()