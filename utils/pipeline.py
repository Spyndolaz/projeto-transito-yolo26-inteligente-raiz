from collections import Counter
import cv2
import numpy as np

from config.config import (CONFIDENCE, COUNTING_LINES, STOP_THRESHOLD, STOP_TIME, TRACKER, TRAIL_LENGTH,
                           LEARNING_ENABLED, LOW_CONFIDENCE_THRESHOLD, INTERMEDIATE_CONFIDENCE_THRESHOLD,
                           MIN_LEARNING_INTERVAL, MAX_LEARNING_CASES, MODEL_VERSION)
from agent.agent import TrafficAgent
from agent.memory import AgentMemory
from learning.collector import CaseCollector
from learning.sampler import LearningSampler
from traffic.events import EventLogger
from utils.counting import LineCounter
from utils.tracking import TrackHistory
from utils.traffic import calculate_congestion


class TrafficProcessor:
    def __init__(self, model, lines=COUNTING_LINES):
        self.model = model
        self.history = TrackHistory(TRAIL_LENGTH, STOP_THRESHOLD, STOP_TIME)
        self.counter = LineCounter(lines)
        self.lines = lines
        self.events = EventLogger()
        self.memory = AgentMemory()
        sampler = LearningSampler(LOW_CONFIDENCE_THRESHOLD, INTERMEDIATE_CONFIDENCE_THRESHOLD,
                                  MIN_LEARNING_INTERVAL, MAX_LEARNING_CASES)
        collector = CaseCollector(sampler, events=self.events) if LEARNING_ENABLED else None
        self.agent = TrafficAgent(self.events, self.memory, collector, LOW_CONFIDENCE_THRESHOLD)
        self.seen_ids, self.statuses = set(), {}

    def process(self, frame):
        result = self.model.track(frame, persist=True, tracker=TRACKER, conf=CONFIDENCE, verbose=False)[0]
        visible, stopped = Counter(), 0
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            ids = result.boxes.id.int().cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            for box, track_id, class_id, confidence in zip(boxes, ids, classes, confidences):
                name = result.names[class_id]
                x1, y1, x2, y2 = box
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                status = self.history.update(track_id, center)
                crossing = self.counter.update(track_id, name, center)
                detection = {"frame_id": int(result.path) if str(result.path).isdigit() else None,
                             "track_id": track_id, "class_id": class_id, "class_name": name,
                             "confidence": round(float(confidence), 4), "bbox": [int(x1), int(y1), int(x2), int(y2)],
                             "center": [int(center[0]), int(center[1])], "status": status, "model_version": MODEL_VERSION}
                if track_id not in self.seen_ids:
                    self.seen_ids.add(track_id)
                    self.events.emit("VEHICLE_DETECTED", **detection)
                    self.events.emit("VEHICLE_ENTERED", **detection)
                if crossing:
                    self.events.emit("VEHICLE_EXITED", line=crossing, **detection)
                if self.statuses.get(track_id) != status:
                    self.statuses[track_id] = status
                    self.events.emit("VEHICLE_STOPPED" if status == "PARADO" else "VEHICLE_MOVING", **detection)
                visible[name] += 1
                stopped += status == "PARADO"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(frame, f"{name} #{track_id} {confidence:.2f} {status}", (x1, max(22, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 0), 2)
                trail = self.history.trail(track_id)
                if len(trail) > 1:
                    cv2.polylines(frame, [np.array(trail, np.int32)], False, (0, 180, 255), 2)
        for name, (a, b) in self.lines.items():
            cv2.line(frame, a, b, (255, 100, 0), 2)
            cv2.putText(frame, name.upper(), a, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
        congestion = calculate_congestion(sum(visible.values()), stopped_vehicles=stopped,
                                          flow_per_minute=self.counter.flow_per_minute())
        # O agente vê registros estruturados após o cálculo de trânsito.
        if result.boxes is not None and result.boxes.id is not None:
            for box, track_id, class_id, confidence in zip(boxes, ids, classes, confidences):
                name = result.names[class_id]
                x1, y1, x2, y2 = box
                self.agent.observe(frame, {"track_id": track_id, "class_id": class_id, "class_name": name,
                                           "confidence": float(confidence), "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                           "center": [(x1+x2)//2, (y1+y2)//2], "status": self.statuses.get(track_id)}, congestion)
        self._draw_stats(frame, visible, stopped, congestion)
        return frame

    def _draw_stats(self, frame, visible, stopped, congestion):
        lines = ["CAMERA INTELIGENTE DE TRANSITO", f"Modelo: YOLO26 {MODEL_VERSION} | Agente: ATIVO",
                 f"Aprendizado: {'ATIVO' if LEARNING_ENABLED else 'DESLIGADO'}", f"Objetos atuais: {sum(visible.values())}"]
        lines += [f"{name.upper()}: {visible.get(name, 0)}" for name in ("car", "bike", "Big car", "people")]
        lines += [f"Entrada: {self.counter.totals['entrada']}  Saida: {self.counter.totals['saida']}",
                  f"Fluxo/min: {self.counter.flow_per_minute()}  Parados: {stopped}",
                  f"Congestionamento: {congestion['level']} (estimativa)", self.agent.explanation()[:75]]
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (14, 28 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)


def run_source(model, source, window_name="Camera Inteligente de Transito"):
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a fonte: {source}")
    processor = TrafficProcessor(model)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        cv2.imshow(window_name, processor.process(frame))
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break
    capture.release()
    cv2.destroyAllWindows()
