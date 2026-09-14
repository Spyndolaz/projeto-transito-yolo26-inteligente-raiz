"""Ajuste estes valores para a câmera e a maquete reais."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = ROOT_DIR / "models" / "production"
# Após promoção, a câmera usa a versão oficial; antes do primeiro ciclo preserva o peso legado.
MODEL_PATH = next(iter(sorted(PRODUCTION_DIR.glob("*.pt"))), ROOT_DIR / "runs" / "transito" / "weights" / "best.pt")
DATA_YAML = ROOT_DIR / "data" / "data.yaml"
RUNS_DIR = ROOT_DIR / "runs"

CAMERA_ID = 0
CONFIDENCE = 0.35
IOU_THRESHOLD = 0.45
TRACKER = "bytetrack.yaml"
TRAIL_LENGTH = 40
STOP_THRESHOLD = 3.0  # pixels entre observações; calibrar para a maquete
STOP_TIME = 3.0       # segundos quase imóvel antes de marcar PARADO

# Linhas e zonas de EXEMPLO. Calibre para a resolução/posição final da câmera.
COUNTING_LINES = {
    "entrada": ((100, 300), (540, 300)),
    "saida": ((100, 500), (540, 500)),
}
ZONES = {
    "ZONA 1": [(100, 100), (300, 100), (300, 280), (100, 280)],
    "ZONA 2": [(320, 100), (540, 100), (540, 280), (320, 280)],
    "ZONA 3": [(100, 320), (540, 320), (540, 600), (100, 600)],
}

# None impede converter pixel/s em km/h sem calibração física.
CALIBRATION_DISTANCE_METERS = None
CALIBRATION_DISTANCE_PIXELS = None

# Aprendizado contínuo controlado. Nenhuma previsão é usada como rótulo sem revisão humana.
LEARNING_ENABLED = True
LOW_CONFIDENCE_THRESHOLD = 0.50
INTERMEDIATE_CONFIDENCE_THRESHOLD = 0.65
MIN_LEARNING_INTERVAL = 5.0
MAX_LEARNING_CASES = 500
MODEL_AUTO_TRAIN = False
MODEL_AUTO_PROMOTE = False
MIN_MODEL_IMPROVEMENT = 0.01  # melhoria mínima de mAP50-95
MODEL_VERSION = "v001"
