import os

SEED_TRAIN = 42
SEED_TEST = 99

MIN_PRESSURE_M = 7.03       # 10 PSI
REQUIRED_PRESSURE_M = 14.06 # 20 PSI
HW_SAFETY_MARGIN_M = 1.0
GAT_Q_ESTIMATE_M3S = 0.005
ESCALATION_THRESHOLD = 0.6
DEFAULT_K = 2
N_SCENARIOS = 500
N_BASELINE_RUNS = 100
Z_THRESHOLD = 3.0
# Planning-level pipe capacity used by HARP's residual graph. A pipe can carry
# the Hazen-Williams flow for whichever is larger: the head difference actually
# predicted across it, or this design hydraulic gradient (m of head per m of
# pipe; 0.01 = 10 m/km, a common distribution-main design limit). Without the
# design floor, near-equilibrium networks (adjacent nodes at ~equal head) get
# ~zero capacity everywhere. Also capped by a maximum design velocity.
HW_DESIGN_GRADIENT = 0.01
MAX_DESIGN_VELOCITY_MS = 2.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
NETWORKS_DIR = os.path.join(DATA_DIR, "networks")
# Output locations can be redirected with env vars (e.g. to Google Drive when
# training on Colab). Env vars are inherited by multiprocessing workers, so
# parallel scenario generation writes to the same place as the parent.
GENERATED_DIR = os.environ.get("HARP_GENERATED_DIR") or os.path.join(DATA_DIR, "generated")
MODELS_DIR = os.environ.get("HARP_MODELS_DIR") or os.path.join(DATA_DIR, "trained_models")
GEO_DIR = os.path.join(DATA_DIR, "geo")
RESULTS_DIR = os.environ.get("HARP_RESULTS_DIR") or os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(GENERATED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(GEO_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

NETWORK_FILES = {
    "Net3": os.path.join(NETWORKS_DIR, "Net3.inp"),
    "Hanoi": os.path.join(NETWORKS_DIR, "Hanoi.inp"),
    "L-Town": os.path.join(NETWORKS_DIR, "L-Town.inp"),
}

PRIORITY_WEIGHTS = {
    "hospital": 5.0,
    "fire_station": 4.0,
    "school": 3.0,
    "residential": 1.0,
    "park": 0.5,
}
