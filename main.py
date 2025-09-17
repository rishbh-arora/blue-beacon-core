

import io, os, re
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parent

CKPT_PATH = Path(os.getenv("CKPT_PATH", BASE / "best (1).pt"))
DEVICE = "cpu"  


EFFECT_CARDS: Dict[str, str] = {
    "oil_sheen": "oil sheen or petroleum slick with rainbow film or black tar on water or shoreline",
    "vessel_wreckage_visible": "visible ship or boat wreckage such as grounded or sunken vessel or torn hull",
    "debris_on_beach": "storm-driven debris or litter piles collected on the beach",
    "exposed_seabed": "sea has receded exposing unusually wide seabed or reef flats",
    "wave_overtopping": "seawater crossing or overtopping steps, seawalls or promenade",
    "high_surf_scene": "large breaking waves with heavy whitewater near the shore",
    "flooded_area": "standing seawater inundation on streets or yards near the coast",
    "structural_damage": "fresh damage to buildings or coastal structures like broken roofs or walls",
}
CALAMITY_CARDS: Dict[str, str] = {
    "oil_spill": "oil spill with rainbow sheen, black slick, or tar balls on water or shoreline",
    "ship_boat_wreckage": "ship or boat wreckage such as grounded or sunken vessel with torn hull and marine debris",
    "debris": "storm-driven debris or litter piles on the beach",
    "tsunami": "tsunami effects like rapid sea drawback exposing seabed or seawater inundation rushing inland",
    "high_tides": "unusually high tide with seawater overtopping steps or promenade without storm surf look",
    "cyclones": "cyclone conditions with extreme surf or storm surge and coastal damage",
    "floods": "coastal flooding with standing seawater on streets or yards near the coast",
    "none": "no calamity is present",
}


FALLBACK_THR = {"tsunami": 0.75, "ship_boat_wreckage": 0.70}

def norm(lbl: str) -> str:
    return str(lbl).strip().lower().replace("-", "_")


CUESETS = {
    "tsunami": [
        r"\bdrawback\b", r"\bsea\s+reced(ed|ing)\b", r"\bexposed\s+seabed\b",
        r"\bwater\s+rushed\s+inland\b", r"\binundation\b", r"\bmultiple\s+surges?\b",
        r"\bseawater\s+cross(ed|ing)\s+(road|street|promenade|seawall)\b",
    ],
    "ship_boat_wreckage": [
        r"\bwreck(age)?\b", r"\bgrounded\b", r"\bsunken\b", r"\bhull\s+(torn|broken)\b",
        r"\bmayday\b", r"\bdistress\b", r"\b(yacht|boat|ship)\b.*\b(aground|broken|damaged)\b",
    ],
}
def cue_boost(desc: str, label: str) -> float:
    text = (desc or "").lower()
    pats = CUESETS.get(label, [])
    hits = sum(1 for pat in pats if re.search(pat, text))
    return min(0.12, [0.0, 0.06, 0.10, 0.12][min(hits, 3)])

class MultiTaskEffB0(nn.Module):
    def __init__(self, n_effects: int, n_calam: int):
        super().__init__()
        try:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        except Exception:
            weights = None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_feat = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.head_effects = nn.Linear(in_feat, n_effects)
        self.head_calam = nn.Linear(in_feat, n_calam)

    def forward(self, x):
        feats = self.backbone.features(x)
        feats = self.pool(feats).flatten(1)
        feats = self.dropout(feats)
        return self.head_effects(feats), self.head_calam(feats)


app = FastAPI(title="Crowdsourced Ocean Hazard Verifier", version="1.0")

MODEL = None
EFFECT_LABELS = None
CALAMITY_LABELS = None
IMG_SIZE = 256
TFM = None
EMBEDDER = None  

@app.on_event("startup")
def load_everything():
    global MODEL, EFFECT_LABELS, CALAMITY_LABELS, IMG_SIZE, TFM, EMBEDDER

    if not CKPT_PATH.exists():
        raise RuntimeError(f"Checkpoint not found: {CKPT_PATH}")

    ckpt = torch.load(str(CKPT_PATH), map_location=DEVICE)
    EFFECT_LABELS = ckpt["effect_classes"]
    CALAMITY_LABELS = ckpt["calamity_classes"]
    IMG_SIZE = ckpt.get("img_size", 256)

    
    for lbl in EFFECT_LABELS:
        EFFECT_CARDS.setdefault(lbl, f"visual evidence of {lbl.replace('_',' ')}")
    for lbl in CALAMITY_LABELS:
        CALAMITY_CARDS.setdefault(lbl, f"calamity: {lbl.replace('_',' ')}")

    MODEL = MultiTaskEffB0(n_effects=len(EFFECT_LABELS), n_calam=len(CALAMITY_LABELS)).to(DEVICE)
    MODEL.load_state_dict(ckpt["state_dict"])
    MODEL.eval()

    global TFM
    TFM = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    
    global EMBEDDER
    EMBEDDER = SentenceTransformer("l3cube-pune/indic-sentence-similarity-sbert", device=DEVICE)
    _ = EMBEDDER.encode(["ok"], normalize_embeddings=True)  # warm-up

@app.get("/")
def root():
    return {"message": "OK. Visit /docs for Swagger UI."}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "ckpt": str(CKPT_PATH),
        "n_effects": len(EFFECT_LABELS) if EFFECT_LABELS else None,
        "n_calamities": len(CALAMITY_LABELS) if CALAMITY_LABELS else None,
        "img_size": IMG_SIZE,
    }

def text_scores(description: str, cards: Dict[str, str]) -> Dict[str, float]:
    """Return {label: score in [0,1]} via cosine(sim(description, card_text))."""
    labels = list(cards.keys())
    if not description or not description.strip():
        return {k: 0.0 for k in labels}
    texts = [description] + [cards[k] for k in labels]
    embs = EMBEDDER.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    v_desc = embs[0:1]
    v_lbls = embs[1:]
    sims = (v_desc @ v_lbls.T).ravel()  # cosine in [-1,1]
    return {labels[i]: float(0.5 * (sims[i] + 1.0)) for i in range(len(labels))}

def predict_one(pil_image: Image.Image, description: str) -> Dict:
    # CNN forward
    x = TFM(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outE, outC = MODEL(x)
        pE = torch.sigmoid(outE).squeeze(0).cpu().numpy()
        pC = torch.sigmoid(outC).squeeze(0).cpu().numpy()

    # top-1 from each head
    e_idx = int(np.argmax(pE)); e_top, p_e = EFFECT_LABELS[e_idx], float(pE[e_idx])
    c_idx = int(np.argmax(pC)); c_top, p_c = CALAMITY_LABELS[c_idx], float(pC[c_idx])
    avg_top_img_score = round((p_e + p_c) / 2.0, 3)

    # text support for those two
    p_txt_e = text_scores(description, {e_top: EFFECT_CARDS[e_top]}).get(e_top, 0.0)
    p_txt_c = text_scores(description, {c_top: CALAMITY_CARDS[c_top]}).get(c_top, 0.0)

    # semantic fallback if any head predicted "none"
    def semantic_on_none(desc: str) -> Dict:
        if norm(e_top) != "none" and norm(c_top) != "none":
            return {"triggered": False, "verdict": "skipped"}
        subset = {
            "tsunami": CALAMITY_CARDS["tsunami"],
            "ship_boat_wreckage": CALAMITY_CARDS["ship_boat_wreckage"],
        }
        base = text_scores(desc, subset)
        sc_tsu = min(1.0, base.get("tsunami", 0.0) + cue_boost(desc, "tsunami"))
        sc_wrk = min(1.0, base.get("ship_boat_wreckage", 0.0) + cue_boost(desc, "ship_boat_wreckage"))

        passes = []
        if sc_tsu >= FALLBACK_THR["tsunami"]: passes.append(("tsunami", sc_tsu))
        if sc_wrk >= FALLBACK_THR["ship_boat_wreckage"]: passes.append(("ship_boat_wreckage", sc_wrk))
        if passes:
            chosen = max(passes, key=lambda t: t[1])
            return {
                "triggered": True,
                "p_txt": {"tsunami": round(sc_tsu, 3), "ship_boat_wreckage": round(sc_wrk, 3)},
                "thresholds": FALLBACK_THR,
                "verdict": chosen[0],
                "flag_analyst": False,
            }
        else:
            return {
                "triggered": True,
                "p_txt": {"tsunami": round(sc_tsu, 3), "ship_boat_wreckage": round(sc_wrk, 3)},
                "thresholds": FALLBACK_THR,
                "verdict": "flag_analyst",
                "flag_analyst": True,
            }

    sem = semantic_on_none(description)

    return {
        "effect_top":   {"label": e_top, "p_img": round(p_e, 3), "p_txt_support": round(p_txt_e, 3)},
        "calamity_top": {"label": c_top, "p_img": round(p_c, 3), "p_txt_support": round(p_txt_c, 3)},
        "avg_top_img_score": round(avg_top_img_score, 3),
        "semantic_on_none": sem,
    }

@app.post("/predict")
async def predict(
    image: UploadFile = File(..., description="Image file (jpg/png)"),
    description: str = Form(..., description="Short text description (any Indic/English)")
):
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    content = await image.read()
    try:
        pil = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to read image.")

    try:
        result = predict_one(pil, description)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
