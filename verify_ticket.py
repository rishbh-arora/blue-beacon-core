# verify_ticket_standalone.py  — CPU-only, Windows-friendly
# Uses SentenceTransformer "l3cube-pune/indic-sentence-similarity-sbert" for multilingual text scoring.

import json, re, os
from typing import Dict, List
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from sentence_transformers import SentenceTransformer, util  # <— swapped in


TF_ENABLE_ONEDNN_OPTS=0
# ========= CONFIG: EDIT THESE FOUR LINES =========
CKPT_PATH = r"best (1).pt"                  # <- your best.pt
IMG_PATH  = r"oilspill_1.jpg"               # <- an image to score
DESC      = "Black liquid in ocean"         # <- user description (any Indic/English)
OUT_JSON  = r""                             # <- optional output path, or "" to skip
# ========= END CONFIG =========

DEVICE = "cpu"  # force CPU

def norm(lbl: str) -> str:
    return str(lbl).strip().lower().replace("-", "_")

# Cards for text similarity (edit wording if you like)
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

# thresholds for semantic fallback (embeddings mapped to [0,1])
FALLBACK_THR = {"tsunami": 0.75, "ship_boat_wreckage": 0.70}

# ---------------- CNN model (same architecture as training) ----------------
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
        self.head_calam   = nn.Linear(in_feat, n_calam)
    def forward(self, x):
        feats = self.backbone.features(x)
        feats = self.pool(feats).flatten(1)
        feats = self.dropout(feats)
        return self.head_effects(feats), self.head_calam(feats)

# --------------- IndicSBERT scorer (multilingual sentence similarity) ---------------
class IndicSBERTScorer:
    def __init__(self, model_id: str = "l3cube-pune/indic-sentence-similarity-sbert", device: str = "cpu"):
        self.embedder = SentenceTransformer(model_id, device=device)

    @torch.no_grad()
    def scores(self, description: str, cards: Dict[str, str]) -> Dict[str, float]:
        labels = list(cards.keys())
        if not description or not description.strip():
            return {k: 0.0 for k in labels}
        texts = [description] + [cards[k] for k in labels]
        embs = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        v_desc = embs[0:1]         # (1, D)
        v_lbls = embs[1:]          # (L, D)
        # cosine in [-1,1] -> map to [0,1]
        sims = (v_desc @ v_lbls.T).ravel()
        return {labels[i]: float(0.5 * (sims[i] + 1.0)) for i in range(len(labels))}

# tiny deterministic cue boost for keywords
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

# ---------------- core pipeline ----------------
def run_ticket(ckpt_path: str, img_path: str, description: str) -> Dict:
    # exist checks
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"best.pt not found: {ckpt_path}")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"image not found: {img_path}")

    # load checkpoint + labels
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    eff_labels = ckpt["effect_classes"]
    cal_labels = ckpt["calamity_classes"]
    img_size   = ckpt.get("img_size", 256)

    # build CNN
    model = MultiTaskEffB0(n_effects=len(eff_labels), n_calam=len(cal_labels)).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # preprocess + predict
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outE, outC = model(x)
        pE = torch.sigmoid(outE).squeeze(0).cpu().numpy()
        pC = torch.sigmoid(outC).squeeze(0).cpu().numpy()

    # top-1 from each head
    e_idx = int(np.argmax(pE)); e_top, p_e = eff_labels[e_idx], float(pE[e_idx])
    c_idx = int(np.argmax(pC)); c_top, p_c = cal_labels[c_idx], float(pC[c_idx])
    avg_top_img_score = round((p_e + p_c) / 2.0, 3)

    # ensure cards cover any custom labels
    for lbl in eff_labels: EFFECT_CARDS.setdefault(lbl, f"visual evidence of {lbl.replace('_',' ')}")
    for lbl in cal_labels: CALAMITY_CARDS.setdefault(lbl, f"calamity: {lbl.replace('_',' ')}")

    # text support via IndicSBERT (multilingual)
    scorer = IndicSBERTScorer(device=DEVICE)
    p_txt_e = scorer.scores(description, {e_top: EFFECT_CARDS[e_top]}).get(e_top, 0.0)
    p_txt_c = scorer.scores(description, {c_top: CALAMITY_CARDS[c_top]}).get(c_top, 0.0)

    # semantic fallback if any head predicts "none"
    def semantic_on_none(desc: str) -> Dict:
        if norm(e_top) != "none" and norm(c_top) != "none":
            return {"triggered": False, "verdict": "skipped"}
        subset = {
            "tsunami": CALAMITY_CARDS["tsunami"],
            "ship_boat_wreckage": CALAMITY_CARDS["ship_boat_wreckage"],
        }
        base = scorer.scores(desc, subset)
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

    sem = semantic_on_none(DESC)

    return {
        "effect_top":   {"label": e_top, "p_img": round(p_e, 3), "p_txt_support": round(p_txt_e, 3)},
        "calamity_top": {"label": c_top, "p_img": round(p_c, 3), "p_txt_support": round(p_txt_c, 3)},
        "avg_top_img_score": avg_top_img_score,
        "semantic_on_none": sem,
    }

# --------- run with the CONFIG above ---------
if __name__ == "__main__":
    out = run_ticket(CKPT_PATH, IMG_PATH, DESC)
    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if OUT_JSON:
        os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            f.write(text)
