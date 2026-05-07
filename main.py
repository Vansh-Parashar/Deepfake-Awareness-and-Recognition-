from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline, AutoConfig
from PIL import Image, ImageChops, ImageFilter
import uvicorn
import io
import numpy as np
import cv2
import os
import tempfile
import scipy.ndimage as ndimage

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# THREE-MODEL ENSEMBLE
# Model 1: umm-maybe  — general AI detector
# Model 2: Organika   — SDXL/modern AI detector  
# Model 3: haywoodsloan — newest, trained on Flux/MJ v6 images
# ─────────────────────────────────────────────────────────────
MODELS = [
    {
        "name": "umm-maybe/AI-image-detector",
        "weight": 0.25,
        "ai_keywords":    {"artificial", "ai", "fake", "generated", "synthetic"},
        "human_keywords": {"human", "real", "authentic", "natural"},
    },
    {
        "name": "Organika/sdxl-detector",
        "weight": 0.35,
        "ai_keywords":    {"artificial", "ai", "fake", "generated", "sdxl", "synthetic"},
        "human_keywords": {"human", "real", "authentic", "natural"},
    },
    {
        "name": "haywoodsloan/ai-image-detector",
        "weight": 0.40,   # highest weight — newest training data
        "ai_keywords":    {"artificial", "ai", "fake", "generated", "synthetic"},
        "human_keywords": {"human", "real", "authentic", "natural"},
    },
]

print("Loading models...")
classifiers = []
for m in MODELS:
    try:
        config = AutoConfig.from_pretrained(m["name"])
        clf = pipeline("image-classification", model=m["name"])
        ai_id, human_id = 0, 1
        for idx, label in config.id2label.items():
            ll = label.lower()
            if any(kw in ll for kw in m["ai_keywords"]):
                ai_id = idx
            elif any(kw in ll for kw in m["human_keywords"]):
                human_id = idx
        classifiers.append({
            "clf": clf, "weight": m["weight"], "name": m["name"],
            "ai_label_id": ai_id, "human_label_id": human_id,
            "ai_keywords": m["ai_keywords"], "human_keywords": m["human_keywords"],
        })
        print(f"  ✓ Loaded: {m['name']}")
    except Exception as e:
        print(f"  ✗ Failed: {m['name']} — {e}")

if not classifiers:
    raise RuntimeError("No models loaded.")
print(f"Engine ready — {len(classifiers)} model(s) loaded.")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ─────────────────────────────────────────────────────────────
# LABEL PARSER
# ─────────────────────────────────────────────────────────────
def parse_model_output(results: list, model_info: dict) -> tuple[float, float]:
    ai_score, human_score = 0.0, 0.0
    for r in results:
        ll = r["label"].lower()
        s  = float(r["score"])
        if any(kw in ll for kw in model_info["ai_keywords"] - {"label_0"}):
            ai_score = s
        elif any(kw in ll for kw in model_info["human_keywords"] - {"label_1"}):
            human_score = s
        elif ll == f"label_{model_info['ai_label_id']}":
            ai_score = s
        elif ll == f"label_{model_info['human_label_id']}":
            human_score = s
    if ai_score == 0.0 and human_score == 0.0:
        ai_score = human_score = 0.5
    return ai_score, human_score


# ─────────────────────────────────────────────────────────────
# RUN ALL MODELS ON ONE IMAGE
# ─────────────────────────────────────────────────────────────
def run_ensemble(img_pil: Image.Image) -> tuple[float, list]:
    img_r = img_pil.resize((224, 224), Image.Resampling.LANCZOS)
    weighted_ai, total_w, details = 0.0, 0.0, []
    for m in classifiers:
        try:
            raw = m["clf"](img_r)
            ai_s, hu_s = parse_model_output(raw, m)
            weighted_ai += ai_s * m["weight"]
            total_w     += m["weight"]
            details.append({
                "model": m["name"].split("/")[-1],
                "ai_score":    round(ai_s * 100, 1),
                "human_score": round(hu_s * 100, 1),
            })
            print(f"    [{m['name'].split('/')[-1]}] ai={ai_s:.3f}")
        except Exception as e:
            print(f"    Model error ({m['name']}): {e}")
    return (weighted_ai / total_w if total_w > 0 else 0.5), details


# ─────────────────────────────────────────────────────────────
# FACE EXTRACTOR
# ─────────────────────────────────────────────────────────────
def extract_face_crop(img_pil: Image.Image, padding: float = 0.30) -> Image.Image | None:
    try:
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        gray   = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces  = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        return img_pil.crop((
            max(0, x - pad_x), max(0, y - pad_y),
            min(img_pil.width,  x + w + pad_x),
            min(img_pil.height, y + h + pad_y),
        ))
    except:
        return None


# ─────────────────────────────────────────────────────────────
# SKIN TEXTURE ANALYSIS
# AI skin is unnaturally smooth — real skin has pores/micro-texture
# ─────────────────────────────────────────────────────────────
def analyze_skin_texture(face_crop: Image.Image) -> float:
    """
    Returns [0,1] where HIGH = more likely AI (too smooth).
    Measures high-frequency texture detail in the skin region.
    Real skin has micro-texture (pores, fine lines).
    AI skin is blurred/smooth at the pixel level.
    """
    try:
        gray  = np.array(face_crop.convert("L"), dtype=np.float32)
        gray  = cv2.resize(gray, (128, 128))

        # Laplacian measures sharpness / texture detail
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_var = np.var(laplacian)

        # Real skin: high texture variance (lots of fine detail)
        # AI skin:   low texture variance (smooth, blurred look)
        # Normalize: low variance → high AI suspicion
        ai_score = float(np.clip(1.0 - (texture_var / 300.0), 0.0, 1.0))
        print(f"    Skin texture variance: {texture_var:.1f} → ai_score: {ai_score:.3f}")
        return ai_score
    except Exception as e:
        print(f"    Skin texture error: {e}")
        return 0.5


# ─────────────────────────────────────────────────────────────
# EYE REGION ANALYSIS
# AI eyes have perfect, symmetric catchlights and unnatural iris patterns
# ─────────────────────────────────────────────────────────────
def analyze_eye_region(face_crop: Image.Image) -> float:
    """
    Returns [0,1] where HIGH = more likely AI.
    Analyzes the upper-third of the face crop (eye region).
    Checks for unnatural symmetry and over-smoothness.
    """
    try:
        w, h  = face_crop.size
        # Eye region is roughly top 40% of face
        eye_region = face_crop.crop((0, 0, w, int(h * 0.40)))
        gray  = np.array(eye_region.convert("L"), dtype=np.float32)

        # Check left-right symmetry — AI eyes are TOO symmetric
        left_half  = gray[:, :gray.shape[1]//2]
        right_half = np.fliplr(gray[:, gray.shape[1]//2:])
        min_w = min(left_half.shape[1], right_half.shape[1])
        symmetry_diff = np.mean(np.abs(
            left_half[:, :min_w] - right_half[:, :min_w]
        ))

        # Real eyes: higher asymmetry diff (natural imperfections)
        # AI eyes:   very low diff (too perfectly symmetric)
        # Low diff → high AI score
        symmetry_ai_score = float(np.clip(1.0 - (symmetry_diff / 15.0), 0.0, 1.0))
        print(f"    Eye symmetry diff: {symmetry_diff:.2f} → ai_score: {symmetry_ai_score:.3f}")
        return symmetry_ai_score
    except Exception as e:
        print(f"    Eye analysis error: {e}")
        return 0.5


# ─────────────────────────────────────────────────────────────
# HAIR EDGE ANALYSIS
# AI hair has unnaturally smooth, blended edges
# Real hair has sharp, stray, chaotic edge patterns
# ─────────────────────────────────────────────────────────────
def analyze_hair_edges(face_crop: Image.Image) -> float:
    """
    Returns [0,1] where HIGH = more likely AI.
    Analyzes edge complexity in the hair/boundary region (top of face crop).
    """
    try:
        gray    = np.array(face_crop.convert("L"), dtype=np.uint8)
        h, w    = gray.shape
        # Hair is at the very top of the face crop
        hair_region = gray[:int(h * 0.20), :]
        hair_region = cv2.resize(hair_region, (128, 32))

        edges = cv2.Canny(hair_region, threshold1=30, threshold2=100)
        edge_density  = np.sum(edges > 0) / edges.size

        # Real hair: high edge density (lots of strand edges)
        # AI hair:   low edge density (blended, smooth boundary)
        # Low density → high AI score
        ai_score = float(np.clip(1.0 - (edge_density / 0.15), 0.0, 1.0))
        print(f"    Hair edge density: {edge_density:.4f} → ai_score: {ai_score:.3f}")
        return ai_score
    except Exception as e:
        print(f"    Hair edge error: {e}")
        return 0.5


# ─────────────────────────────────────────────────────────────
# NOISE PATTERN ANALYSIS
# ─────────────────────────────────────────────────────────────
def get_noise_score(img_pil: Image.Image) -> float:
    try:
        gray      = np.array(img_pil.convert("L"), dtype=np.float32)
        blur      = cv2.GaussianBlur(gray, (3, 3), 0)
        noise     = gray - blur
        patch_sz  = 16
        variances = []
        h, w      = noise.shape
        for y in range(0, h - patch_sz, patch_sz):
            for x in range(0, w - patch_sz, patch_sz):
                variances.append(np.var(noise[y:y+patch_sz, x:x+patch_sz]))
        if not variances:
            return 0.5
        mean_var = np.mean(variances)
        return float(np.clip(1.0 - (mean_var / 80.0), 0.0, 1.0))
    except:
        return 0.5


# ─────────────────────────────────────────────────────────────
# FORENSIC SIGNALS
# ─────────────────────────────────────────────────────────────
def get_forensic_signals(img_bytes: bytes, filename: str = "") -> tuple[float, float]:
    freq_score, ela_score = 0.0, 0.0
    try:
        nparr    = np.frombuffer(img_bytes, np.uint8)
        img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img_gray is not None:
            img_res  = cv2.resize(img_gray, (256, 256)).astype(np.float32)
            dft      = np.fft.fftshift(np.fft.fft2(img_res))
            mag      = np.abs(dft) + 1e-8
            h, w     = mag.shape
            Y, X     = np.ogrid[:h, :w]
            low_mask = (X - w//2)**2 + (Y - h//2)**2 <= 20**2
            lo, hi   = np.mean(mag[low_mask]), np.mean(mag[~low_mask])
            if lo > 0:
                freq_score = float(np.clip((hi/lo - 0.02) / 0.15, 0.0, 1.0))
    except Exception as e:
        print(f"    Freq error: {e}")

    if filename.lower().endswith((".jpg", ".jpeg")):
        try:
            orig = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            buf  = io.BytesIO()
            orig.save(buf, "JPEG", quality=75)
            buf.seek(0)
            diff     = ImageChops.difference(orig, Image.open(buf).convert("RGB"))
            raw_ela  = np.array(diff, dtype=np.float32).mean()
            ela_score = float(np.clip(1.0 - (raw_ela / 20.0), 0.0, 1.0))
        except Exception as e:
            print(f"    ELA error: {e}")

    return freq_score, ela_score


# ─────────────────────────────────────────────────────────────
# MASTER ANALYSIS FUNCTION
# ─────────────────────────────────────────────────────────────
def analyze_single_frame(
    img_pil: Image.Image, img_bytes: bytes, filename: str = ""
) -> dict:
    print(f"\n--- Analyzing: {filename or 'frame'} ---")

    # 1. Full image ensemble
    full_ai, model_details = run_ensemble(img_pil)
    print(f"  Full image ai_score: {full_ai:.3f}")

    # 2. Face-specific deep analysis
    face_crop    = extract_face_crop(img_pil)
    face_found   = face_crop is not None
    face_ai      = None
    skin_score   = 0.5
    eye_score    = 0.5
    hair_score   = 0.5

    if face_found:
        print("  Face detected — running face-specific analysis...")
        face_ai    , _ = run_ensemble(face_crop)
        skin_score      = analyze_skin_texture(face_crop)
        eye_score       = analyze_eye_region(face_crop)
        hair_score      = analyze_hair_edges(face_crop)
        print(f"  Face ensemble ai_score: {face_ai:.3f}")

    # 3. Whole-image forensics
    freq_score, ela_score = get_forensic_signals(img_bytes, filename)
    noise_score = get_noise_score(img_pil)

    # ── SCORING ──────────────────────────────────────────────
    # Model score (most reliable)
    if face_found and face_ai is not None:
        model_score = (full_ai * 0.40) + (face_ai * 0.60)
    else:
        model_score = full_ai

    # Face micro-feature score (only valid when face found)
    if face_found:
        face_micro_score = (
            skin_score * 0.50 +   # skin smoothness is the strongest signal
            eye_score  * 0.30 +   # eye symmetry
            hair_score * 0.20     # hair edges
        )
    else:
        face_micro_score = 0.5    # neutral when no face

    # Forensic score
    forensic_score = (
        freq_score  * 0.40 +
        ela_score   * 0.30 +
        noise_score * 0.30
    )

    # Final weighted combination
    if face_found:
        # When face is present, face micro-features get a meaningful vote
        final_ai_prob = (
            model_score      * 0.60 +
            face_micro_score * 0.25 +
            forensic_score   * 0.15
        )
    else:
        final_ai_prob = (
            model_score    * 0.80 +
            forensic_score * 0.20
        )

    print(f"  model={model_score:.3f} | face_micro={face_micro_score:.3f} | forensic={forensic_score:.3f}")
    print(f"  ► Final AI probability: {final_ai_prob:.3f}")

    # Threshold: 0.52 to be sensitive to modern Instagram AI models
    THRESHOLD = 0.52
    is_ai     = final_ai_prob > THRESHOLD
    confidence = final_ai_prob if is_ai else (1.0 - final_ai_prob)

    return {
        "is_ai":               is_ai,
        "confidence":          round(confidence    * 100, 2),
        "final_ai_probability":round(final_ai_prob * 100, 2),
        "model_details":       model_details,
        "face_analyzed":       face_found,
        "face_ai_score":       round(face_ai    * 100, 2) if face_ai    is not None else None,
        "skin_smoothness":     round(skin_score * 100, 2) if face_found else None,
        "eye_symmetry":        round(eye_score  * 100, 2) if face_found else None,
        "hair_edge_score":     round(hair_score * 100, 2) if face_found else None,
        "freq_anomaly":        round(freq_score  * 100, 2),
        "ela_anomaly":         round(ela_score   * 100, 2),
        "noise_anomaly":       round(noise_score * 100, 2),
    }


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents  = await file.read()
    filename  = file.filename or ""
    fname_low = filename.lower()

    if fname_low.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name

            cap          = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                return {"error": "Could not read video frames"}

            n       = min(5, total_frames)
            indices = [int(i*(total_frames-1)/(n-1)) if n > 1 else 0 for i in range(n)]
            results = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    ok, buf = cv2.imencode(".jpg", frame)
                    if ok:
                        fb   = buf.tobytes()
                        fpil = Image.open(io.BytesIO(fb)).convert("RGB")
                        results.append(analyze_single_frame(fpil, fb, "frame.jpg"))

            cap.release()
            if not results:
                return {"error": "No frames processed"}

            ai_votes   = sum(1 for r in results if r["is_ai"])
            is_ai_fin  = ai_votes > len(results) / 2
            avg_conf   = float(np.mean([r["confidence"] for r in results]))

            return {
                "is_ai":           is_ai_fin,
                "score":           round(avg_conf, 2),
                "frames_analyzed": len(results),
                "ai_frame_votes":  ai_votes,
                "type":            "Temporal Video Analysis",
            }
        except Exception as e:
            return {"error": f"Video failed: {e}"}
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    else:
        try:
            img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
            result  = analyze_single_frame(img_pil, contents, filename)
            return {**result, "type": "Static Image Analysis"}
        except Exception as e:
            return {"error": f"Image failed: {e}"}


@app.post("/debug")
async def debug(file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename or ""
    img_pil  = Image.open(io.BytesIO(contents)).convert("RGB")
    return analyze_single_frame(img_pil, contents, filename)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
