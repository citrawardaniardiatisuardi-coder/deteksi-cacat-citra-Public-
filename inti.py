"""inti.py — semua rumus dari notebook, dipakai juga oleh dashboard."""
import time
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

GRID = 8
SUDUT = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
PROPS = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']


def baca_abu(sumber):
    """Baca gambar jadi matriks abu-abu. sumber = path file atau bytes (dari unggahan)."""
    if isinstance(sumber, (bytes, bytearray)):
        return cv2.imdecode(np.frombuffer(sumber, np.uint8), cv2.IMREAD_GRAYSCALE)
    return cv2.imread(str(sumber), cv2.IMREAD_GRAYSCALE)


def resize_citra(img, n):
    interp = cv2.INTER_AREA if n < max(img.shape[:2]) else cv2.INTER_LINEAR
    return cv2.resize(img, (n, n), interpolation=interp)


def normalisasi(img):
    f = img.astype(np.float64)
    fmin, fmax = f.min(), f.max()
    if fmax - fmin == 0:
        return np.zeros(img.shape, dtype=np.uint8)
    return np.round((f - fmin) / (fmax - fmin) * 255).astype(np.uint8)


def peta_tekstur(img_norm, n):
    k = max(3, n // 32)
    f = img_norm.astype(np.float64)
    rata, rata2 = cv2.blur(f, (k, k)), cv2.blur(f ** 2, (k, k))
    s = np.sqrt(np.maximum(rata2 - rata ** 2, 0))
    D = np.abs(np.log((s + 1) / (np.median(s) + 1)))
    D = cv2.GaussianBlur(D, (0, 0), max(1.0, n / 128))
    return s, D


def segmentasi(img_norm, n):
    _, D = peta_tekstur(img_norm, n)
    t, m = cv2.threshold(normalisasi(D), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m, t, normalisasi(D)


def fitur_glcm(patch, L=16, jarak=(1, 2)):
    q = (patch.astype(np.int32) * L // 256).astype(np.uint8)
    P = graycomatrix(q, distances=list(jarak), angles=SUDUT, levels=L, symmetric=True, normed=True)
    hasil = {}
    for i, d in enumerate(jarak):
        for prop in PROPS:
            hasil[f'{prop}_d{d}'] = np.nan_to_num(graycoprops(P, prop)[i], nan=1.0).mean()
        p = P[:, :, i, :]
        hasil[f'entropy_d{d}'] = (-(p * np.log2(p + 1e-12)).sum(axis=(0, 1))).mean()
    return hasil


def kotak_petak(n, grid=GRID):
    b = np.linspace(0, n, grid + 1).astype(int)
    return [(b[r], b[r + 1], b[c], b[c + 1]) for r in range(grid) for c in range(grid)]


def label_petak(mask512, grid=GRID, ambang=0.25):
    """Label tiap petak dari mask: 1 = cacat (>= 25% tertutup), 0 = normal (0%), -1 = ragu (dibuang)."""
    if mask512 is None:
        return np.zeros(grid * grid, dtype=int)
    m = (mask512 > 127).astype(np.float64)
    lab = []
    for (a0, a1, b0, b1) in kotak_petak(m.shape[0], grid):
        frac = m[a0:a1, b0:b1].mean()
        lab.append(1 if frac >= ambang else (0 if frac == 0 else -1))
    return np.array(lab)


def fitur_petak(img_norm, grid=GRID):
    n = img_norm.shape[0]
    mu, sd = img_norm.mean(), img_norm.std() + 1e-6
    rows = []
    for (a0, a1, b0, b1) in kotak_petak(n, grid):
        pt = img_norm[a0:a1, b0:b1]
        v = list(fitur_glcm(pt).values())
        v += [(pt.mean() - mu) / sd, pt.std() / sd]
        rows.append(v)
    X = np.array(rows)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    Z = np.clip((X - med) / (1.4826 * mad + 1e-6), -20, 20)
    return np.hstack([X, Z])


NAMA_MUTLAK = list(fitur_glcm(np.zeros((8, 8), np.uint8)).keys()) + ['rata2', 'simp_baku']
NAMA_FITUR = NAMA_MUTLAK + ['rel_' + nm for nm in NAMA_MUTLAK]


def proses_citra(img, n):
    """Semua tahap untuk resolusi n, sekalian mencatat waktunya."""
    t0 = time.process_time()
    kecil = resize_citra(img, n)
    norm = normalisasi(kecil)
    mask_seg, t_otsu, peta_D = segmentasi(norm, n)
    X = fitur_petak(norm)
    return {'resize': kecil, 'normal': norm, 'peta_D': peta_D, 'mask_seg': mask_seg,
            'ambang_otsu': t_otsu, 'fitur': X, 'waktu_ms': (time.process_time() - t0) * 1000}


def prediksi(model, X, tau):
    p = model.predict_proba(X)[:, 1]
    S = float(p.max())
    return {'prob_petak': p, 'skor': S, 'cacat': S >= tau}


def peta_panas(prob_petak, n, grid=GRID):
    g = prob_petak.reshape(grid, grid).astype(np.float32)
    return np.clip(cv2.resize(g, (n, n), interpolation=cv2.INTER_CUBIC), 0, 1)
