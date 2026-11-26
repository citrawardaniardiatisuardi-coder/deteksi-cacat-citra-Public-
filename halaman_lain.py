"""Tab Histori (+ ekspor laporan) dan tab Rumus."""
import base64
import datetime as dt
import io
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

FILE_HISTORI = 'hasil/histori.csv'


def judul(no, teks):
    st.markdown(f"<div class='subjudul'><b>{no}</b>{teks}</div>", unsafe_allow_html=True)


def chip(isi):
    html = ''.join(f"<div class='chip'><span>{a}</span><b>{b}</b><i>{c}</i></div>" for a, b, c in isi)
    st.markdown(f"<div class='chip-baris'>{html}</div>", unsafe_allow_html=True)


# ================= HISTORI =================
def catat_histori():
    """Tambahkan hasil deteksi terakhir ke histori.csv (sekali untuk tiap kombinasi file + resolusi + τ)."""
    t = st.session_state.get('terakhir')
    if not t:
        return
    kunci = f"{t['file']}|{t['resolusi']}|{t['tau']:.2f}"
    if st.session_state.get('kunci_histori') == kunci:
        return
    st.session_state['kunci_histori'] = kunci
    baris = {'waktu': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'file': t['file'],
             'resolusi': t['resolusi'], 'tau': round(t['tau'], 2), 'skor_S': round(t['skor'], 4),
             'prediksi': t['prediksi'], 'label_asli': t['label_asli'], 'waktu_ms': round(t['waktu_ms'], 1)}
    pd.DataFrame([baris]).to_csv(FILE_HISTORI, mode='a', index=False, header=not os.path.exists(FILE_HISTORI))


def gambar_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def buat_laporan(W, H, NAMA='', NIM=''):
    """Laporan HTML mandiri: ringkasan eksperimen + grafik + histori."""
    RG = pd.read_csv('hasil/ringkasan.csv').sort_values('resolusi')
    TR = pd.read_csv('hasil/tradeoff.csv') if os.path.exists('hasil/tradeoff.csv') else None
    df = RG.merge(TR[['resolusi', 'total_ms']], on='resolusi') if TR is not None else RG.assign(total_ms=RG['ms_per_gambar'])
    N = df['resolusi'].values
    x, t = (N ** 2).astype(float), df['total_ms'].values
    b = ((x - x.mean()) * (t - t.mean())).sum() / ((x - x.mean()) ** 2).sum()
    a = t.mean() - b * x.mean()
    R2 = 1 - ((t - (a + b * x)) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    ref = df[df['resolusi'] == N.max()].iloc[0]
    layak = df[df['f1'] >= 0.9 * ref['f1']]
    pilih = int(layak['resolusi'].min())
    rp = df[df['resolusi'] == pilih].iloc[0]

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), facecolor='white')
    for kol, w in [('f1', W['utama']), ('recall', W['tua']), ('presisi', W['aksen']), ('auc', W['lembut'])]:
        ax[0].plot(N, df[kol], marker='o', lw=2.5, color=w, label=kol)
    ax[0].set_xscale('log', base=2); ax[0].set_xticks(N, [str(v) for v in N]); ax[0].set_ylim(0, 1.05)
    ax[0].set_title('Kinerja vs resolusi'); ax[0].legend(); ax[0].grid(alpha=.3)
    xx = np.linspace(0, x.max() * 1.05, 50)
    ax[1].scatter(x, t, color=W['utama'], s=60, zorder=3); ax[1].plot(xx, a + b * xx, '--', color=W['tua'])
    ax[1].set_title(f'Waktu vs N²  (t ≈ {a:.0f} + {b:.1e}·N², R² = {R2:.2f})'); ax[1].set_xlabel('N²')
    ax[1].set_ylabel('ms/gambar'); ax[1].grid(alpha=.3)
    fig.tight_layout()
    grafik = gambar_png(fig)

    kolom = ['resolusi', 'tau', 'TP', 'FP', 'FN', 'TN', 'akurasi', 'presisi', 'recall', 'f1', 'auc', 'biaya', 'total_ms']
    tabel = df[kolom].round(3).to_html(index=False, border=0)
    tabel_h = H.tail(15).iloc[::-1].to_html(index=False, border=0) if H is not None and len(H) else '<p>Belum ada.</p>'
    return f"""<!doctype html><html lang="id"><head><meta charset="utf-8">
<title>Laporan Deteksi Cacat · Resolusi</title><style>
body{{font-family:Georgia,serif;background:{W['krem']};color:{W['tua']};margin:0;padding:28px;
background-image:radial-gradient({W['lembut']} 2.5px,transparent 3px);background-size:34px 34px}}
.kotak{{max-width:1000px;margin:auto;background:{W['krem']};border:3px solid {W['tua']};border-radius:22px;
padding:28px 34px;box-shadow:7px 7px 0 {W['utama']}}}
h1{{color:{W['utama']};margin:0 0 4px 0}} h2{{color:{W['utama']};border-bottom:3px dotted {W['aksen']};padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;font-size:14px;font-family:Arial,sans-serif}}
th{{background:{W['utama']};color:{W['krem']};padding:7px}} td{{padding:6px;border-bottom:1px solid {W['lembut']};text-align:center}}
.sorot{{background:{W['krem2']};border-left:6px solid {W['aksen']};padding:12px 16px;border-radius:10px}}
img{{width:100%;border:2px solid {W['tua']};border-radius:12px}} small{{opacity:.75}}
</style></head><body><div class="kotak">
<h1>Laporan Evaluasi Pengaruh Resolusi Kamera pada Deteksi Cacat</h1>
<p><b>{NAMA}</b> · NIM {NIM} · Tugas Matematika 3</p>
<small>Dataset DAGM 2007 · Resize + Normalisasi + Segmentasi Otsu + GLCM + Random Forest ·
dibuat {dt.datetime.now():%d-%m-%Y %H:%M}</small>
<h2>1. Ringkasan kinerja per resolusi (data uji)</h2>{tabel}
<h2>2. Grafik tradeoff</h2><img src="data:image/png;base64,{grafik}">
<h2>3. Kesimpulan</h2><div class="sorot">
Resolusi terkecil dengan F1 ≥ 90% dari {N.max()} px adalah <b>{pilih}×{pilih} px</b>
(F1 = {rp['f1']:.3f}; waktu {rp['total_ms']:.0f} ms vs {ref['total_ms']:.0f} ms, hemat {100 - rp['total_ms'] / ref['total_ms'] * 100:.0f}%).<br>
Model waktu (regresi kuadrat terkecil): t(N) ≈ {a:.1f} + {b:.2e}·N² ms, R² = {R2:.3f};
biaya tetap = {a / (a + b * N.max() ** 2) * 100:.0f}% dari total pada {N.max()} px.</div>
<h2>4. Histori pengujian terakhir</h2>{tabel_h}
<h2>5. Catatan validitas</h2><p>Hasil berupa <b>prototipe indikatif</b>, bukan sertifikasi alat. Data DAGM bersifat sintetis,
label cacat berupa elips kasar (weak label), dan hanya subset kelas yang dipakai.</p>
<p><small>Sitasi: Wieler, M., Hahn, T., &amp; Hamprecht, F. A. (2007). Weakly supervised learning for industrial
optical inspection [Dataset]. DAGM 2007. doi:10.5281/zenodo.8086136</small></p></div></body></html>"""


def tampilkan_histori(W, NAMA='', NIM=''):
    catat_histori()
    H = pd.read_csv(FILE_HISTORI) if os.path.exists(FILE_HISTORI) else None

    judul(1, 'Histori pengujian')
    if H is None or len(H) == 0:
        st.markdown("<div class='kartu kosong'><div class='ikon'>🕘</div><h3>Belum ada histori</h3>"
                    "Setiap gambar yang diuji di tab Deteksi otomatis tercatat di sini.</div>", unsafe_allow_html=True)
    else:
        ada_label = H[H['label_asli'].notna() & (H['label_asli'] != '')]
        benar = (ada_label['prediksi'] == ada_label['label_asli']).mean() * 100 if len(ada_label) else float('nan')
        chip([('Jumlah uji', f'{len(H)}', 'baris tercatat'),
              ('Prediksi benar', '–' if np.isnan(benar) else f'{benar:.0f}%', f'dari {len(ada_label)} gambar berlabel'),
              ('Rata-rata waktu', f"{H['waktu_ms'].mean():.0f} ms", 'per gambar')])
        st.dataframe(H.iloc[::-1], width='stretch', hide_index=True)
        k1, k2 = st.columns([1, 1])
        k1.download_button('⬇️ Unduh histori (CSV)', H.to_csv(index=False).encode(), 'histori.csv', 'text/csv')
        if k2.button('🗑️ Hapus histori'):
            os.remove(FILE_HISTORI)
            st.session_state.pop('kunci_histori', None)
            st.rerun()

    judul(2, 'Ekspor laporan')
    if os.path.exists('hasil/ringkasan.csv'):
        st.markdown("<div class='kartu'>Laporan berisi tabel kinerja 4 resolusi, grafik tradeoff, regresi waktu, "
                    "kesimpulan resolusi kompromi, histori terakhir, dan catatan validitas. "
                    "Buka file-nya di browser, lalu <b>Ctrl + P → Simpan sebagai PDF</b> kalau perlu PDF.</div>",
                    unsafe_allow_html=True)
        st.download_button('📄 Unduh laporan (HTML)', buat_laporan(W, H, NAMA, NIM).encode(),
                           f'laporan_{NIM}_{dt.datetime.now():%Y%m%d_%H%M}.html', 'text/html')
    else:
        st.warning('ringkasan.csv belum ada. Jalankan Langkah 12 di notebook.')


# ================= RUMUS =================
RUMUS = [
    ('🖼️ Citra sebagai fungsi', [r'f:\{0,\dots,N-1\}^2 \to \{0,\dots,255\}'],
     'Citra abu-abu = matriks angka. 0 hitam, 255 putih.'),
    ('📏 Resize (rata-rata blok)', [r's=\frac{512}{N},\qquad g(u,v)=\frac{1}{s^2}\sum_{x=us}^{(u+1)s-1}\sum_{y=vs}^{(v+1)s-1} f(x,y)'],
     'Satu piksel baru = rata-rata kotak s×s piksel lama. Luas cacat turun sebanding (N/512)².'),
    ('🌗 Normalisasi min-max', [r'g=\frac{f-f_{\min}}{f_{\max}-f_{\min}}\times 255'], 'Merentangkan nilai ke 0–255.'),
    ('🔔 Gaussian & konvolusi', [r'G_\sigma(x,y)=\frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}',
                                 r'(f*G)(x,y)=\sum_i\sum_j f(x-i,y-j)\,G(i,j),\qquad \sum_i\sum_j G(i,j)=1'],
     'Rata-rata berbobot tetangga. Dipakai untuk merapikan peta segmentasi.'),
    ('📊 Kekasaran lokal & log rasio', [r's(x,y)=\sqrt{E_W[f^2]-\big(E_W[f]\big)^2}',
                                        r'D(x,y)=\left|\ln\frac{s(x,y)+1}{\operatorname{median}(s)+1}\right|'],
     'Simpangan baku di jendela k×k; logaritma membuat "2× lebih halus" dan "2× lebih kasar" setara.'),
    ('✂️ Ambang Otsu', [r't^*=\arg\max_t\;\omega_0(t)\,\omega_1(t)\,[\mu_0(t)-\mu_1(t)]^2'],
     'Pilih ambang yang membuat dua kelompok paling berjauhan (variansi antar-kelas maksimum).'),
    ('🧩 GLCM & fitur Haralick', [r'q=\left\lfloor \frac{f\cdot L}{256}\right\rfloor,\qquad p_{d,\theta}(i,j)=\frac{\#\{\text{pasangan }(i,j)\}}{\#\{\text{semua pasangan}\}}',
                                 r'\text{Kontras}=\sum_{i,j}(i-j)^2p(i,j)\quad \text{Homogenitas}=\sum_{i,j}\frac{p(i,j)}{1+(i-j)^2}',
                                 r'\text{ASM}=\sum_{i,j}p(i,j)^2\quad \text{Entropi}=-\sum_{i,j}p(i,j)\log_2p(i,j)'],
     'Peluang dua piksel bertetangga bernilai i dan j, pada jarak d dan sudut θ (rata-rata 4 sudut).'),
    ('📐 z-skor robust antar petak', [r'z_k=\frac{x_k-\operatorname{median}(x)}{1{,}4826\cdot\operatorname{MAD}+\varepsilon}'],
     '"Seberapa aneh petak ini dibanding 63 saudaranya?" Median & MAD tahan terhadap nilai ekstrem.'),
    ('🌳 Gini & Random Forest', [r'\text{Gini}=1-\sum_c p_c^2,\qquad \Delta G=G_{\text{induk}}-\Big(\tfrac{N_L}{N}G_L+\tfrac{N_R}{N}G_R\Big)',
                                 r'P(\text{cacat}\mid\mathbf{x})=\frac{1}{T}\sum_{t=1}^{T}P_t(\text{cacat}\mid\mathbf{x})'],
     'Pohon memilih belahan dengan penurunan Gini terbesar; hutan merata-ratakan 200 pohon.'),
    ('🎯 Keputusan gambar', [r'S=\max_{k=1,\dots,64}p_k,\qquad \hat y=\begin{cases}\text{cacat} & S\ge\tau\\ \text{normal} & S<\tau\end{cases}',
                             r'\tau^*=\arg\max_\tau F_1(\tau)\ \text{(pada data validasi)}'],
     'Satu petak mencurigakan sudah cukup untuk menyatakan gambar cacat.'),
    ('📈 Metrik evaluasi', [r'\text{Presisi}=\frac{TP}{TP+FP}\quad \text{Recall}=\frac{TP}{TP+FN}\quad F_1=\frac{2PR}{P+R}',
                            r'\text{Biaya DAGM}=FP+20\cdot FN,\qquad \text{AUC}=\int_0^1\text{TPR}\,d(\text{FPR})=P(S_{\text{cacat}}>S_{\text{normal}})'],
     'F1 = rata-rata harmonik; AUC 0,5 = tebakan koin.'),
    ('⏱️ Regresi waktu (kuadrat terkecil)', [r't(N)=a+b\,N^2,\qquad b=\frac{\sum(x_i-\bar x)(t_i-\bar t)}{\sum(x_i-\bar x)^2},\; a=\bar t-b\bar x',
                                            r'(A^\top A)\begin{bmatrix}a\\b\end{bmatrix}=A^\top\mathbf t,\qquad R^2=1-\frac{\sum(t_i-\hat t_i)^2}{\sum(t_i-\bar t)^2}'],
     'a = biaya tetap (64 petak × GLCM), b·N² = biaya yang sebanding jumlah piksel.'),
]


def tampilkan_rumus():
    judul('∑', 'Matematika di balik dashboard')
    kiri, kanan = st.columns(2)
    for i, (nama, rumus, ket) in enumerate(RUMUS):
        with (kiri if i % 2 == 0 else kanan):
            with st.expander(nama, expanded=i < 2):
                for r in rumus:
                    st.latex(r)
                st.caption(ket)
