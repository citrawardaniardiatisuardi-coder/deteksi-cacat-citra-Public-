"""Tab Evaluasi & Tradeoff: confusion matrix interaktif, ROC, kurva τ, dan tradeoff resolusi."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
from sklearn.metrics import roc_curve, roc_auc_score

import inti


# ---------------- alat bantu ----------------
def gaya(fig, axs, W):
    """Samakan warna grafik matplotlib dengan tema dashboard."""
    fig.patch.set_facecolor(W['krem'])
    for ax in np.atleast_1d(axs):
        ax.set_facecolor(W['krem'])
        for s in ax.spines.values():
            s.set_color(W['tua'])
        ax.tick_params(colors=W['tua'])
        ax.xaxis.label.set_color(W['tua']); ax.yaxis.label.set_color(W['tua'])
        ax.title.set_color(W['utama']); ax.title.set_fontweight('bold')
        ax.grid(alpha=.25, color=W['tua'])


def judul(no, teks):
    st.markdown(f"<div class='subjudul'><b>{no}</b>{teks}</div>", unsafe_allow_html=True)


def chip(isi):
    """isi = list (label, nilai, keterangan) → baris kartu kecil."""
    html = ''.join(f"<div class='chip'><span>{a}</span><b>{b}</b><i>{c}</i></div>" for a, b, c in isi)
    st.markdown(f"<div class='chip-baris'>{html}</div>", unsafe_allow_html=True)


def hitung_cm(y, s, tau):
    pred = s >= tau
    TP = int(((y == 1) & pred).sum()); FN = int(((y == 1) & ~pred).sum())
    FP = int(((y == 0) & pred).sum()); TN = int(((y == 0) & ~pred).sum())
    P = TP / max(TP + FP, 1); R = TP / max(TP + FN, 1)
    return {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN, 'akurasi': (TP + TN) / len(y), 'presisi': P, 'recall': R,
            'f1': 2 * P * R / max(P + R, 1e-12), 'spesifisitas': TN / max(TN + FP, 1), 'biaya': FP + 20 * FN}


@st.cache_data
def baca(path):
    return pd.read_csv(path) if os.path.exists(path) else None


# ================= TAB EVALUASI =================
def tampilkan_evaluasi(MODEL, resolusi, tau, W):
    cmap = LinearSegmentedColormap.from_list('t', [W['krem'], W['lembut'], W['aksen'], W['utama']])
    SK = baca('hasil/skor_uji.csv')
    RG = baca('hasil/ringkasan.csv')
    if SK is None and RG is None:
        st.warning('File hasil belum ada. Jalankan Langkah 12 dan sel 16a di notebook.')
        return

    if SK is not None and f's_{resolusi}' in SK:
        y, s = SK['y'].values, SK[f's_{resolusi}'].values
        m = hitung_cm(y, s, tau)
        sumber = f'dihitung ulang dari {len(y)} skor S data uji pada τ = {tau:.2f} (geser τ di sidebar!)'
    else:
        r = RG[RG['resolusi'] == resolusi].iloc[0]
        m = {k: r[k] for k in ['TP', 'FP', 'FN', 'TN', 'akurasi', 'presisi', 'recall', 'f1', 'spesifisitas', 'biaya']}
        y = s = None
        sumber = f'dari ringkasan.csv (τ* = {r["tau"]:.2f}). Jalankan sel 16a agar bisa interaktif.'

    judul(1, f'Confusion matrix · {resolusi}×{resolusi} px')
    st.caption(sumber)
    k1, k2 = st.columns([1, 1.25])
    with k1:
        M = np.array([[m['TN'], m['FP']], [m['FN'], m['TP']]])
        fig, ax = plt.subplots(figsize=(4.4, 4))
        ax.imshow(M, cmap=cmap)
        nama = [['TN', 'FP'], ['FN', 'TP']]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{M[i, j]}\n{nama[i][j]}', ha='center', va='center', fontsize=17, fontweight='bold',
                        color=W['krem'] if M[i, j] > M.max() / 2 else W['tua'])
        ax.set_xticks([0, 1], ['Normal', 'Cacat']); ax.set_yticks([0, 1], ['Normal', 'Cacat'])
        ax.set_xlabel('Prediksi'); ax.set_ylabel('Sebenarnya')
        gaya(fig, ax, W); ax.grid(False)
        fig.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)
    with k2:
        chip([('Akurasi', f"{m['akurasi']:.3f}", '(TP+TN)/semua'),
              ('Presisi', f"{m['presisi']:.3f}", 'TP/(TP+FP)'),
              ('Recall', f"{m['recall']:.3f}", 'TP/(TP+FN)'),
              ('F1', f"{m['f1']:.3f}", '2PR/(P+R)'),
              ('Spesifisitas', f"{m['spesifisitas']:.3f}", 'TN/(TN+FP)'),
              ('Biaya DAGM', f"{m['biaya']}", 'FP + 20·FN')])
        st.markdown(f"<div class='kartu'><b>Cara membaca:</b> dari {m['TP'] + m['FN']} gambar cacat, "
                    f"<b>{m['TP']}</b> tertangkap dan <b>{m['FN']}</b> lolos. Dari {m['TN'] + m['FP']} gambar normal, "
                    f"<b>{m['FP']}</b> dicurigai keliru (alarm palsu).</div>", unsafe_allow_html=True)

    if y is not None:
        judul(2, 'Pengaruh ambang τ & kurva ROC')
        k1, k2 = st.columns(2)
        grid = np.linspace(0.01, 0.99, 99)
        kur = pd.DataFrame([hitung_cm(y, s, t) for t in grid])
        with k1:
            fig, ax = plt.subplots(figsize=(5.2, 3.8))
            for kol, w in [('presisi', W['aksen']), ('recall', W['tua']), ('f1', W['utama'])]:
                ax.plot(grid, kur[kol], lw=2.5, color=w, label=kol)
            ax.axvline(tau, ls='--', color=W['tua']); ax.text(tau, 0.03, f' τ={tau:.2f}', color=W['tua'])
            ax.set_xlabel('ambang τ'); ax.set_ylim(0, 1.03); ax.set_title('Presisi, recall, F1 terhadap τ')
            ax.legend(facecolor=W['krem'], edgecolor=W['lembut'])
            gaya(fig, ax, W); fig.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)
        with k2:
            fpr, tpr, _ = roc_curve(y, s)
            auc = roc_auc_score(y, s)
            fig, ax = plt.subplots(figsize=(5.2, 3.8))
            ax.fill_between(fpr, tpr, color=W['lembut'], alpha=.6, label=f'luas = AUC = {auc:.3f}')
            ax.plot(fpr, tpr, lw=2.5, color=W['utama'])
            ax.plot([0, 1], [0, 1], ls=':', color=W['tua'], label='tebakan koin (AUC 0,5)')
            ax.scatter([m['FP'] / max(m['FP'] + m['TN'], 1)], [m['recall']], s=90, color=W['aksen'],
                       edgecolor=W['tua'], zorder=3, label=f'posisi τ = {tau:.2f}')
            ax.set_xlabel('FPR = FP/(FP+TN)'); ax.set_ylabel('TPR = recall'); ax.set_title('Kurva ROC')
            ax.legend(facecolor=W['krem'], edgecolor=W['lembut'], loc='lower right')
            gaya(fig, ax, W); fig.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)
        st.caption('AUC = luas di bawah kurva ROC = P(S cacat > S normal), tidak bergantung pada τ.')

    judul(3 if y is not None else 2, '10 fitur paling penting (penurunan Gini rata-rata)')
    imp = pd.Series(MODEL[resolusi]['model'].feature_importances_, index=inti.NAMA_FITUR).nlargest(10)[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(imp.index, imp.values, color=W['utama'], edgecolor=W['tua'])
    ax.set_xlabel('kepentingan')
    gaya(fig, ax, W); fig.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)


# ================= TAB TRADEOFF =================
def tampilkan_tradeoff(W):
    RG = baca('hasil/ringkasan.csv')
    TR = baca('hasil/tradeoff.csv')
    if RG is None:
        st.warning('ringkasan.csv belum ada. Jalankan Langkah 12 di notebook.')
        return
    df = RG.sort_values('resolusi').copy()
    if TR is not None and 'total_ms' in TR:
        df = df.merge(TR[['resolusi', 'total_ms']], on='resolusi')
    else:
        df['total_ms'] = df['ms_per_gambar']
    N = df['resolusi'].values

    # --- regresi kuadrat terkecil t = a + b·N² (dihitung langsung di sini) ---
    x, t = (N ** 2).astype(float), df['total_ms'].values
    b = ((x - x.mean()) * (t - t.mean())).sum() / ((x - x.mean()) ** 2).sum()
    a = t.mean() - b * x.mean()
    R2 = 1 - ((t - (a + b * x)) ** 2).sum() / ((t - t.mean()) ** 2).sum()

    judul(1, 'Kinerja & waktu terhadap resolusi')
    k1, k2 = st.columns(2)
    with k1:
        fig, ax = plt.subplots(figsize=(5.4, 4))
        for kol, w, mk in [('f1', W['utama'], 'o'), ('recall', W['tua'], 's'),
                           ('presisi', W['aksen'], 'D'), ('auc', W['lembut'], '^')]:
            ax.plot(N, df[kol], marker=mk, lw=2.5, ms=8, color=w, label=kol, markeredgecolor=W['tua'])
        ax.axhline(0.5, ls=':', color=W['tua'], label='AUC tebakan koin')
        ax.set_xscale('log', base=2); ax.set_xticks(N, [str(v) for v in N]); ax.set_ylim(0, 1.05)
        ax.set_xlabel('resolusi N'); ax.set_title('Kinerja vs resolusi')
        ax.legend(facecolor=W['krem'], edgecolor=W['lembut'], fontsize=8)
        gaya(fig, ax, W); fig.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)
    with k2:
        fig, ax = plt.subplots(figsize=(5.4, 4))
        xx = np.linspace(0, x.max() * 1.05, 100)
        ax.plot(xx, a + b * xx, '--', color=W['tua'], label=f't = {a:.0f} + {b:.1e}·N²')
        ax.scatter(x, t, s=110, color=W['utama'], edgecolor=W['tua'], zorder=3, label='median terukur')
        for n_, x_, t_ in zip(N, x, t):
            ax.annotate(f'{n_}px', (x_, t_), xytext=(6, 6), textcoords='offset points', color=W['tua'])
        ax.set_xlabel('jumlah piksel N²'); ax.set_ylabel('ms per gambar'); ax.set_title(f'Waktu vs N² (R² = {R2:.3f})')
        ax.legend(facecolor=W['krem'], edgecolor=W['lembut'], fontsize=8)
        gaya(fig, ax, W); fig.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)

    porsi = a / (a + b * N.max() ** 2) * 100
    st.markdown(f"<div class='kartu'><h3>📐 Regresi kuadrat terkecil</h3>"
                f"t(N) ≈ <b>{a:.1f}</b> + <b>{b:.2e}</b>·N² ms &nbsp;·&nbsp; R² = <b>{R2:.3f}</b><br>"
                f"Biaya tetap a (64 petak × GLCM) = <b>{porsi:.0f}%</b> dari waktu total di {N.max()} px. "
                f"Setelah N ≤ 256, suku b·N² sudah kecil, jadi mengecilkan gambar lagi hampir tidak menghemat waktu."
                f"</div>", unsafe_allow_html=True)

    judul(2, 'Peta tradeoff & titik kompromi')
    ref_f1 = df.loc[df['resolusi'] == N.max(), 'f1'].values[0]
    ref_t = df.loc[df['resolusi'] == N.max(), 'total_ms'].values[0]
    df['F1 vs maks (%)'] = df['f1'] / ref_f1 * 100
    df['hemat waktu (%)'] = 100 - df['total_ms'] / ref_t * 100
    layak = df[df['F1 vs maks (%)'] >= 90]
    pilihan = int(layak['resolusi'].min()) if len(layak) else int(N.max())

    k1, k2 = st.columns([1.2, 1])
    with k1:
        fig, ax = plt.subplots(figsize=(5.6, 4))
        for r in df.itertuples():
            pilih = r.resolusi == pilihan
            ax.scatter(r.total_ms, r.f1, s=420 if pilih else 200, zorder=3,
                       color=W['aksen'] if pilih else W['utama'], edgecolor=W['tua'], lw=2)
            ax.annotate(f'{r.resolusi}px' + (' ★' if pilih else ''), (r.total_ms, r.f1),
                        xytext=(10, 6), textcoords='offset points', color=W['tua'], fontweight='bold')
        ax.set_xlabel('ms per gambar'); ax.set_ylabel('F1'); ax.set_title('Kanan-atas = akurat tapi lambat')
        gaya(fig, ax, W); fig.tight_layout(); st.pyplot(fig, width='stretch'); plt.close(fig)
    with k2:
        r = df[df['resolusi'] == pilihan].iloc[0]
        st.markdown(f"<div class='vonis ok'>★ {pilihan}×{pilihan} px"
                    f"<small>resolusi terkecil dengan F1 ≥ 90% dari {N.max()} px</small></div>", unsafe_allow_html=True)
        chip([('F1', f"{r['f1']:.3f}", f"{r['F1 vs maks (%)']:.0f}% dari maksimum"),
              ('Hemat waktu', f"{r['hemat waktu (%)']:.0f}%", f"vs {N.max()} px"),
              ('Biaya DAGM', f"{int(r['biaya'])}", 'FP + 20·FN')])

    tabel = df[['resolusi', 'jml_piksel', 'tau', 'akurasi', 'presisi', 'recall', 'f1', 'auc', 'biaya',
                'total_ms', 'F1 vs maks (%)', 'hemat waktu (%)']].round(3)
    st.dataframe(tabel, width='stretch', hide_index=True)
    st.download_button('⬇️ Unduh tabel tradeoff (CSV)', tabel.to_csv(index=False).encode(),
                       'tradeoff_resolusi.csv', 'text/csv')
