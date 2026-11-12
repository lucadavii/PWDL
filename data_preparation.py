import pathlib, numpy as np, pandas as pd
import matplotlib.pyplot as plt

RAW_DIR = pathlib.Path("data/ECGData")
DEN_DIR = pathlib.Path("data/ECGDataDenoised")
DIAG_XLSX = pathlib.Path("data/Diagnostics.xlsx")

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
T=5000 #number of samples per lead
NL = len(LEADS)
RUN_TH = 500

diag = pd.read_excel(DIAG_XLSX)
filenames = diag["FileName"].to_list()
N = len(filenames)

def zero_runs_per_lead(sig):                         # (T, n_leads) -> (n_leads,)
    sig = np.asarray(sig)
    n = sig.shape[1]
    z = np.vstack([np.zeros((1, n), bool), sig == 0, np.zeros((1, n), bool)])
    d = np.diff(z.astype(np.int8), axis=0)
    out = np.zeros(n, dtype=int)
    for l in range(n):
        s = np.flatnonzero(d[:, l] == 1)
        e = np.flatnonzero(d[:, l] == -1)
        if len(s):
            out[l] = (e - s).max()
    return out

def longest_zero_run(sig):
    return int(zero_runs_per_lead(sig).max())

def convert(folder, out_path):

    paths= {p.stem: p for p in pathlib.Path(folder).rglob("*.csv")}
    X = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, 
                                  shape=(N, T, NL))
    rows = []
    for i, fn in enumerate(filenames):
        r = {"FileName": fn, "missing": fn not in paths, "wrong_shape": False, "has_nan": False,
             "n_zero_leads":0, "n_flat_leads":0, "longest_zero_run":0}
        if not r["missing"]:
            df = pd.read_csv(paths[fn])
            if not set(LEADS) <= set(df.columns): #csv without header
                df = pd.read_csv(paths[fn], header=None, names=LEADS)
            sig = df[LEADS].to_numpy(dtype=np.float32)
            if sig.shape != (T, NL):
                r["wrong_shape"] = True
            else:
                r["has_nan"] = not np.isfinite(sig).all()
                sig = np.nan_to_num(sig)
                zero_per_lead = (sig == 0).mean(axis=0)
                r["n_zero_leads"] = int((zero_per_lead == 1.0).sum())
                r["n_flat_leads"] = int((sig.std(axis=0) < 1e-6).sum())
                r["longest_zero_run"] = longest_zero_run(sig)
                X[i] = sig
        rows.append(r)
        if (i+1) % 1000 == 0:
            print(f"{i+1}/{N} files processed")
    X.flush()
    return X, pd.DataFrame(rows)

print("Converting raw data...")
X_raw, qc_raw = convert(RAW_DIR, "X_raw.npy")
print("Converting denoised data...")
X_den, qc_den = convert(DEN_DIR, "X_den.npy")

for name, q in [("raw", qc_raw), ("den", qc_den)]:
    print(name,
          "| missing:", q.missing.sum(),
          "| wrong shape:", q.wrong_shape.sum(),
          "| NaN:", q.has_nan.sum(),
          "| any all-zero lead:", (q.n_zero_leads > 0).sum(),
          "| all 12 leads zero:", (q.n_zero_leads == 12).sum(),
          "| any flat lead:", (q.n_flat_leads > 0).sum(),
          "| longest zero run:", q.longest_zero_run.max(),
          {th: int((q.longest_zero_run > th).sum()) for th in [10,25,50,100,250,500]})

def is_bad(q):
    return(
        q.missing 
        | q.wrong_shape
        | q.has_nan
        | (q.n_flat_leads > 0)
        | (q.longest_zero_run >= RUN_TH)
    )

keep = ~(is_bad(qc_raw) | is_bad(qc_den)).to_numpy()
print("kept after QC:", keep.sum(), "files out of", N)

np.save("keep_mask.npy", keep)
qc_raw.to_csv("qc_raw.csv", index=False)
qc_den.to_csv("qc_den.csv", index=False)


sus = qc_raw[(qc_raw.longest_zero_run >= RUN_TH) & (qc_raw.n_flat_leads == 0)]
if len(sus):
    fig, axes = plt.subplots(len(sus), 1, figsize=(10, 2 * len(sus)), squeeze=False)
    for ax, i in zip(axes[:, 0], sus.index):
        runs = zero_runs_per_lead(X_raw[i])
        lead = int(runs.argmax())
        ax.plot(np.asarray(X_raw[i][:, lead]), lw=0.7)
        ax.set_title(f"{filenames[i]}  lead {LEADS[lead]}  zero run = {runs[lead]}", fontsize=8)
    plt.tight_layout()
    plt.savefig("suspicious_zero_runs.png", dpi=120)
    plt.close(fig)
    print("saved suspicious_zero_runs.png with", len(sus), "files")

# also the wrong-shape denoised file
print(qc_den[qc_den.wrong_shape].FileName.tolist())