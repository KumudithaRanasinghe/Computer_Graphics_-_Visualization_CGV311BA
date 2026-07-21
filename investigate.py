"""
SAMS – Signature Investigation / Verification
CS402.3 - Computer Graphics and Visualization Coursework
Usage: python investigate.py <student_index>
Example: python investigate.py 001
Collects all signature crops for a student, computes similarity,
and flags potential mismatches or forgeries.
"""

import sys
import os
import cv2
import numpy as np
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Pure OpenCV / NumPy SSIM implementation (no skimage dependency required)
def compute_ssim(img1, img2):
    """Compute Structural Similarity Index (SSIM) between two grayscale images [0, 1]."""
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    kernel_size = (11, 11)
    sigma = 1.5

    mu1 = cv2.GaussianBlur(img1, kernel_size, sigma)
    mu2 = cv2.GaussianBlur(img2, kernel_size, sigma)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, kernel_size, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, kernel_size, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, kernel_size, sigma) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


DB_FILE = "attendance.db"
SIG_DIR = "outputs/processing_steps"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


SIG_RESIZE = (200, 80)   # canonical size for comparison


def load_signature_crop(sheet_name, row_index, img_folder="signing_sheets"):
    """
    Reconstruct the signature cell from the original image.
    Re-runs the crop logic so we don't need to persist crops separately.
    """
    for ext in [".jpeg", ".jpg", ".png"]:
        path = os.path.join(img_folder, sheet_name + ext)
        if os.path.exists(path):
            break
    else:
        return None

    img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape
    # Mirror the logic in SigningSheetProcessor
    top    = int(h * 0.28)
    bot    = int(h * 0.72)
    sig_x0 = int(w * 0.71)

    # Divide body evenly into 6 rows (excluding header)
    num_rows = 6
    row_h    = (bot - top) // 7  # 7 physical rows including header
    y1       = top + (row_index + 1) * row_h  # +1 to skip header
    y2       = y1 + row_h

    cell = img[y1:y2, sig_x0:int(w*0.92)]
    if cell.size == 0:
        return None

    # Binarize
    _, cell_bin = cv2.threshold(cell, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(cell_bin, SIG_RESIZE, interpolation=cv2.INTER_AREA)
    return resized


def get_student_sessions(index):
    """Return all sessions for a student from the DB."""
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(f"Database '{DB_FILE}' not found. Run sams.py first.")
    conn = sqlite3.connect(DB_FILE)
    cur  = conn.cursor()
    cur.execute("""
        SELECT sheet, date, name, present
        FROM attendance
        WHERE student_index = ?
        ORDER BY date ASC
    """, (index,))
    rows = cur.fetchall()
    conn.close()
    return rows


def compute_similarity(sig1, sig2):
    """Return SSIM similarity score between two binary signature images."""
    if sig1 is None or sig2 is None:
        return 0.0
    # Ensure same size
    s2 = cv2.resize(sig2, (sig1.shape[1], sig1.shape[0]))
    score = compute_ssim(sig1.astype(np.float32) / 255.0,
                         s2.astype(np.float32)  / 255.0)
    return float(score)


def investigate(index):
    sessions = get_student_sessions(index)
    if not sessions:
        print(f"No records found for student index '{index}'.")
        sys.exit(1)

    name = sessions[0][2]
    print(f"\n{'='*60}")
    print(f"  SIGNATURE INVESTIGATION — {name} ({index})")
    print(f"{'='*60}")

    # Collect row_index from the attendance table ordering
    all_sheets = list({s[0] for s in sessions})
    all_sheets.sort()

    # Map sheet → student's row index (0-based) for this student
    conn = sqlite3.connect(DB_FILE)
    cur  = conn.cursor()
    signatures = []

    for sheet, date, _, present in sessions:
        if not present:
            signatures.append({
                "sheet": sheet, "date": date,
                "sig": None, "present": False
            })
            continue

        # Find the row index of this student in that sheet
        cur.execute("""
            SELECT rowid FROM attendance
            WHERE sheet = ? AND student_index = ?
        """, (sheet, index))
        r = cur.fetchone()

        # Fallback: get ordering within the sheet
        cur.execute("""
            SELECT student_index FROM attendance
            WHERE sheet = ? ORDER BY id ASC
        """, (sheet,))
        all_in_sheet = [row[0] for row in cur.fetchall()]
        try:
            row_idx = all_in_sheet.index(index)
        except ValueError:
            row_idx = 0

        sig = load_signature_crop(sheet, row_idx)
        signatures.append({
            "sheet": sheet, "date": date,
            "sig": sig, "present": True, "row_idx": row_idx
        })

    conn.close()

    # Filter only present sessions with valid crops
    valid = [s for s in signatures if s["present"] and s["sig"] is not None]
    print(f"  Sessions with signatures: {len(valid)} / {len(sessions)}")

    if len(valid) < 2:
        print("  ℹ Not enough signatures to compare (need ≥ 2 present sessions).")
        _render_single(valid, index, name)
        return

    # Pairwise similarity matrix
    n     = len(valid)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = compute_similarity(valid[i]["sig"], valid[j]["sig"])

    # Compare each to the "reference" (first valid signature)
    ref_sig   = valid[0]["sig"]
    print(f"\n  Reference signature: {valid[0]['date']} (sheet {valid[0]['sheet']})")
    print(f"\n  {'Date':<14} {'Sheet':<8} {'Similarity':>10}  {'Flag'}")
    print(f"  {'-'*50}")

    flags = []
    for s in valid:
        score = compute_similarity(ref_sig, s["sig"])
        flag  = "⚠ MISMATCH" if score < 0.30 else "✓ OK"
        flags.append(flag)
        print(f"  {s['date']:<14} {s['sheet']:<8} {score:>10.3f}  {flag}")

    # ── Visualisation ─────────────────────────────────────────────────────────
    cols = min(n, 5)
    rows_layout = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_layout + 1, cols,
                              figsize=(cols * 3.5, rows_layout * 2.5 + 4))
    fig.patch.set_facecolor("#F7F9FC")
    fig.suptitle(f"Signature Investigation — {name} (Index: {index})",
                 fontsize=14, fontweight="bold")

    # Flatten axes safely
    if rows_layout + 1 == 1:
        axes = np.array([axes])
    if cols == 1:
        axes = axes[:, np.newaxis]

    for idx, s in enumerate(valid):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.imshow(s["sig"], cmap="gray", vmin=0, vmax=255)
        score = compute_similarity(ref_sig, s["sig"])
        colour = "#27AE60" if score >= 0.30 else "#E74C3C"
        ax.set_title(f"{s['date']}\n{flags[idx]}  ({score:.3f})",
                     fontsize=8, color=colour, fontweight="bold")
        ax.axis("off")

    # Hide unused axes in signature rows
    for idx in range(len(valid), rows_layout * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    # Similarity heatmap in last row
    ax_heat = fig.add_subplot(rows_layout + 1, 1, rows_layout + 1)
    im = ax_heat.imshow(sim_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    labels = [s["date"] for s in valid]
    ax_heat.set_xticks(range(n))
    ax_heat.set_yticks(range(n))
    ax_heat.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax_heat.set_yticklabels(labels, fontsize=8)
    ax_heat.set_title("Pairwise Similarity Heatmap (SSIM)", fontweight="bold")
    plt.colorbar(im, ax=ax_heat, orientation="vertical", fraction=0.02, pad=0.02)
    for i in range(n):
        for j in range(n):
            ax_heat.text(j, i, f"{sim_matrix[i,j]:.2f}",
                         ha="center", va="center", fontsize=7,
                         color="black")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(OUT_DIR, f"investigate_{index}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  ✔ Investigation chart saved → {out_path}")
    print(f"{'='*60}\n")


def _render_single(valid, index, name):
    """Fallback: render whatever we have."""
    if not valid:
        return
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.imshow(valid[0]["sig"], cmap="gray")
    ax.set_title(f"Signature — {name} ({index})\n{valid[0]['date']}")
    ax.axis("off")
    out_path = os.path.join(OUT_DIR, f"investigate_{index}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Single signature saved → {out_path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python investigate.py <student_index>")
        sys.exit(1)
    investigate(sys.argv[1])


if __name__ == "__main__":
    main()
