"""
SAMS - Student Attendance Management System
CS402.3 - Computer Graphics and Visualization Coursework
Usage: python sams.py <image_file> <info.xml>
Example: python sams.py 1.jpeg info.xml
"""

import sys
import os
import cv2
import numpy as np
import sqlite3
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime


# ─── CONFIG ──────────────────────────────────────────────────────────────────
DB_FILE   = "attendance.db"
OUT_DIR   = "outputs/processing_steps"
os.makedirs(OUT_DIR, exist_ok=True)


# ─── DATABASE ─────────────────────────────────────────────────────────────────
def init_db():
    """Create (or open) the SQLite database and ensure the table exists."""
    conn = sqlite3.connect(DB_FILE)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            date      TEXT    NOT NULL,
            sheet     TEXT    NOT NULL,
            student_index TEXT NOT NULL,
            name      TEXT    NOT NULL,
            present   INTEGER NOT NULL,   -- 1 = present, 0 = absent
            timestamp TEXT    NOT NULL
        )
    """)
    conn.commit()
    return conn


def save_attendance(conn, date, sheet, student_index, name, present):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO attendance (date, sheet, student_index, name, present, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (date, sheet, student_index, name, 1 if present else 0,
          datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()


# ─── XML PARSING ──────────────────────────────────────────────────────────────
def load_session_info(xml_file, sheet_filename):
    """Return (date, [list of {index, name}]) for the given sheet."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    sheet_base = os.path.basename(sheet_filename)

    for session in root.iter("session"):
        if session.attrib.get("sheet") == sheet_base:
            date     = session.attrib["date"]
            students = [
                {"index": s.attrib["index"], "name": s.attrib["name"]}
                for s in session.iter("student")
            ]
            return date, students

    # Fallback: use first session date but all students
    sessions = list(root.iter("session"))
    if sessions:
        date     = sessions[0].attrib["date"]
        students = [
            {"index": s.attrib["index"], "name": s.attrib["name"]}
            for s in root.iter("student")
        ]
        return date, students

    raise ValueError(f"No session found for sheet '{sheet_base}' in {xml_file}")


# ─── IMAGE PROCESSING ─────────────────────────────────────────────────────────
class SigningSheetProcessor:
    """
    Full pipeline:
      1. Load & resize
      2. Greyscale
      3. Gaussian blur
      4. Binarization (Otsu threshold)
      5. Morphological operations (deskew + denoise)
      6. Perspective / deskew correction
      7. Table-row detection (Hough lines or contours)
      8. Signature-cell extraction per row
      9. Presence detection (ink-pixel density)
    """

    THUMB_W  = 900          # width for display / saving thumbnails
    SIG_COL_FRAC = (0.71, 0.92)   # signature column occupies rightmost portion

    def __init__(self, image_path, sheet_name):
        self.image_path = image_path
        self.sheet_name = sheet_name
        self.steps      = {}   # name → image (for saving)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _thumb(self, img):
        h, w = img.shape[:2]
        ratio = self.THUMB_W / w
        return cv2.resize(img, (self.THUMB_W, int(h * ratio)),
                          interpolation=cv2.INTER_AREA)

    def _save_step(self, name, img):
        path = os.path.join(OUT_DIR,
                            f"{self.sheet_name}_{name}.jpg")
        cv2.imwrite(path, self._thumb(img) if img.ndim == 3
                    else self._thumb(img if img.ndim == 2
                                     else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        self.steps[name] = path
        print(f"  [step] {name:35s} → {path}")

    # ── pipeline stages ──────────────────────────────────────────────────────
    def step1_load(self):
        """Step 1 – Load original image."""
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot open {self.image_path}")
        self.original = img
        print(f"  [step] load                              size={img.shape[1]}×{img.shape[0]}")
        self._save_step("01_original", img)
        return img

    def step2_greyscale(self, img):
        """Step 2 – Convert to greyscale."""
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._save_step("02_greyscale", grey)
        return grey

    def step3_blur(self, grey):
        """Step 3 – Gaussian blur to reduce noise."""
        blurred = cv2.GaussianBlur(grey, (5, 5), 0)
        self._save_step("03_gaussian_blur", blurred)
        return blurred

    def step4_binarize(self, blurred):
        """Step 4 – Otsu binarization."""
        _, binary = cv2.threshold(blurred, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self._save_step("04_binary_otsu", binary)
        return binary

    def step5_morphology(self, binary):
        """Step 5 – Morphological cleanup (open then close)."""
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
        closed  = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
        self._save_step("05_morphology", closed)
        return closed

    def step6_deskew(self, grey, binary):
        """Step 6 – Deskew using Hough line detection."""
        edges  = cv2.Canny(binary, 50, 150, apertureSize=3)
        lines  = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
        angle  = 0.0
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                a = np.degrees(theta) - 90
                if abs(a) < 10:          # only near-horizontal lines
                    angles.append(a)
            if angles:
                angle = np.median(angles)

        h, w   = grey.shape
        center = (w // 2, h // 2)
        M      = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(grey, M, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
        self.deskew_angle = angle
        self._save_step("06_deskewed", deskewed)
        print(f"  [step] deskew angle                      {angle:.2f}°")
        return deskewed

    def step7_detect_rows(self, deskewed, num_students):
        """
        Step 7 – Detect table rows using multi-kernel horizontal projection,
        identifying the double separator line, and extracting student data rows
        while skipping the column header.
        """
        h, w = deskewed.shape
        _, bin_desk = cv2.threshold(deskewed, 0, 255,
                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Multi-kernel horizontal line detection
        all_h = set()
        for kw_frac in [4, 5, 6, 7]:
            h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (w // kw_frac, 1))
            h_mask = cv2.morphologyEx(bin_desk, cv2.MORPH_OPEN, h_kern)
            h_proj = np.sum(h_mask, axis=1) / 255
            thresh = w * 0.10
            for r in np.where(h_proj > thresh)[0]:
                all_h.add(r)

        all_h_sorted = sorted([r for r in all_h if h * 0.15 < r < h * 0.65])
        hgroups = []
        if len(all_h_sorted) > 0:
            g = [all_h_sorted[0]]
            for r in all_h_sorted[1:]:
                if r - g[-1] < 15:
                    g.append(r)
                else:
                    hgroups.append(g)
                    g = [r]
            hgroups.append(g)
        hmids = [int(np.median(grp)) for grp in hgroups]

        # Find double-line separator (thin gap between title header & column headers)
        gaps = [hmids[j+1] - hmids[j] for j in range(len(hmids)-1)]
        median_gap = np.median(gaps) if gaps else 80
        double_seps = [gi for gi, gap in enumerate(gaps) if gap < median_gap * 0.55]

        if not double_seps:
            min_gap_idx = np.argmin(gaps) if gaps else 0
            double_seps = [min_gap_idx]

        ds_idx = double_seps[-1]
        after_ds = hmids[ds_idx+1:]

        if len(after_ds) > 3:
            after_gaps = [after_ds[j+1] - after_ds[j] for j in range(len(after_ds)-1)]
            med_after = np.median(after_gaps)
            filtered_after = [after_ds[0]]
            for idx in range(1, len(after_ds)):
                gap = after_ds[idx] - filtered_after[-1]
                if gap < med_after * 3.0:
                    filtered_after.append(after_ds[idx])
            after_ds = filtered_after

        col_header_top = after_ds[0] if len(after_ds) > 0 else int(h * 0.3)
        col_header_bot = after_ds[1] if len(after_ds) > 1 else int(h * 0.35)
        student_data_lines = after_ds[1:] if len(after_ds) > 1 else after_ds

        rows = [(student_data_lines[j] + 8, student_data_lines[j+1] - 8)
                for j in range(len(student_data_lines)-1)]

        # Vertical line detection for signature column
        table_top = hmids[0] if hmids else int(h * 0.2)
        table_bot = after_ds[-1] if after_ds else int(h * 0.6)
        v_height = table_bot - table_top

        v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(v_height // 3, 10)))
        v_region = bin_desk[table_top:table_bot, :]
        v_mask = cv2.morphologyEx(v_region, cv2.MORPH_OPEN, v_kern)
        v_proj = np.sum(v_mask, axis=0) / 255
        v_cols = np.where(v_proj > v_height * 0.10)[0]

        vgroups = []
        if len(v_cols) > 0:
            vg = [v_cols[0]]
            for c in v_cols[1:]:
                if c - vg[-1] < 30:
                    vg.append(c)
                else:
                    vgroups.append(vg)
                    vg = [c]
            vgroups.append(vg)
        vmids = [int(np.median(grp)) for grp in vgroups]

        if len(vmids) < 6:
            v_kern2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(v_height // 5, 10)))
            v_mask2 = cv2.morphologyEx(v_region, cv2.MORPH_OPEN, v_kern2)
            v_proj2 = np.sum(v_mask2, axis=0) / 255
            v_cols2 = np.where(v_proj2 > v_height * 0.06)[0]
            vgroups2 = []
            if len(v_cols2) > 0:
                vg = [v_cols2[0]]
                for c in v_cols2[1:]:
                    if c - vg[-1] < 30:
                        vg.append(c)
                    else:
                        vgroups2.append(vg)
                        vg = [c]
                vgroups2.append(vg)
            vmids = [int(np.median(grp)) for grp in vgroups2]

        if len(vmids) >= 2:
            sig_full_left = vmids[-2]
            sig_full_right = vmids[-1]
            sig_col_width = sig_full_right - sig_full_left
            sig_left = sig_full_left + int(sig_col_width * 0.20) + 5
            sig_right = sig_full_right - 10
        else:
            sig_left = int(w * 0.71)
            sig_right = int(w * 0.92)

        self.sig_bounds = (sig_left, sig_right)

        # Visualise rows on a colour copy
        vis = cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)
        for y1, y2 in rows:
            cv2.rectangle(vis, (sig_left, y1), (sig_right, y2), (0, 255, 0), 2)
        cv2.line(vis, (sig_left, table_top), (sig_left, table_bot), (255, 0, 0), 2)
        cv2.line(vis, (sig_right, table_top), (sig_right, table_bot), (255, 0, 0), 2)

        self._save_step("07_row_detection", vis)
        self.rows      = rows
        self.table_w   = w
        self.table_top = table_top
        self.table_bot = table_bot
        return rows, bin_desk

    def step8_extract_signatures(self, deskewed, binary, rows):
        """
        Step 8 – Crop signature cells, apply morphological line artifact removal,
        and measure ink density.
        Returns list of (cell_c, ink_density, present_bool).
        """
        sig_x0, sig_x1 = self.sig_bounds
        results        = []

        cell_h  = max((r[1] - r[0]) for r in rows) if rows else 50
        strip_h = cell_h * len(rows) if rows else 50
        strip_w = max(sig_x1 - sig_x0, 50)
        strip   = np.zeros((strip_h, strip_w), dtype=np.uint8)

        for i, (y1, y2) in enumerate(rows):
            cell = binary[y1:y2, sig_x0:sig_x1]
            if cell.size == 0:
                results.append((None, 0.0, False))
                continue

            cell_h_cur, cell_w_cur = cell.shape

            # Clean horizontal table line artifacts
            h_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (max(cell_w_cur * 2 // 3, 5), 1))
            h_artifacts = cv2.morphologyEx(cell, cv2.MORPH_OPEN, h_clean)
            cell_c = cv2.subtract(cell, h_artifacts)

            # Clean vertical table line artifacts
            v_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(cell_h_cur * 2 // 3, 5)))
            v_artifacts = cv2.morphologyEx(cell_c, cv2.MORPH_OPEN, v_clean)
            cell_c = cv2.subtract(cell_c, v_artifacts)

            ink_px   = np.sum(cell_c > 0)
            total_px = cell_c.size
            density  = ink_px / total_px if total_px > 0 else 0.0

            # Presence threshold: >2.0% ink density in cleaned signature cell → signed
            present  = density > 0.02

            # Place cell in strip
            cy = i * cell_h
            ch = min(cell_h, cell_c.shape[0])
            cw = min(strip_w, cell_c.shape[1])
            strip[cy:cy+ch, :cw] = cell_c[:ch, :cw]
            results.append((cell_c, density, present))

        self._save_step("08_signature_cells", strip)
        return results

    def step9_annotate(self, results, students):
        """Step 9 – Annotate the original image with P/A markers."""
        annotated = self.original.copy()
        sig_x0, sig_x1 = getattr(self, 'sig_bounds', (int(annotated.shape[1] * 0.71), int(annotated.shape[1] * 0.92)))

        for i, (y1, y2) in enumerate(self.rows):
            if i >= len(students) or i >= len(results):
                break
            _, density, present = results[i]
            label  = "PRESENT" if present else "ABSENT"
            colour = (0, 180, 0)  if present else (0, 0, 220)
            cy     = (y1 + y2) // 2

            cv2.putText(annotated, label,
                        (sig_x0 + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)
            bar_w = int(density * 500)
            cv2.rectangle(annotated, (sig_x0 + 200, cy - 20),
                          (sig_x0 + 200 + bar_w, cy - 5), colour, -1)

        self._save_step("09_annotated", annotated)
        return annotated

    # ── public entry point ────────────────────────────────────────────────────
    def process(self, students):
        """Run the full pipeline and return list of (student, present) tuples."""
        print("\n── Image Processing Pipeline ──────────────────────────────")
        img      = self.step1_load()
        grey     = self.step2_greyscale(img)
        blurred  = self.step3_blur(grey)
        binary   = self.step4_binarize(blurred)
        morph    = self.step5_morphology(binary)
        deskewed = self.step6_deskew(grey, morph)
        
        rows, bin_desk = self.step7_detect_rows(deskewed, len(students))
        results  = self.step8_extract_signatures(deskewed, bin_desk, rows)
        self.step9_annotate(results, students)
        print("── Pipeline Complete ───────────────────────────────────────\n")

        attendance = []
        for i, student in enumerate(students):
            if i < len(results):
                _, density, present = results[i]
            else:
                present = False
                density = 0.0
            attendance.append({
                "index":   student["index"],
                "name":    student["name"],
                "present": present,
                "density": density
            })
        return attendance


# ─── REPORTING ────────────────────────────────────────────────────────────────
def print_report(date, sheet, attendance):
    total   = len(attendance)
    present = sum(1 for a in attendance if a["present"])
    absent  = total - present

    print("=" * 60)
    print(f" ATTENDANCE REPORT — {sheet}  ({date})")
    print("=" * 60)
    print(f"{'#':<4} {'Index':<8} {'Name':<30} {'Status':<10} {'Ink%'}")
    print("-" * 60)
    for i, a in enumerate(attendance, 1):
        status = "PRESENT" if a["present"] else "ABSENT"
        print(f"{i:<4} {a['index']:<8} {a['name']:<30} {status:<10} {a['density']*100:.1f}%")
    print("-" * 60)
    print(f"  Total: {total}  |  Present: {present}  |  Absent: {absent}")
    print("=" * 60)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) != 3:
        print("Usage: python sams.py <image_file> <info.xml>")
        sys.exit(1)

    image_file = sys.argv[1]
    xml_file   = sys.argv[2]
    sheet_name = os.path.splitext(os.path.basename(image_file))[0]

    if not os.path.exists(image_file):
        print(f"Error: image file '{image_file}' not found.")
        sys.exit(1)
    if not os.path.exists(xml_file):
        print(f"Error: XML file '{xml_file}' not found.")
        sys.exit(1)

    print(f"\n╔══════════════════════════════════════════════╗")
    print(f"║  SAMS – Student Attendance Management System ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║  Sheet : {image_file:<36}║")
    print(f"║  Info  : {xml_file:<36}║")
    print(f"╚══════════════════════════════════════════════╝\n")

    # 1. Parse XML
    date, students = load_session_info(xml_file, image_file)
    print(f"Session date: {date}  |  Students: {len(students)}")

    # 2. Process image
    proc       = SigningSheetProcessor(image_file, sheet_name)
    attendance = proc.process(students)

    # 3. Print report
    print_report(date, image_file, attendance)

    # 4. Save to DB
    conn = init_db()
    for a in attendance:
        save_attendance(conn, date, sheet_name,
                        a["index"], a["name"], a["present"])
    conn.close()
    print(f"\n✔ Attendance saved to '{DB_FILE}'")
    print(f"✔ Processing step images saved in '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
