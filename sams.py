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
        Step 7 – Detect table rows.
        Uses horizontal projection profile to find row separators,
        then splits image into row-height bands equal to num_students rows.
        """
        # Binarize the deskewed image fresh
        _, bin_desk = cv2.threshold(deskewed, 0, 255,
                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Horizontal projection: count ink pixels per row
        h_proj = np.sum(bin_desk, axis=1)

        # Adaptive: find the body of the table by ignoring top/bottom parts
        h, w = deskewed.shape
        top_margin    = int(h * 0.28)
        bottom_margin = int(h * 0.72)
        table_region  = h_proj[top_margin:bottom_margin]

        # Mean-threshold to find blank (separator) rows
        mean_ink = np.mean(table_region)
        separator_rows = np.where(table_region < mean_ink * 0.3)[0] + top_margin

        # Cluster consecutive separator rows into groups → find gaps
        row_bounds = []
        if len(separator_rows) > 0:
            groups = []
            g = [separator_rows[0]]
            for r in separator_rows[1:]:
                if r - g[-1] < 10:
                    g.append(r)
                else:
                    groups.append(g)
                    g = [r]
            groups.append(g)
            # Each group's median is a row boundary
            for grp in groups:
                row_bounds.append(int(np.median(grp)))

        # If we couldn't find enough separators, divide evenly
        if len(row_bounds) < num_students - 1:
            row_bounds = [top_margin + int((bottom_margin - top_margin)
                          * i / num_students)
                          for i in range(1, num_students)]

        # Build (start, end) pairs
        starts = [top_margin]    + row_bounds
        ends   = row_bounds      + [bottom_margin]
        rows   = list(zip(starts, ends))

        # Keep only num_students rows
        rows = rows[:num_students]
        while len(rows) < num_students:
            # pad by subdividing last row if needed
            last = rows[-1]
            mid  = (last[0] + last[1]) // 2
            rows[-1] = (last[0], mid)
            rows.append((mid, last[1]))

        # Visualise rows on a colour copy
        vis = cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)
        for y1, y2 in rows:
            cv2.line(vis, (0, y1), (w, y1), (0, 200, 0), 2)
        cv2.line(vis, (0, bottom_margin), (w, bottom_margin), (0, 200, 0), 2)

        # Mark signature column boundary
        sig_x = int(w * self.SIG_COL_FRAC[0])
        cv2.line(vis, (sig_x, top_margin), (sig_x, bottom_margin), (255, 0, 0), 3)
        self._save_step("07_row_detection", vis)
        self.rows       = rows
        self.table_w    = w
        self.table_top  = top_margin
        self.table_bot  = bottom_margin
        return rows, bin_desk

    def step8_extract_signatures(self, deskewed, binary, rows):
        """
        Step 8 – Crop signature cells and measure ink density.
        Returns list of (row_img, ink_density, present_bool).
        """
        w         = deskewed.shape[1]
        sig_x0    = int(w * self.SIG_COL_FRAC[0])
        sig_x1    = int(w * self.SIG_COL_FRAC[1])
        results   = []

        # Compose a side-by-side strip for visualisation
        cell_h    = max((r[1] - r[0]) for r in rows)
        strip_h   = cell_h * len(rows)
        strip_w   = sig_x1 - sig_x0
        strip     = np.ones((strip_h, strip_w), dtype=np.uint8) * 255

        for i, (y1, y2) in enumerate(rows):
            cell     = binary[y1:y2, sig_x0:sig_x1]
            if cell.size == 0:
                results.append((None, 0.0, False))
                continue
            ink_px   = np.sum(cell > 0)
            total_px = cell.size
            density  = ink_px / total_px

            # Presence threshold: >0.5% ink pixels → signed
            present  = density > 0.005

            # Place cell in strip
            cy = i * cell_h
            ch = min(cell_h, cell.shape[0])
            cw = min(strip_w, cell.shape[1])
            strip[cy:cy+ch, :cw] = cell[:ch, :cw]
            results.append((cell, density, present))

        self._save_step("08_signature_cells", strip)
        return results

    def step9_annotate(self, results, students):
        """Step 9 – Annotate the original image with P/A markers."""
        annotated = self.original.copy()
        h, w      = annotated.shape[:2]
        rows      = self.rows
        sig_x0    = int(w * self.SIG_COL_FRAC[0])

        for i, (y1, y2) in enumerate(rows):
            if i >= len(students):
                break
            _, density, present = results[i]
            label  = "PRESENT" if present else "ABSENT"
            colour = (0, 180, 0)  if present else (0, 0, 220)
            cy     = (y1 + y2) // 2

            cv2.putText(annotated, label,
                        (sig_x0 + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)
            # density bar
            bar_w = int(density * 200)
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
        # Detect num_students + 1 rows to account for the table header row
        rows, bin_desk = self.step7_detect_rows(deskewed, len(students) + 1)
        
        # Discard the first row which is the table header
        if len(rows) > len(students):
            rows = rows[1:]
            
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
