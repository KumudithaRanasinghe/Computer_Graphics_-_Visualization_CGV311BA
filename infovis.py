"""
SAMS – Information Visualization
CS402.3 - Computer Graphics and Visualization Coursework
Usage: python infovis.py <student_index>
Example: python infovis.py 001
Reads from the local SQLite database and renders attendance charts.
"""

import sys
import os
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime

DB_FILE = "attendance.db"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def fetch_student_attendance(index):
    """Return list of {date, present, sheet} dicts for one student."""
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(f"Database '{DB_FILE}' not found. Run sams.py first.")
    conn = sqlite3.connect(DB_FILE)
    cur  = conn.cursor()
    cur.execute("""
        SELECT date, present, sheet, name
        FROM attendance
        WHERE student_index = ?
        ORDER BY date ASC
    """, (index,))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return None, []

    name    = rows[0][3]
    records = [{"date": r[0], "present": bool(r[1]), "sheet": r[2]} for r in rows]
    return name, records


def fetch_class_summary():
    """Return per-session class-level stats for comparison."""
    conn = sqlite3.connect(DB_FILE)
    cur  = conn.cursor()
    cur.execute("""
        SELECT date, COUNT(*) as total,
               SUM(present) as present_count
        FROM attendance
        GROUP BY date
        ORDER BY date ASC
    """)
    rows = cur.fetchall()
    conn.close()
    return [{"date": r[0], "total": r[1], "present": r[2]} for r in rows]


def visualise(index):
    name, records = fetch_student_attendance(index)

    if not records:
        print(f"No attendance records found for student index '{index}'.")
        sys.exit(1)

    print(f"\nStudent: {name}  (Index: {index})")
    print(f"Sessions found: {len(records)}")

    dates   = [r["date"] for r in records]
    present = [1 if r["present"] else 0 for r in records]
    absent  = [0 if r["present"] else 1 for r in records]

    total_sessions = len(records)
    total_present  = sum(present)
    total_absent   = total_sessions - total_present
    attend_pct     = (total_present / total_sessions * 100) if total_sessions else 0

    class_summary = fetch_class_summary()
    class_dates   = [c["date"] for c in class_summary]
    class_pct     = [c["present"] / c["total"] * 100 for c in class_summary]

    # ── Figure layout: 2×2 grid ──────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#F7F9FC")
    fig.suptitle(f"Attendance Summary — {name}  (Index: {index})\n"
                 f"CS402.3 Computer Graphics and Visualization  |  "
                 f"Overall Attendance: {attend_pct:.1f}%",
                 fontsize=15, fontweight="bold", y=0.97)

    # ── 1. Session-by-session bar chart ──────────────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor("#FAFAFA")
    x   = np.arange(len(dates))
    bars = ax1.bar(x, present, color=["#27AE60" if p else "#E74C3C" for p in present],
                   edgecolor="white", linewidth=0.8, width=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.replace("-", "\n") for d in dates], fontsize=8)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Absent", "Present"])
    ax1.set_title("Session-by-Session Attendance", fontweight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylim(-0.1, 1.4)
    for bar, p in zip(bars, present):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.05,
                 "✓" if p else "✗",
                 ha="center", va="bottom",
                 fontsize=14,
                 color="#27AE60" if p else "#E74C3C")
    ax1.spines[["top", "right"]].set_visible(False)

    # ── 2. Pie chart ─────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor("#FAFAFA")
    if total_present > 0 and total_absent > 0:
        wedges, texts, autotexts = ax2.pie(
            [total_present, total_absent],
            labels=["Present", "Absent"],
            colors=["#27AE60", "#E74C3C"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
            shadow=True
        )
        for at in autotexts:
            at.set_fontsize(12)
            at.set_fontweight("bold")
    elif total_present == total_sessions:
        ax2.pie([1], labels=["100% Present"], colors=["#27AE60"],
                startangle=90)
    else:
        ax2.pie([1], labels=["100% Absent"], colors=["#E74C3C"],
                startangle=90)
    ax2.set_title("Overall Attendance Split", fontweight="bold")

    # ── 3. Cumulative attendance line ─────────────────────────────────────────
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor("#FAFAFA")
    cumulative = np.cumsum(present) / (np.arange(len(present)) + 1) * 100
    ax3.plot(x, cumulative, marker="o", color="#2980B9", linewidth=2,
             markersize=8, markerfacecolor="white", markeredgewidth=2,
             label="Student")
    ax3.axhline(y=75, color="#E74C3C", linestyle="--", linewidth=1.5,
                label="75% threshold")

    # Shade region below threshold
    ax3.fill_between(x, cumulative, 75,
                     where=[c < 75 for c in cumulative],
                     alpha=0.15, color="#E74C3C", label="Below threshold")

    ax3.set_xticks(x)
    ax3.set_xticklabels([d.replace("-", "\n") for d in dates], fontsize=8)
    ax3.set_ylim(0, 110)
    ax3.set_yticks(range(0, 110, 10))
    ax3.set_ylabel("Attendance %")
    ax3.set_title("Cumulative Attendance Rate", fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.grid(axis="y", linestyle="--", alpha=0.4)

    # ── 4. Student vs class attendance ───────────────────────────────────────
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor("#FAFAFA")
    xc  = np.arange(len(class_dates))

    # Map student sessions onto class dates
    student_map = {r["date"]: (1 if r["present"] else 0) for r in records}
    stu_pct_per = []
    for cd in class_dates:
        if cd in student_map:
            stu_pct_per.append(student_map[cd] * 100)
        else:
            stu_pct_per.append(0)

    ax4.bar(xc - 0.2, class_pct,   width=0.35, color="#3498DB", label="Class avg %",
            edgecolor="white")
    ax4.bar(xc + 0.2, stu_pct_per, width=0.35, color="#E67E22", label=f"{name} %",
            edgecolor="white")
    ax4.set_xticks(xc)
    ax4.set_xticklabels([d.replace("-", "\n") for d in class_dates], fontsize=8)
    ax4.set_ylim(0, 120)
    ax4.set_yticks(range(0, 120, 20))
    ax4.set_ylabel("Attendance %")
    ax4.set_title("Student vs Class Attendance", fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(OUT_DIR, f"attendance_{index}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n✔ Visualization saved → {out_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"  ATTENDANCE SUMMARY FOR {name} ({index})")
    print("=" * 50)
    print(f"  Total sessions : {total_sessions}")
    print(f"  Present        : {total_present}")
    print(f"  Absent         : {total_absent}")
    print(f"  Attendance %   : {attend_pct:.1f}%")
    print(f"  Status         : {'✓ OK' if attend_pct >= 75 else '⚠ Below 75% threshold'}")
    print("=" * 50)
    for r in records:
        mark = "✓" if r["present"] else "✗"
        print(f"  {r['date']}  {mark}  {'Present' if r['present'] else 'Absent'}")
    print("=" * 50)


def main():
    if len(sys.argv) != 2:
        print("Usage: python infovis.py <student_index>")
        sys.exit(1)
    visualise(sys.argv[1])


if __name__ == "__main__":
    main()
