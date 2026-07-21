import os
import sqlite3
import json
import xml.etree.ElementTree as ET
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import sams

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'xml'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('outputs/processing_steps', exist_ok=True)

# Ensure DB initialized
sams.init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif hasattr(obj, 'item'):  # converts numpy types (np.bool_, np.float64, etc) to python primitives
        return obj.item()
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)

def clear_sheet_attendance(conn, sheet_name):
    cur = conn.cursor()
    cur.execute("DELETE FROM attendance WHERE sheet = ?", (sheet_name,))
    conn.commit()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/sample-images', methods=['GET'])
def get_sample_images():
    sheets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'signing_sheets')
    samples = []
    if os.path.exists(sheets_dir):
        for f in sorted(os.listdir(sheets_dir)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                samples.append(f)
    return jsonify({'samples': samples, 'xml_default': 'info.xml'})

@app.route('/api/process-samples', methods=['POST'])
def process_samples():
    xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'info.xml')
    sheets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'signing_sheets')
    
    if not os.path.exists(xml_path):
        return jsonify({'error': 'info.xml not found'}), 404
        
    sample_files = sorted([f for f in os.listdir(sheets_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not sample_files:
        return jsonify({'error': 'No sample images found in signing_sheets/'}), 404
        
    conn = sams.init_db()
    results = []
    
    for filename in sample_files:
        image_path = os.path.join(sheets_dir, filename)
        sheet_name = os.path.splitext(filename)[0]
        
        try:
            date, students = sams.load_session_info(xml_path, filename)
            proc = sams.SigningSheetProcessor(image_path, sheet_name)
            attendance = proc.process(students)
            
            # Clear old records for this sheet to prevent duplication
            clear_sheet_attendance(conn, sheet_name)
            
            for a in attendance:
                sams.save_attendance(conn, date, sheet_name, a['index'], a['name'], a['present'])
                
            results.append({
                'filename': filename,
                'sheet_name': sheet_name,
                'date': date,
                'total_students': len(students),
                'present_count': sum(1 for a in attendance if a['present']),
                'absent_count': sum(1 for a in attendance if not a['present']),
                'attendance': attendance,
                'steps': proc.steps
            })
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            results.append({
                'filename': filename,
                'error': str(e)
            })
            
    conn.close()
    return jsonify(sanitize_for_json({'status': 'success', 'results': results}))

@app.route('/api/upload-and-process', methods=['POST'])
def upload_and_process():
    if 'images' not in request.files:
        return jsonify({'error': 'No image files provided'}), 400
        
    image_files = request.files.getlist('images')
    xml_file = request.files.get('xml_file')
    
    # Save XML if provided, else use root info.xml
    if xml_file and xml_file.filename != '' and allowed_file(xml_file.filename):
        xml_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(xml_file.filename))
        xml_file.save(xml_path)
    else:
        xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'info.xml')
        
    if not os.path.exists(xml_path):
        return jsonify({'error': 'No info.xml file found or uploaded'}), 400

    conn = sams.init_db()
    results = []
    
    for file in image_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            sheet_name = os.path.splitext(filename)[0]
            
            try:
                date, students = sams.load_session_info(xml_path, filename)
                proc = sams.SigningSheetProcessor(file_path, sheet_name)
                attendance = proc.process(students)
                
                clear_sheet_attendance(conn, sheet_name)
                
                for a in attendance:
                    sams.save_attendance(conn, date, sheet_name, a['index'], a['name'], a['present'])
                    
                results.append({
                    'filename': filename,
                    'sheet_name': sheet_name,
                    'date': date,
                    'total_students': len(students),
                    'present_count': sum(1 for a in attendance if a['present']),
                    'absent_count': sum(1 for a in attendance if not a['present']),
                    'attendance': attendance,
                    'steps': proc.steps
                })
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                results.append({
                    'filename': filename,
                    'error': str(e)
                })
                
    conn.close()
    return jsonify(sanitize_for_json({'status': 'success', 'results': results}))

@app.route('/api/summary', methods=['GET'])
def get_summary():
    if not os.path.exists(sams.DB_FILE):
        return jsonify({'sessions': [], 'students': [], 'overall': {}})

    conn = sqlite3.connect(sams.DB_FILE)
    cur = conn.cursor()

    # Session stats
    cur.execute("""
        SELECT date, sheet, COUNT(*) as total, SUM(present) as present_count
        FROM attendance
        GROUP BY date, sheet
        ORDER BY date ASC
    """)
    session_rows = cur.fetchall()
    sessions = []
    for row in session_rows:
        tot = row[2]
        pres = row[3] or 0
        absent = tot - pres
        pct = round((pres / tot * 100), 1) if tot > 0 else 0
        sessions.append({
            'date': row[0],
            'sheet': row[1],
            'total': tot,
            'present': pres,
            'absent': absent,
            'percentage': pct
        })

    # Student stats
    cur.execute("""
        SELECT student_index, name, COUNT(*) as total_sessions, SUM(present) as present_count
        FROM attendance
        GROUP BY student_index, name
        ORDER BY student_index ASC
    """)
    student_rows = cur.fetchall()
    students = []
    
    total_records = 0
    total_presents = 0

    for row in student_rows:
        idx = row[0]
        name = row[1]
        tot_s = row[2]
        pres_s = row[3] or 0
        abs_s = tot_s - pres_s
        pct = round((pres_s / tot_s * 100), 1) if tot_s > 0 else 0
        
        # Fetch per-session details for this student
        cur.execute("""
            SELECT date, sheet, present
            FROM attendance
            WHERE student_index = ?
            ORDER BY date ASC
        """, (idx,))
        history = [{'date': h[0], 'sheet': h[1], 'present': bool(h[2])} for h in cur.fetchall()]

        total_records += tot_s
        total_presents += pres_s

        students.append({
            'index': idx,
            'name': name,
            'total_sessions': tot_s,
            'present_count': pres_s,
            'absent_count': abs_s,
            'percentage': pct,
            'status': 'Eligible' if pct >= 75.0 else 'Warning (<75%)',
            'history': history
        })

    # Overall stats
    overall_pct = round((total_presents / total_records * 100), 1) if total_records > 0 else 0
    overall = {
        'total_sessions': len(sessions),
        'total_students': len(students),
        'total_records': total_records,
        'total_presents': total_presents,
        'total_absents': total_records - total_presents,
        'overall_percentage': overall_pct
    }

    conn.close()
    return jsonify({
        'sessions': sessions,
        'students': students,
        'overall': overall
    })

@app.route('/api/step-image/<path:filename>')
def serve_step_image(filename):
    # Search in outputs/processing_steps, signing_sheets, or uploads
    dirs = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'processing_steps'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'signing_sheets'),
        app.config['UPLOAD_FOLDER']
    ]
    for d in dirs:
        full_path = os.path.join(d, filename)
        if os.path.exists(full_path):
            return send_from_directory(d, filename)
            
    return jsonify({'error': 'Image not found'}), 404

@app.route('/api/clear-data', methods=['POST'])
def clear_data():
    conn = sqlite3.connect(sams.DB_FILE)
    cur = conn.cursor()
    cur.execute("DELETE FROM attendance")
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'message': 'All attendance records cleared.'})

if __name__ == '__main__':
    print("Starting SAMS Flask Attendance Web App on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
