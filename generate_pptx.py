"""
PharmaLens — Generate PPTX Presentation
Matches the HTML presentation content and dark theme.
"""
import subprocess
import sys

# Install python-pptx
subprocess.check_call([sys.executable, "-m", "pip", "install", "python-pptx", "-q"])

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

# ── Colors ──
BG       = RGBColor(0x0e, 0x0e, 0x0e)
CARD_BG  = RGBColor(0x1b, 0x1b, 0x1b)
BORDER   = RGBColor(0x2d, 0x2d, 0x2d)
GREEN    = RGBColor(0x4a, 0xde, 0x80)
GREEN2   = RGBColor(0x22, 0xc5, 0x5e)
AMBER    = RGBColor(0xf5, 0xb9, 0x42)
CORAL    = RGBColor(0xef, 0x6f, 0x5c)
WARM     = RGBColor(0xeb, 0xe5, 0xd8)
TX       = RGBColor(0xb5, 0xae, 0x9f)
TX2      = RGBColor(0x7a, 0x75, 0x68)
TX3      = RGBColor(0x4e, 0x4a, 0x44)
WHITE    = RGBColor(0xff, 0xff, 0xff)
GDIM     = RGBColor(0x14, 0x35, 0x2a)
ADIM     = RGBColor(0x35, 0x2b, 0x14)
CDIM     = RGBColor(0x35, 0x1d, 0x18)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H

# Use blank layout
blank_layout = prs.slide_layouts[6]

def set_bg(slide, color=BG):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_text(slide, left, top, width, height, text, size=18, color=WARM, bold=False, align=PP_ALIGN.LEFT, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = align
    return txBox

def add_rich_text(slide, left, top, width, height):
    """Returns a text frame for adding multiple formatted runs."""
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    return tf

def add_run(para, text, size=16, color=TX, bold=False, font_name="Calibri"):
    run = para.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.name = font_name
    return run

def add_rect(slide, left, top, width, height, fill_color=CARD_BG, border_color=BORDER):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1)
    shape.adjustments[0] = 0.05
    return shape

def add_table(slide, left, top, width, rows, cols, col_widths=None):
    tbl_shape = slide.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(width), Inches(0.4 * rows))
    tbl = tbl_shape.table
    
    for r in range(rows):
        for c in range(cols):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = RGBColor(0x15, 0x15, 0x15)
            else:
                cell.fill.fore_color.rgb = CARD_BG if r % 2 == 1 else RGBColor(0x17, 0x17, 0x17)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(11)
                p.font.color.rgb = TX
                p.font.name = "Calibri"
            # border
            from pptx.oxml.ns import qn
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            for border_name in ['a:lnL', 'a:lnR', 'a:lnT', 'a:lnB']:
                ln = tcPr.find(qn(border_name))
                if ln is None:
                    from lxml import etree
                    ln = etree.SubElement(tcPr, qn(border_name))
                ln.set('w', '6350')
                sf = ln.find(qn('a:solidFill'))
                if sf is None:
                    from lxml import etree
                    sf = etree.SubElement(ln, qn('a:solidFill'))
                sr = sf.find(qn('a:srgbClr'))
                if sr is None:
                    from lxml import etree
                    sr = etree.SubElement(sf, qn('a:srgbClr'))
                sr.set('val', '2d2d2d')
    
    if col_widths:
        for i, w in enumerate(col_widths):
            tbl.columns[i].width = Inches(w)
    
    return tbl

def set_cell(tbl, r, c, text, size=11, color=TX, bold=False, align=PP_ALIGN.LEFT):
    cell = tbl.cell(r, c)
    cell.text = ""
    p = cell.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = "Calibri"
    p.alignment = align
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE

def add_stat_box(slide, left, top, width, height, value, label, val_color=GREEN):
    add_rect(slide, left, top, width, height, CARD_BG, BORDER)
    add_text(slide, left, top + 0.15, width, 0.6, value, size=28, color=val_color, bold=True, align=PP_ALIGN.CENTER, font_name="Consolas")
    add_text(slide, left, top + height - 0.5, width, 0.3, label, size=9, color=TX3, align=PP_ALIGN.CENTER, font_name="Consolas")

def add_bar(slide, left, top, width, max_width, height, value_text, fill_color, text_color):
    # track
    add_rect(slide, left, top, max_width, height, RGBColor(0x16, 0x16, 0x16), BORDER)
    # fill
    if width > 0.1:
        bar = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        bar.fill.solid()
        bar.fill.fore_color.rgb = fill_color
        bar.line.fill.background()
        bar.adjustments[0] = 0.15
    # label
    add_text(slide, left, top, width, height, value_text, size=10, color=text_color, align=PP_ALIGN.RIGHT, font_name="Consolas")


# ════════════════════════════════════════════
# SLIDE 1: TITLE
# ════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
set_bg(s)
add_text(s, 0, 1.8, 13.333, 0.4, "MCA CAPSTONE PROJECT · PHASE 1", size=12, color=GREEN, bold=True, align=PP_ALIGN.CENTER, font_name="Consolas")
add_text(s, 0, 2.5, 13.333, 1, "PharmaLens", size=54, color=WARM, bold=True, align=PP_ALIGN.CENTER)
add_text(s, 2, 3.6, 9.333, 0.8, "An Explainable AI Framework for Drug–Target Interaction\nPrediction and Candidate Prioritization", size=18, color=TX, align=PP_ALIGN.CENTER)
add_text(s, 2, 4.8, 9.333, 1.5,
    "Student: [Your Name]  ·  Roll: [Roll No]\nGuide: [Guide Name], [Designation]\nDepartment of Computer Applications, [University]\nOctober 2026",
    size=14, color=TX2, align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════
# SLIDE 2: ABSTRACT
# ════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
set_bg(s)
add_text(s, 0.7, 0.5, 5, 0.3, "ABSTRACT", size=11, color=GREEN, bold=True, font_name="Consolas")
add_text(s, 0.7, 0.9, 6, 0.5, "Project Overview", size=28, color=WARM, bold=True)

tf = add_rich_text(s, 0.7, 1.6, 6.5, 2.5)
p = tf.paragraphs[0]
add_run(p, "Drug discovery costs ", 15, TX)
add_run(p, "$1–2 billion", 15, WARM, True)
add_run(p, " and takes ", 15, TX)
add_run(p, "10–15 years", 15, WARM, True)
add_run(p, " per drug. Testing every drug-protein combination experimentally is infeasible.", 15, TX)
p2 = tf.add_paragraph()
p2.space_before = Pt(12)
add_run(p2, "PharmaLens uses machine learning to predict binding affinity from molecular structure and protein sequences. It compares ", 15, TX)
add_run(p2, "7 models", 15, WARM, True)
add_run(p2, " on the KIBA benchmark and provides ", 15, TX)
add_run(p2, "SHAP explainability", 15, WARM, True)
add_run(p2, " for interpretable predictions.", 15, TX)

# Stats
add_stat_box(s, 8.2, 1.6, 2.3, 1.1, "118K", "INTERACTIONS", GREEN)
add_stat_box(s, 10.8, 1.6, 2.3, 1.1, "7", "MODELS", AMBER)
add_stat_box(s, 8.2, 2.9, 2.3, 1.1, "0.421", "BEST RMSE", GREEN)
add_stat_box(s, 10.8, 2.9, 2.3, 1.1, "0.867", "PEARSON r", WARM)

# ════════════════════════════════════════════
# SLIDE 3: INTRODUCTION
# ════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
set_bg(s)
add_text(s, 0.7, 0.5, 5, 0.3, "INTRODUCTION", size=11, color=GREEN, bold=True, font_name="Consolas")
add_text(s, 0.7, 0.9, 8, 0.5, "Problem · Solution · Scope", size=28, color=WARM, bold=True)

cards = [
    ("🚨 Problem", "• Experimental screening is slow & costs billions\n• Combinatorial explosion: thousands of drugs × hundreds of proteins\n• Most AI models lack interpretability", 0.7, 1.7),
    ("💡 Solution", "• ML framework predicting binding from SMILES + protein sequence\n• 7-model comparison: baselines → XGBoost → CNN → GNN\n• SHAP explainability for every prediction", 6.9, 1.7),
    ("🎯 Purpose", "• Accelerate early-stage drug candidate prioritization\n• Provide trustworthy, interpretable predictions\n• Enable batch screening for researchers", 0.7, 4.2),
    ("📐 Scope", "• KIBA benchmark dataset (kinase inhibitors)\n• Continuous binding affinity prediction\n• Web app for interactive prediction & screening\n• Does not replace experimental validation", 6.9, 4.2),
]
for title, body, x, y in cards:
    add_rect(s, x, y, 5.7, 2.2, CARD_BG, BORDER)
    add_text(s, x + 0.2, y + 0.15, 5.3, 0.35, title, size=14, color=WARM, bold=True)
    add_text(s, x + 0.2, y + 0.55, 5.3, 1.5, body, size=12, color=TX)

# ════════════════════════════════════════════
# SLIDE 4: LITERATURE SURVEY
# ════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
set_bg(s)
add_text(s, 0.7, 0.5, 5, 0.3, "LITERATURE SURVEY", size=11, color=GREEN, bold=True, font_name="Consolas")
add_text(s, 0.7, 0.9, 10, 0.5, "Research Papers (IEEE / Scopus Indexed)", size=28, color=WARM, bold=True)

tbl = add_table(s, 0.7, 1.6, 12, 11, 5, [0.4, 4.5, 0.6, 2.2, 4.3])
headers = ["#", "Title & Authors", "Year", "Publication", "Key Contribution"]
for i, h in enumerate(headers):
    set_cell(tbl, 0, i, h, 10, TX3, True)

papers = [
    ("1", "DeepDTA — Öztürk H. et al.", "2018", "Bioinformatics", "CNN on raw SMILES + protein sequence"),
    ("2", "GraphDTA — Nguyen T. et al.", "2021", "Bioinformatics", "GNN on molecular graphs; captures topology"),
    ("3", "KIBA Score — Tang J. et al.", "2014", "J. Chem. Inf. Model.", "Unified bioactivity score (Ki + Kd + IC50)"),
    ("4", "KronRLS — Pahikkala T. et al.", "2015", "Brief. Bioinform.", "Cold-start evaluation protocol"),
    ("5", "SimBoost — He T. et al.", "2017", "J. Cheminform.", "Gradient boosting for DTI"),
    ("6", "TransformerCPI — Chen L. et al.", "2020", "Bioinformatics", "Transformer attention for CPI"),
    ("7", "DeepConv-DTI — Lee I. et al.", "2019", "PLOS Comput. Biol.", "Deep conv on protein subsequences"),
    ("8", "MGraphDTA — Yang Z. et al.", "2022", "Chemical Science", "Multi-scale GNN local + global features"),
    ("9", "AttentionDTA — Zhao Q. et al.", "2022", "IEEE/ACM Trans. CBB", "Attention on drug/protein sub-regions"),
    ("10", "SHAP — Lundberg S. et al.", "2017", "NeurIPS", "Shapley-value feature attribution"),
]
for i, row in enumerate(papers):
    for j, val in enumerate(row):
        c = WARM if j == 1 else TX
        set_cell(tbl, i+1, j, val, 11, c, j == 1)

# ════════════════════════════════════════════
# SLIDE 5: COMPARATIVE STUDY
# ════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
set_bg(s)
add_text(s, 0.7, 0.5, 5, 0.3, "COMPARATIVE STUDY", size=11, color=GREEN, bold=True, font_name="Consolas")
add_text(s, 0.7, 0.9, 10, 0.5, "Existing Systems vs PharmaLens", size=28, color=WARM, bold=True)

tbl = add_table(s, 0.7, 1.6, 12, 9, 5, [2.8, 2.0, 2.0, 2.0, 3.2])
for i, h in enumerate(["Feature", "DeepDTA '18", "GraphDTA '21", "MGraphDTA '22", "PharmaLens"]):
    c = GREEN if i == 4 else TX3
    set_cell(tbl, 0, i, h, 10, c, True)

comp_data = [
    ("Multi-model comparison", "✗", "✗", "✗", "✓ 7 models"),
    ("Traditional ML baselines", "✗", "✗", "✗", "✓ Ridge, RF, XGBoost"),
    ("Deep Learning", "CNN", "GNN", "Multi-GNN", "✓ CNN + GNN"),
    ("Explainability", "✗", "✗", "✗", "✓ SHAP"),
    ("Cold-start evaluation", "✗", "✗", "Partial", "✓ Drug + Target"),
    ("Hyperparameter tuning", "Manual", "Manual", "Manual", "✓ Bayesian (Optuna)"),
    ("Web application", "✗", "✗", "✗", "✓ FastAPI + UI"),
    ("Batch screening", "✗", "✗", "✗", "✓ Drug × Target"),
]
for i, row in enumerate(comp_data):
    for j, val in enumerate(row):
        c = GREEN if j == 4 else (WARM if j == 0 else TX)
        set_cell(tbl, i+1, j, val, 11, c)

# ════════════════════════════════════════════
# SLIDE 6: TOOLS & TECHNOLOGIES
# ════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
set_bg(s)
add_text(s, 0.7, 0.5, 5, 0.3, "TOOLS & TECHNOLOGIES", size=11, color=GREEN, bold=True, font_name="Consolas")
add_text(s, 0.7, 0.9, 6, 0.5, "Stack", size=28, color=WARM, bold=True)

tools = [
    ("🐍 Language", "Python 3.12 · JavaScript ES6+ · HTML5/CSS3", 0.7, 1.7, 3.6),
    ("📊 ML & Data", "scikit-learn 1.3+ · XGBoost 2.0+ · Optuna 3.0+\nNumPy 1.24+ · Pandas 2.0+", 4.6, 1.7, 3.8),
    ("🧠 Deep Learning", "PyTorch 2.0+ · PyTorch Geometric 2.4+", 8.7, 1.7, 3.9),
    ("⚗️ Chemistry", "RDKit 2024.03+\nSMILES validation, descriptors, fingerprints, graphs", 0.7, 3.7, 3.6),
    ("💡 Explainability", "SHAP 0.43+\nTreeExplainer, summary, waterfall, dependence plots", 4.6, 3.7, 3.8),
    ("🌐 Web & Viz", "FastAPI 0.100+ · Uvicorn 0.23+\nMatplotlib 3.7+ · Seaborn 0.12+", 8.7, 3.7, 3.9),
]
for title, body, x, y, w in tools:
    add_rect(s, x, y, w, 1.7, CARD_BG, BORDER)
    add_text(s, x + 0.2, y + 0.12, w - 0.4, 0.3, title, size=13, color=WARM, bold=True)
    add_text(s, x + 0.2, y + 0.5, w - 0.4, 1.1, body, size=11, color=TX)

# ════════════════════════════════════════════
# SLIDE 7: FUNCTIONAL REQUIREMENTS
# ════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
set_bg(s)
add_text(s, 0.7, 0.5, 5, 0.3, "FUNCTIONAL REQUIREMENTS", size=11, color=GREEN, bold=True, font_name="Consolas")
add_text(s, 0.7, 0.9, 8, 0.5, "System Capabilities", size=28, color=WARM, bold=True)

tbl = add_table(s, 0.7, 1.6, 12, 13, 2, [1.2, 10.8])
set_cell(tbl, 0, 0, "ID", 10, TX3, True)
set_cell(tbl, 0, 1, "Requirement", 10, TX3, True)

frs = [
    ("FR-01", "The system shall allow users to input drug molecules via SMILES string or by selecting from the KIBA dataset."),
    ("FR-02", "The system shall allow users to input protein targets via amino acid sequence or by selecting from the dataset."),
    ("FR-03", "The system shall predict binding affinity scores for a given drug–target pair using the trained ML model."),
    ("FR-04", "The system shall validate user inputs (SMILES validity via RDKit, sequence format) and display appropriate error messages."),
    ("FR-05", "The system shall perform batch screening across multiple drug × target combinations and return ranked results."),
    ("FR-06", "The system shall provide SHAP-based explanations for each prediction, showing which molecular features contributed most."),
    ("FR-07", "The system shall display drug information including MW, LogP, TPSA, HBD, HBA, and structural fingerprint details."),
    ("FR-08", "The system shall allow users to compare multiple ML models (Ridge, RF, XGBoost, DeepDTA, GraphDTA) on evaluation metrics."),
    ("FR-09", "The system shall display dataset exploration views including score distributions, drug/target counts, and statistics."),
    ("FR-10", "The system shall expose a REST API (FastAPI) for programmatic access to prediction, screening, and explanation endpoints."),
    ("FR-11", "The system shall provide a web-based dashboard for interactive prediction, screening, model comparison, and exploration."),
    ("FR-12", "The system shall display appropriate disclaimers indicating predictions are computational and do not replace lab validation."),
]
for i, (fid, desc) in enumerate(frs):
    set_cell(tbl, i+1, 0, fid, 11, WARM, True)
    set_cell(tbl, i+1, 1, desc, 11, TX)

# ════════════════════════════════════════════
# SLIDE 8: NON-FUNCTIONAL REQUIREMENTS
# ════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
set_bg(s)
add_text(s, 0.7, 0.5, 5, 0.3, "NON-FUNCTIONAL REQUIREMENTS", size=11, color=GREEN, bold=True, font_name="Consolas")
add_text(s, 0.7, 0.9, 8, 0.5, "Quality Attributes", size=28, color=WARM, bold=True)

tbl = add_table(s, 0.7, 1.6, 12, 15, 2, [2.5, 9.5])
set_cell(tbl, 0, 0, "ID", 10, TX3, True)
set_cell(tbl, 0, 1, "Requirement", 10, TX3, True)

nfrs = [
    ("NFR-01 — Performance", "Single predictions shall be returned within 100ms; batch screening of 200 pairs within 5 seconds."),
    ("NFR-02 — Availability", "The system should remain available and functional during normal operating hours with minimal downtime."),
    ("NFR-03 — Usability", "The web interface should be simple, intuitive, and responsive, allowing users to submit predictions with minimal steps."),
    ("NFR-04 — Security", "API endpoints shall be protected with appropriate input sanitization and rate limiting to prevent misuse."),
    ("NFR-05 — Accuracy", "Predictions shall originate from models trained on the validated KIBA benchmark dataset with documented metrics."),
    ("NFR-06 — Reliability", "The system should handle invalid SMILES, malformed sequences, and errors without crashing."),
    ("NFR-07 — Reproducibility", "All experiments shall use fixed random seeds (42) and persisted splits to ensure reproducible results."),
    ("NFR-08 — Scalability", "The architecture should support increase in drug records, targets, and API requests without major redesign."),
    ("NFR-09 — Modularity", "The software shall use a modular architecture so preprocessing, features, models, evaluation can be independently maintained."),
    ("NFR-10 — Compatibility", "The web application should work correctly on commonly used modern browsers (Chrome, Firefox, Edge)."),
    ("NFR-11 — Responsiveness", "The UI should adapt appropriately to desktop and tablet screen sizes."),
    ("NFR-12 — Interpretability", "Every prediction shall be accompanied by a SHAP-based explanation so users understand the reasoning."),
    ("NFR-13 — Extensibility", "The system should allow future integration of additional datasets, models, or drug databases via modular pipeline."),
    ("NFR-14 — Data Integrity", "Model artifacts and dataset files should not be accidentally modified; checksums and version tracking maintained."),
]
for i, (nid, desc) in enumerate(nfrs):
    set_cell(tbl, i+1, 0, nid, 10, WARM, True)
    set_cell(tbl, i+1, 1, desc, 11, TX)

# ════════════════════════════════════════════
# SLIDE 9: ARCHITECTURE
# ════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
set_bg(s)
add_text(s, 0.7, 0.5, 5, 0.3, "DESIGN", size=11, color=GREEN, bold=True, font_name="Consolas")
add_text(s, 0.7, 0.9, 10, 0.5, "System Architecture — Component Diagram", size=28, color=WARM, bold=True)

# Row 1: Data flow
boxes_r1 = [
    ("«datastore»\nKIBA Dataset", TX, 1.5, 1.7),
    ("«component»\nPreprocessing", TX, 5.0, 1.7),
    ("«component»\nFeature Engineering", TX, 8.5, 1.7),
]
for label, color, x, y in boxes_r1:
    add_rect(s, x, y, 2.5, 0.8, RGBColor(0x12,0x12,0x12), BORDER)
    add_text(s, x, y + 0.05, 2.5, 0.7, label, size=11, color=color, align=PP_ALIGN.CENTER)
add_text(s, 4.0, 1.9, 1, 0.4, "──▶", size=14, color=TX3, align=PP_ALIGN.CENTER)
add_text(s, 7.5, 1.9, 1, 0.4, "──▶", size=14, color=TX3, align=PP_ALIGN.CENTER)

# Row 2: Features
add_rect(s, 2.5, 2.8, 3.5, 0.6, RGBColor(0x12,0x12,0x12), AMBER)
add_text(s, 2.5, 2.85, 3.5, 0.5, "💊 Drug: Descriptors + FP → 1034D", size=11, color=AMBER, align=PP_ALIGN.CENTER)
add_rect(s, 7.3, 2.8, 3.5, 0.6, RGBColor(0x12,0x12,0x12), CORAL)
add_text(s, 7.3, 2.85, 3.5, 0.5, "🧬 Protein: AAC + DPC → 420D", size=11, color=CORAL, align=PP_ALIGN.CENTER)

add_text(s, 4.5, 3.5, 4.3, 0.3, "↓  concat → 1454D  ↓", size=12, color=TX3, align=PP_ALIGN.CENTER)

# Row 3: Models
models = [
    ("Ridge", TX, BORDER, 1.5),
    ("Random Forest", TX, BORDER, 3.5),
    ("★ XGBoost", GREEN, GREEN2, 5.5),
    ("DeepDTA", AMBER, AMBER, 7.5),
    ("GraphDTA", AMBER, AMBER, 9.5),
]
for label, color, bdr, x in models:
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(4.0), Inches(1.8), Inches(0.55)) if False else add_rect(s, x, 4.0, 1.8, 0.55, RGBColor(0x12,0x12,0x12), bdr)
    add_text(s, x, 4.05, 1.8, 0.45, label, size=11, color=color, align=PP_ALIGN.CENTER, bold=True)

add_text(s, 5.5, 4.65, 2.3, 0.3, "▼", size=16, color=TX3, align=PP_ALIGN.CENTER)

# Row 4: Eval + SHAP
add_rect(s, 3.0, 5.1, 3.0, 0.65, RGBColor(0x12,0x12,0x12), BORDER)
add_text(s, 3.0, 5.15, 3.0, 0.55, "«component»\n📈 Evaluation", size=11, color=TX, align=PP_ALIGN.CENTER)
add_text(s, 6.0, 5.25, 1, 0.4, "──▶", size=14, color=TX3, align=PP_ALIGN.CENTER)
add_rect(s, 7.0, 5.1, 3.0, 0.65, RGBColor(0x12,0x12,0x12), BORDER)
add_text(s, 7.0, 5.15, 3.0, 0.55, "«component»\n🧠 SHAP Explainability", size=11, color=TX, align=PP_ALIGN.CENTER)

add_text(s, 5.5, 5.85, 2.3, 0.3, "▼", size=16, color=TX3, align=PP_ALIGN.CENTER)

# Row 5: Deploy
add_rect(s, 3.5, 6.2, 2.5, 0.65, RGBColor(0x12,0x12,0x12), GREEN2)
add_text(s, 3.5, 6.25, 2.5, 0.55, "«service»\n🔌 FastAPI Backend", size=11, color=GREEN, align=PP_ALIGN.CENTER)
add_text(s, 6.0, 6.35, 1, 0.4, "──▶", size=14, color=TX3, align=PP_ALIGN.CENTER)
add_rect(s, 7.0, 6.2, 2.5, 0.65, RGBColor(0x12,0x12,0x12), GREEN2)
add_text(s, 7.0, 6.25, 2.5, 0.55, "«interface»\n🖥️ Web Dashboard", size=11, color=GREEN, align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════
# SLIDE 10: PROCESS FLOW
# ════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
set_bg(s)
add_text(s, 0.7, 0.5, 5, 0.3, "DESIGN", size=11, color=GREEN, bold=True, font_name="Consolas")
add_text(s, 0.7, 0.9, 10, 0.5, "ML Process Flow — Activity Diagram", size=28, color=WARM, bold=True)

# Start node
circle = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(3.2), Inches(1.55), Inches(0.25), Inches(0.25))
circle.fill.solid(); circle.fill.fore_color.rgb = GREEN; circle.line.fill.background()

left_steps = [
    ("① Data Collection — KIBA", TX, BORDER, 1.95),
    ("② Exploratory Data Analysis", TX, BORDER, 2.7),
    ("③ Preprocessing & Cleaning", TX, BORDER, 3.45),
    ("④ Feature Engineering", AMBER, AMBER, 4.2),
    ("⑤ Train / Val / Test Split", AMBER, AMBER, 4.95),
]
for label, color, bdr, y in left_steps:
    add_rect(s, 1.0, y, 5.0, 0.55, RGBColor(0x12,0x12,0x12), bdr)
    add_text(s, 1.0, y + 0.05, 5.0, 0.45, label, size=12, color=color, align=PP_ALIGN.CENTER)
for y in [2.55, 3.3, 4.05, 4.8]:
    add_text(s, 3.2, y, 0.5, 0.2, "↓", size=12, color=TX3, align=PP_ALIGN.CENTER)

right_steps = [
    ("⑥ Model Training (×7)", GREEN, GREEN2, 1.95),
    ("⑦ Hyperparameter Tuning", GREEN, GREEN2, 2.7),
    ("⑧ Cold-Start Evaluation", CORAL, CORAL, 3.45),
    ("⑨ SHAP Explainability", TX, BORDER, 4.2),
    ("⑩ Deployment (API + Web)", GREEN, GREEN2, 4.95),
]
for label, color, bdr, y in right_steps:
    add_rect(s, 7.3, y, 5.0, 0.55, RGBColor(0x12,0x12,0x12), bdr)
    add_text(s, 7.3, y + 0.05, 5.0, 0.45, label, size=12, color=color, align=PP_ALIGN.CENTER)
for y in [2.55, 3.3, 4.05, 4.8]:
    add_text(s, 9.5, y, 0.5, 0.2, "↓", size=12, color=TX3, align=PP_ALIGN.CENTER)

# End node
outer = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(9.6), Inches(5.65), Inches(0.25), Inches(0.25))
outer.fill.background(); outer.line.color.rgb = GREEN; outer.line.width = Pt(2)
inner = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(9.66), Inches(5.71), Inches(0.13), Inches(0.13))
inner.fill.solid(); inner.fill.fore_color.rgb = GREEN; inner.line.fill.background()

# ════════════════════════════════════════════
# SLIDE 11: DATASET & FEATURES
# ════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
set_bg(s)
add_text(s, 0.7, 0.5, 5, 0.3, "IMPLEMENTATION", size=11, color=GREEN, bold=True, font_name="Consolas")
add_text(s, 0.7, 0.9, 8, 0.5, "Dataset & Feature Engineering", size=28, color=WARM, bold=True)

add_stat_box(s, 0.7, 1.7, 2.7, 1.2, "118,254", "INTERACTIONS", GREEN)
add_stat_box(s, 3.7, 1.7, 2.7, 1.2, "2,111", "DRUGS", AMBER)
add_stat_box(s, 6.7, 1.7, 2.7, 1.2, "229", "TARGETS", CORAL)
add_stat_box(s, 9.7, 1.7, 2.9, 1.2, "0 – 17.2", "SCORE RANGE", WARM)

add_rect(s, 0.7, 3.3, 5.7, 2.8, CARD_BG, BORDER)
add_text(s, 0.9, 3.4, 5.3, 0.35, "💊 Drug Representation → 1,034D", size=14, color=WARM, bold=True)
add_text(s, 0.9, 3.85, 5.3, 2, "• 10 molecular descriptors: MW, LogP, TPSA, HBD, HBA, rotatable bonds, aromatic rings, Csp3, heavy atoms, ring count\n\n• 1,024-bit Morgan fingerprint: circular substructure encoding (ECFP4, radius=2)", size=12, color=TX)

add_rect(s, 6.9, 3.3, 5.7, 2.8, CARD_BG, BORDER)
add_text(s, 7.1, 3.4, 5.3, 0.35, "🧬 Protein Representation → 420D", size=14, color=WARM, bold=True)
add_text(s, 7.1, 3.85, 5.3, 2, "• AAC (20D): frequency of each standard amino acid\n\n• DPC (400D): frequency of each consecutive amino acid pair (20 × 20)", size=12, color=TX)

add_text(s, 0.7, 6.3, 12, 0.4, "Per pair: concat(drug₁₀₃₄, protein₄₂₀) = 1,454-dimensional input vector", size=13, color=TX)

# ════════════════════════════════════════════
# SLIDE 12: MODEL RESULTS
# ════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
set_bg(s)
add_text(s, 0.7, 0.5, 5, 0.3, "IMPLEMENTATION", size=11, color=GREEN, bold=True, font_name="Consolas")
add_text(s, 0.7, 0.9, 10, 0.5, "Model Comparison — RMSE (lower is better)", size=28, color=WARM, bold=True)

bar_data = [
    ("★ XGBoost Tuned", 0.421, 0.841, GDIM, GREEN),
    ("DeepDTA", 0.425, 0.841, ADIM, AMBER),
    ("GraphDTA", 0.430, 0.841, ADIM, AMBER),
    ("XGBoost", 0.435, 0.841, GDIM, GREEN),
    ("Random Forest", 0.487, 0.841, RGBColor(0x1e,0x1e,0x1e), TX2),
    ("Ridge", 0.655, 0.841, CDIM, CORAL),
    ("Mean Predictor", 0.841, 0.841, RGBColor(0x1e,0x1e,0x1e), TX3),
]
bar_max_w = 8.5
for i, (name, val, maxv, fill, tcol) in enumerate(bar_data):
    y = 1.65 + i * 0.52
    nc = GREEN if i == 0 else TX
    add_text(s, 0.3, y, 1.8, 0.38, name, size=11, color=nc, align=PP_ALIGN.RIGHT)
    bw = (val / maxv) * bar_max_w
    add_bar(s, 2.3, y, bw, bar_max_w, 0.38, str(val), fill, tcol)

# Stats row
add_stat_box(s, 0.7, 5.6, 2.8, 1.1, "0.867", "BEST PEARSON r", GREEN)
add_stat_box(s, 3.8, 5.6, 2.8, 1.1, "0.750", "BEST R²", AMBER)
add_stat_box(s, 6.9, 5.6, 2.8, 1.1, "5.4 min", "BEST TRAIN TIME", WARM)
add_stat_box(s, 10.0, 5.6, 2.6, 1.1, "100×", "FASTER THAN DL", CORAL)

# ════════════════════════════════════════════
# SLIDE 13: TUNING & COLD-START
# ════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
set_bg(s)
add_text(s, 0.7, 0.5, 5, 0.3, "IMPLEMENTATION", size=11, color=GREEN, bold=True, font_name="Consolas")
add_text(s, 0.7, 0.9, 10, 0.5, "Tuning & Cold-Start", size=28, color=WARM, bold=True)

# LEFT: Tuning card
add_rect(s, 0.7, 1.6, 6.0, 5.2, CARD_BG, BORDER)
add_text(s, 0.9, 1.7, 5.6, 0.35, "⚙️ Optuna Tuning — 50 Trials", size=14, color=WARM, bold=True)

metrics_tune = [("RMSE ↓", 0.435, 0.421, "↓3.2%"), ("Pearson r ↑", 0.857, 0.867, "↑+0.01"), ("R² ↑", 0.733, 0.750, "↑+0.017")]
for i, (metric, before, after, delta) in enumerate(metrics_tune):
    y = 2.3 + i * 1.5
    add_text(s, 0.9, y, 5, 0.25, metric, size=11, color=TX2, font_name="Consolas")
    # Before bar
    add_text(s, 0.9, y + 0.3, 0.6, 0.3, "Before", size=9, color=TX3, align=PP_ALIGN.RIGHT)
    bw_before = (before / 1.0) * 4.5
    add_bar(s, 1.6, y + 0.3, bw_before, 4.8, 0.3, str(before), CDIM if metric.startswith("RMSE") else ADIM, CORAL if metric.startswith("RMSE") else AMBER)
    # After bar
    add_text(s, 0.9, y + 0.65, 0.6, 0.3, "After", size=9, color=GREEN, align=PP_ALIGN.RIGHT)
    bw_after = (after / 1.0) * 4.5
    add_bar(s, 1.6, y + 0.65, bw_after, 4.8, 0.3, f"{after} {delta}", GDIM, GREEN)

# RIGHT: Cold-start vertical bars
add_rect(s, 7.0, 1.6, 5.7, 5.2, CARD_BG, BORDER)
add_text(s, 7.2, 1.7, 5.3, 0.35, "❄️ Cold-Start RMSE — XGBoost", size=14, color=WARM, bold=True)

cold_bars = [
    ("Random", 0.435, GREEN, GREEN2),
    ("Cold-Drug", 0.574, AMBER, AMBER),
    ("Cold-Target", 0.593, CORAL, CORAL),
]
bar_area_h = 3.0
max_val = 0.8
for i, (label, val, tcol, bcol) in enumerate(cold_bars):
    x = 8.0 + i * 1.6
    h = (val / max_val) * bar_area_h
    y = 2.4 + (bar_area_h - h)
    # value label
    add_text(s, x, y - 0.3, 1.0, 0.25, str(val), size=11, color=tcol, bold=True, align=PP_ALIGN.CENTER, font_name="Consolas")
    # bar
    bar_shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(1.0), Inches(h))
    bar_shape.fill.solid()
    # Darker version of the border color
    r_val = int(str(bcol)[0:2], 16) // 3
    g_val = int(str(bcol)[2:4], 16) // 3
    b_val = int(str(bcol)[4:6], 16) // 3
    bar_shape.fill.fore_color.rgb = RGBColor(r_val, g_val, b_val)
    bar_shape.line.color.rgb = bcol
    bar_shape.line.width = Pt(1)
    bar_shape.adjustments[0] = 0.05
    # label
    add_text(s, x, 2.4 + bar_area_h + 0.05, 1.0, 0.25, label, size=11, color=TX, align=PP_ALIGN.CENTER)

add_text(s, 7.5, 5.9, 4.8, 0.3, "▲ +32% drop          ▲ +36% drop", size=11, color=CORAL, align=PP_ALIGN.CENTER)
add_text(s, 7.5, 6.25, 4.8, 0.3, "Unseen proteins harder than unseen drugs", size=10, color=TX3, align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════
# SLIDE 14: SHAP
# ════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
set_bg(s)
add_text(s, 0.7, 0.5, 5, 0.3, "IMPLEMENTATION", size=11, color=GREEN, bold=True, font_name="Consolas")
add_text(s, 0.7, 0.9, 8, 0.5, "SHAP Explainability", size=28, color=WARM, bold=True)

# Feature importance bars
add_text(s, 0.7, 1.6, 5, 0.35, "Feature Category Importance", size=15, color=WARM, bold=True)

add_text(s, 0.7, 2.2, 1.5, 0.35, "💊 Drug", size=13, color=TX, align=PP_ALIGN.RIGHT)
add_bar(s, 2.3, 2.2, 4.5 * 0.67, 4.5, 0.4, "~67%", GDIM, GREEN)

add_text(s, 0.7, 2.75, 1.5, 0.35, "🧬 Protein", size=13, color=TX, align=PP_ALIGN.RIGHT)
add_bar(s, 2.3, 2.75, 4.5 * 0.33, 4.5, 0.4, "~33%", ADIM, AMBER)

add_text(s, 0.7, 3.4, 6, 1, "Drug structural features dominate predictions —\nmolecular substructures determine binding pocket fit.\nConsistent with biochemical expectation.", size=12, color=TX2)

# Key findings card
add_rect(s, 7.0, 1.6, 5.7, 4.5, CARD_BG, BORDER)
add_text(s, 7.2, 1.7, 5.3, 0.35, "Key Findings", size=14, color=WARM, bold=True)
add_text(s, 7.2, 2.2, 5.3, 3.5,
    "• Morgan FP bits — strongest predictors;\n  specific substructures drive binding\n\n"
    "• LogP, MW, TPSA — top molecular descriptors;\n  lipophilicity and size matter\n\n"
    "• Dipeptide composition — main protein-side signal\n\n"
    "• Every prediction decomposable via waterfall plots\n\n"
    "• 37 publication-ready figures generated",
    size=12, color=TX)

# ════════════════════════════════════════════
# SLIDE 15: REFERENCES
# ════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
set_bg(s)
add_text(s, 0.7, 0.5, 5, 0.3, "REFERENCES", size=11, color=GREEN, bold=True, font_name="Consolas")
add_text(s, 0.7, 0.9, 6, 0.5, "IEEE Format", size=28, color=WARM, bold=True)

refs = [
    '[1] H. Öztürk et al., "DeepDTA: deep drug–target binding affinity prediction," Bioinformatics, vol. 34, pp. i821–i829, 2018.',
    '[2] T. Nguyen et al., "GraphDTA: predicting DTI with graph neural networks," Bioinformatics, vol. 37, pp. 1140–1147, 2021.',
    '[3] J. Tang et al., "Making sense of large-scale kinase inhibitor bioactivity," J. Chem. Inf. Model., vol. 54, pp. 735–743, 2014.',
    '[4] T. Pahikkala et al., "Toward more realistic DTI predictions," Brief. Bioinform., vol. 16, pp. 325–337, 2015.',
    '[5] T. He et al., "SimBoost: predicting drug–target binding affinities," J. Cheminform., vol. 9, p. 24, 2017.',
    '[6] L. Chen et al., "TransformerCPI," Bioinformatics, vol. 36, pp. 4406–4414, 2020.',
    '[7] I. Lee et al., "DeepConv-DTI," PLOS Comput. Biol., vol. 15, e1007129, 2019.',
    '[8] Z. Yang et al., "MGraphDTA," Chem. Sci., vol. 13, pp. 816–833, 2022.',
    '[9] Q. Zhao et al., "AttentionDTA," IEEE/ACM Trans. CBB, vol. 20, pp. 228–237, 2022.',
    '[10] S. Lundberg, S. Lee, "A unified approach to interpreting model predictions," NeurIPS, pp. 4765–4774, 2017.',
    '',
    'Software: [11] RDKit — rdkit.org   [12] PyG — pyg.org   [13] SHAP — shap.readthedocs.io',
]
add_text(s, 0.7, 1.5, 12, 5.5, "\n".join(refs), size=12, color=TX)

# ════════════════════════════════════════════
# SLIDE 16: THANK YOU
# ════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
set_bg(s)
add_text(s, 0, 2.0, 13.333, 0.4, "PHASE 1 · 50% COMPLETE", size=12, color=GREEN, bold=True, align=PP_ALIGN.CENTER, font_name="Consolas")
add_text(s, 0, 2.7, 13.333, 1, "Thank You", size=54, color=WARM, bold=True, align=PP_ALIGN.CENTER)
add_text(s, 0, 3.9, 13.333, 0.5, "Questions & Discussion", size=20, color=TX2, align=PP_ALIGN.CENTER)
add_text(s, 0, 5.0, 13.333, 0.5, "💊 PharmaLens — Explainable AI for Drug–Target Interaction Prediction", size=14, color=TX3, align=PP_ALIGN.CENTER)

# ── Save ──
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PharmaLens_Presentation.pptx")
prs.save(out_path)
print(f"\n✅ Saved: {out_path}")
print(f"   16 slides · Dark theme · Matching HTML presentation")
