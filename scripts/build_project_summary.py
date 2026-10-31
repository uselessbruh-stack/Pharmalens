"""Build the project summary Word document from its maintained Markdown source."""

import json
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "docs/PROJECT_SUMMARY.md"
    content = source.read_text(encoding="utf-8")
    benchmark = json.loads((ROOT / "results/reports/final_benchmark_v1.json").read_text())
    names = {
        "NonKinasePanel": "Starter panel",
        "NonKinaseBroad": "Broad non-kinase",
        "KIBA": "KIBA",
        "NonKinase_gpcr": "GPCR",
    }
    rows = ["| Scope | Endpoint | XGBoost | CNN | GNN | Seeds |",
            "|---|---|---:|---:|---:|---:|"]
    for task in sorted(benchmark["tasks"], key=lambda t: tuple(t["scope"])):
        dataset, _, endpoint, _ = task["scope"]
        name = names.get(dataset, dataset.removeprefix("NonKinase_").replace("_", " ").title())
        architectures = task["architectures"]
        values = [f"{architectures[m]['test_mean_rmse']:.4f}" for m in ("xgboost", "deeptda", "gnn")]
        rows.append("| " + " | ".join([name, endpoint, *values,
                                       str(len(architectures["gnn"]["seeds"]))]) + " |")
    table_md = "\n".join(rows)
    content = re.sub(r"<!-- BENCHMARK_TABLE -->|<!-- BENCHMARK_START -->.*?<!-- BENCHMARK_END -->",
                     "<!-- BENCHMARK_START -->\n" + table_md + "\n<!-- BENCHMARK_END -->",
                     content, flags=re.S)
    source.write_text(content, encoding="utf-8")

    doc = Document()
    for border in list(doc.styles.element.iter(qn("w:pBdr"))):
        border.getparent().remove(border)
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)
    section.top_margin = section.bottom_margin = Inches(0.65)
    section.left_margin = section.right_margin = Inches(0.7)
    section.header_distance = section.footer_distance = Inches(0.28)
    for name in ("Normal", "Title", "Subtitle", "Heading 1", "Heading 2", "Heading 3"):
        style = doc.styles[name]
        style.font.name = "Calibri"
        style.font.color.rgb = RGBColor(0, 0, 0)
    normal = doc.styles["Normal"]
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.08
    doc.styles["Title"].font.size = Pt(26)
    doc.styles["Title"].paragraph_format.space_after = Pt(7)
    doc.styles["Heading 1"].font.size = Pt(19)
    doc.styles["Heading 1"].paragraph_format.space_after = Pt(10)
    doc.styles["Heading 2"].font.size = Pt(12)
    doc.styles["Heading 2"].paragraph_format.space_before = Pt(10)
    doc.styles["Heading 2"].paragraph_format.space_after = Pt(6)
    header = section.header.paragraphs[0]
    header.text = "PHARMALENS    /    RESEARCH IMPLEMENTATION REVIEW"
    header.runs[0].font.size = Pt(8)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("31 October 2026    |    ").font.size = Pt(8)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)

    def add_table(lines):
        values = [[v.strip() for v in line.strip().strip("|").split("|")] for line in lines]
        values = [values[0], *values[2:]]
        table = doc.add_table(rows=0, cols=len(values[0]))
        table.autofit = False
        count = len(values[0])
        widths = {3: [1.0, 2.8, 3.07], 4: [2.55, 1.44, 1.44, 1.44],
                  5: [2.0, 1.35, 1.25, 1.1, 1.27],
                  6: [2.15, 0.75, 1.1, 1.05, 1.05, 0.77],
                  2: [2.1, 4.77]}[count]
        numeric_family_table = values[0][0] == "Non-kinase family"
        if numeric_family_table:
            widths = [3.2, 1.835, 1.835]
        for col, width in zip(table.columns, widths):
            col.width = Inches(width)
        for i, vals in enumerate(values):
            row = table.add_row()
            if i == 0:
                repeat = OxmlElement("w:tblHeader")
                row._tr.get_or_add_trPr().append(repeat)
            no_split = OxmlElement("w:cantSplit")
            row._tr.get_or_add_trPr().append(no_split)
            for j, (cell, val) in enumerate(zip(row.cells, vals)):
                cell.width = Inches(widths[j])
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                cell.text = val
                props = cell._tc.get_or_add_tcPr()
                margins = OxmlElement("w:tcMar")
                for edge in ("top", "left", "bottom", "right"):
                    item = OxmlElement("w:" + edge)
                    item.set(qn("w:w"), "65" if edge in ("top", "bottom") else "85")
                    item.set(qn("w:type"), "dxa")
                    margins.append(item)
                props.append(margins)
                borders = OxmlElement("w:tcBorders")
                for edge in ("top", "left", "bottom", "right"):
                    item = OxmlElement("w:" + edge)
                    for key, value in (("val", "single"), ("sz", "4"), ("color", "D9D9D9")):
                        item.set(qn("w:" + key), value)
                    borders.append(item)
                props.append(borders)
                if i == 0:
                    shade = OxmlElement("w:shd")
                    shade.set(qn("w:fill"), "DCE6F1")
                    props.append(shade)
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(1)
                    p.paragraph_format.space_before = Pt(1)
                    p.paragraph_format.line_spacing = 1.0
                    if j > 0 and (count >= 4 or numeric_family_table):
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in p.runs:
                        run.font.size = Pt(9.5)
                        run.bold = i == 0
        doc.add_paragraph().paragraph_format.space_after = Pt(0)

    lines = content.splitlines()
    pos = 0
    while pos < len(lines):
        line = lines[pos].strip()
        if not line or line.startswith("<!--"):
            pos += 1
            continue
        if line.startswith("|"):
            block = []
            while pos < len(lines) and lines[pos].strip().startswith("|"):
                block.append(lines[pos])
                pos += 1
            add_table(block)
            continue
        if line.startswith("# "):
            doc.add_paragraph(line[2:], "Title")
        elif line.startswith("## "):
            doc.add_page_break()
            doc.add_paragraph(line[3:], "Heading 1")
        elif line.startswith("### "):
            doc.add_paragraph(line[4:], "Heading 2")
        else:
            paragraph = doc.add_paragraph(line)
            if line == "Research implementation and benchmark review":
                paragraph.style = doc.styles["Subtitle"]
                paragraph.runs[0].font.size = Pt(12)
        pos += 1
    destination = ROOT / "docs/PharmaLens_Project_Summary.docx"
    doc.save(destination)
    print(destination)


if __name__ == "__main__":
    main()
