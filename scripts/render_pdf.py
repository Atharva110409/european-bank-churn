"""
render_pdf.py
Renders reports/research_paper.md into reports/research_paper.pdf using
an IEEE two-column HTML template and headless Microsoft Edge.
"""

import os
import re
import subprocess


def render_markdown_to_ieee_pdf(
    md_path: str = "reports/research_paper.md",
    html_path: str = "reports/research_paper.html",
    pdf_path: str = "reports/research_paper.pdf"
) -> bool:
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Markdown source not found: {md_path}")

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    css = """
    @page {
        size: A4;
        margin: 18mm 15mm 18mm 15mm;
    }
    body {
        font-family: 'Times New Roman', Times, serif;
        font-size: 10pt;
        line-height: 1.25;
        color: #111;
        margin: 0;
        padding: 0;
    }
    .title-block {
        text-align: center;
        margin-bottom: 18px;
    }
    h1.paper-title {
        font-size: 16pt;
        font-weight: bold;
        margin-bottom: 8pt;
        line-height: 1.2;
    }
    .author-block {
        font-size: 9.5pt;
        margin-bottom: 12pt;
        color: #222;
        line-height: 1.3;
    }
    .abstract-box {
        margin: 0 10mm 14pt 10mm;
        padding: 4pt 0;
        border-top: 0.5pt solid #888;
        border-bottom: 0.5pt solid #888;
    }
    .abstract-box p {
        font-size: 9pt;
        line-height: 1.3;
        margin: 4pt 0;
        text-align: justify;
    }
    .abstract-title {
        font-weight: bold;
        font-style: italic;
    }
    .two-column {
        column-count: 2;
        column-gap: 7mm;
        text-align: justify;
    }
    h2 {
        font-size: 10.5pt;
        text-transform: uppercase;
        text-align: center;
        border-bottom: 0.5pt solid #666;
        margin-top: 12pt;
        margin-bottom: 6pt;
        font-weight: bold;
        break-after: avoid;
    }
    h3 {
        font-size: 9.5pt;
        font-style: italic;
        margin-top: 8pt;
        margin-bottom: 3pt;
        font-weight: bold;
        break-after: avoid;
    }
    p {
        margin-top: 0;
        margin-bottom: 5pt;
        text-indent: 1.2em;
    }
    p.no-indent {
        text-indent: 0;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 8pt 0;
        font-size: 7.5pt;
        line-height: 1.15;
        break-inside: avoid;
    }
    th, td {
        border-top: 0.5pt solid #444;
        border-bottom: 0.5pt solid #444;
        padding: 3pt 3pt;
        text-align: left;
    }
    th {
        font-weight: bold;
        background: #f4f4f4;
    }
    .ref-item {
        font-size: 8pt;
        margin-bottom: 4pt;
        text-indent: -1.5em;
        margin-left: 1.5em;
        line-height: 1.2;
    }
    """

    lines = md_text.split("\n")
    out = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>Research Paper</title>",
        f"<style>{css}</style>",
        "</head>",
        "<body>"
    ]

    i = 0
    n = len(lines)
    title = ""
    author_lines = []

    # Parse title and authors
    while i < n:
        line = lines[i].strip()
        if line.startswith("# "):
            title = line[2:].strip()
            i += 1
            continue
        if line.startswith("---"):
            i += 1
            break
        if line:
            clean_l = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", line)
            author_lines.append(clean_l)
        i += 1

    out.append("<div class='title-block'>")
    out.append(f"<h1 class='paper-title'>{title}</h1>")
    if author_lines:
        out.append(f"<div class='author-block'>{'<br>'.join(author_lines)}</div>")
    out.append("</div>")

    # Abstract box
    while i < n:
        line = lines[i].strip()
        if line.startswith("### Abstract"):
            out.append("<div class='abstract-box'>")
            i += 1
            abs_text = ""
            while i < n and not lines[i].strip().startswith("**Index Terms**"):
                if lines[i].strip():
                    abs_text += " " + lines[i].strip()
                i += 1
            abs_clean = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", abs_text.strip())
            abs_clean = re.sub(r"\*(.*?)\*", r"<em>\1</em>", abs_clean)
            abs_clean = re.sub(r"\$(.*?)\$", r"<em>\1</em>", abs_clean)
            out.append(f"<p><span class='abstract-title'>Abstract—</span>{abs_clean}</p>")
            if i < n and lines[i].strip().startswith("**Index Terms**"):
                terms = lines[i].strip().replace("**Index Terms**", "").replace("—", "").replace("--", "").strip()
                out.append(f"<p class='no-indent'><span class='abstract-title'>Index Terms—</span>{terms}</p>")
                i += 1
            out.append("</div>")
            break
        i += 1

    out.append("<div class='two-column'>")

    table_lines = []
    in_table = False

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("|"):
            in_table = True
            table_lines.append(stripped)
            i += 1
            continue
        elif in_table:
            in_table = False
            headers = [c.strip() for c in table_lines[0].strip("|").split("|")]
            rows = [[c.strip() for c in r.strip("|").split("|")] for r in table_lines[2:]]
            tbl_html = "<table><thead><tr>"
            for h in headers:
                clean_h = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", h)
                tbl_html += f"<th>{clean_h}</th>"
            tbl_html += "</tr></thead><tbody>"
            for r in rows:
                tbl_html += "<tr>"
                for c in r:
                    clean_c = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", c)
                    clean_c = re.sub(r"\*(.*?)\*", r"<em>\1</em>", clean_c)
                    tbl_html += f"<td>{clean_c}</td>"
                tbl_html += "</tr>"
            tbl_html += "</tbody></table>"
            out.append(tbl_html)
            table_lines = []

        if stripped == "---":
            i += 1
            continue

        if stripped.startswith("## "):
            out.append(f"<h2>{stripped[3:]}</h2>")
            i += 1
            continue
        if stripped.startswith("### "):
            out.append(f"<h3>{stripped[4:]}</h3>")
            i += 1
            continue
        if stripped.startswith("[") and "] " in stripped and (stripped[1].isdigit() or stripped[2].isdigit()):
            ref_clean = re.sub(r"\*(.*?)\*", r"<em>\1</em>", stripped)
            out.append(f"<div class='ref-item'>{ref_clean}</div>")
            i += 1
            continue
        if stripped.startswith("$$") and stripped.endswith("$$"):
            eq_text = stripped[2:-2].strip()
            eq_text = eq_text.replace(r"\chi^2", "χ²")
            eq_text = eq_text.replace(r"\sum_{i=1}^{r}", "∑<sub>i=1</sub><sup>r</sup>")
            eq_text = eq_text.replace(r"\sum_{j=1}^{c}", "∑<sub>j=1</sub><sup>c</sup>")
            eq_text = eq_text.replace(r"\frac{(O_{ij} - E_{ij})^2}{E_{ij}}", "((O<sub>ij</sub> − E<sub>ij</sub>)² / E<sub>ij</sub>)")
            eq_text = eq_text.replace(r"\sqrt{\frac{\chi^2}{N \min(r - 1, c - 1)}}", "√[ χ² / (N · min(r − 1, c − 1)) ]")
            eq_text = eq_text.replace(r"\min \left( 1, \min_{k \ge i} \frac{m}{k} p_{(k)} \right)", "min(1, min<sub>k≥i</sub> (m/k) · p<sub>(k)</sub>)")
            eq_text = eq_text.replace(r"\left[ \hat{\theta}_{(0.025B)}, \hat{\theta}_{(0.975B)} \right]", "[ θ̂<sub>(0.025B)</sub>, θ̂<sub>(0.975B)</sub> ]")
            eq_text = eq_text.replace(r"\overline{\text{AP}}_{\text{HGB}} - \overline{\text{AP}}_{\text{LR}} \le 0.01; \text{ otherwise select HGB.}", "AP̄<sub>HGB</sub> − AP̄<sub>LR</sub> ≤ 0.01; otherwise select HGB.")
            eq_text = eq_text.replace(r"\text{Select logistic regression if }", "Select logistic regression if ")
            eq_text = eq_text.replace(r"\quad", " &nbsp;&nbsp;&nbsp;&nbsp; ")
            out.append(f"<p class='no-indent' style='text-align: center; margin: 6pt 0; font-family: Cambria, \"Times New Roman\", serif;'><em>{eq_text}</em></p>")
            i += 1
            continue
        if stripped:
            p_line = stripped
            p_line = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", p_line)
            p_line = re.sub(r"\*(.*?)\*", r"<em>\1</em>", p_line)
            p_line = re.sub(r"\$(.*?)\$", r"<em>\1</em>", p_line)
            out.append(f"<p>{p_line}</p>")
        i += 1

    if in_table and table_lines:
        headers = [c.strip() for c in table_lines[0].strip("|").split("|")]
        rows = [[c.strip() for c in r.strip("|").split("|")] for r in table_lines[2:]]
        tbl_html = "<table><thead><tr>"
        for h in headers:
            clean_h = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", h)
            tbl_html += f"<th>{clean_h}</th>"
        tbl_html += "</tr></thead><tbody>"
        for r in rows:
            tbl_html += "<tr>"
            for c in r:
                clean_c = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", c)
                clean_c = re.sub(r"\*(.*?)\*", r"<em>\1</em>", clean_c)
                tbl_html += f"<td>{clean_c}</td>"
            tbl_html += "</tr>"
        tbl_html += "</tbody></table>"
        out.append(tbl_html)

    out.append("</div>")
    out.append("</body>")
    out.append("</html>")

    full_html = "\n".join(out)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    # Invoke headless Edge
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
    ]
    edge_exe = next((p for p in edge_paths if os.path.exists(p)), None)
    if not edge_exe:
        print("Microsoft Edge executable not found. PDF conversion skipped.")
        return False

    abs_pdf = os.path.abspath(pdf_path)
    abs_html = os.path.abspath(html_path)
    cmd = [
        edge_exe,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={abs_pdf}",
        f"file:///{abs_html}"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(abs_pdf) and os.path.getsize(abs_pdf) > 0:
        print(f"Successfully rendered PDF: {abs_pdf} ({os.path.getsize(abs_pdf)} bytes)")
        return True
    else:
        print(f"Edge execution finished with code {result.returncode}, but PDF not generated.")
        return False


if __name__ == "__main__":
    render_markdown_to_ieee_pdf()
