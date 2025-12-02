import io
import tempfile
from fpdf import FPDF
import matplotlib.pyplot as plt


class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Neuro-TopOpt: Engineering Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def clean_text(text):
    """
    Helper to remove emojis and non-latin characters that break the PDF.
    Replaces common emojis with text equivalents.
    """
    text = str(text)
    replacements = {
        "✅": "[PASS] ",
        "⚠️": "[WARN] ",
        "❌": "[FAIL] ",
        "🚀": "",
        "🏗️": "",
        "🏭": "",
        "📄": ""
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    # Final safety net: Remove any other strange characters
    return text.encode('latin-1', 'replace').decode('latin-1')


def generate_pdf(input_grid, optimized_grid, metrics_dict, load_x, load_y):
    """
    Generates a PDF report with project details and plots.
    """
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # --- SECTION 1: PROJECT SUMMARY ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "1. Project Summary", 0, 1)
    pdf.set_font("Arial", size=10)

    # Write metrics with text cleaning
    for key, value in metrics_dict.items():
        # Clean both key and value to ensure no emojis sneak in
        safe_key = clean_text(key)
        safe_value = clean_text(value)
        pdf.cell(0, 8, f"{safe_key}: {safe_value}", 0, 1)
    pdf.ln(5)

    # --- SECTION 2: BOUNDARY CONDITIONS ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "2. Boundary Conditions Setup", 0, 1)

    # Generate Plot 1
    fig1, ax1 = plt.subplots(figsize=(6, 2))
    ax1.imshow(1 - input_grid, cmap='gray')  # Invert for White Background
    ax1.scatter([load_x], [load_y], c='red', s=50, label='Load')
    ax1.legend()
    ax1.axis('off')
    ax1.set_title("Load & Support Configuration")

    # Save to temp file for PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        fig1.savefig(tmpfile.name, bbox_inches='tight', dpi=150)
        pdf.image(tmpfile.name, x=10, w=170)
    plt.close(fig1)  # Clean up memory
    pdf.ln(5)

    # --- SECTION 3: OPTIMIZED TOPOLOGY ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "3. AI-Optimized Topology", 0, 1)

    # Generate Plot 2 (The Design)
    fig2, ax2 = plt.subplots(figsize=(6, 2))
    ax2.imshow(1 - optimized_grid, cmap='gray')
    ax2.axis('off')
    ax2.set_title("Final Structural Layout")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        fig2.savefig(tmpfile.name, bbox_inches='tight', dpi=150)
        pdf.image(tmpfile.name, x=10, w=170)
    plt.close(fig2)

    # Output to bytes
    return pdf.output(dest='S').encode('latin-1')
