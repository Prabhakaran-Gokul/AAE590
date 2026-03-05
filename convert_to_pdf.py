from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

with open("aae590_ps03.py", "r") as f:
    for line in f:
        pdf.write(5, line)
        pdf.ln()

pdf.output("aae590_ps03.pdf")
