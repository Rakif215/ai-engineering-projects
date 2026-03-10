from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import textwrap

c = canvas.Canvas("Clinical_Guidelines_2024.pdf", pagesize=letter)
c.setFont("Helvetica-Bold", 16)
c.drawString(50, 750, "Clinical Practice Guidelines - Cardiology (2024)")

c.setFont("Helvetica-Bold", 14)
c.drawString(50, 710, "1. Protocol: Acute Myocardial Infarction")
c.setFont("Helvetica", 12)
text = "Patients presenting with Acute Myocardial Infarction should immediately receive administered aspirin (162-325 mg) and P2Y12 receptor inhibitors unless contraindicated. Reperfusion therapy should be initiated within 90 minutes for PCI-capable facilities. Continuous ECG monitoring is required."
y = 690
for line in textwrap.wrap(text, 90):
    c.drawString(50, y, line)
    y -= 15

c.setFont("Helvetica-Bold", 14)
y -= 30
c.drawString(50, y, "2. Protocol: Hypertension Management")
c.setFont("Helvetica", 12)
filler = "First-line antihypertensive medications include thiazide diuretics, calcium channel blockers, and ACE inhibitors or ARBs. Target blood pressure for most adults is < 130/80 mm Hg. Lifestyle modifications remain a cornerstone of early-stage management, including sodium restriction and regular aerobic activity."
y -= 20
for line in textwrap.wrap(filler, 90):
    c.drawString(50, y, line)
    y -= 15

c.setFont("Helvetica-Bold", 14)
y -= 30
c.drawString(50, y, "3. Protocol: Heart Failure Exacerbation")
c.setFont("Helvetica", 12)
text2 = "For patients with acute decompensated heart failure, intravenous loop diuretics are the primary treatment to relieve congestive symptoms. Vasodilators may be considered in patients with excessively high blood pressure, but strictly avoided in hypotensive states. Regular assessment of renal function is critical."
y -= 20
for line in textwrap.wrap(text2, 90):
    c.drawString(50, y, line)
    y -= 15

# Note: We INTENTIONALLY omit pediatric or off-label neurological drug protocols so the QA can fail.
c.save()
