from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import textwrap

c = canvas.Canvas("Company_Employee_Handbook.pdf", pagesize=letter)
c.setFont("Helvetica-Bold", 16)
c.drawString(50, 750, "Acme Corp Employee Handbook - 2024")

c.setFont("Helvetica-Bold", 14)
c.drawString(50, 710, "1. Paid Time Off (PTO)")
c.setFont("Helvetica", 12)
text = "All full-time employees are entitled to 20 days of Paid Time Off (PTO) per calendar year. PTO accrues monthly. Employees must submit PTO requests at least two weeks in advance through the HR Workday portal. A maximum of 5 unused PTO days can be rolled over to the next calendar year. Any additional unused days will be forfeited."
y = 690
for line in textwrap.wrap(text, 90):
    c.drawString(50, y, line)
    y -= 15

c.setFont("Helvetica-Bold", 14)
y -= 30
c.drawString(50, y, "2. Code of Conduct")
c.setFont("Helvetica", 12)
filler = "Acme Corp expects all employees to maintain the highest standard of professional conduct. Interactions with colleagues, clients, and partners should always be respectful. Discrimination or harassment of any kind will not be tolerated. Employees are expected to protect company confidential data and report any security breaches to the IT department immediately."
y -= 20
for line in textwrap.wrap(filler, 90):
    c.drawString(50, y, line)
    y -= 15

c.setFont("Helvetica-Bold", 14)
y -= 30
c.drawString(50, y, "3. Remote Work Policy")
c.setFont("Helvetica", 12)
text2 = "Acme Corp supports a hybrid work environment. Eligible employees may work remotely up to two days per week, subject to manager approval. While working remotely, employees must be available during core business hours, which are 10:00 AM to 3:00 PM Eastern Standard Time (EST). All remote work locations must have a secure, high-speed internet connection."
y -= 20
for line in textwrap.wrap(text2, 90):
    c.drawString(50, y, line)
    y -= 15

c.save()
