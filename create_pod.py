import pandas as pd
from fpdf import FPDF
from datetime import datetime

def generate_pdf_for_shipment(csv_path, output_pdf, shipment_id, selected_columns, logo_path=None):
    df = pd.read_csv(csv_path)
    df = df[df["shipment_id"] == shipment_id]

    if df.empty:
        print(f"Shipment ID {shipment_id} not found in CSV.")
        return

    df = df[selected_columns]
    today = datetime.today().strftime('%Y-%m-%d')


    pdf = FPDF()
    pdf.add_page()


    if logo_path:
        pdf.image(logo_path, x=10, y=10, w=40)
        pdf.set_xy(10, 40)
    else:
        pdf.set_y(10)


    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Your Order Summary", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Date: {today}", ln=True, align="C")

    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Shipment Details for ID: {shipment_id}", ln=True)

    pdf.set_font("Arial", "B", 11)
    col_width = 45
    for col in selected_columns:
        pdf.cell(col_width, 10, col, border=1)
    pdf.ln()

    pdf.set_font("Arial", "", 11)
    for _, row in df.iterrows():
        for col in selected_columns:
            pdf.cell(col_width, 10, str(row[col]), border=1)
        pdf.ln()


    pdf.ln(10)
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(0, 8, "Your Shipment has been succesfully booked\nFor any queries regarding this shipment, contact: support@qbotica.com")

    pdf.output(output_pdf)
    print(f"PDF for {shipment_id} saved as: {output_pdf}")


if __name__ == "__main__":
    csv_file = "orders.csv"
    output_pdf = "order_summary_SHP31391.pdf"
    shipment_id = "SHP31391"
    columns_to_include = ["shipment_id", "origin_zip", "dest_zip", "Quote"]
    logo_file = "test.png"

    generate_pdf_for_shipment(csv_file, output_pdf, shipment_id, columns_to_include, logo_file)
