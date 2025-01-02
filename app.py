import streamlit as st
import pandas as pd
import fpdf
from ollama import Client
import random
import io

class TestDataGenerator:
    def __init__(self):
        self.client = Client()
        
    def generate_content(self, content_type, specifications):
        prompt = f"Generate {content_type} content with these specifications: {specifications}"
        response = self.client.generate(model="llama2:8b", prompt=prompt)
        return response['response']
    
    def create_pdf_with_tables(self, num_tables, rows_per_table):
        pdf = fpdf.FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        content = self.generate_content("business report", 
            f"Create {num_tables} tables with {rows_per_table} rows each about business metrics")
        
        # Add text content
        pdf.multi_cell(0, 10, content)
        
        # Generate and add tables
        for _ in range(num_tables):
            data = [[random.randint(1000, 9999) for _ in range(5)] for _ in range(rows_per_table)]
            col_width = pdf.w / 5.5
            row_height = 10
            
            for row in data:
                for item in row:
                    pdf.cell(col_width, row_height, str(item), 1)
                pdf.ln()
            pdf.ln(10)
            
        return pdf.output(dest='S').encode('latin1')
    
    def create_inconsistent_csv(self, num_rows, missing_rate=0.2):
        data = []
        columns = ['ID', 'Name', 'Value', 'Category', 'Date']
        
        for i in range(num_rows):
            row = {
                'ID': i if random.random() > missing_rate else '',
                'Name': self.generate_content("name", "Generate a random name") if random.random() > missing_rate else '',
                'Value': random.randint(1, 1000) if random.random() > missing_rate else '',
                'Category': random.choice(['A', 'B', 'C']) if random.random() > missing_rate else '',
                'Date': pd.Timestamp.now().strftime('%Y-%m-%d') if random.random() > missing_rate else ''
            }
            data.append(row)
            
        return pd.DataFrame(data)

def main():
    st.title("Test Data Generator")
    
    generator = TestDataGenerator()
    
    with st.sidebar:
        data_type = st.selectbox("Select Data Type", ["PDF with Tables", "Inconsistent CSV"])
        
    if data_type == "PDF with Tables":
        num_tables = st.slider("Number of Tables", 1, 10, 3)
        rows_per_table = st.slider("Rows per Table", 5, 50, 10)
        
        if st.button("Generate PDF"):
            pdf_content = generator.create_pdf_with_tables(num_tables, rows_per_table)
            st.download_button(
                "Download PDF",
                pdf_content,
                "test_data.pdf",
                "application/pdf"
            )
            
    else:  # Inconsistent CSV
        num_rows = st.slider("Number of Rows", 10, 1000, 100)
        missing_rate = st.slider("Missing Data Rate", 0.0, 1.0, 0.2)
        
        if st.button("Generate CSV"):
            df = generator.create_inconsistent_csv(num_rows, missing_rate)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV",
                csv,
                "test_data.csv",
                "text/csv"
            )

if __name__ == "__main__":
    main()