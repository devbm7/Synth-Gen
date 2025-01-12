import streamlit as st
import pandas as pd
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List
import numpy as np

class ColumnSpec:
    def __init__(self, name: str, data_type: str, params: Dict[str, Any]):
        self.name = name
        self.data_type = data_type
        self.params = params
        
    def generate_value(self, missing_rate: float) -> Any:
        if random.random() < missing_rate:
            return None
            
        if self.data_type == "integer":
            return random.randint(self.params.get("min", 0), self.params.get("max", 100))
            
        elif self.data_type == "float":
            min_val = self.params.get("min", 0)
            max_val = self.params.get("max", 100)
            decimals = self.params.get("decimals", 2)
            return round(random.uniform(min_val, max_val), decimals)
            
        elif self.data_type == "datetime":
            start_date = datetime.strptime(self.params.get("start_date", "2020-01-01"), "%Y-%m-%d")
            end_date = datetime.strptime(self.params.get("end_date", "2024-12-31"), "%Y-%m-%d")
            days_between = (end_date - start_date).days
            random_days = random.randint(0, days_between)
            return (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")
            
        elif self.data_type == "categorical":
            return random.choice(self.params.get("categories", ["A", "B", "C"]))
            
        elif self.data_type == "boolean":
            return random.choice([True, False])
            
        elif self.data_type == "text":
            words = self.params.get("word_list", ["Lorem", "Ipsum", "Dolor", "Sit", "Amet"])
            word_count = random.randint(
                self.params.get("min_words", 1),
                self.params.get("max_words", 5)
            )
            return " ".join(random.choices(words, k=word_count))

class EnhancedTestDataGenerator:
    def create_inconsistent_csv(self, columns: List[ColumnSpec], num_rows: int, missing_rate: float = 0.2) -> pd.DataFrame:
        data = []
        for _ in range(num_rows):
            row = {col.name: col.generate_value(missing_rate) for col in columns}
            data.append(row)
        return pd.DataFrame(data)

def render_column_specifics(num_columns: int):
    columns = []
    for i in range(num_columns):
        st.subheader(f"Column {i+1}")
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input(f"Column Name", value=f"column_{i+1}", key=f"name_{i}")
            data_type = st.selectbox(
                "Data Type",
                ["integer", "float", "datetime", "categorical", "boolean", "text"],
                key=f"type_{i}"
            )
        
        with col2:
            params = {}
            if data_type in ["integer", "float"]:
                params["min"] = st.number_input("Minimum", value=0, key=f"min_{i}")
                params["max"] = st.number_input("Maximum", value=100, key=f"max_{i}")
                if data_type == "float":
                    params["decimals"] = st.number_input("Decimal Places", value=2, min_value=0, max_value=10, key=f"dec_{i}")
                    
            elif data_type == "datetime":
                params["start_date"] = st.date_input("Start Date", datetime(2020, 1, 1), key=f"start_{i}").strftime("%Y-%m-%d")
                params["end_date"] = st.date_input("End Date", datetime(2024, 12, 31), key=f"end_{i}").strftime("%Y-%m-%d")
                
            elif data_type == "categorical":
                categories = st.text_input("Categories (comma-separated)", "A,B,C", key=f"cat_{i}")
                params["categories"] = [c.strip() for c in categories.split(",")]
                
            elif data_type == "text":
                word_list = st.text_input("Word List (comma-separated)", "Lorem,Ipsum,Dolor,Sit,Amet", key=f"words_{i}")
                params["word_list"] = [w.strip() for w in word_list.split(",")]
                params["min_words"] = st.number_input("Min Words", value=1, min_value=1, key=f"minw_{i}")
                params["max_words"] = st.number_input("Max Words", value=5, min_value=1, key=f"maxw_{i}")
        
        columns.append(ColumnSpec(name, data_type, params))
    
    return columns

def main():
    st.title("Enhanced Test Data Generator")
    
    num_columns = st.number_input("Number of Columns", min_value=1, max_value=20, value=3)
    columns = render_column_specifics(num_columns)
    
    num_rows = st.slider("Number of Rows", 10, 1000, 100)
    missing_rate = st.slider("Missing Data Rate", 0.0, 1.0, 0.2)
    
    generator = EnhancedTestDataGenerator()
    
    if st.button("Generate CSV"):
        df = generator.create_inconsistent_csv(columns, num_rows, missing_rate)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            csv,
            "test_data.csv",
            "text/csv"
        )
        st.dataframe(df)

if __name__ == "__main__":
    main()