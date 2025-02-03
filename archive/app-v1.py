import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, Any, List, Tuple
from ollama import Client
import re

class DeformationRules:
    CURRENCY_SYMBOLS = ['$', '€', '£', '¥']
    NUMBER_FORMATS = ['1234.56', '1,234.56', '1.234,56', '1 234.56']
    DATE_FORMATS = ['%Y-%m-%d', '%d/%m/%Y', '%m-%d-%Y', '%d-%b-%Y', '%Y/%m/%d']
    TEXT_CASES = ['normal', 'UPPER', 'lower', 'Title Case', 'camelCase']
    
    @staticmethod
    def apply_deformation(value: Any, data_type: str, deform_rate: float, specific_rules: Dict = None) -> str:
        if random.random() > deform_rate or value is None:
            return value
            
        specific_rules = specific_rules or {}
        
        if data_type in ["integer", "float"]:
            if specific_rules.get("add_currency", False):
                symbol = random.choice(DeformationRules.CURRENCY_SYMBOLS)
                try:
                    formatted = format(float(value), ',').replace(',', random.choice([',', '.', ' ']))
                    return f"{symbol}{formatted}"
                except:
                    return value
                    
        elif data_type == "datetime":
            try:
                date_obj = datetime.strptime(str(value), "%Y-%m-%d")
                return date_obj.strftime(random.choice(DeformationRules.DATE_FORMATS))
            except:
                return value
                
        elif data_type == "text":
            try:
                case_style = random.choice(DeformationRules.TEXT_CASES)
                if case_style == "UPPER":
                    return str(value).upper()
                elif case_style == "lower":
                    return str(value).lower()
                elif case_style == "Title Case":
                    return str(value).title()
                elif case_style == "camelCase":
                    words = str(value).split()
                    return words[0].lower() + ''.join(word.capitalize() for word in words[1:])
            except:
                return value
                
        return value

class ContextAwareColumnSpec:
    def __init__(self, name: str, data_type: str, context: str, params: Dict[str, Any], deform_rate: float = 0.0):
        self.name = name
        self.data_type = data_type
        self.context = context
        self.params = params
        self.deform_rate = deform_rate
        
    def generate_value(self, client: Client, missing_rate: float) -> Any:
        if random.random() < missing_rate:
            return None
            
        prompt = self._create_prompt()
        try:
            response = client.generate(model="llama3.1:8b", prompt=prompt)
            value = self._process_response(response['response'].strip())
            return DeformationRules.apply_deformation(value, self.data_type, self.deform_rate, self.params)
        except:
            return self._fallback_generation()
    
    def _create_prompt(self) -> str:
        base_prompt = f"Generate a single {self.data_type} value for column '{self.name}'.Generate only one value and nothing else. do not put the value in quotes. "
        context_prompt = f"Context: {self.context}"
        constraints = self._get_constraints()
        return f"{base_prompt}{context_prompt}{constraints}"
    
    def _get_constraints(self) -> str:
        if self.data_type in ["integer", "float"]:
            return f" Value should be between {self.params.get('min', 0)} and {self.params.get('max', 100)}."
        elif self.data_type == "datetime":
            return f" Date should be between {self.params.get('start_date')} and {self.params.get('end_date')}."
        elif self.data_type == "categorical":
            return f" Value should be one of: {', '.join(self.params.get('categories', []))}."
        return ""
    
    def _process_response(self, response: str) -> Any:
        if self.data_type == "integer":
            return int(float(re.findall(r'-?\d+\.?\d*', response)[0]))
        elif self.data_type == "float":
            return float(re.findall(r'-?\d+\.?\d*', response)[0])
        return response
    
    def _fallback_generation(self) -> Any:
        if self.data_type == "integer":
            return random.randint(self.params.get("min", 0), self.params.get("max", 100))
        elif self.data_type == "float":
            return round(random.uniform(self.params.get("min", 0), self.params.get("max", 100)), 2)
        elif self.data_type == "datetime":
            start = datetime.strptime(self.params.get("start_date", "2020-01-01"), "%Y-%m-%d")
            end = datetime.strptime(self.params.get("end_date", "2024-12-31"), "%Y-%m-%d")
            days = (end - start).days
            return (start + timedelta(days=random.randint(0, days))).strftime("%Y-%m-%d")
        elif self.data_type == "categorical":
            return random.choice(self.params.get("categories", ["A", "B", "C"]))
        return "fallback_value"

class CorrelationRules:
    @staticmethod
    def calculate_age(birth_date: str) -> int:
        birth_year = int(birth_date.split('-')[0])
        current_year = datetime.now().year
        return current_year - birth_year
    
    @staticmethod
    def calculate_birth_date(age: int) -> str:
        birth_year = datetime.now().year - age
        return f"{birth_year}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
    
    @staticmethod
    def get_gender_from_name(name: str, client: Client) -> str:
        prompt = f"What is the most likely gender (Male or Female) for the name: {name}? Reply with just 'Male' or 'Female'"
        try:
            response = client.generate(model="llama3.1:8b", prompt=prompt)
            gender = response['response'].strip().upper()
            return gender if gender in ['MALE', 'FEMALE'] else random.choice(['MALE', 'FEMALE'])
        except:
            return random.choice(['MALE', 'FEMALE'])
    
    @staticmethod
    def get_name_for_gender(gender: str, client: Client, is_first_name: bool = True) -> str:
        name_type = "first name" if is_first_name else "last name"
        prompt = f"Generate one {gender.lower()} {name_type}. Reply with just the name."
        try:
            response = client.generate(model="llama3.1:8b", prompt=prompt)
            return response['response'].strip()
        except:
            return f"{'First' if is_first_name else 'Last'}Name"

class CorrelationManager:
    def __init__(self, global_context: str):
        self.correlations = self._parse_correlations(global_context)
        self.generated_values = {}
        
    def _parse_correlations(self, context: str) -> Dict[str, Dict[str, str]]:
        correlations = {}
        if "Correlate:" not in context:
            return correlations
            
        rules = context.split("Correlate:")[1].split(";")
        for rule in rules:
            if ":" in rule:
                rule_parts = rule.strip().split(":")
                primary = rule_parts[0].strip()
                dependents = [dep.strip() for dep in rule_parts[1].split(",")]
                correlations[primary] = {
                    "type": "logical",
                    "dependents": dependents
                }
        return correlations
    
    def handle_correlation(self, column_name: str, value: Any, client: Client) -> Dict[str, Any]:
        correlated_values = {}
        
        # Age and Birth Date correlation
        if column_name == "age" and "birth_date" in self.get_correlated_columns(column_name):
            correlated_values["birth_date"] = CorrelationRules.calculate_birth_date(int(value))
        elif column_name == "birth_date" and "age" in self.get_correlated_columns(column_name):
            correlated_values["age"] = CorrelationRules.calculate_age(str(value))
            
        # Name and Gender correlation
        if column_name == "first_name" and "gender" in self.get_correlated_columns(column_name):
            correlated_values["gender"] = CorrelationRules.get_gender_from_name(str(value), client)
        elif column_name == "gender":
            if "first_name" in self.get_correlated_columns(column_name):
                correlated_values["first_name"] = CorrelationRules.get_name_for_gender(str(value), client, True)
            if "last_name" in self.get_correlated_columns(column_name):
                correlated_values["last_name"] = CorrelationRules.get_name_for_gender(str(value), client, False)
        
        return correlated_values
    
    def get_correlated_columns(self, column_name: str) -> List[str]:
        correlated = []
        for primary, info in self.correlations.items():
            if primary == column_name:
                correlated.extend(info["dependents"])
            elif column_name in info["dependents"]:
                correlated.append(primary)
        return correlated
    

class DataAnalyzer:
    @staticmethod
    def infer_column_specs(df: pd.DataFrame) -> List[ContextAwareColumnSpec]:
        specs = []
        for col in df.columns:
            data_type, params = DataAnalyzer._infer_column_type(df[col])
            context = DataAnalyzer._generate_column_context(df[col], col, data_type)
            specs.append(ContextAwareColumnSpec(col, data_type, context, params))
        return specs
    
    @staticmethod
    def _infer_column_type(series: pd.Series) -> Tuple[str, Dict]:
        sample = series.dropna()
        if len(sample) == 0:
            return "text", {}
            
        if pd.api.types.is_numeric_dtype(sample):
            if all(sample.astype(str).str.contains(r'^\d+$')):
                return "integer", {
                    "min": int(sample.min()),
                    "max": int(sample.max())
                }
            return "float", {
                "min": float(sample.min()),
                "max": float(sample.max())
            }
            
        if pd.api.types.is_datetime64_any_dtype(sample):
            return "datetime", {
                "start_date": sample.min().strftime("%Y-%m-%d"),
                "end_date": sample.max().strftime("%Y-%m-%d")
            }
            
        unique_vals = sample.nunique()
        if unique_vals < 10:
            return "categorical", {"categories": list(sample.unique())}
            
        return "text", {}
    
    @staticmethod
    def _generate_column_context(series: pd.Series, col_name: str, data_type: str) -> str:
        sample_values = series.dropna().sample(min(5, len(series))).tolist()
        context = f"Column '{col_name}' contains {data_type} values. Examples: {sample_values}"
        
        patterns = DataAnalyzer._detect_patterns(series)
        if patterns:
            context += f". Patterns detected: {patterns}"
        
        return context
    
    @staticmethod
    def _detect_patterns(series: pd.Series) -> str:
        patterns = []
        sample = series.dropna().astype(str)
        
        if all(sample.str.contains(r'\d')):
            patterns.append("contains numbers")
        if all(sample.str.contains(r'[A-Z]')):
            patterns.append("contains uppercase")
        if all(sample.str.contains(r'[@]')):
            patterns.append("likely email addresses")
        if all(sample.str.contains(r'^\+?[\d\s-]+$')):
            patterns.append("likely phone numbers")
        
        return ', '.join(patterns)

class TestDataGenerator:
    def __init__(self):
        self.client = Client()
    
    def create_context_aware_csv(self, columns: List[ContextAwareColumnSpec], 
                               num_rows: int, missing_rate: float, 
                               global_context: str) -> pd.DataFrame:
        correlation_manager = CorrelationManager(global_context)
        data = []
        
        for _ in range(num_rows):
            row = {}
            processed_columns = set()
            
            # Process primary columns first
            for col in columns:
                if col.name in correlation_manager.correlations and col.name not in processed_columns:
                    value = col.generate_value(self.client, missing_rate)
                    row[col.name] = value
                    processed_columns.add(col.name)
                    
                    # Generate correlated values
                    correlated_values = correlation_manager.handle_correlation(col.name, value, self.client)
                    row.update(correlated_values)
                    processed_columns.update(correlated_values.keys())
            
            # Process remaining columns
            for col in columns:
                if col.name not in processed_columns:
                    row[col.name] = col.generate_value(self.client, missing_rate)
            
            data.append(row)
        
        return pd.DataFrame(data)

def get_advanced_params(data_type: str, index: int, defaults: Dict = None) -> Dict[str, Any]:
    defaults = defaults or {}
    params = {}
    
    if data_type in ["integer", "float"]:
        params["min"] = st.number_input("Min", value=defaults.get("min", 0), key=f"min_{index}")
        params["max"] = st.number_input("Max", value=defaults.get("max", 100), key=f"max_{index}")
        params["add_currency"] = st.checkbox("Add Currency Symbols", key=f"curr_{index}")
    
    elif data_type == "datetime":
        params["start_date"] = st.date_input(
            "Start", 
            value=datetime.strptime(defaults.get("start_date", "2020-01-01"), "%Y-%m-%d"),
            key=f"start_{index}"
        ).strftime("%Y-%m-%d")
        params["end_date"] = st.date_input(
            "End",
            value=datetime.strptime(defaults.get("end_date", "2024-12-31"), "%Y-%m-%d"),
            key=f"end_{index}"
        ).strftime("%Y-%m-%d")
    
    elif data_type == "categorical":
        cats = st.text_input(
            "Categories",
            value=",".join(defaults.get("categories", ["A", "B", "C"])),
            key=f"cat_{index}"
        )
        params["categories"] = [c.strip() for c in cats.split(",")]
    
    return params

def sidebar_controls(inferred_specs: List[ContextAwareColumnSpec] = None):
    with st.sidebar:
        st.header("Column Configuration")
        
        if inferred_specs:
            specs = []
            for i, spec in enumerate(inferred_specs):
                st.subheader(spec.name)
                data_type = st.selectbox(
                    "Type",
                    ["text", "integer", "float", "datetime", "categorical"],
                    key=f"type_{i}",
                    index=["text", "integer", "float", "datetime", "categorical"].index(spec.data_type)
                )
                context = st.text_area("Context", value=spec.context, key=f"context_{i}")
                deform_rate = st.slider("Deformation Rate", 0.0, 1.0, 0.0, key=f"deform_{i}")
                
                with st.expander("Advanced Settings"):
                    params = get_advanced_params(data_type, i, spec.params)
                
                specs.append(ContextAwareColumnSpec(spec.name, data_type, context, params, deform_rate))
        else:
            num_columns = st.number_input("Number of Columns", min_value=1, max_value=20, value=3)
            specs = []
            
            for i in range(num_columns):
                st.subheader(f"Column {i+1}")
                name = st.text_input("Name", value=f"column_{i+1}", key=f"name_{i}")
                data_type = st.selectbox(
                    "Type",
                    ["text", "integer", "float", "datetime", "categorical"],
                    key=f"type_{i}"
                )
                context = st.text_area(
                    "Context",
                    placeholder="E.g., 'Generate realistic first names'",
                    key=f"context_{i}"
                )
                deform_rate = st.slider("Deformation Rate", 0.0, 1.0, 0.0, key=f"deform_{i}")
                
                with st.expander("Advanced Settings"):
                    params = get_advanced_params(data_type, i)
                
                specs.append(ContextAwareColumnSpec(name, data_type, context, params, deform_rate))
        
        num_rows = st.number_input("Number of Rows", 10, 1000, 100)
        missing_rate = st.slider("Missing Data Rate", 0.0, 1.0, 0.2)
        
        return specs, num_rows, missing_rate

def main():
    st.title("Context-Aware Test Data Generator")

    st.markdown("""
    ### Correlation Syntax Guide
    Use 'Correlate:' in global context to define correlations. Examples:
    
    ```
    Correlate: age:birth_date
    Correlate: gender:first_name,last_name
    ```
    
    Supported correlations:
    - age ↔ birth_date: Maintains mathematical consistency
    - gender ↔ first_name: Ensures gender-appropriate names
    - gender ↔ last_name: Maintains cultural consistency
    """)
    
    uploaded_file = st.file_uploader("Upload CSV file (optional)", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Original Data Preview")
        st.dataframe(df.head())
        
        generation_mode = st.radio(
            "Generation Mode",
            ["Add to existing data", "Create new dataset"],
            help="Choose whether to append new rows to your CSV or create a separate dataset using its structure"
        )
        
        inferred_specs = DataAnalyzer.infer_column_specs(df)
        suggested_context = f"Dataset with {len(df)} rows and {len(df.columns)} columns. Contains: {', '.join(df.columns)}"
        global_context = st.text_area("Global Context", value=suggested_context)
        specs, num_rows, missing_rate = sidebar_controls(inferred_specs)
    else:
        generation_mode = "Create new dataset"
        global_context = st.text_area(
            "Global Context",
            placeholder="Describe the dataset you want to generate"
        )
        specs, num_rows, missing_rate = sidebar_controls()
    
    if st.button("Generate Data"):
        generator = TestDataGenerator()
        new_df = generator.create_context_aware_csv(specs, num_rows, missing_rate, global_context)
        
        if uploaded_file and generation_mode == "Add to existing data":
            final_df = pd.concat([df, new_df], ignore_index=True)
            st.write(f"### Preview (Original: {len(df)} rows + Generated: {len(new_df)} rows)")
        else:
            final_df = new_df
            st.write(f"### Preview (Generated: {len(new_df)} rows)")
        
        st.dataframe(final_df.head())
        
        filename = "augmented_data.csv" if generation_mode == "Add to existing data" else "generated_data.csv"
        csv = final_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, filename, "text/csv")

if __name__ == "__main__":
    main()