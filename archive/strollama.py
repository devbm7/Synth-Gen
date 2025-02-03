import streamlit as st
import pandas as pd
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from ollama import AsyncClient
import random
from concurrent.futures import ThreadPoolExecutor

class ColumnSchema(BaseModel):
    name: str
    type: str = Field(..., pattern=r'^(text|integer|float|datetime|categorical)$')
    description: str
    constraints: Dict[str, Any] = {}
    examples: List[str] = []
    format: Optional[str] = None
    
class CorrelationRule(BaseModel):
    source: str
    target: str
    rule_type: str
    params: Dict[str, Any] = {}

class DatasetSchema(BaseModel):
    columns: List[ColumnSchema]
    correlations: List[CorrelationRule] = []
    quality_settings: Dict[str, float] = {
        "missing_rate": 0.0,
        "deformation_rate": 0.0
    }
    
class GeneratedValue(BaseModel):
    value: Any
    correlated_values: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class SchemaExtractor:
    def __init__(self, client: AsyncClient):
        self.client = client
        
    async def extract_schema(self, context: str) -> DatasetSchema:
        response = await self.client.chat(
            messages=[{
                'role': 'user',
                'content': f"""Extract dataset schema from this context. Include:
                - Column names, types, and descriptions
                - Any implied correlations between fields
                - Data quality requirements
                Context: {context}"""
            }],
            model='llama3.1:8b',
            format=DatasetSchema.model_json_schema()
        )
        return DatasetSchema.model_validate_json(response.message.content)

class DataGenerator:
    def __init__(self, client: AsyncClient, schema: DatasetSchema):
        self.client = client
        self.schema = schema
        self.executor = ThreadPoolExecutor()

    async def generate_batch(self, batch_size: int) -> pd.DataFrame:
        tasks = []
        for _ in range(batch_size):
            tasks.append(self._generate_row())
        rows = await asyncio.gather(*tasks)
        return pd.DataFrame(rows)

    async def _generate_row(self) -> Dict[str, Any]:
        row = {}
        processed = set()
        
        # Generate correlated fields first
        for correlation in self.schema.correlations:
            if correlation.source not in processed:
                value = await self._generate_value(correlation.source)
                row[correlation.source] = value.value
                row.update(value.correlated_values)
                processed.update([correlation.source] + list(value.correlated_values.keys()))
        
        # Generate remaining fields
        for column in self.schema.columns:
            if column.name not in processed:
                value = await self._generate_value(column.name)
                row[column.name] = value.value
        
        return row

    async def _generate_value(self, column_name: str) -> GeneratedValue:
        column = next(col for col in self.schema.columns if col.name == column_name)
        response = await self.client.chat(
            messages=[{
                'role': 'user',
                'content': f"""Generate a {column.type} value for column '{column.name}'.
                Description: {column.description}
                Constraints: {column.constraints}
                Examples: {column.examples}"""
            }],
            model='llama3.1:8b',
            format=GeneratedValue.model_json_schema()
        )
        return GeneratedValue.model_validate_json(response.message.content)

class DataQualityEnhancer:
    @staticmethod
    def apply_deformations(df: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
        enhanced = df.copy()
        for column in schema.columns:
            if random.random() < schema.quality_settings["deformation_rate"]:
                enhanced[column.name] = enhanced[column.name].apply(
                    lambda x: DataQualityEnhancer._deform_value(x, column)
                )
        return enhanced
    
    @staticmethod
    def _deform_value(value: Any, column: ColumnSchema) -> Any:
        if value is None:
            return value
            
        if column.type in ["integer", "float"]:
            return DataQualityEnhancer._deform_numeric(value)
        elif column.type == "datetime":
            return DataQualityEnhancer._deform_date(value)
        elif column.type == "text":
            return DataQualityEnhancer._deform_text(value)
        return value

async def main():
    st.title("Enhanced Test Data Generator")
    
    context = st.text_area("Dataset Context", height=200)
    if not context:
        return
        
    client = AsyncClient()
    extractor = SchemaExtractor(client)
    
    try:
        with st.spinner("Extracting schema..."):
            schema = await extractor.extract_schema(context)
        
        st.json(schema.model_dump())
        
        num_rows = st.number_input("Number of rows", min_value=1, max_value=1000, value=100)
        batch_size = st.number_input("Batch size", min_value=1, max_value=100, value=10)
        
        if st.button("Generate Data"):
            generator = DataGenerator(client, schema)
            
            progress_bar = st.progress(0)
            dataframes = []
            
            for i in range(0, num_rows, batch_size):
                batch_df = await generator.generate_batch(min(batch_size, num_rows - i))
                dataframes.append(batch_df)
                progress_bar.progress((i + batch_size) / num_rows)
            
            final_df = pd.concat(dataframes, ignore_index=True)
            enhanced_df = DataQualityEnhancer.apply_deformations(final_df, schema)
            
            st.dataframe(enhanced_df)
            
            csv = enhanced_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "generated_data.csv", "text/csv")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())