import streamlit as st
import pandas as pd
import tempfile
import os
from fpdf import FPDF
from supabase import create_client, Client

# Gemini AI imports
import base64
from google import genai
from google.genai import types

# ------------------- CONFIG -------------------
st.set_page_config(page_title="Invision Insolvency Intelligence Solutions", layout="wide")
st.title("Invision Insolvency Intelligence Solutions")

# ---------- SUPABASE CLIENT ----------
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")

@st.cache_resource
def init_supabase_client():
    if SUPABASE_URL and SUPABASE_KEY:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    else:
        st.error("Supabase URL or Key not found in Streamlit secrets.")
        return None

supabase: Client = init_supabase_client()

# --------- UTILITY FUNCTIONS --------------

def call_gemini_ai(input_text):
    """
    Calls Gemini AI (Google GenAI) using google-genai python package.
    Requires:
      - pip install google-genai
      - GEMINI_API_KEY in Streamlit secrets or env
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            return "No API key found in secrets or environment variable."

    try:
        client = genai.Client(api_key=api_key)
        model = "gemini-2.5-flash"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=input_text),
                ],
            ),
        ]
        tools = [
            types.Tool(url_context=types.UrlContext()),
        ]
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            tools=tools,
        )
        out = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            out += chunk.text or ""
        return out
    except Exception as e:
        return f"Failed to call Gemini AI: {str(e)}"

def extract_text_from_file(uploaded_file):
    # Basic extractor: text, pdf, docx, images (optional: add more robust extractors as needed)
    import mimetypes
    file_type = mimetypes.guess_type(uploaded_file.name)[0] or ""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    try:
        if ext in [".txt"]:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
        elif ext == ".pdf":
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(uploaded_file)
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            except Exception:
                text = "Could not extract PDF text."
        elif ext == ".docx":
            try:
                import docx
                doc = docx.Document(uploaded_file)
                text = "\n".join([para.text for para in doc.paragraphs])
            except Exception:
                text = "Could not extract DOCX text."
        elif "image" in file_type or ext in [".png", ".jpg", ".jpeg"]:
            try:
                import pytesseract
                from PIL import Image
                img = Image.open(uploaded_file)
                text = pytesseract.image_to_string(img)
            except Exception:
                text = "Could not extract image text."
        else:
            text = "Unsupported file type."
    except Exception:
        text = "Failed to process file."
    return text[:6000] # Limit input for AI

def assets_template_csv():
    return (
        "Asset Name,Category,Location,Original Value,Date of Acquisition,Depreciation Rate,Current Condition,Comparable Assets Info\n"
        "Machinery,Plant & Machinery,Plant A,1000000,2018-03-01,10%,Good,Similar Machinery at Plant B\n"
        "Building,Real Estate,Factory Campus,5000000,2016-08-15,5%,Needs Repair,Industrial Building in same area\n"
        "Vehicle,Transport,Warehouse,700000,2019-10-23,15%,Fair,Truck Model Z as per market\n"
    )

def generate_pdf(text, title="Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 8, line)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        tmp_pdf.seek(0)
        pdf_bytes = tmp_pdf.read()
    return pdf_bytes

# ------------------- APP LAYOUT -------------------

tab1, tab2, tab3 = st.tabs(["Document Intelligence", "Valuation Reports & Compliance", "NCLT Cases"])

# ----------------- TAB 1: DOCUMENT INTELLIGENCE -----------------
with tab1:
    st.header("Document Intelligence")
    doc_type = st.selectbox(
        "Select Document Type",
        ["Legal Document", "IBC Document", "Invoice", "Other"]
    )
    uploaded_file = st.file_uploader(
        "Upload Document (PDF, DOCX, TXT, PNG, JPG)", 
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg"], 
        key="doc_upload"
    )
    if uploaded_file:
        st.info(f"Uploaded: {uploaded_file.name}")
        extracted_text = extract_text_from_file(uploaded_file)
        st.text_area("Extracted Text (preview)", value=extracted_text[:1500], height=200)
        if st.button("Analyze with AI"):
            with st.spinner("Analyzing document with Gemini AI..."):
                prompt = (
                    f"This is a {doc_type} for insolvency/intelligence context. "
                    f"Please perform a comprehensive audit/analysis and generate a detailed professional report:\n\n"
                    f"{extracted_text}"
                )
                analysis = call_gemini_ai(prompt)
            st.success("AI Analysis Complete")
            st.subheader("Comprehensive Audit/Analysis Report")
            st.text_area("AI Report", value=analysis, height=300)
            st.download_button(
                "Download Analysis Report (TXT)",
                data=analysis,
                file_name="analysis_report.txt",
                mime="text/plain"
            )
            pdf_bytes = generate_pdf(analysis, title="Document Intelligence Report")
            st.download_button(
                "Download Analysis Report (PDF)",
                data=pdf_bytes,
                file_name="analysis_report.pdf",
                mime="application/pdf"
            )

# ----------------- TAB 2: VALUATION REPORT & COMPLIANCE -----------------
with tab2:
    st.header("Valuation Reports & Compliance")
    col1, col2 = st.columns([2, 1])
    with col1:
        asset_file = st.file_uploader(
            "Upload Asset CSV (use template below)", 
            type="csv", key="asset_csv"
        )
    with col2:
        st.markdown("**CSV Template**")
        st.download_button(
            "Download Asset CSV Template",
            data=assets_template_csv(),
            file_name="assets_template.csv",
            mime="text/csv"
        )

    if asset_file:
        try:
            df = pd.read_csv(asset_file)
            st.success("Assets loaded successfully!")
            st.dataframe(df.head(20))
            if st.button("Run IBC Compliant Valuation"):
                asset_csv_text = df.to_csv(index=False)
                ai_prompt = (
                    "You are an IBC-compliant valuation expert. Given these asset details, "
                    "perform a professional, comparable-based, and financial valuation. "
                    "Give a detailed and structured valuation report suitable for compliance, "
                    "listing assumptions, comparable analysis, and rationale. The data:\n\n"
                    f"{asset_csv_text}"
                )
                with st.spinner("Performing Gemini AI-powered valuation..."):
                    valuation_report = call_gemini_ai(ai_prompt)
                st.success("Valuation Complete")
                st.subheader("IBC Compliant Valuation Report")
                st.text_area("Valuation Report", value=valuation_report, height=300)
                st.download_button(
                    "Download Valuation Report (TXT)",
                    data=valuation_report,
                    file_name="valuation_report.txt",
                    mime="text/plain"
                )
                pdf_bytes = generate_pdf(valuation_report, title="IBC Valuation Report")
                st.download_button(
                    "Download Valuation Report (PDF)",
                    data=pdf_bytes,
                    file_name="valuation_report.pdf",
                    mime="application/pdf"
                )
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")

# ----------------- TAB 3: NCLT CASES -----------------
with tab3:
    st.header("NCLT Intelligence Database")

    if supabase:
        st.subheader("Simple Search")
        search_query = st.text_input("Enter keywords to search in 'Content' or 'Regulatory File Name'")
        search_button = st.button("Search")

        if search_button and search_query:
            with st.spinner("Searching..."):
                try:
                    # Simple search across 'Regulatory File Name' and 'Content 1'
                    response = supabase.table("nclt_intelligence").select("*").or_(
                        f"Regulatory_File_Name.ilike.%{search_query}%,Content_1.ilike.%{search_query}%"
                    ).execute()
                    
                    if response.data:
                        df_results = pd.DataFrame(response.data)
                        st.dataframe(df_results)
                    else:
                        st.info("No results found for your simple search query.")
                except Exception as e:
                    st.error(f"Error during simple search: {e}")

        st.subheader("Advanced SQL Editor")
        st.warning("Use with caution. Incorrect SQL queries may lead to errors. Only `SELECT` queries are supported.")
        sql_query_template = """SELECT "Unique ID", "Regulatory File Name", "Content 1" FROM nclt_intelligence WHERE "Content 1" ILIKE '%your_keyword%' LIMIT 10;"""
        advanced_sql_query = st.text_area(
            "Enter your SQL query (SELECT statements only):",
            value=sql_query_template,
            height=200
        )
        execute_sql_button = st.button("Execute SQL Query")

        if execute_sql_button and advanced_sql_query:
            if not advanced_sql_query.strip().upper().startswith("SELECT"):
                st.error("Only SELECT queries are allowed for security reasons.")
            else:
                with st.spinner("Executing SQL query..."):
                    try:
                        # Supabase's `rpc` method can be used for custom SQL functions,
                        # but for direct table queries, `from().select()` is safer.
                        # For a full SQL editor, you'd typically need a custom endpoint
                        # or a more sophisticated query builder.
                        # As a workaround, we'll try to parse and execute a simple SELECT.
                        # THIS IS A SIMPLIFIED EXAMPLE AND NOT A SECURE PRODUCTION SQL EDITOR.
                        # A robust solution would involve proper SQL parsing and sanitization.

                        # Attempt to parse a basic SELECT statement for demonstration
                        # This is highly limited and prone to failure with complex SQL
                        # For a real SQL editor, consider using a library like sqlparse
                        # to extract table and WHERE clauses safely.

                        # A safer approach for a generic SQL editor in Supabase
                        # would be to define views or stored procedures and call them via rpc.
                        # Direct arbitrary SQL execution from client is generally discouraged.

                        # For demonstration, let's just attempt a generic select from 'nclt_intelligence'
                        # based on the entered query structure, but only for `SELECT` clauses
                        # and by extracting conditions. This is still not robust.

                        # A more practical way for limited advanced search:
                        # Allow users to specify column, operator, and value, then build the query.
                        # For full SQL, it's problematic without a backend service.

                        # Let's pivot to a safer "advanced search" that uses Supabase's query builder
                        # but gives users more control over columns and conditions, rather than raw SQL.
                        # However, since the request explicitly asks for "SQL editor", we will
                        # demonstrate a highly *simplified and unsafe* version, with strong warnings.
                        # For production, this part needs a secure backend.

                        # Simulating a direct SQL execution (NOT RECOMMENDED FOR PRODUCTION)
                        # This part would typically be handled by a secure backend API that
                        # validates and executes the SQL, returning results.
                        st.info("Direct SQL execution is simulated here. For production, a secure backend is required.")
                        
                        # Fetch all data and filter in Pandas for demonstration of "SQL-like" features
                        # This is inefficient for large datasets but avoids raw SQL injection in frontend
                        # with `supabase-py`.
                        response = supabase.table("nclt_intelligence").select("*").execute()
                        all_data_df = pd.DataFrame(response.data)

                        # A very basic (and unsafe) attempt to filter based on WHERE clause in text_area
                        # This is *not* a real SQL engine.
                        try:
                            # Extract parts (this is very fragile)
                            parts = advanced_sql_query.upper().split("FROM NCLT_INTELLIGENCE WHERE", 1)
                            if len(parts) == 2:
                                where_clause_str = parts[1].strip().split("LIMIT")[0].strip()
                                
                                # Convert simple ILIKE conditions to Pandas string contains
                                # Example: "Content 1" ILIKE '%keyword%'
                                if "ILIKE" in where_clause_str:
                                    col, _, val = where_clause_str.partition("ILIKE")
                                    col = col.strip().strip('"')
                                    val = val.strip().strip("'").strip("%")
                                    
                                    if col in all_data_df.columns:
                                        filtered_df = all_data_df[all_data_df[col].astype(str).str.contains(val, case=False, na=False)]
                                        st.dataframe(filtered_df.head(10)) # Apply limit manually
                                    else:
                                        st.error(f"Column '{col}' not found for filtering.")
                                else:
                                    st.error("Only simple ILIKE conditions are supported in this demo 'WHERE' clause parser.")
                            else:
                                # If no WHERE clause, just show a sample
                                st.dataframe(all_data_df.head(10))

                        except Exception as e:
                            st.error(f"Error parsing SQL-like query or filtering data: {e}")
                            st.dataframe(all_data_df.head(10)) # Fallback
                        
                    except Exception as e:
                        st.error(f"Error executing advanced SQL query: {e}")
    else:
        st.warning("Supabase client not initialized. Please check your `SUPABASE_URL` and `SUPABASE_KEY` in Streamlit secrets.")


# ------------------- FOOTER INFO -------------------
st.markdown("""
---
**Note:**  
- The app uses Google Gemini AI via google-genai python package.  
- Requires: `pip install streamlit pandas fpdf PyPDF2 python-docx google-genai supabase`  
- For image text extraction: `pip install pytesseract pillow` (and install Tesseract if you want image text extraction).
- Add your API key to Streamlit secrets as `GEMINI_API_KEY`.
- Add your Supabase URL and Key to Streamlit secrets as `SUPABASE_URL` and `SUPABASE_KEY`.
""")
