"""
Streamlit app for Error Classification Dataset Generator
Allows direct testing of all error endpoints
"""

import streamlit as st
import requests
import json
from typing import Dict, List
import pandas as pd
import os

# Detect if running in Docker or locally
API_BASE_URL = "http://api:8000"

st.set_page_config(
    page_title="Error Dataset Generator",
    page_icon="⚠️",
    layout="wide"
)

st.title("⚠️ Error Dataset Generator")
st.markdown("*Direct API testing for Python, ML, and AI error endpoints*")

# Sidebar for endpoint selection
st.sidebar.header("🎯 Select Endpoint")

error_class = st.sidebar.radio(
    "Error Class",
    ["Python", "ML", "AI"],
    horizontal=False
)

# Define endpoints for each class
endpoints_map = {
    "Python": {
        "types": "/python/types",
        "strings": "/python/strings",
        "io": "/python/io",
        "arithmetic": "/python/arithmetic",
        "iteration": "/python/iteration",
        "advanced": "/python/advanced",
        "test": "/python/test",
        "info": "/python/info"
    },
    "ML": {
        "preprocessing": "/ml/preprocessing",
        "training": "/ml/training",
        "metrics": "/ml/metrics",
        "advanced": "/ml/advanced",
        "nextgen": "/ml/nextgen",
        "propagation": "/ml/propagation",
        "info": "/ml/info"
    },
    "AI": {
        "preprocessing": "/ai/preprocessing",
        "vision": "/ai/vision",
        "embeddings": "/ai/embeddings",
        "autograd": "/ai/autograd",
        "transformers": "/ai/transformers",
        "propagation": "/ai/propagation",
        "deeplearning": "/ai/deeplearning",
        "info": "/ai/info"
    }
}

# Get available endpoints for selected class
available_endpoints = endpoints_map[error_class]
endpoint_names = list(available_endpoints.keys())

selected_endpoint = st.sidebar.selectbox(
    "Select Category",
    endpoint_names,
    format_func=lambda x: x.upper() if x != "run-all" else "RUN ALL"
)

# Main content area
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### 🚀 Controls")
    
    if st.button("📊 Execute Endpoint", use_container_width=True, type="primary"):
        endpoint = available_endpoints[selected_endpoint]
        url = f"{API_BASE_URL}{endpoint}"
        
        st.markdown("---")
        st.markdown("### ⏳ Loading...")
        
        try:
            response = requests.get(url, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                st.session_state.last_response = data
                st.success("✅ Request successful!")
            else:
                st.error(f"❌ Error: {response.status_code}")
                st.write(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Connection failed. Is the API running at {API_BASE_URL}?")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

with col2:
    st.markdown("### 📋 Response")
    
    if "last_response" in st.session_state:
        response_data = st.session_state.last_response
        
        # Display response tabs
        tab1, tab2, tab3 = st.tabs(["Summary", "Detailed", "JSON"])
        
        with tab1:
            st.markdown("#### Summary")
            
            if "category" in response_data:
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Category", response_data.get("category", "N/A"))
                with col_b:
                    st.metric("Total", response_data.get("total", "N/A"))
                with col_c:
                    st.metric("Succeeded", response_data.get("succeeded", "N/A"))
                
                col_d, col_e = st.columns(2)
                with col_d:
                    st.metric("Failed", response_data.get("failed", "N/A"))
                with col_e:
                    success_rate = (response_data.get("succeeded", 0) / max(response_data.get("total", 1), 1)) * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")
            
            elif "endpoints" in response_data:
                st.markdown(f"**Error Class:** {response_data.get('class', 'N/A').upper()}")
                st.markdown(f"**Total Errors:** {response_data.get('total_errors', 'N/A')}")
                st.markdown("**Available Endpoints:**")
                for endpoint, count in response_data.get("endpoints", {}).items():
                    st.markdown(f"  - `{endpoint}`: {count} errors")
        
        with tab2:
            st.markdown("#### Error Details")
            
            if "errors" in response_data and response_data["errors"]:
                # Create a table of errors
                errors_list = response_data["errors"]
                df = pd.DataFrame(errors_list)
                
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Summary by error type
                if "type" in df.columns:
                    st.markdown("**Error Types Distribution:**")
                    error_counts = df["type"].value_counts()
                    st.bar_chart(error_counts)
            else:
                st.info("No errors in response")
        
        with tab3:
            st.json(response_data)
    
    else:
        st.info("💡 Click 'Execute Endpoint' to see results")

# Bottom section - Documentation
st.markdown("---")
st.markdown("## 📚 Endpoint Documentation")

doc_col1, doc_col2, doc_col3 = st.columns(3)

with doc_col1:
    st.markdown("### Python Errors")
    st.markdown("""
- **Types**: Type/data structure errors
- **Strings**: Encoding/string errors
- **I/O**: File operation errors
- **Arithmetic**: Math operation errors
- **Iteration**: Loop/control flow errors
- **Advanced**: Mixed advanced Python errors
    """)

with doc_col2:
    st.markdown("### ML Errors")
    st.markdown("""
- **Preprocessing**: Data handling
- **Training**: Model fitting
- **Metrics**: Evaluation errors
- **Advanced**: Complex scenarios
- **Propagation**: Propagation errors                
    """)

with doc_col3:
    st.markdown("### AI Errors")
    st.markdown("""
- **Preprocessing**: Tensor prep
- **Vision**: CNN/CV errors
- **Embeddings**: NLP/embedding
- **Autograd**: Gradient/RNN
- **Transformers**: Transformer arch
- **Propagation**: Error propagation
- **DeepLearning**: PyTorch/TensorFlow                
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
<p><strong>Error Classification Dataset Generator v2.0</strong></p>
<p>570 total error scenarios across Python, ML, and AI frameworks</p>
</div>
""", unsafe_allow_html=True)
