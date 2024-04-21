import re
import torch
import time
import base64
import pandas as pd
import streamlit as st
from parallelBERT import ParallelBERT 

# transformers
from transformers import AutoTokenizer

# Set page configuration
st.set_page_config(page_title='Your ML Model Demo')

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def encode_sequence(seq_list, max_len, tokenizer):
    for seq in seq_list:
        seq = re.sub(r"[UZOB]", "X", seq)

    encoded = tokenizer(
        " ".join(seq_list),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=max_len,
    )
    return encoded

img = get_img_as_base64("./drug_discovery.png")

act_fn = "tanh"
protein_model_name = "facebook/esm2_t6_8M_UR50D"
COMPRESS_MODEL = False
dims = [320*2, 320, 512, 256]
device = torch.device('cpu')

ab_length = 700
virus_length = 512
tokenizer = AutoTokenizer.from_pretrained(
            protein_model_name, do_lower_case=False
        )

# Load your machine learning model
model = ParallelBERT(
    dropout_rate=0.1, 
    protein_model_name=protein_model_name, 
    compress_model=COMPRESS_MODEL,
    act_fn=act_fn,
    dims=dims,
).to(device)

model.load_state_dict(torch.load('Antigen Split Model.pt', map_location=torch.device('cpu')))
model.eval()

# Define CSS for setting the background image
css = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-color: #e5e5f7;
    opacity: 0.8;
    background-image:  url("data:image/png;base64,{img}");
    }}

.prediction-value {{
    background-color: black;
    color: white;
    padding: 10px;
    border-radius: 5px;
    }}
.time-taken {{
    background-color: black;
    color: white;
    padding: 7px;
    border-radius: 5px;
    margin-top: 10px;
}}

.subtitle {{
    font-family: 'Helvetica';
    margin-bottom: 40px;
}}
</style>
"""

# Render the CSS using st.markdown()
st.markdown(css, unsafe_allow_html=True)

# Define Streamlit app layout
st.title('ImmunoAI Nexus')
# Add a subtitle with reduced gap
st.write(
    f'<h5 class="subtitle"><em>Deep learning framework to predict Ag-Ab inhibitory response</em></h5>', 
    unsafe_allow_html=True
)

# Sidebar sections for model selection
st.sidebar.header("Options")
section = st.sidebar.radio("Select Operation: ", ("Data Distribution", "Structure-Based", "Sequence-Based"))


if section == "Structure-Based":
    st.sidebar.markdown("### Structure-Based Prediction Models")
    model_names = ["Model 1", "Model 2", "Model 3"]
    model_name = st.sidebar.selectbox("Select Model", model_names)

elif section == "Sequence-Based":
    st.sidebar.markdown("### Sequence-Based Prediction Models")
    model_names = ["Model A", "Model B", "Model C"]
    model_name = st.sidebar.selectbox("Select Model", model_names)

elif section == "Data Distribution":
    st.sidebar.markdown("### Data Distribution")
    # Code to display data distribution goes here

if section != "Data Distribution":
    # Add input widgets
    ab_light = st.text_input('Enter Antibody Light Chain Sequence:')
    ab_heavy = st.text_input('Enter Antibody Heavy Chain Sequence:')
    ag_input = st.text_input('Enter Antigen Sequence:')

    ab_enc = encode_sequence([ab_light, ab_heavy], ab_length, tokenizer)
    ag_enc = encode_sequence([ag_input], virus_length, tokenizer)

# Perform inference when a button is clicked
if section != "Data Distribution":
    if st.button('Predict'):

        # Perform inference using the selected model
        start_time = time.time()
        # Perform inference using the model
        with torch.no_grad():
            prediction = model(ab_enc, ag_enc)  
        end_time = time.time()
        
        # Convert tensor prediction to value
        prediction_value = round(prediction[0].item(), 4)
        prediction_text = f'Predicted IC50 Value: {prediction_value} μg/ml'  # Adding unit
        
        # Display prediction value with black background
        st.markdown(f'<h2 class="prediction-value">{prediction_text}</h2>', unsafe_allow_html=True)
        
        # Display time taken with black background
        time_taken = round(end_time - start_time, 4)
        time_text = f'Total Inference Time: {time_taken} seconds'
        st.markdown(f'<h3 class="time-taken">{time_text}</h3>', unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == '__main__':
    pass
