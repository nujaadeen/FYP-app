import re
import torch
import time
import base64
import pandas as pd
import streamlit as st
from parallelBERT import ParallelBERT 

# transformers
from transformers import AutoTokenizer

import numpy as np
from parallelGAT import ParallelGAT
from graph import pdb_to_graph

# Torch
import torch

###########################################################################
# Set page configuration
st.set_page_config(page_title='Your ML Model Demo')

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

ab_length = 700
virus_length = 512

def encode_sequence(seq_list, max_len):
    tokenizer = AutoTokenizer.from_pretrained(
        protein_model_name, do_lower_case=False
    )
    
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

def is_all_uppercase_alpha(string):
    # Check if all characters in the string are uppercase alphabetical characters
    return all(char.isalpha() and char.isupper() for char in string)

###########################################################################################################

img = get_img_as_base64("./drug_discovery.png")

###########################################################################################################
# Freeze Parameters
FREEZE_LAYERS = True
FREEZE_ALL = True # True to stop finetuning the encoder
FREEZE_LAYER_COUNT = 6
COMPRESS_MODEL = False
LORA_RANK = 32

# Model parameters
MODEL_SIZE = 'Tiny' # FREEZE_LAYERS 0 - 7
DROPOUT = 0.0
dropout_rate = DROPOUT
act_fn = "tanh"

if MODEL_SIZE == 'Tiny':
    protein_model_name = "facebook/esm2_t6_8M_UR50D"
    dims = [320*2, 320, 512, 256]
elif MODEL_SIZE == 'Small':
    protein_model_name = "facebook/esm2_t12_35M_UR50D"
    dims = [480*2, 480, 512, 256]
elif MODEL_SIZE == 'Medium':
    protein_model_name = "facebook/esm2_t30_150M_UR50D"
    dims = [640*2, 640, 512, 256]
elif MODEL_SIZE == 'Large':
    protein_model_name = "facebook/esm2_t33_650M_UR50D"
    dims = [320, 512, 256]
model = ParallelBERT(
    dropout_rate=0.1, 
    protein_model_name=protein_model_name, 
    compress_model=COMPRESS_MODEL,
    act_fn=act_fn,
    dims=dims,
)

# Freeze Model
if FREEZE_LAYERS:
    model.freeze_layers(all=FREEZE_ALL, n_layers=FREEZE_LAYER_COUNT)

model.load_state_dict(torch.load('Random Split Model.pt', map_location=torch.device('cpu')))
model.eval()
######################################################################################################
device = torch.device('cpu')

mlp_dims = [324, 324//4, 324//16]
model_2 = ParallelGAT(
            in_channels=324,
            hidden_channels=324,
            num_layers=8,
            mlp_dims=mlp_dims,
            skip_connections=True,
            heads=1,
            dropout=0.3
    ).to(device)

model_2.load_state_dict(torch.load("Parallel GAT Model.pt", map_location=torch.device('cpu')))
model_2.eval()


######################################################################################################

# num_node_features = 324
# NUM_LAYERS = 8
# SKIP_CONNECTIONS = True
# SEPARATE_POOLING = False
# DROPOUT = 0.3
# mlp_dims = [num_node_features, num_node_features//4, num_node_features//16]
# device = "cpu"

# model_3 = ComplexGAT(
#             in_channels=num_node_features,
#             hidden_channels=num_node_features,
#             num_layers=NUM_LAYERS,
#             mlp_dims=mlp_dims,
#             skip_connections=SKIP_CONNECTIONS,
#             separate_pooling=SEPARATE_POOLING,
#             heads=1,
#             dropout=DROPOUT
#     ).to(device)

# model_3.load_state_dict(torch.load('Complex GAT Model.pt', map_location=torch.device('cpu')))
# model_3.eval()


######################################################################################################

# Define CSS for setting the background image
css = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image:  url("data:image/png;base64,{img}");
    opacity: 0.8;
    }}
[data-testid="metric-container"] {{
    background-color: black;
    border-radius: 5px;
    padding: 10px;
    }}
stMarkdownContainer
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
st.title('ImmunoAI')

# Add a subtitle with reduced gap
st.write(
    f'<h5 class="subtitle"><em>Deep learning framework to predict Ag-Ab binding affinity</em></h5>', 
    unsafe_allow_html=True
)

# Sidebar sections for model selection
st.sidebar.header("Options")
section = st.sidebar.radio("Select Operation: ", ("Sequence-Based", "Parallel Structure-Based",  "Complex Structure-Based"))

######################################################################################################

if section == "Sequence-Based":
    st.sidebar.markdown("### Sequence-Based Prediction Model Selected")
elif section == "Parallel Structure-Based":
    st.sidebar.markdown("### Parallel Structure-Based Prediction Model Selected")
elif section == "Complex Structure-Based":
    st.sidebar.markdown("### Complex Structure-Based Prediction Model Selected")
# elif section == "Data Distribution":
#     st.sidebar.markdown("### Data Distribution")
#     # Code to display data distribution goes here

######################################################################################################

ab_enc = None
ag_enc = None
data_ab = None
data_ag = None

######################################################################################################

# Check the section and handle input accordingly
if section == "Sequence-Based":
    # Add input widgets for sequences
    ab_light = st.text_input('Enter Antibody Light Chain Sequence:')
    ab_heavy = st.text_input('Enter Antibody Heavy Chain Sequence:')
    ag_input = st.text_input('Enter Antigen Sequence:')

    antibody_seq = ab_heavy + ab_light
    virus_seq = ag_input
    # Check if sequences are valid and encode them
    if is_all_uppercase_alpha(ab_light) and is_all_uppercase_alpha(ab_heavy):
        ab_enc = {k: v.view(1, -1) for k, v in encode_sequence(antibody_seq, ab_length).items()}
    else:
        st.warning("The input antibody data format is not recognized. Please check again.")
    
    if is_all_uppercase_alpha(ag_input):
        ag_enc = {k: v.view(1, -1) for k, v in encode_sequence(virus_seq, virus_length).items()} 

    else:
        st.warning("The input antigen data format is not recognized. Please check again.")

######################################################################################################

elif section == "Parallel Structure-Based":
    ab_pdb_file = st.file_uploader("Upload Antibody PDB File", type=['pdb'])
    ag_pdb_file = st.file_uploader("Upload Antigen PDB File", type=['pdb'])

    if ab_pdb_file is not None:
        # Save the uploaded file to a temporary location
        with open("temp_ab.pdb", "wb") as f:
            f.write(ab_pdb_file.getvalue())

    if ag_pdb_file is not None:
        # Save the uploaded file to a temporary location
        with open("temp_ag.pdb", "wb") as f:
            f.write(ag_pdb_file.getvalue())
    else:
        st.warning("Please upload both antibody and antigen PDB files.")
    
    # Parse PDB files and generate graph data
    data_ab = pdb_to_graph("temp_ab.pdb", pdb_type='separate', heavy_chain='A', light_chain='B', virus='C')
    data_ag = pdb_to_graph("temp_ag.pdb", pdb_type='separate', heavy_chain='A', light_chain='B', virus='C')

##################################################################################
elif section == "Complex Structure-Based":
    pass

# Perform inference when a button is clicked
if section != "Data Distribution" and ((ab_enc and ag_enc) or (data_ab and data_ag)):
    if st.button('Predict'):
        progress_bar = st.progress(0)
        # Perform inference using the selected model
        start_time = time.time()
        # Perform inference using the model
        with torch.no_grad():
            if section == "Sequence-Based":
                prediction = model(ab_enc, ag_enc)
                prediction = np.exp(prediction[0].item())  
            elif section == "Parallel Structure-Based":
                prediction, _ = model_2(data_ab.x, data_ab.edge_index, data_ab.batch, data_ag.x, data_ag.edge_index, data_ag.batch)
                prediction = np.exp(prediction.item())
        progress_bar.progress(100)
        end_time = time.time()
        
        # Convert tensor prediction to value
        prediction_value = round(prediction, 4)

        # Display time taken with black background
        time_taken = round(end_time - start_time, 4)

        # Display prediction value and total inference time side by side using st.columns
        col1, col2 = st.columns(2)
        # Display prediction value with unit using st.metric
        col1.metric(label="Predicted IC50 Value", value=f'{prediction_value} μg/ml', delta="2.53 MSE")
        # Display total inference time with unit using st.metric
        col2.metric(label="Total Inference Time", value=f"{time_taken} seconds")


# Add a placeholder to push the copyright notice to the bottom
st.sidebar.markdown('<div style="flex: 1;"></div>', unsafe_allow_html=True)

# Add copyright notice at the bottom of the sidebar with increased margin-top
st.sidebar.markdown('<p style="text-align: center; color: grey; margin-top: 300px;">Copyright © 2024 Immuno.ai. All rights reserved.</p>', unsafe_allow_html=True)



# Run the Streamlit app
if __name__ == '__main__':
    pass
