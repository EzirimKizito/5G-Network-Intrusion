
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import StandardScaler
import time
import os
import re

# --- Page Config ---
st.set_page_config(layout="wide", page_title="5G NIDD Live Detection")

# --- Configuration & Constants ---
# !!! IMPORTANT: UPDATE THESE PATHS !!!
# Set these to the correct location of your saved model and scaler files
MODEL_PATH = "nidd_model.h5"
SCALER_PATH = "scaler2.joblib"

# # --- Sidebar ---
# st.sidebar.title("⚙️ Configuration & Info")
# st.sidebar.info(f"Model: {os.path.basename(MODEL_PATH)}")
# st.sidebar.info(f"Scaler: {os.path.basename(SCALER_PATH)}")
# # Toggle for debugging information
# debug_mode = st.sidebar.checkbox("Show Debug Information", value=False) # Default to False

# --- Sidebar ---
st.sidebar.title("About & Help")

# Expander for App Information (collapsed by default)
with st.sidebar.expander("ℹ️ About this App", expanded=False):
    st.markdown(
        """
        **Purpose:**
        This application uses a trained machine learning model to predict
        whether a given 5G network flow exhibits characteristics of
        Non-IP Data Delivery (NIDD) anomalies, potentially indicating
        malicious activity.
        """
    )
    st.divider() # Visual separator
    st.subheader("Instructions:")
    st.markdown(
        """
        1.  **Enter Flow Details:** Fill in the specific values for the network flow
            using the input fields on the main page. Default values are provided
            for a known benign instance.
        2.  **(Optional) Paste Data:** You can paste pre-formatted data
            (`Feature Name: Value`) into the 'Paste Instance Data' section and
            click 'Fill Inputs from Pasted Text'. *Note: Encoded values are expected for Proto, Cause, State.*
        3.  **Predict:** Click the **'Predict Flow Status'** button located below
            all the input fields.
        4.  **View Result:** The prediction (Benign/Malicious), risk level, and
            confidence score will appear below the button.
        """
    )
    st.divider() # Visual separator
    st.subheader("Project Details:")
    st.markdown(
        """
        *   **Developer:** Ajayi Ayooluwa Cornelius
        *   **Matric No:** 19/52HP009
        *   **Context:** Final Year Project
        """
    )

# Toggle for debugging information (kept outside the expander)
st.sidebar.divider() # Separator before debug toggle
debug_mode = st.sidebar.checkbox("Show Debug Information", value=False)
if debug_mode:
    st.sidebar.warning("Debug mode is ON. Extra processing details will be shown.")

# --- Load Model and Preprocessing Objects (Cached) ---
@st.cache_resource
def load_prediction_model(path):
    if debug_mode: st.sidebar.write(f"Attempting to load model from: {path}")
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}. Check MODEL_PATH.")
        return None
    try:
        model = load_model(path)
        print(f"Model loaded successfully from {path}.")
        st.sidebar.success("✅ Model Loaded")
        if hasattr(model, 'input_shape'):
            st.sidebar.write(f"Model Input Shape: {model.input_shape}")
        return model
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
        return None

@st.cache_resource
def load_scaler_object(path):
    if debug_mode: st.sidebar.write(f"Attempting to load scaler from: {path}")
    if not os.path.exists(path):
        st.error(f"Scaler file not found: {path}. Check SCALER_PATH.")
        return None
    try:
        scaler = joblib.load(path)
        print(f"Scaler loaded successfully from {path}.")
        st.sidebar.success("✅ Scaler Loaded")
        if hasattr(scaler, 'n_features_in_'):
            st.sidebar.write(f"Scaler expects features: {scaler.n_features_in_}")
        # Optionally display mean/scale in debug mode
        if debug_mode:
            with st.sidebar.expander("Loaded Scaler Details (Debug)"):
                np.set_printoptions(precision=6, suppress=True)
                if hasattr(scaler, 'mean_'): st.write("**Mean:**", scaler.mean_)
                if hasattr(scaler, 'scale_'): st.write("**Scale (Std Dev):**", scaler.scale_)
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler from {path}: {e}")
        return None

# --- Load Artifacts ---
model = load_prediction_model(MODEL_PATH)
scaler = load_scaler_object(SCALER_PATH)

# --- Feature Definitions and Constraints ---
# Based on the model/scaler expecting 29 features
FINAL_FEATURE_COLS = [
    'Dur', 'RunTime', 'Mean', 'Sum', 'Min', 'Max', 'Proto', 'sTtl', 'sHops', 'Cause',
    'TotPkts', 'SrcPkts', 'DstPkts', 'TotBytes', 'SrcBytes', 'DstBytes', 'Offset',
    'sMeanPktSz', 'dMeanPktSz', 'Load', 'SrcLoad', 'DstLoad', 'Rate', 'SrcRate',
    'DstRate', 'State', 'TcpRtt', 'SynAck', 'AckDat'
]
st.sidebar.write(f"Using {len(FINAL_FEATURE_COLS)} features.")
# Optionally display features in sidebar expander
with st.sidebar.expander("View Model Features"):
    st.dataframe(pd.DataFrame({'Feature': FINAL_FEATURE_COLS}), use_container_width=True)


categorical_features_options = {
    'Proto': ['icmp', 'udp', 'tcp', 'sctp', 'ipv6-icmp'],
    'Cause': ['Start', 'Status', 'Shutdown'],
    'State': ['ECO', 'CON', 'REQ', 'TST', 'RST', 'INT', 'FIN', 'URP', 'RSP', 'NRS', 'ACC'],
    'sTtl': [32, 58, 60, 63, 64, 117, 128, 249, 252, 255]
}

# Default instance (using all 29 features)
default_benign_instance = {
    'Dur': 0.073378, 'RunTime': 0.073378, 'Mean': 0.073378, 'Sum': 0.073378,
    'Min': 0.073378, 'Max': 0.073378, 'Proto': 'udp', 'sTtl': 64, 'sHops': 0,
    'Cause': 'Status', 'TotPkts': 7, 'SrcPkts': 4, 'DstPkts': 3, 'TotBytes': 960,
    'SrcBytes': 760, 'DstBytes': 200, 'Offset': 47072, 'sMeanPktSz': 190.0,
    'dMeanPktSz': 66.666664, 'Load': 76753.25, 'SrcLoad': 62143.96875,
    'DstLoad': 14609.28418, 'Rate': 81.768379, 'SrcRate': 40.88419,
    'DstRate': 27.256126, 'State': 'CON', 'TcpRtt': 0.0, 'SynAck': 0.0, 'AckDat': 0.0
}

# --- Preprocessing Definitions ---
proto_mapping = {'icmp': 0, 'udp': 1, 'tcp': 2, 'sctp': 3, 'ipv6-icmp': 4}
cause_mapping = {'Start': 0, 'Status': 1, 'Shutdown': 2}
state_mapping = {'ECO': 0, 'CON': 1, 'REQ': 2, 'TST': 3, 'RST': 4, 'INT': 5, 'FIN': 6, 'URP': 7, 'RSP': 8, 'NRS': 9, 'ACC': 10}
reverse_proto_mapping = {v: k for k, v in proto_mapping.items()}
reverse_cause_mapping = {v: k for k, v in cause_mapping.items()}
reverse_state_mapping = {v: k for k, v in state_mapping.items()}

feature_mappings_to_apply = {
    'Proto': proto_mapping,
    'Cause': cause_mapping,
    'State': state_mapping,
}
label_mapping = {'Benign': 0, 'Malicious': 1}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# --- Preprocessing Functions ---
def apply_manual_encoding_deploy(df, mappings):
    """Applies predefined mappings to categorical features."""
    df_encoded = df.copy()
    cols_to_make_numeric = []

    for col, mapping in mappings.items():
        if col in df_encoded.columns:
            if not isinstance(mapping, dict): continue # Skip if mapping invalid

            if df_encoded.shape[0] != 1: raise ValueError("Encoding expects 1 row.")

            original_value = df_encoded.loc[0, col]

            if pd.isna(original_value):
                df_encoded.loc[0, col] = np.nan
            else:
                mapped_value = mapping.get(original_value)
                if mapped_value is None:
                    if debug_mode: print(f"Warning: Value '{original_value}' not in mapping for '{col}'. Setting NaN.")
                    df_encoded.loc[0, col] = np.nan
                else:
                    try:
                        df_encoded.loc[0, col] = float(mapped_value)
                    except (ValueError, TypeError):
                         if debug_mode: print(f"Warning: Mapped value '{mapped_value}' for '{col}' not numeric. Setting NaN.")
                         df_encoded.loc[0, col] = np.nan

            if col not in cols_to_make_numeric:
                 cols_to_make_numeric.append(col)

    # Explicitly convert columns processed by mappings to numeric
    if debug_mode: st.write(f"--- Debug: Attempting numeric conversion for: {cols_to_make_numeric} ---")
    for col in cols_to_make_numeric:
        if col in df_encoded.columns:
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')

    if debug_mode:
        st.write("--- Debug: DataFrame dtypes after explicit numeric conversion ---")
        st.write(df_encoded.dtypes)
    return df_encoded

def preprocess_single_input(input_dict, scaler_obj, feature_cols_list, mappings_to_apply):
    """Preprocesses a single dictionary input for prediction."""
    if debug_mode:
        st.divider()
        st.write("--- Debug: Preprocessing Start ---")
        st.write("Input Dictionary (from UI):", input_dict)

    input_df = pd.DataFrame([input_dict])
    if debug_mode: st.write("Input DataFrame (1 row) dtypes:", input_df.dtypes)

    # Apply manual encoding
    df_processed = input_df.copy()
    valid_mappings = {k: v for k, v in mappings_to_apply.items() if k in df_processed.columns}
    if debug_mode: st.write("Mappings to Apply:", valid_mappings)
    df_processed = apply_manual_encoding_deploy(df_processed, valid_mappings)
    if debug_mode:
        st.write("DataFrame AFTER Encoding - dtypes:", df_processed.dtypes)
        st.write("DataFrame AFTER Encoding - values:", df_processed)

    # Select features in the exact order required
    try:
        missing_cols = [col for col in feature_cols_list if col not in df_processed.columns]
        if missing_cols: return None, f"Error: Columns missing after encoding: {missing_cols}."

        df_features = df_processed[feature_cols_list]
        if debug_mode:
            st.write(f"DataFrame FEATURES selected ({len(feature_cols_list)}) - dtypes:", df_features.dtypes)
            st.write("Feature Order Used:", df_features.columns.tolist())
            st.write("Feature Values before Scaling:", df_features)

    except KeyError as e: return None, f"Error: Feature column '{e}' not found. Check FINAL_FEATURE_COLS."
    except Exception as e: return None, f"Error selecting features: {e}"

    # Check for NaNs after encoding/selection
    if df_features.isnull().values.any():
        nan_cols = df_features.columns[df_features.isnull().any(axis=0)].tolist()
        if debug_mode: st.warning(f"NaNs detected before scaling in: {nan_cols}. Check encoding/input.")
        return None, f"Error: Missing/invalid values detected before scaling in: {nan_cols}."

    # Final check for non-numeric types
    non_numeric_cols = df_features.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_cols:
        if debug_mode: st.error(f"Non-numeric columns STILL found before scaling: {non_numeric_cols}")
        return None, f"Error: Non-numeric columns found before scaling: {non_numeric_cols}."

    # Apply scaling
    try:
        if debug_mode:
            st.write(f"Scaler expects {scaler_obj.n_features_in_} features.")
            st.write(f"Data shape going into scaler: {df_features.shape}")

        if df_features.shape[1] != scaler_obj.n_features_in_:
             return None, f"Shape mismatch: Scaler needs {scaler_obj.n_features_in_}, data has {df_features.shape[1]}."

        scaled_data = scaler_obj.transform(df_features)
        if debug_mode:
            st.write("Scaled Data (NumPy array):", scaled_data)
            st.write(f"Scaled Data Shape: {scaled_data.shape}")

        if np.isnan(scaled_data).any():
            if debug_mode:
                 nan_producing_cols = df_features.columns[np.isnan(scaled_data).any(axis=0)].tolist()
                 st.warning(f"NaNs after scaling likely came from features: {nan_producing_cols}")
                 st.warning(f"Original values for these features: {df_features[nan_producing_cols].iloc[0]}")
            return None, "Error: NaNs generated during scaling."

    except ValueError as ve: return None, f"Scaling Error: {ve}."
    except Exception as e: return None, f"Error during scaling: {e}"

    if debug_mode: st.write("--- Debug: Preprocessing End ---"); st.divider()
    return scaled_data, None


# --- Initialize Session State ---
default_keys = default_benign_instance.keys()
for key in default_keys:
    if f"input_{key}" not in st.session_state:
        st.session_state[f"input_{key}"] = default_benign_instance[key]
if 'pasted_text' not in st.session_state:
    st.session_state.pasted_text = ""

# --- Paste-and-Fill Section ---
with st.expander("📋 Paste Instance Data (Optional)"):
    st.markdown("""
    Paste data in the format `Feature Name: Value` (one per line).
    *   Assumes Proto, Cause, State values are *encoded numbers* (e.g., `Proto: 1`).
    *   Only features used by the model (`FINAL_FEATURE_COLS`) will be parsed.
    """)
    pasted_text_area = st.text_area("Paste data here:", value=st.session_state.pasted_text, height=200, key="paste_area", label_visibility="collapsed")
    st.session_state.pasted_text = pasted_text_area # Keep text area content persistent across reruns

    if st.button("📄 Fill Inputs from Pasted Text", key="paste_button"):
        st.session_state.pasted_text = pasted_text_area # Update state first
        parsed_values = {}
        error_messages = []
        if st.session_state.pasted_text:
            lines = st.session_state.pasted_text.strip().split('\n')
            current_model_features = FINAL_FEATURE_COLS # Use the definitive list

            for i, line in enumerate(lines):
                line = line.strip()
                if not line: continue
                match = re.match(r"^\s*([a-zA-Z0-9_]+)\s*:\s*(.*)$", line)
                if match:
                    feature = match.group(1).strip()
                    value_str = match.group(2).strip()

                    # Only process features that are actually used by the model
                    if feature not in current_model_features: continue

                    # Handle known categorical features needing reverse mapping
                    if feature == 'Proto':
                        try:
                            val = int(float(value_str))
                            parsed_values[feature] = reverse_proto_mapping.get(val)
                            if parsed_values[feature] is None: error_messages.append(f"L{i+1}: Bad Proto code '{val}'")
                        except ValueError: error_messages.append(f"L{i+1}: Non-numeric Proto '{value_str}'")
                    elif feature == 'Cause':
                        try:
                            val = int(float(value_str))
                            parsed_values[feature] = reverse_cause_mapping.get(val)
                            if parsed_values[feature] is None: error_messages.append(f"L{i+1}: Bad Cause code '{val}'")
                        except ValueError: error_messages.append(f"L{i+1}: Non-numeric Cause '{value_str}'")
                    elif feature == 'State':
                        try:
                            val = int(float(value_str))
                            parsed_values[feature] = reverse_state_mapping.get(val)
                            if parsed_values[feature] is None: error_messages.append(f"L{i+1}: Bad State code '{val}'")
                        except ValueError: error_messages.append(f"L{i+1}: Non-numeric State '{value_str}'")
                    elif feature == 'sTtl':
                        try:
                            val = int(float(value_str))
                            if val not in categorical_features_options['sTtl']: error_messages.append(f"L{i+1}: TTL '{val}' invalid")
                            parsed_values[feature] = val
                        except ValueError: error_messages.append(f"L{i+1}: Non-numeric sTtl '{value_str}'")
                    # Handle other (numeric) features
                    else:
                        try:
                            parsed_values[feature] = float(value_str)
                            int_cols = ['sHops', 'TotPkts', 'SrcPkts', 'DstPkts', 'TotBytes', 'SrcBytes', 'DstBytes', 'Offset']
                            if feature in int_cols: parsed_values[feature] = int(parsed_values[feature])
                        except ValueError: error_messages.append(f"L{i+1}: Non-numeric '{feature}' value '{value_str}'")
                else: error_messages.append(f"L{i+1}: Bad format '{line}'")

            # Update session state
            if parsed_values:
                updated_count = 0
                for key, value in parsed_values.items():
                     if value is not None:
                         st.session_state[f"input_{key}"] = value
                         updated_count += 1
                if updated_count > 0: st.success(f"Applied {updated_count} valid inputs.")
                else: st.warning("No valid inputs found in pasted text.")
                # Force rerun to update widgets immediately after paste
                st.rerun()

            if error_messages:
                st.warning("Parsing issues found:")
                st.caption(", ".join(error_messages)) # More compact error display
        else:
            st.info("Paste area is empty.")


# --- Streamlit App UI ---
st.title("🚀 5G NIDD Live Flow Detection")
st.markdown("Enter the details for a network flow below, or paste data, then click Predict.")

user_inputs = {} # Collect current widget values

st.markdown("##### **Flow Timing & Packet Stats**")
col1, col2, col3 = st.columns(3)
# Use float() and int() to ensure correct types from session state where needed
with col1:
    user_inputs['Dur'] = st.number_input('Duration (s)', min_value=0.0, value=float(st.session_state.input_Dur), format="%.6f", key="dur_w")
    user_inputs['TotPkts'] = st.number_input('Total Packets', min_value=0, value=int(st.session_state.input_TotPkts), step=1, key="totpkts_w")
    user_inputs['SrcPkts'] = st.number_input('Source Packets', min_value=0, value=int(st.session_state.input_SrcPkts), step=1, key="srcpkts_w")
    user_inputs['DstPkts'] = st.number_input('Dest Packets', min_value=0, value=int(st.session_state.input_DstPkts), step=1, key="dstpkts_w")
with col2:
    user_inputs['RunTime'] = st.number_input('Run Time (s)', min_value=0.0, value=float(st.session_state.input_RunTime), format="%.6f", key="runtime_w")
    user_inputs['TotBytes'] = st.number_input('Total Bytes', min_value=0, value=int(st.session_state.input_TotBytes), step=1, key="totbytes_w")
    user_inputs['SrcBytes'] = st.number_input('Source Bytes', min_value=0, value=int(st.session_state.input_SrcBytes), step=1, key="srcbytes_w")
    user_inputs['DstBytes'] = st.number_input('Dest Bytes', min_value=0, value=int(st.session_state.input_DstBytes), step=1, key="dstbytes_w")
with col3:
    user_inputs['Mean'] = st.number_input('Mean Time', min_value=0.0, value=float(st.session_state.input_Mean), format="%.6f", key="mean_w")
    user_inputs['Sum'] = st.number_input('Sum Time', min_value=0.0, value=float(st.session_state.input_Sum), format="%.6f", key="sum_w")
    user_inputs['Min'] = st.number_input('Min Time', min_value=0.0, value=float(st.session_state.input_Min), format="%.6f", key="min_w")
    user_inputs['Max'] = st.number_input('Max Time', min_value=0.0, value=float(st.session_state.input_Max), format="%.6f", key="max_w")
    user_inputs['Offset'] = st.number_input('Offset', min_value=0, value=int(st.session_state.input_Offset), step=1, key="offset_w")

st.divider()
st.markdown("##### **Protocol, State & Routing**")
col1, col2, col3, col4 = st.columns(4)
with col1:
    proto_options = categorical_features_options['Proto']
    current_proto = st.session_state.input_Proto
    proto_index = proto_options.index(current_proto) if current_proto in proto_options else 0
    user_inputs['Proto'] = st.selectbox('Protocol', options=proto_options, index=proto_index, key="proto_w")
with col2:
    sttl_options = categorical_features_options['sTtl']
    current_sttl = st.session_state.input_sTtl
    sttl_index = sttl_options.index(current_sttl) if current_sttl in sttl_options else 0
    user_inputs['sTtl'] = st.selectbox('Source TTL', options=sttl_options, index=sttl_index, key="sttl_w")
with col3:
    user_inputs['sHops'] = st.number_input('Source Hops', min_value=0, value=int(st.session_state.input_sHops), step=1, key="shops_w")
with col4:
    state_options = categorical_features_options['State']
    current_state = st.session_state.input_State
    state_index = state_options.index(current_state) if current_state in state_options else 0
    user_inputs['State'] = st.selectbox('State', options=state_options, index=state_index, key="state_w")

st.divider()
st.markdown("##### **Network Load & Rate**")
col1, col2, col3 = st.columns(3)
with col1:
     user_inputs['Load'] = st.number_input('Load (bps)', min_value=0.0, value=float(st.session_state.input_Load), format="%.2f", key="load_w")
     user_inputs['SrcLoad'] = st.number_input('Source Load (bps)', min_value=0.0, value=float(st.session_state.input_SrcLoad), format="%.2f", key="srcload_w")
     user_inputs['DstLoad'] = st.number_input('Dest Load (bps)', min_value=0.0, value=float(st.session_state.input_DstLoad), format="%.2f", key="dstload_w")
with col2:
    user_inputs['Rate'] = st.number_input('Rate (pps)', min_value=0.0, value=float(st.session_state.input_Rate), format="%.2f", key="rate_w")
    user_inputs['SrcRate'] = st.number_input('Source Rate (pps)', min_value=0.0, value=float(st.session_state.input_SrcRate), format="%.2f", key="srcrate_w")
    user_inputs['DstRate'] = st.number_input('Dest Rate (pps)', min_value=0.0, value=float(st.session_state.input_DstRate), format="%.2f", key="dstrate_w")
with col3:
    user_inputs['sMeanPktSz'] = st.number_input('Src Mean Pkt Size', min_value=0.0, value=float(st.session_state.input_sMeanPktSz), format="%.2f", key="smeansz_w")
    user_inputs['dMeanPktSz'] = st.number_input('Dst Mean Pkt Size', min_value=0.0, value=float(st.session_state.input_dMeanPktSz), format="%.6f", key="dmeansz_w")
    cause_options = categorical_features_options['Cause']
    current_cause = st.session_state.input_Cause
    cause_index = cause_options.index(current_cause) if current_cause in cause_options else 0
    user_inputs['Cause'] = st.selectbox('Cause', options=cause_options, index=cause_index, key="cause_w")

st.divider()
st.markdown("##### **TCP Specific Timing (Enter 0 if N/A)**")
col1, col2, col3 = st.columns(3)
with col1:
    user_inputs['TcpRtt'] = st.number_input('TCP RTT (s)', min_value=0.0, value=float(st.session_state.input_TcpRtt), format="%.6f", key="tcprtt_w")
with col2:
    user_inputs['SynAck'] = st.number_input('SYN-ACK Time (s)', min_value=0.0, value=float(st.session_state.input_SynAck), format="%.6f", key="synack_w")
with col3:
    user_inputs['AckDat'] = st.number_input('ACK Data Time (s)', min_value=0.0, value=float(st.session_state.input_AckDat), format="%.6f", key="ackdat_w")


# --- Prediction Button and Output Area ---
st.divider()
col_btn, col_spacer = st.columns([1, 3]) # Button column less wide
with col_btn:
    predict_button_clicked = st.button("📊 Predict Flow Status", key="predict_button", type="primary", use_container_width=True)

output_placeholder = st.container() # Placeholder for results

with output_placeholder:
    if predict_button_clicked: # Only run prediction logic when button is clicked
        st.markdown("---")
        prediction_label_str = "Processing" # Initial status

        if model and scaler:
            # --- Update session state from current widget values ---
            for key, value in user_inputs.items():
                 st.session_state[f"input_{key}"] = value

            # --- Prepare Input for Preprocessing ---
            current_inputs_for_model = {}
            missing_ui_features = []
            for key in FINAL_FEATURE_COLS:
                if key in user_inputs: current_inputs_for_model[key] = user_inputs[key]
                else: missing_ui_features.append(key) # Should not happen if UI is correct

            if missing_ui_features:
                st.error(f"UI Error: Missing input widgets for: {missing_ui_features}")
                prediction_label_str = "Error"
            else:
                # --- Call Preprocessing ---
                with st.spinner('Preprocessing data...'): # Show spinner during preprocessing
                    preprocessed_data, error_msg = preprocess_single_input(
                        current_inputs_for_model, scaler, FINAL_FEATURE_COLS, feature_mappings_to_apply
                    )

                st.subheader("📈 Prediction Result")

                # --- Handle Preprocessing Outcome ---
                if error_msg:
                    st.warning(f"Preprocessing Error: {error_msg}")
                    prediction_label_str = "Error"
                elif preprocessed_data is not None:
                    # --- Make Prediction ---
                    try:
                        expected_features = model.input_shape[1]
                        if expected_features is not None and preprocessed_data.shape[1] != expected_features:
                             st.error(f"Shape mismatch! Model: {expected_features}, Data: {preprocessed_data.shape[1]}.")
                             prediction_label_str = "Error"
                        else:
                            with st.spinner('Predicting...'): # Show spinner during prediction
                                if debug_mode: st.write(f"--- Debug: Making prediction with data shape: {preprocessed_data.shape} ---")
                                prediction_proba = model.predict(preprocessed_data)[0][0]
                                if debug_mode: st.write(f"--- Debug: Raw prediction probability: {prediction_proba:.8f} ---")

                            # --- Interpret Prediction ---
                            prediction_label_num = 1 if prediction_proba > 0.5 else 0
                            prediction_label_str = reverse_label_mapping.get(prediction_label_num, "Unknown")

                            if prediction_label_num == 1: # Malicious
                                confidence_in_prediction = prediction_proba
                                conf_help = "Certainty score for 'Malicious' prediction."
                            else: # Benign
                                confidence_in_prediction = 1.0 - prediction_proba
                                confidence_in_prediction = confidence_in_prediction
                                conf_help = "Certainty score for 'Benign' prediction."

                            # --- Display Results ---
                            pred_col, conf_col = st.columns(2)
                            with pred_col:
                                if prediction_label_str == 'Malicious':
                                    st.metric("Predicted Class", prediction_label_str, "High Risk", delta_color="inverse")
                                elif prediction_label_str == 'Benign':
                                    st.metric("Predicted Class", prediction_label_str, "Low Risk", delta_color="normal")
                                else: st.metric("Predicted Class", "Unknown") # Fallback
                            with conf_col:
                                Perc_confidence_in_prediction = (confidence_in_prediction*100)
                                st.metric("Prediction Confidence", f"{perc_confidence_in_prediction:.4f}", help=conf_help)
                                st.progress(float(confidence_in_prediction))

                            # Add final status message
                            if prediction_label_str == 'Malicious': st.error("🚨 Potential Threat Detected!")
                            elif prediction_label_str == 'Benign': st.success("✅ Flow Appears Normal.")

                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        if debug_mode: # Show traceback only in debug mode
                             import traceback
                             st.error("Traceback:")
                             st.code(traceback.format_exc())
                        prediction_label_str = "Error"
                else: # Should not happen if error_msg logic is correct
                    st.error("Unknown preprocessing error.")
                    prediction_label_str = "Error"

        # --- Handle Missing Model/Scaler ---
        elif not model:
             st.error("Model not loaded. Check path and file.")
             prediction_label_str = "Error"
        elif not scaler:
             st.error("Scaler not loaded. Check path and file.")
             prediction_label_str = "Error"

        # --- Final Status Message ---
        if prediction_label_str == "Error":
            st.warning("Prediction could not be completed due to errors.")

    else:
         st.info("Click the 'Predict Flow Status' button to generate a prediction.")


# Footer
st.markdown("---")
st.caption("5G NIDD Detection App")
