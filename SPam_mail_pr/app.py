import streamlit as st
import joblib

# Load the saved model
model_path = 'spam_model_pipeline.pkl'
loaded_model = joblib.load(model_path)

# Streamlit App Title
st.title("Spam Email Detector")
st.write("Predict whether multiple emails are spam or not using a trained machine learning model.")

# Placeholder for dynamic input fields
emails = st.session_state.get("emails", [])

if "emails" not in st.session_state:
    st.session_state.emails = []

# Function to add a new email input field
def add_email_field():
    st.session_state.emails.append("")

# Function to clear all email fields
def clear_email_fields():
    st.session_state.emails = []

# Add a button to add new email input fields
st.button("Add Email Field", on_click=add_email_field)

# Email Input Fields
st.write("### Enter Email Content Below")
for idx, email in enumerate(st.session_state.emails):
    st.session_state.emails[idx] = st.text_area(f"Email {idx+1}:", email, key=f"email_{idx}")

# Predict Button
if st.button("Predict"):
    if st.session_state.emails:
        predictions = loaded_model.predict(st.session_state.emails)
        
        # Display results for each email
        for idx, prediction in enumerate(predictions):
            st.write(f"**Email {idx+1}:**")
            if prediction == 1:
                st.error(f"🚨 This email is classified as **SPAM**.")
            else:
                st.success(f"✅ This email is classified as **NOT SPAM (HAM)**.")
    else:
        st.warning("⚠️ Please add at least one email to predict.")

# Clear Button
if st.button("Clear All Emails", on_click=clear_email_fields):
    st.success("All email fields cleared!")

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    """
    **Spam Email Detector** helps you identify spam emails using a pre-trained machine learning model. 
    - **Model Type**: Pre-trained ML pipeline
    - **Prediction Output**: 
        - 1 = Spam
        - 0 = Not Spam (Ham)
    """
)

st.sidebar.title("How It Works")
st.sidebar.markdown(
    """
    1. Use the **Add Email Field** button to create input fields for emails.
    2. Enter the email content in each field.
    3. Click the **Predict** button to classify all emails.
    4. Use the **Clear All Emails** button to reset the input fields.
    """
)

