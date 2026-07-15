import streamlit as st
import joblib
import pandas as pd

model = joblib.load("KNeighborsClassifier_seed_model.pkl")

st.title("🌾 Wheat Seed Type Predictor")
st.write("Identify the wheat seed variety using the KNN algorithm:")

st.header("Enter the seed measurements:")

area = st.number_input("Area", min_value=0.0, value=15.0)
perimeter = st.number_input("Perimeter", min_value=0.0, value=14.5)
compactness = st.number_input("Compactness", min_value=0.0, max_value=1.0, value=0.87)
length = st.number_input("Length", min_value=0.0, value=5.5)
width = st.number_input("Width", min_value=0.0, value=3.2)
asymmetry = st.number_input("Asymmetry Coefficient", min_value=0.0, value=2.5)
groove = st.number_input("Groove Length", min_value=0.0, value=5.0)

if st.button("Generate prediction"):
    input_data = pd.DataFrame([[area, perimeter, compactness, length, width, asymmetry, groove]],
                                columns=['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 
                                         'AsymmetryCoeff', 'Groove'])
    
    prediction = model.predict(input_data)
    
    variety_names = {1: "Kama", 2: "Rosa", 3: "Canadian"}
    predicted_variety = variety_names.get(prediction[0], prediction[0])
    
    st.success(f"🌾 This is the wheat seed variety: **{predicted_variety}** (Type {prediction[0]})")