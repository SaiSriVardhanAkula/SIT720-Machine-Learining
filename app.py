import streamlit as st
import pandas as pd
import joblib

MODEL_FILE = "sydney_house_price_model.pkl"

model = joblib.load(MODEL_FILE)

st.set_page_config(
    page_title="Sydney Housing Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("Sydney Housing Price Prediction")
st.write(
    "Enter property details below to estimate the sale price "
    "using the trained machine learning model."
)

suburb = st.selectbox(
    "Suburb",
    ["Killara", "Freshwater", "Eagle Vale"]
)

postcode_map = {
    "Killara": "2071",
    "Freshwater": "2096",
    "Eagle Vale": "2558"
}

postcode = postcode_map[suburb]

bedrooms = st.number_input(
    "Bedrooms", min_value=1, max_value=10, value=3
)

bathrooms = st.number_input(
    "Bathrooms", min_value=1, max_value=10, value=2
)

parking = st.number_input(
    "Parking Spaces", min_value=0, max_value=10, value=1
)

property_type = st.selectbox(
    "Property Type",
    ["house", "townhouse", "villa", "duplex/semi-detached"]
)

sale_year = st.number_input(
    "Sale Year", min_value=2000, max_value=2030, value=2018
)

sale_month = st.slider(
    "Sale Month", min_value=1, max_value=12, value=6
)

total_room_feature = bedrooms + bathrooms
amenity_count = bedrooms + bathrooms + parking

input_data = pd.DataFrame({
    "Suburb": [suburb],
    "Postcode": [postcode],
    "Bedrooms": [bedrooms],
    "Bathrooms": [bathrooms],
    "Parking": [parking],
    "Property_Type": [property_type],
    "Sale_Year": [sale_year],
    "Sale_Month": [sale_month],
    "Total_Room_Feature": [total_room_feature],
    "Amenity_Count": [amenity_count]
})

if st.button("Predict Sale Price"):
    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Sale Price: ${prediction:,.0f}")

    st.subheader("Property Summary")
    st.write("Suburb:", suburb)
    st.write("Postcode:", postcode)
    st.write("Bedrooms:", bedrooms)
    st.write("Bathrooms:", bathrooms)
    st.write("Parking:", parking)
    st.write("Property Type:", property_type)

st.caption(
    "Prediction is an estimate based on historical housing data "
    "and should not be treated as a professional property valuation."
)
