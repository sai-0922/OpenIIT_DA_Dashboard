import streamlit as st
from  plots import *

def main():
    st.set_page_config(page_title='Dash board',page_icon="🕗", layout="wide", initial_sidebar_state="auto", menu_items=None)
    st.title("Dashboard: Predicting Tourist Arrivals to Tirupati")

    st.markdown("---")
    st.markdown("---")
    
    st.write(pilplot())

    st.markdown("<p style='margin: 30px;'></p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='margin: 30px;'></p>", unsafe_allow_html=True)


    st.header("Google Trends Plots Illustrating Various Search Indices")
    st.write(featureplot())

    st.markdown("<p style='margin: 30px;'></p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='margin: 30px;'></p>", unsafe_allow_html=True)

    st.header("Visualizing Feature Correlations with a Heatmap")
    st.write(heatmap())

    st.markdown("<p style='margin: 30px;'></p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='margin: 30px;'></p>", unsafe_allow_html=True)

    st.header("AutoCorrelation function (ACF) demonstrating seasonality")
    ACFPlot()

    st.markdown("<p style='margin: 30px;'></p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='margin: 30px;'></p>", unsafe_allow_html=True)

    st.header("")
    st.write(plot_original_vs_predicted())

    st.markdown("<p style='margin: 30px;'></p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='margin: 30px;'></p>", unsafe_allow_html=True)

    st.header("Predictions")
    st.write(sarima_nhits_kalman())

    st.markdown("<p style='margin: 30px;'></p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='margin: 30px;'></p>", unsafe_allow_html=True)

    st.header("xgboost model Deviance")
    st.write(xgboost())


if __name__==main():
    main()    