import streamlit as st
# import inflect
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import string
import plotly.express as px
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

# Note - USE "VBA_venv" environment in the local github folder

punctuations = string.punctuation

def prep_text(text):
    # function for preprocessing text

    # remove trailing characters (\s\n) and convert to lowercase
    clean_sents = [] # append clean con sentences
    sent_tokens = sent_tokenize(str(text))
    for sent_token in sent_tokens:
        word_tokens = [str(word_token).strip().lower() for word_token in sent_token.split()]
        word_tokens = [word_token for word_token in word_tokens if word_token not in punctuations]
        clean_sents.append(' '.join((word_tokens)))
    joined = ' '.join(clean_sents).strip(' ')
    return joined


# model name or path to model
checkpoint_1 = "Highway/SubCat"

checkpoint_2 = "Highway/ExtraOver"


@st.cache(allow_output_mutation=True)
def load_model_1():
    return AutoModelForSequenceClassification.from_pretrained(checkpoint_1)


@st.cache(allow_output_mutation=True)
def load_tokenizer_1():
    return AutoTokenizer.from_pretrained(checkpoint_1)


@st.cache(allow_output_mutation=True)
def load_model_2():
    return AutoModelForSequenceClassification.from_pretrained(checkpoint_2)


@st.cache(allow_output_mutation=True)
def load_tokenizer_2():
    return AutoTokenizer.from_pretrained(checkpoint_2)


st.set_page_config(
    page_title="Cost Data Classifier", layout= "wide", initial_sidebar_state="auto", page_icon="💷"
)

st.title("🚦 AI Infrastructure Cost Data Classifier")
# st.header("")

with st.expander("About this app", expanded=False):
    st.write(
        """
        - Artificial Intelligence (AI) and Machine learning (ML) tool for automatic classification of infrastructure cost data for benchmarking
        - Classifies cost descriptions from documents such as Bills of Quantities (BOQs) and Schedule of Rates
        - Can be trained to classify granular and itemised cost descriptions into any predefined categories for benchmarking
        - Contact research team to discuss your data structures and suitability for the app
        - It is best to use this app on a laptop or desktop computer
        """
    )


st.markdown("##### Description")
with st.form(key="my_form"):
    Text_entry = st.text_area(
        "Paste or type infrastructure cost description in the text box below (i.e., input)"
    )
    submitted = st.form_submit_button(label="👉 Get SubCat and ExtraOver!")

if submitted:

    # First prediction

    label_list_1 = [
        'Arrow, Triangle, Circle, Letter, Numeral, Symbol and Sundries',
        'Binder',
        'Cable',
        'Catman Other Adjustment',
        'Cold Milling',
        'Disposal of Acceptable/Unacceptable Material',
        'Drain/Service Duct In Trench',
        'Erection & Dismantling of Temporary Accommodation/Facilities (All Types)',
        'Excavate And Replace Filter Material/Recycle Filter Material',
        'Excavation',
        'General TM Item',
        'Information boards',
        'Joint/Termination',
        'Line, Ancillary Line, Solid Area',
        'Loop Detector Installation',
        'Minimum Lining Visit Charge',
        'Node Marker',
        'PCC Kerb',
        'Provision of Mobile Welfare Facilities',
        'Removal of Deformable Safety Fence',
        'Removal of Line, Ancillary Line, Solid Area',
        'Removal of Traffic Sign and post(s)',
        'Road Stud',
        'Safety Barrier Or Bifurcation (Non-Concrete)',
        'Servicing of Temporary Accommodation/Facilities (All Types) (day)',
        'Tack Coat',
        'Temporary Road Markings',
        'Thin Surface Course',
        'Traffic Sign - Unknown specification',
        'Vegetation Clearance/Weed Control (m2)',
        'Others'
    ]

    joined_clean_sents = prep_text(Text_entry)

    # tokenize
    tokenizer_1 = load_tokenizer_1()
    tokenized_text_1 = tokenizer_1(joined_clean_sents, return_tensors="pt")

    # predict
    model_1 = load_model_1()
    text_logits_1 = model_1(**tokenized_text_1).logits
    predictions_1 = torch.softmax(text_logits_1, dim=1).tolist()[0]
    predictions_1 = [round(a, 3) for a in predictions_1]

    # dictionary with label as key and percentage as value
    pred_dict_1 = (dict(zip(label_list_1, predictions_1)))

    # sort 'pred_dict' by value and index the highest at [0]
    sorted_preds_1 = sorted(pred_dict_1.items(), key=lambda x: x[1], reverse=True)

    # Make dataframe for plotly bar chart
    u_1, v_1 = zip(*sorted_preds_1)
    x_1 = list(u_1)
    y_1 = list(v_1)
    df2 = pd.DataFrame()
    df2['SubCatName'] = x_1
    df2['Likelihood'] = y_1

    c1, c2, c3 = st.columns([1.5, 0.5, 1])

    with c1:
        st.header("SubCatName")
        # plot graph of predictions
        fig = px.bar(df2, x="Likelihood", y="SubCatName", orientation="h")

        fig.update_layout(
            # barmode='stack',
            template='ggplot2',
            font=dict(
                family="Arial",
                size=14,
                color="black"
            ),
            autosize=False,
            width=800,
            height=500,
            xaxis_title="Likelihood of SubCatName",
            yaxis_title="SubCatNames",
            # legend_title="Topics"
        )

        fig.update_xaxes(tickangle=0, tickfont=dict(family='Arial', color='black', size=14))
        fig.update_yaxes(tickangle=0, tickfont=dict(family='Arial', color='black', size=14))
        fig.update_annotations(font_size=14)  # this changes y_axis, x_axis and subplot title font sizes

        # Plot
        st.plotly_chart(fig, use_container_width=False)

    with c3:
        st.header("")
        predicted_1 = st.metric("Predicted SubCatName", sorted_preds_1[0][0])
        Prediction_confidence_1 = st.metric("Prediction confidence", (str(round(sorted_preds_1[0][1]*100, 1))+"%"))

        st.success("Great! SubCatName successfully predicted. ", icon="✅")


    # Second prediction

    label_list_2 = ["False", "True"]

    joined_clean_sents = prep_text(Text_entry)

    # tokenize
    tokenizer_2 = load_tokenizer_2()
    tokenized_text_2 = tokenizer_2(joined_clean_sents, return_tensors="pt")

    # predict
    model_2 = load_model_2()
    text_logits_2 = model_2(**tokenized_text_2).logits
    predictions_2 = torch.softmax(text_logits_2, dim=1).tolist()[0]
    predictions_2 = [round(a_, 3) for a_ in predictions_2]

    # dictionary with label as key and percentage as value
    pred_dict_2 = (dict(zip(label_list_2, predictions_2)))

    # sort 'pred_dict' by value and index the highest at [0]
    sorted_preds_2 = sorted(pred_dict_2.items(), key=lambda x: x[1], reverse=True)

    # Make dataframe for plotly bar chart
    u_2, v_2 = zip(*sorted_preds_2)
    x_2 = list(u_2)
    y_2 = list(v_2)
    df3 = pd.DataFrame()
    df3['ExtraOver'] = x_2
    df3['Likelihood'] = y_2

    d1, d2, d3 = st.columns([1.5, 0.5, 1])

    with d1:
        st.header("ExtraOver")
        # plot graph of predictions
        fig = px.bar(df3, x="Likelihood", y="ExtraOver", orientation="h")

        fig.update_layout(
            # barmode='stack',
            template='ggplot2',
            font=dict(
                family="Arial",
                size=14,
                color="black"
            ),
            autosize=False,
            width=800,
            height=200,
            xaxis_title="Likelihood of ExtraOver",
            yaxis_title="ExtraOver",
            # legend_title="Topics"
        )

        fig.update_xaxes(tickangle=0, tickfont=dict(family='Arial', color='black', size=14))
        fig.update_yaxes(tickangle=0, tickfont=dict(family='Arial', color='black', size=14))
        fig.update_annotations(font_size=14)  # this changes y_axis, x_axis and subplot title font sizes

        # Plot
        st.plotly_chart(fig, use_container_width=False)

    with d3:
        st.header("")
        predicted_2 = st.metric("Predicted ExtraOver", sorted_preds_2[0][0])
        Prediction_confidence_2 = st.metric("Prediction confidence", (str(round(sorted_preds_2[0][1]*100, 1))+"%"))

        st.success("Great! ExtraOver successfully predicted. ", icon="✅")
