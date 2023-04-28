import streamlit as st

from predict import *

st.set_page_config(layout="wide")


def main():
    st.title("STS demo")
    st.write("#### 두 문장간의 유사도를 pearson correlation으로 측정합니다.")
    st.write("snunlp/KR-ELECTRA-discriminator를 fine-tuning한 단일 모델을 사용합니다.")

    with st.form(key="my_form"):
        st.write("### Input your sentence here")
        sentence1 = st.text_input("Enter your first sentence here")
        sentence2 = st.text_input("Enter your second sentence here")
        submit = st.form_submit_button("Submit")
        if submit:
            st.write("### Similarity")
            st.write("0 ~ 5 사이의 값으로 나타납니다. 0에 가까울수록 유사도가 낮고, 5에 가까울수록 유사도가 높습니다.")
            st.write(predict("snunlp/KR-ELECTRA-discriminator", sentence1, sentence2))
            # st.write(predict("klue/roberta-large", sentence1, sentence2))


if __name__ == "__main__":
    main()
