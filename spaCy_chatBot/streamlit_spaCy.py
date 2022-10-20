
#https://blog.jcharistech.com/2020/07/09/simple-nlp-app-with-spacy-streamlit/ 

# Core Pkgs
import streamlit as st 

# NLP Pkgs
import spacy_streamlit
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
#python -m spacy download en_core_web_sm
# nlp = spacy.load('en_core_web_sm"')


def main():
  '''The whole faith problem arose because of poor leadership.
      If Joffery or Tywin were around they would have ended them.
      Tommon so so weak, which is my failed at everything.'''
      
  st.title('Spacy_Streamlit NLP App')
  menu = ['Home',"NER"]
  choice = st.sidebar.selectbox('Menu', menu)
  
  if choice == 'Home':
    st.subheader('Tokenization')
    raw_text = st.text_area("Your Text","Enter text")
    docx = nlp(raw_text)
    if st.button("Tokenize"):
      spacy_streamlit.visualize_tokens(docx,attrs=['text','pos_','dep_','ent_type_'])
  
  elif choice == "NER":
    st.subheader("Named Entity Recognition")
    raw_text = st.text_area('Your Text',"Enter Text Here")
    docx= nlp(raw_text)
    spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)
      
if __name__== '__main__':
  main()