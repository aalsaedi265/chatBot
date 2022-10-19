import spacy
nlp = spacy.load("en_core_web_md")

doc = nlp("I live in Texas for the past two years.")

for entity in doc.ents:
   
  print(entity.text + " | " + entity.label_)
  