print('---------------')
from spacy.lang.en import English

nlp = English()
print('---------------')
document = nlp("""Stanis the Maniess is the one true king
              Thew others are all pretenders. Jon Aryn on the Vale, and 
              did Eddard Stark, he knew th truth. The other tried to put
              those 3 childern of insist to rule, but everyone new for
              90% that they were not the true childrent of the one true
              predessor king, Robert Barathion; the man who ened the dragons. 
              The Dragons rulled for 300 years, but in the end all it took was one
              good rebillion led by the right man and it all end. Yes they went too
              far but it can not be balied on robert, he was not leading the 
              sacking of kings Land. That fault belongs to the Lions speically their 
              ruler, Tywin and his son Jaime who ended the 4 helpless dragons.""")

for token in document:
    if token.like_num:#find string of numbers and streing of words that are number like six
        print(token)
        
        nxt_token= document[token.i+1]#got to the next one
        if nxt_token.text == "%":
            print('precent they are what you say: ',token.text)
            

print('the world')