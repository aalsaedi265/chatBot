from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer


my_bot = ChatBot(name='PyBot',read_only=True, logic_adapters=
                 ['chatterbot.logic.MathematicalEvaluation',#solve math
                  'chatterbot.logic.BestMatch'])#replise from list provided

small_talk=['hello there!', 'hi', 'hey, how you doin?', 'I\m sexy & I know it',
            'always doing great','fine, let get this over with',
            'name and what do you want', 'I m souless machine what do you want']

math_talk=['pythagorean theorem', 'a squared plus  b squared equals c squared']

math_talk2=['law of cosines', 'c**2= a**2 + b**2 - 2 * a * b * cos(gamma)']

list_trainer = ListTrainer(my_bot)

for item in (small_talk, math_talk, math_talk2):
    list_trainer.train(item)
    




corpus_trainer = ChatterBotCorpusTrainer(my_bot)
corpus_trainer.train('chatterbot.corpus.english')

corpus_trainer.train('chatterbot.corpus.english.greetings',
                     'chatterbot.corpus.english.conversations')


# while(True):
#     x = input("meat bag: ")
#     if x =='meat bag: end': break;
#     print(f'T800: {my_bot.get_response(x)}' )
