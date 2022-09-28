class AnswerGenerator:
    """
    base class for answer generator, which is used to preprocess answer for each question
    """
    def __init__(self):
        pass

    def generate_answer(self):
        raise NotImplementedError

from word2number.w2n import word_to_num
import re
import itertools
import string
def get_number_from_word(word):
    """
    transform the word to number
    """
    punctuation = string.punctuation.replace('-', '')
    word = word.strip(punctuation)
    word = word.replace(",", "")
    try:
        number = word_to_num(word)
    except ValueError:
        try:
            number = int(word)
        except ValueError:
            try:
                number = float(word)
            except ValueError:
                if re.match('^\d*1st$', word):  # ending in '1st'
                    number = int(word[:-2])
                elif re.match('^\d*2nd$', word):  # ending in '2nd'
                    number = int(word[:-2])
                elif re.match('^\d*3rd$', word):  # ending in '3rd'
                    number = int(word[:-2])
                elif re.match('^\d+th$', word):  # ending in <digits>th
                    # Many occurrences are when referring to centuries (e.g "the *19th* century")
                    number = int(word[:-2])
                elif len(word) > 1 and word[-2] == '0' and re.match('^\d+s$', word):
                    # Decades, e.g. "1960s".
                    # Other sequences of digits ending with s (there are 39 of these in the training
                    # set), do not seem to be arithmetically related, as they are usually proper
                    # names, like model numbers.
                    number = int(word[:-1])
                elif len(word) > 4 and re.match('^\d+(\.?\d+)?/km[²2]$', word):
                    # per square kilometer, e.g "73/km²" or "3057.4/km2"
                    if '.' in word:
                        number = float(word[:-4])
                    else:
                        number = int(word[:-4])
                elif len(word) > 6 and re.match('^\d+(\.?\d+)?/month$', word):
                    # per month, e.g "1050.95/month"
                    if '.' in word:
                        number = float(word[:-6])
                    else:
                        number = int(word[:-6])
                else:
                    return None
    return number