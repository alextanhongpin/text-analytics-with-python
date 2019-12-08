# Implement a function for standardizing our text data by bringing word tokens to their base or root form using lemmatization.

from pattern.en import tag
from nltk.corpus import wordnet as wn

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()


# Annotate text tokens with POS tags.
def pos_tag_text(text):
    # Convert Penn treebank tag to wordnet tag.
    def penn_to_wn_pos_tag(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None

    tagged_text = tag(text)
    tagged_text_lower = [(word.lower(), penn_to_wn_pos_tag(pos_tag))
                         for word, pos_tag in tagged_text]
    return tagged_text_lower


# Lemmatize text based on POS tag.
def lemmatize_text(text):
    pos_tagged_text = pos_tag_text(text)
    lemmatized_tokens = [
        wnl.lemmatize(word, pos_tag) if pos_tag else word
        for word, pos_tag in pos_tagged_text
    ]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text
