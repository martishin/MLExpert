import numpy as np
from nltk import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


def tldr(text_to_summarize):
    sentence_tokens = np.array(sent_tokenize(text_to_summarize))
    stop_words_set = list(stopwords.words("english"))
    tf_idf_vectorizer = TfidfVectorizer(stop_words=stop_words_set)
    tf_idf = tf_idf_vectorizer.fit_transform(sentence_tokens)
    sentence_tf_idf_sums_matrix = tf_idf.sum(axis=1)
    sentence_tf_idf_sums_array = np.asarray(sentence_tf_idf_sums_matrix).squeeze()
    selected_sentence_indices = np.where(sentence_tf_idf_sums_array > 3)
    summary_sentences = sentence_tokens[selected_sentence_indices]
    summary = " ".join(summary_sentences)
    return summary


if __name__ == "__main__":
    with open("./text.txt", "r") as handle:
        text = handle.read()
    print(tldr(text))
