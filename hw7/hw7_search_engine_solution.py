import pymorphy2
from razdel import sentenize
from razdel import tokenize
from collections import Counter
import requests
import os
import math
import re
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

morph = pymorphy2.MorphAnalyzer()


def get_wikipedia_article(article_title):
    api_url = f"https://ru.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": article_title,
        "format": "json",
    }

    response = requests.get(api_url, params=params, verify=False)

    data = response.json()

    pages = data["query"]["pages"]
    page = next(iter(pages.values()))  # достаём первую страницу
    if "extract" in page:
        return page["extract"]
    else:
        return None


def create_corpus():
    if not os.path.exists("corpus.txt"):
        with open("corpus.txt", "w", encoding="utf-8") as f:
            articles = [
                "Бобровый",
                "Облепиха",
                "Гватемальско-испанские отношения",
                "Сожжение посольства Испании в Гватемале",
                "Булгаков, Михаил Афанасьевич",
                "Булгакова, Варвара Михайловна",
            ]

            corpus = ""
            for article in articles:
                text = get_wikipedia_article(article)
                if text is not None:
                    corpus += text + "\n"
                else:
                    print(f"Не удалось получить текст статьи {article}")

            f.write(corpus)
    else:
        with open("corpus.txt", "r", encoding="utf-8") as f:
            corpus = f.read()

    return corpus


def normalize_sentence(sentence):
    stop_words = set(stopwords.words("russian"))
    sentence = re.sub(r"[^\w\s]", "", sentence)
    tokens = list(tokenize(sentence))
    normalized_tokens = [
        morph.parse(token.text)[0].normal_form
        for token in tokens
        if morph.parse(token.text)[0].normal_form not in stop_words
    ]
    return normalized_tokens


def compute_tf(sentence_tokens):
    return Counter(sentence_tokens)


def compute_df(documents):
    df = Counter()
    for doc in documents:
        unique_words = set(doc)
        df.update(unique_words)
    return df


def compute_idf(word, df, N):
    return math.log(N / (df[word] + 1))


def compute_tfidf(tf, df, N):
    tfidf = {}
    for word, freq in tf.items():
        idf = compute_idf(word, df, N)
        tfidf[word] = freq * idf
    return tfidf


def cosine_similarity(vec1, vec2):
    dot_product = sum(vec1[word] * vec2.get(word, 0) for word in vec1)
    norm_vec1 = math.sqrt(sum(val**2 for val in vec1.values()))
    norm_vec2 = math.sqrt(sum(val**2 for val in vec2.values()))

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)


def rank_documents_by_tf(query_tf, documents_tf):
    similarities = []
    query_norm = math.sqrt(sum(val**2 for val in query_tf.values()))

    for i, doc_tf in enumerate(documents_tf):
        common_words = set(query_tf.keys()) & set(doc_tf.keys())
        dot_product = sum(query_tf[word] * doc_tf[word] for word in common_words)
        doc_norm = math.sqrt(sum(val**2 for val in doc_tf.values()))
        if query_norm == 0 or doc_norm == 0:
            similarity = 0.0
        else:
            similarity = dot_product / (query_norm * doc_norm)

        similarities.append((i, similarity))

    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return sorted_similarities


def rank_documents_by_tfidf(query_tfidf, documents_tfidf):
    similarities = []
    for i, doc_tfidf in enumerate(documents_tfidf):
        similarity = cosine_similarity(query_tfidf, doc_tfidf)
        similarities.append((i, similarity))
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return sorted_similarities


# вероятность релевантности документа (предложения) по отношению к запросу
def language_model(
    query, sentence_tokens, word_frequencies, total_words, lambda_value=0.5, alpha=1.0
):
    counter = Counter(sentence_tokens)
    prob = 1.0
    V = len(word_frequencies)

    for word in query:
        P_ml = (counter[word] + alpha) / (len(sentence_tokens) + alpha * V)
        P_corpus = (word_frequencies.get(word, 0) + alpha) / (total_words + alpha * V)
        prob *= lambda_value * P_ml + (1 - lambda_value) * P_corpus

    return prob


def rank_documents_by_relevance(
    query, documents, word_frequencies, total_words, lambda_value=0.1, alpha=1.0
):
    probabilities = []
    for i, sentence_tokens in enumerate(documents):
        prob = language_model(
            query, sentence_tokens, word_frequencies, total_words, lambda_value, alpha
        )
        probabilities.append((i, prob))
    sorted_probabilities = sorted(probabilities, key=lambda x: x[1], reverse=True)

    return sorted_probabilities


def format_scientific(number, precision=3):
    formatted_number = f"{number:.{precision}e}"
    mantissa, exponent = formatted_number.split("e")
    exponent = int(exponent)

    return f"{mantissa} * 10^{exponent}"


def main():
    corpus = create_corpus()

    # разделяем тексты на предложения
    sentences = list(sentenize(corpus))

    # обработка морфологическим анализатором
    normalized_doc = [normalize_sentence(sent.text) for sent in sentences]

    queries = [
        "На московском острове растёт восьмиметровая облепиха",
        "Гватемальско-испанские отношения были разорваны после того, как посольство Испании было сожжено",
        "Булгаков не смог приехать на похороны матери из-за отсутствия денег",
    ]

    normalized_queries = [normalize_sentence(query) for query in queries]

    # TF для каждого предложения и запроса
    tf_documents = [compute_tf(sent) for sent in normalized_doc]
    tf_queries = [compute_tf(query) for query in normalized_queries]

    # количество предложений, в которых встречалось каждое из слов документа
    df = compute_df(normalized_doc)

    # общее количество предложений
    N = len(normalized_doc)

    # Подсчет TF-IDF для всех предложений и запросов
    tfidf_documents = [compute_tfidf(tf_doc, df, N) for tf_doc in tf_documents]
    tfidf_queries = [compute_tfidf(tf_query, df, N) for tf_query in tf_queries]

    # частоты слов в коллекции
    word_frequencies = Counter(word for sent in normalized_doc for word in sent)

    total_words_in_corpus = sum(word_frequencies.values())

    for qid, query in enumerate(queries):
        print(f"\nЗапрос {qid+1}: {query}")
        normalized_query = normalized_queries[qid]

        # TF ранжирование
        ranked_tf = rank_documents_by_tf(tf_queries[qid], tf_documents)
        print("\nМодель TF:")
        for rank, (idx, sim) in enumerate(ranked_tf[:3], 1):
            print(f"Ранг {rank}, предложение {idx+1}, TF: {sim:.6f}")
            print(f"{sentences[idx].text}\n")

        # TF-IDF ранжирование
        ranked_tfidf = rank_documents_by_tfidf(tfidf_queries[qid], tfidf_documents)
        print("\nМодель TF-IDF:")
        for rank, (idx, sim) in enumerate(ranked_tfidf[:3], 1):
            print(f"Ранг {rank}, предложение {idx+1}, TF-IDF: {sim:.6f}")
            print(f"{sentences[idx].text}\n")

        # языковая модель
        ranked_language_model = rank_documents_by_relevance(
            normalized_query,
            normalized_doc,
            word_frequencies,
            total_words_in_corpus,
            lambda_value=0.5,
            alpha=1.0,
        )
        print("\nЯзыковая модель:")
        for rank, (idx, prob) in enumerate(ranked_language_model[:3], 1):
            print(
                f"Ранг {rank}, предложение {idx+1}, вероятность: {format_scientific(prob)}"
            )
            print(f"{sentences[idx].text}\n")


if __name__ == "__main__":
    main()
