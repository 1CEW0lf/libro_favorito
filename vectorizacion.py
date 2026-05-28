import spacy
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# CARGAR MODELO
# ---------------------------------------------------------
nlp = spacy.load("es_core_news_sm")

# ---------------------------------------------------------
# LEER LIBRO
# ---------------------------------------------------------
with open("data/caperucita_roja.txt", "r", encoding="utf-8") as archivo:
    texto = archivo.read()

# ---------------------------------------------------------
# PROCESAMIENTO
# ---------------------------------------------------------
doc = nlp(texto)

corpus_lematizado = []

for oracion in doc.sents:

    lemas_oracion = [

        token.lemma_.lower()

        for token in oracion

        if not token.is_punct
        and not token.is_space
        and not token.is_stop
    ]

    if lemas_oracion:

        corpus_lematizado.append(" ".join(lemas_oracion))

print(f"Total de oraciones procesadas: {len(corpus_lematizado)}")

# ---------------------------------------------------------
# BAG OF WORDS
# ---------------------------------------------------------
bow_vectorizer = CountVectorizer()

X_bow = bow_vectorizer.fit_transform(corpus_lematizado)

vocab_bow = bow_vectorizer.get_feature_names_out()

print("\nRepresentación Bag of Words")
print(X_bow.shape)

# ---------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------
tfidf_vectorizer = TfidfVectorizer()

X_tfidf = tfidf_vectorizer.fit_transform(corpus_lematizado)

vocab_tfidf = tfidf_vectorizer.get_feature_names_out()

print("\nRepresentación TF-IDF")
print(X_tfidf.shape)

# ---------------------------------------------------------
# FUNCIÓN PARA GRAFICAR
# ---------------------------------------------------------
def graficar_palabras_3d(ax, matriz, vocabulario, titulo, color):

    matriz_palabras = matriz.T

    pca = PCA(n_components=3)

    coords = pca.fit_transform(matriz_palabras.toarray())

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    ax.scatter(x, y, z, c=color, s=70)

    for i, palabra in enumerate(vocabulario):

        ax.text(x[i], y[i], z[i], palabra, fontsize=8)

    ax.set_title(titulo)

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")

# ---------------------------------------------------------
# VISUALIZACIÓN
# ---------------------------------------------------------
fig = plt.figure(figsize=(16, 7))

# --- BoW ---
ax1 = fig.add_subplot(121, projection='3d')

graficar_palabras_3d(
    ax1,
    X_bow,
    vocab_bow,
    "Espacio Vectorial BoW",
    "orange"
)

# --- TF-IDF ---
ax2 = fig.add_subplot(122, projection='3d')

graficar_palabras_3d(
    ax2,
    X_tfidf,
    vocab_tfidf,
    "Espacio Vectorial TF-IDF",
    "teal"
)

plt.tight_layout()

# Guardar imagen
plt.savefig("imagenes/espacio_vectorial.png")

plt.show()

print("\nVisualización generada correctamente")