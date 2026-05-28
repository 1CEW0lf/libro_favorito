import spacy
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# ---------------------------------------------------------
# CARGAR MODELO
# ---------------------------------------------------------
nlp = spacy.load("es_core_news_md")

# ---------------------------------------------------------
# LEER TEXTO
# ---------------------------------------------------------
with open("data/caperucita_roja.txt", "r", encoding="utf-8") as archivo:

    texto = archivo.read()

# ---------------------------------------------------------
# PROCESAR TEXTO
# ---------------------------------------------------------
doc = nlp(texto)

palabras = []
vectores = []

# ---------------------------------------------------------
# EXTRAER EMBEDDINGS
# ---------------------------------------------------------
for token in doc:

    if (
        not token.is_stop
        and not token.is_punct
        and not token.is_space
        and token.has_vector
    ):

        palabras.append(token.lemma_.lower())

        vectores.append(token.vector)

# ---------------------------------------------------------
# REDUCCIÓN DE DIMENSIONALIDAD
# ---------------------------------------------------------
pca = PCA(n_components=2)

coords = pca.fit_transform(vectores)

# ---------------------------------------------------------
# GRAFICAR
# ---------------------------------------------------------
plt.figure(figsize=(12, 10))

x = coords[:, 0]
y = coords[:, 1]

plt.scatter(x, y)

for i, palabra in enumerate(palabras[:100]):

    plt.annotate(palabra, (x[i], y[i]), fontsize=8)

plt.title("Espacio Semántico Distribucional")

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

# ---------------------------------------------------------
# GUARDAR IMAGEN
# ---------------------------------------------------------
plt.savefig("imagenes/espacio_word2vec.png")

plt.show()

print("Visualización semántica generada correctamente")