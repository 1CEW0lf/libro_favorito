import spacy
import re

# ----------------------------------------
# CARGAR MODELO DE SPACY
# ----------------------------------------
nlp = spacy.load("es_core_news_sm")

# ----------------------------------------
# LEER LIBRO
# ----------------------------------------
with open("data/caperucita_roja.txt", "r", encoding="utf-8") as archivo:
    texto = archivo.read()

print("Texto original:\n")
print(texto[:500])

# ----------------------------------------
# NORMALIZACIÓN
# ----------------------------------------

# Minúsculas
texto = texto.lower()

# Eliminar números
texto = re.sub(r'\d+', '', texto)

# Eliminar espacios múltiples
texto = re.sub(r'\s+', ' ', texto)

# ----------------------------------------
# PROCESAMIENTO CON SPACY
# ----------------------------------------
doc = nlp(texto)

tokens_limpios = []

for token in doc:

    # Filtrar:
    # stopwords
    # puntuación
    # espacios

    if not token.is_stop and not token.is_punct and not token.is_space:

        # Lematización
        lemma = token.lemma_.lower()

        tokens_limpios.append(lemma)

# ----------------------------------------
# RESULTADO FINAL
# ----------------------------------------
texto_limpio = " ".join(tokens_limpios)

print("\nTexto limpio y lematizado:\n")
print(texto_limpio[:1000])

# ----------------------------------------
# GUARDAR RESULTADO
# ----------------------------------------
with open("texto_limpio.txt", "w", encoding="utf-8") as salida:
    salida.write(texto_limpio)

print("\nProceso completado correctamente")