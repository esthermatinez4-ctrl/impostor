from sentence_transformers import SentenceTransformer
import numpy as np
from numpy import dot
from numpy.linalg import norm
import random

jugadores = ["Esther", "Inés", "Marta", "Alejandra"]
num_rondas = 3

palabras_secretas = ["playa", "coche", "pizza", "gato", "montaña", "cine", "fútbol"]

puntos_impostor = 0
puntos_sistema = 0

modelo = SentenceTransformer('all-MiniLM-L6-v2')

def similitud_coseno(v1, v2):
    return dot(v1, v2) / (norm(v1) * norm(v2))

print("=== JUEGO DEL IMPOSTOR — MODO RONDAS ===\n")

for ronda in range(1, num_rondas + 1):
    print(f"\n--- RONDA {ronda} ---")

    palabra_secreta = random.choice(palabras_secretas)
    impostor = random.choice(jugadores)

    print(f"Palabra secreta (solo para jugadores reales): {palabra_secreta}")
    print(f"El impostor es: {impostor} (NO la conoce)\n")

    respuestas = {}

    for jugador in jugadores:
        if jugador == impostor:
            palabra = input(f"{jugador}, di una palabra (¡improvisa!): ")
        else:
            palabra = input(f"{jugador}, di una palabra relacionada: ")
        respuestas[jugador] = palabra

    corpus = [palabra_secreta] + list(respuestas.values())
    embeddings = modelo.encode(corpus)

    similitudes = {}
    for i, jugador in enumerate(jugadores):
        similitudes[jugador] = similitud_coseno(embeddings[0], embeddings[i+1])

    sospechoso = min(similitudes, key=similitudes.get)

    print("\nSimilitudes con la palabra secreta:")
    for jugador, sim in similitudes.items():
        print(f"{jugador}: {sim:.3f}")

    print(f"\nEl sistema cree que el impostor es: {sospechoso}")
    if sospechoso == impostor:
        print("✔ El sistema acertó.")
        puntos_sistema += 1
    else:
        print("✘ El impostor engañó al sistema.")
        puntos_impostor += 1
print("\n=== RESULTADO FINAL ===")
print(f"Puntos del sistema: {puntos_sistema}")
print(f"Puntos del impostor: {puntos_impostor}")

if puntos_sistema > puntos_impostor:
    print(" El sistema gana la partida")

elif puntos_impostor > puntos_sistema:
    print(" El impostor gana la partida.")

else:
    print(" Empate.")