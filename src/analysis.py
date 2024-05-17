import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Lire le fichier log
with open('log.txt', 'r') as file:
    logs = file.readlines()

# Initialiser une liste pour stocker les données
data = []

# Utiliser une expression régulière pour extraire les sections "train"
pattern = re.compile(r"train\s*:\s*{([^}]*)}")

# Parcourir chaque ligne et extraire les données
for line in logs:
    match = pattern.search(line)
    if match:
        # Convertir la chaîne de caractères en dictionnaire
        log_entry = eval(f"{{{match.group(1)}}}")
        data.append(log_entry)

# Convertir la liste de dictionnaires en DataFrame pandas
df = pd.DataFrame(data)

# Tracer le minimum global de la perte totale au cours des étapes
plt.figure(figsize=(12, 6))
plt.plot(df['step'], df['pi_loss'], label='Total Loss', color='blue')
plt.xlabel('Step')
plt.ylabel('Total Loss (log scale)')
plt.title('Total Loss Over Training Steps (Log Scale)')
plt.legend()
plt.show()



# # Tracer les pertes au cours des étapes
# plt.figure(figsize=(12, 6))
# plt.plot(df['step'], df['consistency_loss'], label='Consistency Loss')
# plt.plot(df['step'], df['reward_loss'], label='Reward Loss')
# plt.plot(df['step'], df['value_loss'], label='Value Loss')
# plt.plot(df['step'], df['pi_loss'], label='Pi Loss')
# plt.plot(df['step'], df['totalpi'], label='Total Loss')
# plt.xlabel('Step')
# plt.ylabel('Loss')
# plt.title('Losses Over Training Steps')
# plt.legend()
# plt.show()

# # Tracer les récompenses par épisode
# plt.figure(figsize=(12, 6))
# plt.plot(df['episode'], df['episode_reward'], label='Episode Reward')
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.title('Episode Rewards Over Training Episodes')
# plt.legend()
# plt.show()

# # Tracer la norme des gradients au cours des étapes
# plt.figure(figsize=(12, 6))
# plt.plot(df['step'], df['grad_norm'], label='Grad Norm')
# plt.xlabel('Step')
# plt.ylabel('Grad Norm')
# plt.title('Gradient Norm Over Training Steps')
# plt.legend()
# plt.show()
