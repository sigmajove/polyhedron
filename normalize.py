top_weights = [950, 920, 930, 950]
avg = sum(top_weights) / len(top_weights)

for w in top_weights:
    print(w * 1000.0 / avg)

