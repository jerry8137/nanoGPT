with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print('length of text: ', len(text))

# print(text[:1000])
chars = sorted(list(set(text)))
vocabulary_size = len(chars)
print(''.join(chars))
print('vocabulary size: ', vocabulary_size)
