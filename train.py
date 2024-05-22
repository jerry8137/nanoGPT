with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print('length of text: ', len(text))

# print(text[:1000])
chars = sorted(list(set(text)))
vocabulary_size = len(chars)
print(''.join(chars))
print('vocabulary size: ', vocabulary_size)

stoi = {char: i for i, char in enumerate(chars)}
itos = {i: char for i, char in enumerate(chars)}


def encode(input): return [stoi[i] for i in input]
def decode(input): return ''.join([itos[i] for i in input])


print(encode('hello world'))
print(decode(encode('hello world')))
