with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print('length of text: ', len(text))

# print(text[:1000])
chars = sorted(list(set(text)))
vocabulary_size = len(chars)
print(''.join(chars))
print('vocabulary size: ', vocabulary_size)

stoi = {char: i for i, char in enumerate(chars)}


def encode(input):
    output = []
    for i in input:
        output.append(stoi[i])
    return output


print(encode('hello world'))
