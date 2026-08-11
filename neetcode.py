s = ["hello", "world"]

# ---------------- ENCODE ----------------

encoded = ""

for word in s:
    encoded += str(len(word)) + "#" + word

print("Encoded:", encoded)


# ---------------- DECODE ----------------

decoded = []
i = 0

while i < len(encoded):

    # Get the length
    separator = encoded.find("#", i)
    length = int(encoded[i:separator])

    # Move past #
    i = separator + 1

    # Get the actual word
    word = encoded[i:i + length]
    decoded.append(word)

    # Move to the next encoded string
    i += length

print("Decoded:", decoded)