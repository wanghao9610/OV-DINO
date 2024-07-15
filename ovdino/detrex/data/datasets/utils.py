import re


def clean_words_or_phrase(name):
    name = re.sub(r"\(.*\)", "", name)
    name = re.sub(r"_", " ", name)
    name = re.sub(r"-", " ", name)
    name = re.sub(r"  ", " ", name)
    name = name.strip().lower()
    return name


def clean_caption(caption):
    caption = re.sub(r"  ", " ", caption)
    caption = caption.replace(" .", ".")
    caption = caption.replace(" ,", ",")
    caption = caption.strip().lower()
    caption = caption + "." if not caption.endswith(".") else caption
    return caption


if __name__ == "__main__":
    caption = "A group of people are standing in front of some stores"
    caption = "Cricketer celebrates hitting the winning runs to win the group match."
    print(clean_caption(caption))
    print(clean_words_or_phrase(caption))
