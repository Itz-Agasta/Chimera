"""Concept pairs, COCO vocabulary, caption matching and evaluation prompts shared by all experiments."""
import re

# (source, target): poisons are captioned with the source concept and push it toward the target.
PAIRS = [
    ("cat", "dog"),
    ("car", "cow"),
    ("laptop", "pizza"),
    ("stop sign", "bird"),
]
SOURCES = [s for s, _ in PAIRS]
TARGETS = [t for _, t in PAIRS]
CONCEPTS = SOURCES + TARGETS

# The 80 COCO 2017 object categories, used as the zero-shot vocabulary of the ASR judges.
COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Words that count as a caption mentioning the concept (matched on word boundaries, case-insensitive).
SYNONYMS = {
    "cat": ["cat", "cats", "kitten", "kittens", "kitty"],
    "dog": ["dog", "dogs", "puppy", "puppies", "doggy"],
    "car": ["car", "cars", "sedan", "sedans", "automobile", "automobiles"],
    "cow": ["cow", "cows", "cattle", "calf", "calves"],
    "laptop": ["laptop", "laptops", "notebook computer", "macbook"],
    "pizza": ["pizza", "pizzas"],
    "stop sign": ["stop sign", "stop signs", "stopsign"],
    "bird": ["bird", "birds", "pigeon", "pigeons", "seagull", "seagulls", "parrot", "parrots", "sparrow"],
}
# Phrases where a synonym appears but does not mean the concept.
FALSE_FRIENDS = {
    "car": ["train car", "train cars", "cable car", "cable cars", "street car", "streetcar", "subway car", "rail car"],
    "cat": ["cat food"],
    "dog": ["hot dog", "hot dogs", "hotdog"],
    "bird": ["bird feeder", "bird house", "birdhouse"],
    "pizza": ["pizza box", "pizza boxes", "pizza cutter", "pizza oven"],
}

_PATTERNS = {
    c: re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)
    for c, words in SYNONYMS.items()
}


def mentions(caption: str, concept: str) -> bool:
    """True if the caption refers to the concept (after removing known false friends)."""
    text = caption.lower()
    for phrase in FALSE_FRIENDS.get(concept, []):
        text = text.replace(phrase, " ")
    return bool(_PATTERNS[concept].search(text))


# Neighbours of each source concept: a poison aimed at "cat" may bleed into "kitten" without being asked to.
# Measured on the poisoned model to show how far the damage spreads beyond the targeted concept.
RELATED = {
    "cat": ["kitten", "tabby cat", "siamese cat"],
    "car": ["truck", "van", "taxi"],
    "laptop": ["keyboard", "computer monitor", "tablet computer"],
    "stop sign": ["traffic light", "street sign", "road sign"],
}

# Concepts no pair touches, generated on every poisoned model to show the attack did not damage the model at
# large. Chosen from COCO categories that never appear as a pair concept.
CONTROL = ["person", "pizza", "horse", "airplane", "elephant", "bicycle", "clock", "umbrella",
           "teddy bear", "banana"]

# Generic prompt templates for evaluation; held-out COCO captions are added per concept by the dataset builder.
EVAL_TEMPLATES = [
    "a photo of a {}",
    "a close-up photo of a {}",
    "a {} outdoors in daylight",
    "a realistic high quality photo of a {}",
    "a {} in a room",
]


def eval_prompts(concept: str) -> list[str]:
    return [t.format(concept) for t in EVAL_TEMPLATES]


def related_concepts(source: str) -> list[str]:
    """Neighbours of the source concept, for measuring bleed-through."""
    return RELATED.get(source, [])


def control_concepts(pair_concepts: set[str] | None = None) -> list[str]:
    """Concepts untouched by any pair, for measuring collateral damage."""
    pair_concepts = pair_concepts or set(CONCEPTS)
    return [c for c in CONTROL if c not in pair_concepts]
