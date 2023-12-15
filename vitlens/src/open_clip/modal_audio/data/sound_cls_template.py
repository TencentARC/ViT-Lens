SOUND_CLS_TEMPLATE = (
    lambda c: f"The sound of {c.lower()}.",
    lambda c: f"This is the sound of {c.lower()}.",
    lambda c: f"A sound of {c.lower()}.",
    lambda c: f"This is a sound of {c.lower()}.",
)

SOUND_AS_IMAGE_TEMPLATE = (
    lambda c: f"{c}.",
    lambda c: f"An image depicting {c}.",
    lambda c: f"{c}.",
    lambda c: f"An image showing {c}.",
    lambda c: f"{c}.",
    lambda c: f"This is {c}.",
    lambda c: f"A photograph shows {c}.",
    lambda c: f"An image of {c}.",
    lambda c: f"A good image of {c}.",
    lambda c: f"A photo of {c}.",
    lambda c: f"A picture of {c}.",
    lambda c: f"A bright image of {c}.",
)