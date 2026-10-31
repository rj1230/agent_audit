users_data = {
    "user123": {
        "book": {"price": 23, "date": "2026-10-5"},
        "pen": {"price": 7, "date": "2026-10-3"},
    },
    "user124": {"notebook": {"price": 12, "date": "2026-10-6"}},
    "user125": {
        "pencil": {"price": 3, "date": "2026-10-14"},
        "ruler": {"price": 5, "date": "2026-9-24"},
    },
    "user126": {
        "backpack": {"price": 45, "date": "2026-10-2"},
        "water_bottle": {"price": 18, "date": "2026-7-25"},
        "lunch_box": {"price": 14, "date": "2026-10-10"},
    },
    "user127": {"eraser": {"price": 2, "date": "2026-10-10"}},
    "user128": {
        "stapler": {"price": 11, "date": "2026-10-10"},
        "highlighter": {"price": 4, "date": "2026-10-10"},
    },
    "user129": {
        "calculator": {"price": 29, "date": "2026-9-24"},
        "folder": {"price": 6, "date": "2026-10-25"},
        "usb_drive": {"price": 16, "date": "2026-7-27"},
    },
    "user130": {"headphones": {"price": 35, "date": "2026-10-10"}},
    "user131": {
        "desk_lamp": {"price": 28, "date": "2026-10-4"},
        "mouse": {"price": 20, "date": "2026-10-4"},
    },
    "user132": {
        "sketchbook": {"price": 17, "date": "2026-10-14"},
        "paintbrush": {"price": 9, "date": "2026-10-15"},
        "canvas": {"price": 25, "date": "2026-7-30"},
    },
}

# print(users_data["user123"])

import random
import json
from datetime import date, timedelta

# Reproducible dataset generation
random.seed(42)

# Product catalog: product name and base price
products = {
    "book": 23,
    "pen": 7,
    "notebook": 12,
    "pencil": 3,
    "ruler": 5,
    "backpack": 45,
    "water_bottle": 18,
    "lunch_box": 14,
    "eraser": 2,
    "stapler": 11,
    "highlighter": 4,
    "calculator": 29,
    "folder": 6,
    "usb_drive": 16,
    "headphones": 35,
    "desk_lamp": 28,
    "mouse": 20,
    "sketchbook": 17,
    "paintbrush": 9,
    "canvas": 25,
    "keyboard": 32,
    "monitor": 150,
    "charger": 15,
    "phone_case": 12,
    "tablet_stand": 22,
    "webcam": 40,
    "microphone": 55,
    "power_bank": 30,
    "pen_drive": 14,
    "office_chair": 120,
    "watercolor_set": 19,
    "geometry_box": 10,
    "scissors": 6,
    "paper_pack": 8,
    "calculator_scientific": 45,
    "bluetooth_speaker": 38,
    "external_hard_drive": 75,
    "travel_mug": 21,
    "reading_light": 13,
    "wireless_mouse": 25,
}

# Generate 990 additional users
start_date = date(2026, 7, 1)
end_date = date(2026, 10, 31)

date_range = (end_date - start_date).days

for user_num in range(133, 1123):
    user_id = f"user{user_num}"

    # Each user has between 1 and 5 purchases
    purchase_count = random.randint(1, 5)

    selected_products = random.sample(list(products.keys()), purchase_count)

    purchases = {}

    for product in selected_products:
        base_price = products[product]

        # Vary product prices slightly
        price = round(base_price * random.uniform(0.85, 1.15), 2)

        # Generate purchase date
        purchase_date = start_date + timedelta(days=random.randint(0, date_range))

        purchases[product] = {"price": price, "date": purchase_date.isoformat()}

    users_data[user_id] = purchases


# Validate dataset
assert len(users_data) == 1000

total_purchases = sum(len(purchases) for purchases in users_data.values())

print("Dataset generation successful!")
print("Total users:", len(users_data))
print("Total purchases:", total_purchases)

# Save dataset
with open("users_data_1000.json", "w") as f:
    json.dump(users_data, f, indent=4)

print("Saved: users_data_1000.json")

# Preview
print(json.dumps(users_data["user133"], indent=4))
