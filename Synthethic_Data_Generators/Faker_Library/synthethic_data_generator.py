import os
import pandas as pd
import random
from faker import Faker
from mimesis import Person, Address
from mimesis.enums import Gender

# Initialize Faker and Mimesis
fake = Faker()
person = Person()
address = Address()

# Define output folder
output_folder = "Faker_Library"
os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

def generate_student_data(num_records=10):
    """Generates synthetic student data using Faker and Mimesis."""
    data = []
    for _ in range(num_records):
        gender = random.choice(["Male", "Female"])
        
        # Generate name based on gender
        name = person.full_name(gender=Gender.MALE if gender == "Male" else Gender.FEMALE)
        
        # Generate random student data
        student = {
            "Name": name,
            "Age": random.randint(18, 25),
            "Grade": random.choice(["A", "B", "C", "D"]),
            "Gender": gender,
            "Address": address.address(),
            "Phone Number": fake.phone_number(),
            "Email": fake.email(),
            "GPA": round(random.uniform(2.0, 4.0), 2),
            "Math Score": random.randint(50, 100),
            "Science Score": random.randint(50, 100),
            "English Score": random.randint(50, 100)
        }
        data.append(student)

    return pd.DataFrame(data)

# Generate 10 student records
df = generate_student_data(10)

# Save to CSV inside the folder
csv_filename = os.path.join(output_folder, "synthetic_student_data.csv")
df.to_csv(csv_filename, index=False)

print(f"✅ Synthetic dataset saved in '{csv_filename}'")
print(df.head())  # Display first few records
