# generate_passwords.py

import streamlit_authenticator as stauth
import yaml

# Define user credentials
names = ["User One", "User Two"]
usernames = ["user1", "user2"]
emails = ["user1@example.com", "user2@example.com"]
passwords = ["password1", "password2"]  # Replace with secure passwords

# Generate hashed passwords
hashed_passwords = stauth.Hasher(passwords).generate()

# Create the credentials dictionary
credentials = {
    "usernames": {
        usernames[i]: {
            "name": names[i],
            "email": emails[i],
            "password": hashed_passwords[i]
        } for i in range(len(names))
    }
}

# Define the cookie configuration
cookie_config = {
    "name": "media_bias_detection_tool",
    "key": "random_cookie_key_1234567890",  # This will be automatically handled in the code
    "expiry_days": 30
}

# Define preauthorized emails (optional)
preauthorized = {
    "emails": [
        "user1@example.com"
    ]
}

# Combine all configurations
config = {
    "credentials": credentials,
    "cookie": cookie_config,
    "preauthorized": preauthorized
}

# Write the configuration to a YAML file
with open('hashed_passwords.yaml', 'w') as file:
    yaml.dump(config, file)

print("Hashed passwords have been generated and saved to 'hashed_passwords.yaml'.")
