import streamlit_authenticator as sauth

passwords_to_hash=['admin123','user456']
 
hashed_passwords=sauth.Hasher(passwords_to_hash).generate()

for password,hash_val in zip(passwords_to_hash,hashed_passwords):
    print("The script is running")
    print(f'Password {password}')
    print(f'Hash {hash_val}')
    print('-'*10)