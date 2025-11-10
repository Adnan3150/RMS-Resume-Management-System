import bcrypt

class PasswordAuth:
    def hash_password(self, password: str) -> bytes:
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed_password

    def password_authenticate(self, entered_password: str, hashed_password: bytes) -> bool:
        print("verifying password..>!")
        return bcrypt.checkpw(entered_password.encode('utf-8'), hashed_password)
