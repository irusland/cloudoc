import base64
import random

import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from code.constants import SECRET


class SecuredKey:
    """
    SK is only shared by DU but protected from CS
    """

    def __init__(self, m):
        """generates the secured key SK = {S, M1, M2, g}"""

        # randomly generated m-bit vector
        self.S = self.secure_vector(m)

        # randomly generated m×m-dimensional invertible matrices
        self.M1 = np.random.randn(m, m)
        self.M2 = np.random.randn(m, m)

        salt = b'\xc8,J\xe2*\xf6\x95\xfa\xf0\x86!\xbe\x0f!]\xf3'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )

        # key for document set encryption
        self.g = base64.urlsafe_b64encode(kdf.derive(SECRET.encode()))

    def decrypt(self, document: bytes) -> bytes:
        f = Fernet(self.g)
        return f.decrypt(document)

    def encrypt(self, document: bytes) -> bytes:
        f = Fernet(self.g)
        return f.encrypt(document)

    @staticmethod
    def secure_vector(length):
        arr = np.zeros(length)
        arr[:random.randrange(length)] = 1
        np.random.shuffle(arr)
        return arr
