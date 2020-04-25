from code.key import SecuredKey


class Document:
    def __init__(self, path, name, text):
        self.path = path
        self.name = name
        self.text = text

    def __str__(self):
        return self.name

    def decrypt(self, key: SecuredKey):
        return Document(self.path, self.name, key.decrypt(self.text).decode())

    def encrypt(self, key: SecuredKey):
        return Document(self.path, self.name, key.encrypt(self.text.encode()))
