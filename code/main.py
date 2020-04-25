from code.constants import k, DIMENSIONS
from code.key import SecuredKey
from code.logger.logger import Logger, LogLevel
from code.server.server import CloudServer
from code.users.owner import DataOwner
from code.users.user import DataUser
from definitions import DEBUG_LOG


def main():
    Logger.configure(level=LogLevel.CONSOLE, debug_path=DEBUG_LOG)

    owner = DataOwner()
    user = DataUser()
    server = CloudServer()

    key = SecuredKey(DIMENSIONS)

    document_vectors = owner.get_vectors()

    index, secured_documents = owner.encrypt_data(key, document_vectors)

    trapdoor_query_vector = user.get_query_vector(key, owner.model)

    user_result = server.search(secured_documents, index, trapdoor_query_vector, k)
    user_result = user.decrypt(user_result, key)

    owner_result = server.search(secured_documents, document_vectors, user.query_vector, k)
    owner_result = user.decrypt(owner_result, key)

    Logger.info(key)
    Logger.info(f'Query "{" ".join(user.query)}"')
    Logger.info(owner.model.infer_vector(user.query))
    Logger.info(trapdoor_query_vector)

    for line in user_result:
        Logger.info(f'From server {line}')

    for line in owner_result:
        Logger.info(f'From owner  {line}')

    Logger.info(owner.model.wv.most_similar(positive=['bird']))


if __name__ == '__main__':
    main()
