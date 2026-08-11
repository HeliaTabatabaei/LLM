from datetime import datetime
import traceback

from SQlDB.db import DatabaseConnection
from config import connection_string


def updateClarifyMessage(clarify, convertionID):
    try:
        print(connection_string)

        with DatabaseConnection(connection_string) as cursor:
            query = """
                UPDATE [dbo].[Conversations]
                SET 
                    clarify = ?,               
                WHERE chatId = ?
            """
            params = (clarify, convertionID)

            cursor.execute(query, params)

    except Exception as e:
        print(f"Error updateClarifyMessage: {e}", flush=True)
        traceback.print_exc()
