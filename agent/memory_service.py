class MemoryService:

    def __init__(self, db):
        self.db = db


    def get_recent_history(
        self,
        session_id: str,
        limit: int = 10
    ):

        rows = self.db.fetch_all(
            """
            SELECT TOP (:limit)
                role,
                message
            FROM conversations
            WHERE session_id=:session
            ORDER BY id DESC
            """,
            {
                "limit": limit,
                "session": session_id
            }
        )

        return list(reversed(rows))