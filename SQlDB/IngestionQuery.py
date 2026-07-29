from datetime import datetime
import json
import os
from config import connection_string
import pyodbc

def LogStatus(_DocID, _ActionName, _FileName, _Step, _Status, _ErrorMessage, _Timestamp):
    conn = None
    cursor = None

    try:
        if _Timestamp is None:
            _Timestamp = datetime.now()

        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        print(f"LogStatus => DocID={_DocID}, Action={_ActionName}, Step={_Step}, Status={_Status}")

        # اگر نام فایل پاس نشده، از جدول بخوان
        if not _FileName and _DocID != -1:
            cursor.execute("""
                SELECT Title
                FROM [LLMDB].[dbo].[LLM_Documnts]
                WHERE id = ?
            """, (_DocID,))

            row = cursor.fetchone()
            _FileName = row[0] if row else None

        # اگر باز هم FileName نداشتیم
        if not _FileName:
            _FileName = "Unknown"

        # بررسی اینکه آیا برای این DocID رکوردی وجود دارد یا نه
        cursor.execute("""
            SELECT COUNT(*)
            FROM [LLMDB].[dbo].[LLM_Pipeline_Status]
            WHERE DocID = ?
        """, (_DocID,))

        exists = cursor.fetchone()[0] > 0

        if exists:
            query_update = """
                UPDATE [LLMDB].[dbo].[LLM_Pipeline_Status]
                SET 
                    FileName = ?,
                    Step = ?,
                    Status = ?,
                    ErrorMessage = ?,
                    Timestamp = ?,
                    ActionName = ?
                WHERE DocID = ?
            """

            cursor.execute(
                query_update,
                (
                    _FileName,
                    _Step,
                    _Status,
                    _ErrorMessage,
                    _Timestamp,
                    _ActionName,
                    _DocID
                )
            )

        else:
            insert_query = """
                INSERT INTO [LLMDB].[dbo].[LLM_Pipeline_Status]
                    (DocID, FileName, Step, Status, ErrorMessage, Timestamp, ActionName)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """

            cursor.execute(
                insert_query,
                (
                    _DocID,
                    _FileName,
                    _Step,
                    _Status,
                    _ErrorMessage,
                    _Timestamp,
                    _ActionName
                )
            )

        conn.commit()
        print("✅ LogStatus saved successfully")

    except Exception as e:
        print(f"❌ Error in LogStatus: {e}")
        import traceback
        traceback.print_exc()

        if conn:
            conn.rollback()

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def Deactivate_doc_from_sql(doc_id):
    
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()

    query = """
    update [LLMDB].[dbo].[LLM_Documnts]  set isActive=0
    WHERE id = ?
    """

    cursor.execute(query, (doc_id,))
    conn.commit()

    deleted_rows = cursor.rowcount

    cursor.close()
    conn.close()

def load_chunks_from_db():
   
    conn_str = connection_string
    
  
    query = """
    SELECT id, JsonData
    FROM dbo.LLM_Documnts
    WHERE IsActive = 1
    """

    chunks = []

    with pyodbc.connect(conn_str) as conn:
      
        cursor = conn.execute(query)

        for row in cursor:
            
            table_id = row.id
            json_data = row.JsonData
          
            # اگر JSON به صورت string ذخیره شده
            data = json.loads(json_data)
           
            for chunk in data:
                chunk_id = int(chunk["id"])

                qdrant_id = int(f"{table_id}{chunk_id:05d}")
                chunk["metadata"]["doc_id"] = str(table_id)
                chunk["id"] = qdrant_id
                chunks.append(chunk)
              
    return chunks


def load_chunks_from_dbByDocId(docId):

    conn_str =connection_string

    query = f"""
SELECT id, JsonData
FROM dbo.LLM_Documnts
WHERE  id = {docId}
"""
    print(query)
    chunks = []
    table_id=0
    with pyodbc.connect(conn_str) as conn:
        cursor = conn.execute(query)

        for row in cursor:
            table_id = row.id
            json_data = row.JsonData

            # اگر JSON به صورت string ذخیره شده
            data = json.loads(json_data)

            for chunk in data:
                chunk_id = int(chunk["id"])

                qdrant_id = int(f"{table_id}{chunk_id:05d}")
                chunk["metadata"]["doc_id"] = str(table_id)
                chunk["id"] = qdrant_id
                chunks.append(chunk)
             
              
    return chunks,table_id
def SetIsActiveTrue(target_file_path, Doc_id): 
    file_name = os.path.basename(target_file_path)
    conn = None
    cursor = None
    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        
        # ۱. کوئری اصلاح شد: هم id و هم file_name اضافه شدند
        # ۲. مطمئن شوید نام ستون id است (اگر در دیتابیس اسمش Doc_id است، آن را تغییر دهید)
        cursor.execute("""
            UPDATE LLM_Documnts 
            SET IsActive = 1 
            WHERE Title = ? AND IsActive = 0 AND id = ?
        """, (file_name, Doc_id)) # حالا تعداد پارامترها با تعداد ? ها یکی است
        
        conn.commit() 
        print(f"✅ وضعیت {file_name} (ID: {Doc_id}) به فعال تغییر یافت.")
        
    except Exception as e:
        print(f"❌ خطا در فعال‌سازی رکورد: {e}")
        if conn: conn.rollback()
        
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

def SetIsActiveFalse( Doc_id): 
    
    conn = None
    cursor = None
    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        
        # ۱. کوئری اصلاح شد: هم id و هم file_name اضافه شدند
        # ۲. مطمئن شوید نام ستون id است (اگر در دیتابیس اسمش Doc_id است، آن را تغییر دهید)
        cursor.execute("""
            UPDATE LLM_Documnts 
            SET IsActive = 0 
            WHERE id = ? 
        """, ( Doc_id)) # حالا تعداد پارامترها با تعداد ? ها یکی است
        
        conn.commit() 
      
        
    except Exception as e:
        print(f"❌ خطا در فعال‌سازی رکورد: {e}")
        if conn: conn.rollback()
        
    finally:
        if cursor: cursor.close()
        if conn: conn.close()
    
    
def InsertDocsToSql(target_file_path):
    file_name = os.path.basename(target_file_path)
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()

    try:
        # ۱. خواندن محتوای فایل JSON
        with open(target_file_path, 'r', encoding='utf-8') as f:
            json_data = json.dumps(json.load(f), ensure_ascii=False)
    
     
       
        query_insert = """
            INSERT INTO LLM_Documnts (Title, Path, JsonData, IsActive)
            OUTPUT INSERTED.ID
            VALUES (?, ?, ?, 0)
        """
      
        cursor.execute(query_insert, (file_name, target_file_path, json_data))
   
      
        new_id = int(cursor.fetchone()[0])
        
     
        conn.commit()
        
        print(f"✅ رکورد جدید با موفقیت ثبت شد. ID جدید: {new_id}")
        return new_id

    except Exception as e:
        print(f"❌ خطا در درج دیتابیس رخ داد: {str(e)}")
        import traceback
        traceback.print_exc() 
        
        if conn:
            conn.rollback()
        return -1

    finally:
        cursor.close()
        conn.close()
        print("🔌 اتصال به دیتابیس بسته شد.")
def SetALLRecord_IsActiveFalse( ): 
    
    conn = None
    cursor = None
    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        
        # ۱. کوئری اصلاح شد: هم id و هم file_name اضافه شدند
        # ۲. مطمئن شوید نام ستون id است (اگر در دیتابیس اسمش Doc_id است، آن را تغییر دهید)
        cursor.execute("""
            UPDATE LLM_Documnts 
            SET IsActive = 0 
           
        """)
        
        conn.commit() 
      
        
    except Exception as e:
        print(f"❌ خطا در فعال‌سازی رکورد: {e}")
        if conn: conn.rollback()
        
    finally:
        if cursor: cursor.close()
        if conn: conn.close()