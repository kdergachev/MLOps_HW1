import psycopg2
import pickle




def db_decorator(queryf):
    """
    Open connection, run function with this connection, close items
    (pretty much a "with" but as a self defined decorator)
    """
    
    def df_wrapper(*args, **kwargs):
        
        conn = psycopg2.connect(dbname='MLOps', user='postgres', 
                        password='user', host='localhost') # дада, надо прятать
        cursor = conn.cursor()
        # всё как обычно, но даже при ошибке закроет коннекшн (вроде бы)
        try:
            res = queryf(conn, cursor, *args, **kwargs)
        except Exception as e:
            raise e
        finally:
            cursor.close()
            conn.close()
        
        return res
    
    return df_wrapper


@db_decorator
def db_add_model(conn, cursor, model, modelname, idx=None):
    """
    Add a new model to database if idx is not supplied, otherwise rewrites
    an existing row keeping only id
    """
    
    model = psycopg2.Binary(pickle.dumps(model))
    
    if idx is not None:
        q = f"""UPDATE model_store
                SET model_bytes = {model},
                    date = NOW(),
                    model_type = '{modelname}'"""
    
    else:
        q = f"""INSERT INTO model_store (model_bytes, date, model_type) 
                VALUES ({model}, NOW(), '{modelname}')"""
    
    cursor.execute(q)
    conn.commit()


def db_drop_row_by_id(conn, cursor, idx):
    """
    Not used directly
    (was used to delete old model under given index now replaced by update)
    """
    
    q = f"""DELETE FROM model_store
            WHERE id = {idx}"""
    cursor.execute(q)
    conn.commit()


@db_decorator
def db_pure_delete(conn, cursor, idx):
    """
    Drop row with given index from database
    """
    db_drop_row_by_id(conn, cursor, idx)


@db_decorator
def db_fetch_model(conn, cursor, idx):
    """
    returns a model (object) given id from the database
    """
    q = f"""SELECT model_bytes FROM model_store
            WHERE id = {idx}"""
    cursor.execute(q)
    mod_bin = cursor.fetchall()[0][0].tobytes() # cursor -> list -> memory -> bytes
    model = pickle.loads(mod_bin)
    return model